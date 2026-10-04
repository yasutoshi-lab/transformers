"""学習前後のモデルを対数尤度ベースで評価する.

Evaluate base / CPT / SFT models with likelihood-based metrics so that all
conditions are measured with the same yardstick (no answer parsing).

指標:
    holdout_ppl  学習に使っていない本文（holdout_ppl split）の perplexity（ドメイン適応）
    mcq          4 択 QA（cloze 形式）。各選択肢の対数確率で回答を決める（知識獲得）
                 - acc      : 選択肢トークンの対数確率の合計が最大のものを回答とする
                 - acc_norm : 合計を選択肢の文字数で割った値が最大のものを回答とする
    jmmlu        JMMLU（56 科目・0-shot）。「答え:」の直後に続く " A"〜" D" の対数確率で回答を決める
                 （一般知識の忘却を測る。工学系 / 数学系 / その他の 3 グループでも集計）

モデルの指定:
    --adapters に LoRA adapter を順に並べると、その順にベースへ merge してから評価する
    （例: CPT→SFT は ``--adapters runs/cpt/final runs/cpt-sft/final``）。

使い方（eng-cpt/ 直下、GPU0）:
    CUDA_VISIBLE_DEVICES=0 python -m train.eval_lm --name base --output artifacts/eval/base.json
"""

import os

# PyTorch 2.13 の torch._native が Gemma4 の RoPE（bmm_outer_product）を Triton カーネルへ
# 振り替えるが、ws3-arc（RTX PRO 6000 Blackwell / driver 575）では起動時に Illegal instruction で
# 落ちる。torch を import する前に無効化しておく（環境変数で上書き指定も可）。
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "data"
QA_DIR = ROOT / "artifacts" / "qa"
BASE_MODEL = "google/gemma-4-E4B"
MCQ_TEMPLATE = "問題: {question}\n答え: "
JMMLU_DIR = ROOT / "artifacts" / "bench" / "JMMLU" / "test"
JMMLU_TEMPLATE = "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答え:"
JMMLU_LETTERS = [" A", " B", " C", " D"]
JMMLU_GROUPS = {
    "engineering": {"electrical_engineering", "college_physics", "high_school_physics", "conceptual_physics",
                    "college_chemistry", "high_school_chemistry", "college_computer_science",
                    "high_school_computer_science", "computer_security"},
    "math": {"college_mathematics", "high_school_mathematics", "elementary_mathematics",
             "high_school_statistics", "abstract_algebra", "econometrics"},
}


def load_model(base_model, adapters):
    """ベースを読み込み、adapter を順に merge する.

    Args:
        base_model (str): ベースモデルの ID かパス。
        adapters (list[str]): merge する adapter のパス（適用順）。

    Returns:
        PreTrainedModel: 評価用モデル（eval モード・GPU 上）。
    """
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, attn_implementation="sdpa")
    for path in adapters:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model.cuda().eval()


@torch.no_grad()
def holdout_perplexity(model, tokenizer, seq_len):
    """holdout_ppl の各文書を ``seq_len`` の窓に区切り、トークン平均 NLL から perplexity を出す.

    Args:
        model (PreTrainedModel): 評価対象。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        seq_len (int): 窓のトークン長（BOS を含む）。

    Returns:
        dict: ``ppl`` / ``nll`` / ``tokens``、およびカテゴリ別の ``ppl``。
    """
    total_nll, total_tok = 0.0, 0
    by_cat = {}
    with open(DATA_DIR / "holdout_ppl.jsonl") as f:
        docs = [json.loads(line) for line in f]
    for d in docs:
        ids = tokenizer(d["text"], add_special_tokens=False)["input_ids"]
        for s in range(0, len(ids), seq_len - 1):
            window = [tokenizer.bos_token_id] + ids[s : s + seq_len - 1]
            if len(window) < 2:
                continue
            x = torch.tensor([window], device="cuda")
            logits = model(input_ids=x).logits.float()
            nll = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
            n = len(window) - 1
            total_nll += nll
            total_tok += n
            c = by_cat.setdefault(d["category"], [0.0, 0])
            c[0] += nll
            c[1] += n
    return {
        "ppl": math.exp(total_nll / total_tok),
        "nll": total_nll / total_tok,
        "tokens": total_tok,
        "by_category": {k: math.exp(v[0] / v[1]) for k, v in by_cat.items()},
    }


@torch.no_grad()
def choice_logprobs(model, tokenizer, prompt, choices):
    """プロンプトに続く各選択肢の対数確率の合計を返す.

    Args:
        model (PreTrainedModel): 評価対象。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        prompt (str): 問題文を含むプロンプト。
        choices (list[str]): 選択肢。

    Returns:
        list[float]: 選択肢ごとの対数確率の合計。
    """
    p_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    seqs = [p_ids + tokenizer(c, add_special_tokens=False)["input_ids"] for c in choices]
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), tokenizer.pad_token_id, device="cuda")
    att = torch.zeros_like(x)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    logp = torch.log_softmax(model(input_ids=x, attention_mask=att).logits.float(), dim=-1)
    out = []
    for i, s in enumerate(seqs):
        tgt = torch.tensor(s[len(p_ids) :], device="cuda")
        pos = torch.arange(len(p_ids) - 1, len(s) - 1, device="cuda")
        out.append(logp[i, pos, tgt].sum().item())
    return out


def mcq_accuracy(model, tokenizer, limit=0):
    """4 択 QA の正解率（acc / acc_norm）をカテゴリ別にも計算する.

    Args:
        model (PreTrainedModel): 評価対象。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        limit (int): 先頭 N 問だけ評価（0 = 全問）。

    Returns:
        dict: ``n`` / ``acc`` / ``acc_norm`` / ``by_category`` / ``predictions``。
    """
    with open(QA_DIR / "mcq_eval.jsonl") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]
    hits, hits_norm, by_cat, preds = 0, 0, {}, []
    for r in rows:
        lps = choice_logprobs(model, tokenizer, MCQ_TEMPLATE.format(question=r["question"]), r["choices"])
        norm = [lp / max(1, len(c)) for lp, c in zip(lps, r["choices"])]
        pred, pred_norm = max(range(4), key=lps.__getitem__), max(range(4), key=norm.__getitem__)
        hits += pred == r["answer"]
        hits_norm += pred_norm == r["answer"]
        c = by_cat.setdefault(r["category"], [0, 0, 0])
        c[0] += pred == r["answer"]
        c[1] += pred_norm == r["answer"]
        c[2] += 1
        preds.append({"id": r["id"], "answer": r["answer"], "pred": pred, "pred_norm": pred_norm})
    n = len(rows)
    return {
        "n": n,
        "acc": hits / n,
        "acc_norm": hits_norm / n,
        "by_category": {k: {"acc": v[0] / v[2], "acc_norm": v[1] / v[2], "n": v[2]} for k, v in by_cat.items()},
        "predictions": preds,
    }


def jmmlu_group(subject):
    """JMMLU の科目名を集計グループに振り分ける.

    Args:
        subject (str): 科目名（CSV のファイル名）。

    Returns:
        str: ``engineering`` / ``math`` / ``other``。
    """
    for group, subjects in JMMLU_GROUPS.items():
        if subject in subjects:
            return group
    return "other"


def jmmlu_accuracy(model, tokenizer):
    """JMMLU の正解率を全体・グループ別・科目別に計算する.

    Args:
        model (PreTrainedModel): 評価対象。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。

    Returns:
        dict: ``n`` / ``acc`` / ``by_group`` / ``by_subject`` / ``predictions``。

    Raises:
        FileNotFoundError: JMMLU が ``artifacts/bench`` に展開されていない場合。
    """
    files = sorted(JMMLU_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"JMMLU が見つかりません: {JMMLU_DIR}")
    by_subject, by_group, preds = {}, {}, []
    for path in files:
        subject, group = path.stem, jmmlu_group(path.stem)
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            lps = choice_logprobs(model, tokenizer, JMMLU_TEMPLATE.format(**r), JMMLU_LETTERS)
            pred = "ABCD"[max(range(4), key=lps.__getitem__)]
            hit = pred == r["answer"].strip()
            for key, table in ((subject, by_subject), (group, by_group)):
                c = table.setdefault(key, [0, 0])
                c[0] += hit
                c[1] += 1
            preds.append({"subject": subject, "index": i, "answer": r["answer"].strip(), "pred": pred})
    n = sum(v[1] for v in by_subject.values())
    return {
        "n": n,
        "acc": sum(v[0] for v in by_subject.values()) / n,
        "by_group": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in by_group.items()},
        "by_subject": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in by_subject.items()},
        "predictions": preds,
    }


def main():
    """モデルを読み込み、指定した指標を計算して JSON に保存する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="条件名（例: base / cpt / sft / cpt-sft）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-mcq", action="store_true")
    ap.add_argument("--mcq-limit", type=int, default=0)
    ap.add_argument("--jmmlu", action="store_true", help="JMMLU も評価する（数分かかる）")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = load_model(args.base_model, args.adapters)
    result = {"name": args.name, "base_model": args.base_model, "adapters": args.adapters}
    if not args.skip_ppl:
        result["holdout_ppl"] = holdout_perplexity(model, tokenizer, args.seq_len)
        print("holdout_ppl", json.dumps(result["holdout_ppl"], ensure_ascii=False), flush=True)
    if not args.skip_mcq:
        result["mcq"] = mcq_accuracy(model, tokenizer, args.mcq_limit)
        print("mcq", {k: v for k, v in result["mcq"].items() if k != "predictions"}, flush=True)
    if args.jmmlu:
        result["jmmlu"] = jmmlu_accuracy(model, tokenizer)
        print("jmmlu", {k: result["jmmlu"][k] for k in ("n", "acc", "by_group")}, flush=True)
    result["eval_seconds"] = time.time() - t0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
