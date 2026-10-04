"""主実験の 4 条件で同じ質問に自由記述で答えさせ、生成結果を並べて比較する（定性評価）.

Generate free-form answers from the four main conditions for side-by-side comparison.

- 質問は評価用 4 択（qa_eval 範囲＝CPT では読ませたが SFT 用 QA には使っていない範囲）から、
  カテゴリ別に固定シードで選ぶ。選択肢は見せない。
- Base / Base+CPT はチャット形式を学習していないので、SFT データの QA を 3 例前置きした few-shot で
  答えさせる（"\\n\\n質問:" で打ち切る）。SFT 済みの条件はチャットテンプレートで答えさせる。
- 正解の選択肢と根拠文を並記し、Markdown に書き出す。

使い方（eng-cpt/ 直下、GPU は空いている方を指定）:
    CUDA_VISIBLE_DEVICES=1 python -m train.sample_generations
"""

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # train_lora.py と同じ回避策

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train.eval_lm import BASE_MODEL, load_model


ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "artifacts" / "qa"
RUNS = ROOT / "artifacts" / "runs"
CHAT_TEMPLATE_MODEL = "google/gemma-4-E4B-it"
CHAT_TEMPLATE_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
# チャット形式の生成を止めるトークン。base の generation_config は <eos>(1) だけだが、
# -it の公式 generation_config は [<eos>, <turn|>, <|tool_response>] = [1, 106, 50] で止める。
# base の設定のままだと、SFT 済みモデルが <turn|> を出しても止まらず回答を繰り返す
CHAT_EOS_TOKEN_IDS = [1, 106, 50]
FEWSHOT_TEMPLATE = "質問: {q}\n回答: {a}\n\n"
STOP = "\n\n質問:"
# 条件名 → (merge する adapter のリスト, チャット形式か)
CONDITIONS = {
    "Base": ([], False),
    "Base+CPT": ([RUNS / "cpt-f100" / "final"], False),
    "Base+SFT": ([RUNS / "sft-base" / "final"], True),
    "Base+CPT+SFT": ([RUNS / "cpt-f100" / "final", RUNS / "sft-cpt-f100" / "final"], True),
}


def pick_questions(n_per_cat, seed):
    """評価用 4 択からカテゴリ別に質問を選ぶ.

    Args:
        n_per_cat (dict[str, int]): カテゴリ→選ぶ問題数。
        seed (int): 乱数シード。

    Returns:
        list[dict]: 選んだ 4 択問題。
    """
    with open(QA_DIR / "mcq_eval.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    out = []
    for cat, k in n_per_cat.items():
        pool = [r for r in rows if r["category"] == cat]
        out.extend(rng.sample(pool, min(k, len(pool))))
    return out


def fewshot_prefix(seed, n=3):
    """SFT データから few-shot 例を作る（評価範囲と重ならない train 由来の QA）.

    Args:
        seed (int): 乱数シード。
        n (int): 例の数。

    Returns:
        str: few-shot の前置き文字列。
    """
    with open(QA_DIR / "sft.jsonl") as f:
        rows = [json.loads(line) for line in f]
    shots = random.Random(seed).sample(rows, n)
    return "".join(FEWSHOT_TEMPLATE.format(q=r["messages"][0]["content"], a=r["messages"][1]["content"]) for r in shots)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, add_bos, eos_token_id=None):
    """貪欲法で生成し、プロンプトより後ろだけを返す.

    Args:
        model (PreTrainedModel): 生成に使うモデル。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        prompt (str): プロンプト。
        max_new_tokens (int): 最大生成トークン数。
        add_bos (bool): 先頭に BOS を付けるか（チャットテンプレートは自前で付ける）。
        eos_token_id (list[int] | None): 生成を止めるトークン（``None`` ならモデルの既定）。

    Returns:
        str: 生成テキスト（few-shot の場合は次の「質問:」の手前で打ち切る）。
    """
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if add_bos:
        ids = [tokenizer.bos_token_id] + ids
    x = torch.tensor([ids], device="cuda")
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_token_id)
    text = tokenizer.decode(out[0, len(ids):], skip_special_tokens=True)
    return text.split(STOP)[0].strip()


def main():
    """4 条件を順に読み込んで生成し、Markdown と JSON を書き出す.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "report")
    args = ap.parse_args()

    questions = pick_questions({"mechanical": 5, "electrical": 5, "aeronautical": 2}, args.seed)
    prefix = fewshot_prefix(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_MODEL,
                                                            revision=CHAT_TEMPLATE_REVISION).chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    answers = {q["id"]: {} for q in questions}
    for cond, (adapters, chat) in CONDITIONS.items():
        missing = [str(a) for a in adapters if not Path(a).exists()]
        if missing:
            print(f"skip {cond}: adapter がありません {missing}", flush=True)
            continue
        model = load_model(BASE_MODEL, [str(a) for a in adapters])
        for q in questions:
            if chat:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                                       add_generation_prompt=True)
                answers[q["id"]][cond] = generate(model, tokenizer, prompt, args.max_new_tokens, add_bos=False,
                                                  eos_token_id=CHAT_EOS_TOKEN_IDS)
            else:
                prompt = prefix + f"質問: {q['question']}\n回答:"
                answers[q["id"]][cond] = generate(model, tokenizer, prompt, args.max_new_tokens, add_bos=True)
        print(f"done {cond}", flush=True)
        del model
        torch.cuda.empty_cache()

    lines = ["# 生成結果の比較（主実験 4 条件）", "",
             "Base / Base+CPT は few-shot（SFT データの QA 3 例を前置き）、SFT 済みはチャット形式。貪欲法。", ""]
    for q in questions:
        lines += [f"## {q['id']}（{q['book']}）", "", f"**質問**: {q['question']}", "",
                  f"**正解の選択肢**: {q['choices'][q['answer']]}", "", f"**根拠（教科書）**: {q['evidence']}", ""]
        for cond, text in answers[q["id"]].items():
            lines += [f"**{cond}**:", "", "> " + text.replace("\n", "\n> "), ""]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generations.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "generations.json").write_text(json.dumps(
        [{**{k: q[k] for k in ("id", "book", "category", "question", "choices", "answer", "evidence")},
          "outputs": answers[q["id"]]} for q in questions], ensure_ascii=False, indent=1))
    print(f"wrote {args.output_dir / 'generations.md'}")


if __name__ == "__main__":
    main()
