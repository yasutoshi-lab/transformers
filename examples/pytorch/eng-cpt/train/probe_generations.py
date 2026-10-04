"""指定したサンプルについて、主実験 4 条件の自由記述生成を並べて記録する（生成結果の検証用）.

Probe free-form generations of the four main conditions on hand-picked samples.

サンプル定義（JSON。著作物由来の文言を含むため artifacts/ 配下に置く）:
    [{"sample_id": "T1", "domain": "工具", "aspect": "工具材料", "mcq_id": "mcq-0136",
      "question": "<自由記述用に書き換えた質問>"}, ...]

各サンプルの元の 4 択（問題文・選択肢・正解・根拠文）は ``artifacts/qa/mcq_eval.jsonl`` から、
出典ページはコーパスから引いて記録する。出力は 1 行 = 1 サンプル × 1 条件の JSONL。

生成条件:
    - Base / Base+CPT: SFT データの QA 3 例を前置きした few-shot（"\\n\\n質問:" で打ち切り）
    - Base+SFT / Base+CPT+SFT: -it のチャットテンプレート。<eos>/<turn|>/<|tool_response> で停止
    - 貪欲法（do_sample=False）、max_new_tokens と seed は引数で指定（貪欲法では seed は出力に影響しない）

使い方（eng-cpt/ 直下）:
    CUDA_VISIBLE_DEVICES=0 python -m train.probe_generations \\
        --samples artifacts/probe/samples.json --output artifacts/probe/probe_generations.jsonl
"""

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # train_lora.py と同じ回避策

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from train.eval_lm import BASE_MODEL, load_model
from train.sample_generations import (
    CHAT_EOS_TOKEN_IDS,
    CHAT_TEMPLATE_MODEL,
    CHAT_TEMPLATE_REVISION,
    CONDITIONS,
    STOP,
    fewshot_prefix,
)


ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "artifacts" / "qa"
DATA_DIR = ROOT / "artifacts" / "data"


def load_index(path, key):
    """JSONL を読み込み、指定キーで引ける dict にする.

    Args:
        path (pathlib.Path): JSONL ファイル。
        key (str): キーにするフィールド名。

    Returns:
        dict[str, dict]: キー→行。
    """
    with open(path) as f:
        return {r[key]: r for r in map(json.loads, f)}


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, add_bos, eos_token_id):
    """貪欲法で生成し、出力テキスト・生成トークン数・停止理由を返す.

    Args:
        model (PreTrainedModel): 生成に使うモデル。
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        prompt (str): 入力プロンプト（そのまま記録する）。
        max_new_tokens (int): 最大生成トークン数。
        add_bos (bool): 先頭に BOS を付けるか。
        eos_token_id (list[int] | int): 停止トークン。

    Returns:
        dict: ``output`` / ``n_output_tokens`` / ``finish_reason``（``stop`` / ``length`` / ``stop_sequence``）。
    """
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if add_bos:
        ids = [tokenizer.bos_token_id] + ids
    x = torch.tensor([ids], device="cuda")
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_token_id)
    new = out[0, len(ids):].tolist()
    stops = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    text = tokenizer.decode(new, skip_special_tokens=True)
    if STOP in text:
        return {"output": text.split(STOP)[0].strip(), "n_output_tokens": len(new), "finish_reason": "stop_sequence"}
    reason = "stop" if new and new[-1] in stops else ("length" if len(new) >= max_new_tokens else "stop")
    return {"output": text.strip(), "n_output_tokens": len(new), "finish_reason": reason}


def main():
    """サンプルを 4 条件で生成し、JSONL と比較用 Markdown を書き出す.

    Returns:
        None

    Raises:
        KeyError: サンプルの mcq_id が評価用 4 択に存在しない場合。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42, help="学習時と同じ seed（貪欲法では出力に影響しない）")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--fewshot-seed", type=int, default=7, help="few-shot 例の選択（sample_generations と同じ）")
    args = ap.parse_args()

    samples = json.loads(args.samples.read_text())
    mcq = load_index(QA_DIR / "mcq_eval.jsonl", "id")
    docs = {}
    for split in ("train", "qa_eval", "holdout_ppl"):
        docs.update(load_index(DATA_DIR / f"{split}.jsonl", "id"))
    prefix = fewshot_prefix(args.fewshot_seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_MODEL,
                                                            revision=CHAT_TEMPLATE_REVISION).chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    for cond, (adapters, chat) in CONDITIONS.items():
        set_seed(args.seed)
        model = load_model(BASE_MODEL, [str(a) for a in adapters])
        for s in samples:
            m = mcq[s["mcq_id"]]
            src = docs[m["source_id"]]
            if chat:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": s["question"]}], tokenize=False,
                                                       add_generation_prompt=True)
                gen = generate(model, tokenizer, prompt, args.max_new_tokens, False, CHAT_EOS_TOKEN_IDS)
            else:
                prompt = prefix + f"質問: {s['question']}\n回答:"
                gen = generate(model, tokenizer, prompt, args.max_new_tokens, True, tokenizer.eos_token_id)
            records.append({
                "sample_id": s["sample_id"], "domain": s["domain"], "aspect": s["aspect"], "condition": cond,
                "input": {"question": s["question"], "prompt": prompt,
                          "prompt_format": "chat_template" if chat else "few-shot(3)"},
                "output": gen["output"], "n_output_tokens": gen["n_output_tokens"], "finish_reason": gen["finish_reason"],
                "reference": {"correct_choice": m["choices"][m["answer"]], "evidence": m["evidence"]},
                "source": {"mcq_id": m["id"], "original_question": m["question"], "original_choices": m["choices"],
                           "answer_index": m["answer"], "book": m["book"], "chunk_id": m["chunk_id"],
                           "split": src["split"], "pages": src["pages"],
                           "in_cpt_data": src["split"] in ("train", "qa_eval"), "in_sft_data": src["split"] == "train"},
                "generation": {"seed": args.seed, "max_new_tokens": args.max_new_tokens, "do_sample": False,
                               "adapters": [str(a) for a in adapters]},
            })
        print(f"done {cond}", flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = ["# 生成結果の検証（4 条件 × サンプル）", "",
             f"貪欲法 / seed={args.seed} / max_new_tokens={args.max_new_tokens}。"
             "Base・Base+CPT は few-shot、SFT 済みはチャット形式。", ""]
    for s in samples:
        rs = [r for r in records if r["sample_id"] == s["sample_id"]]
        r0 = rs[0]
        lines += [f"## {s['sample_id']}（{s['domain']}・{s['aspect']}）{r0['source']['book']} p.{r0['source']['pages'][0]}-{r0['source']['pages'][1]}",
                  "", f"**質問**: {s['question']}", "", f"**データセット内の正答**: {r0['reference']['correct_choice']}", "",
                  f"**根拠（教科書）**: {r0['reference']['evidence']}", ""]
        for r in rs:
            lines += [f"**{r['condition']}**（{r['n_output_tokens']} tokens, {r['finish_reason']}）:", "",
                      "> " + r["output"].replace("\n", "\n> "), ""]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
