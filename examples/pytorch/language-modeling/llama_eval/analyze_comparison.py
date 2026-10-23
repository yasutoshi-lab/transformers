#!/usr/bin/env python
"""Summarize the Llama-3.2 1B/3B SFT comparison and test the differences.

Reads the lm-eval outputs under ``<art>/eval/<run>/`` (written by ``run_bench.sh`` and
``run_chat_evals.sh``) and prints Markdown tables:

* per-run scores (pretrained 1B/3B, SFT 1B/3B, SFT chat with thinking off/on);
* McNemar exact tests (two-sided binomial on the discordant pairs) for pretrained vs. SFT and
  for thinking off vs. on, pairing the same questions by task and ``doc_id``;
* for the chat generations, how many outputs hit the generation cap and how many of those loop
  (rep-4 > 0.5: share of repeated 4-token windows).

Usage:
    python llama_eval/analyze_comparison.py --art artifacts-cmp --tokenizer artifacts-cmp/tokenizer
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

from transformers import AutoTokenizer


# (label, run dir, task prefix, metric key in the samples, metric key in the results)
BENCH = [
    ("MMLU（5-shot）", "mmlu", "mmlu_", "acc", "acc,none"),
    ("MMLU-Pro", "bench1", "leaderboard_mmlu_pro", "acc", "acc,none"),
    ("JMMLU", "bench1", "jmmlu_", "acc", "acc,none"),
    ("CyberMetric-10000", "cybermetric", "cybermetric_10000", "acc", "acc,none"),
    ("LAMBADA", "bench2", "lambada_openai", "acc", "acc,none"),
    ("HellaSwag（acc_norm）", "bench2", "hellaswag", "acc_norm", "acc_norm,none"),
    ("ARC-Easy", "bench2", "arc_easy", "acc", "acc,none"),
    ("PIQA", "bench2", "piqa", "acc", "acc,none"),
    ("WinoGrande", "bench2", "winogrande", "acc", "acc,none"),
    ("XWinograd(ja)", "bench2", "xwinograd_jp", "acc", "acc,none"),
    ("JCommonsenseQA", "bench2", "jcommonsenseqa_cloze", "acc", "acc,none"),
    ("GSM8K（5-shot、そのまま、flexible）", "gsm8k-plain", "gsm8k", "exact_match", None),
    ("HumanEval（そのまま）", "humaneval-plain", "humaneval", "pass@1", None),
]
# (label, run dir for thinking off, run dir for thinking on, task prefix, generation cap off/on)
CHAT = [
    ("GSM8K（5-shot multiturn、flexible）", "gsm8k-chat-nothink", "gsm8k-chat-think", "gsm8k", (2048, 4096)),
    ("HumanEval（pass@1）", "humaneval-chat-nothink", "humaneval-chat-think", "humaneval_instruct", (1024, 2048)),
]


def load_samples(run_dir: str, prefix: str, metric: str) -> dict[tuple[str, int], float]:
    """Return {(task, doc_id): score} for every samples file of ``prefix`` under ``run_dir``.

    GSM8K writes one record per filter; the ``flexible-extract`` one is used.
    """
    out = {}
    for f in glob.glob(os.path.join(run_dir, "*", "samples_*.jsonl")):
        task = os.path.basename(f)[len("samples_"):].rsplit("_", 1)[0]
        if not task.startswith(prefix):
            continue
        for line in open(f):
            r = json.loads(line)
            if "filter" in r and r["filter"] not in ("none", "flexible-extract", "create_test", "build_predictions_think"):
                continue
            if metric == "exact_match" and r.get("filter") not in ("flexible-extract",):
                continue
            key = metric if metric in r else next(k for k in r if k.startswith(metric))
            out[(task, r["doc_id"])] = float(r[key])
    return out


def load_responses(run_dir: str, prefix: str) -> list[str]:
    """Return the raw generations (one per question) of ``prefix`` under ``run_dir``."""
    out = []
    for f in glob.glob(os.path.join(run_dir, "*", "samples_*.jsonl")):
        if not os.path.basename(f)[len("samples_"):].startswith(prefix):
            continue
        seen = set()
        for line in open(f):
            r = json.loads(line)
            if r["doc_id"] in seen:
                continue
            seen.add(r["doc_id"])
            resp = r["resps"][0]
            out.append(resp[0] if isinstance(resp, list) else resp)
    return out


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on the shared keys.

    Returns:
        (only ``b`` correct, only ``a`` correct, p-value).
    """
    keys = a.keys() & b.keys()
    nb = sum(1 for k in keys if b[k] > 0.5 >= a[k])
    na = sum(1 for k in keys if a[k] > 0.5 >= b[k])
    n = na + nb
    if n == 0:
        return nb, na, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(na, nb) + 1)) / 2**n
    return nb, na, min(1.0, 2 * tail)


def rep4(ids: list[int]) -> float:
    """Share of 4-token windows that already appeared earlier in the sequence."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.2g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts-cmp")
    ap.add_argument("--tokenizer", default="artifacts-cmp/tokenizer")
    args = ap.parse_args()
    ev = os.path.join(args.art, "eval")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for size in ("1b", "3b"):
        base, sft = os.path.join(ev, f"base-{size}"), os.path.join(ev, f"sft-{size}")
        print(f"\n## {size.upper()}: 事前学習モデルと SFT 後（同じ方式）\n")
        print("| タスク | 事前学習モデル | SFT 後 | 差 | SFT だけ正解 / 事前学習モデルだけ正解 | p |")
        print("|---|---:|---:|---:|---:|---:|")
        for label, sub, prefix, metric, _ in BENCH:
            a = load_samples(os.path.join(base, sub), prefix, metric)
            b = load_samples(os.path.join(sft, sub), prefix, metric)
            if not a or not b:
                print(f"| {label} | – | – | – | – | – |")
                continue
            # Grouped tasks (MMLU, JMMLU) report the mean over questions, as lm-eval does.
            sa, sb = sum(a.values()) / len(a), sum(b.values()) / len(b)
            nb, na, p = mcnemar(a, b)
            print(f"| {label} | {sa:.4f} | {sb:.4f} | {sb - sa:+.4f} | {nb:,} / {na:,} | {fmt_p(p)} |")

    print("\n## SFT 後のチャット形式（思考なし: greedy / 思考あり: temperature 0.6・top_p 0.95）\n")
    print("| タスク | モデル | 思考なし | 思考あり | 思考ありだけ正解 / 思考なしだけ正解 | p |")
    print("|---|---|---:|---:|---:|---:|")
    chat_scores = {}
    for label, off, on, prefix, _ in CHAT:
        metric = "exact_match" if prefix == "gsm8k" else "pass@1"
        for size in ("1b", "3b"):
            sft = os.path.join(ev, f"sft-{size}")
            # The thinking-off and -on task names differ (humaneval_instruct_nothink / _think): pair by doc_id.
            a = {k[1]: v for k, v in load_samples(os.path.join(sft, off), prefix, metric).items()}
            b = {k[1]: v for k, v in load_samples(os.path.join(sft, on), prefix, metric).items()}
            chat_scores[(label, size)] = (a, b)
            nb, na, p = mcnemar(a, b)
            print(f"| {label} | {size.upper()} | {sum(a.values()) / len(a):.4f} | {sum(b.values()) / len(b):.4f} | {nb} / {na} | {fmt_p(p)} |")

    print("\n## 1B と 3B（SFT 後、同じ問題）\n")
    print("| タスク | 条件 | 1B | 3B | 3B だけ正解 / 1B だけ正解 | p |")
    print("|---|---|---:|---:|---:|---:|")
    for label, *_ in CHAT:
        for i, cond in enumerate(("思考なし", "思考あり")):
            a, b = chat_scores[(label, "1b")][i], chat_scores[(label, "3b")][i]
            nb, na, p = mcnemar(a, b)
            print(f"| {label} | {cond} | {sum(a.values()) / len(a):.4f} | {sum(b.values()) / len(b):.4f} | {nb} / {na} | {fmt_p(p)} |")

    # lm-eval drops a closed <think> block from the saved output, so the length below is the answer
    # only (or the whole unclosed thinking when the cap was hit).
    print("\n## 生成の上限に達した出力とループ（SFT 後のチャット形式。長さは保存された出力〔閉じた思考は除く〕のトークン数）\n")
    print("| タスク | モデル | 条件 | 上限 | 問題数 | 上限に達した | うちループ（rep-4 > 0.5） | 長さの中央値 |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for label, off, on, prefix, caps in CHAT:
        for size in ("1b", "3b"):
            for cond, sub, cap in (("思考なし", off, caps[0]), ("思考あり", on, caps[1])):
                resps = load_responses(os.path.join(ev, f"sft-{size}", sub), prefix)
                ids = [tok(r, add_special_tokens=False)["input_ids"] for r in resps]
                lens = sorted(len(x) for x in ids)
                hit = [x for x in ids if len(x) >= cap - 8]
                loop = sum(1 for x in hit if rep4(x) > 0.5)
                med = lens[len(lens) // 2] if lens else 0
                print(f"| {label} | {size.upper()} | {cond} | {cap:,} | {len(ids):,} | {len(hit)} | {loop} | {med:,} |")


if __name__ == "__main__":
    main()
