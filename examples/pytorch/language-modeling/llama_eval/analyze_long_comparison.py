#!/usr/bin/env python
"""Compare the short-thinking (low) and long-thinking (-long) SFT runs of Llama-3.2 1B/3B.

Reads ``<art>/eval/{base,sft,sft-long}-{1b,3b}/`` and prints Markdown tables:

* same-method benchmarks: pretrained / SFT low / SFT long, with McNemar tests of long vs. low;
* chat evaluations (GSM8K, HumanEval, CyberMetric-2000) with thinking off / on: low vs. long, and
  thinking on vs. off within the long run;
* generations that hit the cap and how many of those loop (rep-4 > 0.5).

Uses the helpers of ``analyze_comparison.py``.

Usage:
    python llama_eval/analyze_long_comparison.py --art artifacts-cmp --tokenizer artifacts-cmp/tokenizer
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from analyze_comparison import BENCH, fmt_p, load_responses, load_samples, mcnemar, rep4

from transformers import AutoTokenizer


# CyberMetric is measured on the 2,000-question set for every run.
BENCH_CMP = [b for b in BENCH if not b[1] == "cybermetric"] + [
    ("CyberMetric-2000（5-shot）", "cybermetric2000", "cybermetric_2000", "acc", None)
]
# (label, dir off, dir on, task prefix, metric, caps off/on)
CHAT = [
    ("GSM8K（flexible）", "gsm8k-chat-nothink", "gsm8k-chat-think", "gsm8k", "exact_match", (2048, 4096)),
    ("HumanEval（pass@1）", "humaneval-chat-nothink", "humaneval-chat-think", "humaneval_instruct", "pass@1", (1024, 4096)),
    ("CyberMetric-2000（公式の採点）", "cybermetric2000-chat-nothink-evaluator", "cybermetric2000-chat-think-evaluator",
     "cybermetric_2000_chat_evaluator", "exact_match", (1024, 4096)),
]


def load_chat(run_dir: str, prefix: str, metric: str) -> dict[int, float]:
    """Return {doc_id: score} for a chat run (CyberMetric uses its own ``official-answer`` filter)."""
    if prefix.startswith("cybermetric"):
        out = {}
        for f in glob.glob(os.path.join(run_dir, "*", "samples_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                out[r["doc_id"]] = float(r["exact_match"])
        return out
    return {k[1]: v for k, v in load_samples(run_dir, prefix, metric).items()}


def mean(d: dict) -> float:
    return sum(d.values()) / len(d) if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts-cmp")
    ap.add_argument("--tokenizer", default="artifacts-cmp/tokenizer")
    args = ap.parse_args()
    ev = os.path.join(args.art, "eval")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for size in ("1b", "3b"):
        base, low, long_ = (os.path.join(ev, f"{p}-{size}") for p in ("base", "sft", "sft-long"))
        print(f"\n## {size.upper()}: 同じ方式のベンチマーク（事前学習モデル／SFT low／SFT long）\n")
        print("| タスク | 事前学習 | SFT low | SFT long | long だけ正解 / low だけ正解 | p（long vs low） | p（long vs 事前学習） |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label, sub, prefix, metric, _ in BENCH_CMP:
            a, b, c = (load_samples(os.path.join(d, sub), prefix, metric) for d in (base, low, long_))
            if not (a and b and c):
                print(f"| {label} | – | – | – | – | – | – |")
                continue
            nl, nb, p = mcnemar(b, c)
            _, _, pb = mcnemar(a, c)
            print(f"| {label} | {mean(a):.4f} | {mean(b):.4f} | {mean(c):.4f} | {nl:,} / {nb:,} | {fmt_p(p)} | {fmt_p(pb)} |")

    print("\n## SFT 後のチャット形式：low と long\n")
    print("| タスク | モデル | 条件 | low | long | long だけ正解 / low だけ正解 | p |")
    print("|---|---|---|---:|---:|---:|---:|")
    chat = {}
    for label, off, on, prefix, metric, _ in CHAT:
        for size in ("1b", "3b"):
            for cond, sub in (("思考なし", off), ("思考あり", on)):
                lo = load_chat(os.path.join(ev, f"sft-{size}", sub), prefix, metric)
                lg = load_chat(os.path.join(ev, f"sft-long-{size}", sub), prefix, metric)
                chat[(label, size, cond)] = (lo, lg)
                nl, nb, p = mcnemar(lo, lg)
                print(f"| {label} | {size.upper()} | {cond} | {mean(lo):.4f} | {mean(lg):.4f} | {nl} / {nb} | {fmt_p(p)} |")

    print("\n## long の思考なしと思考あり\n")
    print("| タスク | モデル | 思考なし | 思考あり | 思考ありだけ正解 / 思考なしだけ正解 | p |")
    print("|---|---|---:|---:|---:|---:|")
    for label, *_ in CHAT:
        for size in ("1b", "3b"):
            a, b = chat[(label, size, "思考なし")][1], chat[(label, size, "思考あり")][1]
            nb, na, p = mcnemar(a, b)
            print(f"| {label} | {size.upper()} | {mean(a):.4f} | {mean(b):.4f} | {nb} / {na} | {fmt_p(p)} |")

    print("\n## 生成の上限に達した出力とループ（チャット形式、思考あり。長さは保存された出力〔閉じた思考は除く〕のトークン数）\n")
    print("| タスク | モデル | データ | 上限 | 問題数 | 上限に達した | うちループ（rep-4 > 0.5） | 長さの中央値 |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for label, off, on, prefix, metric, caps in CHAT:
        for size in ("1b", "3b"):
            for name, run in (("low", f"sft-{size}"), ("long", f"sft-long-{size}")):
                resps = load_responses(os.path.join(ev, run, on), prefix)
                ids = [tok(r, add_special_tokens=False)["input_ids"] for r in resps]
                lens = sorted(len(x) for x in ids)
                hit = [x for x in ids if len(x) >= caps[1] - 8]
                loop = sum(1 for x in hit if rep4(x) > 0.5)
                med = lens[len(lens) // 2] if lens else 0
                print(f"| {label} | {size.upper()} | {name} | {caps[1]:,} | {len(ids):,} | {len(hit)} | {loop} | {med:,} |")


if __name__ == "__main__":
    main()
