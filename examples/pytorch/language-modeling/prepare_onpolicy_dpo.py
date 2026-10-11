#!/usr/bin/env python
"""Turn the on-policy pairs (judge_onpolicy.py output) into the dataset format of run_ranunculus_dpo.py.

Splits the pairs into train / validation (``--val_ratio``, stratified by language) and drops
pairs longer than ``--max_len`` tokens.

Usage:
    python prepare_onpolicy_dpo.py --pairs artifacts-6/onpolicy/pairs.jsonl --out_dir artifacts-6/onpolicy-data
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    st = Counter()
    keep = []
    for r in rows:
        if len(r["prompt_ids"]) + max(len(r["chosen_ids"]), len(r["rejected_ids"])) > args.max_len:
            st["drop_too_long"] += 1
            continue
        keep.append({k: r[k] for k in ("id", "lang", "prompt_ids", "chosen_ids", "rejected_ids")})
    rng = random.Random(args.seed)
    split = {"train": [], "validation": []}
    for lang in ("en", "ja"):
        rr = [r for r in keep if r["lang"] == lang]
        rng.shuffle(rr)
        n_val = max(1, int(len(rr) * args.val_ratio))
        split["validation"] += rr[:n_val]
        split["train"] += rr[n_val:]
        st[f"train/{lang}"] = len(rr) - n_val
        st[f"validation/{lang}"] = n_val
    d = Path(args.out_dir)
    for k, v in split.items():
        datasets.Dataset.from_list(v).save_to_disk(str(d / k))
    (d / "stats.json").write_text(json.dumps(dict(st), indent=2))
    print(json.dumps(dict(st), indent=2))


if __name__ == "__main__":
    main()
