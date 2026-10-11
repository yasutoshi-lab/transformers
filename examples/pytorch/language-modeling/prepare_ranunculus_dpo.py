#!/usr/bin/env python
"""Tokenize the ranunculus-dpo pairs for DPO on top of Ranunculus-v1-1B-SFT.

Uses the rows with ``quality_ok`` (all English rows and the Japanese rows that passed
the translation check). The pairs have no thinking, so the prompt is rendered with
the v2 chat template and ``enable_thinking=False`` (the template appends the empty
think block); the response is the chosen / rejected text followed by ``<|im_end|>``.
Prompt and response are tokenized separately and concatenated, exactly as at
generation time. Pairs longer than ``--max_len`` tokens are dropped.

Usage:
    python prepare_ranunculus_dpo.py --dataset_dir artifacts-5/dpo-dataset --out_dir artifacts-6/dpo-data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--tokenizer_dir", default="artifacts-5/models/ranunculus-v1-1b-sft-v4")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ds = datasets.load_from_disk(args.dataset_dir)
    st = Counter()
    out = {}
    for split in ("train", "validation"):
        rows = []
        for r in ds[split]:
            key = f"{split}/{r['lang']}"
            st[f"{key}/total"] += 1
            if not r["quality_ok"]:
                st[f"{key}/drop_quality"] += 1
                continue
            prompt = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, enable_thinking=False, tokenize=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["chosen"], add_special_tokens=False)["input_ids"] + [im_end]
            r_ids = tok(r["rejected"], add_special_tokens=False)["input_ids"] + [im_end]
            if len(p_ids) + max(len(c_ids), len(r_ids)) > args.max_len:
                st[f"{key}/drop_too_long"] += 1
                continue
            rows.append({"id": r["id"], "lang": r["lang"], "prompt_ids": p_ids, "chosen_ids": c_ids, "rejected_ids": r_ids})
            st[f"{key}/kept"] += 1
            st[f"{key}/tokens"] += 2 * len(p_ids) + len(c_ids) + len(r_ids)
        out[split] = datasets.Dataset.from_list(rows)
    d = Path(args.out_dir)
    datasets.DatasetDict(out).save_to_disk(str(d))
    (d / "stats.json").write_text(json.dumps(dict(sorted(st.items())), indent=2))
    print(json.dumps(dict(sorted(st.items())), indent=2))
    ex = out["train"][0]
    print(repr(tok.decode(ex["prompt_ids"])[-200:]), "||", repr(tok.decode(ex["chosen_ids"])[:120]))


if __name__ == "__main__":
    main()
