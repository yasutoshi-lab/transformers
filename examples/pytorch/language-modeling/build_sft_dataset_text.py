#!/usr/bin/env python
"""Export the SFT v4 training conversations as a text dataset (for publishing).

Selects exactly the conversations used for SFT v4 (same sources, filters and
4,096-token limit as ``prepare_ranunculus_sft_v2.py --final_turn_only``) and
writes them as ``messages`` lists. In thinking conversations the final assistant
message carries ``reasoning_content`` (the teacher's thinking); in no-thinking
conversations it is an empty string.

Usage:
    python build_sft_dataset_text.py --gen artifacts-5/think-data/gen.jsonl --out_dir artifacts-5/sft-dataset-text
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets
from generate_thinking_sft import build_conversations
from prepare_ranunculus_sft import IDENTITY_RE
from prepare_ranunculus_sft_v2 import TEACHER_RE, encode

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="artifacts-0/tokenizer")
    ap.add_argument("--template", default="artifacts-5/chat_template_v2.jinja")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.chat_template = Path(args.template).read_text()
    gen = {r["id"]: r for r in map(json.loads, open(args.gen, encoding="utf-8"))}
    rows, st = {"train": [], "validation": []}, Counter()
    for c in build_conversations(seed=0, ja_val_ratio=0.02):
        g = gen.get(c["id"])
        if g is not None:
            mode = "think"
            if (g["finish_reason"] != "stop" or not g["reasoning"] or not g["answer"]
                    or IDENTITY_RE.search(g["answer"]) or TEACHER_RE.search(g["answer"])):
                continue
            final = {"role": "assistant", "content": g["answer"], "reasoning_content": g["reasoning"]}
            source = "teacher:nvidia/Gemma-4-26B-A4B-NVFP4"
        else:
            mode = "nothink"
            final = {**c["messages"][-1], "reasoning_content": ""}
            source = "human"
        conv = c["messages"][:-1] + [final]
        if len(encode(conv, tok, final_turn_only=True)["input_ids"]) > args.max_len:
            continue
        msgs = [{"role": m["role"], "content": m["content"], "reasoning_content": m.get("reasoning_content", "")} for m in conv]
        split = "train" if c["split"] == "train" else "validation"
        rows[split].append({"id": c["id"], "lang": c["lang"], "mode": mode, "final_turn_source": source,
                            "messages": msgs})
        st[f"{split}/{c['lang']}/{mode}"] += 1
    out = Path(args.out_dir)
    datasets.DatasetDict({k: datasets.Dataset.from_list(v) for k, v in rows.items()}).save_to_disk(str(out))
    (out / "summary.json").write_text(json.dumps(dict(sorted(st.items())), indent=2))
    print(json.dumps(dict(sorted(st.items())), indent=2))


if __name__ == "__main__":
    main()
