#!/usr/bin/env python
"""Export the Ranunculus pretraining corpus (Wikipedia en/ja, 20231101) as parquet, reproducing the training split.

Re-runs the exact selection of ``prepare_ranunculus_data.py`` (shuffle seed 42, then
train_test_split with the same test_size and seed) and writes the train / validation
articles per language as parquet shards. Before writing, it verifies that the split is
identical to the one used for training by tokenizing the first articles of each split and
comparing them with the beginning of ``artifacts-0/packed/{train,val}_{lang}.bin``.

Usage:
    python export_pt_corpus.py --out_dir artifacts-7/ranunculus-pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from transformers import AutoTokenizer

SEED = 42
VAL_TOKENS_PER_LANG = 10_000_000


def check_prefix(docs, tok, eos_id: int, binpath: Path, n_docs: int = 20) -> int:
    ids = []
    for t in docs.select(range(n_docs))["text"]:
        ids += tok(t)["input_ids"] + [eos_id]
    ref = np.fromfile(binpath, dtype=np.uint32, count=len(ids))
    if not np.array_equal(np.asarray(ids, dtype=np.uint32), ref):
        raise SystemExit(f"split mismatch against {binpath}")
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_dir", default="artifacts-0/tokenizer")
    ap.add_argument("--packed_dir", default="artifacts-0/packed")
    ap.add_argument("--dump_version", default="20231101")
    ap.add_argument("--shard_rows", type=int, default=200_000)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    out = Path(args.out_dir) / "data"
    for lang in ("en", "ja"):
        ds = load_dataset("wikimedia/wikipedia", f"{args.dump_version}.{lang}", split="train").shuffle(seed=SEED)
        split = ds.train_test_split(test_size=min(0.01, VAL_TOKENS_PER_LANG / max(ds.num_rows * 200, 1)), seed=SEED)
        for name, docs, binname in (("train", split["train"], f"train_{lang}.bin"), ("validation", split["test"], f"val_{lang}.bin")):
            n = check_prefix(docs, tok, eos_id, Path(args.packed_dir) / binname)
            print(f"[{lang}/{name}] {docs.num_rows} articles; first 20 articles ({n} tokens) match {binname}", flush=True)
            d = out / lang
            d.mkdir(parents=True, exist_ok=True)
            docs = docs.flatten_indices()
            for i, s in enumerate(range(0, docs.num_rows, args.shard_rows)):
                docs.select(range(s, min(s + args.shard_rows, docs.num_rows))).to_parquet(str(d / f"{name}-{i:05d}.parquet"))
            print(f"[{lang}/{name}] wrote {i + 1} shards", flush=True)


if __name__ == "__main__":
    main()
