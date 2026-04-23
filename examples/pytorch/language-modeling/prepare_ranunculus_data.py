#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare packed Ranunculus-1B PT data (design §6 natural distribution + §7 packing).

Pipeline per language (en / ja):
  1. Download Wikipedia from `wikimedia/wikipedia`.
  2. Shuffle and split off a validation shard (design: 10M tokens each).
  3. Tokenize with the trained 96K tokenizer.
  4. Concatenate with `<|endoftext|>` separators and slice into 8192-token frames.
  5. Write train_{lang}.bin / val_{lang}.bin as uint32 flat arrays.

No temperature sampling is applied. The 2-epoch repeat is handled entirely by
PackedDataset(num_epochs=2) at training time, so each language is written to
disk exactly once. Natural distribution: EN ~89.6% / JA ~10.4%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from transformers import AutoTokenizer


LANGUAGES = ("en", "ja")

# Design §5: 10M tokens held out per language for per-language val loss.
VAL_TOKENS_PER_LANG = 10_000_000

SEED = 42


def tokenize_split(texts, tokenizer, eos_id: int, num_proc: int) -> list[np.ndarray]:
    def _enc(batch):
        return {"ids": [np.asarray(e + [eos_id], dtype=np.uint32) for e in tokenizer(batch["text"])["input_ids"]]}

    mapped = texts.map(_enc, batched=True, num_proc=num_proc, remove_columns=texts.column_names)
    return [np.asarray(ids, dtype=np.uint32) for ids in mapped["ids"]]


def pack_to_fixed_length(arrays: list[np.ndarray], seq_len: int) -> np.ndarray:
    flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint32)
    usable = (flat.size // seq_len) * seq_len
    return flat[:usable]


def prepare_language(
    lang: str,
    dump_version: str,
    tokenizer,
    seq_len: int,
    out_dir: Path,
    num_proc: int,
) -> tuple[int, int]:
    cfg = f"{dump_version}.{lang}"
    print(f"[{lang}] loading {cfg}")
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train")
    ds = ds.shuffle(seed=SEED)

    split = ds.train_test_split(test_size=min(0.01, VAL_TOKENS_PER_LANG / max(ds.num_rows * 200, 1)), seed=SEED)
    train_docs, val_docs = split["train"], split["test"]

    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print(f"[{lang}] tokenizing {train_docs.num_rows} train + {val_docs.num_rows} val docs")
    train_arrays = tokenize_split(train_docs, tokenizer, eos_id, num_proc)
    val_arrays = tokenize_split(val_docs, tokenizer, eos_id, num_proc)

    val_flat = np.concatenate(val_arrays) if val_arrays else np.zeros(0, dtype=np.uint32)
    val_flat = val_flat[: (VAL_TOKENS_PER_LANG // seq_len) * seq_len]

    train_flat = pack_to_fixed_length(train_arrays, seq_len)

    natural_tokens = sum(a.size for a in train_arrays)
    print(
        f"[{lang}] natural={natural_tokens / 1e9:.2f}B tokens written to disk (2-epoch repeat handled by PackedDataset)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"train_{lang}.bin").write_bytes(train_flat.tobytes())
    (out_dir / f"val_{lang}.bin").write_bytes(val_flat.tobytes())
    return int(train_flat.size), int(val_flat.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/packed"))
    parser.add_argument("--dump_version", default="20231101")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    totals: dict[str, tuple[int, int]] = {}
    for lang in LANGUAGES:
        totals[lang] = prepare_language(lang, args.dump_version, tokenizer, args.seq_len, args.out_dir, args.num_proc)

    total_train = sum(v[0] for v in totals.values())
    total_val = sum(v[1] for v in totals.values())
    print(f"[done] total_train={total_train / 1e9:.2f}B total_val={total_val / 1e6:.2f}M")
    for lang, (t, v) in totals.items():
        print(f"  {lang}: train={t / 1e9:.2f}B val={v / 1e6:.2f}M")


if __name__ == "__main__":
    main()
