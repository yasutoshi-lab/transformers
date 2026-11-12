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
"""Tokenize the Freesia v2 pool for stage-2 pretraining and write freesia-pt (the exact documents used).

v2 プールの train（シャッフル済み）を先頭から読み、言語ごとに `--tokens-per-lang` に達するまでトークン化して
`<packed>/<lang>_train.bin`（uint16・文書ごとに <|eos|>）を書く。validation / test はプールの全文書をトークン化する。
使った文書は `<pt-out>/<lang>/{train,validation,test}-*.parquet` に同じ順序で書き出す（freesia-pt）。

Usage:
    python prepare_freesia_pt_v2.py --pool ~/freesia-work/v2/pool --tokenizer artifacts/tokenizer-v2/v32k \
        --packed ~/freesia-work/v2/packed --pt-out ~/freesia-work/v2/freesia-pt
"""

from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer


_TOK = None
_EOS = None


def _init(tok_dir: str) -> None:
    global _TOK, _EOS
    _TOK = AutoTokenizer.from_pretrained(tok_dir)
    _EOS = _TOK.eos_token_id


def _encode(texts: list[str]) -> np.ndarray:
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    return np.concatenate([np.asarray(x + [_EOS], dtype=np.uint16) for x in ids])


def iter_rows(pool: Path, lang: str, split: str, batch: int = 512):
    """Yield batches of rows (dicts) of one pool split in file order."""
    for f in sorted(glob.glob(str(pool / lang / f"{split}-*.parquet"))):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=batch):
            yield b.to_pylist()


class ShardWriter:
    """Write rows to `<split>-NNNNN.parquet` shards of `shard_docs` rows (renamed to -of-M at close)."""

    def __init__(self, out: Path, split: str, schema: pa.Schema, shard_docs: int):
        self.out, self.split, self.schema, self.shard_docs = out, split, schema, shard_docs
        self.buf, self.paths = [], []
        out.mkdir(parents=True, exist_ok=True)

    def add(self, rows):
        self.buf.extend(rows)
        while len(self.buf) >= self.shard_docs:
            self._flush(self.buf[: self.shard_docs])
            self.buf = self.buf[self.shard_docs :]

    def _flush(self, rows):
        p = self.out / f".{self.split}-{len(self.paths):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=self.schema), p, compression="zstd",
                       row_group_size=5000, write_page_index=True)
        self.paths.append(p)

    def close(self):
        if self.buf:
            self._flush(self.buf)
        n = len(self.paths)
        for i, p in enumerate(self.paths):
            p.rename(self.out / f"{self.split}-{i:05d}-of-{n:05d}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--pt-out", type=Path, required=True)
    parser.add_argument("--tokens-per-lang", type=int, default=3_050_000_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shard-docs", type=int, default=100_000)
    args = parser.parse_args()
    args.packed.mkdir(parents=True, exist_ok=True)
    summary = {}
    with Pool(args.workers, initializer=_init, initargs=(str(args.tokenizer),)) as pool:
        for lang in ("ja", "en"):
            schema = pq.read_schema(sorted(glob.glob(str(args.pool / lang / "train-*.parquet")))[0])
            summary[lang] = {}
            for split in ("validation", "test", "train"):
                writer = ShardWriter(args.pt_out / lang, split, schema, args.shard_docs)
                docs = tokens = 0
                with open(args.packed / f"{lang}_{split}.bin", "wb") as fbin:
                    group = []
                    stream = iter_rows(args.pool, lang, split)
                    done = False
                    while not done:
                        group = [b for _, b in zip(range(args.workers * 2), stream)]
                        if not group:
                            break
                        arrays = pool.map(_encode, [[r["text"] for r in b] for b in group])
                        for b, arr in zip(group, arrays):
                            arr.tofile(fbin)
                            tokens += int(arr.size); docs += len(b)
                            writer.add(b)
                            if split == "train" and tokens >= args.tokens_per_lang:
                                done = True
                                break
                        print(f"[{lang}/{split}] docs={docs:,} tokens={tokens:,}", flush=True)
                writer.close()
                summary[lang][split] = {"docs": docs, "tokens": tokens}
            print(lang, json.dumps(summary[lang]), flush=True)
    (args.packed / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.pt_out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("PREPARE_PT_V2_DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
