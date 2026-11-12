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
"""Build the Freesia v2 tokenizer corpus (freesia-tokenizer-pt) from the v2 pool and train byte-level BPE.

train は v2 プールの train（シャッフル済み）の先頭から、言語ごとのバイト数（既定 ja 400MB・en 300MB）に達するまでの文書。
validation / test は v2 プールの validation / test をそのまま使い、圧縮効率（1 トークンあたりの文字数）の測定だけに使う
（BPE の学習には使わない）。語彙サイズは validation の圧縮効率で選び、test の値は報告用に記録する。

Usage:
    python train_freesia_tokenizer_v2.py --pool ~/freesia-work/v2/pool --corpus-out ~/freesia-work/v2/tokenizer-pt \
        --out-dir artifacts/tokenizer-v2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import pre_tokenizers
from tokenizers.trainers import BpeTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_freesia_tokenizer import SPECIAL_TOKENS, build_tokenizer, evaluate, start_memory_guard

from transformers import PreTrainedTokenizerFast


def take_train(pool: Path, lang: str, max_bytes: int, out: Path) -> tuple[Path, dict]:
    """Copy the leading train documents of the pool up to `max_bytes` and dump them as one-doc-per-line text."""
    rows, written = [], 0
    for f in sorted(glob.glob(str(pool / lang / "train-*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pylist()
            for r in t:
                line = r["text"].replace("\n", " ").strip()
                if not line:
                    continue
                rows.append(r)
                written += len(line.encode("utf-8")) + 1
                if written >= max_bytes:
                    break
            if written >= max_bytes:
                break
        if written >= max_bytes:
            break
    d = out / lang
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(f)), d / "train-00000-of-00001.parquet",
                   compression="zstd", row_group_size=5000, write_page_index=True)
    txt = out / f"_txt/train_{lang}.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    with txt.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r["text"].replace("\n", " ").strip() + "\n")
    return txt, {"docs": len(rows), "bytes": written, "last_id": rows[-1]["id"]}


def take_split(pool: Path, lang: str, split: str, out: Path) -> tuple[Path, dict]:
    """Copy a whole held-out split of the pool and dump it as text."""
    t = pq.read_table(sorted(glob.glob(str(pool / lang / f"{split}-*.parquet"))))
    d = out / lang
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, d / f"{split}-00000-of-00001.parquet", compression="zstd", row_group_size=5000, write_page_index=True)
    txt = out / f"_txt/{split}_{lang}.txt"
    with txt.open("w", encoding="utf-8") as fh:
        for x in t.column("text").to_pylist():
            fh.write(x.replace("\n", " ").strip() + "\n")
    return txt, {"docs": t.num_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--corpus-out", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tokenizer-v2"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument("--bytes-ja", type=int, default=400_000_000)
    parser.add_argument("--bytes-en", type=int, default=300_000_000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--body-params", type=int, default=264_300_000)
    parser.add_argument("--min-available-gb", type=float, default=10.0)
    args = parser.parse_args()
    start_memory_guard(args.min_available_gb)

    corpus = {}
    train, val, test = {}, {}, {}
    for lang, nbytes in (("ja", args.bytes_ja), ("en", args.bytes_en)):
        train[lang], corpus[f"{lang}/train"] = take_train(args.pool, lang, nbytes, args.corpus_out)
        val[lang], corpus[f"{lang}/validation"] = take_split(args.pool, lang, "validation", args.corpus_out)
        test[lang], corpus[f"{lang}/test"] = take_split(args.pool, lang, "test", args.corpus_out)
    (args.corpus_out / "corpus_summary.json").write_text(json.dumps(corpus, indent=1))
    print("CORPUS", json.dumps(corpus), flush=True)

    report = []
    for vocab in args.vocab_sizes:
        tok = build_tokenizer()
        trainer = BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL_TOKENS,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2, show_progress=False)
        tok.train([str(p) for p in train.values()], trainer=trainer)
        out = args.out_dir / f"v{vocab // 1024}k"
        out.mkdir(parents=True, exist_ok=True)
        PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
                                mask_token="<|mask|>", model_max_length=1024).save_pretrained(out)
        emb = vocab * args.hidden_size
        row = {"vocab_size": vocab, "path": str(out), "embedding_params": emb,
               "embedding_ratio_300m": emb / (emb + args.body_params),
               "validation": evaluate(tok, val, max_lines=10**9), "test": evaluate(tok, test, max_lines=10**9)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        report.append(row)
    (args.out_dir / "tokenizer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("TOKENIZER_V2_DONE")


if __name__ == "__main__":
    main()
