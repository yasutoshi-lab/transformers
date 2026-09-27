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
"""Tokenize the Freesia pretraining corpus into flat uint16 token streams (design §4.1).

日英それぞれ、文書ごとに `<|eos|>` を付けて 1 本の uint16 配列（.bin）へ書き出す。
先頭 `--val-docs` 文書は検証用に別ファイルへ分ける。途中で止まっても、再実行すると既存の .bin を
読み飛ばして続きから書く（`<lang>.progress.json` に処理済み文書数とトークン数を記録する）。

Usage:
    python prepare_freesia_data.py --tokenizer artifacts/tokenizer/v32k --out-dir artifacts/packed \
        --tokens-per-lang 3050000000
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from freesia_data_sources import iter_language

from transformers import AutoTokenizer


_TOK = None
_EOS = None


def _init(tokenizer_dir: str) -> None:
    global _TOK, _EOS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir)
    _EOS = _TOK.eos_token_id


def _encode(texts: list[str]) -> np.ndarray:
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    return np.concatenate([np.asarray(x + [_EOS], dtype=np.uint16) for x in ids])


def _chunks(it, n):
    while True:
        block = list(islice(it, n))
        if not block:
            return
        yield block


def prepare(lang: str, args) -> dict:
    """Tokenize one language until the token budget is reached.

    Args:
        lang (str): `ja` or `en`.
        args: Parsed CLI arguments.

    Returns:
        dict: Progress record (documents and tokens written).
    """
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / f"{lang}.progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"docs": 0, "tokens": 0, "val_tokens": 0}
    stream = (text for _, text in iter_language(lang))

    if prog["docs"] == 0:
        val_docs = list(islice(stream, args.val_docs))
        with Pool(4, initializer=_init, initargs=(str(args.tokenizer),)) as pool:
            arrays = pool.map(_encode, [val_docs[i : i + 256] for i in range(0, len(val_docs), 256)])
        val = np.concatenate(arrays)
        val.tofile(out / f"{lang}_val.bin")
        prog.update(docs=args.val_docs, val_tokens=int(val.size))
        prog_path.write_text(json.dumps(prog))
    else:
        stream = islice(stream, prog["docs"], None)
        train_path = out / f"{lang}_train.bin"
        if train_path.exists():  # drop a partially written tail from an interrupted run
            os.truncate(train_path, prog["tokens"] * 2)

    with open(out / f"{lang}_train.bin", "ab") as f, Pool(
        args.workers, initializer=_init, initargs=(str(args.tokenizer),)
    ) as pool:
        batches = _chunks(_chunks(stream, 256), args.workers * 4)
        for group in batches:
            for arr in pool.map(_encode, group):
                arr.tofile(f)
                prog["tokens"] += int(arr.size)
            prog["docs"] += sum(len(g) for g in group)
            f.flush()
            prog_path.write_text(json.dumps(prog))
            print(f"[{lang}] docs={prog['docs']:,} tokens={prog['tokens']:,}", flush=True)
            if prog["tokens"] >= args.tokens_per_lang:
                break
    return prog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/packed"))
    parser.add_argument("--tokens-per-lang", type=int, default=3_050_000_000)
    parser.add_argument("--val-docs", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--langs", nargs="+", default=["ja", "en"])
    args = parser.parse_args()
    summary = {lang: prepare(lang, args) for lang in args.langs}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
