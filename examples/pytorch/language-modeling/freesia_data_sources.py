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
"""Pretraining corpus sources for Freesia (design §4.1).

事前学習コーパス（日英 1:1）の読み出しを 1 か所にまとめる。parquet は HF キャッシュから直接読む
（`download_freesia_data.py` で取得済みのファイルだけを使う）。
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator

import pyarrow.parquet as pq


HF_HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))

# language -> list of (name, glob pattern relative to the HF hub cache, text column)
SOURCES: dict[str, list[tuple[str, str, str]]] = {
    "ja": [
        ("fineweb2-edu-ja", "datasets--hotchpotch--fineweb-2-edu-japanese/snapshots/*/sample_10BT/train-*.parquet", "text"),
        ("wikipedia-ja", "datasets--wikimedia--wikipedia/snapshots/*/20231101.ja/*.parquet", "text"),
    ],
    "en": [
        ("fineweb-edu", "datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/*.parquet", "text"),
        ("wikipedia-en", "datasets--wikimedia--wikipedia/snapshots/*/20231101.en/*.parquet", "text"),
    ],
}


def source_files(pattern: str) -> list[str]:
    """Resolve a cache-relative glob into sorted absolute parquet paths.

    Args:
        pattern (str): Glob relative to the HF hub cache.

    Returns:
        list[str]: Sorted file paths (possibly empty if not downloaded yet).
    """
    return sorted(glob.glob(os.path.join(HF_HUB, pattern)))


def iter_texts(pattern: str, column: str = "text", batch_rows: int = 2048) -> Iterator[str]:
    """Stream non-empty texts from parquet files in a deterministic order.

    Args:
        pattern (str): Glob relative to the HF hub cache.
        column (str): Text column name.
        batch_rows (int): Rows per parquet batch.

    Yields:
        str: One document text.
    """
    for path in source_files(pattern):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[column]):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text


def iter_language(lang: str) -> Iterator[tuple[str, str]]:
    """Interleave the sources of a language document by document.

    Web text and Wikipedia are alternated so that any prefix of the stream has a similar mixture.

    Args:
        lang (str): `ja` or `en`.

    Yields:
        tuple[str, str]: `(source_name, text)`.
    """
    iters = [(name, iter_texts(pattern, col)) for name, pattern, col in SOURCES[lang]]
    # 2:1 web:wikipedia interleave (design §4.1: ~2B web + ~1B wikipedia per language)
    weights = {0: 2, 1: 1}
    alive = list(range(len(iters)))
    while alive:
        for idx in list(alive):
            name, it = iters[idx]
            for _ in range(weights.get(idx, 1)):
                try:
                    yield name, next(it)
                except StopIteration:
                    alive.remove(idx)
                    break
