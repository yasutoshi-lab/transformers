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
"""Train Freesia byte-level BPE tokenizers and compare vocab sizes (design §2.1, stage 0).

日英均等のコーパスから byte-level BPE を学習し、候補の語彙サイズ（既定 32,768 と 49,152）ごとに
日英の圧縮効率（1 トークンあたりの文字数）と埋め込み行列の比率を測る。

Usage:
    python train_freesia_tokenizer.py --out-dir artifacts/tokenizer --vocab-sizes 32768 49152
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from itertools import islice
from pathlib import Path

from freesia_data_sources import iter_language
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|mask|>", *[f"<|reserved_{i}|>" for i in range(12)]]
# Japanese has no spaces, so a plain `\p{L}+` rule turns whole sentences into single pre-tokens, which makes
# BPE training explode in memory on ws2-arc (61GiB). Split Japanese by script (kanji / hiragana / katakana)
# with a length cap first; the remaining rules follow the usual byte-level BPE pattern.
SPLIT_REGEX = (
    r"\p{Han}{1,6}|\p{Hiragana}{1,6}|[\p{Katakana}ー]{1,10}|"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+[\p{L}&&[^\p{Han}\p{Hiragana}\p{Katakana}ー]]+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)


def build_tokenizer() -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer (NFC, no lowercasing).

    Returns:
        Tokenizer: Untrained tokenizer.
    """
    tok = Tokenizer(BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=Regex(SPLIT_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def dump_corpus(lang: str, path: Path, max_bytes: int, skip_docs: int = 0) -> Path:
    """Write a language sample (newline separated, one doc per line) to disk once.

    Args:
        lang (str): `ja` or `en`.
        path (Path): Output file.
        max_bytes (int): UTF-8 byte budget.
        skip_docs (int): Documents to skip first (used to make a held-out sample).

    Returns:
        Path: The written (or already existing) file.
    """
    if path.exists() and path.stat().st_size >= max_bytes * 0.95:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for _, text in islice(iter_language(lang), skip_docs, None):
            line = text.replace("\n", " ").strip()
            if not line:
                continue
            f.write(line + "\n")
            written += len(line.encode("utf-8")) + 1
            if written >= max_bytes:
                break
    return path


def evaluate(tok: Tokenizer, heldout: dict[str, Path], max_lines: int = 20000) -> dict[str, float]:
    """Measure characters per token on held-out text.

    Args:
        tok (Tokenizer): Trained tokenizer.
        heldout (dict[str, Path]): Language -> held-out file.
        max_lines (int): Lines used per language.

    Returns:
        dict[str, float]: `chars_per_token_<lang>` for each language.
    """
    out = {}
    for lang, path in heldout.items():
        lines = path.read_text(encoding="utf-8").splitlines()[:max_lines]
        chars = sum(len(x) for x in lines)
        toks = sum(len(e.ids) for e in tok.encode_batch(lines))
        out[f"chars_per_token_{lang}"] = chars / max(toks, 1)
    return out


def start_memory_guard(min_available_gb: float) -> None:
    """Abort the process before the host runs out of memory.

    ws2-arc は RAM 61GiB を他のサービス（VM など）と共有するので、BPE 学習がメモリを使い切って
    OOM killer が他のプロセスを巻き込む前に、自分だけを終了させる。

    Args:
        min_available_gb (float): Abort when MemAvailable falls below this value.
    """

    def _watch():
        while True:
            with open("/proc/meminfo") as f:
                avail_kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable"))
            if avail_kb / 1024**2 < min_available_gb:
                print(f"[memory-guard] MemAvailable={avail_kb / 1024**2:.1f}GiB < {min_available_gb}GiB, aborting", flush=True)
                os._exit(3)
            time.sleep(2)

    threading.Thread(target=_watch, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/tokenizer_corpus_small"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument("--bytes-ja", type=int, default=400_000_000)
    parser.add_argument("--bytes-en", type=int, default=300_000_000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--body-params", type=int, default=264_300_000, help="Freesia-300M body params")
    parser.add_argument("--min-available-gb", type=float, default=10.0)
    args = parser.parse_args()
    start_memory_guard(args.min_available_gb)

    train = {
        "ja": dump_corpus("ja", args.work_dir / "train_ja.txt", args.bytes_ja),
        "en": dump_corpus("en", args.work_dir / "train_en.txt", args.bytes_en),
    }
    # held-out: documents far beyond the training sample
    heldout = {
        "ja": dump_corpus("ja", args.work_dir / "heldout_ja.txt", 20_000_000, skip_docs=3_000_000),
        "en": dump_corpus("en", args.work_dir / "heldout_en.txt", 20_000_000, skip_docs=3_000_000),
    }

    report = []
    for vocab in args.vocab_sizes:
        tok = build_tokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=True,
        )
        tok.train([str(p) for p in train.values()], trainer=trainer)
        out = args.out_dir / f"v{vocab // 1024}k"
        out.mkdir(parents=True, exist_ok=True)
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            pad_token="<|pad|>",
            mask_token="<|mask|>",
            model_max_length=1024,
        )
        fast.save_pretrained(out)
        emb = vocab * args.hidden_size
        row = {"vocab_size": vocab, "path": str(out), "embedding_params": emb,
               "embedding_ratio_300m": emb / (emb + args.body_params), **evaluate(tok, heldout)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        report.append(row)
    (args.out_dir / "tokenizer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
