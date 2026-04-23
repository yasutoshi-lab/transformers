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
"""Train a 128K BPE + byte-fallback tokenizer for Ranunculus (design §4).

Design choices:
  * vocab_size = 128_000 = 127_744 BPE + 256 reserved specials
  * Byte-level pre-tokenizer + NFC normalization (no lowercasing)
  * Balanced 500M tokens × {en, de, ja, zh} corpus for BPE training
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast


LANGUAGES = ("en", "de", "ja", "zh")

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    *[f"<|reserved_{i}|>" for i in range(245)],
]
assert len(SPECIAL_TOKENS) == 256, "design §4 requires exactly 256 reserved slots"


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"
                    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def dump_corpus(dataset_config: str, out_path: Path, max_bytes: int) -> Path:
    """Stream Wikipedia for a language into a UTF-8 newline-separated file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size >= max_bytes:
        return out_path

    ds = load_dataset("wikimedia/wikipedia", dataset_config, split="train", streaming=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            line = text.replace("\n", " ") + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            if written >= max_bytes:
                break
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--corpus_dir", type=Path, default=Path("artifacts/tokenizer_corpus"))
    parser.add_argument("--dump_version", default="20231101")
    parser.add_argument(
        "--bytes_per_language",
        type=int,
        default=2 * 1024 * 1024 * 1024,  # ~2GB per language ≈ 500M tokens target.
    )
    parser.add_argument("--vocab_size_core", type=int, default=127_744)
    args = parser.parse_args()

    corpus_files: list[str] = []
    for lang in LANGUAGES:
        cfg = f"{args.dump_version}.{lang}"
        out = args.corpus_dir / f"{lang}.txt"
        print(f"[corpus] {cfg} -> {out}")
        dump_corpus(cfg, out, args.bytes_per_language)
        corpus_files.append(str(out))

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size_core,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[bpe] training on {corpus_files}")
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        bos_token=None,
        additional_special_tokens=SPECIAL_TOKENS[2:],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.out_dir)
    # Sanity checks from design §4 checklist.
    # vocab_size returns only the BPE core vocab; len() includes added specials.
    assert len(fast) == 128_000, len(fast)
    print(f"[done] saved to {args.out_dir}, vocab_size={len(fast)}")


if __name__ == "__main__":
    main()
