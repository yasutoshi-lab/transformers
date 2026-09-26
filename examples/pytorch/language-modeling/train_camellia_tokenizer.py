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
# See the License for the governing permissions and
# limitations under the License.
"""Train a 128K BPE + byte-fallback tokenizer for Camellia (design sec.4 / M2a).

Spec:
  * vocab_size = 128_000 = 127_744 BPE + 256 reserved specials
  * Byte-level pre-tokenizer + NFC normalization (no lowercasing)
  * Training corpus follows the design sec.4 pretrain mix (as a sample):
      EN 40% / JA 30% / de/fr/es/zh/ko 25% / code 5%
    Byte budgets approximate the token mix (bytes/token differs per
    language; the M2a gate is per-language fertility vs EN, not the exact
    ratio).
  * Fertility gate (M2a): tokens-per-1KB for JA / de / fr / es / zh / ko /
    code must stay within 2x the EN value (design sec.4 / notes).

Sources (all streamed from the HF Hub, `text` column):
  EN:  wikimedia/wikipedia (20231101.en) + HuggingFaceFW/fineweb-edu (sample-10BT)
  JA:  wikimedia/wikipedia (20231101.ja) + hotchpotch/fineweb-2-edu-japanese
  de:  wikimedia/wikipedia (20231101.de) + HuggingFaceFW/fineweb-2 (deu_Latn)
  fr:  wikimedia/wikipedia (20231101.fr) + HuggingFaceFW/fineweb-2 (fra_Latn)
  es:  wikimedia/wikipedia (20231101.es) + HuggingFaceFW/fineweb-2 (spa_Latn)
  zh:  wikimedia/wikipedia (20231101.zh) + HuggingFaceFW/fineweb-2 (cmn_Hani)
  ko:  wikimedia/wikipedia (20231101.ko) + HuggingFaceFW/fineweb-2 (kor_Hang)
  code: bigcode/starcoderdata (Python + Markdown)

Run (tmux, long job):
  tmux new -d -s camellia-m2a "python train_camellia_tokenizer.py 2>&1 | tee /tmp/camellia-m2a.log"\n"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

# Same 256 reserved slots as the Ranunculus tokenizer (design inheritance).
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
    "<|tool_call|>",
    "</tool_call>",
    *[f"<|reserved_{i}|>" for i in range(245)],
]
assert len(SPECIAL_TOKENS) == 256, "design sec.4 requires exactly 256 reserved slots"

GB = 1024 ** 3
MB = 1024 ** 2

# (dataset, config, max_bytes, language_filter) per domain.
# Budgets approximate the design sec.4 token mix (EN 40 / JA 30 / ML 25 / code 5).
SOURCES: dict[str, list[tuple[str, str | None, int, str | None]]] = {
    "en": [
        ("wikimedia/wikipedia", "20231101.en", 0.4 * GB, None),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", 2.6 * GB, None),
    ],
    "ja": [
        ("wikimedia/wikipedia", "20231101.ja", 0.3 * GB, None),
        ("hotchpotch/fineweb-2-edu-japanese", None, 1.7 * GB, None),
    ],
    "de": [
        ("wikimedia/wikipedia", "20231101.de", 0.2 * GB, None),
        ("HuggingFaceFW/fineweb-2", "deu_Latn", 0.8 * GB, None),
    ],
    "fr": [
        ("wikimedia/wikipedia", "20231101.fr", 0.2 * GB, None),
        ("HuggingFaceFW/fineweb-2", "fra_Latn", 0.8 * GB, None),
    ],
    "es": [
        ("wikimedia/wikipedia", "20231101.es", 0.2 * GB, None),
        ("HuggingFaceFW/fineweb-2", "spa_Latn", 0.8 * GB, None),
    ],
    "zh": [
        ("wikimedia/wikipedia", "20231101.zh", 0.2 * GB, None),
        ("HuggingFaceFW/fineweb-2", "cmn_Hani", 0.8 * GB, None),
    ],
    "ko": [
        ("wikimedia/wikipedia", "20231101.ko", 0.2 * GB, None),
        ("HuggingFaceFW/fineweb-2", "kor_Hang", 0.8 * GB, None),
    ],
    "code": [
        ("bigcode/starcoderdata", None, 0.3 * GB, "Python"),
        ("bigcode/starcoderdata", None, 0.3 * GB, "Markdown"),
    ],
}

# GPT-style pre-tokenizer pattern (same as the Ranunculus tokenizer).
_PRETOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(_PRETOKEN_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _stream(dataset: str, config: str | None, language: str | None):
    if config is not None:
        return load_dataset(dataset, config, split="train", streaming=True)
    return load_dataset(dataset, split="train", streaming=True)


def dump_source(dataset: str, config: str | None, out_path: Path, max_bytes: int,
                language: str | None = None) -> None:
    """Stream one dataset (optionally filtered by `language`) into a text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size >= max_bytes:
        print(f"[corpus] skip (exists): {out_path}", flush=True)
        return
    label = f"{dataset} {config or ''} {language or ''}".strip()
    print(f"[corpus] {label} -> {out_path} (cap {max_bytes / GB:.2f} GB)", flush=True)
    ds = _stream(dataset, config, language)
    written = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                if language is not None and row.get("language") != language:
                    continue
                text = row.get("text", "")
                if not text:
                    continue
                line = text.replace("\n", " ") + "\n"
                f.write(line)
                written += len(line.encode("utf-8"))
                if written >= max_bytes:
                    break
    finally:
        try:
            ds.close()
        except Exception:
            pass
    print(f"[corpus] wrote {written / GB:.2f} GB", flush=True)


def fertility(tokenizer: PreTrainedTokenizerFast, corpus_dir: Path) -> dict[str, float]:
    """tokens per 1KB of raw text, per domain file (first 5MB sample)."""
    out: dict[str, float] = {}
    for path in sorted(corpus_dir.glob("*.txt")):
        sample = path.read_bytes()[: 5 * MB].decode("utf-8", errors="ignore")
        n_tokens = len(tokenizer.encode(sample, add_special_tokens=False))
        kb = len(sample.encode("utf-8", errors="ignore")) / 1000
        out[path.stem] = n_tokens / max(kb, 1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts-1/tokenizer-128k"))
    parser.add_argument("--corpus_dir", type=Path, default=Path("artifacts-1/tokenizer-corpus-128k"))
    parser.add_argument("--vocab_size_core", type=int, default=127_744)
    parser.add_argument("--skip_dump", action="store_true", help="reuse existing corpus files")
    args = parser.parse_args()

    if not args.skip_dump:
        for domain, sources in SOURCES.items():
            for i, (dataset, config, max_bytes, language) in enumerate(sources):
                dump_source(dataset, config, args.corpus_dir / f"{domain}_{i}.txt", max_bytes, language)
    corpus_files = sorted(str(p) for p in args.corpus_dir.glob("*.txt"))
    expected = sum(len(v) for v in SOURCES.values())
    assert len(corpus_files) == expected, f"expected {expected} corpus files, found {len(corpus_files)}"

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size_core,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[bpe] training {args.vocab_size_core} on {len(corpus_files)} files", flush=True)
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        bos_token=None,
        additional_special_tokens=SPECIAL_TOKENS[2:],
    )
    assert len(fast) == 128_000, len(fast)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.out_dir)
    print(f"[done] saved to {args.out_dir}, vocab_size={len(fast)}", flush=True)

    # M2a gate: fertility per language / code vs EN (within 2x EN).
    fert = fertility(fast, args.corpus_dir)
    base = fert.get("en_0") or fert.get("en_1")
    report = {
        "base_en_tokens_per_kb": base,
        "fertility_tokens_per_kb": fert,
        "ratios_vs_en": {k: round(v / base, 4) for k, v in fert.items() if not k.startswith("en_")},
    }
    (args.out_dir / "fertility.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    over = [k for k, r in report["ratios_vs_en"].items() if r > 2.0]
    if over:
        print(f"[gate] M2a FAIL: fertility > 2x EN for {over}", flush=True)
    else:
        print("[gate] M2a PASS: all languages/code within 2x EN fertility", flush=True)


if __name__ == "__main__":
    main()
