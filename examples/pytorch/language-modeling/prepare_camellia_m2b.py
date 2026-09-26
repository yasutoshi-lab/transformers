#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE 2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the governing limitations under
# the License.
"""Prepare the Camellia pretrain corpus (design sec.4 / M2b, 60B or 40B tokens).

Pipeline per domain (idempotent — re-running skips finished work):
  1. dump:    write {domain}.txt (train) + val_{domain}.txt. Val is held out
              BEFORE packing (head of the stream; seeded shuffle for local
              Wikipedia). Wikipedia EN/JA is read from the local HF cache
              (Ranunculus-era download) — one epoch, 2-epoch repeat abolished.
  2. sample:  write {domain}_samples.txt (1000 evenly spaced docs) for the
              design sec.4 quality gate (目視確認 before packing).
  3. pack:    tokenize with the 128K tokenizer (M2a output), append the eos
              token (<|endoftext|>) per document — same convention as
              prepare_ranunculus_data.py — and write fixed 8192-token uint32
              frames, truncated to the per-domain token budget. Tokenization
              runs in bounded-memory sequential passes (2GB raw each) so a
              24B-token domain never needs >~4GB RAM.

Domain mix (design sec.4; --total_tokens 40e9 for the 2B fallback):
  EN 40%:  Wikipedia EN (local, ~5.5B tokens) + FineWeb-Edu sample-10BT
  JA 30%:  Wikipedia JA (local, ~1.2B) + fineweb-2-edu-japanese
  de/fr/es/zh/ko 25% (5 x 5%): FineWeb2 per-language subsets
  code 5%: starcoderdata python + markdown (parquet data_files glob)

Wikipedia is consumed in ONE epoch. If the wiki text yields fewer tokens than
the sec.4 "9.5B" figure for EN, the FineWeb part absorbs the remainder
(2026-09-26 measurement: wiki EN text = 19.6GB -> ~5.5B tokens at 128K).
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset

GB = 1024 ** 3
SEED = 42
WIKI_DUMP_VERSION = "20231101"
SAMPLE_DOCS = 1000            # design sec.4 quality gate
PASS_RAW_BYTES = 2 * GB       # per-pass tokenization window (bounds RAM)

# --- domain plan ----------------------------------------------------------------
# frac: share of --total_tokens. wiki: local wikipedia lang (None = web only).
# web / web2: (dataset, config-or-parquet-glob, dump byte cap). The byte cap is
# a safety margin over the token budget (bytes/token varies by language); the
# pack stage truncates to the exact token budget, so excess raw is harmless.
DOMAINS: dict[str, dict] = {
    "en": {
        "frac": 0.40,
        "wiki": "en",
        "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", 70 * GB),
    },
    "ja": {
        "frac": 0.30,
        "wiki": "ja",
        "web": ("hotchpotch/fineweb-2-edu-japanese", None, 62 * GB),
    },
    "deu_Latn": {"frac": 0.05, "wiki": None,
                 "web": ("HuggingFaceFW/fineweb-2", "deu_Latn", 13 * GB)},
    "fra_Latn": {"frac": 0.05, "wiki": None,
                 "web": ("HuggingFaceFW/fineweb-2", "fra_Latn", 13 * GB)},
    "spa_Latn": {"frac": 0.05, "wiki": None,
                 "web": ("HuggingFaceFW/fineweb-2", "spa_Latn", 13 * GB)},
    "cmn_Hani": {"frac": 0.05, "wiki": None,
                 "web": ("HuggingFaceFW/fineweb-2", "cmn_Hani", 17 * GB)},
    "kor_Hang": {"frac": 0.05, "wiki": None,
                 "web": ("HuggingFaceFW/fineweb-2", "kor_Hang", 16 * GB)},
    "code": {
        "frac": 0.05,
        "wiki": None,
        "web": ("bigcode/starcoderdata", "python/train-*.parquet", 7 * GB),
        "web2": ("bigcode/starcoderdata", "markdown/train-*.parquet", 7 * GB),
    },
}

_TEXT_COLS = ("text", "content")  # starcoderdata uses `content`

# module-level encoder hook, set in main() before any Pool (fork inherits it)
_ENC = None


# --- raw text helpers -----------------------------------------------------------

def _doc_line(row) -> str:
    for col in _TEXT_COLS:
        v = row.get(col)
        if v:
            return v.replace("\n", " ") + "\n"
    return ""


# --- dump stage -----------------------------------------------------------------

def dump_wiki(lang: str, out_train: Path, out_val: Path, val_bytes: int) -> None:
    """Local Wikipedia (HF cache) -> val (seeded-shuffle head) + train (1 epoch)."""
    if out_train.exists() and out_train.stat().st_size > 0 and out_val.exists():
        print(f"[dump] skip (exists): wiki {lang}", flush=True)
        return
    cfg = f"{WIKI_DUMP_VERSION}.{lang}"
    print(f"[dump] loading local wikipedia {cfg} ...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train").shuffle(seed=SEED)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(ds)
    val_written = 0
    with out_val.open("w", encoding="utf-8") as fv:
        for row in rows:
            line = _doc_line(row)
            if not line:
                continue
            fv.write(line)
            val_written += len(line.encode("utf-8"))
            if val_written >= val_bytes:
                break
    n = 0
    with out_train.open("w", encoding="utf-8") as ft:
        for row in rows:  # continues after the val head
            line = _doc_line(row)
            if not line:
                continue
            ft.write(line)
            n += 1
            if n % 1_000_000 == 0:
                print(f"[dump] wiki {lang} train: {n} docs", flush=True)
    print(f"[dump] wiki {lang}: val={val_written / GB:.2f}GB train={n} docs", flush=True)


def _stream_source(dataset: str, config: str | None):
    if config is None:
        return load_dataset(dataset, split="train", streaming=True)
    if "/" in config:  # parquet data_files glob (starcoderdata)
        return load_dataset(dataset, data_files={"train": config},
                            split="train", streaming=True)
    return load_dataset(dataset, config, split="train", streaming=True)


def dump_web(dataset: str, config: str | None, out_train: Path, out_val: Path,
             val_bytes: int, max_bytes: int) -> None:
    """Stream one web corpus: head of the stream -> val, then train up to cap."""
    if out_train.exists() and out_train.stat().st_size > 0 and out_val.exists():
        print(f"[dump] skip (exists): {dataset} {config or ''}", flush=True)
        return
    print(f"[dump] {dataset} {config or ''} -> {out_train.name} (cap {max_bytes / GB:.0f}GB)",
          flush=True)
    ds = _stream_source(dataset, config)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = iter(ds)
        val_written = 0
        with out_val.open("w", encoding="utf-8") as fv:
            for row in rows:
                line = _doc_line(row)
                if not line:
                    continue
                fv.write(line)
                val_written += len(line.encode("utf-8"))
                if val_written >= val_bytes:
                    break
        written = 0
        n = 0
        with out_train.open("w", encoding="utf-8") as ft:
            for row in rows:  # continues after the val head
                line = _doc_line(row)
                if not line:
                    continue
                ft.write(line)
                written += len(line.encode("utf-8"))
                n += 1
                if n % 2_000_000 == 0:
                    print(f"[dump] {dataset} {config or ''}: {written / GB:.1f}GB ({n} docs)",
                          flush=True)
                if written >= max_bytes:
                    break
    finally:
        try:
            ds.close()
        except Exception:
            pass
    print(f"[dump] done {dataset} {config or ''}: val={val_written / GB:.2f}GB "
          f"train={written / GB:.1f}GB", flush=True)


def dump_web_append(dataset: str, config: str | None, out_train: Path,
                    max_bytes: int) -> None:
    """Append a second web source to an existing train file (val already dumped)."""
    marker = out_train.parent / f".{out_train.stem}+{dataset.split('/')[0]}+" \
        f"{(config or '').split('/')[0]}.done"
    if marker.exists():
        print(f"[dump] skip (appended): {dataset} {config or ''}", flush=True)
        return
    size0 = out_train.stat().st_size
    ds = _stream_source(dataset, config)
    try:
        with out_train.open("a", encoding="utf-8") as ft:
            n = 0
            for row in ds:
                line = _doc_line(row)
                if not line:
                    continue
                ft.write(line)
                n += len(line.encode("utf-8"))
                if n >= max_bytes:
                    break
    finally:
        try:
            ds.close()
        except Exception:
            pass
    marker.write_text("")
    print(f"[dump] appended {dataset} {config or ''}: "
          f"+{(out_train.stat().st_size - size0) / GB:.1f}GB", flush=True)


def dump_domain(domain: str, plan: dict, raw_dir: Path, val_bytes: int) -> None:
    train_p = raw_dir / f"{domain}.txt"
    val_p = raw_dir / f"val_{domain}.txt"
    if plan.get("wiki"):
        dump_wiki(plan["wiki"], train_p, val_p, val_bytes)
        ds, cfg, cap = plan["web"]
        dump_web_append(ds, cfg, train_p, cap)
    else:
        ds, cfg, cap = plan["web"]
        dump_web(ds, cfg, train_p, val_p, val_bytes, cap)
        if plan.get("web2"):
            ds2, cfg2, cap2 = plan["web2"]
            dump_web_append(ds2, cfg2, train_p, cap2)
    print(f"[dump] domain {domain}: {train_p.stat().st_size / GB:.1f}GB train ready",
          flush=True)


# --- quality gate (design sec.4: 1000 docs per corpus, visual check) -------------

def sample_domain(domain: str, raw_dir: Path) -> Path:
    src = raw_dir / f"{domain}.txt"
    out = raw_dir / f"{domain}_samples.txt"
    if out.exists():
        return out
    size = src.stat().st_size
    step = max(size // SAMPLE_DOCS, 1)
    parts = []
    with src.open("rb") as f:
        for i in range(0, size, step):
            f.seek(i)
            chunk = f.read(8192)
            nl = chunk.find(b"\n")
            if nl > 0:
                parts.append(chunk[:nl].decode("utf-8", errors="ignore"))
                if len(parts) >= SAMPLE_DOCS:
                    break
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[sample] {out.name}: {len(parts)} docs for visual check", flush=True)
    return out


# --- tokenize + pack (bounded-memory sequential passes) --------------------------

def _enc_chunk(args):
    """Tokenize [offset, offset+size) of one raw file; returns uint32 array."""
    path, offset, size = args
    out = []
    total = 0
    with open(path, "rb") as f:
        f.seek(offset)
        if offset > 0:
            f.readline()  # drop the partial line at the chunk boundary
        end = offset + size
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            ids = _ENC(text)
            arr = np.fromiter(ids, dtype=np.uint32)
            arr = np.append(arr, _EOS)
            out.append(arr)
            total += arr.size
            if total > 64 * 1024 * 1024:  # ~64M tokens (256MB) per worker buffer
                break
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint32)


def pack_file(path: Path, out_bin: Path, seq_len: int, budget: int,
              num_proc: int) -> int:
    """Tokenize a raw text file into 8192-token uint32 frames up to `budget`.

    Crash-safe: each pass appends complete frames directly to out_bin and
    records the raw offset in .progress; on resume, a torn tail is truncated
    to a frame boundary and the last pass re-reads from its start (no dup).
    """
    frames_total = (budget // seq_len) * seq_len
    frame_bytes = seq_len * 4
    prog = out_bin.with_suffix(".progress")
    if out_bin.exists() and out_bin.stat().st_size > 0:
        # truncate a torn tail from an interrupted write
        sz = out_bin.stat().st_size
        if sz % frame_bytes:
            with out_bin.open("r+b") as f:
                f.truncate(sz - sz % frame_bytes)
        have = out_bin.stat().st_size // 4
        if have >= frames_total:
            prog.unlink(missing_ok=True)
            print(f"[pack] skip (complete): {out_bin.name} ({have / 1e9:.1f}B)", flush=True)
            return have
        raw_off = int(prog.read_text()) if prog.exists() else 0
        print(f"[pack] resume {out_bin.name}: have={have / 1e9:.2f}B raw_off={raw_off}",
              flush=True)
    else:
        have = 0
        raw_off = 0
    size = path.stat().st_size
    t0 = time.time()
    while have < frames_total and raw_off < size:
        pass_size = min(PASS_RAW_BYTES, size - raw_off,
                        int((frames_total - have) * 4.2))  # bytes ~ tokens*4.2
        bounds = np.linspace(raw_off, raw_off + pass_size, num_proc + 1, dtype=np.int64)
        tasks = [(str(path), int(a), int(b))
                 for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with Pool(num_proc) as pool:
            arrays = list(pool.imap_unordered(_enc_chunk, tasks))
        flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint32)
        take = flat[: frames_total - have]
        with out_bin.open("ab") as f:
            f.write(take.tobytes())
        have += take.size
        raw_off += pass_size
        prog.write_text(str(raw_off))
        print(f"[pack] {out_bin.name}: {have / 1e9:.2f}B / {frames_total / 1e9:.1f}B "
              f"({time.time() - t0:.0f}s)", flush=True)
    prog.unlink(missing_ok=True)
    print(f"[pack] done {out_bin.name}: {have / 1e9:.2f}B tokens "
          f"({have / seq_len} frames)", flush=True)
    return have


# --- main ------------------------------------------------------------------------

def main() -> None:
    global _ENC, _EOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dump", "sample", "pack", "all"], default="all")
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--tokenizer_dir", type=Path, default=Path("artifacts-1/tokenizer"))
    ap.add_argument("--raw_dir", type=Path, default=Path("artifacts-1/m2b-raw"))
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts-1/packed"))
    ap.add_argument("--total_tokens", type=float, default=60e9)
    ap.add_argument("--val_tokens", type=int, default=10_000_000)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--num_proc", type=int, default=32)
    args = ap.parse_args()

    budgets = {d: int(DOMAINS[d]["frac"] * args.total_tokens) for d in args.domains}
    print(f"[plan] total={args.total_tokens / 1e9:.0f}B tokens "
          f"(wiki EN/JA 1 epoch, web absorbs the remainder)")
    for d in args.domains:
        print(f"  {d}: budget={budgets[d] / 1e9:.2f}B")

    val_bytes = int(args.val_tokens * 4)  # ~10M tokens * ~4 bytes/token

    if args.stage in ("dump", "all"):
        for d in args.domains:
            dump_domain(d, DOMAINS[d], args.raw_dir, val_bytes)
        print("[dump] all domains done", flush=True)

    if args.stage in ("sample", "all"):
        for d in args.domains:
            sample_domain(d, args.raw_dir)
        print("[sample] done — STOP and visually check *_samples.txt before packing "
              "(design sec.4 quality gate)", flush=True)

    if args.stage in ("pack", "all"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
        _ENC = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
        _EOS = np.uint32(tok.eos_token_id)  # M2a 128K: eos =  (id 127744)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        grand = 0
        for d in args.domains:
            grand += pack_file(args.raw_dir / f"{d}.txt",
                               args.out_dir / f"train_{d}.bin",
                               args.seq_len, budgets[d], args.num_proc)
            vp = args.raw_dir / f"val_{d}.txt"
            if vp.exists():
                pack_file(vp, args.out_dir / f"val_{d}.bin",
                          args.seq_len, args.val_tokens, max(4, args.num_proc // 4))
        print(f"[pack] ALL DONE: {grand / 1e9:.2f}B train tokens in {args.out_dir}")


if __name__ == "__main__":
    main()
