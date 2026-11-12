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
"""Build the deduplicated, decontaminated and split Freesia v2 pretraining pool (Wikipedia + FineWeb).

1 言語ずつ次を行い、`<out>/<lang>/{train,validation,test}-*.parquet`（列: id / text / source / source_id / url）を書く。

0. Web 文書の行単位の定型文除去（正規化後 `--boiler-min-chars` 文字以上の同じ行が `--boiler-min-docs` 件以上の
   Web 文書に現れたら、その行を全 Web 文書から削る。Wikipedia には適用しない）
1. 短い文書の除外（空白を除いた正規化後の文字数が `--min-chars` 未満）
2. 完全一致の重複（正規化後の本文）と MinHash によるほぼ重複（14 バンド × 8 行。ja は文字 5-gram、
   en は単語 5-gram）を 1 つのまとまりとして束ね、まとまりごとに 1 文書だけ残す（Wikipedia を優先）
3. 評価データ（JMTEB-lite・英語 MTEB のクエリと STS の文）との n-gram 一致（ja 25 文字・en 13 単語）がある文書を除く
4. validation / test を Web 2 : Wikipedia 1 で各 `--heldout-docs` 文書、ハッシュ順で選ぶ（残りが train）
5. train をソースごとに固定 seed でシャッフルし、Web 2 : Wikipedia 1 で交互に並べる

Usage:
    python build_freesia_corpus_v2.py --out ~/freesia-work/v2/pool --langs ja en
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash


HF_HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
SEED = 42
NUM_BANDS, ROWS = 14, 8
NUM_PERM = NUM_BANDS * ROWS
SHINGLE = 5
MAX_SHINGLES = 50_000
NGRAM = {"ja": 25, "en": 13}
WEB, WIKI = 0, 1
PRIORITY = {WIKI: 0, WEB: 1}  # lower wins inside a duplicate cluster

# (source name, glob, kind)
SOURCES = {
    "ja": [
        ("fineweb2-edu-ja", "datasets--hotchpotch--fineweb-2-edu-japanese/snapshots/*/sample_10BT/train-*.parquet", WEB),
        ("wikipedia-ja", "datasets--wikimedia--wikipedia/snapshots/*/20231101.ja/*.parquet", WIKI),
    ],
    "en": [
        ("fineweb-edu", "datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/00[0-2]_00000.parquet", WEB),
        ("wikipedia-en", "datasets--wikimedia--wikipedia/snapshots/*/20231101.en/*.parquet", WIKI),
    ],
}
SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()), ("source", pa.string()),
                    ("source_id", pa.string()), ("url", pa.string())])

WS = re.compile(r"\s+")
WORD = re.compile(r"\w+", re.U)
_rng = np.random.default_rng(SEED)
PERM_A = (_rng.integers(1, 2**63, NUM_PERM, dtype=np.uint64) | np.uint64(1))
PERM_B = _rng.integers(0, 2**63, NUM_PERM, dtype=np.uint64)
ROLL = np.uint64(0x100000001B3)


def norm(text: str) -> str:
    """NFKC + lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def rolling(ids: np.ndarray, k: int = SHINGLE) -> np.ndarray:
    """Hash every k consecutive ids into one uint64 (wrap-around arithmetic)."""
    n = ids.size - k + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(k):
            h = h * ROLL + ids[i : i + n]
    return h


def bands_of(shingles: np.ndarray) -> np.ndarray:
    """MinHash signature -> 14 band hashes."""
    sh = np.unique(shingles)
    if sh.size > MAX_SHINGLES:
        sh = sh[:MAX_SHINGLES]
    with np.errstate(over="ignore"):
        sig = ((PERM_A[:, None] * sh[None, :] + PERM_B[:, None]) >> np.uint64(16)).min(axis=1)
    sig = sig.reshape(NUM_BANDS, ROWS)
    return np.array([xxhash.xxh64_intdigest(sig[b].tobytes(), seed=b) for b in range(NUM_BANDS)], dtype=np.uint64)


# ---------- boilerplate lines ----------
_BOILER = np.zeros(0, dtype=np.uint64)  # sorted hashes of boilerplate lines
_BOILER_MIN_CHARS = 10


def line_hashes(text: str) -> tuple[list[str], np.ndarray]:
    """Split into lines and hash each normalized line (0 for lines shorter than the minimum)."""
    lines = text.split("\n")
    hs = np.zeros(len(lines), dtype=np.uint64)
    for i, ln in enumerate(lines):
        c = WS.sub("", norm(ln))
        if len(c) >= _BOILER_MIN_CHARS:
            hs[i] = xxhash.xxh64_intdigest(c.encode())
    return lines, hs


def strip_boiler(text: str, kind: int) -> tuple[str, int]:
    """Remove boilerplate lines from a web document; returns (text, removed line count)."""
    if kind != WEB or _BOILER.size == 0 or not text:
        return text, 0
    lines, hs = line_hashes(text)
    pos = np.searchsorted(_BOILER, hs)
    hit = (hs != 0) & (pos < _BOILER.size) & (_BOILER[np.minimum(pos, _BOILER.size - 1)] == hs)
    if not hit.any():
        return text, 0
    return "\n".join(ln for ln, h in zip(lines, hit) if not h), int(hit.sum())


def pass0(task):
    """Unique line hashes of every web document in one row group (document frequency counting)."""
    _, kind, _, path, rg = task
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"]).column(0).to_pylist()
    out = []
    for t in texts:
        _, hs = line_hashes(t or "")
        hs = np.unique(hs[hs != 0])
        if hs.size:
            out.append(hs)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


# ---------- worker (pass 1) ----------
_AUTO = None
_LANG = None
_MIN = None


def _init(lang: str, patterns: list[str], min_chars: int, boiler: np.ndarray) -> None:
    global _AUTO, _LANG, _MIN, _BOILER
    import ahocorasick

    _LANG, _MIN, _BOILER = lang, min_chars, boiler
    _AUTO = ahocorasick.Automaton()
    for i, p in enumerate(patterns):
        _AUTO.add_word(p, i)
    _AUTO.make_automaton()


def _contaminated(s: str) -> bool:
    for _ in _AUTO.iter(s):
        return True
    return False


def pass1(task):
    """Compute keys for one row group: length filter, exact hash, MinHash bands, contamination."""
    src_idx, kind, file_idx, path, rg = task
    t = pq.ParquetFile(path).read_row_group(rg, columns=["text"])
    texts = t.column(0).to_pylist()
    n = len(texts)
    stripped = np.zeros(n, dtype=np.int32)
    for i in range(n):
        texts[i], stripped[i] = strip_boiler(texts[i] or "", kind)
    short = np.zeros(n, dtype=bool)
    contam = np.zeros(n, dtype=bool)
    exact = np.zeros(n, dtype=np.uint64)
    bands = np.zeros((n, NUM_BANDS), dtype=np.uint64)
    for i, text in enumerate(texts):
        low = norm(text or "")
        compact = WS.sub("", low)
        if len(compact) < _MIN:
            short[i] = True
            continue
        exact[i] = xxhash.xxh64_intdigest(compact.encode())
        if _LANG == "ja":
            ids = np.frombuffer(compact.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
            contam[i] = _contaminated(compact)
        else:
            words = WORD.findall(low)
            if len(words) < SHINGLE:
                short[i] = True
                continue
            ids = np.fromiter((xxhash.xxh64_intdigest(w.encode()) for w in words), dtype=np.uint64, count=len(words))
            contam[i] = _contaminated(" " + " ".join(words) + " ")
        bands[i] = bands_of(rolling(ids))
    return src_idx, file_idx, rg, short, contam, exact, bands, stripped


# ---------- eval patterns ----------
def eval_patterns(lang: str) -> list[str]:
    """n-gram patterns of the evaluation queries / STS sentences (items shorter than n are skipped)."""
    lm = Path(__file__).resolve().parent
    sys.path.insert(0, str(lm))
    import bench_embed_models as B

    items = []
    if lang == "ja":
        for t in B.JA_RETRIEVAL:
            items += B.load_ja_retrieval(t)[0]
        for t in B.JA_STS:
            a, b, _ = B.load_ja_sts(t)
            items += a + b
        k = NGRAM["ja"]
        pats = set()
        for it in items:
            c = WS.sub("", norm(it))
            pats.update(c[i : i + k] for i in range(len(c) - k + 1))
    else:
        for t in B.EN_RETRIEVAL:
            items += B.load_en_retrieval(t)[0]
        for t in B.EN_STS:
            a, b, _ = B.load_en_sts(t)
            items += a + b
        k = NGRAM["en"]
        pats = set()
        for it in items:
            w = WORD.findall(norm(it))
            pats.update(" " + " ".join(w[i : i + k]) + " " for i in range(len(w) - k + 1))
    return sorted(pats)


# ---------- union-find ----------
def clusters(bands: np.ndarray) -> np.ndarray:
    """Union docs that share any band hash; returns the root index of each doc."""
    n = bands.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for b in range(NUM_BANDS):
        col = bands[:, b]
        order = np.argsort(col, kind="stable")
        sc = col[order]
        same = np.nonzero(sc[1:] == sc[:-1])[0]
        for j in same:
            x, y = find(order[j]), find(order[j + 1])
            if x != y:
                parent[max(x, y)] = min(x, y)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--langs", nargs="+", default=["ja", "en"])
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--heldout-docs", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--shard-docs", type=int, default=100_000)
    parser.add_argument("--boiler-min-docs", type=int, default=5)
    parser.add_argument("--boiler-min-chars", type=int, default=10)
    parser.add_argument("--limit-rowgroups", type=int, default=0, help="smoke test: row groups per source")
    args = parser.parse_args()

    for lang in args.langs:
        t0 = time.time()
        srcs = SOURCES[lang]
        files = [sorted(glob.glob(os.path.join(HF_HUB, g))) for _, g, _ in srcs]
        tasks = []
        for si, (_, _, kind) in enumerate(srcs):
            n_rg = 0
            for fi, f in enumerate(files[si]):
                for rg in range(pq.ParquetFile(f).num_row_groups):
                    if args.limit_rowgroups and n_rg >= args.limit_rowgroups:
                        break
                    tasks.append((si, kind, fi, f, rg)); n_rg += 1
        patterns = eval_patterns(lang)
        print(f"[{lang}] tasks={len(tasks)} eval_patterns={len(patterns)}", flush=True)

        # pass 0: boilerplate lines = lines (>= boiler_min_chars) seen in >= boiler_min_docs web documents
        global _BOILER, _BOILER_MIN_CHARS
        _BOILER_MIN_CHARS = args.boiler_min_chars
        with Pool(args.workers) as pool:
            hs = np.concatenate(list(pool.imap_unordered(pass0, [t for t in tasks if t[1] == WEB], chunksize=1)))
        vals, cnt = np.unique(hs, return_counts=True)
        _BOILER = np.sort(vals[cnt >= args.boiler_min_docs])
        del hs, vals, cnt
        print(f"[{lang}] boilerplate lines={_BOILER.size} {time.time() - t0:.0f}s", flush=True)

        parts = {}
        with Pool(args.workers, initializer=_init, initargs=(lang, patterns, args.min_chars, _BOILER)) as pool:
            for k, res in enumerate(pool.imap_unordered(pass1, tasks, chunksize=1)):
                parts[(res[0], res[1], res[2])] = res[3:]
                if k % 50 == 0:
                    print(f"[{lang}] pass1 {k}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
        keys = sorted(parts)
        short_parts = {k: parts[k][0] for k in keys}
        loc = np.concatenate([np.column_stack([np.full(len(parts[k][0]), v, dtype=np.int64) for v in k]
                                              + [np.arange(len(parts[k][0]), dtype=np.int64)]) for k in keys])
        short = np.concatenate([parts[k][0] for k in keys])
        contam = np.concatenate([parts[k][1] for k in keys])
        exact = np.concatenate([parts[k][2] for k in keys])
        bands = np.concatenate([parts[k][3] for k in keys])
        stripped = np.concatenate([parts[k][4] for k in keys])
        del parts
        src = loc[:, 0]
        kind = np.array([srcs[s][2] for s in src], dtype=np.int8)
        n_all = len(src)
        stats = {"docs_in": {srcs[s][0]: int((src == s).sum()) for s in range(len(srcs))},
                 "boilerplate": {"unique_lines": int(_BOILER.size), "lines_removed": int(stripped.sum()),
                                 "docs_touched": int((stripped > 0).sum())}}

        valid = np.nonzero(~short)[0]
        stats["short_removed"] = int(short.sum())
        ex = exact[valid]
        stats["exact_duplicates"] = int(valid.size - np.unique(ex).size)
        # contiguous row ranges per row group (keys are sorted, so loc follows the same order)
        lens = [len(short_parts[k]) for k in keys]
        offs = np.concatenate([[0], np.cumsum(lens)])
        # source ids for the stable tie-break / split / shuffle hashes; read only the id column
        sid = np.empty(n_all, dtype=object)
        for j, (s, f, r) in enumerate(keys):
            col = pq.ParquetFile(files[s][f]).read_row_group(r, columns=["id"]).column(0).to_pylist()
            sid[offs[j] : offs[j + 1]] = [str(x) for x in col]
        idh = np.fromiter((xxhash.xxh64_intdigest(f"{srcs[s][0]}:{i}".encode(), seed=SEED) for s, i in zip(src, sid)),
                          dtype=np.uint64, count=n_all)

        root = clusters(bands[valid])
        # representative per cluster: (priority, idh) minimum
        pri = np.array([PRIORITY[k] for k in kind[valid]], dtype=np.uint64)
        order = np.lexsort((idh[valid], pri, root))
        first = np.ones(order.size, dtype=bool)
        first[1:] = root[order][1:] != root[order][:-1]
        rep = valid[order[first]]
        stats["near_duplicates_removed"] = int(valid.size - rep.size - stats["exact_duplicates"])
        keep = rep[~contam[rep]]
        stats["decontaminated_removed"] = int(contam[rep].sum())
        stats["docs_kept"] = {srcs[s][0]: int((src[keep] == s).sum()) for s in range(len(srcs))}

        # heldout: Web 2 : Wikipedia 1, smallest split-hash first
        split_key = np.array([xxhash.xxh64_intdigest(int(h).to_bytes(8, "little"), seed=SEED + 1) for h in idh[keep]], dtype=np.uint64)
        shuf_key = np.array([xxhash.xxh64_intdigest(int(h).to_bytes(8, "little"), seed=SEED + 2) for h in idh[keep]], dtype=np.uint64)
        n_wiki = args.heldout_docs // 3
        n_web = args.heldout_docs - n_wiki
        assign = {}
        streams = {}
        for k_, quota in ((WEB, n_web), (WIKI, n_wiki)):
            m = np.nonzero(kind[keep] == k_)[0]
            m = m[np.argsort(split_key[m], kind="stable")]
            assign[("validation", k_)] = keep[m[:quota]]
            assign[("test", k_)] = keep[m[quota : 2 * quota]]
            rest = m[2 * quota :]
            streams[k_] = keep[rest[np.argsort(shuf_key[rest], kind="stable")]]

        def interleave(web, wiki):
            out, i, j = [], 0, 0
            while i < len(web) or j < len(wiki):
                for _ in range(2):
                    if i < len(web):
                        out.append(web[i]); i += 1
                if j < len(wiki):
                    out.append(wiki[j]); j += 1
            return np.array(out, dtype=np.int64)

        final = {
            "validation": interleave(assign[("validation", WEB)], assign[("validation", WIKI)]),
            "test": interleave(assign[("test", WEB)], assign[("test", WIKI)]),
            "train": interleave(streams[WEB], streams[WIKI]),
        }
        stats["split_docs"] = {k: int(v.size) for k, v in final.items()}
        print(f"[{lang}] {json.dumps(stats)}", flush=True)

        # pass 2: gather rows in final order, write shards
        split_names = ["train", "validation", "test"]
        pos_split = np.full(n_all, -1, dtype=np.int8)
        pos_idx = np.full(n_all, -1, dtype=np.int64)
        for si_, split in enumerate(split_names):
            pos_split[final[split]] = si_
            pos_idx[final[split]] = np.arange(final[split].size)
        out_dir = args.out / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        sizes = {s_: final[s_].size for s_ in final}
        buckets, writers = {}, {}

        def shard_name(split, b):
            n_sh = max(1, -(-sizes[split] // args.shard_docs))
            return out_dir / f".tmp-{split}-{b:05d}-of-{n_sh:05d}.parquet"

        def flush(key, rows):
            tbl = pa.table({"pos": [x[0] for x in rows], "text": [x[1] for x in rows], "source": [x[2] for x in rows],
                            "source_id": [x[3] for x in rows], "url": [x[4] for x in rows]})
            if key not in writers:
                writers[key] = pq.ParquetWriter(shard_name(*key), tbl.schema, compression="zstd")
            writers[key].write_table(tbl)

        for j, (s, f, r) in enumerate(keys):
            g = np.arange(offs[j], offs[j + 1])
            g = g[pos_split[g] >= 0]
            if g.size == 0:
                continue
            tb = pq.ParquetFile(files[s][f]).read_row_group(r, columns=["text", "id", "url"])
            texts, ids, urls = (tb.column(c).to_pylist() for c in ("text", "id", "url"))
            for gg in g:
                i = int(loc[gg, 3]); split = split_names[pos_split[gg]]; p = int(pos_idx[gg])
                text, _ = strip_boiler(texts[i] or "", srcs[s][2])
                buckets.setdefault((split, p // args.shard_docs), []).append((p, text, srcs[s][0], str(ids[i]), urls[i] or ""))
            for key in [k for k, v in buckets.items() if len(v) >= 20_000]:
                flush(key, buckets.pop(key))
            if j % 50 == 0:
                print(f"[{lang}] pass2 {j}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
        for key, rows in buckets.items():
            flush(key, rows)
        for w in writers.values():
            w.close()
        for key in sorted(writers):
            tmp = shard_name(*key)
            t = pq.read_table(tmp).sort_by("pos")
            ids = [f"{lang}-{split_}-{p:09d}" for split_, p in [(key[0], int(x)) for x in t.column("pos").to_pylist()]]
            t = pa.table({"id": ids, "text": t.column("text"), "source": t.column("source"),
                          "source_id": t.column("source_id"), "url": t.column("url")}, schema=SCHEMA)
            pq.write_table(t, out_dir / tmp.name[5:], compression="zstd", row_group_size=5000, write_page_index=True)
            tmp.unlink()
        stats["seconds"] = round(time.time() - t0)
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
        print(f"[{lang}] DONE {json.dumps(stats)}", flush=True)
    print("BUILD_V2_DONE")


if __name__ == "__main__":
    main()
