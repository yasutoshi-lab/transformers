"""Rebuild the Freesia v2 test split with a stricter near-duplicate filter.

Per language:
  1. Index every document in freesia-pt train (the documents actually used for pretraining) and validation
     with a low-threshold MinHash LSH (100 permutations, 20 bands x 5 rows).
  2. Candidates are the current test documents, followed by reserve documents from the unused tail of the pool
     train split. Any candidate that shares a band with an indexed document is checked with the exact shingle
     Jaccard, and the candidate is rejected if J >= 0.5. Candidates are also checked against each other.
  3. The test split is refilled to the same size and ratio (2 web : 1 Wikipedia). Surviving original documents
     come first, then reserve documents. The split is written to pool / freesia-pt / tokenizer-pt, and the
     token file is rewritten.
"""
import collections, glob, json, os, sys
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, LM)
import build_freesia_corpus_v2 as V  # noqa: E402

W = os.path.expanduser("~/freesia-work/v2")
PT, POOL, TOKPT, PACKED = f"{W}/freesia-pt", f"{W}/pool", f"{W}/tokenizer-pt", f"{W}/packed"
TOKENIZER = f"{LM}/artifacts/tokenizer-v2/v32k"
NB, NR = 20, 5
NP = NB * NR
_r = np.random.default_rng(7)
A = (_r.integers(1, 2**63, NP, dtype=np.uint64) | np.uint64(1))
B = _r.integers(0, 2**63, NP, dtype=np.uint64)
J_MAX = 0.5
RESERVE = {V.WEB: 3000, V.WIKI: 1500}
SRC_KIND = {"fineweb2-edu-ja": V.WEB, "wikipedia-ja": V.WIKI, "fineweb-edu": V.WEB, "wikipedia-en": V.WIKI}


def shingles(text, lang):
    low = V.norm(text or "")
    if lang == "ja":
        ids = np.frombuffer(V.WS.sub("", low).encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    else:
        w = V.WORD.findall(low)
        ids = np.fromiter((xxhash.xxh64_intdigest(x.encode()) for x in w), dtype=np.uint64, count=len(w))
    return np.unique(V.rolling(ids)) if ids.size >= V.SHINGLE else np.zeros(0, dtype=np.uint64)


def bands(sh):
    if sh.size == 0:
        return np.zeros(NB, dtype=np.uint64)
    s = sh[: V.MAX_SHINGLES]
    with np.errstate(over="ignore"):
        sig = ((A[:, None] * s[None, :] + B[:, None]) >> np.uint64(16)).min(axis=1).reshape(NB, NR)
    return np.array([xxhash.xxh64_intdigest(sig[b].tobytes(), seed=100 + b) or 1 for b in range(NB)], dtype=np.uint64)


def sig_file(args):
    path, lang = args
    texts = pq.read_table(path, columns=["text"]).column(0).to_pylist()
    return path, np.stack([bands(shingles(t, lang)) for t in texts]) if texts else np.zeros((0, NB), dtype=np.uint64)


def jacc(a, b, lang):
    x, y = shingles(a, lang), shingles(b, lang)
    if x.size == 0 or y.size == 0:
        return 0.0
    inter = np.intersect1d(x, y, assume_unique=True).size
    return inter / (x.size + y.size - inter)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    report = {}
    for lang in ("ja", "en"):
        ref_files = sorted(glob.glob(f"{PT}/{lang}/train-*.parquet")) + sorted(glob.glob(f"{PT}/{lang}/validation-*.parquet"))
        with Pool(16) as pool:
            sigs = dict(pool.imap_unordered(sig_file, [(f, lang) for f in ref_files]))
        ref_bands = np.concatenate([sigs[f] for f in ref_files])
        ref_loc = [(f, i) for f in ref_files for i in range(len(sigs[f]))]
        del sigs
        sorted_cols = [(np.sort(ref_bands[:, b]), np.argsort(ref_bands[:, b], kind="stable")) for b in range(NB)]
        print(f"[{lang}] indexed {len(ref_loc):,} reference docs", flush=True)

        # candidates: current test, then the unused tail of the pool train split (reserve)
        test_rows = pq.read_table(sorted(glob.glob(f"{PT}/{lang}/test-*.parquet"))).to_pylist()
        used = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{PT}/{lang}/train-*.parquet"))
        reserve, need, seen = [], dict(RESERVE), 0
        for f in sorted(glob.glob(f"{POOL}/{lang}/train-*.parquet")):
            n = pq.ParquetFile(f).metadata.num_rows
            if seen + n <= used:
                seen += n
                continue
            rows = pq.read_table(f).to_pylist()[max(0, used - seen):]
            seen += n
            for r in rows:
                k = SRC_KIND[r["source"]]
                if need[k] > 0:
                    reserve.append(r); need[k] -= 1
            if all(v <= 0 for v in need.values()):
                break
        cands = [("test", r) for r in test_rows] + [("reserve", r) for r in reserve]
        cand_bands = np.stack([bands(shingles(r["text"], lang)) for _, r in cands])

        # candidate x reference pairs sharing a band
        pairs = collections.defaultdict(set)
        for b in range(NB):
            sv, order = sorted_cols[b]
            lo = np.searchsorted(sv, cand_bands[:, b], side="left")
            hi = np.searchsorted(sv, cand_bands[:, b], side="right")
            for ci in np.nonzero((hi > lo) & (cand_bands[:, b] != 0))[0]:
                pairs[int(ci)].update(int(x) for x in order[lo[ci]:hi[ci]])
        need_txt = collections.defaultdict(set)
        for ci, js in pairs.items():
            for j in js:
                need_txt[ref_loc[j][0]].add(ref_loc[j][1])
        ref_text = {}
        for f, ks in need_txt.items():
            col = pq.read_table(f, columns=["text"]).column(0).to_pylist()
            for k in ks:
                ref_text[(f, k)] = col[k]
        bad, maxj = set(), collections.defaultdict(float)
        for ci, js in pairs.items():
            for j in js:
                jv = jacc(cands[ci][1]["text"], ref_text[ref_loc[j]], lang)
                maxj[ci] = max(maxj[ci], jv)
                if jv >= J_MAX:
                    bad.add(ci)
        # candidates against each other (earlier candidate wins)
        for b in range(NB):
            groups = collections.defaultdict(list)
            for ci, h in enumerate(cand_bands[:, b]):
                if h != 0:
                    groups[int(h)].append(ci)
            for g in groups.values():
                for x in range(len(g)):
                    for y in range(x + 1, len(g)):
                        if g[x] not in bad and g[y] not in bad and jacc(cands[g[x]][1]["text"], cands[g[y]][1]["text"], lang) >= J_MAX:
                            bad.add(g[y])

        quota = {V.WEB: sum(SRC_KIND[r["source"]] == V.WEB for r in test_rows), V.WIKI: sum(SRC_KIND[r["source"]] == V.WIKI for r in test_rows)}
        chosen = {V.WEB: [], V.WIKI: []}
        for ci, (kind_, r) in enumerate(cands):
            k = SRC_KIND[r["source"]]
            if ci not in bad and len(chosen[k]) < quota[k]:
                chosen[k].append(r)
        assert all(len(chosen[k]) == quota[k] for k in quota), {k: len(v) for k, v in chosen.items()}
        web, wiki, final, i, j = chosen[V.WEB], chosen[V.WIKI], [], 0, 0
        while i < len(web) or j < len(wiki):
            for _ in range(2):
                if i < len(web):
                    final.append(web[i]); i += 1
            if j < len(wiki):
                final.append(wiki[j]); j += 1
        for p, r in enumerate(final):
            r["id"] = f"{lang}-test-{p:09d}"
        tbl = pa.Table.from_pylist(final, schema=pq.read_schema(sorted(glob.glob(f"{PT}/{lang}/test-*.parquet"))[0]))
        for root in (PT, POOL, TOKPT):
            for old in glob.glob(f"{root}/{lang}/test-*.parquet"):
                os.remove(old)
            pq.write_table(tbl, f"{root}/{lang}/test-00000-of-00001.parquet", compression="zstd", row_group_size=5000, write_page_index=True)
        ids = tok([r["text"] for r in final], add_special_tokens=False)["input_ids"]
        arr = np.concatenate([np.asarray(x + [tok.eos_token_id], dtype=np.uint16) for x in ids])
        arr.tofile(f"{PACKED}/{lang}_test.bin")
        orig_bad = sorted(ci for ci in bad if ci < len(test_rows))
        report[lang] = {"test_docs": len(final), "tokens": int(arr.size),
                        "original_removed": len(orig_bad),
                        "removed_by_source": dict(collections.Counter(test_rows[ci]["source"] for ci in orig_bad)),
                        "reserve_used": len(final) - (len(test_rows) - len(orig_bad)),
                        "reserve_rejected": len([ci for ci in bad if ci >= len(test_rows)]),
                        "max_jaccard_kept": round(max([maxj[ci] for ci in range(len(cands)) if ci not in bad] or [0.0]), 3)}
        print(lang, json.dumps(report[lang]), flush=True)
    for d in (PT, PACKED):
        p = f"{d}/summary.json"
        s = json.load(open(p))
        for lang in report:
            s[lang]["test"] = {"docs": report[lang]["test_docs"], "tokens": report[lang]["tokens"]}
        json.dump(s, open(p, "w"), indent=1)
    json.dump(report, open(f"{W}/test_rebuild_report.json", "w"), indent=1)
    print("TEST_REBUILD_DONE")


if __name__ == "__main__":
    main()
