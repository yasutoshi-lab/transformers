"""Verify freesia-pt: duplicates within each split and overlap of validation/test with train.

Per language, three checks on the exported documents:
  exact    : identical text after NFKC + lowercase + whitespace removal
  minhash  : near-duplicates (same 14x8 MinHash bands as the pool build; any shared band = candidate)
  head/tail: first / last 200 normalized characters identical (catches shared boilerplate; reported for reference)
For minhash, candidate pairs are confirmed with the actual shingle Jaccard (>= 0.8 counted as near-duplicate).
"""
import glob, json, os, sys, collections
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
import xxhash

sys.path.insert(0, os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling"))
import build_freesia_corpus_v2 as V  # noqa: E402

ROOT = sys.argv[1]
LANG = None


def shingles(text, lang):
    low = V.norm(text)
    if lang == "ja":
        c = V.WS.sub("", low)
        ids = np.frombuffer(c.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    else:
        w = V.WORD.findall(low)
        ids = np.fromiter((xxhash.xxh64_intdigest(x.encode()) for x in w), dtype=np.uint64, count=len(w))
    if ids.size < V.SHINGLE:
        return None
    return V.rolling(ids)


def keys(args):
    path, lang = args
    out = []
    for t in pq.read_table(path, columns=["text"]).column(0).to_pylist():
        c = V.WS.sub("", V.norm(t))
        ex = xxhash.xxh64_intdigest(c.encode())
        hd = xxhash.xxh64_intdigest(c[:200].encode()) if len(c) >= 200 else 0
        tl = xxhash.xxh64_intdigest(c[-200:].encode()) if len(c) >= 200 else 0
        sh = shingles(t, lang)
        b = V.bands_of(sh) if sh is not None else np.zeros(V.NUM_BANDS, dtype=np.uint64)
        out.append((ex, hd, tl, b))
    return path, out


def jaccard(a, b, lang):
    x, y = shingles(a, lang), shingles(b, lang)
    if x is None or y is None:
        return 0.0
    x, y = set(x.tolist()), set(y.tolist())
    return len(x & y) / max(len(x | y), 1)


res = {}
for lang in ("ja", "en"):
    files = {s: sorted(glob.glob(f"{ROOT}/{lang}/{s}-*.parquet")) for s in ("train", "validation", "test")}
    rows = {}
    with Pool(16) as pool:
        for path, out in pool.imap_unordered(keys, [(f, lang) for fs in files.values() for f in fs]):
            rows[path] = out
    split_of, ex, hd, tl, bands, ref = [], [], [], [], [], []
    for s, fs in files.items():
        for f in fs:
            for i, (a, b, c, d) in enumerate(rows[f]):
                split_of.append(s); ex.append(a); hd.append(b); tl.append(c); bands.append(d); ref.append((f, i))
    split_of = np.array(split_of); ex = np.array(ex, dtype=np.uint64); bands = np.stack(bands)
    r = {"docs": dict(collections.Counter(split_of.tolist()))}

    def dup_stats(key, name):
        c = collections.Counter()
        first = {}
        for i, k in enumerate(key):
            if k == 0:
                continue
            if k in first:
                s0, s1 = split_of[first[k]], split_of[i]
                c[f"{s1}~{s0}" if s0 != s1 else f"{s1}:within"] += 1
            else:
                first[k] = i
        r[name] = dict(c)

    # order rows so validation/test come first: overlaps are then counted on the train side
    dup_stats(ex.tolist(), "exact")
    dup_stats(hd, "head200")
    dup_stats(tl, "tail200")

    # minhash candidates: any shared band; confirm with Jaccard >= 0.8
    cand = set()
    for b in range(V.NUM_BANDS):
        col = bands[:, b]
        order = np.argsort(col, kind="stable")
        sc = col[order]
        same = np.nonzero((sc[1:] == sc[:-1]) & (sc[1:] != 0))[0]
        for j in same:
            x, y = int(order[j]), int(order[j + 1])
            cand.add((min(x, y), max(x, y)))
    need = collections.defaultdict(set)
    for x, y in cand:
        for i in (x, y):
            need[ref[i][0]].add(ref[i][1])
    texts = {}
    for f, ks in need.items():
        col = pq.read_table(f, columns=["text"]).column(0).to_pylist()
        for k in ks:
            texts[(f, k)] = col[k]

    def text_of(i):
        return texts[ref[i]]

    mh = collections.Counter()
    for x, y in sorted(cand):
        s0, s1 = split_of[x], split_of[y]
        key = f"{s1}~{s0}" if s0 != s1 else f"{s1}:within"
        tx, ty = text_of(x), text_of(y)
        j = jaccard(tx, ty, lang)
        mh[key + (":J>=0.8" if j >= 0.8 else ":J<0.8")] += 1
    r["minhash_candidates"] = dict(mh)
    res[lang] = r
    print(lang, json.dumps(r), flush=True)
json.dump(res, open(f"{ROOT}/verify.json", "w"), indent=1)
print("VERIFY_PT_DONE")
