"""一節の元の文書（旧 export/freesia-pt）が、v2 のどこにあるかを数える（調査用）。

分類: v2 公開版 train / validation / test（本文の完全一致）、v2 プールの train 部分のうち公開版 train の外（完全一致）、
      本文は違うが先頭 64 字（正規化）が v2 の文書と一致、どこにもない。
"""
import glob, gzip, json, os, re, unicodedata
from collections import Counter
import pyarrow.parquet as pq, xxhash

W = os.path.expanduser("~/freesia-work")
WS = re.compile(r"\s+")


def h(t):
    return xxhash.xxh64_intdigest(t.strip().encode())


def pk(t):
    return xxhash.xxh64_intdigest(WS.sub("", unicodedata.normalize("NFKC", t).lower())[:64].encode())


res = {}
for lang in ("ja", "en"):
    need = {}
    for l in gzip.open(f"{W}/weak-cl/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["lang"] == lang:
            fn, i = r["doc"].rsplit("#", 1)
            need.setdefault(fn, []).append(int(i))
    old = []
    for fn, rows in need.items():
        col = pq.read_table(f"{W}/export/freesia-pt/{lang}/{fn}", columns=["text", "source"])
        for i in rows:
            t = col.column(0)[i].as_py()
            old.append((h(t), pk(t), col.column(1)[i].as_py()))
    full, pref = {}, {}
    for tag, pat in (("pub", f"{W}/v2/freesia-pt/{lang}/*.parquet"), ("pool", f"{W}/v2/pool/{lang}/*.parquet")):
        for f in sorted(glob.glob(pat)):
            split = os.path.basename(f).split("-")[0]
            for t in pq.read_table(f, columns=["text"]).column(0).to_pylist():
                full.setdefault(h(t), f"{tag}_{split}")
                pref.setdefault(pk(t), f"{tag}_{split}")
    c = Counter()
    for hx, px, src in old:
        k = full.get(hx)
        if k is None:
            k = "prefix_" + pref[px] if px in pref else "nowhere"
        c[f"{k}"] += 1
        c[f"{src}:{k}"] += 1
    res[lang] = dict(sorted(c.items()))
    print(lang, res[lang], flush=True)
json.dump(res, open(f"{W}/weak-cl/trace_stats.json", "w"), indent=1)
print("TRACE_DONE")
