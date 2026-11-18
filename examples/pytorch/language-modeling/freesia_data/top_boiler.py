import sys, os, glob, collections, random
sys.path.insert(0, os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling"))
import build_freesia_corpus_v2 as V, pyarrow.parquet as pq
for lang in ("ja", "en"):
    g = V.SOURCES[lang][0][1]; f = sorted(glob.glob(os.path.join(V.HF_HUB, g)))[0]
    pf = pq.ParquetFile(f); df = collections.Counter(); ex = {}
    for rg in range(min(40, pf.num_row_groups)):
        for t in pf.read_row_group(rg, columns=["text"]).column(0).to_pylist():
            seen = set()
            for ln in (t or "").split("\n"):
                c = V.WS.sub("", V.norm(ln))
                if len(c) >= 10 and c not in seen:
                    seen.add(c); df[c] += 1; ex.setdefault(c, ln.strip())
    common = [(k, n) for k, n in df.items() if n >= 5]
    print(f"== {lang}: boiler lines {len(common)}")
    for k, n in sorted(common, key=lambda x: -x[1])[:12]: print(n, ex[k][:90])
    random.seed(1)
    print("-- random sample")
    for k, n in random.sample(common, min(10, len(common))): print(n, ex[k][:90])
