import sys, os, collections
LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, LM)
import build_freesia_corpus_v2 as V
import ahocorasick, glob, pyarrow.parquet as pq
pats = V.eval_patterns("ja")
A = ahocorasick.Automaton()
for i, p in enumerate(pats): A.add_word(p, p)
A.make_automaton()
c = collections.Counter(); docs = 0; n = 0
for g, _ in [(V.SOURCES["ja"][0][1], 0), (V.SOURCES["ja"][1][1], 1)]:
    f = sorted(glob.glob(os.path.join(V.HF_HUB, g)))[0]
    for t in pq.ParquetFile(f).read_row_group(0, columns=["text"]).column(0).to_pylist()[:3000]:
        n += 1
        s = V.WS.sub("", V.norm(t))
        hits = {p for _, p in A.iter(s)}
        if hits: docs += 1; c.update(hits)
print("docs", docs, "/", n)
for p, k in c.most_common(25): print(k, p)
