import glob, hashlib, re, sys, unicodedata, collections, random
import pyarrow.parquet as pq
ROOT, lang = sys.argv[1], sys.argv[2]; WS = re.compile(r"\s+")
nz = lambda t: WS.sub("", unicodedata.normalize("NFKC", t).lower())
val = pq.read_table(glob.glob(f"{ROOT}/{lang}/validation-*.parquet")[0]).to_pylist()
vh = {}
for r in val:
    n = nz(r["text"])
    if len(n) >= 200: vh.setdefault(n[:200], []).append(r)
hits = []
for f in sorted(glob.glob(f"{ROOT}/{lang}/train-*.parquet")):
    t = pq.read_table(f, columns=["text", "source", "url"]).to_pylist()
    for r in t:
        n = nz(r["text"])
        if len(n) >= 200 and n[:200] in vh: hits.append((vh[n[:200]][0], r))
pairs = collections.Counter((a["source"], b["source"]) for a, b in hits)
print("pairs", len(hits), dict(pairs))
# similarity: containment of char 5-grams
def sh(t): n = nz(t); return {n[i:i+5] for i in range(0, max(len(n)-4, 1))}
random.seed(0); samp = random.sample(hits, min(12, len(hits)))
for a, b in samp:
    A, B = sh(a["text"]), sh(b["text"]); j = len(A & B) / len(A | B); c = len(A & B) / min(len(A), len(B))
    print(f"J={j:.2f} C={c:.2f} lenV={len(a['text'])} lenT={len(b['text'])} {a['source']}|{b['source']} {a['url'][:60]} || {b['url'][:60]}")
    print("   V:", a["text"][:100].replace("\n", " "))
