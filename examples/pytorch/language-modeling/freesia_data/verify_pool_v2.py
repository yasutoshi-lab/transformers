"""Re-verify the v2 pool: exact duplicates within/among splits, and head/tail-200 overlaps of validation/test with train."""
import glob, hashlib, json, re, sys, unicodedata, collections
import pyarrow.parquet as pq
ROOT = sys.argv[1]; WS = re.compile(r"\s+")
def keys(t):
    n = WS.sub("", unicodedata.normalize("NFKC", t).lower())
    h = lambda s: hashlib.blake2b(s.encode(), digest_size=8).digest()
    return h(n), (h(n[:200]) if len(n) >= 200 else None), (h(n[-200:]) if len(n) >= 200 else None)
res = {}
for lang in ("ja", "en"):
    seen = {s: [set(), set(), set()] for s in ("train", "validation", "test")}
    dup = collections.Counter(); n = collections.Counter()
    for split in ("validation", "test", "train"):
        for f in sorted(glob.glob(f"{ROOT}/{lang}/{split}-*.parquet")):
            for t in pq.read_table(f, columns=["text"]).column(0).to_pylist():
                n[split] += 1
                for j, k in enumerate(keys(t)):
                    if k is None: continue
                    if k in seen[split][j]: dup[f"{split}:within:{j}"] += 1
                    for other in ("validation", "test", "train"):
                        if other != split and k in seen[other][j]: dup[f"{split}~{other}:{j}"] += 1
                    seen[split][j].add(k)
    res[lang] = {"docs": dict(n), "hits(0=exact,1=head200,2=tail200)": dict(dup)}
    print(lang, json.dumps(res[lang]), flush=True)
    del seen
print("VERIFY_DONE")
