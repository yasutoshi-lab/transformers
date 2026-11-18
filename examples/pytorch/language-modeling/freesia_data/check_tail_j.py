import glob, os, sys, collections
sys.path.insert(0, os.path.expanduser("~/freesia-work/scripts"))
sys.argv = [sys.argv[0], sys.argv[1]]
import importlib.util
spec = importlib.util.spec_from_file_location("vp", os.path.expanduser("~/freesia-work/scripts/verify_pt_v2.py"))
src = open(spec.origin).read().split("res = {}")[0]
ns = {}; exec(compile(src, "vp", "exec"), ns)
import pyarrow.parquet as pq
V = ns["V"]; ROOT = sys.argv[1]
for lang in ("ja", "en"):
    held = {}
    for s in ("validation", "test"):
        for t in pq.read_table(glob.glob(f"{ROOT}/{lang}/{s}-*.parquet")[0], columns=["text"]).column(0).to_pylist():
            c = V.WS.sub("", V.norm(t))
            if len(c) >= 200: held.setdefault(c[-200:], []).append(t); held.setdefault("H" + c[:200], []).append(t)
    js = []
    for f in sorted(glob.glob(f"{ROOT}/{lang}/train-*.parquet")):
        for t in pq.read_table(f, columns=["text"]).column(0).to_pylist():
            c = V.WS.sub("", V.norm(t))
            if len(c) < 200: continue
            for k in (c[-200:], "H" + c[:200]):
                for h in held.get(k, []):
                    js.append(ns["jaccard"](h, t, lang))
    js.sort()
    print(lang, "pairs", len(js), "max J", round(js[-1], 3) if js else None, "J>=0.8:", sum(j >= 0.8 for j in js), "J>=0.5:", sum(j >= 0.5 for j in js))
