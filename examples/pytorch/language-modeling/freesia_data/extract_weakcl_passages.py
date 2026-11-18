"""freesia-pt の train split から段階3 用の一節を抽出する。"""
import pyarrow.parquet as pq, glob, random, json, gzip, re
N = 1_700_000
CUT = {"ja": 500, "en": 1500}
END = {"ja": "。！？", "en": ".!?"}
random.seed(20261024)
for lang in ["ja", "en"]:
    rows = []
    for f in sorted(glob.glob(f"export/freesia-pt/{lang}/train-*.parquet")):
        t = pq.read_table(f, columns=["text", "source"])
        for i, (txt, src) in enumerate(zip(t.column("text").to_pylist(), t.column("source").to_pylist())):
            txt = txt.strip()
            if len(txt) >= 300:
                rows.append((f.split("/")[-1], i, src, txt))
    print(lang, "eligible", len(rows), flush=True)
    random.shuffle(rows)
    with gzip.open(f"weak-cl/passages_{lang}.jsonl.gz", "wt") as out:
        for n, (fn, i, src, txt) in enumerate(rows[:N]):
            p = txt[:CUT[lang]]
            if len(txt) > CUT[lang]:
                k = max(p.rfind(c) for c in END[lang])
                if k >= CUT[lang] * 0.6:
                    p = p[:k + 1]
            rec = {"id": f"{lang}-{n:07d}", "lang": lang, "source": src, "doc": f"{fn}#{i}", "passage": p}
            if lang == "ja":
                rec["style"] = "keyword" if n % 2 == 0 else "question"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(lang, "written", min(N, len(rows)), flush=True)
