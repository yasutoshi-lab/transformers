"""freesia-weak-cl B 案の準備: 旧ペアのうち公開版 freesia-pt の train に元文書があるものを残し、不足分の一節を train から新たに取り出す。

入力: ~/freesia-work/weak-cl/pairs_a.jsonl.gz、export/freesia-pt（旧ペアの取り出し元）、v2/freesia-pt（公開版）
出力: ~/freesia-work/weak-cl/b/pairs_keep.jsonl.gz（残す旧ペア。段階A 通過分、pt_id 付き）
      ~/freesia-work/weak-cl/b/passages_{ja,en}.jsonl.gz（クエリを作る新しい一節。doc は pt_id）

- 新しい一節は、残す旧ペアが使っている文書を除いた train の文書（300 字以上）から取る
- 一節の切り出し方・ja の keyword/question の交互割り当ては extract_weakcl_passages.py と同じ
"""
import glob, gzip, json, os, random, sys
import pyarrow.parquet as pq, xxhash

W = os.path.expanduser("~/freesia-work")
D, OUT = f"{W}/weak-cl", f"{W}/weak-cl/b"
N = {"ja": int(sys.argv[1]), "en": int(sys.argv[2])}
CUT = {"ja": 500, "en": 1500}
END = {"ja": "。！？", "en": ".!?"}
SEED = 20261027


def h(text):
    return xxhash.xxh64_intdigest(text.strip().encode())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = {"ja": [], "en": []}
    for l in gzip.open(f"{D}/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["lang"]].append(r)
    keep_out = gzip.open(f"{OUT}/pairs_keep.jsonl.gz", "wt")
    for lang in ("ja", "en"):
        # 公開版 train の本文ハッシュ → id と、本文・source
        train = {}
        docs = []
        for f in sorted(glob.glob(f"{W}/v2/freesia-pt/{lang}/train-*.parquet")):
            t = pq.read_table(f, columns=["id", "text", "source"])
            for i, x, s in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), t.column("source").to_pylist()):
                train[h(x)] = i
                x = x.strip()
                docs.append((i, s, len(x), x[:CUT[lang]]))  # 一節の切り出しに要る先頭だけ持つ
        # 旧ペアの元文書の本文ハッシュ
        by_file = {}
        for r in rows[lang]:
            fn, i = r["doc"].rsplit("#", 1)
            by_file.setdefault(fn, []).append((int(i), r))
        used, kept = set(), 0
        for fn, items in by_file.items():
            col = pq.read_table(f"{W}/export/freesia-pt/{lang}/{fn}", columns=["text"]).column(0)
            for i, r in items:
                pid = train.get(h(col[i].as_py()))
                if pid is None:
                    continue
                used.add(pid)
                keep_out.write(json.dumps({**r, "pt_id": pid}, ensure_ascii=False) + "\n")
                kept += 1
        del train
        cand = [d for d in docs if d[0] not in used and d[2] >= 300]
        del docs
        print(lang, "kept_old", kept, "train_docs_unused_eligible", len(cand), flush=True)
        random.seed(SEED)
        random.shuffle(cand)
        with gzip.open(f"{OUT}/passages_{lang}.jsonl.gz", "wt") as out:
            for n, (pid, src, ln, p) in enumerate(cand[:N[lang]]):
                if ln > CUT[lang]:
                    k = max(p.rfind(c) for c in END[lang])
                    if k >= CUT[lang] * 0.6:
                        p = p[:k + 1]
                rec = {"id": f"{lang}-b{n:07d}", "lang": lang, "source": src, "doc": pid, "pt_id": pid, "passage": p}
                if lang == "ja":
                    rec["style"] = "keyword" if n % 2 == 0 else "question"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(lang, "new_passages", min(N[lang], len(cand)), flush=True)
    keep_out.close()
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
