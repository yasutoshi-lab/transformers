"""freesia-weak-cl 後処理 段階C: 一致度で絞り、freesia-pt の id を付け、ja/en × train/validation/test の parquet に書き出す。

入力: ~/freesia-work/weak-cl/pairs_a.jsonl.gz、emb/{lang}_{ids,rank,own}、export/freesia-pt（一節の取り出し元）、
      v2/freesia-pt（公開版。元の文書の id と split を本文の一致で引く）
出力: ~/freesia-work/weak-cl/hf/{ja,en}/{train,validation,test}-*.parquet、build_stats.json

- 順位が 10 位以内（TOPK）のペアだけを残す
- 元の文書が公開版 freesia-pt の train にないペアは除く（validation・test の文書が混じらないようにする）
- 言語ごとに固定 seed でシャッフルし、validation・test を各 1 万件、残りを train にする
"""
import glob, gzip, json, os
from collections import Counter
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, xxhash

W = os.path.expanduser("~/freesia-work")
D, E, OUT = f"{W}/weak-cl", f"{W}/weak-cl/emb", f"{W}/weak-cl/hf"
TOPK, NEVAL, SEED = 10, 10_000, 20261026
ROWS_PER_FILE, ROW_GROUP = 200_000, 10_000
SCHEMA = pa.schema([("id", pa.string()), ("query", pa.string()), ("passage", pa.string()), ("query_style", pa.string()),
                    ("source", pa.string()), ("pt_id", pa.string()), ("score", pa.float32()), ("rank", pa.int32())])


def h(text):
    return xxhash.xxh64_intdigest(text.strip().encode())


def pt_index(lang):
    """公開版 freesia-pt の本文ハッシュ → (split, id)。"""
    idx = {}
    for f in sorted(glob.glob(f"{W}/v2/freesia-pt/{lang}/*.parquet")):
        split = os.path.basename(f).split("-")[0]
        t = pq.read_table(f, columns=["id", "text"])
        for i, x in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            idx[h(x)] = (split, i)
    return idx


def export_texts(lang, needed):
    """export/freesia-pt の (ファイル名, 行) → 本文ハッシュ。"""
    by_file = {}
    for fn, i in needed:
        by_file.setdefault(fn, set()).add(i)
    out = {}
    for fn, rows in by_file.items():
        col = pq.read_table(f"{W}/export/freesia-pt/{lang}/{fn}", columns=["text"]).column(0)
        for i in rows:
            out[(fn, i)] = h(col[i].as_py())
    return out


def write(lang, split, recs):
    os.makedirs(f"{OUT}/{lang}", exist_ok=True)
    nf = max(1, -(-len(recs) // ROWS_PER_FILE))
    for k in range(nf):
        part = recs[k * ROWS_PER_FILE:(k + 1) * ROWS_PER_FILE]
        t = pa.Table.from_pylist(part, schema=SCHEMA)
        pq.write_table(t, f"{OUT}/{lang}/{split}-{k:05d}-of-{nf:05d}.parquet", row_group_size=ROW_GROUP,
                       write_page_index=True, compression="zstd")


def main():
    stats = Counter()
    rows = {"ja": [], "en": []}
    for l in gzip.open(f"{D}/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["lang"]].append(r)
    for lang, rs in rows.items():
        ids = [x.strip() for x in open(f"{E}/{lang}_ids.txt")]
        assert ids == [r["id"] for r in rs], "埋め込みと pairs_a の並びが一致しない"
        rank, own = np.load(f"{E}/{lang}_rank.npy"), np.load(f"{E}/{lang}_own.npy")
        keep = [(r, int(k), float(s)) for r, k, s in zip(rs, rank, own) if k <= TOPK]
        stats[f"{lang}/after_stage_a"] = len(rs)
        stats[f"{lang}/after_consistency"] = len(keep)
        docs = [tuple(r["doc"].rsplit("#", 1)) for r, _, _ in keep]
        docs = [(fn, int(i)) for fn, i in docs]
        hx = export_texts(lang, set(docs))
        idx = pt_index(lang)
        recs = []
        for (r, k, s), d in zip(keep, docs):
            split, pid = idx.get(hx[d], ("missing", None))
            stats[f"{lang}/pt_{split}"] += 1
            if split != "train":
                continue
            recs.append({"id": r["id"], "query": r["query"].strip(), "passage": r["passage"], "query_style": r.get("style"),
                         "source": r["source"], "pt_id": pid, "score": round(s, 4), "rank": k})
        del idx
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(recs))
        recs = [recs[i] for i in perm]
        parts = {"validation": recs[:NEVAL], "test": recs[NEVAL:2 * NEVAL], "train": recs[2 * NEVAL:]}
        for split, part in parts.items():
            write(lang, split, part)
            stats[f"{lang}/{split}"] = len(part)
            for st, n in Counter(x["query_style"] for x in part).items():
                stats[f"{lang}/{split}/{st}"] = n
        print(lang, {k: v for k, v in stats.items() if k.startswith(lang)}, flush=True)
    json.dump(dict(sorted(stats.items())), open(f"{D}/build_stats.json", "w"), indent=1)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
