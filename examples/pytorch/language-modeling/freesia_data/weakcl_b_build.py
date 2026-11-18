"""freesia-weak-cl B 案 段階C: 一致度で絞り、ja/en × train/validation/test の parquet に書き出す。

入力: ~/freesia-work/weak-cl/b/pairs_a.jsonl.gz、b/emb/{lang}_{ids,rank,own}
出力: ~/freesia-work/weak-cl/b/hf/{ja,en}/{train,validation,test}-*.parquet、b/build_stats.json

- 順位が 10 位以内（TOPK）のペアだけを残す
- 元の文書（pt_id）が同じペアは先に出た 1 件だけ残す（旧 export の重複文書が公開版で 1 件にまとまったもの。split をまたぐのを防ぐ）
- どのペアも元の文書は公開版 freesia-pt の train（pt_id）。念のため id が train のものかを確かめる
- 言語ごとに固定 seed でシャッフルし、validation・test を各 1 万件、残りを train にする
- 別の文書でも一節の文字列が validation・test と同じものは train から除く
"""
import gzip, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weakcl_build as WB  # noqa: E402

D = os.path.expanduser("~/freesia-work/weak-cl/b")
E = f"{D}/emb"
WB.OUT = f"{D}/hf"


def main():
    stats = Counter()
    rows = {"ja": [], "en": []}
    for l in gzip.open(f"{D}/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["lang"]].append(r)
    for lang, rs in rows.items():
        ids = [x.strip() for x in open(f"{E}/{lang}_ids.txt")]
        assert ids == [r["id"] for r in rs], "埋め込みと pairs_a の並びが一致しない"
        assert all(r["pt_id"].startswith(f"{lang}-train-") for r in rs), "train 以外の文書が混じっている"
        rank, own = np.load(f"{E}/{lang}_rank.npy"), np.load(f"{E}/{lang}_own.npy")
        stats[f"{lang}/after_stage_a"] = len(rs)
        recs, seen = [], set()
        for r, k, s in zip(rs, rank, own):
            if k > WB.TOPK:
                continue
            if r["pt_id"] in seen:
                stats[f"{lang}/same_pt_id"] += 1
                continue
            seen.add(r["pt_id"])
            stats[f"{lang}/after_consistency/{'new' if '-b' in r['id'] else 'old'}"] += 1
            recs.append({"id": r["id"], "query": r["query"].strip(), "passage": r["passage"], "query_style": r.get("style"),
                         "source": r["source"], "pt_id": r["pt_id"], "score": round(float(s), 4), "rank": int(k)})
        stats[f"{lang}/after_consistency"] = len(recs)
        rng = np.random.default_rng(WB.SEED)
        perm = rng.permutation(len(recs))
        recs = [recs[i] for i in perm]
        parts = {"validation": recs[:WB.NEVAL], "test": recs[WB.NEVAL:2 * WB.NEVAL], "train": recs[2 * WB.NEVAL:]}
        held = {x["passage"] for x in parts["validation"] + parts["test"]}
        parts["train"] = [x for x in parts["train"] if x["passage"] not in held]
        stats[f"{lang}/train_dropped_same_passage_as_eval"] = len(recs) - 2 * WB.NEVAL - len(parts["train"])
        for split, part in parts.items():
            WB.write(lang, split, part)
            stats[f"{lang}/{split}"] = len(part)
            for st, n in Counter(x["query_style"] for x in part).items():
                stats[f"{lang}/{split}/{st}"] = n
        print(lang, {k: v for k, v in stats.items() if k.startswith(lang)}, flush=True)
    json.dump(dict(sorted(stats.items())), open(f"{D}/build_stats.json", "w"), indent=1)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
