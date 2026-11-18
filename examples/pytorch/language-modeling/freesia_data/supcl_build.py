"""freesia-supervised-cl: 生成結果に除外規則をかけ、負例をつけて、ja/en × train/validation/test の parquet に書き出す。

使い方: python supcl_build.py GEN.jsonl MINE.jsonl JUDGE.jsonl OUTDIR [NEVAL]
出力: OUTDIR/{ja,en}/{train,validation,test}-*.parquet、OUTDIR/build_stats.json

除外の規則
- 共通: 空・改行を含む・長すぎる・言語不一致・引用符で始まる・評価データとの重なり（weak-cl と同じ判定。全テキスト列にかける）
- retrieval: 一節の写し（weak-cl と同じ）・同じクエリの重複・自分の一節が同じ split のプールで 11 位以下・
  判定で正例が答えに挙がらない・偽の負例を除いた残りの負例が 7 件未満
- nli: 言い換え・矛盾文のどちらかが元の文と同じ（正規化後）、二つが同じ、元の文との長さの比が 0.5〜2 の外、元の文の重複
- classification: 一覧にないラベル・ラベルごとの上限（train 5,000 件、validation・test 250 件）を超えた分
仕上げの確認（満たさなければ止める）
- すべての pt_id（負例を含む）が {lang}-train- で始まり、weak-cl で使った文書を含まない
- split をまたいで pt_id もテキストも重ならない
"""
import glob, gzip, json, os, sys, random
from collections import Counter, defaultdict
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weakcl_rules as R  # noqa: E402
from supcl_prepare import weakcl_docs  # noqa: E402
from supcl_labels import LABEL_LIST  # noqa: E402

W = os.path.expanduser("~/freesia-work")
D = f"{W}/sup-cl"
GEN, MINE, JUDGE, OUT = sys.argv[1:5]
NEVAL = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
NNEG, TOPK = 7, 10
LABEL_CAP = {"train": 5000, "validation": 250, "test": 250}  # classification のラベルごとの上限
MAXLEN = {"ja": 300, "en": 800}  # nli の文の上限
ROWS_PER_FILE, ROW_GROUP = 200_000, 10_000
SCHEMA = pa.schema([("id", pa.string()), ("task", pa.string()), ("query", pa.string()), ("positive", pa.string()),
                    ("negatives", pa.list_(pa.string())), ("query_style", pa.string()), ("source", pa.string()),
                    ("pt_id", pa.string()), ("neg_pt_ids", pa.list_(pa.string()))])


def text_ok(lang, s, maxlen):
    if not s or "\n" in s:
        return "empty_or_newline"
    if len(s) > maxlen:
        return "too_long"
    if (lang == "ja") != bool(R.JA_CHAR.search(s)):
        return "lang_mismatch"
    if s.startswith(R.QUOTE_START):
        return "quote_start"
    return None


def contaminated(lang, *texts):
    return any(R.key_of(lang, t) in R._EXACT[lang] or R.contam(lang, t) for t in texts)


def main():
    R.init()
    stats = Counter()
    mine = {json.loads(l)["id"]: json.loads(l) for l in open(MINE)} if os.path.exists(MINE) else {}
    judge = {}
    if os.path.exists(JUDGE):
        for l in open(JUDGE):
            j = json.loads(l)
            if j.get("answers") is not None:
                judge[j["id"]] = j
    # 再開で同じ id が何度も出るので、出力が取れた最後の行を使う（取れなかった id は最後の行）
    last = {}
    for l in open(GEN):
        r = json.loads(l)
        if r["id"] not in last or r.get("query") or r.get("label") or r.get("paraphrase"):
            last[r["id"]] = r
    rows = defaultdict(list)
    for r in last.values():
        rows[r["lang"]].append(r)
    os.makedirs(OUT, exist_ok=True)
    for lang in ("ja", "en"):
        used = weakcl_docs(lang)
        pool = {}
        for l in gzip.open(f"{D}/pool_{lang}.jsonl.gz", "rt"):
            p = json.loads(l)
            pool[p["pt_id"]] = (p["split"], p["passage"])
        seen = defaultdict(set)
        recs = defaultdict(list)
        for r in rows[lang]:
            t, split = r["task"], r["split"]
            key = f"{lang}/{split}/{t}"
            stats[f"{key}/generated"] += 1
            why, rec = None, None
            if t == "retrieval":
                q = (r.get("query") or "").strip()
                why = text_ok(lang, q, R.MAXLEN[lang]) or ("verbatim_copy" if R.copied(lang, q, r["text"]) else None)
                if not why and contaminated(lang, q, r["text"]):
                    why = "contaminated"
                k = R.key_of(lang, q)
                if not why and k in seen[(split, t)]:
                    why = "duplicate_query"
                m, j = mine.get(r["id"]), judge.get(r["id"])
                if not why and (m is None or m["own_rank"] > TOPK):
                    why = "own_rank_gt10"
                if not why and len(m["cands"]) < NNEG:
                    why = "few_candidates"
                if not why and j is None:
                    why = "not_judged"
                if not why:
                    picked = {j["order"][a - 1] for a in j["answers"] if 1 <= a <= len(j["order"])}
                    if "POS" not in picked:
                        why = "judge_missed_positive"
                    else:
                        negs = [c[0] for c in m["cands"] if c[0] not in picked][:NNEG]
                        stats[f"{key}/false_negatives_removed"] += len(picked) - 1
                        if len(negs) < NNEG:
                            why = "few_negatives_after_judge"
                        elif contaminated(lang, *(pool[x][1] for x in negs)):
                            why = "contaminated_negative"
                        else:
                            assert pool[r["pt_id"]][1] == r["text"], "正例がプールの一節と一致しない"
                            assert all(pool[x][0] == split for x in negs), "負例が別の split から来ている"
                            rec = {"query": q, "positive": r["text"], "negatives": [pool[x][1] for x in negs],
                                   "neg_pt_ids": negs, "query_style": r["style"]}
                if not why:
                    seen[(split, t)].add(k)
            elif t == "nli":
                a, b = (r.get("paraphrase") or "").strip(), (r.get("contradiction") or "").strip()
                s = r["text"]
                why = text_ok(lang, a, MAXLEN[lang]) or text_ok(lang, b, MAXLEN[lang])
                ka, kb, ks = R.key_of(lang, a), R.key_of(lang, b), R.key_of(lang, s)
                if not why and (ka == ks or kb == ks):
                    why = "same_as_source"
                if not why and ka == kb:
                    why = "paraphrase_eq_contradiction"
                if not why and not all(0.5 <= len(x) / len(s) <= 2.0 for x in (a, b)):
                    why = "length_ratio"
                if not why and contaminated(lang, s, a, b):
                    why = "contaminated"
                if not why and ks in seen[(split, t)]:
                    why = "duplicate_source"
                if not why:
                    seen[(split, t)].add(ks)
                    rec = {"query": s, "positive": a, "negatives": [b], "neg_pt_ids": [], "query_style": None}
            else:
                lab = r.get("label")
                if not lab:
                    why = "invalid_label"
                elif contaminated(lang, r["text"]):
                    why = "contaminated"
                else:
                    rec = {"query": r["text"], "positive": lab, "negatives": [x for x in LABEL_LIST[lang] if x != lab],
                           "neg_pt_ids": [], "query_style": None}
            if why:
                stats[f"{key}/drop/{why}"] += 1
                continue
            recs[split].append({"id": r["id"], "task": t, **rec, "source": r["source"], "pt_id": r["pt_id"]})
        # classification はラベルごとに上限を設ける（生成材料は準備でシャッフル済みなので先頭から残す）
        for split in recs:
            cap, cnt, keep = LABEL_CAP[split], Counter(), []
            for x in recs[split]:
                if x["task"] == "classification":
                    cnt[x["positive"]] += 1
                    if cnt[x["positive"]] > cap:
                        stats[f"{lang}/{split}/classification/drop/label_cap"] += 1
                        continue
                keep.append(x)
            recs[split] = keep
        # validation・test は (task) ごとに先頭 NEVAL 件（生成材料は準備でシャッフル済み）
        for split in ("validation", "test"):
            cnt, keep = Counter(), []
            for x in recs[split]:
                if cnt[x["task"]] < NEVAL:
                    cnt[x["task"]] += 1
                    keep.append(x)
            recs[split] = keep
        # 仕上げの確認
        held_text, held_pt = set(), {}
        for split in ("validation", "test"):
            for x in recs[split]:
                for p in [x["pt_id"]] + x["neg_pt_ids"]:
                    assert held_pt.setdefault(p, split) == split, f"pt_id が validation と test にまたがる: {p}"
                if x["task"] != "classification":
                    held_text.update([x["query"], x["positive"], *x["negatives"]])
                else:
                    held_text.add(x["query"])
        tr = []
        for x in recs["train"]:
            texts = [x["query"], x["positive"], *x["negatives"]] if x["task"] != "classification" else [x["query"]]
            if any(t in held_text for t in texts):
                stats[f"{lang}/train/{x['task']}/drop/text_in_heldout"] += 1
                continue
            tr.append(x)
        recs["train"] = tr
        for split, part in recs.items():
            for x in part:
                for p in [x["pt_id"]] + x["neg_pt_ids"]:
                    assert p.startswith(f"{lang}-train-"), p
                    assert p not in used, f"weak-cl の文書: {p}"
                    assert pool[p][0] == split, f"split をまたぐ文書: {p}"
            random.Random(20261029).shuffle(part)
            os.makedirs(f"{OUT}/{lang}", exist_ok=True)
            nf = max(1, -(-len(part) // ROWS_PER_FILE))
            for k in range(nf):
                t = pa.Table.from_pylist(part[k * ROWS_PER_FILE:(k + 1) * ROWS_PER_FILE], schema=SCHEMA)
                pq.write_table(t, f"{OUT}/{lang}/{split}-{k:05d}-of-{nf:05d}.parquet", row_group_size=ROW_GROUP,
                               write_page_index=True, compression="zstd")
            for task, n in Counter(x["task"] for x in part).items():
                stats[f"{lang}/{split}/{task}/kept"] = n
        print(lang, {k: v for k, v in sorted(stats.items()) if k.startswith(lang)}, flush=True)
        del pool
    json.dump(dict(sorted(stats.items())), open(f"{OUT}/build_stats.json", "w"), indent=1, ensure_ascii=False)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
