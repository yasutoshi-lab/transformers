"""freesia-supervised-cl 準備: 文書の単位で split を決め、タスクごとの生成材料と負例探しの対象（プール）を作る。

入力: ~/freesia-work/v2/freesia-pt/{lang}/train-*.parquet（公開版 freesia-pt の train だけ）
      ~/freesia-work/weak-cl/b/（weak-cl で使った文書。hf の全 split・pairs_keep・passages のすべてを除く）
出力: ~/freesia-work/sup-cl/items_{lang}.jsonl.gz（生成材料。id・task・split・pt_id・text・style）
      ~/freesia-work/sup-cl/pool_{lang}.jsonl.gz（負例探しの対象。split・pt_id・passage。1 文書 1 件）
      ~/freesia-work/sup-cl/prepare_stats.json

- 300 字以上の文書を固定 seed でシャッフルし、先頭から validation・test・train の文書群に分ける
- 各文書群の中で、文書を retrieval・nli・classification に重ならないよう割り当てる
- プールはその文書群のすべての文書の先頭の一節（retrieval の正例と同じ切り出し方）。負例は同じ split の中からしか取らない
- 一節の切り出し方は weak-cl と同じ（ja 500 字・en 1500 字で文末に合わせて切る）
"""
import glob, gzip, json, os, random, re
from collections import Counter
import pyarrow.parquet as pq

W = os.path.expanduser("~/freesia-work")
OUT = f"{W}/sup-cl"
SEED = 20261029
CUT = {"ja": 500, "en": 1500}
END = {"ja": "。！？", "en": ".!?"}
GROUP = {"validation": 40_000, "test": 40_000}  # 残りは train
# 文書群ごと・タスクごとの文書数（生成後の除外に備えて目標より多めに取る）
ALLOC = {"validation": {"retrieval": 7_000, "nli": 9_000, "classification": 7_000},
         "test": {"retrieval": 7_000, "nli": 9_000, "classification": 7_000},
         "train": {"retrieval": 500_000, "nli": 300_000, "classification": 130_000}}
STYLES = ("keyword", "factoid", "reasoning")
SENT = {"ja": re.compile(r"[^。！？\n]+[。！？]"), "en": re.compile(r"[^.!?\n]+[.!?]")}
SENT_LEN = {"ja": (20, 150), "en": (50, 350)}


def cut(lang, x):
    """weak-cl と同じ規則で文書の先頭から一節を切り出す。"""
    p = x[:CUT[lang]]
    if len(x) > CUT[lang]:
        k = max(p.rfind(c) for c in END[lang])
        if k >= CUT[lang] * 0.6:
            p = p[:k + 1]
    return p


def weakcl_docs(lang):
    used = set()
    for f in glob.glob(f"{W}/weak-cl/b/hf/{lang}/*.parquet"):
        used.update(pq.read_table(f, columns=["pt_id"]).column(0).to_pylist())
    for fn in (f"{W}/weak-cl/b/pairs_keep.jsonl.gz", f"{W}/weak-cl/b/passages_{lang}.jsonl.gz"):
        for l in gzip.open(fn, "rt"):
            r = json.loads(l)
            if r["lang"] == lang:
                used.add(r["pt_id"])
    return used


def pick_sentence(lang, text, rng):
    lo, hi = SENT_LEN[lang]
    cands = [s.strip() for s in SENT[lang].findall(text)]
    cands = [s for s in cands if lo <= len(s) <= hi and "|" not in s and "http" not in s]
    return rng.choice(cands) if cands else None


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = Counter()
    for lang in ("ja", "en"):
        used = weakcl_docs(lang)
        stats[f"{lang}/weakcl_docs_excluded"] = len(used)
        files = sorted(glob.glob(f"{W}/v2/freesia-pt/{lang}/train-*.parquet"))
        docs = []
        for f in files:
            t = pq.read_table(f, columns=["id", "text", "source"])
            for i, x, s in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), t.column("source").to_pylist()):
                assert i.startswith(f"{lang}-train-"), i
                stats[f"{lang}/train_docs"] += 1
                if i in used:
                    continue
                x = x.strip()
                if len(x) < 300:
                    continue
                docs.append((i, s, cut(lang, x)))
        stats[f"{lang}/eligible_docs"] = len(docs)
        random.seed(SEED)
        random.shuffle(docs)
        groups, k = {}, 0
        for split in ("validation", "test"):
            groups[split] = docs[k:k + GROUP[split]]
            k += GROUP[split]
        groups["train"] = docs[k:]
        need = sum(ALLOC["train"].values())
        assert len(groups["train"]) >= need, (lang, len(groups["train"]), need)
        # タスクの割り当て（nli は本文から文を選ぶので、二度目の読み込みで埋める）
        assign, nli_ids = {}, {}
        for split, g in groups.items():
            k = 0
            for task in ("retrieval", "nli", "classification"):
                for d in g[k:k + ALLOC[split][task]]:
                    assign[d[0]] = (split, task)
                k += ALLOC[split][task]
        with gzip.open(f"{OUT}/pool_{lang}.jsonl.gz", "wt") as out:
            for split, g in groups.items():
                for i, s, p in g:
                    out.write(json.dumps({"split": split, "pt_id": i, "source": s, "passage": p}, ensure_ascii=False) + "\n")
                stats[f"{lang}/{split}/pool"] = len(g)
        rng = random.Random(SEED)
        sents = {}
        for f in files:
            t = pq.read_table(f, columns=["id", "text"])
            for i, x in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
                if assign.get(i, (None, None))[1] == "nli":
                    sents[i] = pick_sentence(lang, x.strip(), rng)
        n = Counter()
        with gzip.open(f"{OUT}/items_{lang}.jsonl.gz", "wt") as out:
            for split, g in groups.items():
                for i, s, p in g:
                    if i not in assign:
                        continue
                    _, task = assign[i]
                    text = sents.get(i) if task == "nli" else p
                    if not text:
                        stats[f"{lang}/{split}/nli_no_sentence"] += 1
                        continue
                    key = f"{split}/{task}"
                    rec = {"id": f"{lang}-{split[:2]}-{task[:3]}-{n[key]:07d}", "lang": lang, "task": task, "split": split,
                           "pt_id": i, "source": s, "text": text}
                    if task == "retrieval":
                        rec["style"] = STYLES[n[key] % 3]
                    n[key] += 1
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for key, v in n.items():
            stats[f"{lang}/{key}"] = v
        print(lang, {k: v for k, v in stats.items() if k.startswith(lang)}, flush=True)
        del docs, groups, assign, sents
    json.dump(dict(sorted(stats.items())), open(f"{OUT}/prepare_stats.json", "w"), indent=1)
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
