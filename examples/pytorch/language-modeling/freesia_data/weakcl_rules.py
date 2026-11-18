"""freesia-weak-cl 後処理 段階A: 機械的な除外・重複除去・汚染除去。

入力: ~/freesia-work/weak-cl/pairs_raw.jsonl.gz（生成結果 340 万件）
出力: ~/freesia-work/weak-cl/pairs_a.jsonl.gz（残ったペア）、stats_a.json（除外理由ごとの件数）
使い方: python3 weakcl_rules.py [入力 出力 stats]（引数なしなら上のパス。B 案では b/ 配下を渡す）

除外の基準
- 空・改行を含む・長すぎる（ja 80 字超・en 250 字超）
- 言語不一致（ja クエリに仮名・漢字がない / en クエリに仮名・漢字がある）
- 引用符（" “ ' ‘）で始まる
- 一節の文字列をそのまま写したもの（正規化後、ja は空白を除いて 20 字、en は 8 語の一致）
- 同じ言語で正規化後に同じクエリ（先に出たものを残す）
- 汚染: 評価クエリ・STS 文との完全一致、または freesia-pt と同じ n-gram（ja 25 字・en 13 語）の共有
"""
import gzip, json, os, re, sys, unicodedata
from collections import Counter
from multiprocessing import Pool

LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, LM)
import build_freesia_corpus_v2 as C  # noqa: E402
import bench_embed_models as B  # noqa: E402

D = os.path.expanduser("~/freesia-work/weak-cl")
WS, WORD = C.WS, C.WORD
JA_CHAR = re.compile(r"[぀-ヿ一-鿿]")
QUOTE_START = tuple("\"“”'‘’")  # 『作品名』のような書名の括弧は自然なクエリなので残す
MAXLEN = {"ja": 80, "en": 250}
COPY = {"ja": 20, "en": 8}


def norm(s):
    return unicodedata.normalize("NFKC", s).lower()


def eval_items(lang):
    items = []
    if lang == "ja":
        for t in B.JA_RETRIEVAL:
            items += B.load_ja_retrieval(t)[0]
        for t in B.JA_STS:
            a, b, _ = B.load_ja_sts(t)
            items += a + b
    else:
        for t in B.EN_RETRIEVAL:
            items += B.load_en_retrieval(t)[0]
        for t in B.EN_STS:
            a, b, _ = B.load_en_sts(t)
            items += a + b
    return items


def key_of(lang, s):
    low = norm(s)
    return WS.sub("", low) if lang == "ja" else " ".join(WORD.findall(low))


_AUTO = {}
_EXACT = {}


def init():
    import ahocorasick
    for lang in ("ja", "en"):
        a = ahocorasick.Automaton()
        for i, p in enumerate(C.eval_patterns(lang)):
            a.add_word(p, i)
        a.make_automaton()
        _AUTO[lang] = a
        _EXACT[lang] = {key_of(lang, x) for x in eval_items(lang)}


def contam(lang, text):
    low = norm(text)
    s = WS.sub("", low) if lang == "ja" else " " + " ".join(WORD.findall(low)) + " "
    for _ in _AUTO[lang].iter(s):
        return True
    return False


def copied(lang, q, p):
    k = COPY[lang]
    if lang == "ja":
        a, b = WS.sub("", norm(q)), WS.sub("", norm(p))
        return len(a) >= k and any(a[i:i + k] in b for i in range(len(a) - k + 1))
    a, b = WORD.findall(norm(q)), WORD.findall(norm(p))
    if len(a) < k:
        return False
    grams = {tuple(b[i:i + k]) for i in range(len(b) - k + 1)}
    return any(tuple(a[i:i + k]) in grams for i in range(len(a) - k + 1))


def judge(line):
    r = json.loads(line)
    lang, q, p = r["lang"], (r.get("query") or "").strip(), r["passage"]
    if not q:
        return "empty", None, line
    if "\n" in q:
        return "newline", None, line
    if len(q) > MAXLEN[lang]:
        return "too_long", None, line
    has_ja = bool(JA_CHAR.search(q))
    if (lang == "ja") != has_ja:
        return "lang_mismatch", None, line
    if q.startswith(QUOTE_START):
        return "quote_start", None, line
    if copied(lang, q, p):
        return "verbatim_copy", None, line
    k = key_of(lang, q)
    if k in _EXACT[lang] or contam(lang, q) or contam(lang, p):
        return "contaminated", None, line
    return None, k, line


def main():
    stats = Counter()
    seen = {"ja": set(), "en": set()}
    src, dst, st = sys.argv[1:4] if len(sys.argv) == 4 else (f"{D}/pairs_raw.jsonl.gz", f"{D}/pairs_a.jsonl.gz", f"{D}/stats_a.json")
    init()  # fork 前に作り、ワーカーと共有する
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as g, \
            Pool(28) as pool:
        for n, (why, k, line) in enumerate(pool.imap(judge, f, chunksize=2000), 1):
            lang = '"lang": "ja"' in line and "ja" or "en"
            if why is None and k in seen[lang]:
                why = "duplicate_query"
            if why:
                stats[f"{lang}/{why}"] += 1
                continue
            seen[lang].add(k)
            stats[f"{lang}/kept"] += 1
            g.write(line)
            if n % 500000 == 0:
                print(n, dict(stats), flush=True)
    json.dump(dict(sorted(stats.items())), open(st, "w"), indent=1)
    print("STAGE_A_DONE", dict(sorted(stats.items())), flush=True)


if __name__ == "__main__":
    main()
