"""freesia-supervised-cl: 生成材料ごとに Gemma-4 で retrieval のクエリ・nli の言い換えと矛盾文・classification のラベルを作る（再開可能）。

使い方: python3 -I supcl_gen.py OUT.jsonl LIMIT CONC PORTS SPLITS
  LIMIT は (言語, split, task) ごとの上限（0 なら全件）。PORTS は "8000,8001"、SPLITS は "train" や "validation,test,train"。
  OUT に既にある id（出力が取れたもの）は飛ばす。8000 は ws2 の vLLM、8001 は ws1 の vLLM へのトンネル。
"""
import gzip, json, os, re, sys, time, threading, urllib.request, itertools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

OUT, LIMIT, CONC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PORTS = [int(p) for p in sys.argv[4].split(",")]
SPLITS = sys.argv[5].split(",")
MODEL = "gemma4"
HERE = os.path.expanduser("~/freesia-work/sup-cl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supcl_labels import LABEL_LIST  # noqa: E402

Q_JA = "条件: 日本語、文章を読んでいない人が検索窓に入力する想定、「この」「その」などで文章を指さない、文章中の語句をそのまま長く写さない、出力は{what}のみ。\n\n文章:\n{p}"
Q_EN = "Rules: English, written by someone who has not read the passage, do not refer to the passage with words like \"this\" or \"the passage\", do not copy long spans verbatim, no quotation marks, output only the {what}.\n\nPassage:\n{p}"
PROMPT = {
    ("ja", "keyword"): "次の文章を検索で見つけたい人が入力しそうな検索クエリを1つだけ作ってください。2〜5語のキーワードを空白で区切った形にし、「概要」「とは」のような汎用語だけで終わらせないでください。\n" + Q_JA.replace("{what}", "クエリ"),
    ("ja", "factoid"): "次の文章に書かれている具体的な事実（人物・数値・日付・場所・名称など）を問う質問を1つだけ作ってください。自然な疑問文にしてください。\n" + Q_JA.replace("{what}", "質問文"),
    ("ja", "reasoning"): "次の文章を読むと答えられる、理由・仕組み・方法・経緯のいずれかを問う質問を1つだけ作ってください。自然な疑問文にしてください。\n" + Q_JA.replace("{what}", "質問文"),
    ("en", "keyword"): "Write exactly one keyword search query (2 to 6 words, no question form) that a user would type to find the following passage.\n" + Q_EN.replace("{what}", "query"),
    ("en", "factoid"): "Write exactly one question about a specific fact in the following passage (a person, number, date, place or name). Make it a natural question.\n" + Q_EN.replace("{what}", "question"),
    ("en", "reasoning"): "Write exactly one question that the following passage answers about a reason, mechanism, method or course of events. Make it a natural question.\n" + Q_EN.replace("{what}", "question"),
    ("ja", "nli"): "次の文をもとに、2つの文を作ってください。\n"
                   "1. paraphrase: 元の文と同じ意味を、語順や言葉を変えて表した文。情報を足したり削ったりしない。\n"
                   "2. contradiction: 元の文とよく似た言い回しのまま、事実を1か所だけ変えて、元の文と矛盾するようにした文。変えるのは数値・日付・人名・地名・組織名・作品名・人や物どうしの関係（誰が何をしたか、何が何に属するか）のいずれかにする。反対の意味の語への置き換え（大→小、増→減、多い→少ないなど）や、否定形にするだけ（〜する→〜しない）の変更はしない。\n"
                   "どちらも日本語の1文にしてください。出力は {{\"paraphrase\": \"...\", \"contradiction\": \"...\"}} の JSON のみ。\n\n文:\n{p}",
    ("en", "nli"): "Write two sentences based on the sentence below.\n"
                   "1. paraphrase: the same meaning in different words and word order, adding or removing no information.\n"
                   "2. contradiction: wording close to the original, but change exactly one fact so that it contradicts the original. Change a number, date, person, place, organization, title, or a relation between people or things (who did what, what belongs to what). Do not just swap a word for its antonym (large to small, increase to decrease, alive to dead) or simply negate the sentence.\n"
                   "Each must be a single English sentence. Output only the JSON {{\"paraphrase\": \"...\", \"contradiction\": \"...\"}}.\n\nSentence:\n{p}",
    ("ja", "classification"): "次の文章の主な話題として最も当てはまるものを、下の一覧から1つだけ選んでください。出力は一覧の表記そのままのラベルのみ。\n一覧: {labels}\n\n文章:\n{p}",
    ("en", "classification"): "Choose the one label from the list below that best fits the main topic of the following passage. Output only the label, exactly as written in the list.\nList: {labels}\n\nPassage:\n{p}",
}


def records():
    fs = [gzip.open(os.path.join(HERE, f"items_{lang}.jsonl.gz"), "rt") for lang in ("ja", "en")]
    seen = Counter()
    for a, b in itertools.zip_longest(*fs):
        for l in (a, b):
            if not l:
                continue
            r = json.loads(l)
            key = (r["lang"], r["split"], r["task"])
            if r["split"] not in SPLITS or (LIMIT and seen[key] >= LIMIT):
                continue
            seen[key] += 1
            yield r


def parse(rec, text):
    t = rec["task"]
    if t == "retrieval":
        return {"query": text.strip()}
    if t == "classification":
        lab = text.strip().strip("「」\"'.。 ")
        return {"label": lab} if lab in LABEL_LIST[rec["lang"]] else {"label": None, "raw": text}
    m = re.search(r"\{[^{}]*\}", text)  # 前後の説明やコードフェンスは捨て、最初の JSON だけを読む
    try:
        d = json.loads(m.group(0))
        return {"paraphrase": str(d["paraphrase"]).strip(), "contradiction": str(d["contradiction"]).strip()}
    except Exception:
        return {"paraphrase": None, "raw": text}


def call(port, rec):
    lang, t = rec["lang"], rec["task"]
    key = (lang, rec["style"]) if t == "retrieval" else (lang, t)
    prompt = PROMPT[key].format(p=rec["text"], labels="、".join(LABEL_LIST["ja"]) if lang == "ja" else ", ".join(LABEL_LIST["en"]))
    body = json.dumps({"model": MODEL, "temperature": 0.7 if t != "classification" else 0.0,
                       "max_tokens": {"retrieval": 64, "nli": 512, "classification": 32}[t],
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            return {**rec, **parse(rec, d["choices"][0]["message"]["content"]), "port": port, "usage": d["usage"]}
        except Exception as e:
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return {**rec, "port": port, "error": err}


def ok(r):
    return bool(r.get("query") or r.get("label") or r.get("paraphrase"))


done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if ok(r):
            done.add(r["id"])
todo = (r for r in records() if r["id"] not in done)
out = open(OUT, "a")
pools = {p: ThreadPoolExecutor(CONC) for p in PORTS}
t0 = time.time()
n = nok = 0
pending = set()
port_cycle = itertools.cycle(PORTS)


def drain(fin):
    global n, nok
    for f in fin:
        r = f.result()
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1; nok += ok(r)
        if n % 10000 == 0:
            out.flush()
            print(f"{time.strftime('%m-%d %H:%M')} done={n} ok={nok} rate={n / (time.time() - t0):.1f}/s", flush=True)


for rec in todo:
    while len(pending) >= CONC * len(PORTS) * 2:
        fin, pending = wait(pending, return_when=FIRST_COMPLETED)
        drain(fin)
    p = next(port_cycle)
    pending.add(pools[p].submit(call, p, rec))
drain(pending)
out.close()
el = time.time() - t0
print(json.dumps({"new": n, "ok": nok, "skipped": len(done), "elapsed_s": round(el, 1),
                  "per_s": round(n / el, 2) if el else 0}), flush=True)
print("GEN_DONE", flush=True)
