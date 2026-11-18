"""freesia-weak-cl B 案: 一節ごとに検索クエリを 1 件生成する（再開可能）。ws2 で動かし、ws2 と ws1 の vLLM を使う。

使い方: python3 -I gen_queries.py OUT.jsonl LIMIT CONC
  b/passages_ja / b/passages_en を交互に読み（短いほうが尽きたら残りを続けて読む）、OUT に既にある id は飛ばす。
  8000 は ws2 の vLLM、8001 は ws1 の vLLM へのトンネル。
"""
import gzip, json, os, sys, time, threading, urllib.request, itertools
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

OUT, LIMIT, CONC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PORTS = [8000, 8001]
MODEL = "gemma4"
HERE = os.path.expanduser("~/freesia-work/weak-cl/b")

PROMPT = {
    "keyword": "次の文章を検索で見つけたい人が入力しそうな検索クエリを1つだけ作ってください。\n"
               "条件: 日本語、2〜5語のキーワードを空白で区切った形、「概要」「とは」のような汎用語だけで終わらせない、"
               "文章中の語句をそのまま長く写さない、出力はクエリのみ。\n\n文章:\n{p}",
    "question": "次の文章が答えになるような質問を1つだけ作ってください。\n"
                "条件: 日本語の自然な疑問文、文章を読んでいない人が検索窓に入力する想定、"
                "文章中の語句をそのまま長く写さない、出力は質問文のみ。\n\n文章:\n{p}",
    "en": "Write exactly one search query that a user would type to find the following passage.\n"
          "Rules: English, a natural question or keyword phrase, do not copy long spans verbatim, "
          "no quotation marks, output only the query.\n\nPassage:\n{p}",
}


def records():
    fa = gzip.open(os.path.join(HERE, "passages_ja.jsonl.gz"), "rt")
    fb = gzip.open(os.path.join(HERE, "passages_en.jsonl.gz"), "rt")
    for a, b in itertools.zip_longest(fa, fb):
        if a:
            yield json.loads(a)
        if b:
            yield json.loads(b)


def call(port, rec):
    kind = rec.get("style") if rec["lang"] == "ja" else "en"
    body = json.dumps({"model": MODEL, "temperature": 0.7, "max_tokens": 64,
                       "messages": [{"role": "user", "content": PROMPT[kind].format(p=rec["passage"])}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            return {**rec, "query": d["choices"][0]["message"]["content"].strip(), "port": port,
                    "usage": d["usage"]}
        except Exception as e:
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return {**rec, "query": None, "port": port, "error": err}


done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if r.get("query"):
            done.add(r["id"])
todo = (r for r in itertools.islice(records(), LIMIT) if r["id"] not in done)

out = open(OUT, "a")
lock = threading.Lock()
pools = {p: ThreadPoolExecutor(CONC) for p in PORTS}
t0 = time.time()
n = ok = 0
pending = set()
port_cycle = itertools.cycle(PORTS)
for rec in todo:
    while len(pending) >= CONC * len(PORTS) * 2:
        fin, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in fin:
            r = f.result()
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1; ok += bool(r.get("query"))
            if n % 10000 == 0:
                out.flush()
                el = time.time() - t0
                print(f"{time.strftime('%m-%d %H:%M')} done={n} ok={ok} rate={n/el:.1f}/s", flush=True)
    p = next(port_cycle)
    pending.add(pools[p].submit(call, p, rec))
for f in pending:
    r = f.result()
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    n += 1; ok += bool(r.get("query"))
out.close()
el = time.time() - t0
print(json.dumps({"new": n, "ok": ok, "skipped": len(done), "elapsed_s": round(el, 1),
                  "per_s": round(n / el, 2) if el else 0}), flush=True)
