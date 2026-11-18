"""freesia-supervised-cl: 負例の候補のうち、実は質問の答えになっているもの（偽の負例）を Gemma-4 に判定させる（再開可能）。

使い方: python3 -I supcl_judge.py GEN.jsonl MINE.jsonl OUT.jsonl CONC PORTS
- own_rank が 10 位以内で、候補が 7 件以上ある行だけを判定する
- 正例と候補（最大 10 件）を固定 seed で混ぜて番号を振り、答えが書かれている文章の番号をすべて挙げさせる
- 出力の answers（番号）と order（番号ごとの pt_id。"POS" は正例）を保存する。正例を挙げなかった行は後で除く
"""
import gzip, json, os, random, re, sys, time, itertools, urllib.request
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

GEN, MINE, OUT, CONC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
PORTS = [int(p) for p in sys.argv[5].split(",")]
MODEL = "gemma4"
D = os.path.expanduser("~/freesia-work/sup-cl")
TOPK, MIN_NEG = 10, 7

PROMPT = {
    "ja": "次の質問と、番号つきの文章 {n} 件があります。質問の答えが書かれている文章（その文章だけを読めば質問に答えられるもの）の番号をすべて挙げてください。"
          "話題が近いだけで答えが書かれていない文章は挙げないでください。\n出力は {{\"answers\": [番号, ...]}} の JSON のみ。該当がなければ {{\"answers\": []}}。\n\n質問: {q}\n\n{docs}",
    "en": "Below are a query and {n} numbered passages. List the numbers of every passage that contains the answer to the query "
          "(a passage that alone lets you answer it). Do not list passages that are only on a related topic.\n"
          "Output only the JSON {{\"answers\": [numbers, ...]}}, or {{\"answers\": []}} if none.\n\nQuery: {q}\n\n{docs}",
}


def load():
    gen = {}
    for l in open(GEN):
        r = json.loads(l)
        if r["task"] == "retrieval" and r.get("query"):
            gen[r["id"]] = r
    need = set()
    mine = []
    for l in open(MINE):
        m = json.loads(l)
        if m["own_rank"] <= TOPK and len(m["cands"]) >= MIN_NEG:
            mine.append(m)
            need.update(c[0] for c in m["cands"])
    text = {}
    for lang in ("ja", "en"):
        for l in gzip.open(f"{D}/pool_{lang}.jsonl.gz", "rt"):
            r = json.loads(l)
            if r["pt_id"] in need:
                text[r["pt_id"]] = r["passage"]
    return gen, mine, text


def call(port, r, m, text):
    order = ["POS"] + [c[0] for c in m["cands"]]
    random.Random(r["id"]).shuffle(order)
    docs = "\n\n".join(f"[{k + 1}] " + (r["text"] if p == "POS" else text[p]) for k, p in enumerate(order))
    prompt = PROMPT[r["lang"]].format(n=len(order), q=r["query"], docs=docs)
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 64,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                                       {"Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            s = d["choices"][0]["message"]["content"]
            mm = re.search(r"\{.*\}", s, re.S)
            ans = json.loads(mm.group(0))["answers"] if mm else None
            ans = [int(a) for a in ans] if isinstance(ans, list) else None
            return {"id": r["id"], "order": order, "answers": ans, "raw": None if ans is not None else s, "port": port}
        except Exception as e:
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return {"id": r["id"], "order": order, "answers": None, "error": err, "port": port}


def main():
    gen, mine, text = load()
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT) if json.loads(l).get("answers") is not None}
    out = open(OUT, "a")
    pools = {p: ThreadPoolExecutor(CONC) for p in PORTS}
    cyc = itertools.cycle(PORTS)
    pending, n, t0 = set(), 0, time.time()

    def drain(fin):
        nonlocal n
        for f in fin:
            out.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            n += 1
            if n % 10000 == 0:
                out.flush()
                print(f"{time.strftime('%m-%d %H:%M')} judged={n} rate={n / (time.time() - t0):.1f}/s", flush=True)

    for m in mine:
        if m["id"] in done:
            continue
        while len(pending) >= CONC * len(PORTS) * 2:
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            drain(fin)
        p = next(cyc)
        pending.add(pools[p].submit(call, p, gen[m["id"]], m, text))
    drain(pending)
    out.close()
    print(json.dumps({"judged": n, "skipped": len(done), "eligible": len(mine), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
