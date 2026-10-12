"""パイロット出力のレビュー用集計（通過率・不合格理由・長さ・抽出サンプル）。"""

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
recs = [json.loads(l) for l in open(path, encoding="utf-8").read().split("\n") if l.strip()]
print("records", len(recs))
by = defaultdict(lambda: [0, 0])
reasons = defaultdict(Counter)
for r in recs:
    k = (r["domain"], r["mode"])
    by[k][0] += 1
    ok = r["passed"] and not r["duplicate"]
    by[k][1] += ok
    for x in r["reasons"]:
        reasons[k][x.split(":")[0]] += 1
    if r["duplicate"]:
        reasons[k]["dup_" + r["duplicate"]] += 1
for k in sorted(by):
    t, ok = by[k]
    print(f"{k[0]:18s} {k[1]:13s} {ok:4d}/{t:4d} {ok / t:5.0%}  {dict(reasons[k])}")
tot = sum(v[0] for v in by.values())
ok = sum(v[1] for v in by.values())
print(f"ALL {ok}/{tot} {ok / tot:.1%}")

kept = [r for r in recs if r["passed"] and not r["duplicate"]]
for mode in ("reasoning", "no_reasoning"):
    ks = [r for r in kept if r["mode"] == mode]
    if not ks:
        continue
    th = [len(r["thinking"]) for r in ks if r["thinking"]]
    rs = [len(r["response"]) for r in ks]
    ct = [r.get("completion_tokens", 0) for r in ks]
    print(mode, "n", len(ks), "thinking chars med", statistics.median(th) if th else 0,
          "resp chars med", statistics.median(rs), "completion tok med/max", statistics.median(ct), max(ct))

# 思考の末尾・言語指示への言及などのパターン
pat = {
    "mentions_user_language": re.compile(r"user (?:asked|wrote|writes|is asking) in (?:English|Japanese)|answer in (?:English|Japanese)", re.I),
    "drafting_markers": re.compile(r"\*Drafting|\*Check:|\*Self-Correction|\*Final", re.I),
    "thinking_has_ja": re.compile(r"[぀-ヿ]"),
}
for name, p in pat.items():
    c = sum(1 for r in kept if r["mode"] == "reasoning" and p.search(r["thinking"]))
    print("kept reasoning with", name, c)
resp_pat = {"resp_starts_prefix": None}
c = sum(1 for r in kept if r["response"].startswith(r["prefix"]) or r["thinking"].startswith(r["prefix"]))
print("kept starting with prefix", c, "/", len(kept))
lead = Counter(r["response"][:12] for r in kept if r["mode"] == "no_reasoning")
print("no_reasoning response heads", lead.most_common(8))

rng = random.Random(0)
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    with open(out, "w", encoding="utf-8") as fh:
        for k in sorted(by):
            pool = [r for r in kept if (r["domain"], r["mode"]) == k]
            for r in rng.sample(pool, min(3, len(pool))):
                fh.write(f"===== {k} prefix={r['prefix']!r} tokens={r.get('completion_tokens')}\n")
                fh.write("PROMPT: " + r["prompt"][:600] + "\n")
                if r["thinking"]:
                    fh.write("THINKING: " + r["thinking"][:1500] + "\n...[tail] " + r["thinking"][-400:] + "\n")
                fh.write("RESPONSE: " + r["response"][:1500] + "\n\n")
        fh.write("\n\n######## REJECTED SAMPLES\n")
        rej = [r for r in recs if not r["passed"]]
        for r in rng.sample(rej, min(15, len(rej))):
            fh.write(f"===== {r['domain']}/{r['mode']} reasons={r['reasons']} finish={r['finish_reason']}\n")
            fh.write("PROMPT: " + r["prompt"][:300] + "\n")
            if r["thinking"]:
                fh.write("THINKING tail: " + r["thinking"][-500:] + "\n")
            fh.write("RESPONSE: " + r["response"][:800] + "\n\n")
