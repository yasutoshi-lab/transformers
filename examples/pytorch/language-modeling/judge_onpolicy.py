#!/usr/bin/env python
"""Score on-policy thinking-mode samples with an LLM judge and build DPO pairs (stage 2).

Samples that did not close ``</think>`` or did not stop with ``<|im_end|>`` get score 0
without asking the judge (unfinished thinking and endless generation are exactly the
failures to push down). For the rest, the judge scores the final answer (after
``</think>``) from 1 to 10. Per prompt, the highest- and lowest-scored samples form a
pair when the chosen one is finished and the score gap is at least ``--min_gap``.

Usage:
    python judge_onpolicy.py --gen artifacts-6/onpolicy/gen_*.jsonl --out artifacts-6/onpolicy/scored.jsonl \
        --pairs_out artifacts-6/onpolicy/pairs.jsonl --endpoints http://127.0.0.1:18100/v1 http://127.0.0.1:18101/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

RUBRIC = """You are a strict evaluator of an AI assistant's answer to the user's last message.
Score the ANSWER from 1 to 10 for helpfulness, factual correctness, following the user's instructions
(language, format, length), and clarity. Penalize repetition, rambling and content unrelated to the question.
Do not reward length by itself: a shorter answer that fully addresses the request is better than a longer one. Reply with JSON only: {"score": <1-10>, "reason": "<one short sentence in Japanese>"}"""


def render(prompt):
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt)


async def score(client, url, convo, answer, model):
    body = {"model": model,
            "messages": [{"role": "system", "content": RUBRIC},
                         {"role": "user", "content": f"CONVERSATION:\n<<<\n{convo}\n>>>\n\nANSWER:\n<<<\n{answer}\n>>>"}],
            "temperature": 0.0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            r = await client.post(f"{url}/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            t = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", t, flags=re.S)
            if m:
                v = json.loads(m.group(0))
                if isinstance(v.get("score"), (int, float)):
                    return v
        except (httpx.HTTPError, KeyError, json.JSONDecodeError):
            await asyncio.sleep(5 * (attempt + 1))
    return None


async def main_async(args):
    rows = [json.loads(l) for f in args.gen for l in open(f, encoding="utf-8")]
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"todo={len(todo)} done={len(done)}", flush=True)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    async with httpx.AsyncClient() as client:
        async def one(k, r):
            url = args.endpoints[k % len(args.endpoints)]
            convo = render(r["prompt"])
            async def s_one(smp):
                if not (smp["closed_think"] and smp["stopped"]):
                    return {"score": 0, "reason": "unfinished (no </think> or no <|im_end|>)", "auto": True}
                ans = smp["text"].split("</think>", 1)[1].strip()
                return await score(client, url, convo, ans, args.model)
            async with sem:
                verdicts = await asyncio.gather(*(s_one(s) for s in r["samples"]))
            if any(v is None for v in verdicts):
                return
            row = {"id": r["id"], "lang": r["lang"], "scores": [v["score"] for v in verdicts], "verdicts": verdicts}
            async with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        await asyncio.gather(*(one(k, r) for k, r in enumerate(todo)))
    # build pairs
    scored = {json.loads(l)["id"]: json.loads(l) for l in out.open(encoding="utf-8")}
    n_pairs = 0
    n_len_drop = 0
    with open(args.pairs_out, "w", encoding="utf-8") as f:
        for r in rows:
            sc = scored.get(r["id"])
            if sc is None:
                continue
            s = sc["scores"]
            hi = max(range(len(s)), key=lambda i: s[i])
            lo = min(range(len(s)), key=lambda i: s[i])
            ch = r["samples"][hi]
            if s[hi] - s[lo] < args.min_gap or not (ch["closed_think"] and ch["stopped"]):
                continue
            # Length guard: drop pairs whose chosen is much longer than the rejected one, so that DPO does not
            # simply learn "longer is better" (stage 1 did). Unfinished rejected samples are always kept.
            rj = r["samples"][lo]
            if rj["closed_think"] and rj["stopped"] and len(ch["ids"]) > args.max_len_ratio * max(len(rj["ids"]), 1):
                n_len_drop += 1
                continue
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "prompt_ids": r["prompt_ids"],
                                "chosen_ids": ch["ids"] + [args.im_end_id], "rejected_ids": r["samples"][lo]["ids"] + ([args.im_end_id] if r["samples"][lo]["stopped"] else []),
                                "chosen_score": s[hi], "rejected_score": s[lo]}, ensure_ascii=False) + "\n")
            n_pairs += 1
    print(f"pairs={n_pairs} dropped_by_length_ratio={n_len_drop}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_out", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--min_gap", type=float, default=2)
    ap.add_argument("--im_end_id", type=int, default=95747)
    ap.add_argument("--max_len_ratio", type=float, default=1.5)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
