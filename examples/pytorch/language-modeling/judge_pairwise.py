#!/usr/bin/env python
"""Blind pairwise comparison of two models' responses with an LLM judge (vLLM servers).

For each prompt the judge sees the conversation and two responses labelled A and B,
without model names, and is asked twice with the order swapped. A model wins a prompt
only if it wins in both orders; otherwise it is a tie (position-consistent judging).

Usage:
    python judge_pairwise.py --data_dir artifacts-5/dpo-dataset --split validation \
        --a artifacts-6/pairwise/sft_v4.jsonl --b artifacts-6/pairwise/dpo.jsonl \
        --out artifacts-6/pairwise/judge.jsonl --endpoints http://127.0.0.1:18100/v1 http://127.0.0.1:18101/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import datasets
import httpx

RUBRIC = """You are an impartial judge comparing two AI assistant responses to the same conversation.
Judge which response better serves the user's last message: helpfulness, correctness of facts, following the
user's instructions (language, format, length), clarity, and absence of repetition or rambling.
Do not prefer a response just because it is longer. Ignore which position (A or B) a response is in.
The responses come from small models and may both be poor; still pick the better one, or "tie" only if they are
truly equal in quality.

Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one short sentence in Japanese>"}"""


def render(prompt: list[dict]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt)


async def ask(client, url, convo, ra, rb, model):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": f"CONVERSATION:\n<<<\n{convo}\n>>>\n\nRESPONSE A:\n<<<\n{ra}\n>>>\n\nRESPONSE B:\n<<<\n{rb}\n>>>"}],
        "temperature": 0.0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
    }
    last = ""
    for attempt in range(4):
        try:
            r = await client.post(f"{url}/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            last = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", last, flags=re.S)
            if m:
                v = json.loads(m.group(0))
                if v.get("winner") in ("A", "B", "tie"):
                    return v
        except (httpx.HTTPError, KeyError, json.JSONDecodeError):
            await asyncio.sleep(5 * (attempt + 1))
    return None if not last else {"winner": "invalid", "raw": last[:300]}


async def main_async(args):
    prompts = {r["id"]: r["prompt"] for r in datasets.load_from_disk(args.data_dir)[args.split]}
    a = {r["id"]: r for r in map(json.loads, open(args.a, encoding="utf-8"))}
    b = {r["id"]: r for r in map(json.loads, open(args.b, encoding="utf-8"))}
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    ids = [i for i in a if i in b and i not in done]
    print(f"todo={len(ids)} done={len(done)}", flush=True)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    async with httpx.AsyncClient() as client:
        async def one(k, i):
            url = args.endpoints[k % len(args.endpoints)]
            convo = render(prompts[i])
            async with sem:
                v1, v2 = await asyncio.gather(ask(client, url, convo, a[i]["response"], b[i]["response"], args.model),
                                              ask(client, url, convo, b[i]["response"], a[i]["response"], args.model))
            if v1 is None or v2 is None:
                return  # transport failure: retry on resume
            w1 = {"A": "a", "B": "b"}.get(v1["winner"], "tie")
            w2 = {"A": "b", "B": "a"}.get(v2["winner"], "tie")
            final = w1 if w1 == w2 else "tie"
            row = {"id": i, "lang": a[i]["lang"], "order_ab": v1, "order_ba": v2, "winner": final}
            async with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        await asyncio.gather(*(one(k, i) for k, i in enumerate(ids)))
    print("finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
