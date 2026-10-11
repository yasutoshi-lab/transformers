#!/usr/bin/env python
"""Score the Japanese translations of the DPO pairs with an LLM judge (vLLM servers).

Each translated text (every prompt turn, ``chosen`` and ``rejected``) is shown to
the judge together with its English source. The judge returns JSON with 1-5
scores for accuracy, fluency and completeness, a flag for "not a translation"
(the text was answered / rewritten instead of translated), and a short reason.
Results are appended to a JSONL file so an interrupted run can be resumed.

Usage:
    python judge_translations.py --ja artifacts-5/dpo-data/pairs_ja_train.jsonl \
        --out artifacts-5/dpo-data/judge_train.jsonl \
        --endpoints http://127.0.0.1:18100/v1 http://127.0.0.1:18101/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

RUBRIC = """You are a strict evaluator of English-to-Japanese translations.
You will receive an English SOURCE text and its Japanese TRANSLATION. Evaluate only the translation quality,
not the quality or correctness of the content itself.

Score each item from 1 to 5:
- accuracy: Is the meaning of the source conveyed correctly? (5 = no meaning errors, 3 = some minor errors,
  1 = meaning is largely wrong)
- fluency: Is the Japanese natural and grammatical? (5 = reads like text written by a native speaker,
  3 = understandable but awkward, 1 = hard to read)
- completeness: Is everything translated, with nothing omitted and nothing added? (5 = nothing omitted or
  added, 3 = a sentence or so omitted/added, 1 = large parts omitted or extra content added)
- not_translation: true if the TRANSLATION is not a translation of the SOURCE (e.g. it answers or carries
  out the source instead of translating it, or is in another language), otherwise false.
Code, commands, URLs and identifiers left in the original form are correct, not errors.

Reply with JSON only, in this exact form:
{"accuracy": <1-5>, "fluency": <1-5>, "completeness": <1-5>, "not_translation": <true|false>, "reason": "<one short sentence in Japanese>"}"""


def parse(text: str) -> dict | None:
    """Extract the JSON verdict from the judge output."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    keys = ("accuracy", "fluency", "completeness", "not_translation")
    return v if all(k in v for k in keys) else None


async def judge(client: httpx.AsyncClient, url: str, src: str, tgt: str, model: str) -> dict:
    """Ask the judge for one verdict, retrying on transport or parse errors."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"SOURCE:\n<<<\n{src}\n>>>\n\nTRANSLATION:\n<<<\n{tgt}\n>>>"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last = ""
    for attempt in range(4):
        try:
            r = await client.post(f"{url}/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            last = r.json()["choices"][0]["message"]["content"] or ""
            v = parse(last)
            if v is not None:
                return v
        except (httpx.HTTPError, KeyError):
            await asyncio.sleep(5 * (attempt + 1))
    return {"error": "unparsable", "raw": last[:500]}


async def main_async(args) -> None:
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    rows = [r for r in map(json.loads, open(args.ja, encoding="utf-8")) if r["id"] not in done]
    print(f"todo={len(rows)} done={len(done)}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    n = [0]
    n_fail = [0]

    async with httpx.AsyncClient() as client:

        async def one(i: int, r: dict) -> None:
            url = args.endpoints[i % len(args.endpoints)]
            en = r["source_en"]
            items = [(f"prompt[{k}]", t["content"], u["content"]) for k, (t, u) in enumerate(zip(en["prompt"], r["prompt"]))]
            items += [("chosen", en["chosen"], r["chosen"]), ("rejected", en["rejected"], r["rejected"])]
            async with sem:
                verdicts = await asyncio.gather(*(judge(client, url, s, t, args.model) for _, s, t in items))
            row = {"id": r["id"], "verdicts": {name: v for (name, _, _), v in zip(items, verdicts)}}
            if any(v.get("error") and not v.get("raw") for v in verdicts):
                # Transport failure (server down): do not record, so a resumed run judges it again.
                async with lock:
                    n_fail[0] += 1
                return
            async with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n[0] += 1
                if n[0] % 200 == 0:
                    print(f"judged {n[0]}/{len(rows)}", flush=True)

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    print(f"finished {n[0]} transport_failures={n_fail[0]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ja", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
