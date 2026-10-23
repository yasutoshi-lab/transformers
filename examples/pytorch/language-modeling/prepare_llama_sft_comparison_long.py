#!/usr/bin/env python
"""Build the long-thinking counterpart of the Llama-3.2 1B/3B SFT comparison dataset.

Same as ``prepare_llama_sft_comparison.py`` except for the thinking conversations, so that the two
SFT runs differ only in the length of the thinking:

* no_reasoning conversations: copied unchanged (same rows, same splits) from the short-thinking
  dataset written by ``prepare_llama_sft_comparison.py`` (``--short_dir``);
* reasoning conversations: every ``high`` row plus ``medium`` rows sampled to reach exactly the
  short dataset's count per domain and split. english uses all ``medium`` and all ``high``.

Held-out splits keep the short dataset's reasoning counts; english splits them evenly between
medium and high, the other domains in proportion to the medium/high counts. A row is dropped when
its seed or normalized prompt was already taken (including by the no_reasoning rows) or when it
exceeds ``--max_len`` tokens.

Usage:
    python prepare_llama_sft_comparison_long.py --tokenizer_dir artifacts-cmp/tokenizer \
        --template chat_templates/llama3_reasoning_toggle.jinja --short_dir artifacts-cmp/sft-data \
        --out_dir artifacts-cmp/sft-data-long
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import datasets
from prepare_llama_reasoning_sft import encode, norm_prompt, to_messages
from prepare_llama_sft_comparison import DOMAINS, load_source

from transformers import AutoTokenizer


SPLITS = ("train", "validation", "test")
HELD_OUT = ("test", "validation")


def held_out_quota(total: int, sizes: dict[str, int], even: bool) -> dict[str, int]:
    """Divide ``total`` held-out conversations between efforts (evenly, or in proportion to ``sizes``)."""
    efforts = sorted(sizes)
    if even:
        q = {e: total // len(efforts) for e in efforts}
    else:
        n = sum(sizes.values())
        q = {e: total * sizes[e] // n for e in efforts}
    for e in sorted(efforts, key=lambda e: -sizes[e])[: total - sum(q.values())]:
        q[e] += 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yasutoshi-lab/gpt-oss-20b-self-distillation")
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--short_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.chat_template = Path(args.template).read_text()
    short = {s: [json.loads(line) for line in open(Path(args.short_dir) / f"{s}.jsonl")] for s in SPLITS}

    stats = Counter()
    rows = {s: [] for s in SPLITS}
    used_seed, used_norm = set(), set()
    # domain -> split -> number of reasoning conversations in the short dataset
    target = {d: Counter() for d in DOMAINS}
    for split, recs in short.items():
        for r in recs:
            if r["mode"] == "reasoning":
                target[r["domain"]][split] += 1
                continue
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            used_seed.add(f"{r['seed_source']}#{r['seed_index']}")
            used_norm.add(norm_prompt(user))
            ex = encode(r["messages"], False, tok)
            ex.update({k: r[k] for k in ("messages", "domain", "mode", "reasoning_effort", "lang", "seed_source", "seed_index")})
            rows[split].append(ex)

    df = load_source(args.source).to_pandas()
    df["seed_key"] = df.seed_source + "#" + df.seed_index.astype(str)
    df["norm"] = df.prompt.map(norm_prompt)
    rng = random.Random(args.seed)

    def take(d: str, effort: str, limit: int) -> list[dict]:
        idx = list(df.index[(df.domain == d) & (df["mode"] == "reasoning") & (df.reasoning_effort == effort)])
        rng.shuffle(idx)
        out = []
        for i in idx:
            if len(out) >= limit:
                break
            row = df.loc[i]
            if row.seed_key in used_seed or row.norm in used_norm:
                stats[f"{d}/{effort}/drop_duplicate"] += 1
                continue
            msgs = to_messages(row, rng)
            ex = encode(msgs, True, tok)
            if len(ex["input_ids"]) > args.max_len:
                stats[f"{d}/{effort}/drop_too_long"] += 1
                continue
            used_seed.add(row.seed_key)
            used_norm.add(row.norm)
            ex.update(messages=msgs, domain=d, mode="reasoning", reasoning_effort=effort, lang=row.prompt_lang,
                      seed_source=row.seed_source, seed_index=int(row.seed_index))
            out.append(ex)
        return out

    for d in DOMAINS:
        need = sum(target[d].values())
        # high first (all of it), then medium up to the short dataset's count.
        picked = {"high": take(d, "high", need)}
        picked["medium"] = take(d, "medium", need - len(picked["high"]))
        assert sum(len(v) for v in picked.values()) == need, (d, need, {k: len(v) for k, v in picked.items()})
        sizes = {e: len(v) for e, v in picked.items() if v}
        start = Counter()
        for split in HELD_OUT:
            q = held_out_quota(target[d][split], sizes, even=(d == "english"))
            for e, n in q.items():
                rows[split] += picked[e][start[e]:start[e] + n]
                start[e] += n
        for e in sizes:
            rows["train"] += picked[e][start[e]:]
        for e in sizes:
            stats[f"{d}/{e}/taken"] = sizes[e]

    for split, r in rows.items():
        for ex in r:
            stats[f"{split}/{ex['domain']}/{ex['mode']}/{ex['reasoning_effort']}"] += 1
            stats[f"{split}/{ex['mode']}/tokens"] += len(ex["input_ids"])
            stats[f"{split}/{ex['mode']}/loss_tokens"] += sum(x != -100 for x in ex["labels"])
        stats[f"{split}/total"] = len(r)
        assert len(r) == len(short[split]), (split, len(r), len(short[split]))
    keys = [{(e["seed_source"], e["seed_index"]) for e in r} for r in rows.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]), "seed overlap between splits"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        random.Random(args.seed).shuffle(r)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in r:
                rec = {k: v for k, v in ex.items() if k not in ("input_ids", "labels")}
                rec["num_tokens"] = len(ex["input_ids"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cols = ("input_ids", "labels", "domain", "mode", "reasoning_effort")
        datasets.Dataset.from_list([{k: ex[k] for k in cols} for ex in r]).save_to_disk(str(out / split))
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
