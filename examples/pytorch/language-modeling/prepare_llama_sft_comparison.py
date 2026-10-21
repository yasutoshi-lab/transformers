#!/usr/bin/env python
"""Build the reasoning on/off SFT dataset for the Llama-3.2-1B / 3B comparison from the gpt-oss-20b data.

Takes ``yasutoshi-lab/gpt-oss-20b-self-distillation`` rows per domain:

    no_reasoning     5,000 sampled per domain (english: all)
    reasoning low    all
    reasoning medium english only, all

A row is dropped when its seed (``seed_source`` + ``seed_index``) or its normalized prompt was already
taken (dedup across domains and modes), or when it exceeds ``--max_len`` tokens after applying
``chat_templates/llama3_reasoning_toggle.jinja``.

Splits are stratified per domain: validation 10% and test 5% of the domain, each half no_reasoning and
half reasoning (english reasoning: half low, half medium); train gets the rest (ratio 8.5 / 1 / 0.5).

Thinking conversations get ``/think`` in the system turn half of the time and no system turn otherwise;
no-thinking conversations always get ``/no_think``. Loss covers the final assistant turn plus
``<|eot_id|>``; the empty think block of a no-thinking conversation is part of the prompt.

Usage:
    python prepare_llama_sft_comparison.py --tokenizer_dir artifacts-llama/tokenizer \
        --template chat_templates/llama3_reasoning_toggle.jinja --out_dir artifacts-cmp/sft-data
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

import datasets
from prepare_llama_reasoning_sft import encode, norm_prompt, to_messages

from transformers import AutoTokenizer


DOMAINS = ["code", "general_knowledge", "math", "japanese", "english", "cybersecurity"]
NONE_PER_DOMAIN = 5000
# (mode, reasoning_effort) groups taken per domain; ``None`` count means all rows.
GROUPS = {d: [("reasoning", "low"), ("no_reasoning", "")] for d in DOMAINS}
GROUPS["english"] = [("reasoning", "low"), ("reasoning", "medium"), ("no_reasoning", "")]
SPLIT_RATIO = {"validation": 0.10, "test": 0.05}


def load_source(source: str) -> datasets.Dataset:
    """Load the source from the Hub, or from a local directory of parquet files."""
    if Path(source).is_dir():
        files = sorted(glob.glob(str(Path(source) / "**" / "*.parquet"), recursive=True))
        return datasets.load_dataset("parquet", data_files=files, split="train")
    return datasets.load_dataset(source, split="train")


def split_quota(n_domain: int, efforts: list[str]) -> dict[str, dict[str, int]]:
    """Return split -> group key -> count for the held-out splits of one domain.

    Each held-out split is half no_reasoning and half reasoning; the reasoning half is divided
    evenly among ``efforts``.
    """
    quota = {}
    for split, ratio in SPLIT_RATIO.items():
        total = round(n_domain * ratio)
        q = {"no_reasoning/": total // 2}
        think = total - total // 2
        for i, e in enumerate(efforts):
            q[f"reasoning/{e}"] = think // len(efforts) + (1 if i < think % len(efforts) else 0)
        quota[split] = q
    return quota


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yasutoshi-lab/gpt-oss-20b-self-distillation")
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.chat_template = Path(args.template).read_text()
    df = load_source(args.source).to_pandas()
    df["seed_key"] = df.seed_source + "#" + df.seed_index.astype(str)
    df["norm"] = df.prompt.map(norm_prompt)

    rng = random.Random(args.seed)
    used_seed, used_norm = set(), set()
    picked = {d: {} for d in DOMAINS}  # domain -> group key -> examples
    stats = Counter()
    # Take the reasoning groups first: they are used in full, so dedup should cost the sampled
    # no_reasoning pool instead.
    for d in DOMAINS:
        for m, e in GROUPS[d]:
            key = f"{m}/{e}"
            idx = list(df.index[(df.domain == d) & (df["mode"] == m) & (df.reasoning_effort == e)])
            rng.shuffle(idx)
            limit = NONE_PER_DOMAIN if m == "no_reasoning" and d != "english" else len(idx)
            out = picked[d][key] = []
            for i in idx:
                if len(out) >= limit:
                    break
                row = df.loc[i]
                stats[f"{d}/{key}/examined"] += 1
                if row.seed_key in used_seed or row.norm in used_norm:
                    stats[f"{d}/{key}/drop_duplicate"] += 1
                    continue
                msgs = to_messages(row, rng)
                ex = encode(msgs, m == "reasoning", tok)
                if len(ex["input_ids"]) > args.max_len:
                    stats[f"{d}/{key}/drop_too_long"] += 1
                    continue
                used_seed.add(row.seed_key)
                used_norm.add(row.norm)
                ex.update(messages=msgs, domain=d, mode=m, reasoning_effort=e, lang=row.prompt_lang,
                          seed_source=row.seed_source, seed_index=int(row.seed_index))
                out.append(ex)
            stats[f"{d}/{key}/taken"] = len(out)

    rows = {"train": [], "validation": [], "test": []}
    for d in DOMAINS:
        n_domain = sum(len(v) for v in picked[d].values())
        efforts = [e for m, e in GROUPS[d] if m == "reasoning"]
        quota = split_quota(n_domain, efforts)
        for key, exs in picked[d].items():
            start = 0
            for split in ("test", "validation"):
                n = quota[split][key]
                rows[split] += exs[start:start + n]
                start += n
            rows["train"] += exs[start:]

    for split, r in rows.items():
        for ex in r:
            stats[f"{split}/{ex['domain']}/{ex['mode']}/{ex['reasoning_effort']}"] += 1
            stats[f"{split}/{ex['mode']}/tokens"] += len(ex["input_ids"])
            stats[f"{split}/{ex['mode']}/loss_tokens"] += sum(x != -100 for x in ex["labels"])
        stats[f"{split}/total"] = len(r)
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
    for m in ("reasoning", "no_reasoning"):
        ex = next(e for e in rows["train"] if e["mode"] == m)
        print(f"=== {m}\n" + tok.decode(ex["input_ids"])[:500])
        print("--- loss region (head) ---\n" + tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[:300])


if __name__ == "__main__":
    main()
