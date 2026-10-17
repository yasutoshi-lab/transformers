#!/usr/bin/env python
"""Build the reasoning on/off SFT dataset for Llama-3.2-1B from the Gemma 4 self-distillation data.

Samples ``yasutoshi-lab/gemma4-26b-a4b-it-self-distillation`` per domain x mode without
reusing a seed or a (normalized) prompt, keeping only conversations that fit in
``--max_len`` tokens after applying ``chat_templates/llama3_reasoning_toggle.jinja``.

Splits (conversations per mode):
    test        43 per domain (5 domains, balanced)
    validation 129 per domain (5 domains, balanced)
    train     4800 per domain, english 1440 (code:general_knowledge:math:japanese:english = 1:1:1:1:0.3)

Thinking conversations get ``/think`` in the system turn half of the time and no system turn
otherwise (the template defaults to thinking on); no-thinking conversations always get ``/no_think``.
Loss covers the final assistant turn plus ``<|eot_id|>``; the empty think block of a no-thinking
conversation is part of the prompt (the template adds it when thinking is off).

Usage:
    python prepare_llama_reasoning_sft.py --tokenizer_dir artifacts-llama/tokenizer \
        --template chat_templates/llama3_reasoning_toggle.jinja --out_dir artifacts-llama/sft-data
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import datasets

from transformers import AutoTokenizer


DOMAINS = ["code", "general_knowledge", "math", "japanese", "english"]
MODES = ["reasoning", "no_reasoning"]
# split -> domain -> conversations per mode. Earlier splits are filled first.
PER_MODE = {
    "test": {d: 43 for d in DOMAINS},
    "validation": {d: 129 for d in DOMAINS},
    "train": {**{d: 4800 for d in DOMAINS[:4]}, "english": 1440},
}
SPLITS = ["test", "validation", "train"]
EMPTY_THINK = "<think>\n\n</think>\n\n"


def norm_prompt(text: str) -> str:
    """Normalize a prompt for duplicate detection (NFKC, collapsed whitespace, lower case)."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


def to_messages(row: dict, rng: random.Random) -> list[dict]:
    """Turn one self-distillation row into a single-turn conversation.

    Args:
        row: Row of the self-distillation dataset.
        rng: Random generator deciding whether a thinking conversation gets ``/think``.

    Returns:
        Messages whose final assistant turn starts with a ``<think>`` block (empty when thinking is off).
    """
    if row["mode"] == "reasoning":
        assistant = f"<think>\n{row['thinking'].strip()}\n</think>\n\n{row['response'].strip()}"
        system = "/think" if rng.random() < 0.5 else None
    else:
        assistant = EMPTY_THINK + row["response"].strip()
        system = "/no_think"
    msgs = [] if system is None else [{"role": "system", "content": system}]
    return msgs + [{"role": "user", "content": row["prompt"].strip()}, {"role": "assistant", "content": assistant}]


def encode(msgs: list[dict], thinking: bool, tok) -> dict:
    """Render with the template and label the final assistant turn.

    The prompt is rendered with ``add_generation_prompt`` and the matching ``enable_thinking``, so the
    empty think block of a no-thinking conversation falls on the prompt side and gets no loss.

    Raises:
        AssertionError: If the prompt is not a token-level prefix of the full conversation.
    """
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    assert full.startswith(prompt)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert ids[: len(p_ids)] == p_ids
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yasutoshi-lab/gemma4-26b-a4b-it-self-distillation")
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.chat_template = Path(args.template).read_text()
    src = datasets.load_dataset(args.source, split="train")
    df = src.to_pandas()
    df["seed_key"] = df.seed_source + "#" + df.seed_index.astype(str)
    df["norm"] = df.prompt.map(norm_prompt)

    rng = random.Random(args.seed)
    used_seed, used_norm = set(), set()
    rows = {s: [] for s in SPLITS}
    stats = Counter()
    for d in DOMAINS:
        for m in MODES:
            idx = list(df.index[(df.domain == d) & (df["mode"] == m)])
            rng.shuffle(idx)
            need = {s: PER_MODE[s][d] for s in SPLITS}
            for i in idx:
                split = next((s for s in SPLITS if need[s] > 0), None)
                if split is None:
                    break
                row = df.loc[i]
                stats[f"{d}/{m}/examined"] += 1
                if row.seed_key in used_seed or row.norm in used_norm:
                    stats[f"{d}/{m}/drop_duplicate"] += 1
                    continue
                msgs = to_messages(row, rng)
                ex = encode(msgs, m == "reasoning", tok)
                if len(ex["input_ids"]) > args.max_len:
                    stats[f"{d}/{m}/drop_too_long"] += 1
                    continue
                used_seed.add(row.seed_key)
                used_norm.add(row.norm)
                need[split] -= 1
                ex.update(messages=msgs, domain=d, mode=m, lang=row.lang, seed_source=row.seed_source,
                          seed_index=int(row.seed_index))
                rows[split].append(ex)
                stats[f"{split}/{m}/tokens"] += len(ex["input_ids"])
                stats[f"{split}/{m}/loss_tokens"] += sum(x != -100 for x in ex["labels"])
            assert all(v == 0 for v in need.values()), (d, m, need)

    # What the model is trained to emit right after the assistant header: <think> for thinking
    # conversations, nothing (prompt side) for no-thinking ones.
    header = tok("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)["input_ids"]
    for split, r in rows.items():
        for ex in r:
            ids, lab = ex["input_ids"], ex["labels"]
            for j in range(len(ids) - len(header)):
                if ids[j:j + len(header)] == header:
                    k = j + len(header)
                    kind = "masked" if lab[k] == -100 else tok.decode([ids[k]])
                    stats[f"{split}/after_assistant_header/{kind}"] += 1
            stats[f"{split}/{ex['domain']}/{ex['mode']}"] += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        random.Random(args.seed).shuffle(r)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in r:
                f.write(json.dumps({k: v for k, v in ex.items() if k not in ("input_ids", "labels")},
                                   ensure_ascii=False) + "\n")
        ds = datasets.Dataset.from_list([{k: ex[k] for k in ("input_ids", "labels", "domain", "mode")} for ex in r])
        ds.save_to_disk(str(out / split))
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    for m in MODES:
        ex = next(e for e in rows["train"] if e["mode"] == m)
        print(f"=== {m}\n" + tok.decode(ex["input_ids"])[:500])
        print("--- loss region (head) ---\n" + tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[:300])


if __name__ == "__main__":
    main()
