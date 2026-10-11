#!/usr/bin/env python
"""Build the Ranunculus-v1-1B SFT dataset from Apache-2.0 conversation data.

Sources (both Apache-2.0 per the Hugging Face dataset metadata):
  * ``llm-jp/oasst2-33k-ja``  -- Japanese conversations (``conversations`` list).
  * ``OpenAssistant/oasst2``  -- message trees; English threads are rebuilt by
    following the top-ranked (rank 0) assistant reply at each turn.

Each conversation is rendered with the ChatML-style chat template, tokenized as a
whole, and labelled so that only assistant content plus its ``<|im_end|>`` carry
loss. Conversations whose assistant turns mention another model / assistant by
name are dropped so the model does not learn to introduce itself as one.

Usage:
    python prepare_ranunculus_sft.py --tokenizer_dir artifacts-0/tokenizer \
        --template artifacts-4/chat_template.jinja --out_dir artifacts-4/data
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import datasets
from transformers import AutoTokenizer

IDENTITY_RE = re.compile(
    r"open\s*-?\s*assistant|オープンアシスタント|chat\s*-?\s*gpt|gpt-?3|gpt-?4|openai|laion|qwen|claude|anthropic|"
    r"\bbard\b|gemini|llama|vicuna|alpaca|mistral",
    re.IGNORECASE,
)


def ja_conversations() -> list[list[dict]]:
    """Return the Japanese conversations as lists of ``{role, content}``."""
    ds = datasets.load_dataset("llm-jp/oasst2-33k-ja", split="train")
    return [list(ex["conversations"]) for ex in ds]


def en_conversations(split: str) -> list[list[dict]]:
    """Rebuild English oasst2 threads following rank-0 assistant replies.

    Args:
        split: ``train`` or ``validation`` split of ``OpenAssistant/oasst2``.

    Returns:
        Conversations that start with a user turn and end with an assistant turn.
    """
    ds = datasets.load_dataset("OpenAssistant/oasst2", split=split)
    msgs = {m["message_id"]: m for m in ds if m["lang"] == "en" and not m["deleted"] and m["review_result"]}
    children = defaultdict(list)
    for m in msgs.values():
        if m["parent_id"] in msgs:
            children[m["parent_id"]].append(m)
    roots = [m for m in msgs.values() if m["parent_id"] is None and m["role"] == "prompter"]

    def best_assistant(mid: str):
        cands = [c for c in children[mid] if c["role"] == "assistant"]
        ranked = [c for c in cands if c["rank"] == 0]
        if ranked:
            return ranked[0]
        return cands[0] if len(cands) == 1 else None

    out = []

    def walk(prompter, path):
        a = best_assistant(prompter["message_id"])
        if a is None:
            return
        path = path + [{"role": "user", "content": prompter["text"]}, {"role": "assistant", "content": a["text"]}]
        follow = [c for c in children[a["message_id"]] if c["role"] == "prompter"]
        ended = True
        for p in follow:
            before = len(out)
            walk(p, path)
            ended = ended and len(out) == before
        if ended:
            out.append(path)

    for r in roots:
        walk(r, [])
    return out


def has_identity(conv: list[dict]) -> bool:
    return any(t["role"] == "assistant" and IDENTITY_RE.search(t["content"]) for t in conv)


def encode(conv: list[dict], tok) -> dict:
    """Render, tokenize and build assistant-only labels for one conversation."""
    text = tok.apply_chat_template(conv, tokenize=False)
    spans, pos = [], 0
    for t in conv:
        head = f"<|im_start|>{t['role']}\n"
        start = text.index(head, pos) + len(head)
        end = start + len(t["content"]) + len("<|im_end|>")
        assert text[start:end] == t["content"] + "<|im_end|>"
        if t["role"] == "assistant":
            spans.append((start, end))
        pos = end
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = [
        i if any(s <= a < e for s, e in spans) else -100 for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    return {"input_ids": enc["input_ids"], "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--ja_val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.chat_template = Path(args.template).read_text()
    stats = Counter()

    ja = ja_conversations()
    random.Random(args.seed).shuffle(ja)
    n_val = int(len(ja) * args.ja_val_ratio)
    groups = {
        ("ja", "train"): ja[n_val:],
        ("ja", "val"): ja[:n_val],
        ("en", "train"): en_conversations("train"),
        ("en", "val"): en_conversations("validation"),
    }

    rows = {"train": [], "val": []}
    for (lang, split), convs in groups.items():
        stats[f"{lang}/{split}/raw"] = len(convs)
        for conv in convs:
            if not conv or conv[0]["role"] != "user" or conv[-1]["role"] != "assistant":
                stats[f"{lang}/{split}/drop_shape"] += 1
                continue
            if has_identity(conv):
                stats[f"{lang}/{split}/drop_identity"] += 1
                continue
            ex = encode(conv, tok)
            if len(ex["input_ids"]) > args.max_len:
                stats[f"{lang}/{split}/drop_too_long"] += 1
                continue
            ex["lang"] = lang
            rows[split].append(ex)
            stats[f"{lang}/{split}/kept"] += 1
            stats[f"{lang}/{split}/tokens"] += len(ex["input_ids"])
            stats[f"{lang}/{split}/loss_tokens"] += sum(x != -100 for x in ex["labels"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        random.Random(args.seed).shuffle(r)
        datasets.Dataset.from_list(r).save_to_disk(str(out / split))
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    # sanity: show one rendered example with its loss region
    ex = rows["train"][0]
    print(tok.decode(ex["input_ids"])[:600])
    print("--- loss region ---")
    print(tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[:400])


if __name__ == "__main__":
    main()
