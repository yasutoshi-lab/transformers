#!/usr/bin/env python
"""Build English DPO pairs from the human rankings in ``OpenAssistant/oasst2``.

For every user (prompter) message that has two or more ranked English assistant
replies, the rank-0 reply becomes ``chosen`` and the lowest-ranked reply becomes
``rejected``. The prompt is the conversation path from the root to that user
message. Pairs whose texts mention another model / assistant by name (same rule as
the SFT data) are dropped. Half of the training pairs are marked for Japanese
translation.

Usage:
    python prepare_oasst2_dpo_pairs.py --out_dir artifacts-5/dpo-data
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import datasets
from prepare_ranunculus_sft import IDENTITY_RE


def build_pairs(split: str, stats: Counter) -> list[dict]:
    """Return DPO pairs from one oasst2 split.

    Args:
        split: ``train`` or ``validation``.
        stats: Counter updated with drop reasons.

    Returns:
        Pairs with ``prompt`` (list of ``{role, content}``), ``chosen`` and ``rejected``.
    """
    ds = datasets.load_dataset("OpenAssistant/oasst2", split=split)
    msgs = {m["message_id"]: m for m in ds if m["lang"] == "en" and not m["deleted"] and m["review_result"]}
    children = defaultdict(list)
    for m in msgs.values():
        if m["parent_id"] in msgs:
            children[m["parent_id"]].append(m)

    def path(mid: str) -> list[dict] | None:
        out = []
        while mid is not None:
            m = msgs.get(mid)
            if m is None:
                return None
            out.append({"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"]})
            mid = m["parent_id"]
        return out[::-1]

    pairs = []
    for mid, kids in children.items():
        if msgs[mid]["role"] != "prompter":
            continue
        ranked = sorted((k for k in kids if k["role"] == "assistant" and k["rank"] is not None), key=lambda k: k["rank"])
        if len(ranked) < 2:
            continue
        stats[f"{split}/candidates"] += 1
        chosen, rejected = ranked[0], ranked[-1]
        prompt = path(mid)
        if prompt is None or prompt[0]["role"] != "user":
            stats[f"{split}/drop_path"] += 1
            continue
        texts = [chosen["text"], rejected["text"]] + [t["content"] for t in prompt if t["role"] == "assistant"]
        if any(IDENTITY_RE.search(t) for t in texts):
            stats[f"{split}/drop_identity"] += 1
            continue
        pairs.append({
            "id": mid,
            "prompt": prompt,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "chosen_rank": chosen["rank"],
            "rejected_rank": rejected["rank"],
            "n_ranked": len(ranked),
        })
    stats[f"{split}/kept"] = len(pairs)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = Counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        pairs = build_pairs(split, stats)
        random.Random(args.seed).shuffle(pairs)
        half = len(pairs) // 2
        for i, p in enumerate(pairs):
            p["translate_to_ja"] = i < half
        stats[f"{split}/to_translate"] = half
        with open(out / f"pairs_en_{split}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "pairs_stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
