#!/usr/bin/env python
"""Export the on-policy thinking DPO data as a text dataset (for publishing).

One row per prompt: the conversation, the 4 sampled thinking-mode responses with their
judge scores and reasons, and the chosen / rejected pair actually used for DPO (or null
if the prompt yielded no pair). The ``split`` of used pairs follows prepare_onpolicy_dpo.py.

Usage:
    python build_onpolicy_dataset_text.py --gen artifacts-6/onpolicy/gen_0.jsonl artifacts-6/onpolicy/gen_1.jsonl \
        --scored artifacts-6/onpolicy/scored.jsonl --data_dir artifacts-6/onpolicy-data --out_dir artifacts-6/onpolicy-dataset-text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    gen = {json.loads(l)["id"]: json.loads(l) for f in args.gen for l in open(f, encoding="utf-8")}
    scored = {json.loads(l)["id"]: json.loads(l) for l in open(args.scored, encoding="utf-8")}
    split_of, pair_of = {}, {}
    for sp in ("train", "validation"):
        for r in datasets.load_from_disk(str(Path(args.data_dir) / sp)):
            split_of[r["id"]] = sp
    out = {"train": [], "validation": [], "unused": []}
    for i, g in gen.items():
        sc = scored.get(i)
        samples = []
        for k, s in enumerate(g["samples"]):
            v = sc["verdicts"][k] if sc else None
            samples.append({"text": s["text"], "closed_think": s["closed_think"], "stopped": s["stopped"],
                            "n_tokens": s["n_tokens"], "score": v["score"] if v else None,
                            "reason": v.get("reason") if v else None, "auto_zero": bool(v and v.get("auto"))})
        chosen = rejected = None
        if i in split_of and sc:
            s = sc["scores"]
            hi = max(range(len(s)), key=lambda k: s[k])
            lo = min(range(len(s)), key=lambda k: s[k])
            chosen, rejected = hi, lo
        row = {"id": i, "lang": g["lang"], "prompt": g["prompt"], "samples": samples,
               "chosen_index": chosen, "rejected_index": rejected,
               "chosen": samples[chosen]["text"] if chosen is not None else None,
               "rejected": samples[rejected]["text"] if rejected is not None else None}
        out[split_of.get(i, "unused")].append(row)
    d = Path(args.out_dir)
    datasets.DatasetDict({k: datasets.Dataset.from_list(v) for k, v in out.items()}).save_to_disk(str(d))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
