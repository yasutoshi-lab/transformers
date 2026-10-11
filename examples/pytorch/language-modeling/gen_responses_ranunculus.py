#!/usr/bin/env python
"""Generate one response per prompt (batched, left-padded) for pairwise comparison.

Prompts are the tokenized DPO validation prompts (chat template, enable_thinking=False),
so both models see exactly the same input. Greedy decoding with repetition_penalty 1.1.

Usage:
    python gen_responses_ranunculus.py --model artifacts-6/models/ranunculus-v1-1b-dpo \
        --data artifacts-6/dpo-data/validation --out artifacts-6/pairwise/dpo.jsonl
"""

from __future__ import annotations

import argparse
import json

import datasets
import torch

from transformers import AutoTokenizer, RanunculusForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--thinking", action="store_true", help="re-render prompts in thinking mode from --dataset_dir")
    ap.add_argument("--dataset_dir", default="artifacts-5/dpo-dataset")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    rows = list(datasets.load_from_disk(args.data))
    if args.thinking:
        msgs = {r["id"]: r["prompt"] for r in datasets.load_from_disk(args.dataset_dir)[args.split]}
        for r in rows:
            text = tok.apply_chat_template(msgs[r["id"]], add_generation_prompt=True, tokenize=False)  # thinking mode
            r["prompt_ids"] = tok(text, add_special_tokens=False)["input_ids"]
    rows.sort(key=lambda r: len(r["prompt_ids"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(rows), args.batch_size):
            chunk = rows[s : s + args.batch_size]
            n = max(len(r["prompt_ids"]) for r in chunk)
            ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, r in enumerate(chunk):
                ids[i, n - len(r["prompt_ids"]):] = torch.tensor(r["prompt_ids"])
                mask[i, n - len(r["prompt_ids"]):] = 1
            with torch.no_grad():
                out = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=args.max_new_tokens,
                                     do_sample=False, repetition_penalty=1.1, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for r, o in zip(chunk, out[:, n:].tolist()):
                stopped = any(t in eos for t in o)
                if stopped:
                    o = o[: min(o.index(t) for t in eos if t in o)]
                text = tok.decode(o, skip_special_tokens=False) if args.thinking else tok.decode(o, skip_special_tokens=True)
                resp = text.split("</think>", 1)[1].strip() if args.thinking and "</think>" in text else text.strip()
                f.write(json.dumps({"id": r["id"], "lang": r["lang"], "response": resp, "raw": text if args.thinking else None,
                                    "closed_think": ("</think>" in text) if args.thinking else None,
                                    "stopped": stopped, "model": args.model}, ensure_ascii=False) + "\n")
            print(f"{min(s + args.batch_size, len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
