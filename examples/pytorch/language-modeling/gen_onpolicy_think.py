#!/usr/bin/env python
"""Sample several thinking-mode responses per prompt from the policy (on-policy DPO, stage 2).

Prompts come from the ranunculus-dpo training split (quality_ok rows; the same prompts
as stage 1, so no new data source). Each prompt is rendered with the v2 chat template
in thinking mode (generation starts right after ``<|im_start|>assistant\\n``) and the
model samples ``--n`` responses. Output keeps the raw generated text, including the
``<think>`` block, and whether ``</think>`` and ``<|im_end|>`` were produced.

Usage:
    python gen_onpolicy_think.py --model artifacts-6/models/ranunculus-v1-1b-dpo --dataset_dir artifacts-5/dpo-dataset \
        --shard 0 --num_shards 2 --out artifacts-6/onpolicy/gen_0.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

import datasets
import torch

from transformers import AutoTokenizer, RanunculusForCausalLM


def select_prompts(dataset_dir: str, per_lang: int, seed: int) -> list[dict]:
    ds = datasets.load_from_disk(dataset_dir)["train"]
    rows = [r for r in ds if r["quality_ok"]]
    rng = random.Random(seed)
    out = []
    for lang in ("en", "ja"):
        cand = [r for r in rows if r["lang"] == lang]
        rng.shuffle(cand)
        out += [{"id": r["id"], "lang": lang, "prompt": r["prompt"]} for r in cand[:per_lang]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_lang", type=int, default=600)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompts_per_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    prompts = select_prompts(args.dataset_dir, args.per_lang, args.seed)[args.shard :: args.num_shards]
    items = []
    for p in prompts:
        text = tok.apply_chat_template(p["prompt"], add_generation_prompt=True, tokenize=False)  # thinking mode (default)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= args.max_prompt_tokens:
            items.append({**p, "prompt_text": text, "prompt_ids": ids})
    items.sort(key=lambda x: len(x["prompt_ids"]))
    torch.manual_seed(args.seed + args.shard)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(items), args.prompts_per_batch):
            chunk = items[s : s + args.prompts_per_batch]
            n = max(len(x["prompt_ids"]) for x in chunk)
            ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, x in enumerate(chunk):
                ids[i, n - len(x["prompt_ids"]):] = torch.tensor(x["prompt_ids"])
                mask[i, n - len(x["prompt_ids"]):] = 1
            with torch.no_grad():
                out = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), do_sample=True, temperature=0.8,
                                     top_p=0.95, num_return_sequences=args.n, max_new_tokens=args.max_new_tokens,
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id)
            gen = out[:, n:].tolist()
            for i, x in enumerate(chunk):
                samples = []
                for o in gen[i * args.n : (i + 1) * args.n]:
                    stopped = any(t in eos for t in o)
                    if stopped:
                        o = o[: min(o.index(t) for t in eos if t in o)]
                    text = tok.decode(o, skip_special_tokens=False)
                    samples.append({"text": text, "ids": o, "stopped": stopped, "closed_think": "</think>" in text, "n_tokens": len(o)})
                f.write(json.dumps({"id": x["id"], "lang": x["lang"], "prompt": x["prompt"], "prompt_text": x["prompt_text"],
                                    "prompt_ids": x["prompt_ids"], "samples": samples}, ensure_ascii=False) + "\n")
            print(f"{min(s + args.prompts_per_batch, len(items))}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
