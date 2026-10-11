#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generation-quality evaluation for Ranunculus base checkpoints.

Two parts:

1. **Continuation**: take documents from the packed validation streams (never
   trained on), prompt with ``<|endoftext|>`` + the first ``prompt_len`` tokens
   of each document and generate ``gen_len`` tokens (greedy and sampling). The
   real next ``gen_len`` tokens of the document are kept as a reference so the
   same metrics can be computed on human text.
2. **Short prompts**: a handful of hand-written prompts, with and without a
   leading ``<|endoftext|>``, to inspect early-EOS behaviour (probability of EOS
   as the first generated token, and the greedy output).

Training data were packed as ``doc + <|endoftext|>`` with no BOS, so a document
start is always preceded by EOS in training; prompting without it is slightly
out of distribution.

Writes ``generations.jsonl`` (one record per prompt x mode) and
``short_prompts.jsonl`` under ``--out``.

Usage:

    CUDA_VISIBLE_DEVICES=0 python gen_eval_ranunculus.py --model <path> --out <dir> \
        --val en=artifacts-0/packed/val_en.bin ja=artifacts-0/packed/val_ja.bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer, RanunculusForCausalLM


SHORT_PROMPTS = {
    "ja": ["人工知能とは", "日本の首都は", "富士山は", "東京大学は", "第二次世界大戦は",
           "光合成とは", "織田信長は", "量子力学は", "京都は", "日本語は"],
    "en": ["Artificial intelligence is", "The capital of France is", "Mount Everest is",
           "The University of Oxford is", "World War II was", "Photosynthesis is",
           "Albert Einstein was", "Quantum mechanics is", "London is", "The English language is"],
}


def find_documents(arr: np.ndarray, eos_id: int, need: int, n: int, seed: int) -> list[np.ndarray]:
    """Return ``n`` documents (token arrays without EOS) that are at least ``need`` tokens long."""
    eos_pos = np.flatnonzero(arr == eos_id)
    starts = np.concatenate([[0], eos_pos[:-1] + 1])
    ends = eos_pos
    # The first segment of the stream may be a truncated document; skip it.
    cand = [(s, e) for s, e in zip(starts[1:], ends[1:]) if e - s >= need]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return [np.asarray(arr[cand[i][0] : cand[i][1]], dtype=np.int64) for i in sorted(idx)]


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, gen_len: int, eos_id: int, pad_id: int, sample: bool, seed: int):
    if sample:
        torch.manual_seed(seed)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=gen_len,
        do_sample=sample,
        temperature=0.8 if sample else None,
        top_p=0.95 if sample else None,
        top_k=0 if sample else None,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
    )
    return out[:, input_ids.shape[1] :].tolist()


def strip_after_eos(ids: list[int], eos_id: int, pad_id: int) -> tuple[list[int], bool]:
    if eos_id in ids:
        return ids[: ids.index(eos_id)], True
    return [t for t in ids if t != pad_id], False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val", nargs="+", required=True, help="lang=path pairs")
    parser.add_argument("--n-docs", type=int, default=200)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    pad_id = tok.pad_token_id
    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.eval()

    # ---- 1. continuation ----
    with (args.out / "generations.jsonl").open("w", encoding="utf-8") as f:
        for pair in args.val:
            lang, path = pair.split("=", 1)
            arr = np.memmap(path, dtype=np.uint32, mode="r")
            docs = find_documents(arr, eos_id, args.prompt_len + args.gen_len, args.n_docs, args.seed)
            for b in range(0, len(docs), args.batch_size):
                chunk = docs[b : b + args.batch_size]
                prompts = torch.tensor(
                    np.stack([np.concatenate([[eos_id], d[: args.prompt_len]]) for d in chunk]), device="cuda"
                )
                outs = {
                    "greedy": generate(model, prompts, args.gen_len, eos_id, pad_id, False, args.seed),
                    "sample": generate(model, prompts, args.gen_len, eos_id, pad_id, True, args.seed + b),
                }
                for i, d in enumerate(chunk):
                    ref = d[args.prompt_len : args.prompt_len + args.gen_len].tolist()
                    base = {
                        "lang": lang,
                        "doc_index": b + i,
                        "prompt_ids": d[: args.prompt_len].tolist(),
                        "prompt_text": tok.decode(d[: args.prompt_len].tolist()),
                    }
                    f.write(json.dumps({**base, "mode": "reference", "ids": ref, "eos": False,
                                        "text": tok.decode(ref)}, ensure_ascii=False) + "\n")
                    for mode, o in outs.items():
                        ids, hit_eos = strip_after_eos(o[i], eos_id, pad_id)
                        f.write(json.dumps({**base, "mode": mode, "ids": ids, "eos": hit_eos,
                                            "text": tok.decode(ids)}, ensure_ascii=False) + "\n")
                print(f"[gen] {lang} {b + len(chunk)}/{len(docs)}", flush=True)

    # ---- 2. short prompts (EOS behaviour) ----
    with (args.out / "short_prompts.jsonl").open("w", encoding="utf-8") as f:
        for lang, prompts in SHORT_PROMPTS.items():
            for p in prompts:
                for lead in (False, True):
                    ids = tok(p, add_special_tokens=False)["input_ids"]
                    if lead:
                        ids = [eos_id, *ids]
                    x = torch.tensor([ids], device="cuda")
                    with torch.no_grad():
                        probs = model(input_ids=x).logits[0, -1].float().softmax(-1)
                    p_eos = probs[eos_id].item()
                    rank_eos = int((probs > probs[eos_id]).sum().item()) + 1
                    o = generate(model, x, 64, eos_id, pad_id, False, args.seed)[0]
                    gids, hit = strip_after_eos(o, eos_id, pad_id)
                    f.write(json.dumps({"lang": lang, "prompt": p, "leading_eos": lead, "p_eos_first": p_eos,
                                        "eos_rank_first": rank_eos, "greedy_len": len(gids), "greedy_eos": hit,
                                        "greedy_text": tok.decode(gids)}, ensure_ascii=False) + "\n")
    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
