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
"""Evaluate a Ranunculus checkpoint on the full per-language validation sets.

Computes token-weighted CE loss over every sequence in each ``val_<lang>.bin``
(not just the first 64 used during training) under bf16 autocast, plus summary
stats of all RMSNorm weights. Prints one JSON line per model.

Usage:

    CUDA_VISIBLE_DEVICES=0 python eval_ranunculus_val.py \
        --model artifacts-0/checkpoints/ranunculus-1b/checkpoint-40500 \
        --val en=artifacts-0/packed/val_en.bin ja=artifacts-0/packed/val_ja.bin
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from packed_dataset import PackedDataset
from torch.utils.data import DataLoader

from transformers import RanunculusForCausalLM


@torch.no_grad()
def eval_loss(model, dataset, batch_size: int) -> tuple[float, int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, n_tok = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        n = ids.numel()
        total += out.loss.float().item() * n
        n_tok += n
    return total / n_tok, n_tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", nargs="+", required=True, help="lang=path pairs")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.eval()

    norms = torch.cat([p.detach().float().flatten() for n, p in model.named_parameters() if n.endswith("norm.weight")])
    result = {
        "model": args.model,
        "norm_weight": {
            "mean": norms.mean().item(),
            "std": norms.std().item(),
            "min": norms.min().item(),
            "max": norms.max().item(),
            "frac_exactly_1": (norms == 1.0).float().mean().item(),
        },
    }
    for pair in args.val:
        lang, path = pair.split("=", 1)
        ds = PackedDataset([path], seq_len=args.seq_len)
        loss, n_tok = eval_loss(model, ds, args.batch_size)
        result[f"val_loss/{lang}"] = loss
        result[f"ppl/{lang}"] = math.exp(loss)
        result[f"n_seq/{lang}"] = len(ds)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
