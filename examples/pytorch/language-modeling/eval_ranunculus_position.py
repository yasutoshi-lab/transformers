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
"""Position-wise validation loss for Ranunculus checkpoints.

Runs every 8,192-token sequence of each ``val_<lang>.bin`` and buckets the
per-token cross-entropy two ways:

- ``frame``: absolute position inside the 8,192-token frame.
- ``doc``: number of tokens since the last ``<|endoftext|>`` (i.e. how much of
  the *same document* the model can see). Packed frames concatenate several
  documents without attention masking, so this is the in-document context
  length; tokens before the first EOS of a frame are counted from the frame
  start (their true document position is unknown) and excluded from ``doc``.

Prints a JSON object with token counts and mean loss per bucket.

Usage:

    CUDA_VISIBLE_DEVICES=0 python eval_ranunculus_position.py --model <path> \
        --val en=artifacts-0/packed/val_en.bin ja=artifacts-0/packed/val_ja.bin
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from packed_dataset import PackedDataset

from transformers import AutoTokenizer, RanunculusForCausalLM


BUCKETS = [0, 128, 512, 1024, 2048, 4096, 8192]


def bucket_of(pos: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(pos, torch.tensor(BUCKETS[1:], device=pos.device), right=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.eval()
    nb = len(BUCKETS) - 1
    result = {"model": args.model, "buckets": [f"{BUCKETS[i]}-{BUCKETS[i + 1]}" for i in range(nb)]}

    for pair in args.val:
        lang, path = pair.split("=", 1)
        ds = PackedDataset([path], seq_len=args.seq_len)
        sums = {k: torch.zeros(nb, dtype=torch.float64, device="cuda") for k in ("frame", "doc")}
        cnts = {k: torch.zeros(nb, dtype=torch.float64, device="cuda") for k in ("frame", "doc")}
        for s in range(0, len(ds), args.batch_size):
            ids = torch.stack([ds[i]["input_ids"] for i in range(s, min(s + args.batch_size, len(ds)))]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids).logits
            # loss for predicting token t+1 from prefix ending at t; context length = t+1
            loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")
            B, T = loss.shape
            frame_pos = torch.arange(1, T + 1, device="cuda").expand(B, T)
            # in-document context: tokens since the last EOS at or before position t
            is_eos = ids[:, :-1] == eos_id
            idx = torch.arange(T, device="cuda").expand(B, T)
            last_eos = torch.where(is_eos, idx, torch.full_like(idx, -1)).cummax(dim=1).values
            doc_pos = idx - last_eos  # 1 right after EOS
            valid_doc = last_eos >= 0
            # do not score predicting EOS itself or the token right after it as "in-document"
            for key, pos, mask in (("frame", frame_pos, torch.ones_like(valid_doc)), ("doc", doc_pos, valid_doc)):
                b = bucket_of(pos)
                for k in range(nb):
                    m = mask & (b == k)
                    sums[key][k] += loss[m].double().sum()
                    cnts[key][k] += m.sum()
        for key in ("frame", "doc"):
            result[f"{lang}/{key}/loss"] = (sums[key] / cnts[key].clamp(min=1)).tolist()
            result[f"{lang}/{key}/tokens"] = cnts[key].tolist()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
