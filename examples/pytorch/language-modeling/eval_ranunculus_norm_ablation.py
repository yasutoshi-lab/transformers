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
"""RMSNorm diagnostic: how much does a trained model rely on its learned norm gains?

For each variant, a group of ``*norm.weight`` tensors is reset to 1.0 and the
full per-language validation loss is measured (same protocol as
eval_ranunculus_val.py: fp32 weights + bf16 autocast, every 8,192-token
sequence).

Variants:

- ``original``: no change.
- ``fold_in_post``: fold ``input_layernorm`` into q/k/v_proj and
  ``post_attention_layernorm`` into gate/up_proj, then set those gains to 1.0.
  This is mathematically the same function, so the loss must match
  ``original`` up to numerical noise (sanity check of the "absorbable" claim).
- ``reset_in_post``: reset input_layernorm + post_attention_layernorm to 1.0
  (absorbable gains; the loss change shows the co-adaptation of other weights).
- ``reset_qk``: reset q_norm + k_norm (not absorbable because RoPE follows).
- ``reset_final``: reset model.norm (not absorbable because embeddings are tied).
- ``reset_qk_final``: reset q_norm + k_norm + model.norm.
- ``reset_all``: reset every ``*norm.weight``.

Usage:

    CUDA_VISIBLE_DEVICES=0 python eval_ranunculus_norm_ablation.py --model <path> \
        --val en=artifacts-0/packed/val_en.bin ja=artifacts-0/packed/val_ja.bin
"""

from __future__ import annotations

import argparse
import json

import torch
from eval_ranunculus_val import eval_loss
from packed_dataset import PackedDataset

from transformers import RanunculusForCausalLM


GROUPS = {
    "in_post": ("input_layernorm.weight", "post_attention_layernorm.weight"),
    "qk": ("q_norm.weight", "k_norm.weight"),
    "final": ("model.norm.weight",),
}
VARIANTS = {
    "original": (),
    "reset_in_post": ("in_post",),
    "reset_qk": ("qk",),
    "reset_final": ("final",),
    "reset_qk_final": ("qk", "final"),
    "reset_all": ("in_post", "qk", "final"),
}


def matches(name: str, groups) -> bool:
    for g in groups:
        for suffix in GROUPS[g]:
            if name == suffix or name.endswith("." + suffix) or (suffix == "model.norm.weight" and name == suffix):
                return True
    return False


@torch.no_grad()
def fold_in_post(model) -> None:
    for layer in model.model.layers:
        g = layer.input_layernorm.weight.data.clone()
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.weight.data.mul_(g[None, :])
        layer.input_layernorm.weight.data.fill_(1.0)
        g = layer.post_attention_layernorm.weight.data.clone()
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
            proj.weight.data.mul_(g[None, :])
        layer.post_attention_layernorm.weight.data.fill_(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    vals = {p.split("=", 1)[0]: PackedDataset([p.split("=", 1)[1]], seq_len=8192) for p in args.val}

    def load():
        return RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()

    result = {"model": args.model}
    model = load()
    original = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("norm.weight")}
    counts = {g: sum(1 for n in original if matches(n, (g,))) for g in GROUPS}
    result["n_tensors"] = counts
    for variant, groups in VARIANTS.items():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in original:
                    p.copy_(original[n])
                    if matches(n, groups):
                        p.fill_(1.0)
        result[variant] = {lang: eval_loss(model, ds, args.batch_size)[0] for lang, ds in vals.items()}
        print(f"[norm-ablation] {variant} {result[variant]}", flush=True)
    del model
    torch.cuda.empty_cache()
    model = load()
    fold_in_post(model)
    result["fold_in_post"] = {lang: eval_loss(model, ds, args.batch_size)[0] for lang, ds in vals.items()}
    print(f"[norm-ablation] fold_in_post {result['fold_in_post']}", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
