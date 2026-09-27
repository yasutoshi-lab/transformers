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
"""Measure Freesia training throughput and memory on random tokens (stage 0 smoke test, design §7.2).

段階0 の smoke test。乱数トークンで 100M / 300M の事前学習（LM・穴埋め）と、埋め込みモード（Bloom 学習・
per-head マスク）の順伝播・逆伝播の速度とメモリを測る。

Usage:
    python smoke_freesia_throughput.py --out ~/freesia-work/logs/throughput.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from transformers import FreesiaConfig, FreesiaForPreTraining
from transformers.models.freesia.modeling_freesia import MODE_LM, MODE_MASKED

SIZES = {
    "100m": dict(num_hidden_layers=8),
    "300m": dict(num_hidden_layers=28),
}


def bench_pretrain(size: str, mode: int, mbs: int, steps: int = 12, seq: int = 1024) -> dict:
    cfg = FreesiaConfig(vocab_size=32768, **SIZES[size])
    model = FreesiaForPreTraining(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        ids = torch.randint(16, 32768, (mbs, seq), device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, labels=ids, mode=mode).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t)
    tps = mbs * seq / (sum(times) / len(times))
    res = {"size": size, "mode": "lm" if mode == MODE_LM else "masked", "micro_batch": mbs,
           "tok_per_s": round(tps), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "params": sum(p.numel() for p in model.parameters())}
    del model, opt
    torch.cuda.empty_cache()
    return res


def bench_embed(size: str, chunk: int, seq: int = 512, steps: int = 8) -> dict:
    cfg = FreesiaConfig(vocab_size=32768, **SIZES[size])
    model = FreesiaForPreTraining(cfg).model.cuda().train()
    model.gradient_checkpointing_enable()
    model.bloom_override, model.bloom_floor = "learned", 0.1
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        ids = torch.randint(16, 32768, (chunk, seq), device="cuda")
        att = torch.ones_like(ids)
        att[:, seq // 2 :] = 0  # half padding to exercise the per-head float mask path
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.encode(ids, att)
        e.sum().backward()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t)
    res = {"size": size, "mode": "embed(bloom learned, padded)", "chunk": chunk, "seq": seq,
           "tok_per_s_fwd_bwd": round(chunk * seq / (sum(times) / len(times))),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del model
    torch.cuda.empty_cache()
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = []
    for size, mbs in (("100m", 16), ("300m", 8)):
        for mode in (MODE_LM, MODE_MASKED):
            r = bench_pretrain(size, mode, mbs)
            print(json.dumps(r), flush=True)
            results.append(r)
        r = bench_embed(size, chunk=64)
        print(json.dumps(r), flush=True)
        results.append(r)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
