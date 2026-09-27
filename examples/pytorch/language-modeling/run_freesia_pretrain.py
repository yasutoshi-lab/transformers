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
"""Freesia Janus pretraining (design §2.3, stages 1a and 2).

1 つのスクリプトで 3 アームを扱う。
* `objective: lm`     … causal LM のみ（全ステップ LM モード）
* `objective: masked` … 穴埋め（MNTP）のみ（全ステップ穴埋めモード・双方向）
* `objective: janus`  … LM と穴埋めを `janus_masked_ratio` の割合で交互に

途中停止に備え、`save_minutes` ごとに重み・最適化器・ステップ・乱数状態をまとめて保存し、
同じ `output_dir` で再実行すると最新のチェックポイントから続きを学習する（設計書 §7.6）。
データは日英の uint16 トークン列を 1:1 で交互に読み、ステップ番号から読み出し位置が決まる（再開しても同じ順序）。

Usage:
    python run_freesia_pretrain.py --config configs/freesia-100m-janus.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from transformers import AutoTokenizer, FreesiaConfig, FreesiaForPreTraining
from transformers.models.freesia.modeling_freesia import MODE_LM, MODE_MASKED


class TokenStream:
    """Deterministic batch reader over flat uint16 token files (one per language).

    ステップ s・マイクロバッチ m の読み出し位置は (s, m) だけで決まるので、再開しても同じデータ順になる。

    Args:
        paths (dict[str, Path]): Language -> .bin path.
        seq_len (int): Sequence length (tokens per sample).
    """

    def __init__(self, paths: dict[str, Path], seq_len: int):
        self.seq_len = seq_len
        self.data = {k: np.memmap(v, dtype=np.uint16, mode="r") for k, v in paths.items()}
        self.langs = sorted(self.data)
        self.n_seq = {k: len(v) // seq_len for k, v in self.data.items()}

    def batch(self, index: int, batch_size: int) -> torch.Tensor:
        """Return `batch_size` sequences; languages alternate per sequence.

        Args:
            index (int): Global micro-batch index.
            batch_size (int): Sequences per micro-batch.

        Returns:
            torch.Tensor: `(batch_size, seq_len)` int64 token ids.
        """
        rows = []
        for j in range(batch_size):
            g = index * batch_size + j
            lang = self.langs[g % len(self.langs)]
            k = (g // len(self.langs)) % self.n_seq[lang]
            s = k * self.seq_len
            rows.append(np.asarray(self.data[lang][s : s + self.seq_len], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))


def mask_for_mntp(ids: torch.Tensor, rate: float, mask_id: int, vocab: int, gen: torch.Generator, special_max: int):
    """Build MNTP inputs/labels (BERT-style 80/10/10 replacement).

    Args:
        ids (torch.Tensor): `(batch, seq)` original tokens.
        rate (float): Fraction of positions to predict.
        mask_id (int): `<|mask|>` id.
        vocab (int): Vocabulary size.
        gen (torch.Generator): RNG (deterministic per step).
        special_max (int): Ids below this are special tokens and are never selected.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(inputs, labels)`; labels are -100 except at selected positions.
    """
    probs = torch.rand(ids.shape, generator=gen)
    select = (probs < rate) & (ids >= special_max)
    select[:, 0] = False  # position 0 has no previous position to predict it from (MNTP shift)
    labels = torch.where(select, ids, torch.full_like(ids, -100))
    inputs = ids.clone()
    r = torch.rand(ids.shape, generator=gen)
    inputs[select & (r < 0.8)] = mask_id
    rand_pos = select & (r >= 0.8) & (r < 0.9)
    inputs[rand_pos] = torch.randint(special_max, vocab, (int(rand_pos.sum()),), generator=gen)
    return inputs, labels


def lr_at(step: int, cfg: dict) -> float:
    warm, total = cfg["warmup_steps"], cfg["max_steps"]
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def mode_for_step(step: int, cfg: dict) -> int:
    obj = cfg["objective"]
    if obj == "lm":
        return MODE_LM
    if obj == "masked":
        return MODE_MASKED
    # janus: deterministic interleave with the requested masked ratio
    ratio = cfg.get("janus_masked_ratio", 0.5)
    return MODE_MASKED if math.floor((step + 1) * ratio) > math.floor(step * ratio) else MODE_LM


def save_checkpoint(out: Path, model, opt, step: int, tokens: int, cfg: dict) -> None:
    ckpt = out / "checkpoint-latest"
    tmp = out / "checkpoint-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True)
    torch.save(
        {"optimizer": opt.state_dict(), "step": step, "tokens": tokens, "torch_rng": torch.get_rng_state(),
         "cuda_rng": torch.cuda.get_rng_state_all(), "py_rng": random.getstate(), "np_rng": np.random.get_state()},
        tmp / "trainer_state.pt",
    )
    (tmp / "progress.json").write_text(json.dumps({"step": step, "tokens": tokens, "config": cfg}, indent=2))
    if ckpt.exists():
        old = out / "checkpoint-old"
        if old.exists():
            import shutil

            shutil.rmtree(old)
        ckpt.rename(old)
    tmp.rename(ckpt)


@torch.no_grad()
def evaluate(model, val: TokenStream, cfg: dict, tok_mask_id: int, special_max: int) -> dict:
    model.eval()
    out = {}
    gen = torch.Generator().manual_seed(1234)
    for lang in val.langs:
        single = TokenStream({lang: Path(cfg["val_files"][lang])}, cfg["seq_len"])
        for mode, name in ((MODE_LM, "lm"), (MODE_MASKED, "masked")):
            losses = []
            for i in range(cfg.get("eval_batches", 8)):
                ids = single.batch(i, cfg["micro_batch_size"])
                if mode == MODE_MASKED:
                    inputs, labels = mask_for_mntp(ids, cfg["mask_rate"], tok_mask_id, model.config.vocab_size, gen, special_max)
                else:
                    inputs, labels = ids, ids
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(inputs.cuda(), labels=labels.cuda(), mode=mode).loss
                losses.append(loss.item())
            out[f"val_{name}_loss_{lang}"] = float(np.mean(losses))
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    special_max = 16  # the first 16 ids are special tokens (train_freesia_tokenizer.py)
    ckpt = out / "checkpoint-latest"
    if ckpt.exists():
        model = FreesiaForPreTraining.from_pretrained(ckpt, torch_dtype=torch.float32)
    else:
        mcfg = FreesiaConfig(
            vocab_size=len(tok), pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id, mask_token_id=tok.mask_token_id, **cfg["model"],
        )
        torch.manual_seed(cfg.get("seed", 42))
        model = FreesiaForPreTraining(mcfg)
    model.cuda().train()
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.dim() < 2 or "norm" in name or "embed" in name or "bloom" in name else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    step, tokens = 0, 0
    if ckpt.exists():
        st = torch.load(ckpt / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        step, tokens = st["step"], st["tokens"]
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        print(f"resumed from step={step} tokens={tokens:,}", flush=True)

    train = TokenStream({k: Path(v) for k, v in cfg["train_files"].items()}, cfg["seq_len"])
    val = TokenStream({k: Path(v) for k, v in cfg["val_files"].items()}, cfg["seq_len"])
    accum, mbs = cfg["grad_accum"], cfg["micro_batch_size"]
    log = open(out / "train_log.jsonl", "a")
    last_save, t_window, tok_window = time.time(), time.time(), 0

    while step < cfg["max_steps"]:
        mode = mode_for_step(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        gen = torch.Generator().manual_seed(cfg.get("seed", 42) * 1_000_003 + step)
        loss_sum = 0.0
        for m in range(accum):
            ids = train.batch(step * accum + m, mbs)
            if mode == MODE_MASKED:
                inputs, labels = mask_for_mntp(ids, cfg["mask_rate"], tok.mask_token_id, model.config.vocab_size, gen, special_max)
            else:
                inputs, labels = ids, ids
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs.cuda(non_blocking=True), labels=labels.cuda(non_blocking=True), mode=mode).loss / accum
            loss.backward()
            loss_sum += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += accum * mbs * cfg["seq_len"]
        tok_window += accum * mbs * cfg["seq_len"]

        if step % cfg.get("log_every", 10) == 0:
            dt = time.time() - t_window
            rec = {"step": step, "tokens": tokens, "mode": "masked" if mode == MODE_MASKED else "lm",
                   "loss": loss_sum, "grad_norm": float(gnorm), "lr": lr_at(step - 1, cfg), "tok_per_s": tok_window / dt,
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9, "time": time.strftime("%F %T")}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_window, tok_window = time.time(), 0
        if step % cfg.get("eval_every", 500) == 0 or step == cfg["max_steps"]:
            rec = {"step": step, "tokens": tokens, **evaluate(model, val, cfg, tok.mask_token_id, special_max)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if time.time() - last_save > cfg.get("save_minutes", 30) * 60 or step == cfg["max_steps"]:
            save_checkpoint(out, model, opt, step, tokens, cfg)
            last_save = time.time()
            print(f"saved checkpoint at step={step}", flush=True)

    model.save_pretrained(out / "final", safe_serialization=True)
    tok.save_pretrained(out / "final")
    (out / "DONE").write_text(json.dumps({"step": step, "tokens": tokens}))
    print("TRAINING_FINISHED", flush=True)


if __name__ == "__main__":
    main()
