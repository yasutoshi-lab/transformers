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
"""Ranunculus-v1-1B continued pretraining with fp32 master weights.

The original PT run cast the model with ``.to(torch.bfloat16)`` so there were no
fp32 master weights, and every RMSNorm weight stayed at exactly 1.0 (updates of
~lr were below bf16's rounding threshold at 1.0). This entrypoint loads an
existing checkpoint in **fp32**, builds a **fresh** AdamW (the saved optimizer
state is bf16 and is not reused), and lets ``TrainingArguments(bf16=True)``
handle mixed precision via autocast.

Usage (DDP 2 GPU):

    NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 run_ranunculus_continue.py \
        --config configs/ranunculus-v1-1b-fp32-continue.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from packed_dataset import PackedDataset
from ranunculus_callbacks import ParamNormCallback, PerLanguageEvalCallback, TokensPerSecCallback
from run_ranunculus_pretrain import RanunculusTrainer, load_yaml

from transformers import AutoTokenizer, RanunculusForCausalLM, TrainerCallback, TrainingArguments


class WandbPerLanguageEvalCallback(PerLanguageEvalCallback):
    """PerLanguageEvalCallback that also pushes ``val_loss/*`` to wandb.

    The base class only injects metrics when ``logs`` is passed to
    ``on_step_end``, which Trainer never does, so the original run kept val loss
    only in stdout.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        model = kwargs["model"]
        metrics = {f"val_loss/{lang}": self._eval_one(model, ds) for lang, ds in self.datasets_by_lang.items()}
        print(f"[step {state.global_step}] {metrics}", flush=True)
        if state.is_world_process_zero:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({**metrics, "train/global_step": state.global_step}, commit=False)
            except ImportError:
                pass


class StaleParamCallback(TrainerCallback):
    """Report parameters whose values did not change since the previous check.

    Keeps an fp32 CPU snapshot of every trainable parameter and, every ``every``
    steps, prints how many tensors are bit-identical to the snapshot. Also prints
    summary stats of all ``*norm.weight`` tensors so the RMSNorm fix is visible.

    Args:
        every: Check interval in global steps.
    """

    def __init__(self, every: int = 100):
        self.every = every
        self._snapshot: dict[str, torch.Tensor] | None = None

    def _take(self, model) -> dict[str, torch.Tensor]:
        return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    def on_train_begin(self, args, state, control, **kwargs):
        self._snapshot = self._take(kwargs["model"])

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0 or not state.is_world_process_zero:
            return
        model = kwargs["model"]
        current = self._take(model)
        stale = [n for n, t in current.items() if torch.equal(t, self._snapshot[n])]
        norm_vals = torch.cat([t.flatten() for n, t in current.items() if n.endswith("norm.weight")])
        print(
            f"[step {state.global_step}] stale_params={len(stale)}/{len(current)} "
            f"norm_weight mean={norm_vals.mean():.6f} std={norm_vals.std():.6f} "
            f"min={norm_vals.min():.6f} max={norm_vals.max():.6f}"
            + (f" stale_examples={stale[:5]}" if stale else ""),
            flush=True,
        )
        self._snapshot = current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    cli_args = parser.parse_args()

    cfg = load_yaml(cli_args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])

    # fp32 で読み込む（.to(torch.bfloat16) はしない）。混合精度は bf16=True の autocast に任せる。
    model = RanunculusForCausalLM.from_pretrained(
        cfg["init_from"], dtype=torch.float32, attn_implementation=cfg.get("attn_implementation", "sdpa")
    )
    model.config.use_cache = False
    assert all(p.dtype == torch.float32 for p in model.parameters()), "master weights must be fp32"
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    seq_len = cfg["data"]["seq_len"]
    train_ds = PackedDataset(cfg["data"]["train_bins"], seq_len=seq_len, num_epochs=cfg["data"].get("num_epochs", 1))
    val_ds_by_lang = {lang: PackedDataset([path], seq_len=seq_len) for lang, path in cfg["data"]["val_bins"].items()}

    args = TrainingArguments(**cfg["training"])

    callbacks = [
        WandbPerLanguageEvalCallback(
            val_ds_by_lang,
            every=cfg.get("per_lang_eval_every", 500),
            batch_size=cfg.get("per_lang_eval_batch_size", 1),
            max_batches=cfg.get("per_lang_eval_max_batches", 64),
        ),
        ParamNormCallback(every=cfg.get("param_norm_every", 1000)),
        TokensPerSecCallback(every=cfg.get("tokens_per_sec_every", 100)),
        StaleParamCallback(every=cfg.get("stale_param_every", 100)),
    ]

    trainer = RanunculusTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[continue] init_from={cfg['init_from']} dtype={next(model.parameters()).dtype}", flush=True)
    trainer.train(resume_from_checkpoint=cli_args.resume)
    trainer.save_model(cfg["final_dir"])


if __name__ == "__main__":
    main()
