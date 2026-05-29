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
"""Ranunculus-1B pretraining entrypoint (design §5 + §10 + §11).

Usage (single GPU):

    CUDA_VISIBLE_DEVICES=0 python run_ranunculus_pretrain.py --config configs/ranunculus-1b.yaml

Usage (DDP 2 GPU):

    torchrun --nproc_per_node=2 run_ranunculus_pretrain.py --config configs/ranunculus-1b.yaml

The YAML file holds everything: model hyperparameters, packed .bin paths per
language, and the `TrainingArguments` dict. Only the PT stage is covered here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from packed_dataset import PackedDataset
from ranunculus_callbacks import (
    ParamNormCallback,
    PerLanguageEvalCallback,
    TokensPerSecCallback,
)

from transformers import (
    AutoTokenizer,
    RanunculusConfig,
    RanunculusForCausalLM,
    Trainer,
    TrainingArguments,
)


class RanunculusTrainer(Trainer):
    """Trainer override that enforces the design §5 weight-decay exclusion list."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # bias / RMSNorm.weight / embed_tokens.weight are excluded from WD.
            if param.dim() < 2 or "norm.weight" in name or "embed_tokens.weight" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        args = self.args
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            fused=torch.cuda.is_available(),
        )
        return self.optimizer


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES override (e.g. '0')")
    cli_args = parser.parse_args()

    # torchrun が LOCAL_RANK を設定している場合は DDP モード。
    # CUDA_VISIBLE_DEVICES を上書きすると torchrun のデバイス割り当てが壊れるため skip する。
    is_ddp = "LOCAL_RANK" in os.environ
    if cli_args.gpu is not None and not is_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu

    cfg = load_yaml(cli_args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])

    model_cfg = RanunculusConfig(**cfg["model"])
    if cfg.get("attn_implementation"):
        model_cfg._attn_implementation = cfg["attn_implementation"]
    model = RanunculusForCausalLM(model_cfg).to(torch.bfloat16)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    seq_len = cfg["data"]["seq_len"]
    num_epochs = cfg["data"].get("num_epochs", 2)
    train_ds = PackedDataset(cfg["data"]["train_bins"], seq_len=seq_len, num_epochs=num_epochs)
    val_ds_by_lang = {lang: PackedDataset([path], seq_len=seq_len) for lang, path in cfg["data"]["val_bins"].items()}

    args = TrainingArguments(**cfg["training"])

    callbacks = [
        PerLanguageEvalCallback(
            val_ds_by_lang,
            every=cfg.get("per_lang_eval_every", 500),
            batch_size=cfg.get("per_lang_eval_batch_size", 1),
            max_batches=cfg.get("per_lang_eval_max_batches", 64),
        ),
        ParamNormCallback(every=cfg.get("param_norm_every", 1000)),
        TokensPerSecCallback(every=cfg.get("tokens_per_sec_every", 100)),
    ]

    trainer = RanunculusTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=cli_args.resume or cfg.get("resume"))
    trainer.save_model(cfg["final_dir"])


if __name__ == "__main__":
    main()
