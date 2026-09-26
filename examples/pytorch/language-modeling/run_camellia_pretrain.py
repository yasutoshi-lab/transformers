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
"""Camellia pretraining entrypoint (design §5 / §6 / §7).

Usage (single GPU, M3 tiny smoke):

    CUDA_VISIBLE_DEVICES=0 python run_camellia_pretrain.py --config configs/camellia-tiny.yaml

Usage (DDP 2 GPU, M4 smoke / M5 full):

    torchrun --nproc_per_node=2 run_camellia_pretrain.py --config configs/camellia-3b.yaml

The YAML holds: model hyper-parameters, packed .bin paths per domain, Muon /
AdamW optimizer settings, MoE / MTP hooks, and the `TrainingArguments` dict.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from camellia_callbacks import (
    LayerTypeGradNormCallback,
    MoEBiasAndMonitorCallback,
    MtpLambdaCallback,
    PerDomainEvalCallback,
    TokensPerSecMfuCallback,
)
from camellia_muon import build_camellia_optimizer
from packed_dataset import PackedDataset
from transformers import AutoTokenizer, CamelliaConfig, CamelliaForCausalLM, Trainer, TrainingArguments


class CamelliaTrainer(Trainer):
    """Trainer override: Muon/AdamW hybrid optimizer (design §5)."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        muon_cfg = self.camellia_muon_cfg
        self.optimizer = build_camellia_optimizer(
            self.model,
            muon_lr=muon_cfg.get("muon_lr", 0.02),
            adamw_lr=muon_cfg.get("adamw_lr", 3.0e-4),
            weight_decay=muon_cfg.get("weight_decay", 0.1),
            betas=(muon_cfg.get("adam_beta1", 0.9), muon_cfg.get("adam_beta2", 0.95)),
            epsilon=muon_cfg.get("adam_epsilon", 1.0e-8),
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

    is_ddp = "LOCAL_RANK" in os.environ
    if cli_args.gpu is not None and not is_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu

    cfg = load_yaml(cli_args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])

    model_cfg = CamelliaConfig(**cfg["model"])
    # Packed intra-document boundary marker (design §7 実装前確定事項 1)
    model_cfg.eos_token_id = tokenizer.convert_tokens_to_ids("\x08")
    if cfg.get("attn_implementation"):
        model_cfg._attn_implementation = cfg["attn_implementation"]
    model = CamelliaForCausalLM(model_cfg).to(torch.bfloat16)
    # Kernel-contract fp32 parameters (M0 finding, fla-core 0.5.2):
    # A_log / dt_bias stay fp32, e_score_correction_bias stays fp32.
    for name, p in model.named_parameters():
        if "A_log" in name or "dt_bias" in name:
            p.data = p.data.float()
    for name, b in model.named_buffers():
        if "e_score_correction_bias" in name:
            b.data = b.data.float()
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if cfg.get("torch_compile", False):
        model = torch.compile(model)

    seq_len = cfg["data"]["seq_len"]
    num_epochs = cfg["data"].get("num_epochs", 1)
    train_ds = PackedDataset(cfg["data"]["train_bins"], seq_len=seq_len, num_epochs=num_epochs)
    val_ds_by_domain = {
        domain: PackedDataset([path], seq_len=seq_len) for domain, path in cfg["data"].get("val_bins", {}).items()
    }

    training = dict(cfg["training"])
    training.setdefault("ddp_find_unused_parameters", False)
    args = TrainingArguments(**training)

    muon_cfg = cfg.get("muon", {})
    trainer = CamelliaTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[
            MoEBiasAndMonitorCallback(
                bias_update_speed=cfg.get("moe", {}).get("bias_update_speed", 0.001),
                monitor_every=cfg.get("moe", {}).get("monitor_every", 100),
            ),
            MtpLambdaCallback(),
            PerDomainEvalCallback(
                val_ds_by_domain,
                every=cfg.get("per_domain_eval_every", 500),
                batch_size=cfg.get("per_domain_eval_batch_size", 1),
                max_batches=cfg.get("per_domain_eval_max_batches", 64),
            ),
            LayerTypeGradNormCallback(every=cfg.get("layer_grad_norm_every", 10)),
            TokensPerSecMfuCallback(
                every=cfg.get("throughput_every", 100),
                n_active=cfg.get("mfu", {}).get("n_active", 0.76e9),
                peak_tflops_per_gpu=cfg.get("mfu", {}).get("peak_tflops_per_gpu", 250.0),
            ),
        ],
    )
    trainer.camellia_muon_cfg = muon_cfg
    trainer.train(resume_from_checkpoint=cli_args.resume or cfg.get("resume"))
    trainer.save_model(cfg.get("final_dir", "artifacts-1/models/camellia"))
    trainer.save_state()


if __name__ == "__main__":
    main()
