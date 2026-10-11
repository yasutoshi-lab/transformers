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
"""Ranunculus-v1-1B full-parameter SFT (fp32 master weights + bf16 autocast).

Loads the continued-pretraining model in fp32, resets the embedding rows of the
chat delimiters (``<|im_start|>`` / ``<|im_end|>``, never seen in pretraining) to
the mean of the regular vocabulary, and trains on the dataset produced by
``prepare_ranunculus_sft.py`` (assistant-only loss). The pretraining validation
sets are evaluated periodically to monitor forgetting.

Usage (DDP 2 GPU):

    NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 run_ranunculus_sft.py \
        --config configs/ranunculus-v1-1b-sft.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasets
import torch
from packed_dataset import PackedDataset
from run_ranunculus_continue import WandbPerLanguageEvalCallback
from run_ranunculus_pretrain import RanunculusTrainer, load_yaml

from transformers import AutoTokenizer, GenerationConfig, RanunculusForCausalLM, TrainingArguments


class SFTCollator:
    """Right-pad ``input_ids`` / ``labels`` to the longest sequence in the batch.

    Args:
        pad_id: Token id used for padding ``input_ids``.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def init_chat_tokens(model, tokenizer, tokens: list[str], n_regular: int) -> dict[str, float]:
    """Set the embedding rows of ``tokens`` to the mean of the regular vocabulary.

    Args:
        model: Model whose input embeddings (tied to the LM head) are edited.
        tokenizer: Tokenizer used to look up the token ids.
        tokens: Special tokens to initialize.
        n_regular: Number of regular (non-special) vocabulary entries at the front.

    Returns:
        L2 norm of each row before the reset, for the work log.
    """
    emb = model.get_input_embeddings().weight
    before = {}
    with torch.no_grad():
        mean = emb[:n_regular].mean(dim=0)
        for t in tokens:
            i = tokenizer.convert_tokens_to_ids(t)
            before[t] = float(emb[i].norm())
            emb[i] = mean
    return before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    cli_args = parser.parse_args()

    cfg = load_yaml(cli_args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    tokenizer.chat_template = Path(cfg["chat_template"]).read_text()

    model = RanunculusForCausalLM.from_pretrained(
        cfg["init_from"], dtype=torch.float32, attn_implementation=cfg.get("attn_implementation", "sdpa")
    )
    model.config.use_cache = False
    assert all(p.dtype == torch.float32 for p in model.parameters()), "master weights must be fp32"
    if cli_args.resume is None:
        tokens = cfg.get("reset_tokens", ["<|im_start|>", "<|im_end|>"])
        before = init_chat_tokens(model, tokenizer, tokens, cfg["n_regular_vocab"])
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[sft] reset chat token rows to vocab mean; row norms before: {before}", flush=True)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    extra = cfg["data"].get("drop_columns", ["lang"])
    train_ds = datasets.load_from_disk(cfg["data"]["train"]).remove_columns(extra)
    eval_ds = datasets.load_from_disk(cfg["data"]["val"]).remove_columns(extra)
    pt_val = {lang: PackedDataset([p], seq_len=cfg["pt_val"]["seq_len"]) for lang, p in cfg["pt_val"]["bins"].items()}

    args = TrainingArguments(**cfg["training"])
    trainer = RanunculusTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[
            WandbPerLanguageEvalCallback(
                pt_val, every=cfg["pt_val"]["every"], batch_size=1, max_batches=cfg["pt_val"]["max_batches"]
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=cli_args.resume)

    final = cfg["final_dir"]
    trainer.save_model(final)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen = GenerationConfig(
            eos_token_id=[im_end, tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id, do_sample=False
        )
        gen.save_pretrained(final)
        print(f"[sft] saved {final}", flush=True)


if __name__ == "__main__":
    main()
