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
"""Ranunculus-v1-1B DPO (fp32 master weights + bf16 autocast, DDP).

Standard DPO (Rafailov et al., 2023, arXiv:2305.18290) with the sigmoid loss:
``-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`` where each term is the
sum of response-token log-probabilities. The reference model is the SFT model;
its log-probabilities are computed once before training (``--precompute_ref``)
and stored with the data, so only the policy is kept in memory.

Usage (DDP 2 GPU):

    NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 run_ranunculus_dpo.py --config configs/ranunculus-v1-1b-dpo.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from run_ranunculus_continue import WandbPerLanguageEvalCallback
from packed_dataset import PackedDataset
from run_ranunculus_pretrain import RanunculusTrainer, load_yaml

from transformers import AutoTokenizer, RanunculusForCausalLM, TrainingArguments


def build_batch(rows: list[dict], key: str, pad_id: int) -> dict[str, torch.Tensor]:
    """Concatenate prompt + response for ``key`` (chosen / rejected) and right-pad.

    Labels are -100 on the prompt and padding, so only response tokens are scored.
    """
    seqs = [r["prompt_ids"] + r[f"{key}_ids"] for r in rows]
    n = max(len(s) for s in seqs)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), n), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (r, s) in enumerate(zip(rows, seqs)):
        ids[i, : len(s)] = torch.tensor(s)
        p = len(r["prompt_ids"])
        labels[i, p : len(s)] = torch.tensor(r[f"{key}_ids"])
        mask[i, : len(s)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class DPOCollator:
    """Stack chosen and rejected sequences into one padded batch (chosen first)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        both = [{"prompt_ids": r["prompt_ids"], "x_ids": r["chosen_ids"]} for r in rows] + [
            {"prompt_ids": r["prompt_ids"], "x_ids": r["rejected_ids"]} for r in rows
        ]
        batch = build_batch(both, "x", self.pad_id)
        batch["ref_chosen_logp"] = torch.tensor([r["ref_chosen_logp"] for r in rows], dtype=torch.float32)
        batch["ref_rejected_logp"] = torch.tensor([r["ref_rejected_logp"] for r in rows], dtype=torch.float32)
        return batch


def sequence_logp(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Sum of log-probabilities of the label tokens for each sequence."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    lp = torch.gather(F.log_softmax(logits, dim=-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


class DPOTrainer(RanunculusTrainer):
    """RanunculusTrainer (weight-decay exclusions) with the DPO loss and DPO metrics."""

    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.model_accepts_loss_kwargs = False
        self._acc = {"n": 0, "acc": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0}

    def _dpo(self, model, inputs):
        logp = sequence_logp(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        b = logp.shape[0] // 2
        rc = self.beta * (logp[:b] - inputs["ref_chosen_logp"])
        rr = self.beta * (logp[b:] - inputs["ref_rejected_logp"])
        loss = -F.logsigmoid(rc - rr).mean()
        return loss, rc.detach(), rr.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, rc, rr = self._dpo(model, inputs)
        if model.training:
            a = self._acc
            a["n"] += rc.numel()
            a["acc"] += (rc > rr).float().sum().item()
            a["margin"] += (rc - rr).sum().item()
            a["chosen"] += rc.sum().item()
            a["rejected"] += rr.sum().item()
        return (loss, None) if return_outputs else loss

    def log(self, logs, start_time=None):
        a = self._acc
        if a["n"] and "loss" in logs:
            logs.update({"rewards/accuracy": a["acc"] / a["n"], "rewards/margin": a["margin"] / a["n"],
                         "rewards/chosen": a["chosen"] / a["n"], "rewards/rejected": a["rejected"] / a["n"]})
            self._acc = {k: 0 if k == "n" else 0.0 for k in a}
        super().log(logs, start_time)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.model
        model.eval()
        tot = {"n": 0, "loss": 0.0, "acc": 0.0, "margin": 0.0}
        for inputs in self.get_eval_dataloader(eval_dataset):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss, rc, rr = self._dpo(model, inputs)
            k = rc.numel()
            tot["n"] += k
            tot["loss"] += loss.item() * k
            tot["acc"] += (rc > rr).float().sum().item()
            tot["margin"] += (rc - rr).sum().item()
        t = torch.tensor([tot["n"], tot["loss"], tot["acc"], tot["margin"]], device=self.args.device, dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(t)
        n = t[0].item()
        metrics = {f"{metric_key_prefix}_loss": t[1].item() / n, f"{metric_key_prefix}_rewards/accuracy": t[2].item() / n,
                   f"{metric_key_prefix}_rewards/margin": t[3].item() / n, f"{metric_key_prefix}_pairs": n}
        self.log(metrics)
        model.train()
        return metrics


@torch.no_grad()
def precompute_reference(cfg, tokenizer) -> None:
    """Compute the SFT (reference) log-probs for every pair and save them with the data (rank 0 only)."""
    model = RanunculusForCausalLM.from_pretrained(cfg["init_from"], dtype=torch.bfloat16).cuda().eval()
    pad = tokenizer.pad_token_id
    for split, path in (("train", cfg["data"]["train"]), ("validation", cfg["data"]["val"])):
        ds = datasets.load_from_disk(path)
        rows = list(ds)
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["prompt_ids"]) - len(rows[i]["chosen_ids"]))
        out_c, out_r = [0.0] * len(rows), [0.0] * len(rows)
        bs = cfg.get("ref_batch_size", 8)
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            chunk = [rows[i] for i in idx]
            for key, out in (("chosen", out_c), ("rejected", out_r)):
                b = {k: v.cuda() for k, v in build_batch(chunk, key, pad).items()}
                lp = sequence_logp(model, **b)
                for i, v in zip(idx, lp.tolist()):
                    out[i] = v
        ds = ds.add_column("ref_chosen_logp", out_c).add_column("ref_rejected_logp", out_r)
        ds.save_to_disk(path + "-ref")
        print(f"[dpo] reference log-probs for {split}: {len(rows)} pairs -> {path}-ref", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precompute_ref", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    cli = parser.parse_args()
    cfg = load_yaml(cli.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["init_from"])
    if cli.precompute_ref:
        precompute_reference(cfg, tokenizer)
        return

    model = RanunculusForCausalLM.from_pretrained(cfg["init_from"], dtype=torch.float32,
                                                  attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.config.use_cache = False
    assert all(p.dtype == torch.float32 for p in model.parameters()), "master weights must be fp32"
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    cols = ["id", "lang"]
    train_ds = datasets.load_from_disk(cfg["data"]["train"] + "-ref").remove_columns(cols)
    eval_ds = datasets.load_from_disk(cfg["data"]["val"] + "-ref").remove_columns(cols)
    pt_val = {lang: PackedDataset([p], seq_len=cfg["pt_val"]["seq_len"]) for lang, p in cfg["pt_val"]["bins"].items()}
    args = TrainingArguments(**cfg["training"])
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DPOCollator(tokenizer.pad_token_id), processing_class=tokenizer, beta=cfg["beta"],
        callbacks=[WandbPerLanguageEvalCallback(pt_val, every=cfg["pt_val"]["every"], batch_size=1,
                                                max_batches=cfg["pt_val"]["max_batches"])],
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[dpo] init_from={cfg['init_from']} beta={cfg['beta']} train_pairs={len(train_ds)}", flush=True)
    trainer.evaluate()
    trainer.train(resume_from_checkpoint=cli.resume)
    trainer.evaluate()
    final = cfg["final_dir"]
    trainer.save_model(final)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        import shutil
        shutil.copy(Path(cfg["init_from"]) / "generation_config.json", Path(final) / "generation_config.json")
        print(f"[dpo] saved {final}", flush=True)


if __name__ == "__main__":
    main()
