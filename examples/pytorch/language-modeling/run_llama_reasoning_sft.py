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
"""Llama-3.2-1B full-parameter SFT with switchable thinking on a single 16GB GPU.

Same recipe as the Ranunculus-v1-1B SFT (``run_ranunculus_sft.py``: fp32 master weights + bf16
autocast, AdamW with the Ranunculus weight-decay exclusion list, chat delimiters reset to the vocab
mean), with two changes needed to fit 16GB:

* The AdamW state and an fp32 copy of the weights live in pinned CPU memory (``HostOffloadAdamW``).
  Gradients are accumulated and clipped on the GPU; the update runs in the fused CPU AdamW and is
  copied back. The arithmetic is the same as ``torch.optim.AdamW``.
* Logits are never materialized (Liger fused linear cross entropy via ``use_liger_kernel``).

Usage:

    python run_llama_reasoning_sft.py --config configs/llama-3.2-1b-reasoning-sft.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from run_ranunculus_pretrain import RanunculusTrainer, load_yaml
from run_ranunculus_sft import SFTCollator, init_chat_tokens

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback, TrainingArguments


class HostOffloadAdamW(torch.optim.Optimizer):
    """AdamW whose fp32 master weights and moments are kept in pinned CPU memory.

    ``param_groups`` hold the GPU parameters so that the LR scheduler and gradient clipping work as
    usual; ``step`` copies the GPU gradients to the host, runs the fused CPU AdamW, and copies the
    updated weights back.

    Args:
        param_groups: Parameter groups of GPU parameters (with per-group ``weight_decay``).
        lr: Learning rate.
        betas: Adam betas.
        eps: Adam epsilon.
    """

    def __init__(self, param_groups, lr: float, betas: tuple[float, float], eps: float):
        super().__init__(param_groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0})
        self.pairs = []
        host_groups = []
        for g in self.param_groups:
            hps = []
            for p in g["params"]:
                hp = p.detach().to("cpu", copy=True).pin_memory()
                hp.grad = torch.zeros_like(hp).pin_memory()
                hps.append(hp)
                self.pairs.append((p, hp))
            host_groups.append({**{k: v for k, v in g.items() if k != "params"}, "params": hps})
        self.host = torch.optim.AdamW(host_groups, lr=lr, betas=betas, eps=eps, fused=True)

    def _sync_hparams(self) -> None:
        for g, hg in zip(self.param_groups, self.host.param_groups):
            for k in ("lr", "betas", "eps", "weight_decay"):
                hg[k] = g[k]

    @torch.no_grad()
    def step(self, closure=None):
        self._sync_hparams()
        for p, hp in self.pairs:
            hp.grad.copy_(p.grad, non_blocking=True)
        torch.cuda.synchronize()
        self.host.step()
        for p, hp in self.pairs:
            p.copy_(hp, non_blocking=True)
        torch.cuda.synchronize()

    def state_dict(self):
        return {"host": self.host.state_dict(), "param_groups": [{k: v for k, v in g.items() if k != "params"}
                                                                 for g in self.param_groups]}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        self.host.load_state_dict(state_dict["host"])
        for g, saved in zip(self.param_groups, state_dict["param_groups"]):
            g.update(saved)
        # The model weights are restored from the checkpoint before the optimizer state; refresh the
        # host master copy from them.
        for p, hp in self.pairs:
            hp.copy_(p.detach().cpu())


class MomentOffloadAdamW(torch.optim.Optimizer):
    """AdamW that keeps only the moments in pinned CPU memory and updates the fp32 weights on the GPU.

    For hosts whose RAM cannot hold an fp32 master copy as well (Llama-3.2-3B: moments 26GB vs. 51GB
    for ``HostOffloadAdamW``). Each step streams a parameter's ``exp_avg`` / ``exp_avg_sq`` to the GPU,
    applies the ``torch.optim.AdamW`` update (decoupled weight decay, bias correction) and copies the
    moments back.

    Args:
        param_groups: Parameter groups of GPU fp32 parameters (with per-group ``weight_decay``).
        lr: Learning rate.
        betas: Adam betas.
        eps: Adam epsilon.
    """

    def __init__(self, param_groups, lr: float, betas: tuple[float, float], eps: float):
        super().__init__(param_groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0})
        for g in self.param_groups:
            for p in g["params"]:
                self._init_state(p)

    def _init_state(self, p) -> None:
        st = self.state[p]
        st["step"] = torch.zeros((), dtype=torch.float32)
        st["exp_avg"] = torch.zeros(p.shape, dtype=torch.float32, device="cpu").pin_memory()
        st["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device="cpu").pin_memory()

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                t = float(st["step"])
                m = st["exp_avg"].to(p.device, non_blocking=True)
                v = st["exp_avg_sq"].to(p.device, non_blocking=True)
                grad = p.grad
                p.mul_(1 - lr * wd)
                m.lerp_(grad, 1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                bc1, bc2 = 1 - b1**t, 1 - b2**t
                denom = (v.sqrt() / bc2**0.5).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
                st["exp_avg"].copy_(m, non_blocking=True)
                st["exp_avg_sq"].copy_(v, non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # The base class moves the moments to the parameter's device; put them back in pinned host memory.
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].to("cpu").pin_memory()
                st["step"] = st["step"].to("cpu")


OPTIMIZERS = {"host_offload": HostOffloadAdamW, "moment_offload": MomentOffloadAdamW}


class OffloadTrainer(RanunculusTrainer):
    """RanunculusTrainer with the AdamW state offloaded to the host.

    ``optimizer_cls`` picks the offload scheme (``HostOffloadAdamW`` unless the config says otherwise).
    """

    optimizer_cls = HostOffloadAdamW

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Same exclusion list as RanunculusTrainer: bias / RMSNorm.weight / embed_tokens.weight.
            if param.dim() < 2 or "norm.weight" in name or "embed_tokens.weight" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        args = self.args
        self.optimizer = self.optimizer_cls(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Liger skips the logits only in training mode by default; a full 8,192 x 128,256 fp32 logit
        # tensor does not fit next to the weights in 16GB, so skip them in evaluation as well.
        if not model.training:
            inputs = {**inputs, "skip_logits": True}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


class PeakMemoryCallback(TrainerCallback):
    """Add the peak GPU memory since the previous log to the logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()


class SaveAtEpochEndCallback(TrainerCallback):
    """Save the weights at the end of each epoch (except the last) for the 1 vs 2 epoch comparison."""

    def __init__(self, out_dir: str, tokenizer):
        self.out_dir = out_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if state.is_world_process_zero and epoch < args.num_train_epochs:
            path = f"{self.out_dir}/epoch{epoch}"
            model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"[sft] saved {path} at step {state.global_step}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="trial run: stop after this many steps")
    cli_args = parser.parse_args()

    cfg = load_yaml(cli_args.config)
    OffloadTrainer.optimizer_cls = OPTIMIZERS[cfg.get("optimizer", "host_offload")]
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    tokenizer.chat_template = Path(cfg["chat_template"]).read_text()
    tokenizer.pad_token = cfg["pad_token"]

    model = AutoModelForCausalLM.from_pretrained(
        cfg["init_from"], dtype=torch.float32, attn_implementation=cfg.get("attn_implementation", "sdpa")
    )
    model.config.use_cache = False
    assert all(p.dtype == torch.float32 for p in model.parameters()), "master weights must be fp32"
    assert model.lm_head.weight is model.get_input_embeddings().weight, "embeddings must be tied"
    if cli_args.resume is None:
        before = init_chat_tokens(model, tokenizer, cfg["reset_tokens"], cfg["n_regular_vocab"])
        print(f"[sft] reset chat token rows to vocab mean; row norms before: {before}", flush=True)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    extra = cfg["data"].get("drop_columns", ["domain"])
    train_ds = datasets.load_from_disk(cfg["data"]["train"]).remove_columns(extra + ["mode"])
    val = datasets.load_from_disk(cfg["data"]["val"]).remove_columns(extra)
    # Validation loss per thinking mode (eval_reasoning_loss / eval_no_reasoning_loss).
    eval_ds = {m: val.filter(lambda x, m=m: x["mode"] == m).remove_columns(["mode"]) for m in ("reasoning", "no_reasoning")}

    training = dict(cfg["training"])
    if cli_args.max_steps:
        training.update(max_steps=cli_args.max_steps, save_strategy="no", eval_steps=cli_args.max_steps, logging_steps=1)
    args = TrainingArguments(**training)
    trainer = OffloadTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[PeakMemoryCallback(), SaveAtEpochEndCallback(cfg["final_dir"], tokenizer)],
    )
    trainer.train(resume_from_checkpoint=cli_args.resume)
    if cli_args.max_steps:
        return

    final = cfg["final_dir"]
    model.config.use_cache = True
    trainer.save_model(final)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        gen = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=[eot, tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        gen.save_pretrained(final)
        print(f"[sft] saved {final}", flush=True)


if __name__ == "__main__":
    main()
