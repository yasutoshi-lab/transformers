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
"""Custom Trainer callbacks for Camellia PT (design §6 monitoring + §7 hooks).

* MoEBiasAndMonitorCallback: auxiliary-loss-free sign-based router bias
  update (global whole-batch counts, design §7 実装前確定事項 2) plus the
  §6 MoE health panel (load distribution / MaxVio / dead experts / expert
  output norms / router bias range).
* MtpLambdaCallback: MTP loss weight schedule (0.3 -> 0.1 at
  `mtp_lambda_switch_frac` of total steps, design §1 MTP).
* PerDomainEvalCallback: forward-only per-domain validation loss.
* LayerTypeGradNormCallback: grad norm split by KDA / MLA layers (design §6).
* TokensPerSecMfuCallback: throughput + MFU estimate (M4 fallback gate).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import TrainerCallback


logger = logging.getLogger(__name__)


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class MoEBiasAndMonitorCallback(TrainerCallback):
    """Sign-based router bias update (every optimizer step) + MoE health panel.

    The bias update follows DeepSeek-V3 auxiliary-loss-free balancing:
    per-expert token counts are all-reduced across DDP ranks so the whole
    global batch drives the over/under decision (each rank then applies the
    identical update to its replicated `e_score_correction_bias`).

    Args:
        bias_update_speed: gamma of the sign update (design: 0.001).
        monitor_every: Steps between MoE health panel logs.
    """

    def __init__(self, bias_update_speed: float = 0.001, monitor_every: int = 100):
        self.gamma = bias_update_speed
        self.monitor_every = monitor_every
        self._window_counts: torch.Tensor | None = None
        self._window_norms: torch.Tensor | None = None
        self._window_hits: torch.Tensor | None = None

    def _moe_model(self, kwargs):
        model = kwargs["model"]
        # DDP / accelerator wrappers expose the raw model
        while hasattr(model, "module"):
            model = model.module
        if not hasattr(model, "moe_layers"):
            return None
        return model

    def on_step_end(self, args, state, control, **kwargs):
        model = self._moe_model(kwargs)
        if model is None or state.global_step == 0:
            return

        # --- global whole-batch counts for this optimizer step ---
        counts = model.moe_load_stats.detach().clone()  # [L, E] local
        _all_reduce_sum(counts)
        total_tokens = counts.sum().clamp_min(1.0)

        # --- sign-based bias update (V3 aux-loss-free) ---
        with torch.no_grad():
            for i, layer in enumerate(model.moe_layers):
                bias = layer.mlp.gate.e_score_correction_bias
                load = counts[i] / total_tokens[i]
                avg = 1.0 / bias.numel()
                delta = torch.where(
                    load > avg,
                    torch.tensor(-self.gamma, device=bias.device),
                    torch.tensor(self.gamma, device=bias.device),
                )
                bias.add_(delta)
        model.moe_load_stats.zero_()

        # --- monitoring window accumulation ---
        if self._window_counts is None or self._window_counts.shape[0] != counts.shape[0]:
            self._window_counts = torch.zeros_like(counts)
            self._window_norms = torch.zeros_like(counts)
            self._window_hits = torch.zeros_like(counts)
        self._window_counts.add_(counts)
        norms = model.moe_out_norm_stats.detach().clone()
        hits = model.moe_out_hit_stats.detach().clone()
        _all_reduce_sum(norms)
        _all_reduce_sum(hits)
        self._window_norms.add_(norms)
        self._window_hits.add_(hits)
        model.moe_out_norm_stats.zero_()
        model.moe_out_hit_stats.zero_()

        if state.global_step % self.monitor_every != 0:
            return

        # --- §6 MoE health panel (window averages) ---
        load = self._window_counts / self._window_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [L, E]
        avg_load = 1.0 / load.shape[-1]
        maxvio = float((load - avg_load).abs().max().item())
        dead = int((self._window_counts.sum(dim=0) == 0).sum().item())
        active = float(self._window_counts.sum(dim=0).gt(0).float().mean().item())
        out_rms = self._window_norms / self._window_hits.clamp_min(1.0)
        metrics = {
            "moe/maxvio": maxvio,
            "moe/load_max": float(load.max().item()),
            "moe/load_min": float(load.min().item()),
            "moe/dead_experts": dead,
            "moe/expert_utilization": active,
            "moe/out_rms_max": float(out_rms.max().item()),
            "moe/out_rms_p99": float(out_rms.flatten().kthvalue(int(0.99 * out_rms.numel()) + 1).values.item()),
        }
        bias_abs_max = max(
            float(layer.mlp.gate.e_score_correction_bias.abs().max().item()) for layer in model.moe_layers
        )
        metrics["moe/router_bias_abs_max"] = bias_abs_max

        logger.info("step %d moe: %s", state.global_step, metrics)
        print(f"[step {state.global_step}] moe {metrics}", flush=True)
        if "logs" in kwargs and isinstance(kwargs["logs"], dict):
            kwargs["logs"].update(metrics)
        self._window_counts.zero_()
        self._window_norms.zero_()
        self._window_hits.zero_()


class MtpLambdaCallback(TrainerCallback):
    """Switch the MTP loss weight at `mtp_lambda_switch_frac * max_steps`.

    Design §1 MTP: lambda stays at `mtp_lambda_high` (0.3) until the switch
    step (final ~30% of training, V3 10T/14.8T ratio), then `mtp_lambda_low`
    (0.1). Not a 50/50 midpoint switch.
    """

    def __init__(self):
        self._switch_step: int | None = None
        self._announced = False

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        while hasattr(model, "module"):
            model = model.module
        cfg = model.config
        if cfg.num_nextn_predict_layers > 0 and args.max_steps is not None:
            self._switch_step = int(args.max_steps * cfg.mtp_lambda_switch_frac)
            model.mtp_lambda = cfg.mtp_lambda_high
            logger.info(
                "MTP lambda: %.2f until step %d, then %.2f", cfg.mtp_lambda_high, self._switch_step, cfg.mtp_lambda_low
            )

    def on_step_end(self, args, state, control, **kwargs):
        if self._switch_step is None or state.global_step < self._switch_step:
            return
        model = kwargs["model"]
        while hasattr(model, "module"):
            model = model.module
        if not self._announced:
            logger.info("step %d: MTP lambda -> %.2f", state.global_step, model.config.mtp_lambda_low)
            self._announced = True
        model.mtp_lambda = model.config.mtp_lambda_low


class PerDomainEvalCallback(TrainerCallback):
    """Forward-only eval on per-domain validation sets, logs `val_loss/{domain}`.

    Args:
        datasets_by_domain: `{"en": dataset, "ja": dataset, ...}`.
        every: Evaluation period in global steps.
        batch_size: Micro batch size for eval.
        max_batches: Cap per domain for speed.
    """

    def __init__(self, datasets_by_domain: dict, every: int = 500, batch_size: int = 1, max_batches: int | None = 64):
        self.datasets_by_domain = datasets_by_domain
        self.every = every
        self.batch_size = batch_size
        self.max_batches = max_batches

    @torch.no_grad()
    def _eval_one(self, model, dataset) -> float:
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        was_training = model.training
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for i, batch in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            n = batch["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
        if was_training:
            model.train()
        return total_loss / max(total_tokens, 1)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        model = kwargs["model"]
        metrics = {f"val_loss/{domain}": self._eval_one(model, ds) for domain, ds in self.datasets_by_domain.items()}
        logger.info("step %d: %s", state.global_step, metrics)
        print(f"[step {state.global_step}] {metrics}", flush=True)
        if "logs" in kwargs and isinstance(kwargs["logs"], dict):
            kwargs["logs"].update(metrics)


class LayerTypeGradNormCallback(TrainerCallback):
    """Grad norm split by layer type (KDA vs MLA vs MoE) — design §6.

    Gradients are read on the last micro-batch of each optimizer step
    (after accumulation, before `zero_grad`).
    """

    def __init__(self, every: int = 10):
        self.every = every
        self._substep = 0
        self._pending: dict[str, float] | None = None

    def _norms(self, model) -> dict[str, float]:
        kda_sq, mla_sq, moe_sq = 0.0, 0.0, 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            sq = float(p.grad.detach().float().pow(2).sum().item())
            if ".self_attn." in name:
                # KDA vs MLA: the KDA layer has k_conv1d, MLA has kv_a_proj
                is_mla = "kv_a_proj" in name or "q_norm" in name
                if is_mla:
                    mla_sq += sq
                else:
                    kda_sq += sq
            elif ".mlp." in name:
                moe_sq += sq
        return {"grad_norm/kda_attn": kda_sq**0.5, "grad_norm/mla_attn": mla_sq**0.5, "grad_norm/moe": moe_sq**0.5}

    def on_substep_end(self, args, state, control, **kwargs):
        self._substep += 1
        if self._substep == args.gradient_accumulation_steps:
            model = kwargs["model"]
            while hasattr(model, "module"):
                model = model.module
            self._pending = self._norms(model)

    def on_step_end(self, args, state, control, **kwargs):
        if self._pending is not None and (state.global_step % self.every == 0):
            logger.info("step %d: %s", state.global_step, self._pending)
            print(f"[step {state.global_step}] {self._pending}", flush=True)
            if "logs" in kwargs and isinstance(kwargs["logs"], dict):
                kwargs["logs"].update(self._pending)
        self._pending = None
        self._substep = 0


class TokensPerSecMfuCallback(TrainerCallback):
    """Throughput (tokens/s, all-GPU) and MFU estimate — M4 fallback gate.

    MFU = 6 * N_active * tokens / (peak_flops * n_gpus * time). `n_active`
    and `peak_tflops_per_gpu` come from the YAML (design §2 FLOPs, §5).

    Args:
        every: Window length in steps.
        n_active: Active parameter count (full config ~0.76e9).
        peak_tflops_per_gpu: bf16 dense peak per GPU (RTX PRO 6000 Blackwell).
    """

    def __init__(self, every: int = 100, n_active: float = 0.76e9, peak_tflops_per_gpu: float = 250.0):
        self.every = every
        # YAML float pitfall: unsigned-exponent scalars (3.0e7) parse as str
        # under PyYAML (YAML 1.1) — cast defensively.
        self.n_active = float(n_active)
        self.peak_tflops_per_gpu = float(peak_tflops_per_gpu)
        self._t0: float | None = None
        self._tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        seq_len = getattr(getattr(kwargs.get("train_dataloader"), "dataset", None), "seq_len", None) or 8192
        tokens = args.per_device_train_batch_size * args.gradient_accumulation_steps * seq_len * args.world_size
        self._tokens += tokens
        if self._t0 is None:
            self._t0 = time.time()
            return
        if state.global_step % self.every != 0:
            return
        elapsed = time.time() - self._t0
        tps = self._tokens / max(elapsed, 1e-9)
        flops = 6 * self.n_active * self._tokens
        peak = self.peak_tflops_per_gpu * 1e12 * args.world_size * elapsed
        mfu = flops / max(peak, 1e-9)
        metrics = {"throughput/tokens_per_sec": tps, "mfu": mfu}
        logger.info("step %d: tokens_per_sec=%.0f mfu=%.3f", state.global_step, tps, mfu)
        print(f"[step {state.global_step}] tokens_per_sec={tps:.0f} mfu={mfu:.3f}", flush=True)
        if "logs" in kwargs and isinstance(kwargs["logs"], dict):
            kwargs["logs"].update(metrics)
        self._t0 = time.time()
        self._tokens = 0


__all__: Iterable[str] = (
    "MoEBiasAndMonitorCallback",
    "MtpLambdaCallback",
    "PerDomainEvalCallback",
    "LayerTypeGradNormCallback",
    "TokensPerSecMfuCallback",
)
