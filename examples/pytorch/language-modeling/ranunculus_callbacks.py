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
"""Custom Trainer callbacks for Ranunculus PT (design §10 monitoring)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PerLanguageEvalCallback(TrainerCallback):
    """Run a forward-only eval on per-language validation sets and log `val_loss/{lang}`.

    Args:
        datasets_by_lang: `{"en": dataset, "de": dataset, ...}` mapping.
        every: How often (in global steps) to evaluate.
        batch_size: Micro batch size for eval. Should divide the GPU budget.
        max_batches: If set, cap each language to this many batches for speed.
    """

    def __init__(self, datasets_by_lang: dict, every: int = 500, batch_size: int = 1, max_batches: int | None = 64):
        self.datasets_by_lang = datasets_by_lang
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
        metrics = {f"val_loss/{lang}": self._eval_one(model, ds) for lang, ds in self.datasets_by_lang.items()}
        logger.info("step %d: %s", state.global_step, metrics)
        print(f"[step {state.global_step}] {metrics}", flush=True)
        if "logs" in kwargs and isinstance(kwargs["logs"], dict):
            kwargs["logs"].update(metrics)


class ParamNormCallback(TrainerCallback):
    """Log per-layer parameter L2 norm every `every` steps."""

    def __init__(self, every: int = 1000):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        model = kwargs["model"]
        norms = {}
        for name, p in model.named_parameters():
            # One entry per top-level transformer block keeps the log compact.
            parts = name.split(".")
            if "layers" in parts:
                idx = parts.index("layers")
                key = ".".join(parts[: idx + 2])
            else:
                key = name
            norms[f"param_norm/{key}"] = (
                norms.get(f"param_norm/{key}", 0.0) + float(p.detach().float().norm().item()) ** 2
            )
        norms = {k: v**0.5 for k, v in norms.items()}
        logger.info("step %d param_norms: %s", state.global_step, norms)
        print(f"[step {state.global_step}] param_norms: {norms}", flush=True)


class TokensPerSecCallback(TrainerCallback):
    """Log training throughput (tokens/sec) averaged over windows of `every` steps."""

    def __init__(self, every: int = 100):
        self.every = every
        self._window_start: float | None = None
        self._window_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Each optimizer step sees micro_bs * grad_accum sequences of length seq_len.
        seq_len = getattr(kwargs.get("train_dataloader"), "dataset", None)
        seq_len = getattr(seq_len, "seq_len", None) or args.max_seq_length or 8192
        tokens = args.per_device_train_batch_size * args.gradient_accumulation_steps * seq_len
        self._window_tokens += tokens

        if self._window_start is None:
            self._window_start = time.time()
            return

        if state.global_step % self.every != 0:
            return

        elapsed = time.time() - self._window_start
        tps = self._window_tokens / max(elapsed, 1e-9)
        logger.info("step %d: tokens_per_sec=%.1f", state.global_step, tps)
        print(f"[step {state.global_step}] tokens_per_sec={tps:.1f}", flush=True)
        self._window_start = time.time()
        self._window_tokens = 0


__all__: Iterable[str] = ("PerLanguageEvalCallback", "ParamNormCallback", "TokensPerSecCallback")
