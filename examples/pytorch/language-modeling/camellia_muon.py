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
"""Muon / AdamW hybrid optimizer for Camellia PT (design §5).

Muon (Keller et al.; nanochat reference implementation ported):
SGD-momentum + Nesterov, then Newton-Schulz orthogonalization of the 2D
update (quintic iteration, bf16), aspect-ratio scaled step size, decoupled
weight decay.

Muon Split (GLM-5 insight, design §1): attention projection matrices are
orthogonalized per head — the weight is viewed as `[heads, *, *]` and the NS
iteration runs batched over heads — instead of as one flat 2D matrix.

AdamW handles embedding / norms / router / scalar parameters.

A single optimizer exposes one param group per (kind, shape) so the standard
HF Trainer LR scheduler (cosine with min-lr) scales every group's `lr` from
its own initial value.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz zeroth-power (orthogonalization) of G, quintic iteration.

    Runs in bf16; supports batched (3D) inputs so Muon Split orthogonalizes
    per-head slices — and NaiveMoe per-expert weights — in one kernel launch.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


if os.environ.get("CAMELLIA_MUON_COMPILE", "1") == "1":
    _zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5)


def _muon_split_view(p: Tensor, split_heads: int | None) -> tuple[Tensor, tuple[int, ...]]:
    """View a 2D parameter as a batched (per-head) matrix for Muon Split.

    Args:
        p: 2D weight tensor.
        split_heads: If set, `p.size(0)` must be divisible by it and the weight
            is viewed as `[split_heads, p0/split_heads, p1]`.

    Returns:
        (reshaped view, original shape).
    """
    if split_heads is None:
        return p, p.shape
    assert p.size(0) % split_heads == 0, f"Muon Split: {p.shape} not divisible by {split_heads}"
    return p.view(split_heads, p.size(0) // split_heads, p.size(1)), p.shape


class CamelliaMuonAdamW(Optimizer):
    """Hybrid optimizer: Muon for 2D/3D matrices, AdamW for the rest.

    Args:
        muon_params: 2D/3D matrix parameters (attention projections, expert
            FFNs, dense FFNs). 3D parameters (NaiveMoe `gate_up_proj` /
            `down_proj`) are orthogonalized per expert directly.
        adamw_params: scalar / 1D / embedding parameters (embedding, norms,
            router, A_log, dt_bias, e_score_correction_bias).
        muon_lr: Muon (internal SGD) learning rate. Design initial 0.02, M4 tune.
        adamw_lr: AdamW learning rate (design 3.0e-4).
        momentum: Muon momentum (0.95).
        nesterov: Nesterov-style momentum (recommended).
        ns_steps: Newton-Schulz iterations (5).
        weight_decay: Decoupled weight decay applied to both parameter kinds.
        betas / epsilon: AdamW hyper-parameters.
        split_heads: Mapping id(param) -> number of heads for Muon Split.
        no_wd_ids: Parameter ids (AdamW group only) exempt from weight decay,
            per design §5 `weight_decay_exclude` (bias / norm / embedding /
            router).
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        muon_lr: float = 0.02,
        adamw_lr: float = 3.0e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        epsilon: float = 1.0e-8,
        split_heads: dict[int, int] | None = None,
        no_wd_ids: set[int] | None = None,
    ):
        split_heads = split_heads or {}
        no_wd_ids = no_wd_ids or set()
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)

        muon_defaults = dict(
            lr=muon_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            use_muon=True,
        )

        # One group per unique shape so same-shape matrices share the loop.
        groups: list[dict] = []
        by_shape: dict[tuple[int, ...], list[Tensor]] = {}
        for p in muon_params:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        for shape, params in by_shape.items():
            groups.append(dict(params=params, **muon_defaults))
        for p in adamw_params:
            # Design §5 weight_decay_exclude = [bias, norm.weight,
            # embed_tokens.weight, router] — those AdamW params get wd = 0.
            wd = 0.0 if id(p) in no_wd_ids else weight_decay
            groups.append(dict(params=[p], lr=adamw_lr, betas=betas, eps=epsilon, weight_decay=wd, use_muon=False))

        super().__init__(groups, defaults={})
        self._split_heads = split_heads

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon"):
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    split = self._split_heads.get(id(p))
                    gm, orig_shape = _muon_split_view(g, split)
                    U = _zeropower_via_newtonschulz5(gm, steps=group["ns_steps"]).view(orig_shape)
                    scale = max(1, p.size(-2) // p.size(-1)) ** 0.5 if p.ndim >= 2 else 1.0
                    p.add_(U, alpha=-group["lr"] * scale)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] / bias_correction1
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

        return loss


def build_camellia_optimizer(
    model,
    muon_lr: float = 0.02,
    adamw_lr: float = 3.0e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    epsilon: float = 1.0e-8,
) -> CamelliaMuonAdamW:
    """Partition `model` parameters into Muon (matrices) and AdamW (rest) groups.

    Muon Split heads (per-head orthogonalization) are derived structurally:
      * MLA attention (`CamelliaMLAAttention`): q_proj / kv_b_proj / o_proj
        split by `num_attention_heads` (kv_b_proj rows are per-head
        (nope+v) blocks; o_proj rows are per-head v blocks).
      * KDA attention (`CamelliaKDAAttention`): q/k/v/o_proj and f_b/g_b
        split by `kda_num_heads`.

    Args:
        model: A `CamelliaForCausalLM`.
        muon_lr / adamw_lr / weight_decay / betas / epsilon: see `CamelliaMuonAdamW`.

    Returns:
        The hybrid optimizer.
    """
    from transformers.models.camellia.modeling_camellia import CamelliaKDAAttention, CamelliaMLAAttention

    cfg = model.config
    muon: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    split_heads: dict[int, int] = {}
    no_wd_ids: set[int] = set()
    seen_ids: set[int] = set()

    def take(name: str, p: torch.nn.Parameter, split: int | None = None):
        if not p.requires_grad:
            return
        # Tied embedding: `lm_head.weight` is the same object as
        # `model.embed_tokens.weight` (tie_word_embeddings=True). Without this
        # guard the embedding would be stepped twice per optimizer step.
        if id(p) in seen_ids:
            return
        seen_ids.add(id(p))
        if p.ndim < 2:
            adamw.append(p)
        elif "embed_tokens" in name or "lm_head" in name:
            adamw.append(p)
        elif any(tag in name for tag in ("norm", "A_log", "dt_bias", "e_score_correction_bias")):
            adamw.append(p)
        elif "gate.weight" in name:  # router: keep off Muon (bias-update interference, design §5)
            adamw.append(p)
        else:
            muon.append(p)
            if split is not None:
                split_heads[id(p)] = split
        # Design §5: weight_decay_exclude = [bias, norm.weight,
        # embed_tokens.weight, router].
        if any(tag in name for tag in ("norm", "embed_tokens", "lm_head", "dt_bias")) or "gate.weight" in name:
            no_wd_ids.add(id(p))

    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn, CamelliaMLAAttention):
            for name, p in attn.named_parameters():
                if (
                    name.endswith("q_proj.weight")
                    or name.endswith("kv_b_proj.weight")
                    or name.endswith("o_proj.weight")
                ):
                    take(f"layers.self_attn.{name}", p, split=cfg.num_attention_heads)
                else:
                    take(f"layers.self_attn.{name}", p)
        elif isinstance(attn, CamelliaKDAAttention):
            for name, p in attn.named_parameters():
                if name.endswith(
                    (
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "o_proj.weight",
                        "f_b_proj.weight",
                        "g_b_proj.weight",
                    )
                ):
                    take(f"layers.self_attn.{name}", p, split=cfg.kda_num_heads)
                else:
                    take(f"layers.self_attn.{name}", p)
        for name, p in layer.mlp.named_parameters():
            take(f"layers.mlp.{name}", p)  # NaiveMoe 3D params: batched per expert
        for name, p in layer.input_layernorm.named_parameters():
            take(f"layers.input_layernorm.{name}", p)
        for name, p in layer.post_attention_layernorm.named_parameters():
            take(f"layers.post_attention_layernorm.{name}", p)

    # Prefix the module path so the string-based Muon/AdamW / weight-decay
    # rules in `take` can see the parameter's identity. `named_parameters()`
    # alone yields just "weight", which would misroute the 2D embedding into
    # the Muon group (design §5: embedding -> AdamW).
    for name, p in model.model.embed_tokens.named_parameters():
        take(f"embed_tokens.{name}", p)
    for name, p in model.lm_head.named_parameters():
        take(f"lm_head.{name}", p)
    for name, p in model.model.norm.named_parameters():
        take(f"model.norm.{name}", p)
    if getattr(model, "mtp", None) is not None:
        for name, p in model.mtp.named_parameters():
            if "self_attn" in name:
                if name.endswith(("q_proj.weight", "kv_b_proj.weight", "o_proj.weight")):
                    take(f"mtp.{name}", p, split=cfg.num_attention_heads)
                else:
                    take(f"mtp.{name}", p)
            else:
                take(f"mtp.{name}", p)

    return CamelliaMuonAdamW(
        muon,
        adamw,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        betas=betas,
        epsilon=epsilon,
        split_heads=split_heads,
        no_wd_ids=no_wd_ids,
    )
