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
"""PyTorch Camellia model.

Camellia = DeepSeek-V3 style MoE (fine-grained experts + shared expert +
auxiliary-loss-free routing) with a Kimi-Linear style KDA:MLA = 3:1 hybrid
attention layout (NoPE MLA), one MTP head (D=1), trained with a Muon/AdamW
hybrid optimizer. See the pretraining design report for the full spec.

fla-core 0.5.2 API notes (verified on sm_120, see m0_kda_smoke.py):
  * `fused_kda_gate(g, A_log, dt_bias)` expects `g` pre-reshaped to `[..., H, K]`
    and `A_log` / `dt_bias` as fp32 CUDA tensors.
  * `ShortConvolution` with `cu_seqlens` requires the packed convention
    (batch dim = 1, flattened token stream).
"""

import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3Model,
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
)

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    raise ImportError("Please run `pip install -U fla-core`")

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@auto_docstring(checkpoint="yasutoshi-lab/camellia-3b")
@strict
class CamelliaConfig(DeepseekV3Config):
    r"""
    Configuration for Camellia: DeepSeek-V3 style MoE with a KDA:MLA 3:1
    hybrid attention layout (NoPE MLA), one MTP head (D=1).

    KDA / hybrid fields:
    - `linear_attn_ratio`: number of consecutive KDA layers before each MLA
      layer. 3 gives the [KDA, KDA, KDA, MLA] x 7 layout for 28 layers.
    - `kda_num_heads` / `kda_head_dim` / `short_conv_kernel_size`: KDA layer
      geometry (projection size = kda_num_heads * kda_head_dim).

    MTP fields:
    - `num_nextn_predict_layers`: MTP depth (1 = predict one extra token).
    - `mtp_lambda_high` / `mtp_lambda_low` / `mtp_lambda_switch_frac`: MTP loss
      weight schedule. The weight is `mtp_lambda_high` until
      `mtp_lambda_switch_frac * total_steps`, then `mtp_lambda_low` (V3 style:
      the switch lands in the final ~30% of training, not at the midpoint).

    Packed training:
    - `eos_token_id` doubles as the intra-document boundary marker used to
      build `cu_seqlens` for the KDA / varlen-MLA paths.

    Args:
        n_group (`int`, defaults to `8`): Number of expert groups for
            group-limited routing (V3). Each group holds
            `n_routed_experts // n_group` experts.
        topk_group (`int`, defaults to `4`): Number of groups considered for
            routing (V3).
        first_k_dense_replace (`int`, defaults to `1`): Number of initial
            layers using a dense FFN instead of MoE (V3).
        rope_interleave (`bool`, defaults to `True`): Whether the rotary
            position embedding uses interleaved weights. Unused under NoPE;
            kept for config compatibility.
        linear_attn_ratio (`int`, defaults to `3`): Number of consecutive KDA
            layers before each MLA layer. 3 gives the
            [KDA, KDA, KDA, MLA] x 7 layout for 28 layers.
        kda_num_heads (`int`, defaults to `12`): KDA attention heads
            (projection size = kda_num_heads * kda_head_dim).
        kda_head_dim (`int`, defaults to `128`): KDA per-head dimension.
        short_conv_kernel_size (`int`, defaults to `4`): KDA short
            convolution kernel size.
        num_nextn_predict_layers (`int`, defaults to `1`): MTP depth
            (1 = predict one extra token).
        mtp_lambda_high (`float`, defaults to `0.3`): MTP loss weight until
            the switch step.
        mtp_lambda_low (`float`, defaults to `0.1`): MTP loss weight after
            the switch step.
        mtp_lambda_switch_frac (`float`, defaults to `0.7`): Fraction of
            total steps at which the MTP loss weight switches (V3 style:
            the switch lands in the final ~30% of training, not at the
            midpoint).
    """

    model_type = "camellia"

    # --- Camellia-3B defaults (design report §1) ---
    vocab_size: int = 128_000
    hidden_size: int = 1_536
    intermediate_size: int = 4_224
    moe_intermediate_size: int = 320
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    n_shared_experts: int = 1
    n_routed_experts: int = 64
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 384
    q_lora_rank: int | None = None
    qk_rope_head_dim: int = 0  # NoPE
    v_head_dim: int | None = 96
    qk_nope_head_dim: int = 96
    n_group: int | None = 8
    topk_group: int | None = 4
    num_experts_per_tok: int | None = 6
    first_k_dense_replace: int | None = 1
    norm_topk_prob: bool | None = True
    max_position_embeddings: int = 8_192
    tie_word_embeddings: bool = True
    rope_parameters: object = None

    # --- KDA hybrid (design §1) ---
    linear_attn_ratio: int = 3
    kda_num_heads: int = 12
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4

    # --- MTP (design §1) ---
    num_nextn_predict_layers: int = 1
    mtp_lambda_high: float = 0.3
    mtp_lambda_low: float = 0.1
    mtp_lambda_switch_frac: float = 0.7

    def __post_init__(self, **kwargs):
        # NoPE: rotary embedding is constructed (base __init__) but never called;
        # give it a valid rope_parameters dict to keep construction happy.
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10_000.0}
        super().__post_init__(**kwargs)


def compute_cu_seqlens(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Build packed `cu_seqlens` from intra-document EOS markers.

    Args:
        input_ids: `[B, T]` token ids. Each document in a packed frame ends with
            `eos_token_id`.
        eos_token_id: The boundary token id.

    Returns:
        `LongTensor` of shape `[1 + num_docs]` covering the flattened `B*T`
        stream, or `None` when no boundary is found.
    """
    if input_ids.shape[0] * input_ids.shape[1] == 0:
        return None
    B, T = input_ids.shape
    S = B * T
    positions = torch.nonzero(input_ids == eos_token_id, as_tuple=False)
    if positions.numel() == 0:
        return None
    flat = positions[:, 0] * T + positions[:, 1] + 1
    flat = torch.unique(flat, sorted=True)
    # drop zero-length segments (consecutive boundaries)
    keep = flat[1:] > flat[:-1]
    flat = torch.cat([flat[:1], flat[1:][keep]])
    # A frame may end mid-document (packing cuts at arbitrary token offsets);
    # the trailing partial document becomes its own segment so `cu_seqlens`
    # always covers the full stream (contract of the fla varlen kernels).
    if int(flat[-1]) < S:
        flat = torch.cat([flat, torch.tensor([S], dtype=torch.long, device=flat.device)])
    return torch.cat([torch.zeros(1, dtype=torch.long, device=input_ids.device), flat.to(torch.long)])


def _adjust_cu_seqlens(cu_seqlens: torch.Tensor | None, B: int, T: int) -> torch.Tensor | None:
    """Map packed `cu_seqlens` from the B*T stream to the MTP B*(T-1) stream.

    The MTP stream drops each batch row's last position, so a boundary at
    flat position p in batch b = p // T moves to p - b (the row end of the
    full stream, p = B*T, maps to B*(T-1)). Zero-length segments (a
    1-token document ending exactly at a row end) are dropped.
    """
    if cu_seqlens is None:
        return None
    rest = cu_seqlens[1:]
    shifted = rest - rest // T
    cu = torch.cat([cu_seqlens[:1], shifted])
    keep = cu[1:] > cu[:-1]
    return torch.cat([cu[:1], cu[1:][keep]])


class CamelliaKDAAttention(nn.Module):
    """Kimi Delta Attention layer (fla-core KDA), Kimi-Linear parameter layout.

    Training path only (chunk kernel). With `cu_seqlens` the layer operates on
    the flattened packed stream ([1, B*T, ...] convention required by both
    `ShortConvolution` and `chunk_kda`) and resets the recurrent state at every
    intra-document boundary.
    """

    def __init__(self, config: CamelliaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.kda_num_heads
        self.head_dim = config.kda_head_dim
        self.conv_size = config.short_conv_kernel_size
        projection = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, projection, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection, bias=False)
        self.q_conv1d = ShortConvolution(projection, kernel_size=self.conv_size, activation="silu")
        self.k_conv1d = ShortConvolution(projection, kernel_size=self.conv_size, activation="silu")
        self.v_conv1d = ShortConvolution(projection, kernel_size=self.conv_size, activation="silu")
        # A_log / dt_bias stay fp32 (kernel contract, verified on sm_120)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(projection, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection, self.hidden_size, bias=False)
        self.o_proj._is_residual_proj = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, T, _ = hidden_states.shape
        if cu_seqlens is not None:
            h = hidden_states.reshape(1, B * T, -1)  # packed convention: B must be 1
        else:
            h = hidden_states
        q = self.q_conv1d(self.q_proj(h), cu_seqlens=cu_seqlens)[0]
        k = self.k_conv1d(self.k_proj(h), cu_seqlens=cu_seqlens)[0]
        v = self.v_conv1d(self.v_proj(h), cu_seqlens=cu_seqlens)[0]
        # 0.5.2 API: g must be [..., H, K] before the fused gate kernel
        g = self.f_b_proj(self.f_a_proj(h)).view(*h.shape[:-1], self.num_heads, self.head_dim)
        g = fused_kda_gate(g, self.A_log, self.dt_bias)
        beta = self.b_proj(h).float().sigmoid()
        if cu_seqlens is not None:
            q = q.view(1, B * T, self.num_heads, self.head_dim)
            k = k.view(1, B * T, self.num_heads, self.head_dim)
            v = v.view(1, B * T, self.num_heads, self.head_dim)
            g = g.view(1, B * T, self.num_heads, self.head_dim)
            beta = beta.view(1, B * T, self.num_heads)
            mode = "chunk" if B * T > 64 else "fused_recurrent"
        else:
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
            g = g.view(B, T, self.num_heads, self.head_dim)
            beta = beta.view(B, T, self.num_heads)
            mode = "chunk" if T > 64 else "fused_recurrent"
        kernel = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o, _ = kernel(
            q=q, k=k, v=v, g=g, beta=beta,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        o = o.view(B, T, self.num_heads, self.head_dim)
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(B, T, self.num_heads, self.head_dim)
        o = self.o_norm(o, gate)
        o = self.o_proj(o.reshape(B, T, self.num_heads * self.head_dim))
        return o, None


class CamelliaMLAAttention(nn.Module):
    """DeepSeek MLA with NoPE + per-head QK-Norm + varlen (cu_seqlens) support.

    Training path: low-rank KV (kv_lora_rank) is expanded to per-head k/v and
    attention runs through `flash_attn_varlen_func` when `cu_seqlens` is given
    (intra-document boundaries), otherwise through SDPA with a causal mask.
    """

    def __init__(self, config: CamelliaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.qk_nope_head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_nope_head_dim, bias=False)
        # QK-Norm (Qwen3 / nanochat style, per head)
        self.q_norm = DeepseekV3RMSNorm(self.qk_nope_head_dim, eps=config.rms_norm_eps)
        self.k_norm = DeepseekV3RMSNorm(self.qk_nope_head_dim, eps=config.rms_norm_eps)
        self.kv_a_proj = nn.Linear(config.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.o_proj._is_residual_proj = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.qk_nope_head_dim)
        q = self.q_norm(q)
        kv = self.kv_a_layernorm(self.kv_a_proj(hidden_states))
        kv = self.kv_b_proj(kv).view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k = self.k_norm(k)

        if cu_seqlens is not None:
            S = B * T
            if flash_attn_varlen_func is not None:
                max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
                qf = q.reshape(S, self.num_heads, self.qk_nope_head_dim)
                kf = k.reshape(S, self.num_heads, self.qk_nope_head_dim)
                vf = v.reshape(S, self.num_heads, self.v_head_dim)
                o = flash_attn_varlen_func(
                    qf, kf, vf,
                    cu_seqlens=cu_seqlens, cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_len, max_seqlen_k=max_len,
                    softmax_scale=self.scaling, causal=True,
                )
                o = o.view(B, T, self.num_heads, self.v_head_dim)
            else:  # SDPA fallback: per-document causal attention.
                # Documents are contiguous segments of the flattened stream,
                # so a block-diagonal mask is equivalent to causal attention
                # per segment. (A 2-D bool [S, S] mask is mishandled by SDPA
                # on torch 2.11 / sm_120 — verified 2026-09-26.)
                qf = q.reshape(S, self.num_heads, self.qk_nope_head_dim)
                kf = k.reshape(S, self.num_heads, self.qk_nope_head_dim)
                vf = v.reshape(S, self.num_heads, self.v_head_dim)
                out_parts = []
                start = 0
                for length in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist():
                    end = start + length
                    out_parts.append(
                        F.scaled_dot_product_attention(
                            qf[start:end].transpose(0, 1),
                            kf[start:end].transpose(0, 1),
                            vf[start:end].transpose(0, 1),
                            is_causal=True,
                            scale=self.scaling,
                        ).transpose(0, 1)
                    )
                    start = end
                o = torch.cat(out_parts, dim=0).view(B, T, self.num_heads, self.v_head_dim)
        else:
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=self.scaling)
            o = o.transpose(1, 2)

        o = self.o_proj(o.reshape(B, T, self.num_heads * self.v_head_dim))
        return o, None


class CamelliaNaiveMoe(DeepseekV3NaiveMoe):
    """Expert collection with per-expert output RMS statistics (design §6).

    The parent model attaches an `[E]` fp32 counter (`_out_norm_stats`) and a
    hit counter (`_out_hit_stats`); the forward loop accumulates the RMS of
    each expert's contribution (before the top-k weighting) for the outlier-
    expert / Localized Activation Blow-up monitoring.
    """

    _out_norm_stats: torch.Tensor | None
    _out_hit_stats: torch.Tensor | None

    def __init__(self, config):
        super().__init__(config)
        self._out_norm_stats = None
        self._out_hit_stats = None

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            if self._out_norm_stats is not None:
                with torch.no_grad():
                    rms = current_hidden_states.float().pow(2).mean(dim=-1).sqrt().mean().clamp_max(1e6)
                    self._out_norm_stats[expert_idx].add_(rms)
                    self._out_hit_stats[expert_idx].add_(1)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class CamelliaMoE(DeepseekV3MoE):
    """DeepSeek-V3 MoE block with per-layer routed-expert load statistics.

    The parent model attaches shared fp32 counters to each MoE block:
    `_load_stats` `[E]` (routed token counts per expert) and, on the expert
    collection, `_out_norm_stats` / `_out_hit_stats` `[E]` (per-expert output
    RMS, see `CamelliaNaiveMoe`). Training-side callbacks all-reduce the
    counters for the global whole-batch load distribution (design §7
    実装前確定事項 2 / §6 MaxVio・expert 出力ノルム).
    """

    _load_stats: torch.Tensor | None

    def __init__(self, config: CamelliaConfig):
        super().__init__(config)
        self.experts = CamelliaNaiveMoe(config)
        # Shared-expert down_proj is a residual-stream projection
        # (design §7 item 3).
        self.shared_experts.down_proj._is_residual_proj = True
        self._load_stats = None

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        if self._load_stats is not None:
            with torch.no_grad():
                counts = topk_indices.view(-1).bincount(minlength=self.n_routed_experts)
                self._load_stats.add_(counts.to(self._load_stats.dtype).to(self._load_stats.device))
        return hidden_states


class CamelliaDecoderLayer(DeepseekV3DecoderLayer):
    """Pre-norm decoder layer. Attention type is fixed by layer index:
    `layer_idx % (linear_attn_ratio + 1) == linear_attn_ratio` -> MLA, else KDA.
    FFN is dense for `layer_idx < first_k_dense_replace`, MoE otherwise.
    """

    def __init__(self, config: CamelliaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_mla = layer_idx % (config.linear_attn_ratio + 1) == config.linear_attn_ratio
        if self.is_mla:
            self.self_attn = CamelliaMLAAttention(config, layer_idx)
        else:
            self.self_attn = CamelliaKDAAttention(config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = CamelliaMoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)
            self.mlp.down_proj._is_residual_proj = True
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CamelliaMTPModule(nn.Module):
    """One-step Multi-Token-Prediction module (V3 style, D=1).

    Input: `[B, T, 2D]` = concat(embed(x_{t+1}), hidden_t) for t = 0..T-2.
    One full block: fc(2D->D) + RMSNorm + MLA (NoPE) + RMSNorm + dense FFN.
    The model's final norm and lm_head are shared with the main path.
    """

    def __init__(self, config: CamelliaConfig):
        super().__init__()
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CamelliaMLAAttention(config, layer_idx=config.num_hidden_layers)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepseekV3MLP(config)
        self.mlp.down_proj._is_residual_proj = True

    def forward(self, combined: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.fc(combined)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class CamelliaPreTrainedModel(DeepseekV3PreTrainedModel):
    config: CamelliaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CamelliaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "A_log", "dt_bias"]
    _tp_plan: dict = {}
    _pp_plan: dict = {}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Residual-stream projections (o_proj / down_proj) are re-initialized
        # with std / sqrt(2 * L) for pre-norm stability at initialization
        # (design §7 item 3, L = 28 -> scale ~= 0.00267).
        if getattr(module, "_is_residual_proj", False):
            scaled = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
            init.normal_(module.weight, mean=0.0, std=scaled)
        elif isinstance(module, CamelliaMoE):
            pass  # handled below through the NaiveMoe branch
        # Post-generation note: the generated file flattens `CamelliaNaiveMoe`
        # to a plain nn.Module, so it must be matched here (not via its
        # DeepseekV3NaiveMoe base) for the scaled down_proj init to apply.
        elif isinstance(module, CamelliaNaiveMoe):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(
                module.down_proj,
                mean=0.0,
                std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
            )
        elif isinstance(module, CamelliaKDAAttention):
            init.normal_(module.dt_bias, mean=0.0, std=0.1)


@auto_docstring
class CamelliaModel(DeepseekV3Model):
    def __init__(self, config: CamelliaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CamelliaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Per-MoE-layer routed token counters + per-expert output RMS stats
        self.moe_layers = [layer for layer in self.layers if isinstance(layer.mlp, CamelliaMoE)]
        n_experts = config.n_routed_experts
        self.register_buffer(
            "moe_load_stats",
            torch.zeros(len(self.moe_layers), n_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "moe_out_norm_stats",
            torch.zeros(len(self.moe_layers), n_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "moe_out_hit_stats",
            torch.zeros(len(self.moe_layers), n_experts, dtype=torch.float32),
            persistent=False,
        )
        for i, layer in enumerate(self.moe_layers):
            layer.mlp._load_stats = self.moe_load_stats[i]
            layer.mlp.experts._out_norm_stats = self.moe_out_norm_stats[i]
            layer.mlp.experts._out_hit_stats = self.moe_out_hit_stats[i]

        self.post_init()

    def reset_moe_load_stats(self) -> None:
        self.moe_load_stats.zero_()
        self.moe_out_norm_stats.zero_()
        self.moe_out_hit_stats.zero_()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Packed intra-document boundaries (design §7 実装前確定事項 1):
        # KDA resets its recurrent state and MLA masks across documents.
        cu_seqlens = None
        if input_ids is not None and self.config.eos_token_id is not None:
            cu_seqlens = compute_cu_seqlens(input_ids, self.config.eos_token_id)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, cu_seqlens=cu_seqlens, **kwargs)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class CamelliaForCausalLM(DeepseekV3ForCausalLM):
    """NTP + MTP (D=1) training objective.

    `loss = L_ntp + mtp_lambda * L_mtp` where `mtp_lambda` is a mutable
    attribute (set by the training loop from the config schedule).
    """

    def __init__(self, config: CamelliaConfig):
        super().__init__(config)
        self.model = CamelliaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.num_nextn_predict_layers > 0:
            self.mtp = CamelliaMTPModule(config)
        else:
            self.mtp = None
        self.mtp_lambda: float = config.mtp_lambda_high
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        mtp_loss = None
        if labels is not None and self.mtp is not None and input_ids is not None:
            B, T = input_ids.shape[0], input_ids.shape[1]
            cu_seqlens = compute_cu_seqlens(input_ids, self.config.eos_token_id) if self.config.eos_token_id is not None else None
            next_embeds = self.model.embed_tokens(torch.roll(input_ids, shifts=-1, dims=1))
            combined = torch.cat([next_embeds[:, :-1], hidden_states[:, :-1]], dim=-1)  # [B, T-1, 2D]
            mtp_cu = _adjust_cu_seqlens(cu_seqlens, B, T)
            mtp_hidden = self.mtp(combined, cu_seqlens=mtp_cu)
            mtp_logits = self.lm_head(self.model.norm(mtp_hidden))
            # labels = input_ids (unshifted, PackedDataset convention).
            # NTP: logits[t] predicts x_{t+1} -> pair logits[:T-1] with labels[1:].
            # MTP: mtp_logits[t] predicts x_{t+2} -> pair mtp_logits[:T-2] with labels[2:]
            # (the last MTP position has no in-sequence target x_T).
            mtp_logits = mtp_logits[:, :-1]
            shift_logits = mtp_logits.contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 2:].contiguous().view(-1)
            mtp_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            ntp_logits = logits[:, :-1].contiguous().view(-1, self.config.vocab_size)
            ntp_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(ntp_logits, ntp_labels, ignore_index=-100) + self.mtp_lambda * mtp_loss
        elif labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "CamelliaConfig",
    "CamelliaForCausalLM",
    "CamelliaModel",
    "CamelliaPreTrainedModel",
]
