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
"""PyTorch Freesia model.

Freesia の本体。Transformer の骨格は Qwen3 の部品（RMSNorm・RoPE・SwiGLU MLP・QK-Norm）を使い、
次の独自要素を加える（設計書 §2）。

* Bloom Attention（`FreesiaAttention`）: head ごとのゲート g で未来位置のスコアに log(g) を足す。
  g=0 で causal、g=1 で双方向。LM モードでは閉、穴埋めモードでは開に固定し、埋め込みモードで学習する
* モード埋め込み（`FreesiaModel.mode_embed`）: LM / 穴埋め / 埋め込みのどのモードかを入力に足して伝える
* Petal Pooling（`FreesiaPetalPooling`）: 学習する花弁クエリの cross-attention 出力を mean pooling に足す。
  出力射影をゼロ初期化するので、学習開始時は mean pooling と一致する
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ..qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb
from .configuration_freesia import FreesiaConfig


MODE_LM = 0
MODE_MASKED = 1
MODE_EMBED = 2

# Bloom gate overrides used by the ablations (design §6, stage 1b).
BLOOM_LEARNED = "learned"  # per-head learned gate, clipped from below by the schedule floor
BLOOM_CLOSED = "closed"  # exact causal attention
BLOOM_OPEN = "open"  # exact bidirectional attention
BLOOM_FLOOR_ONLY = "floor_only"  # all heads share the schedule floor (Conan-v2 style global soft mask)


class FreesiaRMSNorm(Qwen3RMSNorm):
    pass


class FreesiaMLP(Qwen3MLP):
    def __init__(self, config: FreesiaConfig):
        super().__init__(config)
        self.down_proj._is_residual_proj = True


class FreesiaAttention(nn.Module):
    """Multi-head attention with Bloom gates.

    Bloom ゲート付きの MHA。未来位置（j > i）のスコアにだけ head ごとのバイアス log(g_h) を足す。

    Args:
        config (FreesiaConfig): Model configuration.
        layer_idx (int): Index of the layer (used only for debugging and gate logging).
    """

    def __init__(self, config: FreesiaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.o_proj._is_residual_proj = True
        self.q_norm = FreesiaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = FreesiaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.bloom_logit = nn.Parameter(torch.full((self.num_heads,), float(config.bloom_gate_init)))

    def bloom_gate(self) -> torch.Tensor:
        """Return the current gate values g_h in [0, 1].

        Returns:
            torch.Tensor: Gate values with shape `(num_heads,)`.
        """
        return torch.sigmoid(self.bloom_logit.float())

    def _future_bias(self, override: str, floor: float, dtype: torch.dtype) -> torch.Tensor:
        """Build the per-head additive bias for future positions.

        Args:
            override (str): One of `learned`, `closed`, `open`, `floor_only`.
            floor (float): Current lower bound of the gate from the bloom schedule (0-1).
            dtype (torch.dtype): Dtype of the attention scores.

        Returns:
            torch.Tensor: Bias with shape `(num_heads,)`.

        Raises:
            ValueError: If `override` is unknown.
        """
        min_log = math.log(self.config.bloom_min_gate)
        floor_log = math.log(max(floor, self.config.bloom_min_gate))
        device = self.bloom_logit.device
        if override == BLOOM_LEARNED:
            # effective gate = floor + (1 - floor) * sigmoid(logit): the schedule floor sets a minimum opening,
            # and the learned part always receives gradient (a hard max() would cut it once the floor wins).
            floor_c = min(max(floor, 0.0), 1.0)
            g_eff = floor_c + (1.0 - floor_c) * torch.sigmoid(self.bloom_logit.float())
            bias = torch.log(g_eff.clamp(min=self.config.bloom_min_gate))
        elif override == BLOOM_FLOOR_ONLY:
            bias = torch.full((self.num_heads,), floor_log, device=device)
        elif override == BLOOM_OPEN:
            bias = torch.zeros(self.num_heads, device=device)
        elif override == BLOOM_CLOSED:
            bias = torch.full((self.num_heads,), torch.finfo(dtype).min / 2, device=device)
        else:
            raise ValueError(f"unknown bloom override: {override}")
        return bias.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.Tensor | None,
        mode: int,
        bloom_override: str,
        bloom_floor: float,
    ) -> torch.Tensor:
        """Apply attention.

        Args:
            hidden_states (torch.Tensor): Input with shape `(batch, seq, hidden)`.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): RoPE `(cos, sin)`.
            valid_mask (torch.Tensor | None): Bool mask `(batch, seq)`, True for real tokens. None = no padding.
            mode (int): `MODE_LM`, `MODE_MASKED` or `MODE_EMBED`.
            bloom_override (str): Gate override used in the embedding mode.
            bloom_floor (float): Current schedule floor of the gate.

        Returns:
            torch.Tensor: Output with shape `(batch, seq, hidden)`.
        """
        bsz, seq, _ = hidden_states.shape
        shape = (bsz, seq, self.num_heads, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        neg = torch.finfo(q.dtype).min / 2
        if mode == MODE_LM and valid_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mode == MODE_MASKED or (mode == MODE_EMBED and bloom_override == BLOOM_OPEN):
            attn_mask = None if valid_mask is None else valid_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            override = BLOOM_CLOSED if mode == MODE_LM else bloom_override
            future = torch.ones(seq, seq, device=q.device, dtype=torch.bool).triu(1)
            head_bias = self._future_bias(override, bloom_floor, q.dtype)
            attn_mask = future[None, None, :, :].to(q.dtype) * head_bias[None, :, None, None]
            if valid_mask is not None:
                attn_mask = attn_mask + (~valid_mask)[:, None, None, :].to(q.dtype) * neg
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


class FreesiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FreesiaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = FreesiaAttention(config, layer_idx)
        self.mlp = FreesiaMLP(config)
        self.input_layernorm = FreesiaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FreesiaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, valid_mask, mode, bloom_override, bloom_floor):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, valid_mask, mode, bloom_override, bloom_floor
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class FreesiaPetalPooling(nn.Module):
    """Petal Pooling: learned petal queries on top of masked mean pooling.

    学習する K 枚の花弁クエリが最終層に cross-attention をかけ、その平均を mean pooling に足す。
    出力射影をゼロ初期化するので、学習開始時の出力は mean pooling（を RMSNorm した値）と一致する。

    Args:
        config (FreesiaConfig): Model configuration.
    """

    def __init__(self, config: FreesiaConfig):
        super().__init__()
        self.petals = nn.Parameter(torch.zeros(config.petal_count, config.hidden_size))
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, bias=False, batch_first=True
        )
        self.attn.out_proj._zero_init = True
        self.norm = FreesiaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        """Pool token states into one L2-normalized vector per sequence.

        Args:
            hidden_states (torch.Tensor): Final hidden states `(batch, seq, hidden)`.
            pool_mask (torch.Tensor): `(batch, seq)`, 1 for tokens to aggregate (body text, not the
                instruction prefix nor padding).

        Returns:
            torch.Tensor: Embeddings `(batch, hidden)` in float32, L2-normalized.
        """
        mask = pool_mask.bool()
        m = mask.to(hidden_states.dtype).unsqueeze(-1)
        mean = (hidden_states * m).sum(1) / m.sum(1).clamp(min=1.0)
        queries = self.petals.to(hidden_states.dtype).unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        petal_out, _ = self.attn(queries, hidden_states, hidden_states, key_padding_mask=~mask, need_weights=False)
        pooled = self.norm(mean + petal_out.mean(1))
        return F.normalize(pooled.float(), dim=-1)


class FreesiaPreTrainedModel(PreTrainedModel):
    config: FreesiaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FreesiaDecoderLayer"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if getattr(module, "_is_residual_proj", False):
            nn.init.normal_(module.weight, mean=0.0, std=std / math.sqrt(2 * self.config.num_hidden_layers))
        if getattr(module, "_zero_init", False):
            nn.init.zeros_(module.weight)
        if isinstance(module, FreesiaAttention):
            module.bloom_logit.fill_(float(self.config.bloom_gate_init))
        if isinstance(module, FreesiaPetalPooling):
            nn.init.normal_(module.petals, mean=0.0, std=std)
        if isinstance(module, FreesiaModel):
            nn.init.normal_(module.mode_embed.weight, mean=0.0, std=std)


class FreesiaModel(FreesiaPreTrainedModel):
    """Freesia encoder backbone with Bloom Attention and Petal Pooling.

    Freesia の本体。`forward` は最終層の隠れ状態を、`encode` は 768 次元の正規化済み埋め込みを返す。
    Bloom ゲートの扱いは属性 `bloom_override`（既定 `learned`）と `bloom_floor`（スケジュールの下限、既定 0）で制御する。

    Args:
        config (FreesiaConfig): Model configuration.
    """

    def __init__(self, config: FreesiaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mode_embed = nn.Embedding(config.num_modes, config.hidden_size)
        self.layers = nn.ModuleList([FreesiaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = FreesiaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.petal_pooling = FreesiaPetalPooling(config)
        self.bloom_override = BLOOM_LEARNED
        self.bloom_floor = 0.0
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def bloom_gates(self, effective: bool = False) -> torch.Tensor:
        """Return all Bloom gate values.

        Args:
            effective (bool): If True, return the gates actually applied in the learned mode, i.e.
                `floor + (1 - floor) * g` with the current `bloom_floor`.

        Returns:
            torch.Tensor: `(num_layers, num_heads)` gate values in [0, 1].
        """
        gates = torch.stack([layer.self_attn.bloom_gate() for layer in self.layers])
        if effective:
            floor = min(max(float(self.bloom_floor), 0.0), 1.0)
            gates = floor + (1.0 - floor) * gates
        return gates

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        mode: int = MODE_EMBED,
        **kwargs,
    ) -> BaseModelOutput:
        """Run the backbone.

        Args:
            input_ids (torch.LongTensor): `(batch, seq)` token ids.
            attention_mask (torch.Tensor | None): `(batch, seq)`, 1 for real tokens. None = no padding.
            mode (int): `MODE_LM`, `MODE_MASKED` or `MODE_EMBED`.

        Returns:
            BaseModelOutput: `last_hidden_state` with shape `(batch, seq, hidden)`.
        """
        hidden_states = self.embed_tokens(input_ids) + self.mode_embed.weight[mode].to(self.embed_tokens.weight.dtype)
        seq = input_ids.shape[1]
        position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        valid_mask = None
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            valid_mask = attention_mask.bool()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings, valid_mask, mode, self.bloom_override, self.bloom_floor
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode texts into 768-dim L2-normalized embeddings (embedding mode).

        Args:
            input_ids (torch.LongTensor): `(batch, seq)` token ids.
            attention_mask (torch.Tensor): `(batch, seq)`, 1 for real tokens.
            pool_mask (torch.Tensor | None): `(batch, seq)`, 1 for tokens to aggregate. Defaults to
                `attention_mask`. Set the instruction prefix to 0 to exclude it from pooling.

        Returns:
            torch.Tensor: `(batch, hidden)` float32 embeddings.
        """
        hidden = self.forward(input_ids, attention_mask, mode=MODE_EMBED).last_hidden_state
        return self.petal_pooling(hidden, attention_mask if pool_mask is None else pool_mask)


class FreesiaForPreTraining(FreesiaPreTrainedModel):
    """Freesia with a tied LM head for Janus pretraining.

    Janus 事前学習用。LM モード（次トークン予測）と穴埋めモード（MNTP: 位置 i の `<mask>` を i-1 の出力で当てる）の
    どちらも「1 つ前の位置の出力で次のラベルを予測する」同じ損失で学習する。穴埋めモードでは、マスクしていない位置の
    ラベルを -100 にしておく。

    Args:
        config (FreesiaConfig): Model configuration.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: FreesiaConfig):
        super().__init__(config)
        self.model = FreesiaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mode: int = MODE_LM,
        **kwargs,
    ) -> CausalLMOutput:
        """Compute the Janus pretraining loss.

        Args:
            input_ids (torch.LongTensor): `(batch, seq)` input ids (masked ids in the masked mode).
            labels (torch.LongTensor | None): `(batch, seq)` targets aligned with `input_ids`; -100 is ignored.
            attention_mask (torch.Tensor | None): `(batch, seq)`, 1 for real tokens.
            mode (int): `MODE_LM` or `MODE_MASKED`.

        Returns:
            CausalLMOutput: `loss` (when labels are given) and `logits`.
        """
        hidden = self.model(input_ids, attention_mask, mode=mode).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)


__all__ = [
    "FreesiaForPreTraining",
    "FreesiaModel",
    "FreesiaPreTrainedModel",
]
