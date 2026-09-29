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
"""Freesia model configuration.

Freesia は日英テキスト埋め込み用にスクラッチで学習する独立モデル。
Transformer 本体に加えて、3 つの独自要素を持つ（設計書 §2）。

* Bloom Attention: 層・head ごとの学習可能ゲートで、未来位置への attention を 0（causal）〜1（双方向）で開く
* Janus 事前学習: 同じ重みで causal LM と MNTP 型の穴埋め予測を交互に学習し、モード埋め込みで区別する
* Petal Pooling: 学習する花弁クエリの cross-attention 出力を mean pooling に足して埋め込みにする
"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="yasutoshi-lab/freesia-300m")
@strict
class FreesiaConfig(PreTrainedConfig):
    r"""
    bloom_gate_init (`float`, *optional*, defaults to -4.0):
        Initial logit of the Bloom gates. `sigmoid(-4.0) ~= 0.018`, i.e. almost closed (causal) at the start
        of the embedding stage.
    bloom_min_gate (`float`, *optional*, defaults to 1e-4):
        Numerical floor of the Bloom gate used when converting the gate to a log-space attention bias.
    num_modes (`int`, *optional*, defaults to 3):
        Number of mode embeddings. 0 = causal LM, 1 = masked (MNTP) prediction, 2 = embedding.
    petal_count (`int`, *optional*, defaults to 8):
        Number of learned petal queries used by Petal Pooling.
    pooling_mode (`str`, *optional*, defaults to "petal"):
        Pooling used by `FreesiaModel.encode`: "petal" (Petal Pooling), "mean" (masked mean + RMSNorm) or
        "last" (last pooled token + RMSNorm). "mean" / "last" are ablations (design §6, stage 1b).
    mask_token_id (`int`, *optional*):
        Id of the `<mask>` token used by the masked (MNTP) prediction mode.

    ```python
    >>> from transformers import FreesiaConfig, FreesiaModel

    >>> configuration = FreesiaConfig()  # Freesia-300M
    >>> model = FreesiaModel(configuration)
    ```
    """

    model_type = "freesia"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32_768
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 1_024
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_dropout: float | int = 0.0
    bloom_gate_init: float = -4.0
    bloom_min_gate: float = 1e-4
    num_modes: int = 3
    petal_count: int = 8
    pooling_mode: str = "petal"
    use_cache: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    mask_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10_000.0}
        super().__post_init__(**kwargs)


__all__ = ["FreesiaConfig"]
