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
"""PyTorch Ranunculus model."""

import math

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
)


@auto_docstring(checkpoint="yasutoshi-lab/ranunculus-1b")
@strict
class RanunculusConfig(Qwen3Config):
    r"""
    ```python
    >>> from transformers import RanunculusModel, RanunculusConfig

    >>> # Initializing a Ranunculus-1B style configuration
    >>> configuration = RanunculusConfig()

    >>> # Initializing a model from the Ranunculus-1B style configuration
    >>> model = RanunculusModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ranunculus"

    vocab_size: int = 96_000
    hidden_size: int = 1_536
    intermediate_size: int = 4_096
    num_hidden_layers: int = 35
    num_attention_heads: int = 12
    num_key_value_heads: int | None = 3
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 8_192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 35
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 500_000.0}
        super().__post_init__(**kwargs)


class RanunculusAttention(Qwen3Attention):
    def __init__(self, config: RanunculusConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.o_proj._is_residual_proj = True


class RanunculusMLP(Qwen3MLP):
    def __init__(self, config: RanunculusConfig):
        super().__init__(config)
        self.down_proj._is_residual_proj = True


class RanunculusPreTrainedModel(Qwen3PreTrainedModel):
    config: RanunculusConfig
    _no_split_modules = ["RanunculusDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        # Design §8 residual scaling: o_proj / down_proj are the projections
        # that write back into the residual stream, so their initial std is
        # shrunk by 1/sqrt(2 * num_hidden_layers) to keep pre-norm activations
        # stable at the start of training.
        if getattr(module, "_is_residual_proj", False):
            scaled = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
            nn.init.normal_(module.weight, mean=0.0, std=scaled)


class RanunculusModel(Qwen3Model):
    pass


class RanunculusForCausalLM(Qwen3ForCausalLM):
    pass


__all__ = [
    "RanunculusConfig",
    "RanunculusForCausalLM",
    "RanunculusModel",
    "RanunculusPreTrainedModel",
]
