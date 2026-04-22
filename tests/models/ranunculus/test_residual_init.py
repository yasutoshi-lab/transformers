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
"""Unit test for Ranunculus §8 residual-scaling weight init."""

import math
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import RanunculusConfig, RanunculusForCausalLM


@require_torch
class RanunculusResidualInitTest(unittest.TestCase):
    """Design §8: o_proj / down_proj init std = initializer_range / sqrt(2 * num_hidden_layers)."""

    # Use enough layers/hidden so per-tensor std estimates are stable.
    CONFIG_KWARGS = {
        "vocab_size": 256,
        "hidden_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
    }

    def _build_model(self):
        config = RanunculusConfig(**self.CONFIG_KWARGS)
        torch.manual_seed(0)
        return RanunculusForCausalLM(config), config

    def test_residual_projections_use_scaled_std(self):
        model, config = self._build_model()
        expected_std = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)

        checked = 0
        for name, module in model.named_modules():
            if not getattr(module, "_is_residual_proj", False):
                continue
            checked += 1
            std = module.weight.detach().float().std().item()
            # Allow a 5% tolerance; per-tensor std has ~1/sqrt(fan-in) sampling noise.
            self.assertLess(
                abs(std - expected_std) / expected_std,
                0.05,
                msg=f"{name} std={std:.6f} vs expected={expected_std:.6f}",
            )

        # 2 residual projections (o_proj + down_proj) per layer.
        self.assertEqual(checked, 2 * config.num_hidden_layers)

    def test_non_residual_projections_use_default_std(self):
        model, config = self._build_model()
        expected_std = config.initializer_range

        non_residual_suffixes = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
        )
        checked = 0
        for name, param in model.named_parameters():
            if not any(name.endswith(suffix) for suffix in non_residual_suffixes):
                continue
            checked += 1
            std = param.detach().float().std().item()
            self.assertLess(
                abs(std - expected_std) / expected_std,
                0.05,
                msg=f"{name} std={std:.6f} vs expected={expected_std:.6f}",
            )

        # 5 non-residual projections per layer: q/k/v/gate/up.
        self.assertEqual(checked, 5 * config.num_hidden_layers)

    def test_residual_flags_present(self):
        model, _ = self._build_model()
        flagged = {name for name, module in model.named_modules() if getattr(module, "_is_residual_proj", False)}
        # Every layer must contribute exactly its o_proj and down_proj.
        for i in range(self.CONFIG_KWARGS["num_hidden_layers"]):
            self.assertIn(f"model.layers.{i}.self_attn.o_proj", flagged)
            self.assertIn(f"model.layers.{i}.mlp.down_proj", flagged)
