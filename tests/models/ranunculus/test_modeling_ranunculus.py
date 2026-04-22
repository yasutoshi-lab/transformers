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
"""Testing suite for the PyTorch Ranunculus model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers import RanunculusModel

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class RanunculusModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = RanunculusModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Non-zero dropout breaks TP backward tests because RNG state diverges
        # between the TP and non-TP runs; Ranunculus trains with dropout = 0 anyway.
        self.attention_probs_dropout_prob = 0.0


@require_torch
class RanunculusModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = RanunculusModelTester

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True
