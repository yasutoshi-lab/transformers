<!--Copyright 2026 yasutoshi-lab and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-22 and added to Hugging Face Transformers on 2026-04-22.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/FlashAttention-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Ranunculus

Ranunculus is a compact multilingual Dense decoder-only LM series by yasutoshi-lab. The 700M variant, `ranunculus-700m`, is pre-trained from scratch on Wikipedia in four languages (English, Japanese, Chinese, German) for roughly 15.1B tokens on a single GPU.

Architecturally it mirrors [Qwen3](qwen3): 30 transformer layers, hidden size 1280, 16 query heads with 4 KV heads (GQA 4:1), head_dim 80, intermediate size 3584, tied input/output embeddings, and RoPE with `rope_theta=500000`. Attention includes Qwen3-style per-head `q_norm`/`k_norm` RMSNorms for training stability.

Residual projections (`o_proj` and `down_proj`) are initialized with a reduced standard deviation of `0.02 / sqrt(2 * num_hidden_layers)` to keep the residual stream stable in a deep pre-norm architecture (see design report §8).

The example below shows how to load the model with [`AutoModelForCausalLM`].

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("yasutoshi-lab/ranunculus-700m")
model = AutoModelForCausalLM.from_pretrained(
    "yasutoshi-lab/ranunculus-700m",
    dtype=torch.bfloat16,
    device_map="auto",
)

inputs = tokenizer("The key to multilingual modeling is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## RanunculusConfig

[[autodoc]] RanunculusConfig

## RanunculusModel

[[autodoc]] RanunculusModel
    - forward

## RanunculusForCausalLM

[[autodoc]] RanunculusForCausalLM
    - forward
