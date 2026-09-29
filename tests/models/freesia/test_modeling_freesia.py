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
"""Unit tests for Freesia (Bloom Attention / Janus / Petal Pooling)."""

import tempfile

import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, FreesiaConfig, FreesiaForPreTraining, FreesiaModel
from transformers.models.freesia.modeling_freesia import (
    BLOOM_CLOSED,
    BLOOM_LEARNED,
    BLOOM_OPEN,
    MODE_EMBED,
    MODE_LM,
    MODE_MASKED,
)


def tiny_config(**kw):
    base = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        head_dim=16, max_position_embeddings=32, petal_count=3, pad_token_id=0, mask_token_id=1,
    )
    base.update(kw)
    return FreesiaConfig(**base)


def _model():
    torch.manual_seed(0)
    return FreesiaModel(tiny_config()).eval()


def test_closed_gate_matches_causal_lm_mode():
    model = _model()
    ids = torch.randint(2, 128, (2, 10))
    lm = model(ids, mode=MODE_LM).last_hidden_state
    model.bloom_override = BLOOM_CLOSED
    emb = model(ids, mode=MODE_EMBED).last_hidden_state
    # same attention pattern; only the mode embedding differs -> compare after equalizing mode vectors
    model.mode_embed.weight.data[MODE_EMBED] = model.mode_embed.weight.data[MODE_LM]
    emb = model(ids, mode=MODE_EMBED).last_hidden_state
    torch.testing.assert_close(lm, emb, atol=1e-5, rtol=1e-5)


def test_open_gate_matches_masked_mode():
    model = _model()
    ids = torch.randint(2, 128, (2, 10))
    model.mode_embed.weight.data[MODE_EMBED] = model.mode_embed.weight.data[MODE_MASKED]
    masked = model(ids, mode=MODE_MASKED).last_hidden_state
    model.bloom_override = BLOOM_OPEN
    emb = model(ids, mode=MODE_EMBED).last_hidden_state
    torch.testing.assert_close(masked, emb, atol=1e-5, rtol=1e-5)


def test_causal_prefix_invariance_in_lm_mode():
    model = _model()
    ids = torch.randint(2, 128, (1, 10))
    full = model(ids, mode=MODE_LM).last_hidden_state
    prefix = model(ids[:, :6], mode=MODE_LM).last_hidden_state
    torch.testing.assert_close(full[:, :6], prefix, atol=1e-5, rtol=1e-5)


def test_learned_gate_receives_gradient_and_floor_applies():
    model = FreesiaModel(tiny_config()).train()
    ids = torch.randint(2, 128, (2, 10))
    mask = torch.ones_like(ids)
    model.bloom_override = BLOOM_LEARNED
    model.bloom_floor = 0.3
    model.encode(ids, mask).sum().backward()
    grads = [layer.self_attn.bloom_logit.grad for layer in model.layers]
    assert all(g is not None for g in grads)
    bias = model.layers[0].self_attn._future_bias(BLOOM_LEARNED, 0.3, torch.float32)
    assert torch.all(bias >= torch.log(torch.tensor(0.3)) - 1e-6)


def test_padding_does_not_change_real_tokens():
    model = _model()
    ids = torch.randint(2, 128, (1, 8))
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 8), torch.zeros(1, 4)], dim=1).long()
    for override in (BLOOM_OPEN, BLOOM_LEARNED):
        model.bloom_override = override
        a = model.encode(ids, torch.ones_like(ids))
        b = model.encode(padded, mask)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_petal_pooling_starts_as_mean_pooling():
    model = _model()
    ids = torch.randint(2, 128, (2, 10))
    mask = torch.ones_like(ids)
    hidden = model(ids, mask, mode=MODE_EMBED).last_hidden_state
    expected = F.normalize(model.petal_pooling.norm(hidden.mean(1)).float(), dim=-1)
    torch.testing.assert_close(model.encode(ids, mask), expected, atol=1e-5, rtol=1e-5)


def test_pretraining_loss_both_modes_and_tied_head():
    torch.manual_seed(0)
    model = FreesiaForPreTraining(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(2, 128, (2, 12))
    out = model(ids, labels=ids, mode=MODE_LM)
    out.loss.backward()
    labels = torch.full_like(ids, -100)
    masked = ids.clone()
    masked[:, 5] = 1
    labels[:, 5] = ids[:, 5]
    out2 = model(masked, labels=labels, mode=MODE_MASKED)
    assert torch.isfinite(out.loss) and torch.isfinite(out2.loss)


def test_save_and_load_with_auto_classes():
    torch.manual_seed(0)
    model = FreesiaForPreTraining(tiny_config()).eval()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert AutoConfig.from_pretrained(d).model_type == "freesia"
        enc = AutoModel.from_pretrained(d).eval()
        ids = torch.randint(2, 128, (1, 6))
        torch.testing.assert_close(
            enc(ids, mode=MODE_LM).last_hidden_state, model.model(ids, mode=MODE_LM).last_hidden_state
        )


def test_from_pretrained_directly_on_model_class():
    torch.manual_seed(0)
    model = FreesiaForPreTraining(tiny_config()).eval()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        enc = FreesiaModel.from_pretrained(d).eval()
        again = FreesiaForPreTraining.from_pretrained(d).eval()
        ids = torch.randint(2, 128, (1, 6))
        ref = model.model(ids, mode=MODE_LM).last_hidden_state
        torch.testing.assert_close(enc(ids, mode=MODE_LM).last_hidden_state, ref)
        torch.testing.assert_close(again.model(ids, mode=MODE_LM).last_hidden_state, ref)


def test_learned_gate_gets_gradient_even_when_floor_dominates():
    model = FreesiaModel(tiny_config(bloom_gate_init=-8.0)).train()
    model.bloom_override = BLOOM_LEARNED
    model.bloom_floor = 0.9  # floor far above sigmoid(-8)
    ids = torch.randint(2, 128, (2, 10))
    model.encode(ids, torch.ones_like(ids)).sum().backward()
    grad = model.layers[1].self_attn.bloom_logit.grad
    assert grad is not None and torch.any(grad != 0)
    eff = model.bloom_gates(effective=True)
    assert torch.all(eff >= 0.9 - 1e-6)


def test_pooling_modes_mean_and_last():
    model = _model()
    ids = torch.randint(2, 128, (2, 10))
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    petal = model.encode(ids, mask)  # petal starts equal to mean pooling
    model.config.pooling_mode = "mean"
    torch.testing.assert_close(model.encode(ids, mask), petal, atol=1e-5, rtol=1e-5)
    model.config.pooling_mode = "last"
    hidden = model(ids, mask, mode=MODE_EMBED).last_hidden_state
    expected = F.normalize(model.petal_pooling.norm(torch.stack([hidden[0, 9], hidden[1, 6]])).float(), dim=-1)
    torch.testing.assert_close(model.encode(ids, mask), expected, atol=1e-5, rtol=1e-5)
