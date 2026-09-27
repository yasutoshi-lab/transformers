#!/bin/bash
# Stage 0 smoke test for the contrastive and evaluation scripts (random-init 100M).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
$PY - <<'PY'
from transformers import AutoTokenizer, FreesiaConfig, FreesiaForPreTraining
t = AutoTokenizer.from_pretrained("artifacts/tokenizer/v32k")
m = FreesiaForPreTraining(FreesiaConfig(vocab_size=len(t), num_hidden_layers=8, pad_token_id=t.pad_token_id,
    eos_token_id=t.eos_token_id, bos_token_id=t.bos_token_id, mask_token_id=t.mask_token_id))
m.save_pretrained("artifacts/runs/smoke-init")
print("saved", len(t))
PY
$PY run_freesia_contrastive.py --config configs/smoke-probe.yaml
$PY eval_freesia_jmteb_lite.py --model artifacts/runs/smoke-probe/final --out artifacts/runs/smoke-probe/jmteb_lite.json
echo SMOKE_CT_OK
