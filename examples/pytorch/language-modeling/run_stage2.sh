#!/bin/bash
# Freesia stage 2 (design §6): Freesia-300M causal-LM pretraining (~6B tokens, ~4 days on ws2-arc A6000).
# Resumable: re-running continues from artifacts/runs/freesia-300m-lm/checkpoint-latest. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
if [ ! -f artifacts/runs/freesia-300m-lm/DONE ]; then
  echo "[$(date '+%F %T')] pretrain 300m-lm"
  $PY run_freesia_pretrain.py --config configs/freesia-300m-lm.yaml 2>&1 | tee -a "$LOG/pretrain-300m-lm.log"
fi
echo "[$(date '+%F %T')] STAGE2_FINISHED"
