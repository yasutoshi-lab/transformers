#!/bin/bash
# Freesia stage 1a (design §6): 100M x 3 arms (lm / masked / janus), each = pretrain -> probe -> quick eval.
# Every step is resumable and skipped when its DONE marker exists, so re-running this script after an
# interruption continues where it stopped. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
for ARM in ${ARMS:-lm masked janus}; do
  RUN=artifacts/runs/freesia-100m-$ARM
  if [ ! -f "$RUN/DONE" ]; then
    echo "[$(date '+%F %T')] pretrain $ARM"
    $PY run_freesia_pretrain.py --config configs/freesia-100m-$ARM.yaml 2>&1 | tee -a "$LOG/pretrain-100m-$ARM.log"
  fi
  PROBE=artifacts/runs/freesia-100m-$ARM-probe
  if [ ! -f "$PROBE/DONE" ]; then
    echo "[$(date '+%F %T')] probe $ARM"
    $PY run_freesia_contrastive.py --config configs/freesia-100m-$ARM-probe.yaml 2>&1 | tee -a "$LOG/probe-100m-$ARM.log"
  fi
  if [ ! -f "$PROBE/jmteb_lite.json" ]; then
    echo "[$(date '+%F %T')] eval $ARM"
    $PY eval_freesia_jmteb_lite.py --model "$PROBE/final" --out "$PROBE/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-100m-$ARM.log"
  fi
done
echo "[$(date '+%F %T')] STAGE1A_FINISHED"
