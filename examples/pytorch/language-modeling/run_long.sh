#!/bin/bash
# Freesia long contrastive check (design §6 follow-up): LM-pretrained 100M, full supervised data (1 epoch),
# Bloom learned (retuned) vs closed vs open. Each run = contrastive -> quick eval; re-running resumes. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
for R in ${RUNS:-long-bloom long-closed long-open}; do
  NAME=freesia-100m-lm-$R-probe
  OUT=artifacts/runs/$NAME
  if [ ! -f "$OUT/DONE" ]; then
    echo "[$(date '+%F %T')] probe $R"
    $PY run_freesia_contrastive.py --config configs/$NAME.yaml 2>&1 | tee -a "$LOG/probe-100m-lm-$R.log"
  fi
  if [ ! -f "$OUT/jmteb_lite.json" ]; then
    echo "[$(date '+%F %T')] eval $R"
    $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-100m-lm-$R.log"
  fi
done
echo "[$(date '+%F %T')] LONG_FINISHED"
