#!/bin/bash
# Freesia stage 1b (design §6): ablations of the Freesia-specific parts on the LM-pretrained 100M,
# plus seed-43 repeats of the three stage-1a probes. Each run = contrastive probe -> quick eval.
# Every step is skipped when its output exists, so re-running continues after an interruption. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
RUNS=${RUNS:-"lm-1b-closed lm-1b-open lm-1b-floor lm-1b-mean lm-1b-last lm-s43 janus-s43 masked-s43"}
for R in $RUNS; do
  NAME=freesia-100m-$R-probe
  OUT=artifacts/runs/$NAME
  if [ ! -f "$OUT/DONE" ]; then
    echo "[$(date '+%F %T')] probe $R"
    $PY run_freesia_contrastive.py --config configs/$NAME.yaml 2>&1 | tee -a "$LOG/probe-100m-$R.log"
  fi
  if [ ! -f "$OUT/jmteb_lite.json" ]; then
    echo "[$(date '+%F %T')] eval $R"
    $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-100m-$R.log"
  fi
done
echo "[$(date '+%F %T')] STAGE1B_FINISHED"
