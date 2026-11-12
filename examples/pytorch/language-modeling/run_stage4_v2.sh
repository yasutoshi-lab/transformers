#!/bin/bash
# Freesia stage 4 v2 (design §6): supervised contrastive training of Freesia-300M v2
# (yasutoshi-lab/freesia-supervised-cl train, ~1.59M rows, from stage 3 v2) -> quick eval. Re-running resumes. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
OUT=artifacts/runs/freesia-300m-sup
if [ ! -f "$OUT/DONE" ]; then
  echo "[$(date '+%F %T')] train 300m-sup"
  $PY run_freesia_contrastive.py --config configs/freesia-300m-sup-v2.yaml 2>&1 | tee -a "$LOG/stage4-300m-sup-v2.log"
fi
if [ ! -f "$OUT/jmteb_lite.json" ]; then
  echo "[$(date '+%F %T')] eval 300m-sup"
  $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-300m-sup-v2.log"
fi
echo "[$(date '+%F %T')] STAGE4_FINISHED"
