#!/bin/bash
# Freesia stage 4 (design §6): supervised contrastive training of Freesia-300M from the stage 3 model
# (mainstream: Bloom closed, mean pooling) -> quick eval -> upload as Freesia-300m-Embedding. Re-running resumes. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
OUT=artifacts/runs/freesia-300m-sup
if [ ! -f "$OUT/DONE" ]; then
  echo "[$(date "+%F %T")] train 300m-sup"
  $PY run_freesia_contrastive.py --config configs/freesia-300m-sup.yaml 2>&1 | tee -a "$LOG/stage4-300m-sup.log"
fi
if [ ! -f "$OUT/jmteb_lite.json" ]; then
  echo "[$(date "+%F %T")] eval 300m-sup"
  $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-300m-sup.log"
fi
if [ ! -f "$OUT/EMBEDDING_UPLOADED" ]; then
  echo "[$(date "+%F %T")] upload Freesia-300m-Embedding"
  $PY upload_freesia_embedding.py --run "$OUT" 2>&1 | tee -a "$LOG/upload-300m-sup.log"
fi
echo "[$(date "+%F %T")] STAGE4_FINISHED"
