#!/bin/bash
# Freesia stage 4 v3: supervised contrastive training without NLI (retrieval + classification),
# lr 1e-5, focal 0, from stage 3 v2 -> JMTEB-lite + English bench. Re-running resumes. Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
mkdir -p "$LOG"
OUT=artifacts/runs/freesia-300m-sup-v3
if [ ! -f "$OUT/DONE" ]; then
  echo "[$(date '+%F %T')] train 300m-sup-v3"
  $PY run_freesia_contrastive.py --config configs/freesia-300m-sup-v3.yaml 2>&1 | tee -a "$LOG/stage4-300m-sup-v3.log"
fi
if [ ! -f "$OUT/jmteb_lite.json" ]; then
  echo "[$(date '+%F %T')] eval 300m-sup-v3"
  $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/eval-300m-sup-v3.log"
fi
if [ ! -f artifacts/bench/freesia-300m-sup-v3-en.json ]; then
  echo "[$(date '+%F %T')] bench-en 300m-sup-v3"
  $PY bench_embed_models.py --backend freesia --model "$OUT/final" --langs en --out artifacts/bench/freesia-300m-sup-v3-en.json 2>&1 | tee -a "$LOG/bench-sup-v3.log"
fi
echo "[$(date '+%F %T')] STAGE4_FINISHED"
