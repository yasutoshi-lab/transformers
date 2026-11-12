#!/bin/bash
# Freesia stage 4 v2 ablation: 4 short runs (300 steps each) from stage 3 v2 -> JMTEB-lite + English MTEB subset.
# Re-running skips finished parts (each run resumes from its checkpoint-latest). Run inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}
for n in base a-retrieval b-ret-nli c-lowlr; do
  OUT=artifacts/runs/abl4-$n
  if [ ! -f "$OUT/DONE" ]; then
    echo "[$(date '+%F %T')] train $n"
    $PY run_freesia_contrastive.py --config configs/abl4-$n.yaml 2>&1 | tee -a "$LOG/abl4-$n.log"
  fi
  if [ ! -f "$OUT/jmteb_lite.json" ]; then
    echo "[$(date '+%F %T')] jmteb $n"
    $PY eval_freesia_jmteb_lite.py --model "$OUT/final" --out "$OUT/jmteb_lite.json" 2>&1 | tee -a "$LOG/abl4-$n-eval.log"
  fi
  if [ ! -f "artifacts/bench/abl4-$n-en.json" ]; then
    echo "[$(date '+%F %T')] bench-en $n"
    $PY bench_embed_models.py --backend freesia --model "$OUT/final" --langs en --out "artifacts/bench/abl4-$n-en.json" 2>&1 | tee -a "$LOG/abl4-$n-eval.log"
  fi
  echo "[$(date '+%F %T')] RUN_DONE $n"
done
echo "[$(date '+%F %T')] ABLATION_FINISHED"
