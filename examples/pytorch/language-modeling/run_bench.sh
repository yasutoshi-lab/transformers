#!/bin/bash
# Compare Freesia-300m-Embedding with similar-size embedding models (ja: JMTEB-lite, en: MTEB subset). Re-running skips finished models.
set -uo pipefail
cd "$(dirname "$0")"
OUT=artifacts/bench
mkdir -p "$OUT"
LOG=${FREESIA_LOG_DIR:-$HOME/freesia-work/logs}/bench.log
if [ ! -f "$OUT/freesia-300m-embedding.json" ]; then
  echo "[$(date "+%F %T")] bench freesia" | tee -a "$LOG"
  ../../../.venv/bin/python bench_embed_models.py --backend freesia --model artifacts/runs/freesia-300m-sup/final --out "$OUT/freesia-300m-embedding.json" 2>&1 | tee -a "$LOG"
fi
for m in intfloat/multilingual-e5-base cl-nagoya/ruri-v3-310m google/embeddinggemma-300m ibm-granite/granite-embedding-278m-multilingual; do
  f="$OUT/$(basename $m).json"
  [ -f "$f" ] && continue
  echo "[$(date "+%F %T")] bench $m" | tee -a "$LOG"
  ~/freesia-work/bench-venv/bin/python bench_embed_models.py --backend st --model "$m" --out "$f" 2>&1 | tee -a "$LOG"
done
echo "[$(date "+%F %T")] BENCH_ALL_FINISHED" | tee -a "$LOG"
