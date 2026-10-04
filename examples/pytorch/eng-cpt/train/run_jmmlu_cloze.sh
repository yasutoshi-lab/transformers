#!/usr/bin/env bash
# 主実験 4 条件の JMMLU を cloze 方式で評価する（記号回答の形式崩れと知識の忘却を切り分ける）。
# 使い方（eng-cpt/ 直下）: bash train/run_jmmlu_cloze.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TRANSFORMERS_VERBOSITY=error
PY=../../../.venv/bin/python
OUT=artifacts/eval/jmmlu_cloze
RUNS=artifacts/runs
mkdir -p "$OUT"
run() {
  local name=$1; shift
  if [[ -f "$OUT/$name.json" ]]; then echo "[$(date '+%F %T')] skip $name"; return; fi
  echo "[$(date '+%F %T')] eval $name"
  "$PY" -m train.eval_lm --name "$name" --output "$OUT/$name.json" --skip-ppl --skip-mcq --jmmlu-cloze "$@"
}
run base
run cpt-f100-ep3 --adapters "$RUNS/cpt-f100/final"
run sft-base-ep2 --adapters "$RUNS/sft-base/final"
run sft-cpt-f100-ep2 --adapters "$RUNS/cpt-f100/final" "$RUNS/sft-cpt-f100/final"
echo "[$(date '+%F %T')] all done"
