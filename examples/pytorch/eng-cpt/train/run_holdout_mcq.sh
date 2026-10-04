#!/usr/bin/env bash
# CPT で一度も読ませていない範囲（holdout_ppl）から作った 4 択で、主実験 4 条件を評価する。
# qa_eval 由来の 4 択での CPT の効果が「読ませた本の暗記」か「知識の一般化」かを切り分ける。
# 前提: ENG_CPT_QA_DIR=artifacts/qa_holdout ENG_CPT_MCQ_SPLIT=holdout_ppl で
#       python -m qagen.build_qa --steps mcq rebalance manifest を実行済み
# 使い方（eng-cpt/ 直下）: bash train/run_holdout_mcq.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TRANSFORMERS_VERBOSITY=error
PY=../../../.venv/bin/python
MCQ=artifacts/qa_holdout/mcq_eval.jsonl
OUT=artifacts/eval/holdout_mcq
RUNS=artifacts/runs
mkdir -p "$OUT"
run() {
  local name=$1; shift
  if [[ -f "$OUT/$name.json" ]]; then echo "[$(date '+%F %T')] skip $name"; return; fi
  echo "[$(date '+%F %T')] eval $name"
  "$PY" -m train.eval_lm --name "$name" --output "$OUT/$name.json" --skip-ppl --mcq-file "$MCQ" "$@"
}
run base
run cpt-f100-ep3 --adapters "$RUNS/cpt-f100/final"
run sft-base-ep2 --adapters "$RUNS/sft-base/final"
run sft-cpt-f100-ep2 --adapters "$RUNS/cpt-f100/final" "$RUNS/sft-cpt-f100/final"
echo "[$(date '+%F %T')] all done"
