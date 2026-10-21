#!/usr/bin/env bash
# Llama-3.2 1B/3B の SFT 比較の一連の処理（ws1-arc で 1 つのサービスとして実行する）。
#   1. 1B の SFT → 2. 3B の SFT → 3. 事前学習モデルと SFT 後のベンチマーク（CyberMetric を含む）
#   → 4. SFT 後のチャット形式の評価（GSM8K・HumanEval × 思考なし／思考あり）
# 終わった段階は artifacts-cmp/pipeline/<段階>.done に記録し、再実行すると続きから始める。
# 学習が途中で止まった場合は、最新のチェックポイントから再開する。
# Usage: bash llama_eval/run_comparison_pipeline.sh
set -u
cd "$(dirname "$0")/.."
export ART=artifacts-cmp CYBERMETRIC=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_ALLOW_CODE_EVAL=1
PY=../../../.venv/bin/python
DONE=$ART/pipeline
mkdir -p $DONE $ART/logs
step() { echo "[pipeline] $(date '+%F %T') $1"; }

run_stage() {  # run_stage <name> <command...>
  local name=$1; shift
  if [ -f $DONE/$name.done ]; then step "$name already done, skip"; return 0; fi
  step "$name START"
  local t0=$(date +%s)
  "$@" || { step "$name FAILED (exit $?)"; exit 1; }
  echo "$(date '+%F %T') elapsed_sec=$(( $(date +%s) - t0 ))" > $DONE/$name.done
  step "$name END ($(( $(date +%s) - t0 )) s)"
}

train() {  # train <size>
  local cfg=configs/llama-3.2-$1-sft-comparison.yaml
  local ckpt=$(ls -d $ART/checkpoints/llama-3.2-$1-sft-comparison/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  $PY run_llama_reasoning_sft.py --config $cfg ${ckpt:+--resume $ckpt} >> $ART/logs/sft-$1.log 2>&1
}

run_stage sft-1b train 1b
run_stage sft-3b train 3b
for s in 1b 3b; do
  run_stage bench-base-$s bash llama_eval/run_bench.sh base-$s meta-llama/Llama-3.2-${s^^}
  run_stage bench-sft-$s bash llama_eval/run_bench.sh sft-$s $ART/models/llama-3.2-$s-sft-comparison
  run_stage chat-sft-$s bash llama_eval/run_chat_evals.sh sft-$s $ART/models/llama-3.2-$s-sft-comparison
done
step "ALL DONE"
