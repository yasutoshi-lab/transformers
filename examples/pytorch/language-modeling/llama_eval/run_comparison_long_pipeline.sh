#!/usr/bin/env bash
# Llama-3.2 1B/3B の SFT 比較の対照実験（長い思考版、-long）を ws1-arc で 1 つのサービスとして流す。
#   0. 思考ありの生成上限を 4,096 に統一したので、短い思考版の HumanEval・CyberMetric-2000（思考あり）を測り直す
#      （上限 2,048 の結果は *-cap2048 に残す）
#   1. 1B の SFT → 2. 3B の SFT（データ以外は短い思考版と同じ設定）
#   3. SFT 後のベンチマーク（同じ方式）→ チャット形式（GSM8K・HumanEval）→ CyberMetric-2000（対数尤度、チャット形式）
# 事前学習モデルの評価は、短い思考版のときに同じ環境・設定で測った結果を使う。
# 終わった段階は artifacts-cmp/pipeline/<段階>.done に記録し、再実行すると続きから始める。
# Usage: bash llama_eval/run_comparison_long_pipeline.sh
set -u
cd "$(dirname "$0")/.."
export ART=artifacts-cmp CYBERMETRIC=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_ALLOW_CODE_EVAL=1
PY=../../../.venv/bin/python
DONE=$ART/pipeline
mkdir -p $DONE $ART/logs
step() { echo "[pipeline-long] $(date '+%F %T') $1"; }

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
  # The data is built elsewhere and copied in; wait for the marker written after the copy is verified.
  until [ -f $ART/sft-data-long/READY ]; do sleep 60; done
  local cfg=configs/llama-3.2-$1-sft-comparison-long.yaml
  local ckpt=$(ls -d $ART/checkpoints/llama-3.2-$1-sft-comparison-long/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  $PY run_llama_reasoning_sft.py --config $cfg ${ckpt:+--resume $ckpt} >> $ART/logs/sft-long-$1.log 2>&1
}

keep_cap2048() {  # keep_cap2048 <eval path without extension>
  [ -e $1 ] && [ ! -e $1-cap2048 ] && mv $1 $1-cap2048 && mv $1.log $1-cap2048.log
  return 0
}

recap_low() {
  for s in 1b 3b; do
    keep_cap2048 $ART/eval/sft-$s/humaneval-chat-think
    keep_cap2048 $ART/eval/sft-$s/cybermetric2000-chat-think-evaluator
    STEPS=humaneval-chat-think bash llama_eval/run_chat_evals.sh sft-$s $ART/models/llama-3.2-$s-sft-comparison || return 1
  done
  bash llama_eval/run_cybermetric_chat.sh think evaluator
}

cybermetric_long() {
  VARIANT=-long bash llama_eval/run_cybermetric_chat.sh nothink evaluator && VARIANT=-long bash llama_eval/run_cybermetric_chat.sh think evaluator
}

run_stage recap-low-think4096 recap_low
run_stage sft-long-1b train 1b
run_stage sft-long-3b train 3b
for s in 1b 3b; do
  run_stage bench-sft-long-$s bash llama_eval/run_bench.sh sft-long-$s $ART/models/llama-3.2-$s-sft-comparison-long
  run_stage chat-sft-long-$s bash llama_eval/run_chat_evals.sh sft-long-$s $ART/models/llama-3.2-$s-sft-comparison-long
done
run_stage cybermetric-long cybermetric_long
step "ALL DONE"
