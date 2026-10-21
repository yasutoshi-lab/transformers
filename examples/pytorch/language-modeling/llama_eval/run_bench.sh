#!/usr/bin/env bash
# Llama-3.2-1B（素）と SFT 後のベンチマーク一式。Ranunculus の run_sft_v4_evals.sh と同じタスク定義・設定に、
# MMLU（5-shot）と HumanEval（pass@1、生成コードを実行して採点）を加える。GPU 1 枚で順に実行する。
# Usage: bash llama_eval/run_bench.sh <name> <model_path_or_id>
# チャット形式の GSM8K・HumanEval は run_chat_evals.sh で測る。
# 環境変数: ART（成果物のディレクトリ、既定 artifacts-llama）、CYBERMETRIC=1 で CyberMetric-10000（5-shot）も測る。
set -u
cd "$(dirname "$0")/.."
NAME=$1
M=$2
ART=${ART:-artifacts-llama}
OUT=$ART/eval/$NAME
mkdir -p $OUT
LMEVAL=$ART/lm-eval-venv/bin/lm_eval
TASKS=llama_eval/tasks
export HF_ALLOW_CODE_EVAL=1
step() { echo "[bench:$NAME] $(date '+%F %T') $1"; }

step "bench1 START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --include_path $TASKS \
  --tasks leaderboard_mmlu_pro,jmmlu --batch_size auto --seed 1234 --log_samples --output_path $OUT/bench1 > $OUT/bench1.log 2>&1
step "bench2 START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --include_path $TASKS \
  --tasks lambada_openai,hellaswag,arc_easy,piqa,winogrande,xwinograd_jp,jcommonsenseqa_cloze --batch_size auto --seed 1234 --log_samples --output_path $OUT/bench2 > $OUT/bench2.log 2>&1
step "mmlu START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --tasks mmlu --num_fewshot 5 --batch_size auto --seed 1234 --log_samples --output_path $OUT/mmlu > $OUT/mmlu.log 2>&1
step "gsm8k-plain START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --tasks gsm8k --num_fewshot 5 --batch_size 32 --seed 1234 --log_samples --output_path $OUT/gsm8k-plain > $OUT/gsm8k-plain.log 2>&1
step "humaneval-plain START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --tasks humaneval --batch_size 32 --seed 1234 --confirm_run_unsafe_code --log_samples --output_path $OUT/humaneval-plain > $OUT/humaneval-plain.log 2>&1
if [ "${CYBERMETRIC:-0}" = 1 ]; then
  step "cybermetric START"; $LMEVAL --model hf --model_args pretrained=$M,dtype=bfloat16 --include_path $TASKS --tasks cybermetric_10000 --batch_size auto --seed 1234 --log_samples --output_path $OUT/cybermetric > $OUT/cybermetric.log 2>&1
fi
step "END"
echo "[bench:$NAME] EXIT=0"
