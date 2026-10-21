#!/usr/bin/env bash
# SFT 後のモデルのチャット形式の評価（GSM8K・HumanEval × 思考なし／思考あり）。
# 思考なしは greedy、思考ありは temperature 0.6 / top_p 0.95 のサンプリング（Qwen3・DeepSeek-R1 のモデルカードの推奨値）。
# 生成上限は GSM8K が 思考なし 2,048 / 思考あり 4,096、HumanEval が 思考なし 1,024 / 思考あり 2,048。
# Usage: bash llama_eval/run_chat_evals.sh <name> <model_path>
set -u
cd "$(dirname "$0")/.."
NAME=$1
M=$2
OUT=artifacts-llama/eval/$NAME
mkdir -p $OUT artifacts-llama/cache
LMEVAL=artifacts-llama/lm-eval-venv/bin/lm_eval
TASKS=llama_eval/tasks
export HF_ALLOW_CODE_EVAL=1
THINK="pretrained=$M,dtype=bfloat16,enable_thinking=True,think_end_token=</think>"
NOTHINK="pretrained=$M,dtype=bfloat16,enable_thinking=False"
SAMPLE="do_sample=True,temperature=0.6,top_p=0.95"
step() { echo "[chat:$NAME] $(date '+%F %T') $1"; }

step "gsm8k-chat-nothink START"; $LMEVAL --model hf --model_args "$NOTHINK" --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn --gen_kwargs max_gen_toks=2048 --batch_size 8 --seed 1234 --use_cache artifacts-llama/cache/$NAME-gsm8k-chat-nothink --log_samples --output_path $OUT/gsm8k-chat-nothink > $OUT/gsm8k-chat-nothink.log 2>&1
step "gsm8k-chat-think START"; $LMEVAL --model hf --model_args "$THINK" --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn --gen_kwargs $SAMPLE,max_gen_toks=4096 --batch_size 8 --seed 1234 --use_cache artifacts-llama/cache/$NAME-gsm8k-chat-think --log_samples --output_path $OUT/gsm8k-chat-think > $OUT/gsm8k-chat-think.log 2>&1
step "humaneval-chat-nothink START"; $LMEVAL --model hf --model_args "$NOTHINK" --include_path $TASKS --tasks humaneval_instruct_nothink --apply_chat_template --batch_size 16 --seed 1234 --confirm_run_unsafe_code --log_samples --output_path $OUT/humaneval-chat-nothink > $OUT/humaneval-chat-nothink.log 2>&1
step "humaneval-chat-think START"; $LMEVAL --model hf --model_args "$THINK" --include_path $TASKS --tasks humaneval_instruct_think --apply_chat_template --gen_kwargs $SAMPLE,max_gen_toks=2048 --batch_size 16 --seed 1234 --confirm_run_unsafe_code --log_samples --output_path $OUT/humaneval-chat-think > $OUT/humaneval-chat-think.log 2>&1
step "END"
echo "[chat:$NAME] EXIT=0"
