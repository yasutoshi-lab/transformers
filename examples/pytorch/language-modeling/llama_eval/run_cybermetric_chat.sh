#!/usr/bin/env bash
# CyberMetric-2000 の測り直し。事前学習モデルと SFT 後は対数尤度（5-shot、MMLU と同じ形式）、
# SFT 後はチャット形式（公式 README のプロンプト、<xml>X</xml> で採点）でも測る。
#   nothink: 対数尤度（事前学習モデル・SFT 後の 1B/3B）と、SFT 後 1B/3B のチャット形式・思考なし（greedy、上限 1,024）
#   think:   SFT 後 1B/3B のチャット形式・思考あり（temperature 0.6 / top_p 0.95、上限 4,096。2026-10-24 に 2,048 から統一）
#   FORMAT（第 2 引数）: readme（README のプロンプト、<xml>X</xml>）または evaluator（CyberMetric_evaluator.py の
#   プロンプト、ANSWER: X）。evaluator では対数尤度は測らない（readme の nothink で測り済み）
#   VARIANT（環境変数）: SFT 後のモデルの接尾辞（例 -long → models/llama-3.2-1b-sft-comparison-long、出力は eval/sft-long-1b）。
#   VARIANT を付けた nothink では、そのモデルの対数尤度も測る
# Usage: [VARIANT=-long] bash llama_eval/run_cybermetric_chat.sh nothink|think [readme|evaluator]
set -u
cd "$(dirname "$0")/.."
PHASE=$1
VARIANT=${VARIANT:-}
FORMAT=${2:-readme}
ART=artifacts-cmp
LMEVAL=$ART/lm-eval-venv/bin/lm_eval
TASKS=llama_eval/tasks
SYS=$'You are a helpful AI assistant.\nInstructions:\na. Carefully read the question.\nb. Choose the correct answer (A, B, C, or D) only.\nc. Do NOT include any explanation or additional text in the response.\nd. Always return the answer in this XML format: \'<xml>answer</xml>\'. For example, if the correct answer is D, then return <xml>D</xml>.\n\n'
if [ "$FORMAT" = evaluator ]; then
  SYS="You are a security expert who answers questions."; TASK=cybermetric_2000_chat_evaluator; SUFFIX=-evaluator
else
  TASK=cybermetric_2000_chat; SUFFIX=
fi
step() { echo "[cybermetric:$PHASE$SUFFIX] $(date '+%F %T') $1"; }

if [ "$PHASE" = nothink ] && [ "$FORMAT" = readme ]; then
  for s in 1b 3b; do
    for m in base-$s:meta-llama/Llama-3.2-${s^^} sft-$s:$ART/models/llama-3.2-$s-sft-comparison; do
      name=${m%%:*}; path=${m#*:}; out=$ART/eval/$name/cybermetric2000
      step "$name loglik START"
      $LMEVAL --model hf --model_args pretrained=$path,dtype=bfloat16 --include_path $TASKS --tasks cybermetric_2000 \
        --batch_size auto --seed 1234 --log_samples --output_path $out > $out.log 2>&1
    done
  done
fi
if [ "$PHASE" = nothink ] && [ -n "$VARIANT" ]; then
  for s in 1b 3b; do
    out=$ART/eval/sft$VARIANT-$s/cybermetric2000; mkdir -p $(dirname $out)
    step "sft$VARIANT-$s loglik START"
    $LMEVAL --model hf --model_args pretrained=$ART/models/llama-3.2-$s-sft-comparison$VARIANT,dtype=bfloat16 --include_path $TASKS \
      --tasks cybermetric_2000 --batch_size auto --seed 1234 --log_samples --output_path $out > $out.log 2>&1
  done
fi
for s in 1b 3b; do
  M=$ART/models/llama-3.2-$s-sft-comparison$VARIANT
  if [ "$PHASE" = nothink ]; then
    ARGS="pretrained=$M,dtype=bfloat16,enable_thinking=False"; GEN=""
  else
    ARGS="pretrained=$M,dtype=bfloat16,enable_thinking=True,think_end_token=</think>"
    GEN="--gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=4096"
  fi
  out=$ART/eval/sft$VARIANT-$s/cybermetric2000-chat-$PHASE$SUFFIX; mkdir -p $(dirname $out)
  step "sft$VARIANT-$s chat START"
  $LMEVAL --model hf --model_args "$ARGS" --include_path $TASKS --tasks $TASK --apply_chat_template \
    --system_instruction "$SYS" $GEN --batch_size 16 --seed 1234 --use_cache $ART/cache/sft$VARIANT-$s-cybermetric2000-$PHASE$SUFFIX-cap4096 \
    --log_samples --output_path $out > $out.log 2>&1
done
step "END"
