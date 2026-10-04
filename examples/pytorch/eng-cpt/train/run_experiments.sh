#!/usr/bin/env bash
# eng-cpt の全実験を GPU0 で順に実行する（ws3-arc 想定。eng-cpt/ 直下で実行）。
#
#   主実験（2×2）: Base / Base+CPT / Base+SFT / Base+CPT+SFT
#   副実験（量の曲線）: CPT の train 使用率 25% / 50% / 100% × 1〜3 エポック
#
# 各条件の学習は 1 回のみ（Curator 判断）。ばらつきは評価側のブートストラップで示す。
# 完了済みの run・評価は飛ばすので、中断しても同じコマンドで再開できる。
#
# 使い方:
#   tmux new-session -d -s eng-cpt-exp "bash train/run_experiments.sh 2>&1 | tee -a artifacts/runs/experiments.log"
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_VERBOSITY=error
PY=../../../.venv/bin/python
RUNS=artifacts/runs
EVAL=artifacts/eval
mkdir -p "$RUNS" "$EVAL"

CPT_EPOCHS=3
SFT_EPOCHS=2

log() { echo "[$(date '+%F %T')] $*"; }

# 学習（完了済み = <dir>/run_summary.json がある）
train() {
  local name=$1; shift
  if [[ -f "$RUNS/$name/run_summary.json" ]]; then log "skip train $name"; return; fi
  log "train $name: $*"
  "$PY" -m train.train_lora --output-dir "$RUNS/$name" "$@"
}

# 評価（完了済み = 出力 JSON がある）
evaluate() {
  local name=$1; shift
  if [[ -f "$EVAL/$name.json" ]]; then log "skip eval $name"; return; fi
  log "eval $name: $*"
  "$PY" -m train.eval_lm --name "$name" --output "$EVAL/$name.json" --jmmlu "$@"
}

# エポックごとの checkpoint を順に評価する（checkpoint-<step> をステップ数の昇順に ep1, ep2, ...）
# 注意: パスに "-" を含む（cpt-f100 等）ので、区切り文字ではなく末尾の数字だけで数値ソートする
evaluate_epochs() {
  local run=$1; shift
  local ep=0
  for ckpt in $(ls -d "$RUNS/$run"/checkpoint-* | sed -E 's/.*checkpoint-([0-9]+)$/\1 &/' | sort -n | cut -d' ' -f2); do
    ep=$((ep + 1))
    log "map $run-ep$ep -> $ckpt"
    evaluate "$run-ep$ep" --adapters "$@" "$ckpt"
  done
}

evaluate base

# ---- CPT（量の曲線を兼ねる） ----
for frac in 100 50 25; do
  train "cpt-f$frac" --mode cpt --train-fraction "$(echo "scale=2; $frac/100" | bc)" --epochs "$CPT_EPOCHS"
  evaluate_epochs "cpt-f$frac"
done

# ---- SFT（Base+SFT と Base+CPT+SFT） ----
train sft-base --mode sft --epochs "$SFT_EPOCHS" --batch-size 8 --grad-accum 2
evaluate_epochs sft-base

train sft-cpt-f100 --mode sft --epochs "$SFT_EPOCHS" --batch-size 8 --grad-accum 2 \
  --merge-adapter "$RUNS/cpt-f100/final"
evaluate_epochs sft-cpt-f100 "$RUNS/cpt-f100/final"

log "all done"
