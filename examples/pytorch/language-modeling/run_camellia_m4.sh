#!/usr/bin/env bash
# =============================================================================
# Camellia M4 オークストレーション（設計書 §7: MFU 計測 + checkpoint resume 検証）
#
#   Run A: step 0 → 1000（checkpoint-1000 保存完了後に torchrun へ SIGTERM）
#   Run B: --resume 最新 checkpoint → 2000
#
# 冪等: 中断後に再実行すると既存 checkpoint から継続する。
# 完了時は /tmp/camellia-m4-orch.log に [m4] M4 COMPLETE を書いて終了。
#
# 起動（tmux）:
#   tmux new -d -s camellia-m4 "bash /home/gotech/python-files/transformers/examples/pytorch/language-modeling/run_camellia_m4.sh"
# =============================================================================
set -u
cd /home/gotech/python-files/transformers/examples/pytorch/language-modeling
PY=/home/gotech/python-files/transformers/.venv/bin
ORCH_LOG=/tmp/camellia-m4-orch.log
OUT=artifacts-1/checkpoints/camellia-3b-m4
CFG=configs/camellia-3b-m4.yaml
export WANDB_PROJECT=camellia
export WANDB_DIR=$PWD
# 64 論理コア（32 物理）÷ 2 rank。データローダ worker は memmap スライスのみ
# なので 16 スレッドで十分（過剰订阅 → gloo 同期・MFU 計測を劣化させない）
export OMP_NUM_THREADS=16

log() { echo "[m4] $(date '+%F %T') $*" | tee -a "$ORCH_LOG"; }

latest_ckpt() {
  ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1
}

# --- 冪等ガード: 既に完了なら即終了 ---
if [ -f "$OUT/checkpoint-2000/trainer_state.json" ]; then
  log "M4 COMPLETE (checkpoint-2000 exists)"
  exit 0
fi

# --- Run A: 0 → 1000、checkpoint-1000 保存完了で停止 ---
if [ ! -f "$OUT/checkpoint-1000/trainer_state.json" ]; then
  log "=== run A: step 0 -> 1000 (ckpt-1000 保存完了で SIGTERM) ==="
  WANDB_NAME="camellia-3b-m4-runA" \
    "$PY/torchrun" --nproc_per_node=2 run_camellia_pretrain.py \
    --config "$CFG" > /tmp/camellia-m4-runA.log 2>&1 &
  MAIN=$!
  (
    sent=0
    while true; do
      if [ -f "$OUT/checkpoint-1000/trainer_state.json" ]; then
        sleep 30  # 保存フラッシュ完了を待つ（trainer_state は保存の末尾に書かれる）
        pkill -TERM -P "$MAIN" 2>/dev/null
        kill -TERM "$MAIN" 2>/dev/null
        sent=1
        log "checkpoint-1000 保存完了 → run A を停止 (SIGTERM)"
        break
      fi
      kill -0 "$MAIN" 2>/dev/null || break
      sleep 30
    done
    if [ "$sent" = 1 ]; then
      # torchrun の优雅 shutdown 待ち（最大 10 分）、過ぎたら強制
      for _ in $(seq 1 60); do
        kill -0 "$MAIN" 2>/dev/null || break
        sleep 10
      done
      if kill -0 "$MAIN" 2>/dev/null; then
        pkill -KILL -P "$MAIN" 2>/dev/null
        kill -KILL "$MAIN" 2>/dev/null
        log "run A が TERM 後に退出せず → SIGKILL"
      fi
    fi
  ) &
  WATCH=$!
  wait "$MAIN"
  A_RC=$?
  kill "$WATCH" 2>/dev/null
  log "run A 終了 (rc=$A_RC)"
else
  log "checkpoint-1000 既存在 → run A をスキップ（中断からの再開）"
fi

# --- Run B: 最新 checkpoint から resume → 2000 ---
LATEST=$(latest_ckpt)
if [ -n "$LATEST" ]; then
  log "=== run B: resume checkpoint-$LATEST -> 2000 ==="
  WANDB_NAME="camellia-3b-m4-runB" \
    "$PY/torchrun" --nproc_per_node=2 run_camellia_pretrain.py \
    --config "$CFG" --resume "$PWD/$OUT/checkpoint-$LATEST" \
    > /tmp/camellia-m4-runB.log 2>&1
  B_RC=$?
  log "run B 終了 (rc=$B_RC)"
  if [ -f "$OUT/checkpoint-2000/trainer_state.json" ]; then
    grep -E "tokens_per_sec|mfu" /tmp/camellia-m4-runB.log 2>/dev/null | tail -5 >> "$ORCH_LOG"
    log "M4 COMPLETE"
    exit 0
  fi
  log "M4 未完: checkpoint-2000 なし — runB.log を確認し本スクリプトを再実行して再開"
  exit 1
else
  log "run A が checkpoint 保存前に終了 (rc=$A_RC) — /tmp/camellia-m4-runA.log を確認"
  exit 1
fi
