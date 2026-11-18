#!/bin/bash
# freesia-supervised-cl 本番: 生成（ws2+ws1）→ ws2 の vLLM を止めてクエリ埋め込み・負例候補 → vLLM を戻して判定（ws2+ws1）→ ビルド
# 各段は再開可能。途中で止まったらこのスクリプトをもう一度動かす（済んだ段は飛ばす）。
set -euo pipefail
W=~/freesia-work; S=$W/scripts; PY=~/python_project/transformers/.venv/bin/python
cd $W
wait_vllm() { until curl -sf localhost:8000/v1/models >/dev/null && curl -sf localhost:8001/v1/models >/dev/null; do sleep 20; done; }
echo "[$(date '+%F %T')] start"
if [ ! -f sup-cl/full_gen.done ]; then
  docker start vllm-gemma4-ws2 >/dev/null; wait_vllm
  python3 -I $S/supcl_gen.py sup-cl/full_gen.jsonl 0 128 8000,8001 validation,test,train
  python3 -I $S/supcl_gen.py sup-cl/full_gen.jsonl 0 128 8000,8001 validation,test,train  # 失敗した行をもう一度
  touch sup-cl/full_gen.done
fi
echo "[$(date '+%F %T')] gen done"
if [ ! -f sup-cl/full_mine.done ]; then
  docker stop vllm-gemma4-ws2 >/dev/null
  $PY $S/supcl_embed.py query sup-cl/full_gen.jsonl
  $PY $S/supcl_mine.py sup-cl/full_gen.jsonl sup-cl/full_mine.jsonl
  touch sup-cl/full_mine.done
fi
echo "[$(date '+%F %T')] mine done"
if [ ! -f sup-cl/full_judge.done ]; then
  docker start vllm-gemma4-ws2 >/dev/null; wait_vllm
  python3 -I $S/supcl_judge.py sup-cl/full_gen.jsonl sup-cl/full_mine.jsonl sup-cl/full_judge.jsonl 64 8000,8001
  python3 -I $S/supcl_judge.py sup-cl/full_gen.jsonl sup-cl/full_mine.jsonl sup-cl/full_judge.jsonl 64 8000,8001
  touch sup-cl/full_judge.done
fi
echo "[$(date '+%F %T')] judge done"
rm -rf sup-cl/hf
$PY $S/supcl_build.py sup-cl/full_gen.jsonl sup-cl/full_mine.jsonl sup-cl/full_judge.jsonl sup-cl/hf
echo "[$(date '+%F %T')] FULL_DONE"
