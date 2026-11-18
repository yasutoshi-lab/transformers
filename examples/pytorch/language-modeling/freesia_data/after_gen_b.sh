#!/bin/bash
# B 案: 生成の完了を待って vLLM（ws2・ws1）を止め、後処理を始める
cd ~/freesia-work
until grep -q "\"new\":" logs/gen_b.log; do sleep 60; done
echo "[$(date "+%F %T")] gen finished: $(tail -1 logs/gen_b.log)" >> logs/weakcl_b_post.log
docker stop vllm-gemma4-ws2 >> logs/weakcl_b_post.log 2>&1
ssh ws1 docker stop vllm-gemma4-ws1 >> logs/weakcl_b_post.log 2>&1
tmux kill-window -t freesia:ws1tun
scripts/run_b_post.sh
