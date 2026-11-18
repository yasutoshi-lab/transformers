#!/bin/bash
# ws1 が止まったら（8001 が 2 分続けて応答しない）、判定を ws2（8000）だけに切り替える
cd ~/freesia-work
fail=0
while true; do
  if curl -sf -m5 localhost:8001/v1/models >/dev/null; then fail=0; else fail=$((fail+1)); fi
  [ $fail -ge 4 ] && break
  sleep 30
done
echo "[$(date "+%F %T")] ws1 down -> switch to ws2 only" | tee -a logs/supcl_full.log
pkill -f "scripts/run_supcl_full.sh" ; sleep 1; pkill -f "supcl_judge.py"; sleep 5
tmux kill-window -t freesia:ws1tun 2>/dev/null
f=sup-cl/full_judge.jsonl
python3 - "$f" <<"PY"
import sys
p=sys.argv[1]; b=open(p,"rb").read()
if b and not b.endswith(b"\n"):
    open(p,"wb").write(b[:b.rfind(b"\n")+1]); print("trimmed partial line")
PY
tmux kill-window -t freesia:supfull 2>/dev/null
tmux new-window -d -t freesia -n supfull "bash -lc \"scripts/run_supcl_full_ws2.sh 2>&1 | tee -a logs/supcl_full.log\"; exec bash"
echo "[$(date "+%F %T")] restarted with run_supcl_full_ws2.sh" | tee -a logs/supcl_full.log
