#!/bin/bash
# freesia-weak-cl B 案の後処理を通しで流す（生成の完了後に実行）。各段は出力があれば飛ばす。
set -euo pipefail
cd ~/freesia-work
PY=~/python_project/transformers/.venv/bin/python
B=weak-cl/b
LOG=logs/weakcl_b_post.log
echo "[$(date '+%F %T')] start" >> $LOG

# 1. 残す旧ペアと新しいペアを 1 本にする（旧ペアを先に置き、重複除去では旧ペアを残す）
if [ ! -s $B/pairs_raw.jsonl.gz ]; then
  $PY -I - <<'PY' >> $LOG 2>&1
import gzip, json, os
B = os.path.expanduser("~/freesia-work/weak-cl/b")
KEYS = ("id", "lang", "style", "source", "doc", "pt_id", "passage", "query")
n = 0
with gzip.open(f"{B}/pairs_raw.jsonl.gz.tmp", "wt") as g:
    for l in gzip.open(f"{B}/pairs_keep.jsonl.gz", "rt"):
        g.write(l); n += 1
    for l in open(f"{B}/queries_b.jsonl"):
        r = json.loads(l)
        g.write(json.dumps({k: r.get(k) for k in KEYS}, ensure_ascii=False) + "\n"); n += 1
os.replace(f"{B}/pairs_raw.jsonl.gz.tmp", f"{B}/pairs_raw.jsonl.gz")
print("MERGE_DONE", n, flush=True)
PY
fi

# 2. 規則・重複除去・汚染除去
[ -s $B/stats_a.json ] || $PY -I scripts/weakcl_rules.py $B/pairs_raw.jsonl.gz $B/pairs_a.jsonl.gz $B/stats_a.json >> $LOG 2>&1

# 3. 埋め込み（旧ペアは使い回す）→ 全体の一節に対する順位 → parquet
$PY -I scripts/weakcl_b_embed.py >> $LOG 2>&1
$PY -I scripts/weakcl_rank.py $HOME/freesia-work/$B/emb >> $LOG 2>&1
$PY -I scripts/weakcl_b_build.py >> $LOG 2>&1
echo "[$(date '+%F %T')] POST_DONE" >> $LOG
