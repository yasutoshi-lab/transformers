#!/bin/bash
# freesia-supervised-cl 試作の後半: プール埋め込みの完了を待ち、クエリ埋め込み → 負例候補 → 偽の負例の判定（ws1）→ ビルド
set -euo pipefail
W=~/freesia-work; S=$W/scripts; PY=~/python_project/transformers/.venv/bin/python
cd $W
until grep -q EMBED_DONE logs/supcl_embed_pool.log; do grep -q Traceback logs/supcl_embed_pool.log && exit 1; sleep 60; done
$PY $S/supcl_embed.py query sup-cl/pilot_gen.jsonl
$PY $S/supcl_mine.py sup-cl/pilot_gen.jsonl sup-cl/pilot_mine.jsonl
python3 -I $S/supcl_judge.py sup-cl/pilot_gen.jsonl sup-cl/pilot_mine.jsonl sup-cl/pilot_judge.jsonl 64 8001
rm -rf sup-cl/pilot_hf
$PY $S/supcl_build.py sup-cl/pilot_gen.jsonl sup-cl/pilot_mine.jsonl sup-cl/pilot_judge.jsonl sup-cl/pilot_hf
echo PILOT_DONE
