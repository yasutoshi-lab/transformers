#!/usr/bin/env python
"""Upload the finished stage 4 model as the Freesia-300m-Embedding private repository.

段階4（教師あり対照学習・王道構成）の `final/` を、`yasutoshi-lab/Freesia-300m-Embedding`（**private**）へ
アップロードする（重み・tokenizer・学習ログ・評価結果・モデルカード）。完了したら `<run>/EMBEDDING_UPLOADED` を作る。
認証は HF の保存済みログイン（または環境変数 HF_TOKEN）を使い、トークンの値はどこにも書き出さない。

Usage:
    python upload_freesia_embedding.py --run artifacts/runs/freesia-300m-sup
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


CARD = """---
license: other
language: [ja, en]
tags: [freesia, embedding, sentence-similarity, research, private]
---

# Freesia-300m-Embedding

Japanese/English text embedding model built from scratch (Freesia-300M, ~292M parameters).
**Private research artifact** — not an official release.

- Lineage: causal LM pretraining (`freesia-300m-lm`) → weakly supervised contrastive (`freesia-300m-weak`,
  ruri-dataset-v2-pt 3M pairs) → supervised contrastive (`{run}`, this model)
- Configuration ("mainstream"): Bloom attention gate fixed **closed** (causal), **mean pooling**, 768 dims
- Query format: `Instruct: <task instruction>\\nQuery:<text>`; documents are encoded without a prefix
- Code: `yasutoshi-lab/transformers` branch `freesia` (`FreesiaModel`, `bloom_override="closed"`)
- Training data licenses vary (some sources are non-commercial or unverified); research use only.

JMTEB-lite quick evaluation (test splits, retrieval nDCG@10 / STS Spearman):

```json
{summary}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", default="yasutoshi-lab/Freesia-300m-Embedding")
    args = parser.parse_args()
    run = args.run
    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    summary = json.loads((run / "jmteb_lite.json").read_text())
    readme = run / "EMBEDDING_README.md"
    readme.write_text(CARD.format(run=run.name, summary=json.dumps(summary, ensure_ascii=False, indent=2)))
    api.upload_folder(repo_id=args.repo, folder_path=str(run / "final"), path_in_repo=".",
                      commit_message=f"Upload {run.name} final weights")
    for extra in ("train_log.jsonl", "jmteb_lite.json"):
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(run / extra), path_in_repo=extra,
                        commit_message=f"Add {extra}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=str(readme), path_in_repo="README.md",
                    commit_message="Add model card")
    (run / "EMBEDDING_UPLOADED").write_text(json.dumps({"repo": args.repo, "time": time.strftime("%F %T")}))
    print("[" + time.strftime("%F %T") + f"] EMBEDDING_UPLOADED {run.name} -> {args.repo}", flush=True)


if __name__ == "__main__":
    main()
