#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Upload finished Freesia runs to the Hugging Face Hub as private repositories.

`artifacts/runs/<run>/DONE` がある学習済みの run を、`yasutoshi-lab/<run>` の **private** リポジトリへ
アップロードする（重み・tokenizer・設定・学習ログ・評価結果）。アップロード済みの run は
`<run>/UPLOADED` を作って飛ばす。`--watch` を付けると一定間隔で繰り返す（tmux で常駐させる用途）。
認証は HF の保存済みログイン（または環境変数 HF_TOKEN）を使い、トークンの値はどこにも書き出さない。

Usage:
    python upload_freesia_to_hub.py --runs-dir artifacts/runs --watch 600
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
tags: [freesia, embedding, research, private]
---

# {repo}

Freesia research checkpoint (stage {stage}). **Private research artifact** — not an official release.

- Run: `{run}`
- Kind: {kind}
- Code: `yasutoshi-lab/transformers` branch `freesia` (`FreesiaModel` / `FreesiaForPreTraining`)
- Training data licenses vary (some sources are non-commercial or unverified); research use only.

Files: `final/` weights and tokenizer, `train_log.jsonl`, and (for probe runs) `jmteb_lite.json` quick evaluation.

```json
{summary}
```
"""


def upload_run(api: HfApi, run_dir: Path, owner: str) -> str:
    """Upload one finished run.

    Args:
        api (HfApi): Hub client.
        run_dir (Path): Run directory containing `final/` and `DONE`.
        owner (str): Hub namespace.

    Returns:
        str: Repository id.
    """
    repo = f"{owner}/{run_dir.name}"
    api.create_repo(repo, private=True, exist_ok=True)
    kind = "contrastive probe (embedding mode)" if run_dir.name.endswith("-probe") else "Janus-family pretraining"
    summary = {"done": json.loads((run_dir / "DONE").read_text())}
    if (run_dir / "jmteb_lite.json").exists():
        summary["jmteb_lite"] = json.loads((run_dir / "jmteb_lite.json").read_text())
    readme = run_dir / "HUB_README.md"
    readme.write_text(CARD.format(repo=repo, stage="1a", run=run_dir.name, kind=kind,
                                  summary=json.dumps(summary, ensure_ascii=False, indent=2)))
    api.upload_folder(repo_id=repo, folder_path=str(run_dir / "final"), path_in_repo=".",
                      commit_message=f"Upload {run_dir.name} final weights")
    for extra in ("train_log.jsonl", "jmteb_lite.json"):
        if (run_dir / extra).exists():
            api.upload_file(repo_id=repo, path_or_fileobj=str(run_dir / extra), path_in_repo=extra,
                            commit_message=f"Add {extra}")
    api.upload_file(repo_id=repo, path_or_fileobj=str(readme), path_in_repo="README.md", commit_message="Add model card")
    return repo


def ready(run_dir: Path) -> bool:
    if not (run_dir / "DONE").exists() or not (run_dir / "final").exists() or (run_dir / "UPLOADED").exists():
        return False
    # probe runs are uploaded only after their quick evaluation exists
    return not run_dir.name.endswith("-probe") or (run_dir / "jmteb_lite.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--owner", default="yasutoshi-lab")
    parser.add_argument("--watch", type=int, default=0, help="seconds between scans (0 = run once)")
    args = parser.parse_args()
    api = HfApi()
    while True:
        for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir() and p.name.startswith("freesia-")):
            if ready(run_dir):
                try:
                    repo = upload_run(api, run_dir, args.owner)
                    (run_dir / "UPLOADED").write_text(json.dumps({"repo": repo, "time": time.strftime("%F %T")}))
                    print(f"[{time.strftime('%F %T')}] UPLOADED {run_dir.name} -> {repo}", flush=True)
                except Exception as e:  # noqa: BLE001 - retry on the next scan
                    print(f"[{time.strftime('%F %T')}] UPLOAD_ERROR {run_dir.name}: {type(e).__name__}: {e}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
