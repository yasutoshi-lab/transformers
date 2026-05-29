"""
Ranunculus-1B を HuggingFace Hub の private リポジトリにアップロードするスクリプト。

使い方:
  uv run python upload_to_hub.py [--repo-name REPO_NAME]
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


MODEL_DIR = Path(__file__).parent / "models/ranunculus-1b-final"
SRC_DIR = Path(__file__).parent.parent.parent.parent / "src/transformers/models/ranunculus"

CODE_FILES = [
    "configuration_ranunculus.py",
    "modeling_ranunculus.py",
]

AUTO_MAP = {
    "AutoConfig": "configuration_ranunculus.RanunculusConfig",
    "AutoModelForCausalLM": "modeling_ranunculus.RanunculusForCausalLM",
}

# 相対インポート (from ...X) → 絶対インポート (from transformers.X) に変換するパターン
_RELATIVE_IMPORT_RE = re.compile(r"^(from )\.\.\.(.+)", re.MULTILINE)
_LOCAL_IMPORT_RE = re.compile(r"^(from )\.\.\.models\.ranunculus\.(configuration_ranunculus)", re.MULTILINE)


def _fix_imports(src_text: str) -> str:
    """相対インポートを transformers の絶対インポートに変換する。"""
    # from ...models.ranunculus.X → from .X (同梱ファイル参照はそのまま)
    text = re.sub(
        r"^from \.\.\.models\.ranunculus\.",
        "from .",
        src_text,
        flags=re.MULTILINE,
    )
    # from ...X → from transformers.X
    text = re.sub(
        r"^from \.\.\.([\w])",
        r"from transformers.\1",
        text,
        flags=re.MULTILINE,
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-name",
        default="Ranunculus-1B",
        help="Hub のリポジトリ名（ユーザー名は自動取得）",
    )
    args = parser.parse_args()

    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{args.repo_name}"

    print(f"リポジトリ: {repo_id}  (private)")
    create_repo(repo_id, private=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- モデルファイルをコピー ---
        for f in MODEL_DIR.iterdir():
            shutil.copy2(f, tmp / f.name)

        # --- config.json に auto_map を追加 ---
        config_path = tmp / "config.json"
        config = json.loads(config_path.read_text())
        config["auto_map"] = AUTO_MAP
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

        # --- カスタムコードファイルをインポート変換して追加 ---
        for fname in CODE_FILES:
            src = (SRC_DIR / fname).read_text()
            (tmp / fname).write_text(_fix_imports(src))

        # --- 全ファイルをアップロード ---
        files = sorted(tmp.iterdir())
        print(f"\nアップロードするファイル ({len(files)} 件):")
        for f in files:
            size = f.stat().st_size
            print(f"  {f.name:<50s} {size / 1024 / 1024:6.1f} MB")

        print("\nアップロード中...")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add Ranunculus-1B pretrained on Wikipedia",
        )

    print(f"\n完了: https://huggingface.co/{repo_id}")
    print("\n利用方法:")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True, private=True)')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True, private=True)')


if __name__ == "__main__":
    main()
