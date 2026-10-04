"""前処理済みコーパスを HF datasets 形式にして Hub の private リポジトリへアップロードする.

Upload the preprocessed CPT corpus to a private Hugging Face dataset repository.

``artifacts/data/{train,qa_eval,holdout_ppl}.jsonl`` を ``DatasetDict`` として読み込み、
3 split の Parquet として push する。データセットカード（README.md）には
本文を含まない数値（``stats.json`` の要約）とスキーマだけを書く。

安全策:
    - push の前に private でリポジトリを作成し、private であることを確認してから送る。
      既存リポジトリが public だった場合は送らずに中断する。
    - ``drop_samples.jsonl``（除去ページの本文抜粋）はアップロードしない。

使い方（eng-cpt/ 直下で実行）:
    python upload_dataset_to_hub.py --repo-id yasutoshi-lab/eng-textbook-cpt-ja
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import DatasetCard, HfApi

from data_prep.books import BOOKS
from data_prep.corpus import SPLITS


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "artifacts" / "data"

CARD_TEMPLATE = """---
language:
- ja
pretty_name: 工学系教科書 CPT コーパス（日本語）
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- continual-pretraining
- engineering
- ocr
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: qa_eval
    path: data/qa_eval-*
  - split: holdout_ppl
    path: data/holdout_ppl-*
---

# eng-textbook-cpt-ja

工学系教科書 20 冊（機械・材料・加工 10 冊 / 電気・電子・通信 8 冊 / 航空 2 冊）の OCR 結果を
クリーニングした、軽量な継続事前学習（CPT）検証用の日本語コーパス。
`google/gemma-4-E4B` への LoRA CPT → SFT で、工学知識の獲得とコストを測る検証に使う。

> **取り扱い注意**: 本文は市販書籍に由来する。社内の研究目的に限って使い、
> **このリポジトリを public にしない・外部へ再配布しない**こと。

## Splits

| split | 文書数 | ページ | トークン（Gemma tokenizer） | 用途 |
|---|---|---|---|---|
{split_rows}

- 分割は書籍ごとに「除去後ページの相対位置」で連続ブロックとして切り出している
  （qa_eval = 30〜40%、holdout_ppl = 60〜70%、残りが train）。
- `train` は CPT と SFT 用 QA の生成元。`qa_eval` は CPT には含めるが SFT 用 QA には使わない。
  `holdout_ppl` は学習に一切使わない。

## Schema

| key | 型 | 内容 |
|---|---|---|
| `id` | string | `<書籍名>-<書籍内通し番号 4 桁>` |
| `book` | string | 書籍名 |
| `category` | string | `mechanical` / `electrical` / `aeronautical` |
| `split` | string | `train` / `qa_eval` / `holdout_ppl` |
| `pages` | list[int] | 元書籍のページ範囲 `[開始, 終了]` |
| `n_chars` | int | `text` の文字数 |
| `n_tokens` | int | `text` の Gemma tokenizer トークン数（特殊トークンなし） |
| `text` | string | クリーニング済み本文（学習に使うのはこのフィールドのみ） |

## 前処理

入力 {pages_raw} ページ / {chars_raw} 文字。

1. 文字正規化: NFKC、JIS 外漢字のみ簡体字→日本字体（OpenCC 変換 {glyph_converted} 字・OCR 誤字の補正 {glyph_ocr_fixed} 字・判別不能でげた記号 〓 に置換 {glyph_unknown_marked} 字）
2. 行クリーニング: 柱 {line_running_head} 行・ページ番号行 {line_page_number} 行の除去、段組み改行の接合 {line_joined} 箇所
3. ページ除去（計 {pages_dropped} ページ）: {drop_list}
4. 重複除去: MD5 完全一致 + MinHash（文字 5-gram, Jaccard ≥ 0.85）

前処理コード: `yasutoshi-lab/transformers` の `gemma4-eng-cpt` ブランチ `examples/pytorch/eng-cpt/`

## 対象書籍

{book_list}
"""


def build_card(stats):
    """stats.json の要約からデータセットカードの本文を作る.

    Args:
        stats (dict): ``stats.json`` の内容。

    Returns:
        str: データセットカード（YAML メタデータ付き Markdown）。
    """
    s = stats["summary"]
    sp = s["splits"]
    uses = {"train": "CPT・SFT 用 QA の生成元", "qa_eval": "4 択 QA による知識獲得の評価", "holdout_ppl": "perplexity によるドメイン適応の評価"}
    split_rows = "\n".join(
        f"| {name} | {sp[f'{name}_docs']} | {sp[f'{name}_pages']:,} | {sp[f'{name}_tokens']:,} | {uses[name]} |" for name in SPLITS
    )
    drops = s["page_drops"]
    drop_list = "、".join(f"{k.removeprefix('drop_')} {v}" for k, v in drops.items())
    book_list = "\n".join(f"- **{cat}**: " + "、".join(titles) for cat, titles in BOOKS.items())
    lc = s["line_cleaning"]
    return CARD_TEMPLATE.format(
        split_rows=split_rows,
        pages_raw=f"{s['pages_raw']:,}",
        chars_raw=f"{s['chars_raw']:,}",
        glyph_converted=lc.get("glyph_converted", 0),
        glyph_ocr_fixed=lc.get("glyph_ocr_fixed", 0),
        glyph_unknown_marked=lc.get("glyph_unknown_marked", 0),
        line_running_head=lc.get("line_running_head", 0),
        line_page_number=lc.get("line_page_number", 0),
        line_joined=lc.get("line_joined", 0),
        pages_dropped=sum(drops.values()),
        drop_list=drop_list,
        book_list=book_list,
    )


def ensure_private_repo(api, repo_id):
    """データセットリポジトリを private で用意し、private であることを確認する.

    Args:
        api (HfApi): Hub API クライアント。
        repo_id (str): ``<user>/<name>``。

    Returns:
        None

    Raises:
        RuntimeError: リポジトリが public だった場合（本文を送らずに中断する）。
    """
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if not info.private:
        raise RuntimeError(f"{repo_id} が public です。書籍本文を含むため push を中断しました。")


def main():
    """コーパスを読み込み、private リポジトリへ push してカードを更新する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    stats = json.loads((args.data_dir / "stats.json").read_text())
    ds = load_dataset("json", data_files={s: str(args.data_dir / f"{s}.jsonl") for s in SPLITS})
    print(ds)

    api = HfApi()
    ensure_private_repo(api, args.repo_id)
    ds.push_to_hub(args.repo_id, private=True, commit_message="前処理済みコーパスを追加")
    DatasetCard(build_card(stats)).push_to_hub(args.repo_id, repo_type="dataset", commit_message="データセットカードを追加")
    print(f"uploaded: https://huggingface.co/datasets/{args.repo_id} (private={api.dataset_info(args.repo_id).private})")


if __name__ == "__main__":
    main()
