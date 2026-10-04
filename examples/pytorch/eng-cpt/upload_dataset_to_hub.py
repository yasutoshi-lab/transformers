"""前処理済みコーパス / SFT データを HF datasets 形式にして Hub の private リポジトリへアップロードする.

Upload the preprocessed CPT corpus or the SFT QA set to a private Hugging Face dataset repository.

``--kind corpus``: ``artifacts/data/{train,qa_eval,holdout_ppl}.jsonl`` を 3 split で push し、
                  ``manifest.json`` をリポジトリ直下に置く。
``--kind sft``   : ``artifacts/qa/sft.jsonl`` を train split で push し、``qa_manifest.json`` と
                  生成の生出力 ``raw/sft_raw.jsonl``（後処理の再現に必要）を置く。
データセットカード（README.md）には本文を含まない数値・スキーマ・再現情報だけを書く。

安全策:
    - push の前に private でリポジトリを作成し、private であることを確認してから送る。
      既存リポジトリが public だった場合は送らずに中断する。
    - ``drop_samples.jsonl``（除去ページの本文抜粋）はアップロードしない。

使い方（eng-cpt/ 直下で実行）:
    python upload_dataset_to_hub.py --kind corpus --repo-id yasutoshi-lab/eng-textbook-cpt-ja
    python upload_dataset_to_hub.py --kind sft --repo-id yasutoshi-lab/eng-textbook-sft-ja
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import DatasetCard, HfApi

from data_prep.books import BOOKS
from data_prep.corpus import SPLITS


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "artifacts" / "data"
QA_DIR = Path(__file__).resolve().parent / "artifacts" / "qa"

SFT_CARD_TEMPLATE = """---
language:
- ja
pretty_name: 工学系教科書 SFT データ（日本語 QA）
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-generation
tags:
- sft
- engineering
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# eng-textbook-sft-ja

[`yasutoshi-lab/eng-textbook-cpt-ja`](https://huggingface.co/datasets/yasutoshi-lab/eng-textbook-cpt-ja) の
`train` split（工学系教科書 20 冊の本文）から、`nvidia/Gemma-4-26B-A4B-NVFP4` で生成した日本語の QA。
`google/gemma-4-E4B` への CPT → SFT 検証の SFT 段階で使う。

> **取り扱い注意**: 元の本文は市販書籍に由来し、QA はその内容に基づく。社内の研究目的に限って使い、
> **このリポジトリを public にしない・外部へ再配布しない**こと。

## 件数

| 項目 | 値 |
|---|---|
| 件数 | {rows:,} |
| カテゴリ | {categories} |
| 生成元チャンク | {chunks:,}（約 1,500 字ずつ。1 チャンクあたり最大 4 組を生成） |
| 生成件数 → 除去 | {raw:,} 組 → 参照表現 {drop_ref} 組・重複 {drop_dup} 組を除去 |
| 生成トークン | 入力 {prompt_tokens:,} / 出力 {completion_tokens:,} |

評価用 4 択（コーパスの `qa_eval` 範囲から生成）とは生成元の範囲が重ならない。

## Schema

| key | 型 | 内容 |
|---|---|---|
| `id` | string | `sft-<通し番号 5 桁>` |
| `source_id` | string | 生成元文書の id（コーパスの `id`） |
| `chunk_id` | string | 生成元チャンク（`<source_id>#<チャンク番号>`） |
| `book` / `category` | string | 生成元の書籍名 / カテゴリ |
| `messages` | list | `[{{"role": "user", ...}}, {{"role": "assistant", ...}}]` の 1 往復 |

## 生成と後処理

- 質問は抜粋を読んでいない人にも通じる形（「本文」「図3」などの参照を禁止）、回答は抜粋の内容のみで 2〜5 文
- JSON スキーマ固定で生成。JSON 文字列内の未エスケープ LaTeX（`\\frac` が改ページ+`rac` になる等）は復元済み
- 生成コード: `yasutoshi-lab/transformers` の `gemma4-eng-cpt` ブランチ `examples/pytorch/eng-cpt/qagen/`

## 再現情報

- 生成モデル: `{generator}` revision `{generator_revision}`（vLLM `{vllm_image}` / `{vllm_image_id}`）
- 生成の生出力: `raw/sft_raw.jsonl`。これに後処理（`python -m qagen.build_qa --postprocess-only --steps sft`）を
  かけると `data/` の内容とバイト一致する。LLM 生成そのものは seed を指定しても完全一致は保証されない
- 作成経緯（手作業の工程を含む）・プロンプトとパラメータ・出力の sha256 は `qa_manifest.json` を参照
- manifest 作成時のコード: `yasutoshi-lab/transformers` `{code_commit}`
"""

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

## 再現情報

- 作成コード: `yasutoshi-lab/transformers` `{code_commit}`（`gemma4-eng-cpt` ブランチ `examples/pytorch/eng-cpt/`）
- トークナイザ: `{tokenizer}` revision `{tokenizer_revision}`
- 依存: {environment}
- 入力: OCR JSON {n_inputs} 件（各 sha256 は `manifest.json`）。同じ入力・コード・依存で再実行すると出力はバイト一致する
- 出力の sha256 / 行数・パラメータの全量は `manifest.json` を参照
"""


def build_card(stats, manifest):
    """stats.json の要約と manifest からデータセットカードの本文を作る.

    Args:
        stats (dict): ``stats.json`` の内容。
        manifest (dict): ``manifest.json`` の内容。

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
        code_commit=manifest["code"].get("commit", "unknown")[:10],
        tokenizer=manifest["params"]["tokenizer"], tokenizer_revision=manifest["params"]["tokenizer_revision"],
        environment=", ".join(f"{k} {v}" for k, v in manifest["environment"].items()),
        n_inputs=len(manifest["inputs"]),
    )


def build_sft_card(rows, qa_stats, qa_manifest):
    """SFT データ・qa_stats.json・qa_manifest.json からデータセットカードを作る.

    Args:
        rows (list[dict]): sft.jsonl の行。
        qa_stats (dict): ``artifacts/qa/qa_stats.json`` の内容。
        qa_manifest (dict): ``artifacts/qa/qa_manifest.json`` の内容。

    Returns:
        str: データセットカード（YAML メタデータ付き Markdown）。
    """
    c = qa_stats["sft"]["counts"]
    u = qa_stats["sft"]["usage"]
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return SFT_CARD_TEMPLATE.format(
        rows=len(rows), categories=" / ".join(f"{k} {v:,}" for k, v in cats.items()),
        chunks=c["sft_chunks"], raw=c["sft_raw_items"], drop_ref=c.get("sft_drop_source_ref", 0),
        drop_dup=c.get("sft_drop_dup", 0), prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"],
        generator=qa_manifest["generator"]["model"], generator_revision=qa_manifest["generator"]["revision"],
        vllm_image=qa_manifest["generator"]["vllm_image"], vllm_image_id=qa_manifest["generator"]["vllm_image_id"],
        code_commit=qa_manifest["code"].get("commit", "unknown")[:10],
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
    ap.add_argument("--kind", choices=["corpus", "sft"], default="corpus")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    if args.kind == "corpus":
        stats = json.loads((args.data_dir / "stats.json").read_text())
        manifest = json.loads((args.data_dir / "manifest.json").read_text())
        ds = load_dataset("json", data_files={s: str(args.data_dir / f"{s}.jsonl") for s in SPLITS})
        card, message = build_card(stats, manifest), "前処理済みコーパスを追加"
        extra_files = {"manifest.json": args.data_dir / "manifest.json"}
    else:
        path = QA_DIR / "sft.jsonl"
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        qa_manifest = json.loads((QA_DIR / "qa_manifest.json").read_text())
        ds = load_dataset("json", data_files={"train": str(path)})
        card = build_sft_card(rows, json.loads((QA_DIR / "qa_stats.json").read_text()), qa_manifest)
        message = "SFT データを追加"
        extra_files = {"qa_manifest.json": QA_DIR / "qa_manifest.json", "raw/sft_raw.jsonl": QA_DIR / "sft_raw.jsonl"}
    for name, local in extra_files.items():
        if not local.exists():
            raise FileNotFoundError(f"{local} がありません（manifest を先に作成してください）")
    print(ds)

    api = HfApi()
    ensure_private_repo(api, args.repo_id)
    ds.push_to_hub(args.repo_id, private=True, commit_message=message)
    for name, local in extra_files.items():
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=args.repo_id, repo_type="dataset",
                        commit_message=f"{name} を追加（再現性の記録）")
    DatasetCard(card).push_to_hub(args.repo_id, repo_type="dataset", commit_message="データセットカードを更新")
    print(f"uploaded: https://huggingface.co/datasets/{args.repo_id} (private={api.dataset_info(args.repo_id).private})")


if __name__ == "__main__":
    main()
