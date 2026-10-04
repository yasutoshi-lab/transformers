# eng-cpt: 工学系教科書による Gemma-4-E4B の軽量 CPT 検証

`google/gemma-4-E4B`（base）に工学系教科書で軽い継続事前学習（CPT, LoRA）をかけ、
SFT と組み合わせて学習前後の知識獲得・コストを測る検証用ディレクトリ。

> **注意（著作権）**: 入力は市販書籍の OCR で、社内の研究目的に限って使う。
> このブランチは PUBLIC fork 上にあるため、**書籍由来のデータは `artifacts/` 以外に置かない**
> （`artifacts/` は `.gitignore` 済み）。

## ディレクトリ構成

```
eng-cpt/
├── prepare_eng_cpt_data.py   # 前処理のエントリポイント（S1〜S6 を順に実行し manifest.json を出力）
├── upload_dataset_to_hub.py  # HF Hub の private データセットへアップロード（manifest・生出力も送る）
├── requirements-prep.txt     # 前処理・後処理・アップロード環境の固定バージョン
├── requirements-train.txt    # QA 生成クライアント・学習・評価環境（ws3-arc）の固定バージョン
├── data_prep/                # 前処理ステージごとのモジュール
│   ├── books.py              #   対象書籍 20 冊（mechanical / electrical / aeronautical）
│   ├── glyphs.py             #   S1 文字正規化（NFKC・簡体字→日本字体・OCR 誤字補正）
│   ├── lines.py              #   S2 行クリーニング（柱・ページ番号・段組み改行の接合）
│   ├── page_filter.py        #   S3 ページ除去ルールと閾値（閾値はファイル冒頭に集約）
│   ├── dedup.py              #   S4 重複除去（MD5 完全一致 + MinHash 5-gram, J≥0.85）
│   ├── corpus.py             #   S5/S6 分割（連続ブロック）と JSONL 書き出し
│   ├── stats.py              #   stats.json / drop_samples.jsonl の集計
│   └── provenance.py         #   manifest 用の出所記録（sha256・git commit・依存バージョン・revision）
├── qagen/                    # SFT 用 QA・評価用 4 択の生成（vLLM / ws3-arc GPU1）
│   ├── build_qa.py           #   生成→後処理→誤答作り直し→manifest を 1 本で実行（--postprocess-only あり）
│   ├── generate_qa.py        #   チャンク分割・生成・フィルタ・自己検証
│   ├── rebalance_mcq.py      #   4 択の誤答を正解と長さ・粒度をそろえて作り直す
│   ├── prompts.py            #   プロンプトと JSON スキーマ
│   ├── docker-compose.ws3-arc.yml  # 生成サーバ（モデル revision 固定）
│   ├── history/              #   実際の作成経緯（manifest に取り込む）
│   └── repair_escapes.py / backfill_mcq_verify_cache.py  # 2026-10-04 分の 1 回限りの移行処理
├── train/                    # LoRA 学習・評価・集計（ws3-arc GPU0）
├── tools/                    # ルール検証・レポート用の分析ツール
│   ├── inspect_drops.py      #   除去ページを理由別にサンプル表示（誤判定の目視確認）
│   ├── audit_glyphs.py       #   JIS 外漢字の監査（生 OCR / 出力の両方）
│   └── book_stats.py         #   書籍別の統計表（TSV / Markdown）
└── artifacts/                # 出力（git 追跡外）
    └── data/
```

## 実行方法

リポジトリ直下の `.venv`（Python 3.12）を使う。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r examples/pytorch/eng-cpt/requirements-prep.txt

cd examples/pytorch/eng-cpt
../../../.venv/bin/python prepare_eng_cpt_data.py            # 前処理
../../../.venv/bin/python -m tools.inspect_drops --reason toc  # 除去の目視確認
../../../.venv/bin/python -m tools.audit_glyphs --stage output # 字形の残存検査
../../../.venv/bin/python -m tools.book_stats --format md      # レポート用の表
../../../.venv/bin/python upload_dataset_to_hub.py --kind corpus --repo-id yasutoshi-lab/eng-textbook-cpt-ja
../../../.venv/bin/python upload_dataset_to_hub.py --kind sft --repo-id yasutoshi-lab/eng-textbook-sft-ja
```

Hub 上のデータセット（いずれも **private**）: [`yasutoshi-lab/eng-textbook-cpt-ja`](https://huggingface.co/datasets/yasutoshi-lab/eng-textbook-cpt-ja)（前処理コーパス）、[`yasutoshi-lab/eng-textbook-sft-ja`](https://huggingface.co/datasets/yasutoshi-lab/eng-textbook-sft-ja)（SFT 用 QA 6,591 件）。
push 前に private であることを確認し、public なら中断する。`drop_samples.jsonl` は送らない。

```python
from datasets import load_dataset
ds = load_dataset("yasutoshi-lab/eng-textbook-cpt-ja")  # HF_TOKEN が必要
```

`--raw-dir`（既定: Gaia の `raw/books`）、`--out-dir`、`--tokenizer`（既定: `google/gemma-4-E4B`）、
`--dedup-threshold` を指定できる。

## 再現手順と再現性の担保

| データ | 再現の基準 | 記録 |
|---|---|---|
| CPT コーパス | 同じ入力・コード・依存で再実行すると**バイト一致**する（乱数を使わない） | `artifacts/data/manifest.json`（入力 20 冊の sha256・commit・依存・トークナイザ revision・出力 sha256） |
| SFT 用 QA / 評価用 4 択 | LLM 生成は seed を渡しても完全一致は保証されない。**保存した生出力から後処理を再実行してバイト一致する**ことを基準とする | `artifacts/qa/qa_manifest.json`（生成モデル・revision・vLLM イメージ・プロンプト・パラメータ・出力 sha256・作成経緯） |

```bash
cd examples/pytorch/eng-cpt
# 1) CPT コーパス（このホスト）
../../../.venv/bin/python prepare_eng_cpt_data.py                 # → artifacts/data/ と manifest.json
# 2) QA（ws3-arc。先に qagen/docker-compose.ws3-arc.yml で vLLM を GPU1 に起動）
../../../.venv/bin/python -m qagen.build_qa                         # 生成→後処理→誤答作り直し→manifest
../../../.venv/bin/python -m qagen.build_qa --postprocess-only \
    --steps mcq sft rebalance                                       # LLM なしで生出力から作り直す（一致確認用）
# 3) アップロード（manifest と生出力も一緒に送る）
../../../.venv/bin/python upload_dataset_to_hub.py --kind corpus --repo-id yasutoshi-lab/eng-textbook-cpt-ja
../../../.venv/bin/python upload_dataset_to_hub.py --kind sft --repo-id yasutoshi-lab/eng-textbook-sft-ja
```

注意点:

- 入力の `raw/books/*.json` は Gaia の git で追跡されていない。入力の同一性は manifest の sha256 で確認する
- 2026-10-04 に作った QA は `build_qa.py` 整備前に手作業で作った。経緯と、生出力から再現できる範囲・できない範囲
  （誤答作り直しの試行ごとの生出力は未保存）は `qagen/history/2026-10-04.json` に記録している
- 生成サーバの `local/vllm-openai:vtm` は ws3-arc のローカルイメージ（ラベル上の元は `vllm/vllm-openai:v0.25.1-cu129-ubuntu2404`）。
  別の機体で使う場合は公式イメージで代替し、その旨を manifest に残す

## 出力データの仕様

### `{train,qa_eval,holdout_ppl}.jsonl` — 1 行 1 文書

連続するページ（同じ分割・ページ番号が連続）を 1 文書にまとめたもの。

| key | 型 | 内容 |
|---|---|---|
| `id` | str | `<書籍名>-<書籍内通し番号 4 桁>`（例: `機械加工学の基礎-0000`） |
| `book` | str | 書籍名（`raw/books/<book>.json` のファイル名） |
| `category` | str | `mechanical` / `electrical` / `aeronautical` |
| `split` | str | `train` / `qa_eval` / `holdout_ppl`（ファイル名と同じ） |
| `pages` | [int, int] | 文書に含まれる元書籍のページ範囲 `[開始, 終了]` |
| `n_chars` | int | `text` の文字数 |
| `n_tokens` | int | `text` の Gemma tokenizer トークン数（特殊トークンなし） |
| `text` | str | クリーニング済み本文。CPT で学習に使うのはこのフィールドだけ |

分割の役割:

| split | 書籍内の位置 | 用途 |
|---|---|---|
| `train` | 下記以外 | CPT に使う。SFT 用 QA の生成元 |
| `qa_eval` | 除去後ページの 30〜40% | CPT には含めるが SFT 用 QA には使わない。4 択 QA で知識獲得を測る |
| `holdout_ppl` | 除去後ページの 60〜70% | 学習に一切使わない。perplexity でドメイン適応を測る |

### `stats.json` — 数値のみ（本文を含まない）

- `summary.pages_raw` / `chars_raw`: 入力の総ページ数・総文字数
- `summary.line_cleaning`: 文字・行レベルの処理件数（`glyph_converted` / `glyph_ocr_fixed` /
  `glyph_unknown_marked` / `line_running_head` / `line_page_number` / `line_joined`）
- `summary.page_drops`: 除去理由ごとのページ数
- `summary.splits`: 分割ごとの `*_docs` / `*_pages` / `*_chars` / `*_tokens`
- `books.<書籍名>`: 上記の書籍別内訳（`salvaged_figure_pages` = 図主体から解説文を救出したページ数）

### `drop_samples.jsonl` — 除去ページの記録

`{book, page, reason, head}`（`head` は冒頭 160 字。近似重複は `dup_of` 付き）。
本文の抜粋を含むため `artifacts/` の外へ出さない。
