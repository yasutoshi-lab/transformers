# Ranunculus-1B 開発メモ

このリポジトリは huggingface/transformers の fork です。  
`ranunculus` ブランチに Ranunculus-1B（日英 Wikipedia スクラッチ学習モデル）の実装・学習・推論スクリプトを追加しています。

---

## 環境構築

### 前提

| 項目 | バージョン |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu129（CUDA 12.9） |
| GPU | RTX PRO 6000 Blackwell × 2（推論・学習は必ず `CUDA_VISIBLE_DEVICES=0` で GPU 0 のみ指定） |

> GPU を 2 枚のまま起動すると DataParallel の `broadcast_coalesced` でクラッシュします（cu129 + multi-GPU DataParallel の既知問題）。

### リポジトリのクローンとブランチ切り替え

```bash
git clone https://github.com/yasutoshi-lab/transformers.git
cd transformers
git checkout ranunculus
```

### uv による仮想環境セットアップ

```bash
# 仮想環境を作成して editable install
uv venv
uv pip install -e ".[torch]"

# 品質チェック・テストツールも使う場合
uv pip install -e ".[dev]"

# 学習スクリプトの追加依存
uv pip install -r examples/pytorch/language-modeling/requirements.txt
```

インストール確認：

```bash
uv run python -c "import transformers; print(transformers.__version__)"
# 5.6.0.dev0
```

---

## ディレクトリ構成

```
examples/pytorch/language-modeling/
├── artifacts/                      # .gitignore 対象（モデル重み・データ等）
│   ├── tokenizer/                  # 学習済みトークナイザー
│   └── packed/                     # packed .bin データ
├── configs/
│   └── ranunculus-1b.yaml          # 学習設定ファイル
├── train_ranunculus_tokenizer.py   # トークナイザー学習
├── prepare_ranunculus_data.py      # データ前処理
├── run_ranunculus_pretrain.py      # 事前学習
└── run_ranunculus_inference.py     # 対話推論
```

---

## 学習手順

### 1. トークナイザーの学習

96K BPE トークナイザーを Wikipedia（英語・日本語）から学習します。

```bash
cd examples/pytorch/language-modeling

uv run python train_ranunculus_tokenizer.py \
  --output-dir artifacts/tokenizer
```

### 2. データの前処理

Wikipedia をダウンロードし、8192 トークンの packed `.bin` ファイルを生成します（言語ごとに `train_{lang}.bin` / `val_{lang}.bin`）。

```bash
uv run python prepare_ranunculus_data.py \
  --tokenizer-dir artifacts/tokenizer \
  --output-dir artifacts/packed
```

生成されるファイル：

```
artifacts/packed/
├── train_en.bin
├── train_ja.bin
├── val_en.bin
└── val_ja.bin
```

### 3. 設定ファイルの編集

`configs/ranunculus-1b.yaml` を開き、パスを環境に合わせて確認します（デフォルトは `artifacts/` 配下を参照しているためそのまま動く想定）。

主要パラメータ：

| 項目 | 値 |
|---|---|
| 有効バッチサイズ | 524,288 tokens（micro=4 × accum=16 × seq=8192） |
| 最大ステップ数 | 40,500 |
| ピーク学習率 | 3e-4（cosine decay、min 3e-5） |
| ウォームアップ | 2,000 steps |
| Attention 実装 | `sdpa`（デフォルト） |

### 4. 事前学習の実行

```bash
cd examples/pytorch/language-modeling

CUDA_VISIBLE_DEVICES=0 uv run python run_ranunculus_pretrain.py \
  --config configs/ranunculus-1b.yaml
```

チェックポイントは `artifacts/checkpoints/ranunculus-1b/` に、最終モデルは `artifacts/models/ranunculus-1b-final/` に保存されます。

#### WandB を使わない場合

`configs/ranunculus-1b.yaml` の `training.report_to` を変更：

```yaml
training:
  report_to: none
```

---

## 推論

### 対話推論スクリプト

```bash
cd examples/pytorch/language-modeling

# ローカルモデルから起動（デフォルト）
CUDA_VISIBLE_DEVICES=0 uv run python run_ranunculus_inference.py

# HuggingFace Hub から起動
CUDA_VISIBLE_DEVICES=0 uv run python run_ranunculus_inference.py --from-hub
```

起動後は `Prompt>` にテキストを入力するとモデルが続きを生成します。`exit` または `Ctrl+C` で終了。

#### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model-dir` | `./models/ranunculus-1b-final` | ローカルモデルのパス |
| `--repo-id` | `yasutoshi-lab/Ranunculus-v1-1B` | Hub リポジトリ ID |
| `--from-hub` | `False` | Hub からロードする |
| `--gpu` | `0` | 使用する GPU インデックス |
| `--max-new-tokens` | `200` | 最大生成トークン数 |
| `--temperature` | `0.8` | サンプリング温度 |
| `--top-p` | `0.95` | top-p サンプリング |

```bash
# 生成長・パラメータを変えたい場合の例
CUDA_VISIBLE_DEVICES=0 uv run python run_ranunculus_inference.py \
  --max-new-tokens 300 --temperature 0.9
```

### HuggingFace Hub へのアップロード

```bash
cd examples/pytorch/language-modeling

uv run python upload_to_hub.py --repo-name Ranunculus-v1-1B
```

モデルファイルとカスタムコード（`configuration_ranunculus.py` / `modeling_ranunculus.py`）をまとめて private リポジトリにアップロードします。

---

## モデル

- **Hub**: [yasutoshi-lab/Ranunculus-v1-1B](https://huggingface.co/yasutoshi-lab/Ranunculus-v1-1B)（private）
- **パラメータ数**: 1.01B
- **学習データ**: Wikipedia（英語・日本語）、約 21.2B tokens
- **最終 loss**: 2.28 / Perplexity ≈ 9.7

Hub から直接ロードする場合（`trust_remote_code=True` が必要）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "yasutoshi-lab/Ranunculus-v1-1B",
    trust_remote_code=True,
    token=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "yasutoshi-lab/Ranunculus-v1-1B",
    trust_remote_code=True,
    token=True,
)
```
