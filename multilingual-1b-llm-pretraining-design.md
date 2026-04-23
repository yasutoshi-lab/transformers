---
type: clip
created: 2026-04-22
updated: 2026-04-23
sources: []
tags: [llm, pretraining, architecture, qwen3, tokenizer, multilingual, hyperparameters, gpu-training, design-report, ranunculus]
confidence: high
---

# Ranunculus-1B 事前学習設計レポート（多言語 SLM・Qwen3 ベース・EN/JA）

> **モデル名**: `ranunculus-1b`（Curator オリジナル多言語 SLM シリーズ "Ranunculus" の 1B 版）
>
> **改訂履歴**: 2026-04-22 初版（Ranunculus-700M・4言語/15.1B tokens）→ 2026-04-23 に 1B・EN/JA 2言語・2 epoch 構成へ再設計。

## Context

単一 GPU（NVIDIA RTX PRO 6000 Blackwell 96GB）で、2言語（英・日）の Wikipedia コーパスを用いて **1B パラメータ**規模の LLM を**スクラッチから事前学習**するための、アーキテクチャ〜訓練ハイパーパラメータまでの完全設計レポート。

### 設計前提

| 項目 | 値 |
|---|---|
| パラメータ規模 | **1.015B**（Dense、通称 "1B"） |
| ベースアーキテクチャ | [[wiki/concepts/qwen3|Qwen3 スタイル]] |
| 対応言語 | 英語(en) / 日本語(ja) の **2言語** |
| データセット | wikimedia/wikipedia |
| 総学習トークン | **21.2B tokens**（source 10.6B × 2 epoch） |
| 想定コンテキスト長 | 8,192 tokens |
| GQA 比率 | 4:1 |
| 訓練 GPU | RTX PRO 6000 Blackwell 96GB × 1 |
| 学習フロー | **PT** → CPT → SFT → GRPO（本レポートは PT のみ対象） |
| MoE | **不採用**（Dense 構成） |

### データセット配分

```
English  (en):  6.41M 記事 / ~9.5B tokens (source)
Japanese (ja):  1.39M 記事 / ~1.1B tokens (source)
───────────────────────────────────────
Source 合計:                ~10.6B tokens
2 epoch 単純反復:
  English : 19.0B tokens   (全データ 2 周)
  Japanese:  2.2B tokens   (全データ 2 周)
───────────────────────────────────────
総学習トークン:            ~21.2B tokens
```

D/N 比 = 21.2B / 1.015B = **20.89** → [[wiki/concepts/chinchilla-scaling-law|Chinchilla 20:1]] にほぼ完全に整合し **compute-optimal** な配置となる。

### 2 epoch 単純反復の方針

本設計では**温度サンプリングを行わず**、EN/JA それぞれの全データを単純に 2 周する。これは自然配分のままの訓練となる：

- **自然配分**: EN 89.6% / JA 10.4%
- **日本語性能のトレードオフ**: JA は全体の約 10% に留まるため、前身 Ranunculus-700M（4言語・温度 α=0.5 で JA 15%）より日本語タスクでの性能は劣後する可能性がある。これは "EN/JA を絞ることでシンプルさとスケールを優先する" 設計判断の帰結である（詳細は [[#設計判断の記録|Notes §設計判断の記録]]）。

---

## Findings

### 1. アーキテクチャ最終設計

[[wiki/concepts/qwen3]] を縮小した Dense Decoder-only 構成。Qwen3-4B の設計哲学（深さ優先・q/k ノルム・2-RMSNorm）を 1B 規模に移植する。head_dim は Qwen3-4B と同一の 128 を採用することで、ヘッドあたりの表現力を維持する。

#### 主要ハイパーパラメータ

| ハイパーパラメータ | 値 | 根拠 |
|---|---|---|
| `hidden_size` | **1,536** | Qwen3 の深さ優先哲学に沿う中庸値（Qwen2.5-1.5B 同等） |
| `num_hidden_layers` | **35** | 層数を稼いで表現力を確保（Qwen3-4B=36層にほぼ匹敵） |
| `num_attention_heads` | **12** | hidden / head_dim = 1536 / 128 |
| `num_key_value_heads` | **3** | GQA 4:1（12÷4）。KV キャッシュを 1/4 に圧縮 |
| `head_dim` | **128** | Qwen3-4B と同じ head_dim（ヘッドあたり表現力の維持） |
| `intermediate_size` | **4,096** | hidden × 2.67（128 倍数で tensor core 効率◎） |
| `hidden_act` | `silu` | [[wiki/concepts/swiglu|SwiGLU]] |
| `rms_norm_eps` | `1e-6` | Qwen3-4B 準拠 |
| `rope_theta` | **500,000** | 8K context では Llama-3 ベースで十分 |
| `max_position_embeddings` | **8,192** | 指定通り |
| `tie_word_embeddings` | **True** | 1B 規模でも embedding 14% を占めるため継続採用 |
| q_norm / k_norm | **あり**（RMSNorm, head_dim=128 単位） | Qwen3 スタイルの訓練安定化の核 |
| RMSNorm 配置 | **2 箇所**（input + post-attention） | Llama-3 / Qwen3 共通 |
| Attention 種別 | Full Attention（SWA 不採用） | 8K スケールでは不要 |
| MoE | **不採用** | 単一 GPU での訓練を優先 |

#### Qwen3-4B との比較

| 項目 | Qwen3-4B | 本設計（1B） | 縮小比 |
|---|---|---|---|
| hidden_size | 2,560 | 1,536 | 0.60× |
| num_layers | 36 | 35 | 0.97× |
| num_heads (Q/KV) | 32/8 | 12/3 | 0.375× |
| head_dim | 128 | 128 | **1.00×** |
| intermediate | 9,728 | 4,096 | 0.42× |
| MLP ratio | 3.8× | 2.67× | — |
| 総パラメータ | 4.0B | 1.015B | 0.254× |

**設計方針**: 層数をほぼ維持（36→35）しつつ、hidden を 60% に、MLP 比率を 3.8→2.67 へ控えめにして 1B に着地。**head_dim は 128 を維持**する点が 700M 版（head_dim=80）との本質的な違いで、ヘッドあたりの Q/K 表現力を Qwen3-4B と同等に保てる。

---

### 2. パラメータ計算詳細

vocab_size = 96,000（後述の tokenizer 設計に整合）での精密計算。

#### Per-layer パラメータ内訳

```
Attention:
  q_proj:   1536 × 1536 =  2,359,296
  k_proj:   1536 ×  384 =    589,824
  v_proj:   1536 ×  384 =    589,824
  o_proj:   1536 × 1536 =  2,359,296
  q_norm + k_norm:            256
  ───────────────────────────────────
  小計:                   5,898,496

MLP (SwiGLU):
  gate_proj: 1536 × 4096 =  6,291,456
  up_proj:   1536 × 4096 =  6,291,456
  down_proj: 4096 × 1536 =  6,291,456
  ───────────────────────────────────
  小計:                  18,874,368

Norms (input + post-attn):
  2 × 1536          =         3,072
───────────────────────────────────
Per layer total:         24,775,936
```

#### 全体合計

```
35 layers × 24,775,936   =   867,157,760
Embedding (tied, 96K × 1536): 147,456,000
Final RMSNorm                 :       1,536
─────────────────────────────────────────────
Total                    : 1,014,615,296  ≈ 1.015B ✓
```

#### vocab_size 感度

| vocab_size | Embedding | 総パラメータ |
|---|---|---|
| 65,536 | 101 M | 968 M |
| 80,000 | 123 M | 990 M |
| **96,000** | **147 M** | **1,015 M** ✓ |
| 128,000 | 197 M | 1,064 M |
| 151,936 | 233 M | 1,101 M |

---

### 3. HuggingFace config.json

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 1536,
  "num_hidden_layers": 35,
  "num_attention_heads": 12,
  "num_key_value_heads": 3,
  "head_dim": 128,
  "intermediate_size": 4096,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "rope_theta": 500000.0,
  "max_position_embeddings": 8192,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "vocab_size": 96000,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "use_cache": true
}
```

---

### 4. Tokenizer 設計

#### 基本方針

| 項目 | 設定 | 理由 |
|---|---|---|
| アルゴリズム | **BPE + byte-fallback** | GPT-2/Llama-3/Qwen3 系互換、未知文字を UNK にしない堅牢性 |
| 実装 | HuggingFace `tokenizers` 独自訓練 | Qwen3 エコシステムに自然に接続 |
| vocab_size | **96,000**（本体 95,744 + 特殊トークン 256） | EN/JA 2言語に必要十分かつ 128 倍数で GPU 効率◎ |
| 正規化 | **NFC**（lowercase/アクセント除去なし） | 日本語濁点/半濁点の揺れ対策 |
| Pre-tokenization | Qwen3/GPT-4 系 CJK 対応正規表現 | `\p{L}` で EN/JA 両言語の文字クラス対応 |

#### 採用しなかった選択肢

| 戦略 | 却下理由 |
|---|---|
| Qwen3 tokenizer (152K) 流用 | 2言語では過剰、embedding 233M で総パラメータ超過 |
| Llama-3 tokenizer (128K) 流用 | 2言語では過剰、日本語圧縮効率が独自訓練より劣る |
| vocab 64K に縮小 | 日本語圧縮効率が落ち、EN/JA 2言語分の語彙として手狭 |
| vocab 128K 維持 | embedding 197M で総 1.06B となり 1B 超過。2言語では圧縮効率の上積み小 |
| SentencePiece Unigram | Qwen3 エコシステムとの互換性が BPE より弱い |

**96K の選定**: EN/JA 2 言語なら 96K は「byte-fallback 前提で日本語を過不足なくカバーしつつ、embedding を 147M に抑える」均衡点。本体語彙 95,744 + 特殊 256 で丁度 128 倍数（750×128）に揃う。

#### 特殊トークン予約（256 枠）

SFT/GRPO 段階での embedding 再初期化を避けるため**事前予約**する：

| トークン | 用途 | 想定使用ステージ |
|---|---|---|
| `<|endoftext|>` | 文書終端 | PT から |
| `<|pad|>` | パディング | PT から |
| `<|im_start|>` / `<|im_end|>` | チャット区切り | SFT |
| `<|system|>` / `<|user|>` / `<|assistant|>` | ロール | SFT |
| `<think>` / `</think>` | 思考モード | GRPO |
| `<tool_call>` / `</tool_call>` | ツール呼び出し | 将来拡張 |
| `<|reserved_0|>` 〜 `<|reserved_244|>` | 将来拡張用 | — |

合計 **256 枠**を確保し、本語彙は 95,744 で BPE 訓練する。

#### 訓練コーパス戦略

本学習は EN:JA = 90:10 の自然配分（単純 2 epoch）だが、**tokenizer 訓練時は均等配分**する：

| 言語 | 割当 | 理由 |
|---|---|---|
| English | 500M tokens | EN 圧縮効率の維持 |
| Japanese | 500M tokens | 日本語語彙を意図的に確保（本訓練の自然配分 10% に対して tokenizer で圧縮性能を担保） |
| **合計** | **1B tokens** | 均等配分 |

**自然配分が EN 90% に偏るにも関わらず tokenizer は均等配分とする理由**: 本訓練時の JA トークン効率が低すぎると loss が言語別に不均衡になる。tokenizer 段階で JA 語彙を十分取り込むことで、本訓練での有効トークン数を確保する。

#### 実装雛形

```python
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token=None))
tokenizer.normalizer = normalizers.NFC()

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(
        pattern=Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"),
        behavior="isolated",
        invert=False,
    ),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])

tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=95_744,            # 256 特殊トークン分を差し引く
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train(files=balanced_wiki_files, trainer=trainer)

tokenizer.add_special_tokens([
    "<|endoftext|>", "<|pad|>",
    "<|im_start|>", "<|im_end|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    *[f"<|reserved_{i}|>" for i in range(245)],
])
```

#### 予測される圧縮効率

| 言語 | 想定 chars/token | 参考値（Qwen3 152K） |
|---|---|---|
| English | ~4.0 | 4.1 |
| Japanese | ~1.8 | 2.0 |

Qwen3 (152K) より若干劣るが、2言語特化で byte-fallback 併用のため実用上問題なし。

---

### 5. ハイパーパラメータ設計

#### 統合設定（YAML 形式）

```yaml
# ============ Optimizer ============
optimizer:            AdamW
beta1:                0.9
beta2:                0.95            # Llama-3/Qwen3 準拠
epsilon:              1e-8
weight_decay:         0.1
weight_decay_exclude: [bias, norm.weight, embed_tokens.weight]

# ============ LR Schedule ============
scheduler:            cosine_with_min_lr   # transformers の SchedulerType では cosine_with_min_lr を使用
peak_lr:              3.0e-4          # 1B Dense の安全域（Qwen3 スケーリングから）
min_lr:               3.0e-5          # peak の 10%
warmup_steps:         2000            # 全体の約 4.94%
warmup_start_lr:      0.0

# ============ Batch ============
sequence_length:      8192
micro_batch_size:     4               # RTX PRO 6000 Blackwell 96GB 単体で安全マージン
gradient_accumulation: 16
effective_batch:      64              # seqs
tokens_per_step:      524_288         # ≈ 0.5M tokens

# ============ Training Length ============
total_tokens:         21_200_000_000  # EN 9.5B + JA 1.1B の 2 epoch
total_steps:          40_500          # 21.2B / 524K
checkpoint_every:     2000            # 約 20 回保存
num_epochs:           2               # 全データを 2 周（温度サンプリング不使用）

# ============ Precision ============
compute_dtype:        bfloat16
param_dtype:          bfloat16
master_weight_dtype:  float32
gradient_dtype:       float32
attention_impl:       flash_attn_2    # 必須
gradient_checkpointing: true

# ============ Stability ============
gradient_clipping:    1.0
initializer_range:    0.02
dropout:              0.0             # Chinchilla-optimal では不要
attention_dropout:    0.0

# ============ Data Mixing ============
# 温度サンプリングは不使用（自然配分 EN 90% / JA 10% のまま 2 epoch）
seed:                 42
```

#### 各設計の根拠

##### Optimizer: AdamW

| 選択肢 | 判定 | 理由 |
|---|---|---|
| **AdamW** | **採用** | 成熟・安定・ほぼ全 LLM の実績 |
| [[wiki/concepts/muon-optimizer\|Muon]] | 不採用 | 初学習でのリスク。CPT ステージで実験候補 |

- **beta2 = 0.95**: 0.999 は小規模モデル + 短期訓練で分散推定が過度に滑らかになり初期学習が停滞する。Llama-3 / Qwen3 の選択に準拠
- **weight_decay 除外**: bias / RMSNorm スケール / embedding は decay すると表現力低下・低頻度語不利化のリスク

##### 学習率: peak 3e-4

Qwen3 系小規模モデルの経験則に基づくスケーリング：

```
Qwen3-4B 参考値:  ~5e-4
1B スケール:      5e-4 × (1000/4000)^0.15 ≈ 4.08e-4
安全マージン込み: 3.0e-4
```

前身 700M 版と同一の 3e-4 を採用する。1B でも十分安全域に収まり、再現性の観点からも推奨。

- **min_lr = peak × 0.1**: 完全にゼロに落とすと後段平滑化で不利
- **warmup 2,000 steps（~1.05B tokens）**: 2言語・1B 規模で十分。総ステップ数 40,500 に対して 4.94% と、700M 版の 6.9% より比率は低いが絶対量（1B tokens）で判断。1B 規模・自然配分・1 epoch 単位の観点で 2,000 は最小限ながら十分

##### バッチサイズ: 有効 0.5M tokens

- [[wiki/concepts/chinchilla-scaling-law|Chinchilla 系]]の経験則で、1B 規模は **0.5〜1M tokens/batch** が compute-optimal
- 1M 超は gradient noise scale 不足で収束遅延
- 0.5M は RTX PRO 6000 Blackwell 96GB 単体で安定運用可能（後述 VRAM 計算参照）

#### VRAM 使用量詳細

| 項目 | 計算 | VRAM |
|---|---|---|
| モデル重み (bf16) | 1.015B × 2 | 2.0 GB |
| 勾配 (fp32) | 1.015B × 4 | 4.1 GB |
| Adam m+v (fp32) | 1.015B × 8 | 8.1 GB |
| Master weights (fp32) | 1.015B × 4 | 4.1 GB |
| **Optimizer 合計** | | **18.3 GB** |
| 活性化 (micro_bs=4, grad_ckpt=ON) | 概算 | ~25 GB |
| KV + workspace | 概算 | ~5 GB |
| **総合計** | | **~48 GB / 96 GB** |

大きな余裕がある。gradient_checkpointing OFF + micro_bs=8 への拡張も理論上可能だが、**初回訓練は安全構成を優先**。

---

### 6. データサンプリング戦略: 自然配分 × 2 epoch

**温度サンプリングを不使用**とし、EN/JA それぞれのコーパスを全データ 2 周する単純構成とする。

| 言語 | Source | 2 epoch | 自然配分比率 |
|---|---|---|---|
| English | 9.5B | **19.0B** | **89.6%** |
| Japanese | 1.1B | **2.2B** | **10.4%** |
| **合計** | **10.6B** | **21.2B** | 100% |

#### 単純 2 epoch 方式の理由

- **実装の単純さ**: 温度サンプリング用の per-document sampling probability 管理が不要。`PackedDataset` が各言語の .bin を順に読むだけで済む
- **シード固定の再現性**: 各 epoch で同一の文書順（ただしインターリーブする）を使用でき、resume 時の state 再現が容易
- **2 言語構成との親和性**: 4 言語時代（Ranunculus-700M）では温度 α=0.5 で JA/ZH を補強する意義が大きかったが、2言語では EN/JA の 2 極のみでバランス調整の自由度が低く、単純反復の透明性を優先
- **Chinchilla D/N 適合**: 21.2B / 1.015B = 20.89 → ほぼ 20:1 に自然に合致する

#### 日本語性能のトレードオフ

自然配分では JA が全体の約 10% に留まる。これは以下の帰結をもたらす：

- JA タスクの perplexity / val_loss は EN より高止まりする可能性が高い
- Ranunculus-700M の設計では温度 α=0.5 で JA 15% まで引き上げていた（2.07× 繰り返し相当）ため、それとの比較では **JA は実質的に訓練量が減っている**
- Wikipedia JA は品質が高いので 2 周は過学習リスクも許容範囲と見込める
- 後段 CPT で日本語特化コーパスを追加することで、この不均衡を補正する設計とする（本 PT の範囲外）

#### 言語インターリーブ

2 epoch の実装は以下の順序で行う：

```
Epoch 1: [EN 文書群 19.0B ÷ 2 = 9.5B] + [JA 文書群 2.2B ÷ 2 = 1.1B] をインターリーブ
Epoch 2: 同じ文書群をもう一度（文書順は epoch 内で再シャッフル）
```

インターリーブ単位は「文書」レベルで、epoch 内の言語比率は源の 9.5:1.1 を保つ。

---

### 7. Sequence Packing

8K 固定長に文書を連結してパディング無駄を排除：

```
[doc_A ... <|endoftext|>] [doc_B ...] [doc_C ... <|endoftext|>] ... → 8192 tokens
```

#### Attention マスク戦略

| 戦略 | 判定 | 理由 |
|---|---|---|
| **Full attention**（文書境界無視） | **採用** | 1B 規模でも性能差はほぼなし（Llama/Qwen 経験則） |
| Cross-doc mask | 不採用 | flash_attn_varlen 必要で実装複雑、利益に対し過剰 |

---

### 8. 重み初期化

```python
initializer_range = 0.02
```

- `embed_tokens`: `N(0, 0.02²)`
- `Linear`: `N(0, 0.02²)`
- `RMSNorm.weight`: `ones(...)`
- **残差スケーリング**: `o_proj` と `down_proj` のみ `N(0, (0.02/√(2*n_layers))²)` で初期化
  - 35 層 → scale = 0.02 / √70 ≈ **0.00239**
  - Pre-Norm + 深い層での勾配爆発を防ぐ

---

### 9. 訓練時間・リソース見積もり

#### 総演算量

```
FLOPs = 6 × N × D
      = 6 × 1.015B × 21.2B
      = 1.291 × 10²⁰ FLOPs
```

700M 版の 6.34×10¹⁹ から約 **2.04 倍**に増加。

#### RTX PRO 6000 Blackwell 96GB の実効性能

| 項目 | 値 |
|---|---|
| ピーク bf16（dense） | ~252 TFLOPS |
| 実効 MFU | 30〜40% |
| 実効スループット | ~90〜100 TFLOPS |

#### 所要時間

| MFU 前提 | 訓練時間 |
|---|---|
| 40%（楽観） | ~14.9 日 |
| 30%（標準） | ~19.9 日 |
| 25%（保守） | ~24.9 日 |

**実用見積もり: 15〜25 日連続稼働**

---

### 10. モニタリング指標

| 指標                   | 頻度         | 警戒ライン             |
| -------------------- | ---------- | ----------------- |
| train_loss           | 10 steps   | 急上昇 → 即停止         |
| grad_norm            | 10 steps   | > 10 注意、> 50 停止   |
| lr                   | 1 step     | スケジュール通り確認        |
| tokens/sec           | 100 steps  | 急落 → I/O ボトルネック疑う |
| VRAM 使用量             | 100 steps  | OOM 予防            |
| per-lang val_loss    | 500 steps  | **EN/JA** を別々に追跡  |
| param norm (layer 別) | 1000 steps | 異常発散を検出           |

[[wiki/concepts/wandb|wandb]] 統合推奨。

---

### 11. チェックポイント戦略

```
2,000 ステップごと (≈ 1B tokens ごと) に保存
総 20 回 + 最終チェックポイント = 21 個
各 ~14GB (bf16 モデル + fp32 optimizer 状態) × 21 = ~294GB ストレージ
（save_total_limit=21 で常時上限管理、古いものから自動削除可能）
```

**保持すべき情報**:
- モデル重み (bf16)
- Optimizer 状態 (fp32)
- RNG 状態（再現性）
- スケジューラ状態
- ステップ数・epoch

---

### 12. 学習パイプライン全体フロー（文脈確認）

本レポートの対象範囲は **PT ステージのみ**。後段ステージとの接続を明示する：

```
[PT: 本レポート]     → [CPT]          → [SFT]           → [GRPO]
21.2B tokens           ドメインコーパス    指示応答ペア       選好データ
2 言語 × 2 epoch       JA 特化補正 など   チャット能力付与   推論/整合強化
40,500 steps           数千 steps         数千 steps        数百〜数千 steps
全パラメータ訓練       LoRA/部分凍結      LoRA or 全        LoRA 前提
```

PT 段階でのトークン予約・チェックポイント方針は**後続ステージを見据えた設計**になっている：
- 特殊トークン 256 枠予約 → SFT/GRPO で embedding 再初期化不要
- WSD 検討済み（ただし Cosine で PT 完了予定）
- 中間チェックポイント 20 個保存 → 後段ステージの起点選択肢確保
- **CPT 段階での日本語特化データ追加**を前提とした PT 設計（JA 10% の不足を後段で補正）

---

## 訓練前チェックリスト

### Tokenizer 検証

- [ ] 2 言語（EN/JA）サンプル文の encode/decode でラウンドトリップ完全一致
- [ ] UTF-8 任意テキストが byte-fallback で復元可能
- [ ] 特殊トークン 256 枠すべて登録済み
- [ ] `tokenizer.vocab_size` == 96,000
- [ ] 絵文字・珍しい記号が UNK にならない
- [ ] 日本語「今日は良い天気です。」が 5〜7 token 程度
- [ ] 英語「The quick brown fox jumps over the lazy dog.」が 9〜11 token 程度
- [ ] HuggingFace `AutoTokenizer.from_pretrained()` で保存/読み込み成功

### 訓練環境

- [x] データの事前トークン化完了（`artifacts/packed/train_{en,ja}.bin` / `val_{en,ja}.bin`）
- [x] 各言語 validation set を 10M tokens 分確保（訓練データから分離済み）
- [ ] Flash Attention 2 ソースビルド（`pip install flash-attn --no-build-isolation --no-binary :all:`）完了後、`configs/ranunculus-1b.yaml` の `attn_implementation` を `flash_attention_2` に変更
- [x] PyTorch **2.11.0+cu129** インストール済み（ドライバ CUDA 12.9 対応）
- [x] `accelerate >= 1.1.0` / `wandb >= 0.26.0` インストール済み
- [x] wandb プロジェクト作成・認証済み（`~/.netrc`）
- [ ] 冷却・電源安定性確認（3〜4 週間連続稼働）
- [x] GPU 指定 `--gpu 0` の動作確認（2 枚検出時の DataParallel クラッシュ回避）
- [x] 最小実行テスト: 3 steps で loss 降下確認（11.76 → 安定）
- [ ] チェックポイント保存/復旧テスト（2,000 steps 時点）

---

## Notes

### 設計判断の記録

#### なぜ 700M から 1B にスケールアップしたか

- 700M は Chinchilla compute-optimal の下限に近く、少しでもデータが増えると under-train 気味になる
- 単一 RTX PRO 6000 Blackwell 96GB では 1B 規模まで安全に訓練できる VRAM 余裕がある（活性化込みで ~48GB / 96GB）
- 1B は "小さい割に表現力がある" 実用域のラインで、CPT/SFT/GRPO 後段のチューニング効果が顕在化しやすい

#### なぜ EN/JA 2 言語に絞ったか

- 4 言語（EN/DE/JA/ZH）版は tokenizer 設計・データ配分・per-lang 評価すべての複雑度が高く、**scratch 学習の初回試行**としては過剰
- DE/ZH はいずれもシングル GPU でのパイロット規模としては "存在感が薄い"（DE は 3.5B で中規模、ZH は 1B で少数派）
- EN/JA の 2 極に絞ることで、実装・評価・デバッグすべての粒度が上がり、設計判断の因果が追跡可能になる
- 将来 ZH/DE を加える際は CPT で継続学習するか、Ranunculus-1b-multilingual のような別枝として扱う

#### なぜ単純 2 epoch か（温度サンプリング不使用）

- 4 言語時代の温度 α=0.5 は **JA/ZH を漢字補強のため意図的に繰り返す**ためのテクニックだったが、2 言語では繰り返し設計の自由度が下がる
- 自然配分のままでも D/N = 20.89 で Chinchilla にほぼ完全準拠。温度を入れてもパラメータ数 × トークン数の総量は変わらない
- シンプルさ > 局所最適: PT 段階では**言語間バランスよりも訓練の透明性・再現性**を優先
- JA 10% の不足は後段 CPT で明示的に補正する前提（PT 段階で全部を解こうとしない設計）

#### なぜ MoE を採用しないか
- 単一 GPU 訓練では MoE のメモリ消費（全エキスパート保持）がメリットを殺す
- 1B 規模では MoE の表現力上積みより Dense の安定性を優先
- 後段 CPT/SFT での追加訓練コストが MoE では増加

#### なぜ深さ優先（35 層）か
- Qwen3 系の実証: 同パラメータ数なら層数を稼ぐ方が表現力が高い
- Llama-3.2-1B は hidden=2048・layers=16 の幅優先設計だが、多言語性能では深さ優先が有利との報告あり
- 推論レイテンシは若干上がるが、1B 規模では実用範囲

#### なぜ head_dim=128 か（700M 版との変更）
- 700M 版は hidden=1280・heads=16 の制約で head_dim=80 を採用したが、ヘッドあたりの Q/K 表現力がやや窮屈だった
- 1B 版は hidden=1536・heads=12 で head_dim=128 を採用可能となり、**Qwen3-4B と同等の head_dim** を確保
- head_dim=128 は flash-attn / xformers などほぼ全てのカーネルで最適化されている "標準サイズ"

#### なぜ vocab=96K か（700M 版の 128K からの変更）
- 4 言語から 2 言語に減ったため 128K は過剰で、embedding 197M で総 1.06B となり目標の 1B を超過する
- 96K は 2 言語（EN/JA）で byte-fallback 前提なら圧縮効率の劣化が小さく、embedding 147M で総 1.015B に収まる
- 95,744 + 256 = 96,000 = 750 × 128 でアラインメント良好

#### なぜ Cosine を WSD より優先するか
- 初回訓練では実績ある Cosine が安全
- WSD は中間チェックポイントを再利用する設計で、初回 PT では必要性が低い
- 後続 CPT で WSD を試す選択肢は残る

### 将来のアップグレード余地

- **Muon optimizer**: CPT 段階で実験導入を検討
- **Dual RoPE + SWA**: 長文コンテキスト拡張時（16K+）に有効
- **Sliding Window Attention**: 8K 内では不要だが、将来 32K/128K 拡張時に導入
- **Cross-document attention masking**: 30B tokens 超のスケールアップ時に導入検討
- **多言語再拡張**: ZH/DE は CPT で後から継続学習で加える / 別枝の Ranunculus-1b-multilingual として分ける選択肢

---

## 実装計画

本節は、上記設計を **transformers リポジトリを fork した `ranunculus` ブランチ内で、他の公式モデル（Qwen3 / Llama / Gemma 等）と同じレイアウトで実装する**計画を記述する。modular 機構で Qwen3 を最大限継承し、差分のみを最小量のコードで表現する。

### ブランチ戦略

```
upstream: huggingface/transformers (main)
     │
     └─ fork: yasutoshi-lab/transformers
              │
              └─ ranunculus ブランチ  ← 全開発はこのブランチ内
```

- `git clone <fork> && git checkout -b ranunculus` の上、以下に示す全ファイルを追加
- upstream への追従は**必要なときだけ手動で** `git merge origin/main`（自動追従はしない）
- **upstream `huggingface/transformers` への PR 提出は行わない**（純粋に個人開発用ブランチ）。したがって transformers の "Mandatory Agentic contribution policy" は適用外。ただし `make fix-repo` / `make style` は modular 整合性維持のため必ず通す
- 学習済み重み＋tokenizer は transformers ブランチには含めず、`huggingface.co/yasutoshi-lab/ranunculus-1b` リポジトリへ push

---

### リポジトリ内ファイル配置（transformers ブランチ）

他のモデルと完全に同じパターン：`src/transformers/models/<name>/` に 4 ファイル、`tests/models/<name>/` に 2 ファイル、`docs/source/en/model_doc/<name>.md`、`examples/pytorch/` に訓練スクリプト、中央レジストリに登録。

```
transformers/  (ranunculus ブランチ)
├── src/transformers/models/ranunculus/       ★ Qwen3 と同型
│   ├── __init__.py                           # _LazyModule パターン
│   ├── configuration_ranunculus.py           # RanunculusConfig（Qwen3Config 継承）
│   ├── modular_ranunculus.py                 # ★ 主な編集対象：Qwen3 継承 + 残差 init 差分
│   └── modeling_ranunculus.py                # make fix-repo で自動生成、手編集禁止
├── tests/models/ranunculus/
│   ├── __init__.py
│   ├── test_modeling_ranunculus.py           # Qwen3 テストを継承
│   └── test_residual_init.py                 # ★ 残差スケール init の unit test
├── docs/source/en/model_doc/ranunculus.md    # モデルカード
├── examples/pytorch/language-modeling/
│   ├── run_ranunculus_pretrain.py            # ★ 新規: メイン PT スクリプト
│   ├── train_ranunculus_tokenizer.py         # ★ 新規: BPE+byte-fallback 訓練
│   ├── prepare_ranunculus_data.py            # ★ 新規: Wikipedia DL+2epoch packing
│   ├── packed_dataset.py                     # np.memmap ベースの Dataset
│   └── ranunculus_callbacks.py               # PerLanguageEval / ParamNorm / TokensPerSec
├── configs/ranunculus-1b.yaml                # モデル/訓練/データのユーザー設定
└── (既存ファイルへの 1 行追記のみ)
    ├── src/transformers/models/__init__.py               # from . import ranunculus
    ├── src/transformers/models/auto/configuration_auto.py # ("ranunculus", "RanunculusConfig")
    ├── src/transformers/models/auto/modeling_auto.py      # ("ranunculus", "RanunculusForCausalLM")
    └── docs/source/en/_toctree.yml                       # - local: model_doc/ranunculus
```

**中央ファイルには 1 行ずつしか触らない**ことで upstream の merge コンフリクトを最小化する。

---

### モデル実装（`modular_ranunculus.py`）

Ranunculus はアーキテクチャ的に Qwen3 そのもの（hyperparam と残差 init のみ差分）。modular で Qwen3 を継承し、**差分は 2 種類のみ**：

1. `RanunculusConfig` のデフォルト値（設計 §1 の 1B スペック）
2. `_init_weights` で `o_proj` / `down_proj` を `0.02/√(2L)` に再初期化（設計 §8）

```python
# src/transformers/models/ranunculus/modular_ranunculus.py
import math
import torch.nn as nn
from ..qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel,
    Qwen3Attention, Qwen3MLP,
)
from .configuration_ranunculus import RanunculusConfig


class RanunculusPreTrainedModel(Qwen3PreTrainedModel):
    config: RanunculusConfig
    _no_split_modules = ["RanunculusDecoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        # 設計 §8 残差スケーリング: 末端 o_proj / down_proj のみ std 縮小
        if getattr(module, "_is_residual_proj", False):
            scaled = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
            nn.init.normal_(module.weight, mean=0.0, std=scaled)


class RanunculusAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.o_proj._is_residual_proj = True


class RanunculusMLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj._is_residual_proj = True


class RanunculusModel(Qwen3Model):
    pass


class RanunculusForCausalLM(Qwen3ForCausalLM):
    pass


__all__ = [
    "RanunculusForCausalLM",
    "RanunculusModel",
    "RanunculusPreTrainedModel",
]
```

`make fix-repo` が `modeling_ranunculus.py` を自動生成する。手で編集するのは **modular のみ**。

#### `configuration_ranunculus.py`

Qwen3Config のデフォルト値を Ranunculus-1B 用に差し替える：

```python
from ..qwen3.configuration_qwen3 import Qwen3Config


class RanunculusConfig(Qwen3Config):
    model_type = "ranunculus"
    vocab_size: int = 96_000
    hidden_size: int = 1_536
    num_hidden_layers: int = 35
    num_attention_heads: int = 12
    num_key_value_heads: int = 3
    head_dim: int = 128
    intermediate_size: int = 4_096
    max_position_embeddings: int = 8_192
    tie_word_embeddings: bool = True
    use_sliding_window: bool = False
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 500_000.0}
        super().__post_init__(**kwargs)
```

#### 中央レジストリへの登録（必須・各 1 行）

| ファイル | 追記内容 |
|---|---|
| `src/transformers/models/__init__.py` | `from . import ranunculus` |
| `src/transformers/models/auto/configuration_auto.py` | `MODEL_NAMES_MAPPING` / `CONFIG_MAPPING_NAMES` に `("ranunculus", "RanunculusConfig")` |
| `src/transformers/models/auto/modeling_auto.py` | `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` に `("ranunculus", "RanunculusForCausalLM")` |
| `docs/source/en/_toctree.yml` | `- local: model_doc/ranunculus` |

登録後、`AutoModelForCausalLM.from_pretrained(<path>)` が `config.json` の `"model_type": "ranunculus"` で解決できるようになる。

---

### Tokenizer

Ranunculus は Qwen3 と**同じ tokenizer 仕様**（BPE + byte-fallback + 96K）なので、**`tokenization_ranunculus.py` は作らない**。`PreTrainedTokenizerFast` のまま `tokenizer.json` / `tokenizer_config.json` を生成し、保存先の `AutoTokenizer.from_pretrained()` で読み込む。

訓練スクリプト `examples/pytorch/language-modeling/train_ranunculus_tokenizer.py` は設計 §4 の雛形を実装：

1. **コーパス抽出**: `datasets.load_dataset("wikimedia/wikipedia", "20231101.{en,ja}")` から各言語 500M tokens 相当を抽出（NFC のみ正規化）
2. **BPE 訓練**: 設計 §4 の `BpeTrainer(vocab_size=95_744, byte_fallback=True)`、2 ファイルを行レベル均等インターリーブ
3. **特殊 256 枠追加**: 設計 §4 のリスト順に `add_special_tokens`
4. **HF ラップ**: `PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|pad|>", ...)` → `save_pretrained("artifacts/tokenizer/")`

#### 検証

- 2 言語各 1,000 文ラウンドトリップ一致（NFC 後で比較）
- 絵文字・珍しい Unicode 500 サンプルで byte-fallback が UNK を出さない
- 圧縮効率が設計 §4 予測値 ±0.3 chars/token 以内
- `len(tokenizer) == 96_000`

---

### データパイプライン（`examples/pytorch/language-modeling/`）

transformers に相当機能がないため自前で書くが、`datasets` の `load_dataset` / `Dataset.map(batched=True, num_proc=...)` を徹底活用する。

#### `prepare_ranunculus_data.py`

1. **ダウンロード**: 各言語の Wikipedia を `datasets.load_dataset("wikimedia/wikipedia", "20231101.{lang}")` で取得し `data/raw/{lang}.parquet` に保存
2. **train / val 分離**: 各言語 10M tokens を val に抽出
3. **tokenize + packing**: `AutoTokenizer.from_pretrained("artifacts/tokenizer/")` で tokenize → `<|endoftext|>` で連結 → 8192 トークン固定長に切り出し → `data/packed/{train,val}_{lang}.bin`（uint32 フラット列）
4. **2 epoch 構成**: 本訓練用には `train_{lang}.bin` を**そのまま 2 回読み込む**実装とする（`PackedDataset` のコンストラクタに `num_epochs=2` を渡す）。別案として `train_{lang}_epoch{1,2}.bin` を作り文書順を epoch 毎にシャッフルしたい場合は pre-shuffle した 2 本を用意する

tokenize は `Dataset.map(batched=True, num_proc=16)` でプロセス並列。

#### `packed_dataset.py`

```python
class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, bin_paths: list[str], seq_len: int = 8192, num_epochs: int = 1):
        self.arrs = [np.memmap(p, dtype=np.uint32, mode="r") for p in bin_paths]
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        per_file = [len(a) // seq_len for a in self.arrs]
        self.offsets = np.cumsum([0] + per_file)
        self.per_epoch_len = int(self.offsets[-1])
    def __len__(self): return self.per_epoch_len * self.num_epochs
    def __getitem__(self, i):
        i = i % self.per_epoch_len            # epoch 境界でラップ
        fi = int(np.searchsorted(self.offsets[1:], i, side="right"))
        local = i - self.offsets[fi]
        ids = self.arrs[fi][local * self.seq_len : (local + 1) * self.seq_len]
        ids = torch.from_numpy(ids.astype(np.int64))
        return {"input_ids": ids, "labels": ids}
```

固定長で pad 不要、`labels = input_ids` → `RanunculusForCausalLM` の内部 shift + CE がそのまま動く。`DataCollatorForLanguageModeling` は使わない（デフォルト collate で十分）。`num_epochs=2` により `__len__` が倍になり、Trainer の `max_steps` 計算と整合する。

---

### 訓練スクリプト（`run_ranunculus_pretrain.py`）

transformers の `examples/pytorch/language-modeling/run_clm.py` を**丸ごとコピーせず**、Ranunculus 専用の薄い独自スクリプトとして実装する（機能を絞ることで読みやすさを優先）。

骨子：

```python
# examples/pytorch/language-modeling/run_ranunculus_pretrain.py
from transformers import (
    AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments,
    RanunculusConfig, RanunculusForCausalLM,
)
from packed_dataset import PackedDataset
from ranunculus_callbacks import (
    PerLanguageEvalCallback, ParamNormCallback, TokensPerSecCallback,
)
import torch


class RanunculusTrainer(Trainer):
    def create_optimizer(self):
        # 設計 §5: bias / RMSNorm / embed を weight_decay から除外
        if self.optimizer is not None: return self.optimizer
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if p.dim() < 2 or "norm.weight" in n or "embed_tokens.weight" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        a = self.args
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": a.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=a.learning_rate, betas=(a.adam_beta1, a.adam_beta2),
            eps=a.adam_epsilon, fused=True,
        )
        return self.optimizer


def main(yaml_path: str):
    cfg = load_yaml(yaml_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    model_cfg = RanunculusConfig(**cfg["model"])
    if cfg.get("attn_implementation"):                    # yaml の attn_implementation キーから取得
        model_cfg._attn_implementation = cfg["attn_implementation"]
    model = RanunculusForCausalLM(model_cfg)              # 残差 init は _init_weights で自動
    model.gradient_checkpointing_enable()

    train_ds = PackedDataset(cfg["train_bins"], num_epochs=cfg.get("num_epochs", 2))
    val_ds_by_lang = {lang: PackedDataset([p]) for lang, p in cfg["val_bins"].items()}

    args = TrainingArguments(**cfg["training"])           # 下表参照
    trainer = RanunculusTrainer(
        model=model, args=args, train_dataset=train_ds,
        tokenizer=tokenizer,
        callbacks=[
            PerLanguageEvalCallback(val_ds_by_lang, every=500),
            ParamNormCallback(every=1000),
            TokensPerSecCallback(every=100),
        ],
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume"))
    trainer.save_model(cfg["final_dir"])
```

#### `TrainingArguments`（`configs/ranunculus-1b.yaml` で指定）

```python
TrainingArguments(
    output_dir="artifacts/ranunculus-1b/pt/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    max_steps=40_500,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
    warmup_steps=2000,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_steps=2000, save_total_limit=21,
    logging_steps=10,
    eval_strategy="no",                       # 言語別 eval は Callback で実装
    dataloader_num_workers=4, dataloader_pin_memory=True,
    report_to=["wandb"], run_name="ranunculus-1b",
    seed=42,
    torch_compile=False,
)
```

#### Callbacks（`ranunculus_callbacks.py`）

Trainer 標準に無い 3 つだけ：

- `PerLanguageEvalCallback`: 500 ステップ毎に EN/JA の val .bin それぞれで forward し、`val_loss/{lang}` を wandb 記録
- `ParamNormCallback`: 1000 ステップ毎に layer 別 `param_norm` を wandb 記録
- `TokensPerSecCallback`: 100 ステップ毎に `batch × seq_len / elapsed` を wandb 記録

#### checkpoint・resume・最終エクスポート

`TrainingArguments(save_steps=2000)` + `Trainer.train(resume_from_checkpoint="artifacts/ranunculus-1b/pt/checkpoint-XXXX")` でモデル・optimizer・scheduler・RNG state まですべて保存/復旧される。自前の checkpoint.py は**不要**。

`trainer.save_model("artifacts/ranunculus-1b/final/")` で `config.json` + `model.safetensors` + tokenizer が HF レイアウト出力。検証は `AutoModelForCausalLM.from_pretrained("artifacts/ranunculus-1b/final/")` が成功し短文生成が通ること。HuggingFace Hub 公開時のリポジトリ名も `ranunculus-1b` で統一。

---

### テスト（`tests/models/ranunculus/`）

transformers の `ModelTesterMixin` を継承して Qwen3 のテストをそのまま再利用する：

```python
# tests/models/ranunculus/test_modeling_ranunculus.py
from ..qwen3.test_modeling_qwen3 import Qwen3ModelTester, Qwen3ModelTest
from transformers import RanunculusConfig, RanunculusForCausalLM, RanunculusModel


class RanunculusModelTester(Qwen3ModelTester):
    config_class = RanunculusConfig


class RanunculusModelTest(Qwen3ModelTest):
    all_model_classes = (RanunculusModel, RanunculusForCausalLM)
    pipeline_model_mapping = {"text-generation": RanunculusForCausalLM}
    model_tester_class = RanunculusModelTester
```

加えて Ranunculus 固有の 1 ファイル：

- `test_residual_init.py`: tiny config で `RanunculusForCausalLM(config)` 生成後に、全 `o_proj.weight` / `down_proj.weight` の std が `0.02/√(2L)` の ±5% 以内、他の `*_proj.weight` は 0.02 の ±5% 以内

---

### 実装順序（マイルストーン）

| Step | 成果物                                                                     | 所要 | 検証                                                                    |
| ---- | --------------------------------------------------------------------- | ---- | --------------------------------------------------------------------- |
| M1   | ranunculus ブランチ作成 + `models/ranunculus/` 4 ファイル + 中央登録 + テスト 2 ファイル | 1 日  | `make fix-repo && make style && make typing && pytest tests/models/ranunculus/ -x` 全通過 |
| M2   | tokenizer 訓練 + ラウンドトリップ                                                 | 2〜3 日 | 設計 §4 検証項目                                                             |
| M3   | データパイプライン + packed .bin 生成 (2 epoch 対応)                                 | 2 日  | packed .bin が設計配分、`PackedDataset(num_epochs=2)` が `__len__ × 2` を返す   |
| M4   | `run_ranunculus_pretrain.py` smoke run (100 steps)                       | 1〜2 日 | loss 単調減少、grad_norm < 5、tokens/sec 計測、resume 成功                         |
| M5   | 本訓練 21.2B tokens                                                        | 15〜25 日 | §10 の指標を wandb で監視、per-lang val loss が収束                               |
| M6   | model card 執筆 + HF Hub push                                             | 0.5 日 | `AutoModelForCausalLM.from_pretrained("yasutoshi-lab/ranunculus-1b")` 読み込み成功 |

M1〜M4 合計でおよそ **6〜8 日**。

---

### 検証計画

#### transformers ワークフロー（M1 で必ず通す）

- `make fix-repo`: `modular_ranunculus.py` → `modeling_ranunculus.py` 自動変換が整合すること
- `make style`: ruff lint/format
- `make typing`: ty 型チェック
- `pytest tests/models/ranunculus/ -x`: モデル構造テスト（Qwen3 テストを継承しているため、hyperparam が tiny で合えばそのまま通る）

#### Ranunculus 固有テスト

- `test_residual_init.py`: 上記 §テスト
- tokenizer ラウンドトリップ（2 言語 × 1,000 文）
- packed .bin の形式（長さが 8192 の倍数、`<|endoftext|>` 出現数 ≥ 文書数 − 1）

#### Smoke run（本訓練前の必須関門）

1. **tiny**: `hidden=128, layers=2, vocab=8K, seq_len=512, batch=1, accum=1` で 100 steps、loss が `log(8K) ≈ 9` 付近から 0.5 以上降下
2. **full-config 100 steps**: 本番 config + 本番データで 100 steps、tokens/sec が想定域（sdpa: ≈ 12k tokens/sec、flash_attention_2 ビルド後: ≈ 20k tokens/sec 以上）
3. **full-config 2,000 steps**: 1 checkpoint ぶん回し、`resume_from_checkpoint` が成功
4. 全通過でフル訓練へ

---

### リスクと対処

| リスク                                                              | 検出方法                       | 対処                                                                                                                                     |
| ---------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `make fix-repo` で modular → modeling 変換が失敗                        | M1 CI 失敗                   | `docs/source/en/modular_transformers.md` を参照。Gemma3 の継承パターンを手本にする                                                                       |
| AutoModel 登録漏れで `model_type="ranunculus"` が解決できない                 | `from_pretrained` 失敗       | 中央 3 ファイル（`models/__init__.py` / `auto/configuration_auto.py` / `auto/modeling_auto.py`）への追記を M1 チェックリストに明記                                 |
| 残差 init の `_is_residual_proj` フラグが modular → modeling 変換で失われる     | `test_residual_init.py` 失敗 | `RanunculusAttention.__init__` / `RanunculusMLP.__init__` が変換後も残ることを確認。残らない場合は `_init_weights` を名前ベース（`"o_proj.weight"` / `"down_proj.weight"`）で判定する実装に切り替え |
| upstream `main` との merge コンフリクト                                   | rebase 時                   | 中央ファイルへの追記は各 1 行のみ。Ranunculus 以外のファイルには一切触らない                                                                                            |
| transformers バージョン内 API 破壊                                        | M1 CI / M4 smoke 失敗        | `ranunculus` ブランチは任意タイミングで upstream を取り込めるので commit 固定は不要。代わりに**動く commit を tag で保存**（`ranunculus-m4-passed` 等）                            |
| flash-attn 非互換                                                    | M4 smoke 失敗（ImportError: undefined symbol）                | PyPI の事前ビルド wheel は PyTorch バージョンによってシンボルミスマッチが発生する。`pip install flash-attn --no-build-isolation --no-binary :all:` でソースビルド（15〜30 分）。解決前は `configs/ranunculus-1b.yaml` の `attn_implementation: sdpa` で稼働継続（MFU ≈ 29%、約 12k tokens/sec）                                                      |
| Trainer デフォルトの weight_decay 除外が設計 §5 と不一致                         | 初回 smoke の param group 観察  | `RanunculusTrainer.create_optimizer` オーバーライドで明示制御                                                                                        |
| 日本語の圧縮効率未達                                                        | M2 tokenizer 検証            | tokenizer 訓練コーパスの言語配分を 300M/700M 等 JA 寄りに再バランス                                                                                          |
| **JA 10% 配分による日本語性能不足**                                           | M5 per-lang val_loss       | PT 段階では許容（設計判断による）。後段 CPT で日本語特化コーパスを追加して補正                                                                                              |
| 本訓練中の loss 発散                                                     | §10 の train_loss / grad_norm 監視 | 直近 checkpoint から `resume_from_checkpoint`、`learning_rate` を 2.0e-4 に下げて再開                                                              |
| I/O ボトルネックで tokens/sec 急落                                         | `TokensPerSecCallback`     | packed .bin を NVMe ローカル SSD に配置、`dataloader_num_workers` を 8 に増やす                                                                        |
| **3〜4 週間連続稼働**中の電源/冷却障害                                           | ハード側監視                     | `save_steps` を 1000 に短縮、UPS 有効化                                                                                                          |

---

### 参照ファイルインデックス

開発時に頻繁に開くファイル：

- `src/transformers/models/qwen3/modular_qwen3.py` — **`modular_ranunculus.py` の一次手本**
- `src/transformers/models/qwen3/configuration_qwen3.py` — `RanunculusConfig` のフィールド継承元
- `src/transformers/models/qwen3/modeling_qwen3.py` — 継承元の forward ロジック確認
- `src/transformers/models/gemma3/modular_gemma3.py` — より複雑な modular 継承の参考例（Llama + Gemma 継承のパターン）
- `tests/models/qwen3/test_modeling_qwen3.py` — `test_modeling_ranunculus.py` の手本
- `examples/pytorch/language-modeling/run_clm.py` — 訓練スクリプトの骨組み参考
- `docs/source/en/modular_transformers.md` — modular 機構の公式ガイド
- [[wiki/clips/flash-attention-install-notes]] — Flash Attention 2 セットアップ

---

## Connections

### アーキテクチャ基盤
- [[wiki/concepts/qwen3]] — ベースアーキテクチャ
- [[wiki/concepts/decoder-only-llm-architectures]] — Decoder-only 族の共通骨子
- [[wiki/concepts/grouped-query-attention]] — GQA 4:1 設計
- [[wiki/concepts/rmsnorm]] — 2-RMSNorm + q/k ノルム
- [[wiki/concepts/swiglu]] — SiLU ゲート MLP
- [[wiki/concepts/rotary-position-embedding]] — 標準 RoPE
- [[wiki/concepts/transformer-gradient-flow]] — Pre-Norm + 残差初期化の安定性

### 学習理論
- [[wiki/concepts/scaling-laws]] — パラメータ/トークン比の理論基盤
- [[wiki/concepts/chinchilla-scaling-law]] — 20:1 最適比
- [[wiki/concepts/compute-optimal-training]] — compute vs inference 最適

### 実装・ツール
- [[wiki/concepts/tokenization]] — BPE 方式の基礎
- [[wiki/concepts/bpe]] — BPE + byte-fallback の詳細
- [[wiki/concepts/llm-data-pipeline]] — 事前学習データパイプライン
- [[wiki/concepts/kv-cache]] — GQA との関係
- [[wiki/concepts/flash-attention]] — 必須実装
- [[wiki/concepts/huggingface-trainer]] — Trainer API
- [[wiki/concepts/wandb]] — 実験トラッキング

### 関連設計資料
- [[wiki/sources/multilingual-pretraining-design-dialogue]] — 多言語事前学習設計判断の先行議論（4 言語時代）
- [[wiki/clips/model-architectures-catalog]] — 52 モデル分のアーキテクチャカタログ
- [[wiki/clips/llm-core-tech-catalog]] — LLM 中核技術カタログ
- [[wiki/clips/flash-attention-install-notes]] — Flash Attention 導入メモ
- [[wiki/clips/llm-training-experiment-log]] — 訓練実験カタログ

### 比較対象モデル
- [[wiki/concepts/llama-3]] — シンプル構成との比較
- [[wiki/concepts/gemma-3]] — 4-RMSNorm + Dual RoPE との比較
- [[wiki/entities/nemophila]] — 既存のオリジナル LLM シリーズ（nanochat ベース・57.93M）
- [[wiki/entities/nanochat]] — 単一スクリプト完結型の参考実装

---

## 実行環境前提

### ハードウェア・ソフトウェア構成

| 項目 | 値 | 備考 |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell × **2** | 設計は 1 GPU 前提。訓練時は必ず `--gpu 0` を指定して GPU 0 のみ使用 |
| VRAM | 97,887 MiB × 2 | 単体で sdpa + micro_bs=4 + grad_ckpt で ≈ 25 GB 消費 |
| CUDA Driver | 575.51.03（CUDA 12.9 対応） | CUDA 13.0 要求の PyTorch は不可 |
| PyTorch | **2.11.0+cu129** | cu130 はドライバ非互換で `torch.cuda.is_available()` が False になる |
| accelerate | ≥ 1.1.0 | Trainer 必須 |
| wandb | ≥ 0.26.0 | `report_to: none` で無効化可 |

### 依存関係インストール

```bash
# PyTorch（cu129 ビルドを明示）
uv pip install torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# Trainer 必須
uv pip install "accelerate>=1.1.0" wandb

# Flash Attention 2（ソースビルド必須・15〜30 分）
# 事前ビルド wheel は PyTorch バージョンによってシンボルミスマッチが発生するため
# --no-binary :all: でソースコンパイルする
pip install flash-attn --no-build-isolation --no-binary :all:
# ビルド完了後、configs/ranunculus-1b.yaml の attn_implementation を flash_attention_2 に変更
# ビルド前は sdpa のまま運用可（MFU ≈ 29%、約 12k tokens/sec）
```

### GPU 指定について

本環境は GPU が 2 枚搭載されているため、`--gpu 0` を省略すると Trainer が DataParallel を起動してクラッシュする。**すべての実行コマンドに `--gpu 0` を付けること**。

> `run_ranunculus_pretrain.py` は `--gpu 0` 指定時に `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` を設定する。

---

## コマンド実行フロー

**作業ディレクトリ: `examples/pytorch/language-modeling/`**（プロジェクトルートではない）

`run_ranunculus_pretrain.py` が `from packed_dataset import PackedDataset` / `from ranunculus_callbacks import ...` をパッケージ外の bare import で参照するため、すべてのコマンドをこのディレクトリから実行する。

```bash
cd /path/to/transformers/examples/pytorch/language-modeling
```

```
Step 1 (tokenizer) → Step 2 (data) → Step 3-a (smoke) → Step 3-b (本訓練)
```

各ステップは前ステップの成果物に依存するため、必ず順番に実行する。

### Step 1: Tokenizer 訓練（M2）

```bash
uv run python train_ranunculus_tokenizer.py \
  --out_dir artifacts/tokenizer \
  --corpus_dir artifacts/tokenizer_corpus \
  --dump_version 20231101
```

- EN/JA 各 500M tokens を Wikipedia からダウンロード
- BPE 95,744 語 + 特殊 256 枠 = 96,000 vocab のトークナイザーを訓練
- 成果物: `artifacts/tokenizer/`

### Step 2: データ準備（M3）

```bash
uv run python prepare_ranunculus_data.py \
  --tokenizer_dir artifacts/tokenizer \
  --out_dir artifacts/packed \
  --dump_version 20231101 \
  --seq_len 8192 \
  --num_proc 16
```

- EN/JA の Wikipedia を tokenize → 8192 トークン固定長に packing
- 各言語 10M tokens を val に分離（訓練データから除外）
- 2 epoch 反復はディスクに書かず `PackedDataset(num_epochs=2)` に委ねる
- 成果物: `artifacts/packed/train_{en,ja}.bin` / `val_{en,ja}.bin`

### Step 3-a: Smoke run（M4・本訓練前の必須関門）

`configs/ranunculus-1b.yaml` の `training.max_steps` を `100` に絞って実行:

```bash
uv run python run_ranunculus_pretrain.py \
  --config configs/ranunculus-1b.yaml \
  --gpu 0
```

確認項目:
- loss が単調減少する
- `grad_norm < 5`
- `tokens/sec` が想定域（sdpa: 約 12k tokens/sec、flash_attention_2: 約 20k tokens/sec 以上）
- `resume_from_checkpoint` が成功する

### Step 3-b: 本訓練（M5）

`configs/ranunculus-1b.yaml` の `training.max_steps` を `40500` に戻して実行:

```bash
uv run python run_ranunculus_pretrain.py \
  --config configs/ranunculus-1b.yaml \
  --gpu 0
```

途中再開:

```bash
uv run python run_ranunculus_pretrain.py \
  --config configs/ranunculus-1b.yaml \
  --gpu 0 \
  --resume artifacts/ranunculus-1b/pt/checkpoint-XXXX
```

`configs/ranunculus-1b.yaml` 主要値:

```yaml
tokenizer_dir: artifacts/tokenizer
final_dir: artifacts/ranunculus-1b/final

data:
  seq_len: 8192
  num_epochs: 2
  train_bins:
    - artifacts/packed/train_en.bin
    - artifacts/packed/train_ja.bin
  val_bins:
    en: artifacts/packed/val_en.bin
    ja: artifacts/packed/val_ja.bin

attn_implementation: sdpa   # flash-attn ソースビルド完了後は flash_attention_2 に変更

model:
  vocab_size: 96000
  hidden_size: 1536
  num_hidden_layers: 35
  num_attention_heads: 12
  num_key_value_heads: 3
  head_dim: 128
  intermediate_size: 4096
  max_position_embeddings: 8192
  tie_word_embeddings: true
  rms_norm_eps: 1.0e-6

training:
  output_dir: artifacts/ranunculus-1b/pt
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 16
  max_steps: 40500
  learning_rate: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  warmup_steps: 2000
  lr_scheduler_type: cosine_with_min_lr
  lr_scheduler_kwargs:
    min_lr_rate: 0.1
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  save_steps: 2000
  save_total_limit: 21
  logging_steps: 10
  eval_strategy: "no"
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  report_to:
    - wandb
  run_name: ranunculus-1b
  seed: 42
```
