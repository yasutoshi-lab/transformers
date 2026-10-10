# self-distillation-dataset

`openai/gpt-oss-20b` 自身で自己蒸留用データセットを生成するパイプライン。
推論は vLLM（ws1-arc / RTX A6000）、データは HuggingFace dataset へアップロードする。

## 要件

- reasoning あり / なしを同量生成
- 途中切れ・言語混入を規則ベースで除去
- 一般知識・コード・数学・日本語・対話の 5 ドメイン
- 多様なプロンプトと書き出し（プレフィックス）、重複除去
- 50 万件（バッチ並列数は抑えめ）

## 使い方

```bash
# 1. vLLM サーバー（ws1-arc）
docker run --rm --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/data \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model openai/gpt-oss-20b --served-model-name gpt-oss-20b \
  --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.80

# 2. パイロット
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/generate.py --total 1000 --output out/pilot.jsonl

# 3. 本走（HF へアップロード）
.venv/bin/python src/generate.py --total 500000 --output out/full.jsonl \
  --upload-to-hf yasutoshi-lab/gpt-oss-20b-self-distillation-500k
```

## レコード形式

```json
{
  "prompt": "...",
  "prefix": "結論から述べると",
  "mode": "reasoning",
  "thinking": "…",
  "response": "…",
  "finish_reason": "stop",
  "domain": "math",
  "lang": "ja",
  "passed": true,
  "reasons": [],
  "duplicate": null
}
```

## 設計上の判断

- 特権情報（参照解）は与えない。自己蒸留で特権情報を与えると thinking モデルで
  fork suppression により性能が劣化する（wiki 側 `sources/rethinking-on-policy-self-distillation-for-thinking-models`）。
- 生成温度は公式推奨の 1.0 / top_p 1.0。
- `finish_reason == "length"` を除去するのは、生徒の失敗 prefix をそのまま持ち込む
  問題の実務形（wiki 側 `concepts/prefix-failure`）。
- 近似重複の閾値 0.80 は低め。高すぎると「多様な書き出し」の意図が消える。
- 言語判定は response に対して行う。思考（`analysis`）は Curator 方針により英語 only に統一
  （仮名・漢字を 1 文字でも含むものは不合格）。
