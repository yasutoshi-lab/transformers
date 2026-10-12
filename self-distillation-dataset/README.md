# self-distillation-dataset（Gemma 4 版）

`google/gemma-4-26B-A4B-it` 自身で自己蒸留用データセットを生成するパイプライン。
gpt-oss-20b 版（`gpt-oss-self-distill` ブランチ）の方式を Gemma 4 へ移植したもの。
推論は vLLM（ws3-arc / RTX PRO 6000 Blackwell × 2、GPU ごとに 1 サーバー）。

## 要件

- reasoning あり / なしを同量生成
- 途中切れ・言語混入・重複を規則ベースで除去（モデルによる採点は使わない）
- 5 ドメインを構成比どおりに生成（数学 20% / 日本語 10% / 英語 10% / 一般知識 30% / コード 30%）
- シードは Apache-2.0 のデータセットに限る（使うのはプロンプトのみ）
- 多様な書き出し（プレフィックス）
- 目標は**通過数** 50 万件

## シード

| ドメイン | データセット（すべて Apache-2.0） | 絞り込み |
|---|---|---|
| math | `AI-MO/NuminaMath-CoT` | 15 万件に間引き |
| code | `m-a-p/CodeFeedback-Filtered-Instruction`、`glaiveai/glaive-code-assistant-v3` | glaive は 5 万件に間引き |
| general_knowledge | `TIGER-Lab/WebInstructSub`、`HuggingFaceTB/smoltalk`（smol-magpie-ultra） | WebInstructSub は socratic / stackexchange から 6 万件。magpie は情報探索・助言・推論・データ分析 |
| english | `llm-jp/oasst2-33k-en`、`HuggingFaceTB/smoltalk`（smol-magpie-ultra） | magpie は作文・編集・ブレストから 6 万件 |
| japanese | `llm-jp/oasst2-33k-ja`、`llm-jp/magpie-sft-v1.0`、`Aratako/Magpie-Tanuki-8B-97k` | 仮名を含む依頼文のみ。magpie-sft は 4 万件、Tanuki は 3 万件に間引き |

japanese 以外のドメインは、仮名・漢字を含む依頼文を除く（中国語の依頼文が混ざるため）。

## gpt-oss 版からの変更点

| 項目 | gpt-oss 版 | Gemma 4 版 |
|---|---|---|
| 思考の形式 | Harmony（analysis / final チャネル） | `<|channel>thought\n…<channel|>` の後ろに応答 |
| 思考あり / なし | チャネルヘッダ（analysis / final）を明示 | `enable_thinking=True` + `<|channel>thought\n` / `enable_thinking=False`（空の思考が入る） |
| 思考の深さ | reasoning_effort を low / medium / high で混合 | Gemma 4 に段階指定がないため混合しない |
| サンプリング | temperature 1.0 / top_p 1.0 | モデル同梱の generation_config（temperature 1.0 / top_p 0.95 / top_k 64） |
| 系列長 | 8,192 | 16,384。生成上限（思考あり 12,288 / なし 8,192）をプロンプト長から逆算して切り詰める |
| 並列 | 1 サーバー・ドメイン順に処理 | 複数サーバーへ round-robin。(domain, mode) の進捗が quota 比で遅いものから供給 |
| シードの使い方 | 1 シード最大 10 回の想定で quota を算出 | 1 シードは mode ごとに 1 回。プールを使い切った (domain, mode) は他が補う |
| 構成比 | ドメイン等分 | `--domain-ratio` で指定（既定 20/10/10/30/30） |

## 使い方

```bash
# 1. vLLM サーバー（GPU ごとに 1 コンテナ）。イメージ同梱の CUDA compat ライブラリは
#    ホストのドライバ（575）と衝突する（Error 803）ため外してから起動する
for g in 0 1; do
  docker run -d --name gemma4-sd-gpu$g --gpus "\"device=$g\"" --ipc=host \
    -p 127.0.0.1:$((8110+g)):8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 --entrypoint bash vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
    -c 'rm -f /etc/ld.so.conf.d/*compat*.conf && ldconfig && exec vllm serve "$@"' _ \
    --model google/gemma-4-26B-A4B-it --served-model-name gemma-4-26b-a4b-it \
    --max-model-len 16384 --max-num-seqs 128 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
done

# 2. パイロット
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd src && ../.venv/bin/python generate.py --target-kept 2000 --concurrency 128 \
  --output ../out/gemma4-pilot.jsonl
../.venv/bin/python review.py ../out/gemma4-pilot.jsonl ../out/gemma4-pilot-samples.txt

# 3. 本走（中断しても同じコマンドで再開できる）
../.venv/bin/python generate.py --target-kept 500000 --concurrency 128 \
  --output ../out/gemma4-full.jsonl
```

## レコード形式

```json
{
  "prompt": "...",
  "prefix": "Let me set up the notation",
  "mode": "reasoning",
  "domain": "math",
  "seed_source": "openai/gsm8k/train",
  "seed_index": 123,
  "prompt_tokens": 85,
  "thinking": "…",
  "response": "…",
  "thinking_closed": true,
  "finish_reason": "stop",
  "completion_tokens": 1024,
  "error": "",
  "lang": "en",
  "model": "google/gemma-4-26B-A4B-it",
  "passed": true,
  "reasons": [],
  "duplicate": ""
}
```

学習に使うのは `passed == true` かつ `duplicate == ""` のレコード。`thinking` / `response` は
書き出し（prefix）を先頭に含む。

## 設計上の判断

- 特権情報（参照解）は与えない。自己蒸留で特権情報を与えると thinking モデルで
  fork suppression により性能が劣化する（wiki 側 `sources/rethinking-on-policy-self-distillation-for-thinking-models`）。
- 書き出しは思考ヘッダの後ろに置く（gpt-oss 版で、ヘッダ前に置くと継ぎ目が記号断片になった）。
- 思考は英語（引用符外の仮名・漢字 3% まで）。ただし japanese 域は、英語で推論した後に
  日本語の回答を思考内で下書きする例が多く、一律の基準では思考ありの 7 割弱が落ちた。
  japanese 域では思考の前半に 5%、全体に 30% までの日本語を許す。
- 英語のシードを japanese 域に回す場合は、言語指示を system ではなくユーザープロンプト末尾に
  付け、記録に残す（日本語ネイティブのシードには付けない）。
- `english_cot_leak` は gpt-oss 版の `We (need|can|must) …` から、メタ推論の言い回し
  （"The user asks …" 等）に絞った。Gemma 4 の英語の解説文を誤検出したため。
