"""
Ranunculus-v1-1B 対話推論スクリプト

使い方（ローカルモデル）:
  uv run python run_ranunculus_inference.py

使い方（Hub モデル）:
  uv run python run_ranunculus_inference.py --from-hub

オプション:
  --model-dir      ローカルモデルのパス（デフォルト: ./models/ranunculus-1b-final）
  --repo-id        Hub のリポジトリ ID（デフォルト: yasutoshi-lab/Ranunculus-v1-1B）
  --from-hub       Hub からロードする
  --gpu            使用する GPU インデックス（デフォルト: 0）
  --max-new-tokens 最大生成トークン数（デフォルト: 200）
  --temperature    サンプリング温度（デフォルト: 0.8）
  --top-p          top-p サンプリング（デフォルト: 0.95）

終了するには "exit" または Ctrl+C を入力してください。
"""

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).parent / "models/ranunculus-1b-final"
DEFAULT_REPO_ID = "yasutoshi-lab/Ranunculus-v1-1B"


def load_model(args):
    if args.from_hub:
        print(f"[Hub] {args.repo_id} をロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.repo_id, trust_remote_code=True, token=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.repo_id,
            trust_remote_code=True,
            token=True,
            dtype=torch.bfloat16,
        )
    else:
        print(f"[Local] {args.model_dir} をロード中...")
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir), dtype=torch.bfloat16
        )

    # 訓練設定の use_cache=False を推論用に上書き
    model.config.use_cache = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, device, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.1,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    generated_ids = output_ids[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = len(generated_ids) / elapsed
    return text, len(generated_ids), elapsed, tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--from-hub", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    t_load = time.perf_counter()
    tokenizer, model, device = load_model(args)
    load_time = time.perf_counter() - t_load

    total_params = sum(p.numel() for p in model.parameters())
    print(f"デバイス : {device}")
    print(f"パラメータ: {total_params / 1e9:.3f}B")
    print(f"ロード時間: {load_time:.1f}s")
    print(f"最大生成長: {args.max_new_tokens} tokens  temperature={args.temperature}  top_p={args.top_p}")
    print("\n'exit' で終了  |  Ctrl+C でも終了できます\n")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します")
            break

        if not prompt:
            continue
        if prompt.lower() == "exit":
            print("終了します")
            break

        text, n_tok, elapsed, tps = generate(tokenizer, model, device, prompt, args)
        print(f"\n{prompt}{text}")
        print(f"\n--- {n_tok} tokens / {elapsed:.2f}s / {tps:.1f} tok/s ---")


if __name__ == "__main__":
    main()
