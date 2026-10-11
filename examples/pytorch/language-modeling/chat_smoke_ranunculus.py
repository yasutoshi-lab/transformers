#!/usr/bin/env python
"""Generate replies for a fixed set of prompts and save inputs/outputs as JSON.

With ``--mode chat`` the prompt is wrapped by the chat template (SFT model). With
``--mode raw`` the prompt is fed as plain text (PT model, which never saw the
chat format).

Usage:
    python chat_smoke_ranunculus.py --model artifacts-4/models/ranunculus-v1-1b-sft \
        --out artifacts-4/eval/chat_smoke.json
"""

from __future__ import annotations

import argparse
import json

import torch

from transformers import AutoTokenizer, RanunculusForCausalLM

PROMPTS = [
    "日本の首都はどこですか？",
    "自己紹介をしてください。",
    "カレーライスの作り方を簡単に教えてください。",
    "光合成とは何か、小学生にもわかるように説明してください。",
    "次の文を英語に翻訳してください：今日はとても良い天気です。",
    "健康のためにできる習慣を3つ挙げてください。",
    "What is the capital of France?",
    "Explain what a neural network is in two sentences.",
    "Give me three tips for writing clean code.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--mode", choices=["chat", "raw"], default="chat")
    ap.add_argument("--enable_thinking", choices=["default", "true", "false"], default="default",
                    help="passed to the chat template (v2 template only); default leaves it unset")
    ap.add_argument("--max_new_tokens_think", type=int, default=0, help="override max_new_tokens when thinking")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = RanunculusForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    settings = {
        "greedy": dict(do_sample=False, repetition_penalty=1.1),
        "sample": dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1),
    }
    eos = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if i is not None]
    rows = []
    for p in PROMPTS:
        if args.mode == "chat":
            kw = {} if args.enable_thinking == "default" else {"enable_thinking": args.enable_thinking == "true"}
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, **kw)
        else:
            text = p + "\n"
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        row = {"model": args.model, "mode": args.mode, "enable_thinking": args.enable_thinking, "prompt": p, "input_text": text}
        for name, kw in settings.items():
            torch.manual_seed(0)
            out = model.generate(ids, max_new_tokens=args.max_new_tokens_think or args.max_new_tokens, eos_token_id=eos, **kw)
            gen = out[0, ids.shape[1]:].tolist()
            row[name] = tok.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            row[f"{name}_stopped_by_im_end"] = tok.convert_tokens_to_ids("<|im_end|>") in gen
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    meta = {"settings": settings, "max_new_tokens": args.max_new_tokens_think or args.max_new_tokens, "enable_thinking": args.enable_thinking, "seed": 0, "dtype": "bfloat16"}
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
