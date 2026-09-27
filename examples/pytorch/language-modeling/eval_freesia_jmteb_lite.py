#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quick JMTEB-lite evaluation for Freesia checkpoints (stage 1 comparisons, design §5).

段階1 のアーム比較用の簡易評価。`sbintuitions/JMTEB-lite` の検索 5 タスク（nDCG@10）と STS 2 タスク
（Spearman）の test 分割だけを使う。正式な JMTEB の点数ではないので、アーム間の相対比較にだけ使う。

Usage:
    python eval_freesia_jmteb_lite.py --model artifacts/runs/xxx/final --out artifacts/runs/xxx/jmteb_lite.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import glob
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

from transformers import AutoTokenizer, FreesiaModel


HF_HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))


def _table(config: str, split: str) -> dict:
    """Read one JMTEB-lite parquet (downloaded snapshot) as a column dict."""
    path = glob.glob(os.path.join(HF_HUB, f"datasets--sbintuitions--JMTEB-lite/snapshots/*/data/{config}/{split}.parquet"))[0]
    return pq.read_table(path).to_pydict()


RETRIEVAL_TASKS = ["jaqket", "mrtydi", "jagovfaqs_22k", "miracl-retrieval", "jacwir-retrieval"]
STS_TASKS = ["jsts", "jsick"]
QUERY_INSTRUCTION = "Instruct: 質問に関連する文書を検索する\nQuery:"
STS_INSTRUCTION = "Instruct: 意味が似ている文を検索する\nQuery:"


@torch.no_grad()
def encode(model, tok, texts: list[str], prefix: str = "", batch_size: int = 128, max_len: int = 512) -> np.ndarray:
    """Encode texts; the instruction prefix is attended to but excluded from pooling.

    Args:
        model (FreesiaModel): Model in eval mode on CUDA.
        tok: Tokenizer.
        texts (list[str]): Input texts.
        prefix (str): Instruction prefix ("" for documents).
        batch_size (int): Batch size.
        max_len (int): Max tokens.

    Returns:
        np.ndarray: `(len(texts), hidden)` float32 embeddings.
    """
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    eos = tok.eos_token_id
    out = []
    order = np.argsort([-len(t) for t in texts])
    for s in range(0, len(texts), batch_size):
        idx = order[s : s + batch_size]
        body = tok([texts[i] for i in idx], add_special_tokens=False, truncation=True,
                   max_length=max_len - len(pre_ids) - 1)["input_ids"]
        seqs = [pre_ids + b + [eos] for b in body]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        pool = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, : len(x)] = torch.tensor(x)
            att[r, : len(x)] = 1
            pool[r, len(pre_ids) : len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.encode(ids.cuda(), att.cuda(), pool.cuda())
        out.append((idx, e.float().cpu().numpy()))
    res = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
    for idx, e in out:
        res[idx] = e
    return res


def _parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:  # noqa: BLE001
        return [v]


def ndcg_at_10(scores: np.ndarray, doc_ids: list, relevant: list[set]) -> float:
    vals = []
    for i, rel in enumerate(relevant):
        top = np.argsort(-scores[i])[:10]
        dcg = sum(1.0 / math.log2(r + 2) for r, j in enumerate(top) if doc_ids[j] in rel)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(rel), 10)))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals))


def run_retrieval(model, tok, task: str) -> float:
    q = _table(f"{task}-query", "test")
    c = _table(f"{task}-corpus", "corpus")
    titles = c.get("title") or [""] * len(c["text"])
    texts = [((ti or "") + " " + tx).strip() for ti, tx in zip(titles, c["text"])]
    doc_ids = [str(x) for x in c["docid"]]
    d_emb = encode(model, tok, texts)
    q_emb = encode(model, tok, list(q["query"]), QUERY_INSTRUCTION)
    rel = [set(str(x) for x in _parse_list(v)) for v in q["relevant_docs"]]
    scores = []
    for s in range(0, len(q_emb), 256):
        scores.append(torch.from_numpy(q_emb[s : s + 256]).cuda() @ torch.from_numpy(d_emb).cuda().T)
    scores = torch.cat(scores).float().cpu().numpy()
    return ndcg_at_10(scores, doc_ids, rel)


def run_sts(model, tok, task: str) -> float:
    ds = _table(task, "test")
    a = encode(model, tok, list(ds["sentence1"]), STS_INSTRUCTION)
    b = encode(model, tok, list(ds["sentence2"]), STS_INSTRUCTION)
    sims = (a * b).sum(1)
    return float(spearmanr(sims, np.asarray(ds["label"], dtype=float)).correlation)


def evaluate_model(model, tok) -> dict:
    """Run all quick tasks.

    Args:
        model (FreesiaModel): Model (eval mode, CUDA).
        tok: Tokenizer.

    Returns:
        dict: Per-task scores and averages.
    """
    res = {}
    for t in RETRIEVAL_TASKS:
        res[f"retrieval/{t}"] = run_retrieval(model, tok, t)
        print(t, res[f"retrieval/{t}"], flush=True)
    for t in STS_TASKS:
        res[f"sts/{t}"] = run_sts(model, tok, t)
        print(t, res[f"sts/{t}"], flush=True)
    res["avg_retrieval"] = float(np.mean([res[f"retrieval/{t}"] for t in RETRIEVAL_TASKS]))
    res["avg_sts"] = float(np.mean([res[f"sts/{t}"] for t in STS_TASKS]))
    res["avg_all"] = float(np.mean([v for k, v in res.items() if "/" in k]))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--bloom-override", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = FreesiaModel.from_pretrained(args.model).cuda().eval()
    meta = json.loads((args.model / "freesia_embed_meta.json").read_text()) if (args.model / "freesia_embed_meta.json").exists() else {}
    model.bloom_override = args.bloom_override or meta.get("bloom_override", "learned")
    model.bloom_floor = float(meta.get("bloom_floor", 0.0))
    res = {"model": str(args.model), "bloom_override": model.bloom_override, **evaluate_model(model, tok)}
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
