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
"""Freesia contrastive training in the embedding mode (design §3, stages 1a-probe / 1b / 3 / 4).

埋め込みモードで InfoNCE を学習する。主な仕様（設計書 §3）:
* task-homogeneous batching（1 ステップ = 1 データソース）
* 1 クエリ = 正例 1 + ハードネガティブ k（プールから毎回抽出）。in-batch negative は検索タスクだけ
* 偽負例マスク（正例スコア + margin を超える候補を除外）と focal 型の重み付け
* GradCache: 全系列を勾配なしで埋め込み → 損失と埋め込みへの勾配 → 小分けに再計算して逆伝播
* Bloom ゲートの下限スケジュール（0 → floor_max を最初の floor_warmup_ratio で線形に）
* 途中停止に備えて `save_minutes` ごとにチェックポイントを保存し、再実行で続きから学習する

Usage:
    python run_freesia_contrastive.py --config configs/freesia-100m-janus-probe.yaml
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from transformers import AutoTokenizer, FreesiaModel
from transformers.models.freesia.modeling_freesia import MODE_EMBED, MODE_LM, MODE_MASKED


HF_HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
INSTRUCTIONS = {
    "retrieval_ja": "質問に関連する文書を検索する",
    "retrieval_en": "Given a question, retrieve passages that answer the question",
    "nli_ja": "意味が似ている文を検索する",
    "nli_en": "Retrieve semantically similar text",
    "classification_en": "Classify the given text",
    "clustering_en": "Identify the topic or theme of the given text",
}
F2LLM_TYPES = {
    "classification": {"amazon_counterfactual", "amazon_polarity", "imdb", "toxic_conversations",
                       "tweet_sentiment_extraction", "banking77", "emotion", "massive_intent", "massive_scenario",
                       "mtop_domain", "mtop_intent", "cola"},
    "clustering": {"arxiv_clustering_s2s", "biorxiv_clustering_s2s", "medrxiv_clustering_s2s",
                   "reddit_clustering_s2s", "stackexchange_clustering_s2s", "twentynewsgroups"},
    "nli": {"snli", "mnli", "anli", "sts12", "sts22", "stsbenchmark"},
}


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            return [v]
    return [v] if v else []


def load_sources(cfg: dict, rng: random.Random) -> list[dict]:
    """Load (query, positive, negatives) triples per data source.

    Args:
        cfg (dict): Config with `ja_sources` / `en_sources` and per-language query budgets.
        rng (random.Random): RNG used for subsampling.

    Returns:
        list[dict]: Sources with `name`, `kind`, `lang`, `inbatch`, `rows`.
    """
    sources = []
    # Japanese: cl-nagoya/ruri-v3-dataset-ft (anc / pos / neg list)
    ja_files = {}
    for path in glob.glob(os.path.join(HF_HUB, "datasets--cl-nagoya--ruri-v3-dataset-ft/snapshots/*/*/*.parquet")):
        ja_files.setdefault(Path(path).parent.name, []).append(path)
    for name in sorted(ja_files):
        if cfg.get("ja_sources") and name not in cfg["ja_sources"]:
            continue
        rows = []
        for p in sorted(ja_files[name]):
            t = pq.read_table(p, columns=["anc", "pos", "neg"]).to_pylist()
            rows += [(r["anc"], r["pos"], _as_list(r["neg"])) for r in t if r["anc"] and r["pos"]]
        kind = "nli" if name == "nli" else "retrieval"
        sources.append({"name": f"ja/{name}", "kind": kind, "lang": "ja", "inbatch": kind == "retrieval", "rows": rows})
    # English: codefuse-ai/F2LLM (query / passage / negative_1..n)
    for path in sorted(glob.glob(os.path.join(HF_HUB, "datasets--codefuse-ai--F2LLM/snapshots/*/*.parquet"))):
        name = Path(path).stem
        if cfg.get("en_sources") and name not in cfg["en_sources"]:
            continue
        kind = next((k for k, v in F2LLM_TYPES.items() if name in v), "retrieval")
        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if c == "query" or c == "passage" or c.startswith("negative_")]
        cap = cfg.get("max_rows_per_source", 200_000)
        rows = []
        for batch in pf.iter_batches(batch_size=4096, columns=cols):
            for r in batch.to_pylist():
                negs = [r[c] for c in cols if c.startswith("negative_") and r[c]]
                if r["query"] and r["passage"]:
                    rows.append((r["query"], r["passage"], negs))
            if len(rows) >= cap:
                break
        sources.append({"name": f"en/{name}", "kind": kind, "lang": "en", "inbatch": kind == "retrieval", "rows": rows})

    # per-language query budget, proportional to source size
    for lang in ("ja", "en"):
        budget = cfg.get(f"{lang}_queries")
        group = [s for s in sources if s["lang"] == lang]
        total = sum(len(s["rows"]) for s in group)
        if budget and total > budget:
            for s in group:
                keep = max(1, int(len(s["rows"]) * budget / total))
                s["rows"] = rng.sample(s["rows"], min(keep, len(s["rows"])))
    sources = [s for s in sources if len(s["rows"]) >= cfg["batch_queries"]]
    for s in sources:
        print(f"source {s['name']:<40} kind={s['kind']:<14} rows={len(s['rows']):,}", flush=True)
    return sources


class Batcher:
    """Deterministic task-homogeneous batch schedule (reproducible after resume)."""

    def __init__(self, sources: list[dict], batch_queries: int, seed: int):
        self.sources = sources
        self.bq = batch_queries
        self.seed = seed
        sizes = np.array([len(s["rows"]) for s in sources], dtype=float)
        self.probs = sizes / sizes.sum()
        self.steps_per_epoch = int(sizes.sum() // batch_queries)

    def get(self, step: int, k_neg: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + step)
        si = rng.choice(len(self.sources), p=self.probs)
        src = self.sources[si]
        idx = rng.choice(len(src["rows"]), size=self.bq, replace=False)
        qs, docs = [], []
        for i in idx:
            q, p, negs = src["rows"][i]
            negs = [n for n in negs if n and n != p]
            if len(negs) >= k_neg:
                pick = [negs[j] for j in rng.choice(len(negs), size=k_neg, replace=False)]
            else:  # pad with negatives borrowed from other rows of the same source
                pick = list(negs)
                while len(pick) < k_neg:
                    other = src["rows"][rng.integers(len(src["rows"]))][1]
                    if other != p:
                        pick.append(other)
            qs.append(q)
            docs.append([p] + pick)
        return src, qs, docs


def tokenize(tok, texts: list[str], prefix: str, max_len: int):
    pre = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    body = tok(texts, add_special_tokens=False, truncation=True, max_length=max_len - len(pre) - 1)["input_ids"]
    seqs = [pre + b + [tok.eos_token_id] for b in body]
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    pool = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, x in enumerate(seqs):
        ids[r, : len(x)] = torch.tensor(x)
        att[r, : len(x)] = 1
        pool[r, len(pre) : len(x)] = 1
    return ids, att, pool


def encode_chunks(model, batch, chunk: int, grad: bool):
    ids, att, pool = batch
    outs = []
    for s in range(0, ids.size(0), chunk):
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model.encode(ids[s : s + chunk].cuda(), att[s : s + chunk].cuda(), pool[s : s + chunk].cuda()))
    return torch.cat(outs)


def contrastive_loss(q: torch.Tensor, d: torch.Tensor, k1: int, inbatch: bool, cfg: dict) -> torch.Tensor:
    """InfoNCE with false-negative masking and focal weighting.

    Args:
        q (torch.Tensor): `(B, H)` query embeddings.
        d (torch.Tensor): `(B * k1, H)` document embeddings, `k1 = 1 + k_neg`, positive first per query.
        k1 (int): Documents per query.
        inbatch (bool): Use all documents in the batch as candidates (retrieval tasks only).
        cfg (dict): `temperature`, `false_neg_margin`, `focal_gamma`.

    Returns:
        torch.Tensor: Scalar loss.
    """
    B = q.size(0)
    tau = cfg["temperature"]
    sims = q @ d.T  # (B, B*k1) cosine (both normalized)
    pos_idx = torch.arange(B, device=q.device) * k1
    pos = sims[torch.arange(B), pos_idx]
    if not inbatch:
        own = torch.zeros_like(sims, dtype=torch.bool)
        for i in range(B):
            own[i, i * k1 : (i + 1) * k1] = True
        sims = sims.masked_fill(~own, float("-inf"))
    fn = sims > (pos[:, None] + cfg["false_neg_margin"])
    fn[torch.arange(B), pos_idx] = False
    logits = (sims / tau).masked_fill(fn, float("-inf"))
    logp = F.log_softmax(logits.float(), dim=-1)[torch.arange(B), pos_idx]
    gamma = cfg.get("focal_gamma", 0.0)
    w = (1 - logp.exp()).clamp(min=0).pow(gamma).detach() if gamma > 0 else torch.ones_like(logp)
    return -(w * logp).sum() / w.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])

    ckpt = out / "checkpoint-latest"
    resume = ckpt.exists()
    model = FreesiaModel.from_pretrained(ckpt if resume else cfg["init_from"], torch_dtype=torch.float32)
    if not resume:
        src_mode = {"masked": MODE_MASKED, "lm": MODE_LM}[cfg.get("init_embed_mode_from", "masked")]
        with torch.no_grad():
            model.mode_embed.weight[MODE_EMBED] = model.mode_embed.weight[src_mode]
    model.bloom_override = cfg.get("bloom_override", "learned")
    model.cuda().train()
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    rng = random.Random(cfg.get("seed", 42))
    sources = load_sources(cfg, rng)
    batcher = Batcher(sources, cfg["batch_queries"], cfg.get("seed", 42))
    max_steps = cfg.get("max_steps") or int(batcher.steps_per_epoch * cfg.get("epochs", 1))
    print(f"steps={max_steps} (epoch={batcher.steps_per_epoch})", flush=True)

    bloom = [p for n, p in model.named_parameters() if "bloom_logit" in n]
    rest = [p for n, p in model.named_parameters() if "bloom_logit" not in n]
    opt = torch.optim.AdamW(
        [{"params": rest, "lr_mult": 1.0}, {"params": bloom, "lr_mult": cfg.get("bloom_lr_mult", 100.0), "weight_decay": 0.0}],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.999), fused=True,
    )
    step = 0
    if resume:
        st = torch.load(ckpt / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        step = st["step"]
        print(f"resumed at step {step}", flush=True)

    warm = int(max_steps * cfg.get("warmup_ratio", 0.05))
    floor_steps = max(1, int(max_steps * cfg.get("floor_warmup_ratio", 0.2)))
    log = open(out / "train_log.jsonl", "a")
    last_save, t0 = time.time(), time.time()
    k1 = 1 + cfg["k_neg"]
    while step < max_steps:
        lr = cfg["lr"] * (step + 1) / warm if step < warm else cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, max_steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        model.bloom_floor = cfg.get("floor_max", 0.3) * min(1.0, step / floor_steps)

        src, qs, docs = batcher.get(step, cfg["k_neg"])
        instr = INSTRUCTIONS[f"{src['kind']}_{src['lang']}"]
        qb = tokenize(tok, qs, f"Instruct: {instr}\nQuery:", cfg["max_query_len"])
        db = tokenize(tok, [x for group in docs for x in group], "", cfg["max_doc_len"])
        # GradCache pass 1: embeddings without graph
        with torch.no_grad():
            q_emb = encode_chunks(model, qb, cfg["chunk"], grad=False)
            d_emb = encode_chunks(model, db, cfg["chunk"], grad=False)
        q_emb.requires_grad_(True)
        d_emb.requires_grad_(True)
        loss = contrastive_loss(q_emb, d_emb, k1, src["inbatch"], cfg)
        loss.backward()
        # pass 2: recompute with graph, chunk by chunk, and inject the cached gradients
        for batch, cache in ((qb, q_emb.grad), (db, d_emb.grad)):
            ids, att, pool = batch
            for s in range(0, ids.size(0), cfg["chunk"]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = model.encode(ids[s : s + cfg["chunk"]].cuda(), att[s : s + cfg["chunk"]].cuda(), pool[s : s + cfg["chunk"]].cuda())
                e.backward(cache[s : s + cfg["chunk"]])
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.get("log_every", 10) == 0:
            gates = model.bloom_gates().detach().cpu()
            eff = model.bloom_gates(effective=True).detach().cpu()
            rec = {"step": step, "loss": loss.item(), "src": src["name"], "lr": lr, "grad_norm": float(gnorm),
                   "bloom_floor": model.bloom_floor, "gate_mean": float(gates.mean()),
                   "gate_by_layer": [round(float(x), 4) for x in gates.mean(1)],
                   "eff_gate_by_layer": [round(float(x), 4) for x in eff.mean(1)],
                   "sec_per_step": (time.time() - t0) / cfg.get("log_every", 10), "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            t0 = time.time()
        if time.time() - last_save > cfg.get("save_minutes", 30) * 60 or step == max_steps:
            tmp = out / "checkpoint-tmp"
            model.save_pretrained(tmp)
            torch.save({"optimizer": opt.state_dict(), "step": step}, tmp / "trainer_state.pt")
            if ckpt.exists():
                import shutil

                shutil.rmtree(ckpt)
            tmp.rename(ckpt)
            last_save = time.time()

    final = out / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    (final / "freesia_embed_meta.json").write_text(json.dumps(
        {"bloom_override": model.bloom_override, "bloom_floor": model.bloom_floor,
         "bloom_gates": model.bloom_gates().tolist(),
         "bloom_gates_effective": model.bloom_gates(effective=True).tolist(), "config": cfg}, ensure_ascii=False, indent=2))
    (out / "DONE").write_text(json.dumps({"step": step}))
    print("TRAINING_FINISHED", flush=True)


if __name__ == "__main__":
    main()
