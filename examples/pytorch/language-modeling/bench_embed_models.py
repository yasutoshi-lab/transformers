#!/usr/bin/env python
"""Compare Freesia with similar-size embedding models on Japanese (JMTEB-lite) and English (MTEB subset) tasks.

日本語は `sbintuitions/JMTEB-lite` の検索 5 タスク（nDCG@10）と STS 2 タスク（Spearman）、
英語は MTEB の検索 3 タスク（SciFact・TREC-COVID・SCIDOCS、BEIR 形式・nDCG@10）と STS 3 タスク
（SICK-R・STS13・STS15、Spearman）。全モデルを同じデータ・同じ前処理・同じ指標・同じ最大長（512 トークン）で測る。
各モデルのクエリ／文書の接頭辞は、それぞれの公式モデルカードの指定に従う（MODELS）。

Freesia は `--backend freesia`（fork の venv）、他は `--backend st`（sentence-transformers の venv）で実行する。

Usage:
    python bench_embed_models.py --backend st --model intfloat/multilingual-e5-base --out results/e5.json
    python bench_embed_models.py --backend freesia --model artifacts/runs/freesia-300m-sup/final --out results/freesia.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr


HF_HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
MAX_LEN = 512

JA_RETRIEVAL = ["jaqket", "mrtydi", "jagovfaqs_22k", "miracl-retrieval", "jacwir-retrieval"]
JA_STS = ["jsts", "jsick"]
EN_RETRIEVAL = ["scifact", "trec-covid", "scidocs"]
EN_STS = ["sickr-sts", "sts13-sts", "sts15-sts"]

# prompts from each model card: (retrieval query, retrieval document, STS)
MODELS = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: ", "query: "),
    "cl-nagoya/ruri-v3-310m": ("検索クエリ: ", "検索文書: ", ""),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: ", "task: sentence similarity | query: "),
    "ibm-granite/granite-embedding-278m-multilingual": ("", "", ""),
}
FREESIA_PROMPTS = {
    "ja": ("Instruct: 質問に関連する文書を検索する\nQuery:", "", "Instruct: 意味が似ている文を検索する\nQuery:"),
    "en": ("Instruct: Given a question, retrieve passages that answer the question\nQuery:", "",
           "Instruct: Retrieve semantically similar text\nQuery:"),
}


# ---------- data ----------
def _snapshot(repo: str) -> str:
    return glob.glob(os.path.join(HF_HUB, f"datasets--{repo.replace('/', '--')}/snapshots/*"))[0]


def _parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:  # noqa: BLE001
        return [v]


def load_ja_retrieval(task: str):
    root = _snapshot("sbintuitions/JMTEB-lite")
    q = pq.read_table(f"{root}/data/{task}-query/test.parquet").to_pydict()
    c = pq.read_table(f"{root}/data/{task}-corpus/corpus.parquet").to_pydict()
    titles = c.get("title") or [""] * len(c["text"])
    docs = [((ti or "") + " " + tx).strip() for ti, tx in zip(titles, c["text"])]
    rels = [{str(x): 1 for x in _parse_list(v)} for v in q["relevant_docs"]]
    return list(q["query"]), [str(x) for x in c["docid"]], docs, rels


def load_ja_sts(task: str):
    t = pq.read_table(f"{_snapshot('sbintuitions/JMTEB-lite')}/data/{task}/test.parquet").to_pydict()
    return list(t["sentence1"]), list(t["sentence2"]), np.asarray(t["label"], dtype=float)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_en_retrieval(task: str):
    root = _snapshot(f"mteb/{task}")
    qrels = {}
    with open(f"{root}/qrels/test.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")[:3]
            if int(float(score)) > 0:
                qrels.setdefault(qid, {})[did] = int(float(score))
    queries = {r["_id"]: r["text"] for r in _read_jsonl(f"{root}/queries.jsonl")}
    corpus = _read_jsonl(f"{root}/corpus.jsonl")
    qids = [q for q in qrels if q in queries]
    docs = [((r.get("title") or "") + " " + (r.get("text") or "")).strip() for r in corpus]
    return [queries[q] for q in qids], [r["_id"] for r in corpus], docs, [qrels[q] for q in qids]


def load_en_sts(task: str):
    root = _snapshot(f"mteb/{task}")
    files = sorted(glob.glob(f"{root}/**/test*.jsonl*", recursive=True)) + sorted(glob.glob(f"{root}/**/test*.parquet", recursive=True))
    path = files[0]
    if path.endswith(".parquet"):
        rows = pq.read_table(path).to_pylist()
    else:
        import gzip
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return [r["sentence1"] for r in rows], [r["sentence2"] for r in rows], np.asarray([r["score"] for r in rows], dtype=float)


# ---------- metrics ----------
def ndcg_at_10(scores: torch.Tensor, doc_ids: list, rels: list[dict]) -> float:
    top = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
    vals = []
    for i, rel in enumerate(rels):
        dcg = sum(rel.get(doc_ids[j], 0) / math.log2(r + 2) for r, j in enumerate(top[i]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals))


# ---------- encoders ----------
class STEncoder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        self.model.max_seq_length = MAX_LEN
        self.prompts = MODELS[name]

    def __call__(self, texts, kind: str, lang: str) -> np.ndarray:
        prefix = {"query": self.prompts[0], "doc": self.prompts[1], "sts": self.prompts[2]}[kind]
        e = self.model.encode([prefix + t for t in texts], batch_size=128, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return e.astype(np.float32)


class FreesiaEncoder:
    def __init__(self, path: str):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_freesia_jmteb_lite import encode
        from transformers import AutoTokenizer, FreesiaModel
        self._encode = encode
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = FreesiaModel.from_pretrained(path).cuda().eval()
        meta_path = Path(path) / "freesia_embed_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.model.bloom_override = meta.get("bloom_override", "closed")
        self.model.bloom_floor = float(meta.get("bloom_floor", 0.0))

    def __call__(self, texts, kind: str, lang: str) -> np.ndarray:
        q, d, s = FREESIA_PROMPTS[lang]
        prefix = {"query": q, "doc": d, "sts": s}[kind]
        e = self._encode(self.model, self.tok, list(texts), prefix, max_len=MAX_LEN)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)


# ---------- runner ----------
def run_retrieval(enc, lang, queries, doc_ids, docs, rels) -> float:
    d = torch.from_numpy(enc(docs, "doc", lang)).cuda()
    q = torch.from_numpy(enc(queries, "query", lang)).cuda()
    scores = torch.cat([q[s : s + 256] @ d.T for s in range(0, len(q), 256)])
    return ndcg_at_10(scores, doc_ids, rels)


def run_sts(enc, lang, s1, s2, gold) -> float:
    a, b = enc(s1, "sts", lang), enc(s2, "sts", lang)
    return float(spearmanr((a * b).sum(1), gold).correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["st", "freesia"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--langs", nargs="+", default=["ja", "en"])
    args = parser.parse_args()
    enc = STEncoder(args.model) if args.backend == "st" else FreesiaEncoder(args.model)
    res = {"model": args.model, "max_len": MAX_LEN}
    t0 = time.time()
    plan = []
    if "ja" in args.langs:
        plan += [("ja", "retrieval", t) for t in JA_RETRIEVAL] + [("ja", "sts", t) for t in JA_STS]
    if "en" in args.langs:
        plan += [("en", "retrieval", t) for t in EN_RETRIEVAL] + [("en", "sts", t) for t in EN_STS]
    for lang, kind, task in plan:
        with torch.no_grad():
            if kind == "retrieval":
                data = load_ja_retrieval(task) if lang == "ja" else load_en_retrieval(task)
                score = run_retrieval(enc, lang, *data)
            else:
                data = load_ja_sts(task) if lang == "ja" else load_en_sts(task)
                score = run_sts(enc, lang, *data)
        res[f"{lang}/{kind}/{task}"] = score
        print(f"[{time.time() - t0:7.0f}s] {lang}/{kind}/{task} {score:.4f}", flush=True)
    for lang in args.langs:
        for kind in ("retrieval", "sts"):
            vals = [v for k, v in res.items() if k.startswith(f"{lang}/{kind}/")]
            res[f"{lang}/avg_{kind}"] = float(np.mean(vals))
        res[f"{lang}/avg_all"] = float(np.mean([v for k, v in res.items() if k.startswith(f"{lang}/") and k.count("/") == 2]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("BENCH_DONE", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
