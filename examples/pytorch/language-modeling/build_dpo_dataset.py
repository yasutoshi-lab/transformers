#!/usr/bin/env python
"""Assemble the English/Japanese oasst2 DPO dataset and summarize the translation scores.

English rows are the untranslated half of the pairs; Japanese rows are the
translated half with the judge's scores attached. A Japanese row is marked
``quality_ok`` when no text was cut off by the length limit, every verdict was
parsed, nothing was judged "not a translation", and every text scored at least
3 for accuracy and completeness. English rows are always ``quality_ok``.

Usage:
    python build_dpo_dataset.py --data_dir artifacts-5/dpo-data --out_dir artifacts-5/dpo-dataset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets


def verdict_ok(v: dict) -> bool:
    return "error" not in v and not v["not_translation"] and v["accuracy"] >= 3 and v["completeness"] >= 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    d = Path(args.data_dir)
    summary = {}
    splits = {}
    for split in ("train", "validation"):
        en = [json.loads(l) for l in open(d / f"pairs_en_{split}.jsonl", encoding="utf-8")]
        ja = {r["id"]: r for r in map(json.loads, open(d / f"pairs_ja_{split}.jsonl", encoding="utf-8"))}
        judge = {r["id"]: r["verdicts"] for r in map(json.loads, open(d / f"judge_{split}.jsonl", encoding="utf-8"))}
        rows, st, scores = [], Counter(), Counter()
        for p in en:
            base = {k: p[k] for k in ("id", "chosen_rank", "rejected_rank", "n_ranked")}
            if not p["translate_to_ja"]:
                rows.append({**base, "lang": "en", "prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
                             "quality_ok": True, "judge": None})
                st["en"] += 1
                continue
            r, v = ja[p["id"]], judge[p["id"]]
            ok = (not r["truncated"]) and all(verdict_ok(x) for x in v.values())
            for x in v.values():
                st["ja_texts"] += 1
                if "error" in x:
                    st["ja_texts_unparsable"] += 1
                    continue
                for k in ("accuracy", "fluency", "completeness"):
                    scores[f"{k}={x[k]}"] += 1
                st["ja_texts_not_translation"] += bool(x["not_translation"])
            st["ja"] += 1
            st["ja_quality_ok"] += ok
            st["ja_truncated"] += r["truncated"]
            rows.append({**base, "lang": "ja", "prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"],
                         "quality_ok": ok, "judge": json.dumps(v, ensure_ascii=False)})
        splits[split] = datasets.Dataset.from_list(rows)
        summary[split] = {"counts": dict(st), "score_hist": dict(sorted(scores.items()))}
    out = Path(args.out_dir)
    datasets.DatasetDict(splits).save_to_disk(str(out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
