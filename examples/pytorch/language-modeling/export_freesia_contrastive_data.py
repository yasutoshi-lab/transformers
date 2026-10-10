#!/usr/bin/env python
"""Export the exact contrastive training data of a Freesia run as parquet (for reproducible re-training).

`run_freesia_contrastive.py` と同じ読み込み関数・同じ乱数（config の seed）でデータを組み立て、
学習に使った (query, positive, negatives) をデータソースごとの parquet に書き出す。行の順序も学習時と同じなので、
同じ config・seed で Batcher に渡せば同じバッチ順を再現できる。`sources.json` にソース名・種類・件数・順序を残す。

Usage:
    python export_freesia_contrastive_data.py --config configs/freesia-300m-weak.yaml --out-dir artifacts/export/weak
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from run_freesia_contrastive import INSTRUCTIONS, load_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rng = random.Random(cfg.get("seed", 42))  # same RNG stream as training
    sources = load_sources(cfg, rng)
    out = args.out_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    meta = []
    for order, s in enumerate(sources):
        fname = s["name"].replace("/", "__") + ".parquet"
        table = pa.table({
            "query": [r[0] for r in s["rows"]],
            "positive": [r[1] for r in s["rows"]],
            "negatives": pa.array([list(r[2]) for r in s["rows"]], type=pa.list_(pa.string())),
        })
        pq.write_table(table, out / "data" / fname, compression="zstd")
        meta.append({"order": order, "name": s["name"], "kind": s["kind"], "lang": s["lang"],
                     "inbatch": s["inbatch"], "rows": len(s["rows"]), "file": f"data/{fname}"})
        print(f"wrote {fname} rows={len(s['rows']):,}", flush=True)
        s["rows"] = None  # free memory as we go
    (out / "sources.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out / "instructions.json").write_text(json.dumps(INSTRUCTIONS, ensure_ascii=False, indent=2))
    (out / "train_config.yaml").write_text(args.config.read_text())
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
