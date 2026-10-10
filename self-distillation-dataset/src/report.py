"""生成済み JSONL を現行フィルタルールで再評価し、統計を出して HF へアップロードする。

パイロット実行中にルールが変わった場合（例: thinking 英語 only の追加）、既存ファイルに
再適用して「適用後」の通過率を出すためのツール。生成は再実行しない。

実行例:
  python3 src/report.py --input out/pilot.jsonl --upload-to-hf yasutoshi-lab/gpt-oss-20b-self-distillation-pilot
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import dedup
import filters


def reevaluate(path: Path, threshold: float = 0.80) -> tuple[list[dict], dict]:
    """JSONL を現行ルールで再評価する。

    Args:
        path: 入力 JSONL。
        threshold: 近似重複の Jaccard 閾値。

    Returns:
        (レコード list, 統計 dict)。レコードには `passed` / `reasons` / `duplicate` が
        現行ルールで上書きされる。
    """
    records: list[dict] = []
    skipped = 0
    # splitlines() は U+2028/U+2029/U+0085 等でも切れるため、JSON 文字列内の
    # 行区切り文字でレコードが分断される。'\n' だけで切る。
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1  # 进程中断で切られた行
    if skipped:
        print(f"[skip] {skipped} malformed line(s)")
    deduper = dedup.Deduplicator(threshold=threshold)
    reasons: Counter = Counter()
    by_mode: Counter = Counter()
    by_domain: Counter = Counter()

    for rec in records:
        verdict = filters.evaluate(rec)
        rec["passed"] = verdict.passed
        rec["reasons"] = verdict.reasons
        rec["duplicate"] = None
        by_mode[(rec["mode"], verdict.passed)] += 1
        by_domain[(rec["domain"], verdict.passed)] += 1
        if not verdict.passed:
            for r in verdict.reasons:
                reasons[r.split(":")[0]] += 1
        elif deduper.is_duplicate(rec["domain"], rec["prompt"], rec["response"]):
            rec["duplicate"] = "exact_or_near"
            reasons["duplicate"] += 1

    stats = {
        "total": len(records),
        "malformed": skipped,
        "passed": sum(1 for r in records if r["passed"] and not r["duplicate"]),
        "reasons": dict(reasons),
        "by_mode": {f"{m}:{p}": c for (m, p), c in sorted(by_mode.items())},
        "by_domain": {f"{d}:{p}": c for (d, p), c in sorted(by_domain.items())},
    }
    return records, stats


def main() -> None:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="JSONL 再評価・統計・HF アップロード")
    parser.add_argument("--input", required=True)
    parser.add_argument("--upload-to-hf", default="")
    parser.add_argument("--dedup-threshold", type=float, default=0.80)
    args = parser.parse_args()

    records, stats = reevaluate(Path(args.input), args.dedup_threshold)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.upload_to_hf:
        from datasets import Dataset  # noqa: PLC0415

        Dataset.from_list(records).push_to_hub(args.upload_to_hf)
        print(f"uploaded: {args.upload_to_hf}")


if __name__ == "__main__":
    main()
