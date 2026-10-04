"""除去ページを理由別にサンプル表示し、除去ルールの誤判定を目視確認する.

使い方（eng-cpt/ 直下で実行）:
    python -m tools.inspect_drops                       # 全理由から各 6 件
    python -m tools.inspect_drops --reason toc -n 20    # 特定の理由だけ
    python -m tools.inspect_drops --book 溶接I --all    # 特定の書籍を全件
"""

import argparse
import collections
import json
import random
from pathlib import Path


DEFAULT_SAMPLES = Path(__file__).resolve().parent.parent / "artifacts" / "data" / "drop_samples.jsonl"


def load_samples(path, reason=None, book=None):
    """drop_samples.jsonl を読み込み、理由別にまとめる.

    Args:
        path (pathlib.Path): drop_samples.jsonl のパス。
        reason (str | None): 指定した理由だけに絞る。
        book (str | None): 指定した書籍だけに絞る。

    Returns:
        dict[str, list[dict]]: 理由→サンプルのリスト（件数の多い順）。
    """
    by_reason = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (reason and r["reason"] != reason) or (book and r["book"] != book):
                continue
            by_reason[r["reason"]].append(r)
    return dict(sorted(by_reason.items(), key=lambda kv: -len(kv[1])))


def main():
    """引数に従って除去サンプルを表示する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--reason")
    ap.add_argument("--book")
    ap.add_argument("-n", type=int, default=6, help="理由ごとの表示件数")
    ap.add_argument("--all", action="store_true", help="サンプリングせず全件表示")
    ap.add_argument("--width", type=int, default=150, help="抜粋の表示文字数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    for reason, rows in load_samples(args.samples, args.reason, args.book).items():
        print(f"\n######## {reason} ({len(rows)})")
        shown = rows if args.all else random.sample(rows, min(args.n, len(rows)))
        for r in shown:
            print(f"--[{r['book']} p{r['page']}] {r['head'][: args.width]!r}")


if __name__ == "__main__":
    main()
