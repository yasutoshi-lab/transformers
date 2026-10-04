"""stats.json から書籍別・全体の前処理統計を表形式で出力する.

前処理レポートに貼る数値表の元データを作る。``--format md`` で Markdown 表を出す。

使い方（eng-cpt/ 直下で実行）:
    python -m tools.book_stats
    python -m tools.book_stats --format md
"""

import argparse
import json
import statistics
from pathlib import Path

from data_prep.corpus import SPLITS


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "data"
COLUMNS = ["書籍", "生ページ", "除去", "除去率", "train_tok", "qa_eval_tok", "holdout_tok"]


def book_rows(books):
    """書籍ごとの集計行を作る.

    Args:
        books (dict[str, dict]): stats.json の ``books``。

    Returns:
        list[list]: ``COLUMNS`` に対応する値のリスト。
    """
    rows = []
    for book, s in books.items():
        dropped = sum(v for k, v in s.items() if k.startswith("drop_"))
        rows.append([
            book,
            s["pages_raw"],
            dropped,
            f"{dropped / s['pages_raw']:.1%}",
            s.get("train_tokens", 0),
            s.get("qa_eval_tokens", 0),
            s.get("holdout_ppl_tokens", 0),
        ])
    return rows


def doc_length_stats(data_dir):
    """分割ごとの文書数とトークン長の分布を計算する.

    Args:
        data_dir (pathlib.Path): 前処理の出力ディレクトリ。

    Returns:
        dict[str, dict[str, int]]: 分割名→``docs / min / median / max``。
    """
    out = {}
    for sp in SPLITS:
        with open(data_dir / f"{sp}.jsonl") as f:
            toks = [json.loads(line)["n_tokens"] for line in f]
        out[sp] = {"docs": len(toks), "min": min(toks), "median": int(statistics.median(toks)), "max": max(toks)}
    return out


def print_table(rows, fmt):
    """行のリストを TSV か Markdown 表で表示する.

    Args:
        rows (list[list]): ``COLUMNS`` に対応する値のリスト。
        fmt (str): ``"tsv"`` または ``"md"``。

    Returns:
        None
    """
    if fmt == "md":
        print("| " + " | ".join(COLUMNS) + " |")
        print("|" + "---|" * len(COLUMNS))
        for r in rows:
            print("| " + " | ".join(str(v) for v in r) + " |")
    else:
        print("\t".join(COLUMNS))
        for r in rows:
            print("\t".join(str(v) for v in r))


def main():
    """統計表・全体要約・文書長分布を表示する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--format", choices=["tsv", "md"], default="tsv")
    args = ap.parse_args()

    stats = json.loads((args.data_dir / "stats.json").read_text())
    print_table(book_rows(stats["books"]), args.format)
    print("\n# 全体要約")
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=1))
    print("\n# 文書長（トークン）")
    for sp, d in doc_length_stats(args.data_dir).items():
        print(f"{sp}: docs={d['docs']} min={d['min']} median={d['median']} max={d['max']}")


if __name__ == "__main__":
    main()
