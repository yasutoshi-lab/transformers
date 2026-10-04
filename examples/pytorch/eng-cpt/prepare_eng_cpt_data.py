"""工学系教科書 OCR JSON から軽量 CPT 用コーパスを構築する前処理パイプライン.

Build a light continual-pretraining (CPT) corpus from OCR'd engineering textbooks.

入力は Gaia の ``raw/books/<title>.json``（``[{name, page, content}]`` 形式の OCR 結果）。
各ステージの処理は ``data_prep/`` のモジュールに分かれており、本スクリプトは
それらを順に呼び出すだけのエントリポイントである。

Stages:
    S1 文字正規化   (data_prep.glyphs)      : NFKC・簡体字→日本字体・OCR 誤字補正
    S2 行クリーニング(data_prep.lines)       : 柱・ページ番号行の除去、段組み改行の接合
    S3 ページ除去   (data_prep.page_filter) : 奥付・目次・索引・図主体・言語比率など
    S4 重複除去     (data_prep.dedup)       : 完全一致ハッシュ + MinHash 近似重複
    S5 分割         (data_prep.corpus)      : 書籍ごとの連続ブロックで 3 分割
    S6 文書化       (data_prep.corpus)      : 連続ページを結合して JSONL 出力

注意: 出力は著作物由来のため ``artifacts/`` 配下（git 追跡外）にのみ書き出す。
"""

import argparse
import collections
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from data_prep.books import DEFAULT_RAW_DIR, iter_books
from data_prep.corpus import SplitWriter, build_record, group_pages
from data_prep.dedup import DEFAULT_THRESHOLD, Deduplicator
from data_prep.glyphs import GlyphConverter, normalize_page
from data_prep.lines import clean_lines, find_running_heads
from data_prep.page_filter import filter_page
from data_prep.stats import PipelineStats


def parse_args():
    """コマンドライン引数を解釈する.

    Returns:
        argparse.Namespace: ``raw_dir / out_dir / tokenizer / dedup_threshold``。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "artifacts" / "data"))
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")
    ap.add_argument("--dedup-threshold", type=float, default=DEFAULT_THRESHOLD)
    return ap.parse_args()


def clean_books(raw_dir, stats):
    """S1〜S3: 全書籍を読み込み、正規化・行クリーニング・ページ除去を行う.

    Args:
        raw_dir (str): OCR JSON のディレクトリ。
        stats (PipelineStats): 集計先。

    Returns:
        list[tuple[str, str, int, str]]: 残ったページ ``(category, book, page_no, text)``。

    Raises:
        FileNotFoundError: 対象書籍の JSON が存在しない場合。
    """
    converter = GlyphConverter()
    kept = []
    for category, book in iter_books():
        raw = json.loads((Path(raw_dir) / f"{book}.json").read_text())
        bs = stats.start_book(book, raw)
        norm = [(int(p["page"]), normalize_page(p["content"] or "", converter, stats.global_counts)) for p in raw]
        heads = find_running_heads([t for _, t in norm])
        for page_no, text in norm:
            text = clean_lines(text, heads, stats.global_counts)
            reason, text, salvaged = filter_page(text)
            if salvaged:
                bs["salvaged_figure_pages"] += 1
            if reason:
                stats.record_drop(book, page_no, reason, text)
                continue
            kept.append((category, book, page_no, text))
    return kept


def deduplicate(pages, threshold, stats):
    """S4: 完全一致・近似重複のページを除去する.

    Args:
        pages (list[tuple[str, str, int, str]]): ``clean_books`` の結果。
        threshold (float): MinHash LSH の Jaccard 閾値。
        stats (PipelineStats): 集計先。

    Returns:
        list[tuple[str, str, int, str]]: 重複を除いたページ。
    """
    dedup = Deduplicator(threshold)
    kept = []
    for category, book, page_no, text in pages:
        verdict, dup_of = dedup.check(f"{book}:{page_no}", text)
        if verdict == "exact_dup":
            stats.books[book]["drop_exact_dup"] += 1
            continue
        if verdict == "near_dup":
            stats.record_drop(book, page_no, "near_dup", text, dup_of=dup_of)
            continue
        kept.append((category, book, page_no, text))
    return kept


def write_corpus(pages, out_dir, tokenizer_id, stats):
    """S5/S6: 書籍ごとに分割・文書化して JSONL へ書き出す.

    Args:
        pages (list[tuple[str, str, int, str]]): 重複除去後のページ。
        out_dir (pathlib.Path): 出力ディレクトリ。
        tokenizer_id (str): トークン数の計測に使う HF Hub のモデル ID。
        stats (PipelineStats): 集計先。

    Returns:
        None
    """
    tok = Tokenizer.from_file(hf_hub_download(tokenizer_id, "tokenizer.json"))
    count_tokens = lambda text: len(tok.encode(text, add_special_tokens=False).ids)  # noqa: E731
    by_book = collections.defaultdict(list)
    for row in pages:
        by_book[row[1]].append(row)
    with SplitWriter(out_dir) as writer:
        for book, rows in by_book.items():
            for gi, group in enumerate(group_pages(rows)):
                record = build_record(book, gi, group, count_tokens)
                writer.write(record)
                stats.record_doc(record, len(group["pages"]))


def main():
    """パイプライン全体（S1〜S6）を実行し、要約を標準出力へ表示する.

    Returns:
        None
    """
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = PipelineStats()
    pages = clean_books(args.raw_dir, stats)
    pages = deduplicate(pages, args.dedup_threshold, stats)
    write_corpus(pages, out, args.tokenizer, stats)
    summary = stats.write(out)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
