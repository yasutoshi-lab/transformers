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

再現性: 入力 JSON の sha256・コードの commit・依存バージョン・トークナイザの revision・
パラメータ・出力の sha256 を ``manifest.json`` に記録する（本文は含めない）。

注意: 出力は著作物由来のため ``artifacts/`` 配下（git 追跡外）にのみ書き出す。
"""

import argparse
import collections
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from data_prep import provenance
from data_prep.books import DEFAULT_RAW_DIR, iter_books
from data_prep.corpus import HOLDOUT_PPL_RANGE, QA_EVAL_RANGE, SPLITS, SplitWriter, build_record, group_pages
from data_prep.dedup import DEFAULT_THRESHOLD, NGRAM, NUM_PERM, Deduplicator
from data_prep.glyphs import GlyphConverter, normalize_page
from data_prep.lines import clean_lines, find_running_heads
from data_prep.page_filter import filter_page
from data_prep.stats import PipelineStats


# トークン数の計測に使うトークナイザの revision（Hub の更新で n_tokens が変わらないよう固定）
TOKENIZER_REVISION = "411aa17b749aa952df1359d2dcea73917a544d9a"
PACKAGES = ["opencc", "datasketch", "tokenizers", "huggingface_hub"]


def parse_args():
    """コマンドライン引数を解釈する.

    Returns:
        argparse.Namespace: ``raw_dir / out_dir / tokenizer / tokenizer_revision / dedup_threshold``。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "artifacts" / "data"))
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")
    ap.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
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


def write_corpus(pages, out_dir, tokenizer_id, tokenizer_revision, stats):
    """S5/S6: 書籍ごとに分割・文書化して JSONL へ書き出す.

    Args:
        pages (list[tuple[str, str, int, str]]): 重複除去後のページ。
        out_dir (pathlib.Path): 出力ディレクトリ。
        tokenizer_id (str): トークン数の計測に使う HF Hub のモデル ID。
        tokenizer_revision (str): トークナイザの revision（commit sha）。
        stats (PipelineStats): 集計先。

    Returns:
        None
    """
    tok = Tokenizer.from_file(hf_hub_download(tokenizer_id, "tokenizer.json", revision=tokenizer_revision))
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


def write_manifest(args, out_dir):
    """入力・コード・環境・パラメータ・出力を ``manifest.json`` に記録する.

    Args:
        args (argparse.Namespace): 実行時の引数。
        out_dir (pathlib.Path): 出力ディレクトリ。

    Returns:
        dict: 書き出した manifest。
    """
    raw_dir = Path(args.raw_dir)
    manifest = {
        "kind": "eng-cpt corpus",
        "created_at": provenance.now_iso(),
        "code": provenance.git_info(paths=["examples/pytorch/eng-cpt"]),
        "command": "python prepare_eng_cpt_data.py " + " ".join(
            f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()),
        "environment": provenance.package_versions(PACKAGES),
        "inputs": [{"book": book, "category": category, **provenance.file_record(raw_dir / f"{book}.json", raw_dir)}
                   for category, book in iter_books()],
        "input_dir": str(raw_dir),
        "params": {
            "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
            "dedup_threshold": args.dedup_threshold, "minhash_num_perm": NUM_PERM, "minhash_ngram": NGRAM,
            "qa_eval_range": QA_EVAL_RANGE, "holdout_ppl_range": HOLDOUT_PPL_RANGE,
        },
        "outputs": [provenance.file_record(out_dir / f"{s}.jsonl", out_dir) for s in SPLITS]
                   + [provenance.file_record(out_dir / "stats.json", out_dir)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


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
    write_corpus(pages, out, args.tokenizer, args.tokenizer_revision, stats)
    summary = stats.write(out)
    write_manifest(args, out)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
