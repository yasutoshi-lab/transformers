"""JIS 外漢字（簡体字・OCR 誤字）の混入を監査する.

2 つのモードがある。

- ``--stage raw``（既定）: 生 OCR に含まれる JIS 外漢字を頻度順に文脈付きで列挙し、
  OpenCC で解決できるか・``OCR_CHAR_FIXES`` に登録済みかを表示する。
  対応表に足すべき字を洗い出すのに使う。
- ``--stage output``: 出力 JSONL に JIS 外漢字とげた記号（〓）が残っていないかを検査する。

使い方（eng-cpt/ 直下で実行）:
    python -m tools.audit_glyphs --unresolved-only
    python -m tools.audit_glyphs --stage output
"""

import argparse
import collections
import json
import unicodedata
from pathlib import Path

from data_prep.books import DEFAULT_RAW_DIR, iter_books
from data_prep.corpus import SPLITS
from data_prep.glyphs import OCR_CHAR_FIXES, OCR_UNKNOWN_MARK, GlyphConverter, is_cjk_ideograph, is_jis


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "data"
CONTEXT = 8


def _context(text, i):
    """i 文字目の前後を 1 行に整えて返す.

    Args:
        text (str): 対象テキスト。
        i (int): 注目する文字の位置。

    Returns:
        str: 前後 ``CONTEXT`` 文字の抜粋（改行は空白に置換）。
    """
    return text[max(0, i - CONTEXT) : i + CONTEXT].replace("\n", " ")


def audit_raw(raw_dir, unresolved_only):
    """生 OCR の JIS 外漢字を頻度順に表示する.

    Args:
        raw_dir (str): OCR JSON のディレクトリ。
        unresolved_only (bool): OpenCC で解決できない字だけを表示する。

    Returns:
        None
    """
    converter = GlyphConverter()
    counts, contexts = collections.Counter(), collections.defaultdict(list)
    for _, book in iter_books():
        for page in json.loads((Path(raw_dir) / f"{book}.json").read_text()):
            text = unicodedata.normalize("NFKC", page["content"] or "")
            for i, ch in enumerate(text):
                if not is_cjk_ideograph(ch) or is_jis(ch):
                    continue
                if unresolved_only and is_jis(converter(ch)):
                    continue
                counts[ch] += 1
                if len(contexts[ch]) < 3:
                    contexts[ch].append(_context(text, i))
    print(f"種類 {len(counts)} / 出現 {sum(counts.values())}")
    print("字\tコード\t件数\tOpenCC\t対応表\t文脈")
    for ch, k in counts.most_common():
        conv = converter(ch)
        opencc_col = conv if is_jis(conv) else "-"
        fix_col = OCR_CHAR_FIXES.get(ch, "-")
        print(f"{ch}\tU+{ord(ch):04X}\t{k}\t{opencc_col}\t{fix_col}\t" + " | ".join(contexts[ch]))


def audit_output(data_dir):
    """出力 JSONL に JIS 外漢字・げた記号が残っていないか検査する.

    Args:
        data_dir (pathlib.Path): 前処理の出力ディレクトリ。

    Returns:
        None
    """
    for sp in SPLITS:
        non_jis, marks = 0, []
        with open(data_dir / f"{sp}.jsonl") as f:
            for line in f:
                text = json.loads(line)["text"]
                for i, ch in enumerate(text):
                    if is_cjk_ideograph(ch) and not is_jis(ch):
                        non_jis += 1
                    elif ch == OCR_UNKNOWN_MARK:
                        marks.append(_context(text, i))
        print(f"{sp}: JIS外漢字 {non_jis} / げた記号 {len(marks)}")
        for m in marks:
            print(f"    〓 {m!r}")


def main():
    """引数に従って監査を実行する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["raw", "output"], default="raw")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--unresolved-only", action="store_true", help="OpenCC で解決できない字だけ表示")
    args = ap.parse_args()
    if args.stage == "raw":
        audit_raw(args.raw_dir, args.unresolved_only)
    else:
        audit_output(args.data_dir)


if __name__ == "__main__":
    main()
