"""S2 行クリーニング: 柱・ページ番号行の除去と段組み改行の接合."""

import collections
import re


# 柱の検出: ページ先頭/末尾の各 EDGE_LINES 行のうち、HEAD_MAX_LEN 文字以下で
# 数字を伏せた形が書籍内で HEAD_MIN_PAGES ページ以上に現れる行を柱とみなす
EDGE_LINES = 2
HEAD_MAX_LEN = 30
HEAD_MIN_PAGES = 4
# 段組み改行の接合: 前行が JOIN_MIN_PREV_LEN 文字以上で文末記号なし、
# かつ次行がひらがな・句読点で始まる場合に連結する
JOIN_MIN_PREV_LEN = 10

RE_PAGE_NUM = re.compile(r"^\s*[-—]?\s*\d{1,3}\s*[-—]?\s*$")
RE_CONT_HEAD = re.compile(r"^[ぁ-ん、。,.]")
SENT_END = tuple("。．.！？!?）」』】")


def _head_key(line):
    """柱判定用に行中の数字を伏せた正規形を返す.

    Args:
        line (str): 対象行。

    Returns:
        str: 数字列を ``#`` に置換した文字列。
    """
    return re.sub(r"\d+", "#", line)


def find_running_heads(pages):
    """書籍内で繰り返し現れるページ先頭/末尾行（柱）を検出する.

    Args:
        pages (list[str]): 正規化済みページ本文のリスト。

    Returns:
        set[str]: 柱とみなす行（数字を伏せた正規形）の集合。
    """
    counter = collections.Counter()
    for text in pages:
        lines = [ln for ln in text.split("\n") if ln]
        for ln in lines[:EDGE_LINES] + lines[-EDGE_LINES:]:
            if len(ln) <= HEAD_MAX_LEN:
                counter[_head_key(ln)] += 1
    return {k for k, v in counter.items() if v >= HEAD_MIN_PAGES}


def _join_column_breaks(lines, stats):
    """段組み OCR で文中改行された行を前の行へ接合する.

    Args:
        lines (list[str]): 柱・ページ番号を除いた行のリスト。
        stats (collections.Counter): ``line_joined`` の加算先。

    Returns:
        list[str]: 接合後の行のリスト。
    """
    joined = []
    for ln in lines:
        prev = joined[-1] if joined else ""
        if (
            prev
            and ln
            and len(prev) >= JOIN_MIN_PREV_LEN
            and "|" not in ln
            and not prev.endswith(SENT_END)
            and RE_CONT_HEAD.match(ln)
        ):
            joined[-1] += ln
            stats["line_joined"] += 1
        else:
            joined.append(ln)
    return joined


def clean_lines(text, heads, stats):
    """S2: 柱・ページ番号行を除去し、段組みによる文中改行を接合する.

    Args:
        text (str): 正規化済みページ本文。
        heads (set[str]): ``find_running_heads`` が返す柱の正規形集合。
        stats (collections.Counter): ``line_page_number`` / ``line_running_head`` /
            ``line_joined`` の加算先。

    Returns:
        str: 行クリーニング後の本文。
    """
    lines = text.split("\n")
    nonblank = [i for i, ln in enumerate(lines) if ln]
    edge = set(nonblank[:EDGE_LINES] + nonblank[-EDGE_LINES:])
    kept = []
    for i, ln in enumerate(lines):
        if ln and RE_PAGE_NUM.match(ln):
            stats["line_page_number"] += 1
            continue
        if i in edge and _head_key(ln) in heads:
            stats["line_running_head"] += 1
            continue
        kept.append(ln)
    joined = _join_column_breaks(kept, stats)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(joined)).strip()
