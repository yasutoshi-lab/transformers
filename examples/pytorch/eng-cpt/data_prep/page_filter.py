"""S3 ページ除去: 本文以外のページと品質の低いページを判定する.

判定は上から順に評価し、最初に当たったルール名を除去理由とする。
数式・表・コードは工学知識の担い手なので、言語比率は「数式・表・コードを除いた
地の文」で測る（初版では数式や表のページを大量に誤って除去していたため）。
"""

import re

from .lines import SENT_END


# ---- 閾値（前処理レポートで参照するため 1 か所にまとめる） ----
EMPTY_MAX_CHARS = 50            # 空白込みでこれ未満なら empty
TOO_SHORT_MAX_CHARS = 100       # 空白除去後でこれ未満なら too_short
HEAD_LINES = 3                  # 見出し判定に使う先頭行数
INDEX_MIN_LINES = 10            # 索引行の比率判定を行う最小行数
INDEX_LINE_RATIO = 0.5          # 索引らしい行の比率がこれ以上なら index
TOC_MIN_LINES = 5               # 目次行の比率判定を行う最小行数
TOC_LINE_RATIO = 0.3            # リーダ付き行の比率がこれ以上なら toc
COLOPHON_MIN_KEYWORDS = 2       # 奥付語がこの種類数以上なら colophon
FIGURE_MIN_LINES = 8            # 図主体判定を行う最小行数
FIGURE_LONG_LINE = 20           # これを超える長さの行を「文章行」とみなす
FIGURE_MAX_LONG_CHARS = 150     # 文章行の合計文字数がこれ未満
FIGURE_MAX_PROSE_CHARS = 300    # かつ地の文がこれ未満なら figure_dominant
RATIO_MIN_PROSE_CHARS = 50      # 言語比率の判定を行う地の文の最小文字数
MIN_JA_RATIO = 0.3              # 日本語文字の比率がこれ未満なら low_japanese
MIN_HIRAGANA_RATIO = 0.05       # ひらがなの比率がこれ未満なら low_hiragana
CODE_ASCII_RATIO = 0.8          # ASCII 比率がこれを超える行はコード/英文として地の文から除く
SALVAGE_MIN_LINE = 20           # 図主体ページから救出する解説文の最小行長
SALVAGE_MIN_CHARS = 50          # 救出結果がこれ以上ならページを残す

# ---- ルール（正規表現） ----
RE_ISBN = re.compile(r"ISBN|C\d{4}\s*¥")
RE_COLOPHON = re.compile(r"定価|発行所|発行者|印刷所|検定済|無断で複写|無断転載|著作権法")
RE_CREDITS = re.compile(r"作成委員|監修委員|執筆者|著者略歴|著者紹介|編集委員|執筆協力|執筆者紹介")
RE_PREFACE = re.compile(r"^.{0,12}(まえがき|はしがき|序文|あとがき|刊行にあたって|監修の言葉)", re.MULTILINE)
RE_TOC_HEAD = re.compile(r"目\s*次|もくじ|CONTENTS", re.IGNORECASE)
RE_TOC_LINE = re.compile(r"(\.{3,}|…{2,}|·{3,}|・{3,}|—{2,}|-{3,})\s*\d{1,3}\s*$")
RE_INDEX_HEAD = re.compile(r"^\s*(索\s*引|さくいん|INDEX)", re.IGNORECASE)
RE_INDEX_LINE = re.compile(r"^[^|$]{1,30}\s[\d,\s—-]+$|^【.】$")
RE_REFS = re.compile(r"^\s*(引用文献|参考文献|参考図書|〈引用文献〉|〈参考文献〉)")
RE_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", re.DOTALL)
RE_MATH_ENV = re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.DOTALL)
RE_TABLE_ROW = re.compile(r"^.*\|.*$", re.MULTILINE)
RE_HTML = re.compile(r"<[^>]+>")
RE_JA = re.compile("[\u3040-\u30ff\u4e00-\u9fff]")
RE_HIRA = re.compile("[\u3040-\u309f]")


def _strip_ws(text):
    """空白類をすべて取り除く.

    Args:
        text (str): 対象テキスト。

    Returns:
        str: 空白・改行を除いた文字列。
    """
    return re.sub(r"\s", "", text)


def _ratio(lines, pred):
    """述語を満たす行の比率を返す.

    Args:
        lines (list[str]): 対象行。
        pred (Callable[[str], bool]): 判定関数。

    Returns:
        float: 満たす行の比率。``lines`` が空なら 0.0。
    """
    return sum(bool(pred(ln)) for ln in lines) / len(lines) if lines else 0.0


def prose_only(text):
    """数式・数式環境・表・HTML タグ・コード/英文行を除いた地の文だけを返す.

    Args:
        text (str): ページ本文。

    Returns:
        str: 言語比率の判定に使う地の文。
    """
    text = RE_MATH_ENV.sub("", RE_MATH.sub("", text))
    text = RE_HTML.sub("", RE_TABLE_ROW.sub("", text))
    keep = []
    for ln in text.split("\n"):
        body = _strip_ws(ln)
        if body and (sum(ch.isascii() for ch in body) / len(body) > CODE_ASCII_RATIO or "\\" in ln):
            continue
        keep.append(ln)
    return "\n".join(keep)


def salvage_sentences(text):
    """図主体ページから、図ラベルを捨てて解説文（文末記号で終わる行）だけを救出する.

    Args:
        text (str): 図主体と判定されたページ本文。

    Returns:
        str: 救出した解説文。該当がなければ空文字列。
    """
    return "\n".join(ln for ln in text.split("\n") if len(ln) >= SALVAGE_MIN_LINE and ln.endswith(SENT_END))


def classify_drop(text):
    """S3: ページを除去すべきか判定し、理由を返す.

    処理概要: 本文以外のページ（奥付・クレジット・まえがき・索引・目次・参考文献）を
    見出しやキーワードで判定し、次に短すぎるページ・図主体ページ・言語比率の
    低いページを判定する。評価順は除去理由の帰属に影響するため変更しないこと。

    Args:
        text (str): 行クリーニング後のページ本文。

    Returns:
        str | None: 除去理由（ルール名）。残す場合は ``None``。
    """
    lines = [ln for ln in text.split("\n") if ln]
    head = "\n".join(lines[:HEAD_LINES])
    n = len(lines)
    if len(text) < EMPTY_MAX_CHARS:
        return "empty"
    if RE_ISBN.search(text) or len(set(RE_COLOPHON.findall(text))) >= COLOPHON_MIN_KEYWORDS:
        return "colophon"
    if RE_CREDITS.search(text):
        return "credits"
    if RE_PREFACE.search(head):
        return "preface"
    plain = [ln for ln in lines if "|" not in ln]
    if RE_INDEX_HEAD.search(head) or (
        len(plain) >= INDEX_MIN_LINES and _ratio(plain, RE_INDEX_LINE.match) >= INDEX_LINE_RATIO
    ):
        return "index"
    if RE_TOC_HEAD.search(head) or (n >= TOC_MIN_LINES and _ratio(lines, RE_TOC_LINE.search) >= TOC_LINE_RATIO):
        return "toc"
    if RE_REFS.search(head):
        return "references"
    if len(_strip_ws(text)) < TOO_SHORT_MAX_CHARS:
        return "too_short"
    prose = _strip_ws(prose_only(text))
    long_chars = sum(len(ln) for ln in lines if len(ln) > FIGURE_LONG_LINE)
    if n >= FIGURE_MIN_LINES and long_chars < FIGURE_MAX_LONG_CHARS and len(prose) < FIGURE_MAX_PROSE_CHARS:
        return "figure_dominant"
    if len(prose) >= RATIO_MIN_PROSE_CHARS and len(RE_JA.findall(prose)) / len(prose) < MIN_JA_RATIO:
        return "low_japanese"
    if len(prose) >= RATIO_MIN_PROSE_CHARS and len(RE_HIRA.findall(prose)) / len(prose) < MIN_HIRAGANA_RATIO:
        return "low_hiragana"
    return None


def filter_page(text):
    """除去判定を行い、図主体ページは解説文の救出を試みる.

    Args:
        text (str): 行クリーニング後のページ本文。

    Returns:
        tuple[str | None, str, bool]: ``(除去理由, 残す本文, 救出したか)``。
        除去理由が ``None`` ならページを残す。
    """
    reason = classify_drop(text)
    if reason == "figure_dominant":
        salvaged = salvage_sentences(text)
        if len(salvaged) >= SALVAGE_MIN_CHARS:
            return None, salvaged, True
    return reason, text, False
