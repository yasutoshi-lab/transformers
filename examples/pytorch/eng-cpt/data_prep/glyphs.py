"""S1 文字正規化: NFKC・簡体字→日本字体・OCR 誤字補正.

OCR（GLM-OCR）は日本語の漢字を形の近い JIS 外の字に読み違えることがある。
JIS X 0213 に収録されている字は「正しい日本語」とみなして一切変更せず、
JIS 外の字だけを次の順で日本の字体へ寄せる。

1. OpenCC（s2t → t2jp）で簡体字を日本の字体へ変換する
2. 変換できない字は OCR 誤字の対応表（``OCR_CHAR_FIXES``）で補正する
3. それでも不明な字は推測で埋めず ``OCR_UNKNOWN_MARK`` に置き換える
"""

import re
import unicodedata

import opencc


# OCR が形の似た JIS 外漢字に誤認識した字の補正（文脈から判定）
# 語句単位: 同じ誤字が文脈で別の字になるもの
OCR_PHRASE_FIXES = {"深緰り": "深絞り", "歓鎢": "鉄", "力犑雜": "が複雑"}
# 文字単位: 文脈で一意に決まるもの。「推定」は文脈上もっとも妥当だが確証が弱いもの
OCR_CHAR_FIXES = {
    "櫩": "欄", "屝": "扉", "緰": "締", "翱": "靭", "叾": "書", "鍲": "種", "頟": "領",
    "鐧": "鋼", "氖": "気", "攃": "機", "隡": "外", "縴": "緩", "溉": "漑", "溵": "濃",
    "絉": "紫", "鐼": "鎖", "摃": "損", "焨": "帰", "娱": "娯", "繌": "継", "譨": "譲",
    "砼": "砕", "鎷": "鋳", "鎻": "鎖", "犑": "複", "癎": "癇", "鎢": "鉄", "黳": "電",
    "鐊": "誘", "瀃": "満", "犤": "犠", "銌": "鋳", "膱": "臓", "緁": "縦", "郋": "部",
    "勭": "勤", "寕": "寧", "鳯": "鳳", "陹": "階", "鉬": "鋼", "瀱": "濃", "橝": "線",
    "乛": "弧", "啃": "哨",
    "鎟": "総",  # 推定: 総形フライス
    "楃": "極",  # 推定: 指極（人体寸法）
    "鈽": "鋳",  # 推定: 鋳鋼
    "髂": "粉",  # 推定: 紙粉
    "鋱": "越",  # 推定: 越前職人
}
# 人名・社名など文脈から確定できない字は推測で埋めず、げた記号に置き換える
OCR_UNKNOWN_MARK = "〓"

# OpenCC t2jp が日本の常用字体にしない文字の手動補正
CHAR_OVERRIDES = {"豔": "艶", "韌": "靭"}

JIS_ENCODING = "euc_jis_2004"
RE_TEXTCIRCLED = re.compile(r"\$?\\textcircled\{(\d{1,2})\}\$?")


def is_cjk_ideograph(ch):
    """CJK 統合漢字（基本ブロック）かどうかを判定する.

    Args:
        ch (str): 1 文字。

    Returns:
        bool: U+4E00–U+9FFF に含まれれば ``True``。
    """
    return "一" <= ch <= "鿿"


def is_jis(ch):
    """JIS X 0213 で表現できる文字かどうかを判定する.

    Args:
        ch (str): 1 文字。

    Returns:
        bool: ``euc_jis_2004`` でエンコードできれば ``True``。
    """
    try:
        ch.encode(JIS_ENCODING)
        return True
    except UnicodeEncodeError:
        return False


class GlyphConverter:
    """JIS 外の漢字を日本の字体へ寄せる 1 文字変換器.

    処理概要: OpenCC の s2t → t2jp を 1 文字単位で適用し、常用字体にならない
    字は ``CHAR_OVERRIDES`` で補正する。OCR 誤字の補正やげた記号への置換は
    ``to_jp_glyphs`` 側の責務で、このクラスは OpenCC 変換だけを受け持つ。

    Attributes:
        s2t (opencc.OpenCC): 簡体字→繁体字の変換器。
        t2jp (opencc.OpenCC): 繁体字→日本の字体の変換器。
    """

    def __init__(self):
        self.s2t = opencc.OpenCC("s2t")
        self.t2jp = opencc.OpenCC("t2jp")

    def __call__(self, ch):
        """1 文字を日本の字体へ変換する.

        Args:
            ch (str): 変換対象の 1 文字。

        Returns:
            str: 変換後の文字（JIS 内に収まるとは限らない）。
        """
        converted = self.t2jp.convert(self.s2t.convert(ch))
        return CHAR_OVERRIDES.get(converted, converted)


def to_jp_glyphs(text, converter, stats):
    """JIS 外の漢字だけを日本の字体へ変換し、JIS 外の字を残さない.

    処理概要: JIS 内の字はそのまま残す。JIS 外の字は OpenCC 変換 →
    OCR 誤字対応表 → げた記号の順に試し、それぞれの件数を ``stats`` に加算する。

    Args:
        text (str): NFKC 済みテキスト。
        converter (Callable[[str], str]): 1 文字を変換する関数（``GlyphConverter``）。
        stats (collections.Counter): ``glyph_converted`` / ``glyph_ocr_fixed`` /
            ``glyph_unknown_marked`` を加算するカウンタ。

    Returns:
        str: 変換後のテキスト。
    """
    out = []
    for ch in text:
        if is_cjk_ideograph(ch) and not is_jis(ch):
            new = converter(ch)
            if is_jis(new):
                stats["glyph_converted"] += 1
                ch = new
            elif ch in OCR_CHAR_FIXES:
                stats["glyph_ocr_fixed"] += 1
                ch = OCR_CHAR_FIXES[ch]
            else:
                stats["glyph_unknown_marked"] += 1
                ch = OCR_UNKNOWN_MARK
        out.append(ch)
    return "".join(out)


def normalize_page(text, converter, stats):
    """S1: ページ本文の文字レベル正規化を行う.

    処理概要: LaTeX の丸数字（``\\textcircled{1}``）を ①… に置換し、NFKC 正規化、
    語句単位の OCR 誤字補正、JIS 外漢字の変換、空白の圧縮と行頭末の空白除去を行う。

    Args:
        text (str): OCR の生テキスト。
        converter (Callable[[str], str]): 簡体字→日本字体の 1 文字変換関数。
        stats (collections.Counter): 変換件数の集計先。

    Returns:
        str: 正規化済みテキスト。
    """
    text = RE_TEXTCIRCLED.sub(lambda m: chr(0x2460 + int(m.group(1)) - 1), text)
    text = unicodedata.normalize("NFKC", text)
    for wrong, right in OCR_PHRASE_FIXES.items():
        stats["glyph_ocr_fixed"] += text.count(wrong)
        text = text.replace(wrong, right)
    text = to_jp_glyphs(text, converter, stats)
    text = re.sub(r"[ \t　]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n"))
