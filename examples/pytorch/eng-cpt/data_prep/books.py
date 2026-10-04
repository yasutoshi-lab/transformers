"""CPT コーパスの対象とする工学系教科書の一覧.

Gaia の ``raw/books/<title>.json`` のファイル名（拡張子なし）をカテゴリ別に列挙する。
数学（数学I / 数学II）は工学の基礎ではあるが、Curator 判断で対象外とした。
"""

DEFAULT_RAW_DIR = "/home/ubuntu/Documents/project/Gaia-Personal/raw/books"

BOOKS = {
    "mechanical": [
        "機械加工学の基礎",
        "材料加工プロセス",
        "材料基礎工学",
        "溶接I",
        "板金工作法及びプレス加工法",
        "流体力学",
        "はじめての表面処理技術",
        "これでわかるプラスチック技術",
        "生産技術の実践手法がよく分かる本",
        "絵とき機械図面の読み方かき方",
    ],
    "electrical": [
        "電気基礎I",
        "電気基礎II",
        "電子回路",
        "通信技術",
        "電子情報技術",
        "電子製図",
        "電気電子実習I",
        "工業技術基礎",
    ],
    "aeronautical": [
        "プロが教える飛行機のメカニズム",
        "飛行機の仕組みパーフェクト辞典",
    ],
}


def iter_books():
    """カテゴリと書籍名の組を定義順に列挙する.

    Yields:
        tuple[str, str]: ``(category, book_title)``。
    """
    for category, titles in BOOKS.items():
        for title in titles:
            yield category, title
