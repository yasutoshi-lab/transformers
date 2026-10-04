"""S5/S6 分割と文書化: train / qa_eval / holdout_ppl への振り分けと JSONL 出力.

分割は書籍ごとに「除去後ページの相対位置」で連続ブロックとして切り出す。
隣接ページ経由の情報漏れを避けるため、ランダムなページ単位の分割はしない。

    train        CPT に使い、SFT 用 QA の生成元にもなる
    qa_eval      CPT には含めるが SFT 用 QA には使わない（知識獲得を 4 択 QA で測る）
    holdout_ppl  学習に一切使わない（perplexity でドメイン適応を測る）
"""

import json

from .lines import SENT_END


SPLITS = ("train", "qa_eval", "holdout_ppl")
QA_EVAL_RANGE = (0.30, 0.40)
HOLDOUT_PPL_RANGE = (0.60, 0.70)


def split_of(idx, n, qa_range=QA_EVAL_RANGE, ppl_range=HOLDOUT_PPL_RANGE):
    """書籍内の相対位置からページの分割先を決める.

    Args:
        idx (int): 書籍内（除去後）のページ順位。
        n (int): 書籍内（除去後）のページ数。
        qa_range (tuple[float, float]): qa_eval ブロックの相対位置範囲。
        ppl_range (tuple[float, float]): holdout_ppl ブロックの相対位置範囲。

    Returns:
        str: ``train`` / ``qa_eval`` / ``holdout_ppl`` のいずれか。
    """
    r = idx / n
    if ppl_range[0] <= r < ppl_range[1]:
        return "holdout_ppl"
    if qa_range[0] <= r < qa_range[1]:
        return "qa_eval"
    return "train"


def join_pages(texts):
    """連続ページを 1 文書に結合し、ページ跨ぎで途切れた文を接合する.

    Args:
        texts (list[str]): 連続するページ本文のリスト。

    Returns:
        str: 結合後の文書テキスト。前ページが文末記号で終わる場合は空行で区切り、
        そうでなければ区切りなしで連結する。
    """
    out = texts[0]
    for t in texts[1:]:
        sep = "\n\n" if out.rstrip().endswith(SENT_END) else ""
        out = out.rstrip() + sep + t.lstrip()
    return out


def group_pages(rows):
    """1 冊分のページを、分割先が同じで頁番号が連続する塊にまとめる.

    Args:
        rows (list[tuple[str, str, int, str]]): ``(category, book, page_no, text)`` の
            リスト（書籍内のページ順）。

    Returns:
        list[dict]: ``{"split", "category", "pages", "texts"}`` の塊のリスト。
    """
    n = len(rows)
    groups = []
    for idx, (category, _, page_no, text) in enumerate(rows):
        sp = split_of(idx, n)
        if groups and groups[-1]["split"] == sp and page_no - groups[-1]["pages"][-1] == 1:
            groups[-1]["pages"].append(page_no)
            groups[-1]["texts"].append(text)
        else:
            groups.append({"split": sp, "category": category, "pages": [page_no], "texts": [text]})
    return groups


def build_record(book, index, group, count_tokens):
    """塊 1 つを JSONL の 1 レコードに変換する.

    Args:
        book (str): 書籍名。
        index (int): 書籍内の塊の通し番号。
        group (dict): ``group_pages`` が返す塊。
        count_tokens (Callable[[str], int]): トークン数を返す関数。

    Returns:
        dict: ``id / book / category / split / pages / n_chars / n_tokens / text`` を持つレコード。
    """
    text = join_pages(group["texts"])
    return {
        "id": f"{book}-{index:04d}",
        "book": book,
        "category": group["category"],
        "split": group["split"],
        "pages": [group["pages"][0], group["pages"][-1]],
        "n_chars": len(text),
        "n_tokens": count_tokens(text),
        "text": text,
    }


class SplitWriter:
    """分割ごとの JSONL ファイルへレコードを書き出すコンテキストマネージャ.

    Attributes:
        out_dir (pathlib.Path): 出力ディレクトリ。
        files (dict[str, TextIO]): 分割名→ファイルハンドル。
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.files = {}

    def __enter__(self):
        self.files = {s: open(self.out_dir / f"{s}.jsonl", "w") for s in SPLITS}
        return self

    def __exit__(self, *exc):
        for f in self.files.values():
            f.close()

    def write(self, record):
        """レコードを所属する分割のファイルへ 1 行書き出す.

        Args:
            record (dict): ``build_record`` が返すレコード。

        Returns:
            None
        """
        self.files[record["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")
