"""統計の集計と書き出し: stats.json と drop_samples.jsonl.

``stats.json`` は書籍本文を含まない数値だけのファイルで、前処理レポートの一次データになる。
``drop_samples.jsonl`` は除去ページの冒頭抜粋を含むため ``artifacts/`` の外へ出さないこと。
"""

import collections
import json


DROP_SAMPLE_HEAD_CHARS = 160


class PipelineStats:
    """パイプライン全体と書籍ごとの件数を集計する.

    Attributes:
        global_counts (collections.Counter): 文字・行レベルの処理件数（全書籍合計）。
        books (dict[str, collections.Counter]): 書籍名→ページ・トークン等の件数。
        split_totals (collections.Counter): 分割ごとの文書・ページ・文字・トークン数。
        drop_samples (list[dict]): 除去ページの理由と冒頭抜粋。
    """

    def __init__(self):
        self.global_counts = collections.Counter()
        self.books = {}
        self.split_totals = collections.Counter()
        self.drop_samples = []

    def start_book(self, book, raw_pages):
        """書籍の集計を開始し、生ページ数・生文字数を記録する.

        Args:
            book (str): 書籍名。
            raw_pages (list[dict]): OCR JSON のページリスト。

        Returns:
            collections.Counter: その書籍の集計カウンタ。
        """
        bs = collections.Counter(
            pages_raw=len(raw_pages), chars_raw=sum(len(p["content"] or "") for p in raw_pages)
        )
        self.books[book] = bs
        return bs

    def record_drop(self, book, page_no, reason, text, dup_of=None):
        """除去したページを件数と抜粋サンプルに記録する.

        Args:
            book (str): 書籍名。
            page_no (int): ページ番号。
            reason (str): 除去理由。
            text (str): ページ本文（冒頭のみ保存する）。
            dup_of (str | None): 近似重複の場合の重複元キー。

        Returns:
            None
        """
        self.books[book][f"drop_{reason}"] += 1
        sample = {"book": book, "page": page_no, "reason": reason, "head": text[:DROP_SAMPLE_HEAD_CHARS]}
        if dup_of is not None:
            sample["dup_of"] = dup_of
        self.drop_samples.append(sample)

    def record_doc(self, record, n_pages):
        """書き出した文書を書籍別・分割別の合計に加算する.

        Args:
            record (dict): ``corpus.build_record`` が返すレコード。
            n_pages (int): 文書に含まれるページ数。

        Returns:
            None
        """
        sp = record["split"]
        bs = self.books[record["book"]]
        bs[f"{sp}_pages"] += n_pages
        bs[f"{sp}_tokens"] += record["n_tokens"]
        bs[f"{sp}_chars"] += record["n_chars"]
        self.split_totals[f"{sp}_docs"] += 1
        self.split_totals[f"{sp}_pages"] += n_pages
        self.split_totals[f"{sp}_tokens"] += record["n_tokens"]
        self.split_totals[f"{sp}_chars"] += record["n_chars"]

    def summary(self):
        """全体の要約を作る.

        Returns:
            dict: ``pages_raw / chars_raw / line_cleaning / page_drops / splits``。
        """
        drops = collections.Counter()
        for bs in self.books.values():
            for k, v in bs.items():
                if k.startswith("drop_"):
                    drops[k] += v
        return {
            "pages_raw": sum(b["pages_raw"] for b in self.books.values()),
            "chars_raw": sum(b["chars_raw"] for b in self.books.values()),
            "line_cleaning": dict(self.global_counts),
            "page_drops": dict(drops.most_common()),
            "splits": dict(self.split_totals),
        }

    def write(self, out_dir):
        """``stats.json`` と ``drop_samples.jsonl`` を書き出す.

        Args:
            out_dir (pathlib.Path): 出力ディレクトリ。

        Returns:
            dict: 書き出した要約（``summary()`` の結果）。
        """
        summary = self.summary()
        with open(out_dir / "stats.json", "w") as f:
            json.dump({"summary": summary, "books": self.books}, f, ensure_ascii=False, indent=1)
        with open(out_dir / "drop_samples.jsonl", "w") as f:
            for d in self.drop_samples:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return summary
