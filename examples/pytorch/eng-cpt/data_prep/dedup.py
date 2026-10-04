"""S4 重複除去: 完全一致ハッシュ + MinHash による近似重複検出."""

import hashlib
import re

from datasketch import MinHash, MinHashLSH


NUM_PERM = 128
NGRAM = 5
DEFAULT_THRESHOLD = 0.85


def minhash(text, num_perm=NUM_PERM, n=NGRAM):
    """空白を除いた文字 n-gram の MinHash を計算する.

    Args:
        text (str): 対象テキスト。
        num_perm (int): 置換数。
        n (int): 文字 n-gram の n。

    Returns:
        MinHash: 計算済み MinHash。
    """
    m = MinHash(num_perm=num_perm)
    s = re.sub(r"\s", "", text)
    for i in range(max(1, len(s) - n + 1)):
        m.update(s[i : i + n].encode())
    return m


class Deduplicator:
    """ページを順に受け取り、既出ページとの重複を判定する.

    処理概要: 空白を除いた本文の MD5 で完全一致を、文字 5-gram の MinHash LSH で
    近似重複（推定 Jaccard が閾値以上）を検出する。先に現れたページを残し、
    後から現れた重複ページを除去対象とする。

    Attributes:
        seen_hash (set[str]): 登録済みページの MD5。
        lsh (MinHashLSH): 登録済みページの MinHash インデックス。
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.seen_hash = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)

    def check(self, key, text):
        """重複を判定し、重複でなければページを登録する.

        Args:
            key (str): ページの識別子（``"<book>:<page>"``）。
            text (str): ページ本文。

        Returns:
            tuple[str | None, str | None]: ``(判定, 重複元キー)``。判定は
            ``"exact_dup"`` / ``"near_dup"`` / ``None``（重複なし・登録済み）。
        """
        h = hashlib.md5(re.sub(r"\s", "", text).encode()).hexdigest()
        if h in self.seen_hash:
            return "exact_dup", None
        mh = minhash(text)
        dup = self.lsh.query(mh)
        if dup:
            return "near_dup", dup[0]
        self.seen_hash.add(h)
        self.lsh.insert(key, mh)
        return None, None
