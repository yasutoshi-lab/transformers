"""重複除去（完全一致 ＋ MinHash LSH 近似重複）。

2 段で判定する:

1. **完全一致**: 正規化（空白圧縮・小文字化）した prompt+response の SHA1。
2. **近似重複**: 応答本文の shingle を MinHash（128 hash）で表現し、LSH（threshold
   既定 0.80）で照合。同一ドメイン内でのみ照合する（ドメインをまたぐ近似重複は
   別サンプルとして残す）。shingle は prose では character 4-gram、短い応答
   （`SHORT_TEXT_MAX` 未満）では word 3-gram — 短い対話ターンでは character 4-gram が
   過発火するため。

近似重複の閾値は低めに置く。自己蒸留データは同一プロンプトからの再生成が
高確率で類似するため、閾値を高くすると「多様な書き出し」の意図が失われる。
"""

import hashlib
import re
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

_WS = re.compile(r"\s+")
SHINGLE_SIZE = 4
WORD_SHINGLE_SIZE = 3
NUM_PERM = 128
# これ未満の応答は character 4-gram が過発火する（短い対話ターン）。word shingle に切り替える
SHORT_TEXT_MAX = 200


def normalize(text: str) -> str:
    """照合用正規化（空白圧縮・小文字化）を適用する。

    Args:
        text: 対象テキスト。

    Returns:
        正規化済み文字列。
    """
    return _WS.sub(" ", (text or "").strip().lower())


def shingles(text: str, size: int = SHINGLE_SIZE) -> list[str]:
    """character n-gram shingle を返す。

    Args:
        text: 対象テキスト。
        size: n-gram の長さ。

    Returns:
        shingle の list。テキストが size 未満の場合は全体を 1 shingle として扱う。
    """
    t = normalize(text)
    if len(t) < size:
        return [t] if t else []
    return [t[i : i + size] for i in range(len(t) - size + 1)]


def word_shingles(text: str, size: int = WORD_SHINGLE_SIZE) -> list[str]:
    """word n-gram shingle を返す（短い応答用）。

    Args:
        text: 対象テキスト。
        size: n-gram の語数。

    Returns:
        shingle の list。
    """
    words = normalize(text).split()
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


@dataclass
class DedupStats:
    """重複除去の累計統計。

    Attributes:
        total: 入力件数。
        exact: 完全一致で除去した件数。
        near: 近似重複で除去した件数。
        kept: 残した件数。
    """

    total: int = 0
    exact: int = 0
    near: int = 0

    @property
    def kept(self) -> int:
        """残した件数を返す。

        Returns:
            残した件数。
        """
        return self.total - self.exact - self.near


class Deduplicator:
    """完全一致 ＋ MinHash LSH による重複除去器。

    Attributes:
        threshold: 近似重複の Jaccard 閾値。
    """

    def __init__(self, threshold: float = 0.80) -> None:
        """初期化する。

        Args:
            threshold: 近似重複の Jaccard 閾値。
        """
        self.threshold = threshold
        self._exact: set[str] = set()
        # ドメイン別 index。単一 index だと docstring の per-domain 主張と異なり、
        # ドメインをまたぐ近似重複で除去される。
        self._lsh: dict[str, MinHashLSH] = {}
        self.stats = DedupStats()

    def _index(self, domain: str) -> MinHashLSH:
        """ドメイン別の LSH index を取得する。

        Args:
            domain: 対象ドメイン。

        Returns:
            該当ドメインの MinHashLSH（無ければ生成）。
        """
        if domain not in self._lsh:
            self._lsh[domain] = MinHashLSH(threshold=self.threshold, num_perm=NUM_PERM)
        return self._lsh[domain]

    def is_duplicate(self, domain: str, prompt: str, response: str) -> str | None:
        """1 レコードが重複かを判定し、重複種別を返す。

        Args:
            domain: ドメイン名（近似重複の照合範囲）。
            prompt: プロンプト。
            response: 応答本文。

        Returns:
            "exact" / "near" / None（重複でない）。判定と同時に内部状態を更新する。
        """
        self.stats.total += 1

        key = hashlib.sha1(f"{normalize(prompt)}\t{normalize(response)}".encode()).hexdigest()
        if key in self._exact:
            self.stats.exact += 1
            return "exact"
        self._exact.add(key)

        mh = MinHash(num_perm=NUM_PERM)
        source = word_shingles(response) if len(normalize(response)) < SHORT_TEXT_MAX else shingles(response)
        for sh in source:
            mh.update(sh.encode("utf-8"))

        lsh_key = f"{domain}:{key}"
        index = self._index(domain)
        matches = index.query(mh)
        if matches:
            self.stats.near += 1
            return "near"
        index.insert(lsh_key, mh)
        return None
