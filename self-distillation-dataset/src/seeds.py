"""プロンプトシードプールと書き出し（プレフィックス）プール。

シードは ws1-arc の HuggingFace キャッシュからオフラインで読む（ネットワーク往復なし）。
10 万件規模ではプールが尽きるため、ドメイン別の quota はプールサイズで上限をかける。

ドメインは Curator の指定に沿って 5 系統:
general_knowledge / code / math / japanese / dialogue

設計判断（2026-10-10 のレビューに基づく）:

- `openai/humaneval` は gated（HTTP 401）で、この環境に HF_TOKEN が无いため取得できない。
  非 gated 名義の `openai/openai_humaneval` と `google-research-datasets/mbpp` を使う。
- プールが空のドメインは黙って skip しない（quota が静かに再配分され、実行は健全に見えて
  ドメインが欠落する）。`load_seed_pool()` は RuntimeError を出す。
- prefix はチャネルで分ける。`analysis` チャネルに回答調の書き出しを注入すると、モデルが
  会話調から推論調へ切り替わる橋渡しが句読点の断片になる（wiki 側 `concepts/prefix-failure`）。
  したがって reasoning モードでは推論調の英語 prefix を使う。
- reasoning_effort は Curator 方針（2026-10-10）で medium を最大割合とし、low / high を
  含める。既定挙動（medium）を維持しつつ多様性を入れる。
"""

from __future__ import annotations

import random
from collections import defaultdict

# (dataset, config, field, domain) — field はプロンプトとして使う列
SEED_SOURCES: list[tuple[str, str, str, str]] = [
    ("cais/mmlu", "all", "question", "general_knowledge"),
    ("openai/openai_humaneval", "openai_humaneval", "prompt", "code"),
    ("google-research-datasets/mbpp", "full", "text", "code"),
    ("TIGER-Lab/MathInstruct", "default", "instruction", "math"),
    ("openai/gsm8k", "main", "question", "math"),
    ("Inoichan/OpenO1-SFT-JA", "default", "instruction", "japanese"),
    ("Kendamarron/OpenMathInstruct-2-ja-CoT", "default", "instruction", "japanese"),
    ("yongtae-jp/orca_dpo_pairs_ja", "default", "prompt", "dialogue"),
]

# 列名がデータセットごとに違うため、この順で最初の命中を使う
FIELD_FALLBACKS = ["instruction", "question", "prompt", "problem"]

# final チャネル用の書き出し（回答調）
PREFIX_POOLS: dict[str, list[str]] = {
    "general_knowledge": [
        "結論から述べると",
        "前提を整理したうえで",
        "要点を 3 点に絞ると",
        "よくある誤解を先に潰すと",
        "To begin with",
        "In short",
    ],
    "code": [
        "以下の実装で実現できます",
        "まず設計方針を述べる",
        "注意点として",
        "Here is a minimal implementation",
        "Approach first, then the code",
    ],
    "math": [
        "条件を整理すると",
        "まず自明な場合を分ける",
        "答えは次の通り",
        "Let me set up the notation",
        "We start from the given constraints",
    ],
    "japanese": [
        "承知しました",
        "日本語で答えます",
        "以下に整理します",
        "依頼の趣旨を踏まえると",
    ],
    "dialogue": [
        "承知しました",
        "なるほど",
        "続けてください",
        "Sure",
        "I can help with that",
    ],
}

# analysis チャネル用の書き出し（推論調・英語）。thinking は英語 only の Curator 方針と揃える。
REASONING_PREFIX_POOLS: dict[str, list[str]] = {
    "general_knowledge": [
        "Let me restate what is being asked",
        "First, the fact to check is",
        "Before answering, I separate what is known from what is assumed",
    ],
    "code": [
        "Let me restate the requirement",
        "First, check the input format and constraints",
        "Approach: identify the smallest correct implementation",
    ],
    "math": [
        "Let me set up the notation",
        "First, check the trivial case",
        "I restate the given constraints before computing",
    ],
    "japanese": [
        "Let me restate the request",
        "First, confirm what the user is asking for",
        "The user writes in Japanese, so the answer must be in Japanese",
    ],
    "dialogue": [
        "Let me restate the context",
        "First, what the user expects from this turn",
        "I check the tone and the length expected here",
    ],
}

# reasoning_effort の混合（Curator 方針: medium を最大割合、low / high を含む）
REASONING_EFFORT_MIX: dict[str, float] = {"medium": 0.50, "low": 0.25, "high": 0.25}

# effort ごとの生成上限（思考が final チャネルに到達する余地を残す）
EFFORT_MAX_TOKENS: dict[str, int] = {"low": 2048, "medium": 4096, "high": 6144}


def load_seed_pool() -> dict[str, list[str]]:
    """キャッシュ済み HF データセットからドメイン別プロンプトプールを組み立てる。

    Returns:
        domain -> prompt list。空文字・短すぎるものは除外する。

    Raises:
        RuntimeError: どのドメインもプールが空の場合（黙って skip しない）。
        ImportError: `datasets` が未インストールの場合。
    """
    from datasets import load_dataset  # noqa: PLC0415

    pool: dict[str, list[str]] = defaultdict(list)
    for name, config, field, domain in SEED_SOURCES:
        try:
            ds_dict = load_dataset(name, config)
            keys = list(ds_dict.keys())
            if not keys:
                print(f"[seed skip] {name}: no split")
                continue
            split = "train" if "train" in keys else keys[0]
            ds = ds_dict[split]
        except Exception as exc:  # キャッシュ欠落・config 名違い
            print(f"[seed skip] {name}: {exc}")
            continue
        columns = set(ds.column_names)
        key = field if field in columns else next((f for f in FIELD_FALLBACKS if f in columns), None)
        if key is None:
            print(f"[seed skip] {name}: no prompt column in {sorted(columns)}")
            continue
        for row in ds:
            text = str(row.get(key, "")).strip()
            if len(text) >= 20:
                pool[domain].append((text, f"{name}/{split}"))

    empty = [d for d in sorted({d for _, _, _, d in SEED_SOURCES}) if not pool.get(d)]
    if empty:
        raise RuntimeError(f"seed pool is empty for: {', '.join(empty)}")
    for domain in sorted(pool):
        print(f"[seed pool] {domain}: {len(pool[domain])}")
    return pool


class SeedPool:
    """非復元サンプリングするシードプール。

    プールサイズが quota を上回る場合は非復元で引き、尽きた場合は折り返す
    （折り返し分は `seed_index` が重複するため、呼び出し側で quota をプールサイズで
    上限化する）。

    Attributes:
        pool: domain -> prompt list。
    """

    def __init__(self, pool: dict[str, list[str]], rng: random.Random) -> None:
        """初期化する。

        Args:
            pool: ドメイン別プール。
            rng: 乱数生成器。
        """
        self.pool = pool
        self._order = {d: list(range(len(v))) for d, v in pool.items()}
        for order in self._order.values():
            rng.shuffle(order)
        self._cursor = {d: 0 for d in pool}

    def size(self, domain: str) -> int:
        """プールの件数を返す。

        Args:
            domain: 対象ドメイン。

        Returns:
            件数。
        """
        return len(self.pool.get(domain, []))

    def sample(self, domain: str) -> tuple[str, int, str]:
        """1 件を非復元で引き、(prompt, seed_index, seed_source) を返す。

        Args:
            domain: 対象ドメイン。

        Returns:
            (プロンプト, プール内 index, 取得元 dataset/split)。
        """
        order = self._order[domain]
        if not order:
            return "", -1, ""
        i = self._cursor[domain]
        if i >= len(order):
            self._cursor[domain] = 0
            i = 0
        idx = order[i]
        self._cursor[domain] = i + 1
        text, source = self.pool[domain][idx]
        return text, idx, source


def is_japanese_text(text: str) -> bool:
    """仮名・漢字を含むかを判定する。"""
    return any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text)


def sample_prefix(domain: str, mode: str, rng: random.Random, prompt: str = "") -> str:
    """書き出しプレフィックスを 1 件サンプリングする。

    no_reasoning の書き出しは final チャネルの言語を決めるため、プロンプトの言語に
    合わせた候補から選ぶ（英語プロンプトに日本語 prefix を付けると応答が日本語になり
    language_mixed で落ちる）。japanese 域は域の定義上、日本語 prefix を使う。

    Args:
        domain: 対象ドメイン。
        mode: "reasoning" / "no_reasoning"。
        rng: 乱数生成器。
        prompt: 元プロンプト（no_reasoning の言語合わせに使う）。

    Returns:
        書き出し文字列。
    """
    if mode == "reasoning":
        return rng.choice(REASONING_PREFIX_POOLS[domain])
    pool = PREFIX_POOLS[domain]
    want_ja = domain == "japanese" or is_japanese_text(prompt)
    matched = [p for p in pool if is_japanese_text(p) == want_ja]
    return rng.choice(matched or pool)


def sample_effort(rng: random.Random) -> str:
    """reasoning_effort を混合比率からサンプリングする。

    Args:
        rng: 乱数生成器。

    Returns:
        "low" / "medium" / "high"。
    """
    r = rng.random()
    acc = 0.0
    for effort, weight in REASONING_EFFORT_MIX.items():
        acc += weight
        if r < acc:
            return effort
    return "medium"
