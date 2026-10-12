"""プロンプトシードプールと書き出し（プレフィックス）プール。

シードは HuggingFace のデータセットから読む（初回はダウンロード、以降はキャッシュ）。

ドメインと構成比（Curator 判断 2026-10-12）:
math 20% / japanese 10% / english 10% / general_knowledge 30% / code 30%

設計判断:

- シードは Apache-2.0 のデータセットだけから取る（Curator 判断 2026-10-12）。
  使うのはプロンプト（ユーザー発話）のみで、各データセットの回答は使わない。
- japanese 域は日本語ネイティブの依頼文を使う（仮名を含むものだけ）。それ以外の域は
  英語の依頼文に揃える（仮名・漢字を含むものは除く。中国語の依頼文も混ざるため）。
- 大きいソースは固定シードで間引き、域内でソースが偏りすぎないようにする。
- プールが空のドメインは黙って skip しない（quota が静かに再配分され、実行は健全に見えて
  ドメインが欠落する）。`load_seed_pool()` は RuntimeError を出す。
- prefix はチャネルで分ける。思考チャネルに回答調の書き出しを注入すると、会話調から推論調へ
  切り替わる橋渡しが句読点の断片になる（gpt-oss 版で確認）。reasoning モードでは推論調の
  英語 prefix を使う。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedSource:
    """シードの取得元 1 件。

    Attributes:
        name: HF データセット名。
        config: config 名（無ければ None）。
        column: プロンプトとして使う列。会話形式（list）なら最初のユーザー発話を使う。
        domain: 割り当てるドメイン。
        split: 使う split。
        where: 列名 -> 採用する値の集合。すべて満たす行だけを使う。
        limit: 行数の上限。超える場合は固定シードで間引く。
    """

    name: str
    config: str | None
    column: str
    domain: str
    split: str = "train"
    where: dict[str, frozenset] = field(default_factory=dict)
    limit: int | None = None


_MAGPIE_OK = frozenset({"good", "excellent", "average"})

# すべて Apache-2.0（HF のデータセットカードで確認、2026-10-12）
SEED_SOURCES: list[SeedSource] = [
    SeedSource("AI-MO/NuminaMath-CoT", None, "problem", "math", limit=150_000),
    SeedSource("m-a-p/CodeFeedback-Filtered-Instruction", None, "query", "code"),
    SeedSource("glaiveai/glaive-code-assistant-v3", None, "question", "code", limit=50_000),
    SeedSource(
        "TIGER-Lab/WebInstructSub", None, "question", "general_knowledge",
        where={"source": frozenset({"socratic", "stackexchange"})}, limit=60_000,
    ),
    SeedSource(
        "HuggingFaceTB/smoltalk", "smol-magpie-ultra", "messages", "general_knowledge",
        where={
            "category": frozenset({"information-seeking", "advice-seeking", "reasoning", "data-analysis"}),
            "quality": _MAGPIE_OK,
        },
    ),
    SeedSource("llm-jp/oasst2-33k-en", None, "conversations", "english"),
    SeedSource(
        "HuggingFaceTB/smoltalk", "smol-magpie-ultra", "messages", "english",
        where={"category": frozenset({"creative-writing", "editing", "brainstorming"}), "quality": _MAGPIE_OK},
        limit=60_000,
    ),
    SeedSource("llm-jp/oasst2-33k-ja", None, "conversations", "japanese"),
    SeedSource("llm-jp/magpie-sft-v1.0", None, "conversations", "japanese", limit=40_000),
    SeedSource("Aratako/Magpie-Tanuki-8B-97k", None, "instruction", "japanese", limit=30_000),
]

MIN_SEED_CHARS = 20

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
    "english": [
        "To start with the key point",
        "In short",
        "Here is my answer",
        "Let me lay this out clearly",
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
        # 旧 "Let me restate the request" は依頼文を日本語のまま復唱させ、思考が日本語化した。
        # 旧 "The user writes in Japanese, ..." は英語シード＋日本語指示だった当時の japanese 域で事実に反した
        "First, confirm what the user is asking for",
        "Let me break the task into steps",
        "Let me work through this in English, then write the answer in Japanese",
    ],
    "english": [
        "Let me restate what is being asked",
        "First, identify the audience and the purpose",
        "Let me plan the structure before writing",
    ],
}

# reasoning_effort の混合（Curator 方針: medium を最大割合、low / high を含む）
REASONING_EFFORT_MIX: dict[str, float] = {"medium": 0.50, "low": 0.25, "high": 0.25}

# effort ごとの生成上限（思考が final チャネルに到達する余地を残す）
EFFORT_MAX_TOKENS: dict[str, int] = {"low": 2048, "medium": 4096, "high": 6144}


def _first_user_turn(conversation: list) -> str:
    """会話ログから最初のユーザー発話を取り出す。

    Args:
        conversation: {"role", "content"} の list。

    Returns:
        最初のユーザー発話。無ければ空文字。
    """
    return next((str(t.get("content", "")) for t in conversation if t.get("role") == "user"), "")


def has_kana(text: str) -> bool:
    """仮名を含むかを判定する（中国語と日本語を分ける）。"""
    return any("\u3040" <= c <= "\u30ff" for c in text)


def _select_rows(ds, src: SeedSource):
    """where で絞り、limit を超えれば固定シードで間引いた Dataset を返す。

    Args:
        ds: datasets.Dataset。
        src: 取得元。

    Returns:
        絞り込み後の Dataset（元の行順を保つ）。
    """
    idx = range(len(ds))
    for col, allowed in src.where.items():
        # ds[col][i] は大きい表で 1 件ごとに遅い（230 万行で 1 時間超）。Arrow から一括で取り出す
        values = ds.data.column(col).to_pylist()
        idx = [i for i in idx if values[i] in allowed]
    idx = list(idx)
    if src.limit is not None and len(idx) > src.limit:
        rng = random.Random(f"{src.name}/{src.config}/{src.domain}")
        idx = sorted(rng.sample(idx, src.limit))
    return ds.select(idx)


def load_seed_pool() -> dict[str, list[tuple[str, str]]]:
    """HF データセットからドメイン別プロンプトプールを組み立てる。

    Returns:
        domain -> [(prompt, 取得元)]。短すぎるもの・域の言語に合わないものは除く。

    Raises:
        RuntimeError: プールが空のドメインがある場合（黙って skip しない）。
        ImportError: `datasets` が未インストールの場合。
    """
    from datasets import load_dataset  # noqa: PLC0415

    pool: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src in SEED_SOURCES:
        label = f"{src.name}/{src.config}" if src.config else src.name
        try:
            ds = load_dataset(src.name, src.config, split=src.split)
        except Exception as exc:  # ネットワーク・config 名違い
            print(f"[seed skip] {label}: {exc}")
            continue
        ds = _select_rows(ds, src)
        added = 0
        for value in ds[src.column]:
            text = (_first_user_turn(value) if isinstance(value, list) else str(value or "")).strip()
            if len(text) < MIN_SEED_CHARS:
                continue
            matches_lang = has_kana(text) if src.domain == "japanese" else not is_japanese_text(text)
            if not matches_lang:
                continue
            pool[src.domain].append((text, f"{label}/{src.split}"))
            added += 1
        print(f"[seed source] {label} -> {src.domain}: {added}")

    empty = [d for d in sorted({s.domain for s in SEED_SOURCES}) if not pool.get(d)]
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
