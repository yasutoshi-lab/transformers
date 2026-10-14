"""プロンプトシードプールと書き出し（プレフィックス）プール。

シードは ws1-arc の HuggingFace キャッシュから読む（未取得なら初回にダウンロードする）。
プールが尽きる規模では、ドメイン別の quota はプールサイズで上限をかける。

ドメインは Curator の指定に沿って 6 系統:
general_knowledge / code / math / japanese / english / dialogue

設計判断（2026-10-10 のレビューに基づく）:

- シードは train split のみ。2026-10-14、cais/mmlu と openai/openai_humaneval の test split を
  使っていたため外した（評価データの混入）。general_knowledge は WebInstructSub に置き換えた。
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

import filters

# 学習データの汚染を避けるため、シードは train split だけから取る（test / validation は使わない）
SEED_SPLIT = "train"
# 大規模ソースを間引くときの固定シード（resume 時もプールが同じになる）
SAMPLE_SEED = 42


def _first_user_turn(row: dict) -> str:
    """会話形式の行から最初の user 発話を取り出す。

    Args:
        row: `conversations` 列を持つ行。要素は {"from","value"} か {"role","content"}。

    Returns:
        最初の user 発話。見つからなければ空文字。
    """
    for turn in row.get("conversations") or []:
        role = turn.get("from") or turn.get("role")
        if role in ("human", "user"):
            return str(turn.get("value") or turn.get("content") or "")
    return ""


# dataset: HF 名 / config: None なら既定 / field: プロンプト列名か抽出関数 / domain /
# keep: 行の採否（None なら全件）/ sample: 固定シードで間引く件数（None なら全件）
SEED_SOURCES: list[dict] = [
    {"dataset": "google-research-datasets/mbpp", "config": "full", "field": "text", "domain": "code"},
    {
        "dataset": "glaiveai/glaive-code-assistant-v3",
        "field": "question",
        "domain": "code",
        "sample": 74_000,
    },
    {"dataset": "TIGER-Lab/MathInstruct", "field": "instruction", "domain": "math"},
    {"dataset": "openai/gsm8k", "config": "main", "field": "question", "domain": "math"},
    {"dataset": "Inoichan/OpenO1-SFT-JA", "field": "instruction", "domain": "japanese"},
    {"dataset": "Kendamarron/OpenMathInstruct-2-ja-CoT", "field": "instruction", "domain": "japanese"},
    {"dataset": "llm-jp/oasst2-33k-en", "field": _first_user_turn, "domain": "english"},
    {
        "dataset": "TIGER-Lab/WebInstructSub",
        "field": "question",
        "domain": "general_knowledge",
        # mathstackexchange は math ドメインと重なるので除外
        "keep": lambda row: row["source"] in {"socratic", "stackexchange"},
        "sample": 62_000,
    },
    {"dataset": "yongtae-jp/orca_dpo_pairs_ja", "field": "prompt", "domain": "dialogue"},
]

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
        "Sure",
        "Here is a clear answer",
        "Short answer first",
        "Good question",
        "Let me break this down",
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
    "english": [
        "Let me restate what the user wants",
        "First, figure out what kind of answer fits here",
        "The user is asking for",
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


def load_seed_pool(domains: set[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """HF データセットからドメイン別プロンプトプールを組み立てる。

    シードは `SEED_SPLIT`（train）だけから取る。train が無いデータセットは他の split に
    流さずエラーにする（評価用 split の混入を防ぐ）。`keep` で行を絞り、`sample` 指定が
    あれば `SAMPLE_SEED` で非復元に間引く。

    Args:
        domains: 読み込むドメイン。None なら全ドメイン。

    Returns:
        domain -> [(prompt, "<dataset>/<split>")]。20 文字未満のプロンプトは除外する。

    Raises:
        RuntimeError: 対象ドメインのプールが空の場合（黙って skip しない）。
        ValueError: データセットに train split が無い、またはプロンプト列が無い場合。
        ImportError: `datasets` が未インストールの場合。
    """
    from datasets import load_dataset  # noqa: PLC0415

    sources = [src for src in SEED_SOURCES if domains is None or src["domain"] in domains]
    pool: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src in sources:
        name, field = src["dataset"], src["field"]
        try:
            ds = load_dataset(name, src.get("config"), split=SEED_SPLIT)
        except ValueError as exc:  # 指定 split が無い
            raise ValueError(f"{name}: no {SEED_SPLIT} split") from exc
        if isinstance(field, str) and field not in ds.column_names:
            raise ValueError(f"{name}: no column {field!r} in {sorted(ds.column_names)}")
        if src.get("keep"):
            ds = ds.filter(src["keep"])
        indices = list(range(len(ds)))
        if src.get("sample") and src["sample"] < len(indices):
            indices = sorted(random.Random(SAMPLE_SEED).sample(indices, src["sample"]))
        for row in ds.select(indices):
            text = (field(row) if callable(field) else str(row.get(field) or "")).strip()
            if len(text) >= 20:
                pool[src["domain"]].append((text, f"{name}/{SEED_SPLIT}"))
        print(f"[seed source] {name}: {len(indices)} rows", flush=True)

    empty = sorted({src["domain"] for src in sources} - {d for d in pool if pool[d]})
    if empty:
        raise RuntimeError(f"seed pool is empty for: {', '.join(empty)}")
    for domain in sorted(pool):
        print(f"[seed pool] {domain}: {len(pool[domain])}", flush=True)
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
    # 漢字を 1 字含むだけで日本語扱いにすると、中国語の書名などを引用した英語プロンプトに
    # 日本語 prefix が付き、応答が日本語になって language_mixed で落ちる。filters と同じ
    # 支配的な文字種の判定を使い、prefix の言語を language_mixed の期待言語とそろえる
    want_ja = domain == "japanese" or filters.detect_language(prompt) == "ja"
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
