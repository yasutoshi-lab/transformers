"""規則ベースのサンプル品質フィルタ。

自己蒸留データセット（gemma-4-26B-A4B-it 自身の生成）から、以下を除去する:

- 途中切れ（vLLM の `finish_reason == "length"`）
- 言語混入（プロンプトの想定言語と応答の文字種が食い違う）
- 空・極短・思考チャネル欠落（reasoning ありモード）

判定はすべて規則ベース（LLM judge 不使用）。理由: 50 万件規模で judge を併用すると
生成コストが二重化し、かつ判定自体が確率論的になるため再現性が失われる。

文字種の判定は Unicode の block 範囲に基づく（`unicodedata` の名前ではなく
コードポイント範囲で照合するため、外部依存がない）。
"""

import re
from dataclasses import dataclass

# Unicode block 範囲（コードポイント）
SCRIPTS = {
    "hiragana": (0x3040, 0x309F),
    "katakana": (0x30A0, 0x30FF),
    "kanji": (0x4E00, 0x9FFF),
    "latin": (0x0041, 0x005A, 0x0061, 0x007A),
    "cyrillic": (0x0400, 0x04FF),
    "hangul": (0xAC00, 0xD7AF),
    "arabic": (0x0600, 0x06FF),
}

# 日本語応答で許容するラテン文字の割合（技術用語・コード識別子）
JA_LATIN_RATIO_LIMIT = 0.30
# 英語応答で許容する仮名・漢字の割合
EN_KANA_KANJI_RATIO_LIMIT = 0.10
# 想定言語と異なる文字種がこれ以上あれば混入扱い。1 文字は 10 万件規模ではノイズ
# （引用・固有名詞の 1 文字で落ちる）ため 3 に上げた
FOREIGN_SCRIPT_MIN = 3

# 応答最低長。既定 40。対話域は短いターンが正当なので domain 別
MIN_RESPONSE_CHARS_BY_DOMAIN = {"default": 40, "dialogue": 20}
MIN_RESPONSE_CHARS = MIN_RESPONSE_CHARS_BY_DOMAIN["default"]

_WS = re.compile(r"\s+")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# LaTeX 数式（\[..\] / \(..\) / $$..$$ / $..$）。\text{} 内の語や変数がラテン文字として
# 計上され、日本語の解説でも latin_ratio で落ちるため、言語判定から外す
_MATH = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)


def strip_code_blocks(text: str) -> str:
    """fenced code block と inline code を除く。

    code 域では識別子・変数名が latin として計上され、日本語の解説文でも
    `JA_LATIN_RATIO_LIMIT` を超えやすい。コード本文は言語判定から外す。

    Args:
        text: 対象テキスト。

    Returns:
        コード部分を除したテキスト。
    """
    text = _FENCE.sub(" ", text or "")
    text = _MATH.sub(" ", text)
    return _INLINE_CODE.sub(" ", text)


def script_counts(text: str) -> dict[str, int]:
    """テキスト中の文字種別文字数を返す。

    Args:
        text: 対象テキスト。

    Returns:
        script 名 -> 文字数 の dict。判定に使用しない文字（数字・記号・空白）は計上しない。
    """
    counts = {name: 0 for name in SCRIPTS}
    for ch in text:
        code = ord(ch)
        for name, rng in SCRIPTS.items():
            if len(rng) == 4:  # latin は 2 範囲
                if rng[0] <= code <= rng[1] or rng[2] <= code <= rng[3]:
                    counts[name] += 1
                    break
            elif rng[0] <= code <= rng[1]:
                counts[name] += 1
                break
    return counts


def detect_language(text: str) -> str:
    """テキストの支配的な文字種から想定言語を推定する。

    Args:
        text: 対象テキスト。

    Returns:
        "ja" / "en" / "other"。文字判定（字母）が 0 件の場合は "other"。
    """
    counts = script_counts(text)
    ja = counts["hiragana"] + counts["katakana"] + counts["kanji"]
    en = counts["latin"]
    total = sum(counts.values())
    if total == 0:
        return "other"
    if ja >= en:
        return "ja"
    return "en"


def is_language_mixed(prompt: str, response: str, domain: str = "") -> tuple[bool, str]:
    """プロンプトの想定言語と応答の文字種が混入しているかを判定する。

    Args:
        prompt: 元プロンプト。
        response: 応答本文（思考を除く final）。
        domain: 対象ドメイン。code の場合はコード本文を判定から除く。

    Returns:
        (混入フラグ, 理由文字列)。混入でない場合は (False, "")。
    """
    # japanese 域は域の定義そのものが「日本語で答える」ため、英語プロンプトでも ja 固定
    expected = "ja" if domain == "japanese" else detect_language(prompt)
    body = strip_code_blocks(response) if domain in ("code", "japanese", "math") else response
    counts = script_counts(body)
    letters = sum(counts.values())
    if letters == 0:
        # code 域の応答がコードブロックだけ（HumanEval の関数補完など）は正当な応答
        if domain == "code" and _FENCE.search(response or ""):
            return False, ""
        return True, "no_letters"

    if expected == "ja":
        latin_ratio = counts["latin"] / letters
        if latin_ratio > JA_LATIN_RATIO_LIMIT:
            return True, f"latin_ratio={latin_ratio:.2f}"
        for script in ("cyrillic", "hangul", "arabic"):
            if counts[script] >= FOREIGN_SCRIPT_MIN:
                return True, f"foreign_script={script}"
        return False, ""

    if expected == "en":
        ja = counts["hiragana"] + counts["katakana"] + counts["kanji"]
        if ja / letters > EN_KANA_KANJI_RATIO_LIMIT:
            return True, f"kana_kanji_ratio={ja / letters:.2f}"
        return False, ""

    return False, ""


_QUOTED = re.compile(r"「[^」]*」|『[^』]*』|\"[^\"\n]*\"|“[^”\n]*”")
# 引用符外の thinking に許す仮名・漢字の割合（下書きの数語など）。Curator 判断 2026-10-11
THINKING_JA_RATIO_LIMIT = 0.03
# japanese 域の緩和（Curator 判断 2026-10-12）。英語で推論した後に日本語の回答を thinking 内で
# 下書きする例が多く、上の一律基準では思考ありの 7 割弱が落ちた。推論部分にあたる前半は英語を
# 求めたまま、全体では下書きぶんの日本語を許す
JAPANESE_THINKING_JA_RATIO_LIMIT = 0.30
JAPANESE_THINKING_HEAD_JA_RATIO_LIMIT = 0.05


def _ja_ratio(text: str) -> float:
    """文字種カウント上の仮名・漢字の割合を返す。

    Args:
        text: 判定対象。

    Returns:
        仮名・漢字の割合。文字が無ければ 0.0。
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts["hiragana"] + counts["katakana"] + counts["kanji"]) / total


def thinking_is_english(thinking: str, domain: str = "") -> bool:
    """思考チャネルが実質英語で書かれているかを判定する。

    Curator 方針: reasoning ありモードの thinking は英語で書く。引用符・コード内の日本語は
    除外し、引用符外の仮名・漢字が `THINKING_JA_RATIO_LIMIT` 以下なら英語とみなす。
    japanese 域は回答の下書きを許すため、前半が `JAPANESE_THINKING_HEAD_JA_RATIO_LIMIT`
    以下かつ全体が `JAPANESE_THINKING_JA_RATIO_LIMIT` 以下なら英語とみなす。

    Args:
        thinking: 思考チャネル本文。
        domain: 対象ドメイン。

    Returns:
        実質英語なら True。
    """
    body = _QUOTED.sub(" ", strip_code_blocks(thinking))
    if domain == "japanese":
        return (
            _ja_ratio(body) <= JAPANESE_THINKING_JA_RATIO_LIMIT
            and _ja_ratio(body[: len(body) // 2]) <= JAPANESE_THINKING_HEAD_JA_RATIO_LIMIT
        )
    return _ja_ratio(body) <= THINKING_JA_RATIO_LIMIT


_JUNK_CHARS = set("…・-–—”“\"?.")
JUNK_HEAD_RATIO_LIMIT = 0.30


def head_is_junk(text: str) -> bool:
    """先頭の段落が記号・省略記号主体の断片になっているかを判定する。

    prefix の挿入位置が壊れていると thinking / response の頭が `…?……` のような
    断片になる。先頭段落（最大 120 文字）の記号比率で検出する。

    Args:
        text: 判定対象（thinking または response）。

    Returns:
        先頭が空に近い、または記号比率が上限を超えれば True。
    """
    head = text.strip().split("\n\n")[0][:120]
    if len(head) < 3:
        return True
    junk = sum(1 for c in head if c in _JUNK_CHARS)
    return junk / len(head) > JUNK_HEAD_RATIO_LIMIT


# thinking が隠れた指示（system / developer）を引用していないか。学習時に存在しない指示への言及は除く
_INSTRUCTION_CITE = re.compile(
    r"developer (?:instruction|message|says|said|wants)|system (?:prompt|message|instruction)|"
    r"per (?:the )?(?:developer |system )?instruction|the instruction says|instructed to|"
    r"Reason in English|Think in English|analysis channel|thought channel|language of the user'?s request",
    re.IGNORECASE,
)
# Gemma 4 の制御トークン（思考 <|channel>…<channel|>、ターン <|turn>…<turn|> など）が本文に残っていないか
_SPECIAL_TOKEN = re.compile(
    r"<\|(?:channel|turn|think|tool_call|tool_response|tool|\"|image|audio|video)|"
    r"<(?:channel|turn|tool_call|tool_response|tool)\|>"
)

# no_reasoning の応答に、利用者や回答そのものを三人称で語るメタ推論が混入していないか。
# gpt-oss 版の `We (need|can|must) ...` は Gemma 4 では英語解説の自然な文（"We can rewrite ..."）
# まで落とした（パイロット 1 で 30 件すべて誤検出）ため、メタ推論の言い回しに絞る
_ENGLISH_COT = re.compile(
    r"\bThe user (?:asks|asked|wants|is asking|writes|wrote)\b|"
    r"\bWe need to (?:answer|respond|reply|produce|output|write the answer)\b|"
    r"\bLet'?s (?:craft|produce|draft) (?:the|an|a) (?:answer|response|reply)\b"
)


# 応答の途中で誤りに気づいて言い直している（"・・・あ、失礼しました。計算し直します"、"Wait, let's re-examine"）。
# 思考なしの応答で目立つ（パイロット 2 で japanese 域の思考なし 1 割前後）。誤答を含んだまま
# 応答が続くため除く。思考（thinking）内の自己訂正は推論の一部なので対象外
_SELF_CORRECTION = re.compile(
    r"計算ミス|失礼しました|計算し直|訂正(?:します|とお詫び|と最終)|誤りがありました|間違えました|"
    r"\bI made (?:a|an) (?:mistake|error)\b|\bWait[,.!]|\bOops\b|"
    r"\b(?:let'?s|let me) re-?(?:examine|calculate|check|evaluate)\b",
    re.IGNORECASE,
)


@dataclass
class FilterResult:
    """1 サンプルのフィルタ判定結果。

    Attributes:
        passed: 合格かどうか。
        reasons: 不合格理由（合格時は空）。
    """

    passed: bool
    reasons: list[str]


def evaluate(record: dict) -> FilterResult:
    """1 レコードを規則ベースで評価する。

    Args:
        record: 以下のキーを持つ dict。
            - prompt (str)
            - response (str): final チャネル本文
            - thinking (str): analysis チャネル本文（reasoning なしモードでは空）
            - finish_reason (str): vLLM の finish_reason
            - mode (str): "reasoning" / "no_reasoning"

    Returns:
        FilterResult。
    """
    reasons: list[str] = []

    if record.get("error"):
        reasons.append("request_failed")

    finish = record.get("finish_reason")
    if finish == "length":
        reasons.append("truncated")
    elif finish not in ("stop", "length", ""):
        reasons.append(f"finish_reason:{finish}")

    domain = record.get("domain", "")
    min_chars = MIN_RESPONSE_CHARS_BY_DOMAIN.get(domain, MIN_RESPONSE_CHARS_BY_DOMAIN["default"])
    response = (record.get("response") or "").strip()
    if len(response) < min_chars:
        reasons.append("too_short")

    if _SPECIAL_TOKEN.search(response) or _SPECIAL_TOKEN.search(record.get("thinking") or ""):
        reasons.append("special_token_leak")
    elif response and head_is_junk(response):
        reasons.append("junk_head")

    if _SELF_CORRECTION.search(response):
        reasons.append("response_self_correction")

    mixed, why = is_language_mixed(record.get("prompt", ""), response, domain)
    if mixed:
        reasons.append(f"language_mixed:{why}")

    if record.get("mode") == "reasoning":
        thinking = (record.get("thinking") or "").strip()
        if not record.get("thinking_closed", True):
            reasons.append("thinking_unclosed")
        elif not thinking:
            reasons.append("empty_thinking")
        elif not thinking_is_english(thinking, domain):
            reasons.append("thinking_not_english")
        elif head_is_junk(thinking):
            reasons.append("junk_thinking_head")
        elif _INSTRUCTION_CITE.search(thinking):
            reasons.append("thinking_cites_instruction")
    elif record.get("mode") == "no_reasoning" and _ENGLISH_COT.search(strip_code_blocks(response)):
        reasons.append("english_cot_leak")

    return FilterResult(passed=not reasons, reasons=reasons)
