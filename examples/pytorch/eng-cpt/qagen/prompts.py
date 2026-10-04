"""QA 生成・検証用のプロンプトと JSON スキーマ."""

SFT_PROMPT = """あなたは工学教育の専門家です。以下の教科書の抜粋をもとに、工学の知識を問う質問と回答を最大{n}組作ってください。

# 規則
- 質問は、抜粋を読んでいない人にも意味が通るように書く。「本文」「この文章」「上記」「図3」「表2」「式(5)」のような抜粋への参照は使わない。
- 回答は抜粋に書かれている内容だけに基づく。抜粋にない数値・事実を補わない。
- 回答は 2〜5 文で、理由・条件・仕組みが分かるように説明する。
- 質問の種類を散らす（定義、理由・仕組み、比較、手順、用途、計算など）。同じ事実を言い換えた質問を重複させない。
- 抜粋が目次・図のラベル・表の断片だけで工学の知識を含まない場合は、items を空の配列にする。

# 抜粋（書籍: {book}）
{text}
"""

MCQ_PROMPT = """あなたは工学の試験問題の作成者です。以下の教科書の抜粋をもとに、工学の専門知識を問う 4 択問題を最大{n}問作ってください。

# 規則
- 正解は抜粋に明記されている事実にする。一般常識だけで解ける問題は作らない。
- 誤答（distractors）は 3 つ。同じ分野でもっともらしいが、抜粋に照らして明確に誤りであるものにする。
- 正解と誤答は、長さ・言い回し・粒度をそろえる。「すべて正しい」「どれでもない」は使わない。
- 問題文は、抜粋を読んでいない人にも意味が通るように書く。「本文」「この文章」「上記」「図3」「表2」のような参照は使わない。
- evidence には、正解の根拠となる抜粋中の文をそのまま 1 文だけ書き写す。
- 抜粋が目次・図のラベル・表の断片だけで問題を作れない場合は、items を空の配列にする。

# 抜粋（書籍: {book}）
{text}
"""

VERIFY_PROMPT = """以下の抜粋だけを根拠に、4 択問題に答えてください。選択肢の番号（1〜4）だけを answer に入れてください。

# 抜粋
{text}

# 問題
{question}

1. {c0}
2. {c1}
3. {c2}
4. {c3}
"""

SFT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"],
            },
        }
    },
    "required": ["items"],
}

MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "correct": {"type": "string"},
                    "distractors": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                    "evidence": {"type": "string"},
                },
                "required": ["question", "correct", "distractors", "evidence"],
            },
        }
    },
    "required": ["items"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer", "enum": [1, 2, 3, 4]}},
    "required": ["answer"],
}
