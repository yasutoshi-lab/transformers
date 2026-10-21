import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_think(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """思考ありの回答から実行するコードを作る。

    `</think>` より後の回答から最初の ```python ブロックを取り出す。ブロックに問題の関数の
    定義があればそのまま使い（問題文の import を前に付ける）、無ければ問題文に本体としてつなげる。
    """
    import re

    def one(r: str, doc: dict) -> str:
        ans = r.split("</think>")[-1]
        m = re.search(r"```(?:python|py)?\n(.*?)(?:```|$)", ans, re.S)
        code = m.group(1) if m else ans
        if f"def {doc['entry_point']}(" in code:
            head = "\n".join(l for l in doc["prompt"].splitlines() if l.startswith(("import ", "from ")))
            return head + "\n" + code
        return doc["prompt"] + code

    return [[one(r, doc) for r in resp] for resp, doc in zip(resps, docs)]
