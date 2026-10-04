"""4 択問題の誤答を、正解と長さ・粒度をそろえて作り直す.

Regenerate distractors so that the correct choice is not identifiable by its length.

初版の 4 択（``mcq_eval.jsonl``）では、正解が最長の選択肢である割合が 39%（偶然なら 25%）あり、
「一番長い選択肢を選ぶ」だけで正解率が上がってしまっていた。
各問題について、正解の文字数の ±``LEN_TOLERANCE`` に収まる誤答 3 つを生成させ、
条件を満たさなければ最大 ``MAX_ATTEMPTS`` 回まで再生成する。最後に抜粋を見せた自己検証を通す。

入出力（``artifacts/qa/``）:
    mcq_eval.jsonl     入力（初回実行時に mcq_eval_v1.jsonl として退避）→ 作り直し後で上書き
    rebalance_stats.json  件数と長さ偏りの前後比較

使い方（eng-cpt/ 直下で実行）:
    python -m qagen.rebalance_mcq
"""

import argparse
import asyncio
import collections
import json
import random
import shutil

from openai import AsyncOpenAI

from qagen.generate_qa import OUT_DIR, SHUFFLE_SEED, _norm, call_json, load_chunks, shuffle_choices, verify_mcq


LEN_TOLERANCE = 0.25
MIN_LEN_SLACK = 3        # 正解が短い場合は ±3 文字までを許容する
MAX_ATTEMPTS = 3

REBALANCE_PROMPT = """以下の 4 択問題について、誤答（distractors）だけを 3 つ作り直してください。

# 規則
- 各誤答の長さは、正解（{n_chars} 文字）とほぼ同じ（{lo}〜{hi} 文字）にする。
- 正解と同じ書き方・粒度・専門性にそろえる（正解だけが詳しい・具体的、という差を作らない）。
- 抜粋に照らして明確に誤りである内容にする。正解の言い換えや部分的に正しい内容は避ける。
- 「すべて正しい」「どれでもない」は使わない。

# 抜粋
{text}

# 問題
{question}

# 正解
{correct}
"""

REBALANCE_SCHEMA = {
    "type": "object",
    "properties": {"distractors": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
    "required": ["distractors"],
}


def length_bounds(correct):
    """正解の文字数から、誤答に許す文字数の範囲を返す.

    Args:
        correct (str): 正解の選択肢。

    Returns:
        tuple[int, int]: ``(下限, 上限)``。
    """
    n = len(correct)
    slack = max(MIN_LEN_SLACK, round(n * LEN_TOLERANCE))
    return max(1, n - slack), n + slack


def longest_is_correct_rate(rows):
    """正解が最長の選択肢である割合を返す.

    Args:
        rows (list[dict]): ``choices`` / ``answer`` を持つ 4 択問題。

    Returns:
        float: 割合（偶然なら 0.25）。
    """
    hit = sum(max(range(4), key=lambda i: len(r["choices"][i])) == r["answer"] for r in rows)
    return hit / len(rows)


async def regenerate(client, item, chunk_text, sem, stats):
    """1 問の誤答を、長さ条件を満たすまで最大 ``MAX_ATTEMPTS`` 回生成する.

    Args:
        client (AsyncOpenAI): vLLM クライアント。
        item (dict): 4 択問題（``choices`` / ``answer``）。
        chunk_text (dict[str, str]): chunk_id→抜粋本文。
        sem (asyncio.Semaphore): 同時実行数の制御。
        stats (collections.Counter): 件数の加算先。

    Returns:
        dict: ``options``（先頭が正解）を差し替えた item。条件を満たせなければ最後の試行結果。
    """
    correct = item["choices"][item["answer"]]
    lo, hi = length_bounds(correct)
    prompt = REBALANCE_PROMPT.format(n_chars=len(correct), lo=lo, hi=hi, text=chunk_text[item["chunk_id"]],
                                     question=item["question"], correct=correct)
    best = [c for i, c in enumerate(item["choices"]) if i != item["answer"]]
    for attempt in range(MAX_ATTEMPTS):
        async with sem:
            try:
                parsed, _ = await call_json(client, prompt, REBALANCE_SCHEMA, 0.7, 800)
            except Exception:  # noqa: BLE001 — 失敗した試行は次の試行へ
                continue
        ds = [d.strip() for d in parsed["distractors"]]
        if len({_norm(x) for x in ds + [correct]}) < 4:
            continue
        best = ds
        if all(lo <= len(d) <= hi for d in ds):
            stats[f"length_ok_attempt{attempt + 1}"] += 1
            return {**item, "options": [correct] + ds}
    stats["length_not_met"] += 1
    return {**item, "options": [correct] + best}


async def run(args):
    """誤答を作り直し、シャッフル・自己検証して mcq_eval.jsonl を上書きする.

    Args:
        args (argparse.Namespace): コマンドライン引数。

    Returns:
        None
    """
    path, backup = OUT_DIR / "mcq_eval.jsonl", OUT_DIR / "mcq_eval_v1.jsonl"
    if not backup.exists():
        shutil.copy(path, backup)
    with open(backup) as f:
        rows = [json.loads(line) for line in f]
    chunk_text = {c["chunk_id"]: c["text"] for c in load_chunks("qa_eval")}
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    stats = collections.Counter(input=len(rows))
    sem = asyncio.Semaphore(args.concurrency)
    items = await asyncio.gather(*(regenerate(client, r, chunk_text, sem, stats) for r in rows))

    rng = random.Random(SHUFFLE_SEED)
    for it in items:
        it["choices"], it["answer"] = shuffle_choices(it, rng)
    verified = await verify_mcq(client, items, chunk_text, args.concurrency, stats)
    out_rows = [{"id": r["id"], **{k: r[k] for k in ("source_id", "chunk_id", "book", "category", "question")},
                 "choices": r["choices"], "answer": r["answer"], "evidence": r["evidence"]} for r in verified]
    with open(path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["output"] = len(out_rows)
    report = {"counts": dict(stats),
              "longest_is_correct": {"before": longest_is_correct_rate(rows), "after": longest_is_correct_rate(out_rows)}}
    (OUT_DIR / "rebalance_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main():
    """コマンドライン引数を解釈して実行する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
