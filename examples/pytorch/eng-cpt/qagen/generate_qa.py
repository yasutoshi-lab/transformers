"""前処理済みコーパスから SFT 用 QA と評価用 4 択問題を生成する.

Generate SFT QA pairs (from ``train``) and 4-choice eval questions (from ``qa_eval``)
with a local vLLM server (OpenAI-compatible API).

流れ:
    1. 文書を段落境界で約 ``CHUNK_CHARS`` 字のチャンクに分ける
    2. チャンクごとに LLM へ JSON スキーマ固定で生成させ、生出力を ``*_raw.jsonl`` へ追記する
       （チャンク単位で追記するため、中断しても処理済みチャンクを飛ばして再開できる）
    3. 機械フィルタ（抜粋への参照表現・重複・空・選択肢の重複）をかける
    4. 4 択問題は、抜粋を見せてモデル自身に解かせ、正解できたものだけを残す
    5. 選択肢を固定シードでシャッフルし、最終ファイルを書き出す

出力（``artifacts/qa/``、git 追跡外）:
    sft.jsonl       {id, source_id, chunk_id, book, category, messages: [user, assistant]}
    mcq_eval.jsonl  {id, source_id, chunk_id, book, category, question, choices[4], answer(0-3), evidence}
    qa_stats.json   各段階の件数

使い方（eng-cpt/ 直下で実行。vLLM は qagen/docker-compose.ws3-arc.yml）:
    python -m qagen.generate_qa --task sft
    python -m qagen.generate_qa --task mcq
"""

import argparse
import asyncio
import collections
import json
import random
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from qagen.prompts import MCQ_PROMPT, MCQ_SCHEMA, SFT_PROMPT, SFT_SCHEMA, VERIFY_PROMPT, VERIFY_SCHEMA


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "data"
OUT_DIR = ROOT / "artifacts" / "qa"
CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 300
ITEMS_PER_CHUNK = 4
SHUFFLE_SEED = 20261004

TASKS = {
    "sft": {"split": "train", "prompt": SFT_PROMPT, "schema": SFT_SCHEMA, "temperature": 0.7, "max_tokens": 3000},
    "mcq": {"split": "qa_eval", "prompt": MCQ_PROMPT, "schema": MCQ_SCHEMA, "temperature": 0.3, "max_tokens": 3000},
}
# 抜粋を前提にした（自己完結しない）表現。含む QA は捨てる
RE_SOURCE_REF = re.compile(
    r"本文|この文章|上記|抜粋|本書|この章|図\s*[\d０-９]|表\s*[\d０-９]|式\s*[\(（]?[\d０-９]|点\s*[A-ZＡ-Ｚ](?![A-Za-z])"
)


def chunk_document(text, size=CHUNK_CHARS):
    """文書を段落境界で約 ``size`` 字のチャンクに分ける.

    処理概要: 空行区切りの段落を順に詰め、``size`` を超えたら区切る。
    1 段落が ``size`` の 2 倍を超える場合は改行単位で分割する。末尾の短すぎる
    チャンクは直前のチャンクへ併合する。

    Args:
        text (str): 文書本文。
        size (int): 目標チャンク長（文字数）。

    Returns:
        list[str]: チャンクのリスト。
    """
    units = []
    for para in re.split(r"\n{2,}", text):
        if len(para) > size * 2:
            units.extend(ln for ln in para.split("\n") if ln.strip())
        elif para.strip():
            units.append(para)
    chunks, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) > size:
            chunks.append(buf)
            buf = u
        else:
            buf = f"{buf}\n\n{u}" if buf else u
    if buf:
        if chunks and len(buf) < MIN_CHUNK_CHARS:
            chunks[-1] += "\n\n" + buf
        else:
            chunks.append(buf)
    return chunks


def load_chunks(split):
    """分割の JSONL を読み込み、チャンク単位のタスクに展開する.

    Args:
        split (str): ``train`` / ``qa_eval``。

    Returns:
        list[dict]: ``{chunk_id, source_id, book, category, text}`` のリスト。
    """
    out = []
    with open(DATA_DIR / f"{split}.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            for i, text in enumerate(chunk_document(doc["text"])):
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                out.append({"chunk_id": f"{doc['id']}#{i:02d}", "source_id": doc["id"], "book": doc["book"],
                            "category": doc["category"], "text": text})
    return out


# LLM が JSON 文字列内の LaTeX のバックスラッシュをエスケープせずに書くと、"\frac" の "\f" などが
# JSON の制御文字として解釈されて壊れる（例: "\frac" → 改ページ + "rac"）。パース後に復元する。
LATEX_CONTROL_RESTORE = {"\x0c": "\\f", "\x08": "\\b", "\t": "\\t", "\r": "\\r"}
RE_INLINE_MATH = re.compile(r"\$[^$]*\$")


def fix_latex_escapes(value):
    """JSON パースで制御文字化した LaTeX コマンドを復元する（dict / list は再帰的に処理）.

    処理概要: 改ページ・バックスペース・タブ・復帰を ``\\f`` ``\\b`` ``\\t`` ``\\r`` に戻す。
    改行は通常の文中改行と区別できないため、``$...$`` の中で英字が続く場合（``\\nu`` 等）だけ戻す。

    Args:
        value (str | dict | list | object): パース済みの値。

    Returns:
        str | dict | list | object: 復元後の値（文字列以外はそのまま）。
    """
    if isinstance(value, dict):
        return {k: fix_latex_escapes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fix_latex_escapes(v) for v in value]
    if not isinstance(value, str):
        return value
    for ch, rep in LATEX_CONTROL_RESTORE.items():
        value = value.replace(ch, rep)
    return RE_INLINE_MATH.sub(lambda m: re.sub(r"\n(?=[A-Za-z])", r"\\n", m.group()), value)


async def call_json(client, prompt, schema, temperature, max_tokens):
    """JSON スキーマ固定で 1 回生成し、パース済みの dict を返す.

    Args:
        client (AsyncOpenAI): vLLM の OpenAI 互換クライアント。
        prompt (str): ユーザープロンプト。
        schema (dict): 出力 JSON スキーマ。
        temperature (float): サンプリング温度。
        max_tokens (int): 最大生成トークン数。

    Returns:
        tuple[dict, dict]: ``(パース結果, usage)``。

    Raises:
        json.JSONDecodeError: 出力が JSON として読めない場合（長さ打ち切りなど）。
    """
    resp = await client.chat.completions.create(
        model="qagen",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
    return fix_latex_escapes(json.loads(resp.choices[0].message.content)), usage


async def generate(client, task, chunks, raw_path, concurrency):
    """未処理のチャンクだけ生成し、生出力を ``raw_path`` に追記する.

    Args:
        client (AsyncOpenAI): vLLM クライアント。
        task (dict): ``TASKS`` の要素。
        chunks (list[dict]): ``load_chunks`` の結果。
        raw_path (pathlib.Path): 生出力の JSONL（再開時は既存行を読んで処理済みを判定）。
        concurrency (int): 同時リクエスト数。

    Returns:
        None
    """
    done = set()
    if raw_path.exists():
        with open(raw_path) as f:
            done = {json.loads(line)["chunk_id"] for line in f}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"chunks: total={len(chunks)} done={len(done)} todo={len(todo)}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    progress = collections.Counter()

    async def worker(c):
        prompt = task["prompt"].format(n=ITEMS_PER_CHUNK, book=c["book"], text=c["text"])
        async with sem:
            try:
                parsed, usage = await call_json(client, prompt, task["schema"], task["temperature"], task["max_tokens"])
                rec = {**{k: c[k] for k in ("chunk_id", "source_id", "book", "category")},
                       "items": parsed.get("items", []), "usage": usage}
            except Exception as e:  # noqa: BLE001 — 失敗チャンクは記録せず、再実行時に再試行する
                progress["error"] += 1
                print(f"error {c['chunk_id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                return
        async with lock:
            with open(raw_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            progress["ok"] += 1
            if progress["ok"] % 50 == 0:
                print(f"progress {progress['ok']}/{len(todo)}", flush=True)

    await asyncio.gather(*(worker(c) for c in todo))
    print(f"generated: ok={progress['ok']} error={progress['error']}", flush=True)


def read_raw(raw_path):
    """生出力を読み、チャンク情報付きの item に平坦化する.

    Args:
        raw_path (pathlib.Path): 生出力の JSONL。

    Returns:
        tuple[list[dict], dict]: ``(items, usage 合計)``。
    """
    items, usage = [], collections.Counter()
    with open(raw_path) as f:
        for line in f:
            rec = json.loads(line)
            usage.update(rec["usage"])
            for it in rec["items"]:
                items.append({**{k: rec[k] for k in ("chunk_id", "source_id", "book", "category")}, **it})
    return items, dict(usage)


def _norm(s):
    """重複判定用に空白と句読点を除いた文字列を返す.

    Args:
        s (str): 対象文字列。

    Returns:
        str: 正規化済み文字列。
    """
    return re.sub(r"[\s、。,.．？?！!]", "", s)


def filter_sft(items, stats):
    """SFT 用 QA に機械フィルタをかける.

    Args:
        items (list[dict]): ``read_raw`` の結果。
        stats (collections.Counter): 除去件数の加算先。

    Returns:
        list[dict]: フィルタ後の item。
    """
    seen, kept = set(), []
    for it in items:
        q, a = it["question"].strip(), it["answer"].strip()
        if len(q) < 8 or len(a) < 10:
            stats["sft_drop_too_short"] += 1
        elif RE_SOURCE_REF.search(q) or RE_SOURCE_REF.search(a):
            stats["sft_drop_source_ref"] += 1
        elif _norm(q) in seen:
            stats["sft_drop_dup"] += 1
        else:
            seen.add(_norm(q))
            kept.append({**it, "question": q, "answer": a})
    return kept


def filter_mcq(items, stats):
    """4 択問題に機械フィルタをかける.

    Args:
        items (list[dict]): ``read_raw`` の結果。
        stats (collections.Counter): 除去件数の加算先。

    Returns:
        list[dict]: フィルタ後の item。
    """
    seen, kept = set(), []
    for it in items:
        q = it["question"].strip()
        opts = [it["correct"].strip()] + [d.strip() for d in it["distractors"]]
        if len(opts) != 4 or any(not o for o in opts):
            stats["mcq_drop_malformed"] += 1
        elif len({_norm(o) for o in opts}) < 4:
            stats["mcq_drop_dup_choice"] += 1
        elif RE_SOURCE_REF.search(q) or any(RE_SOURCE_REF.search(o) for o in opts):
            stats["mcq_drop_source_ref"] += 1
        elif _norm(q) in seen:
            stats["mcq_drop_dup"] += 1
        else:
            seen.add(_norm(q))
            kept.append({**it, "question": q, "options": opts})
    return kept


def shuffle_choices(item, rng):
    """選択肢をシャッフルし、正解の位置（0〜3）を返す.

    Args:
        item (dict): ``options[0]`` が正解の item。
        rng (random.Random): 固定シードの乱数生成器。

    Returns:
        tuple[list[str], int]: ``(choices, answer_index)``。
    """
    order = list(range(4))
    rng.shuffle(order)
    return [item["options"][i] for i in order], order.index(0)


async def verify_mcq(client, items, chunk_text, concurrency, stats):
    """抜粋を見せてモデル自身に解かせ、正解できた問題だけを残す.

    Args:
        client (AsyncOpenAI): vLLM クライアント。
        items (list[dict]): ``filter_mcq`` の結果（``choices`` / ``answer`` 付与済み）。
        chunk_text (dict[str, str]): chunk_id→抜粋本文。
        concurrency (int): 同時リクエスト数。
        stats (collections.Counter): 件数の加算先。

    Returns:
        list[dict]: 検証を通過した item。
    """
    sem = asyncio.Semaphore(concurrency)

    async def check(it):
        prompt = VERIFY_PROMPT.format(text=chunk_text[it["chunk_id"]], question=it["question"],
                                      c0=it["choices"][0], c1=it["choices"][1], c2=it["choices"][2], c3=it["choices"][3])
        async with sem:
            try:
                parsed, _ = await call_json(client, prompt, VERIFY_SCHEMA, 0.0, 20)
                return parsed["answer"] - 1 == it["answer"]
            except Exception:  # noqa: BLE001 — 検証できない問題は採用しない
                return False

    ok = await asyncio.gather(*(check(it) for it in items))
    stats["mcq_drop_verify_failed"] += sum(not x for x in ok)
    return [it for it, x in zip(items, ok) if x]


def write_jsonl(path, rows):
    """行のリストを JSONL で書き出す.

    Args:
        path (pathlib.Path): 出力先。
        rows (list[dict]): 書き出す行。

    Returns:
        None
    """
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def run(args):
    """生成 → フィルタ →（4 択は検証）→ 書き出しまでを実行する.

    Args:
        args (argparse.Namespace): コマンドライン引数。

    Returns:
        None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task = TASKS[args.task]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    chunks = load_chunks(task["split"])
    if args.limit:
        chunks = chunks[: args.limit]
    raw_path = OUT_DIR / f"{args.task}_raw.jsonl"
    t0 = time.time()
    await generate(client, task, chunks, raw_path, args.concurrency)
    gen_seconds = time.time() - t0

    items, usage = read_raw(raw_path)
    stats = collections.Counter({f"{args.task}_chunks": len(chunks), f"{args.task}_raw_items": len(items)})
    common = ("source_id", "chunk_id", "book", "category")
    if args.task == "sft":
        kept = filter_sft(items, stats)
        rows = [{"id": f"sft-{i:05d}", **{k: it[k] for k in common},
                 "messages": [{"role": "user", "content": it["question"]},
                              {"role": "assistant", "content": it["answer"]}]} for i, it in enumerate(kept)]
        write_jsonl(OUT_DIR / "sft.jsonl", rows)
    else:
        kept = filter_mcq(items, stats)
        rng = random.Random(SHUFFLE_SEED)
        for it in kept:
            it["choices"], it["answer"] = shuffle_choices(it, rng)
        chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
        kept = await verify_mcq(client, kept, chunk_text, args.concurrency, stats)
        rows = [{"id": f"mcq-{i:04d}", **{k: it[k] for k in common}, "question": it["question"],
                 "choices": it["choices"], "answer": it["answer"], "evidence": it["evidence"]} for i, it in enumerate(kept)]
        write_jsonl(OUT_DIR / "mcq_eval.jsonl", rows)
    stats[f"{args.task}_final"] = len(rows)

    stats_path = OUT_DIR / "qa_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    # コスト算出用: 生成の経過秒（再開時は今回分のみ。全件分は runs に累積する）
    prev = all_stats.get(args.task, {}).get("runs", [])
    runs = prev + [{"chunks_generated": len(chunks), "generate_seconds": gen_seconds, "concurrency": args.concurrency}]
    all_stats[args.task] = {"counts": dict(stats), "usage": usage, "runs": runs}
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=1))
    print(json.dumps(all_stats[args.task], ensure_ascii=False, indent=1))


def main():
    """コマンドライン引数を解釈して生成を実行する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N チャンクだけ処理（試験用）")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
