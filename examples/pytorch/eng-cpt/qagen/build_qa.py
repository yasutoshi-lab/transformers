"""SFT 用 QA と評価用 4 択を、生成から最終ファイル・manifest まで 1 本で作る.

Build the SFT QA set and the 4-choice eval set end-to-end, then write a provenance manifest.

工程（``--steps`` で選択、既定は全工程）:
    mcq        4 択を生成（失敗チャンクは ``--max-rounds`` 回まで再試行）→ フィルタ → シャッフル → 自己検証
    sft        SFT 用 QA を生成（同上）→ フィルタ
    rebalance  4 択の誤答を正解と長さ・粒度をそろえて作り直す → 自己検証
    manifest   入力・コード・環境・生成条件・出力を ``artifacts/qa/qa_manifest.json`` に記録

``--postprocess-only`` を付けると LLM を呼ばず、保存済みの生出力と検証キャッシュから
最終ファイルだけを作り直す（再現性の確認に使う）。

使い方（eng-cpt/ 直下。vLLM は qagen/docker-compose.ws3-arc.yml で起動しておく）:
    python -m qagen.build_qa
    python -m qagen.build_qa --postprocess-only --steps mcq sft rebalance manifest
    python -m qagen.build_qa --steps manifest --history qagen/history/2026-10-04.json
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from data_prep import provenance
from qagen import generate_qa, rebalance_mcq


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = generate_qa.OUT_DIR
DATA_DIR = generate_qa.DATA_DIR
COMPOSE_FILE = Path(__file__).resolve().parent / "docker-compose.ws3-arc.yml"
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.py"

# 生成に使うモデルと vLLM イメージ（compose と同じ値。revision は compose の --revision と一致させる）
GENERATOR_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
GENERATOR_REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
DRAFT_MODEL = "google/gemma-4-26B-A4B-it-assistant"
DRAFT_REVISION = "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"
VLLM_IMAGE = "local/vllm-openai:vtm"
PACKAGES = ["openai", "opencc", "datasketch", "tokenizers", "huggingface_hub"]
OUTPUT_FILES = ["build_timings.json", "sft_raw.jsonl", "sft.jsonl", "mcq_raw.jsonl", "mcq_verify_cache.jsonl", "mcq_eval_v1.jsonl",
                "rebalance_raw.jsonl", "rebalance_verify_cache.jsonl", "mcq_eval.jsonl", "qa_stats.json",
                "rebalance_stats.json"]


def todo_count(task):
    """生出力にまだ無いチャンクの数を返す.

    Args:
        task (str): ``sft`` / ``mcq``。

    Returns:
        int: 未生成のチャンク数。
    """
    chunks = generate_qa.load_chunks(generate_qa.TASKS[task]["split"])
    raw = OUT_DIR / f"{task}_raw.jsonl"
    done = set()
    if raw.exists():
        with open(raw) as f:
            done = {json.loads(line)["chunk_id"] for line in f}
    return sum(c["chunk_id"] not in done for c in chunks)


def run_task(task, args):
    """1 タスクを生成→後処理まで実行する（失敗チャンクは最大 ``max_rounds`` 回まで再試行）.

    Args:
        task (str): ``sft`` / ``mcq``。
        args (argparse.Namespace): コマンドライン引数。

    Returns:
        None

    Raises:
        RuntimeError: 再試行しても未生成のチャンクが残った場合。
    """
    ns = argparse.Namespace(task=task, base_url=args.base_url, concurrency=args.concurrency, limit=0,
                            postprocess_only=args.postprocess_only)
    if args.postprocess_only:
        asyncio.run(generate_qa.run(ns))
        return
    # 各ラウンドで未生成チャンクだけを生成し後処理まで行う（自己検証はキャッシュされるので重複しない）
    for round_ in range(1, args.max_rounds + 1):
        print(f"[{task}] generation round {round_}", flush=True)
        asyncio.run(generate_qa.run(ns))
        if todo_count(task) == 0:
            return
    raise RuntimeError(f"{task}: {todo_count(task)} チャンクが未生成のままです")


def write_manifest(args):
    """QA 作成の出所を ``qa_manifest.json`` に記録する.

    Args:
        args (argparse.Namespace): コマンドライン引数（``history`` を含む）。

    Returns:
        dict: 書き出した manifest。
    """
    corpus_manifest = DATA_DIR / "manifest.json"
    manifest = {
        "kind": "eng-cpt QA (SFT + 4-choice eval)",
        "created_at": provenance.now_iso(),
        "code": provenance.git_info(paths=["examples/pytorch/eng-cpt"]),
        "environment": provenance.package_versions(PACKAGES),
        "inputs": {
            "corpus_manifest": provenance.file_record(corpus_manifest, DATA_DIR) if corpus_manifest.exists() else None,
            "splits": [provenance.file_record(DATA_DIR / f"{s}.jsonl", DATA_DIR) for s in ("train", "qa_eval")],
        },
        "generator": {
            "model": GENERATOR_MODEL, "revision": GENERATOR_REVISION,
            "speculative_draft": DRAFT_MODEL, "draft_revision": DRAFT_REVISION,
            "vllm_image": VLLM_IMAGE, "vllm_image_id": args.vllm_image_id or provenance.docker_image_id(VLLM_IMAGE),
            "compose": provenance.file_record(COMPOSE_FILE, ROOT),
            "prompts": provenance.file_record(PROMPTS_FILE, ROOT),
            "thinking": False, "response_format": "json_schema",
        },
        "params": {
            "chunk_chars": generate_qa.CHUNK_CHARS, "min_chunk_chars": generate_qa.MIN_CHUNK_CHARS,
            "items_per_chunk": generate_qa.ITEMS_PER_CHUNK, "shuffle_seed": generate_qa.SHUFFLE_SEED,
            "tasks": {k: {kk: v[kk] for kk in ("split", "temperature", "max_tokens")}
                      for k, v in generate_qa.TASKS.items()},
            "rebalance": {"len_tolerance": rebalance_mcq.LEN_TOLERANCE, "min_len_slack": rebalance_mcq.MIN_LEN_SLACK,
                          "max_attempts": rebalance_mcq.MAX_ATTEMPTS, "temperature": 0.7},
            "verify": {"temperature": 0.0},
        },
        "outputs": [provenance.file_record(OUT_DIR / name, OUT_DIR) for name in OUTPUT_FILES if (OUT_DIR / name).exists()],
        "history": json.loads(Path(args.history).read_text()) if args.history else None,
    }
    (OUT_DIR / "qa_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"wrote {OUT_DIR / 'qa_manifest.json'}")
    return manifest


def main():
    """指定された工程を順に実行する.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", nargs="+", default=["mcq", "sft", "rebalance", "manifest"],
                    choices=["mcq", "sft", "rebalance", "manifest"])
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-rounds", type=int, default=3, help="失敗チャンクの再試行を含む生成の最大回数")
    ap.add_argument("--postprocess-only", action="store_true", help="LLM を呼ばず、生出力とキャッシュから作り直す")
    ap.add_argument("--history", default=None, help="実際の作成経緯を記した JSON（manifest に取り込む）")
    ap.add_argument("--vllm-image-id", default=None, help="vLLM イメージが無いホストで manifest を作る場合に指定")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timings_path = OUT_DIR / "build_timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else []
    for step in args.steps:
        print(f"== step: {step}", flush=True)
        t0 = time.time()
        if step in ("mcq", "sft"):
            run_task(step, args)
        elif step == "rebalance":
            asyncio.run(rebalance_mcq.run(argparse.Namespace(base_url=args.base_url, concurrency=args.concurrency,
                                                             postprocess_only=args.postprocess_only)))
        else:
            write_manifest(args)
        # コスト算出用: 工程ごとの経過秒（生成・自己検証・作り直しを含む実時間）
        timings.append({"step": step, "seconds": time.time() - t0, "postprocess_only": args.postprocess_only,
                        "concurrency": args.concurrency, "finished_at": provenance.now_iso()})
        timings_path.write_text(json.dumps(timings, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
