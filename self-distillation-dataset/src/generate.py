"""gpt-oss-20b 自身による自己蒸留データ生成パイプライン。

設計方針（wiki 側の知見と 2026-10-10 のレビューに基づく）:

- 教師は外部に置かない。生成モデル = 対象モデル自身（openai/gpt-oss-20b）。
  特権情報（参照解）を与えない素朴 on-policy 生成のみを使う（特権情報を与えると
  thinking モデルで fork suppression が起きる —
  wiki/sources/rethinking-on-policy-self-distillation-for-thinking-models）。
- reasoning あり / なしを同量生成する。Harmony 形式では生成開始チャネルが
  `analysis` か `final` かで決まるため、chat template の `enable_thinking` で制御し、
  書き出し（プレフィックス）を明示的に付与する。
- prefix はチャネルで分ける。`analysis` に回答調の書き出しを注入すると橋渡しが
  句読点の断片になる（wiki 側 `concepts/prefix-failure`）。reasoning は推論調の英語、
  no_reasoning はドメイン言語の回答調。
- reasoning_effort は Curator 方針で medium を最大割合、low / high を含む。
- 途中切れは `finish_reason == "length"` で除去。
- 言語混入・重複は規則ベースで除去（filters.py / dedup.py）。thinking は英語 only。
- 目標は**通過数**（Curator 決定）。ドメイン別 quota はプールサイズで上限をかける。

実行例:

  # vLLM サーバー（ws1-arc）
  docker run --rm --gpus all -p 8000:8000 \\
    -v ~/.cache/huggingface:/data \\
    vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \\
    --model openai/gpt-oss-20b --served-model-name gpt-oss-20b \\
    --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.80

  # 通過数 10 万件（dialogue は今回の実行から除外）
  python3 src/generate.py --target-kept 100000 --concurrency 32 \\
    --output out/full.jsonl --upload-to-hf yasutoshi-lab/gpt-oss-20b-self-distillation-100k
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

import dedup
import filters
import seeds

CHANNEL_FINAL = "<|start|>assistant<|channel|>final<|message|>"
# chat template は `<|start|>assistant` で終わりチャネルヘッダを付けない。prefix は
# 必ずヘッダの後ろに置く（ヘッダ前に置くと壊れた位置に入りモデルがヘッダを出し直す）。
GEN_HEADER = {
    "reasoning": "<|channel|>analysis<|message|>",
    "no_reasoning": "<|channel|>final<|message|>",
}
END_TOKEN = "<|end|>"

# json.dumps は U+2028 / U+2029 / U+0085 / \x0b / \x1c-\x1e をエスケープしないため、
# これらを含むレコードは行読み取り（splitlines）で分断される。生成側で置換する。
LINE_SEPARATORS = dict.fromkeys([0x2028, 0x2029, 0x0085, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E], " ")

# japanese 域は域の定義が「日本語で答える」。seed が英語プロンプトでも、言語指示は隠れた
# system ではなくユーザープロンプト末尾に付けて record に残す（thinking が存在しない
# system 指示を引用すると、学習時に再現できない特権情報になるため）。
JAPANESE_USER_SUFFIX = "\n\n日本語で、簡単な説明を添えて答えてください。"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Reason in English in the analysis channel. "
    "Answer in the language of the user's request."
)


def sanitize(text: str) -> str:
    """行区切りとして振る舞う文字を空白に置換する。

    Args:
        text: 対象テキスト。

    Returns:
        置換済み文字列。
    """
    return (text or "").translate(LINE_SEPARATORS)


def render_prompt(tokenizer, messages: list[dict], mode: str, effort: str) -> str:
    """Harmony 形式のプロンプト文字列を組み立てる。

    Args:
        tokenizer: gpt-oss の tokenizer（chat_template を持つ）。
        messages: chat messages（system / user）。
        mode: "reasoning" または "no_reasoning"。
        effort: reasoning_effort（reasoning モードのみ使用）。

    Returns:
        生成開始チャネルヘッダまでを含む prompt 文字列。
    """
    kw = {"reasoning_effort": effort} if mode == "reasoning" else {}
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=(mode == "reasoning"),
        tokenize=False,
        **kw,
    )


def parse_completion(raw: str, mode: str, prefix: str = "") -> tuple[str, str]:
    """vLLM の completion 本文を thinking / final に分離する。

    Args:
        raw: 生成された生テキスト（チャネルヘッダを含む）。
        mode: "reasoning" または "no_reasoning"。
        prefix: プロンプト側に置いた書き出し。生成本文は prefix の続きなので先頭に戻す。

    Returns:
        (thinking, response)。reasoning なしモードでは thinking は空。
    """
    raw = (prefix + raw).replace(END_TOKEN, "").strip()
    if mode == "reasoning":
        parts = raw.split(CHANNEL_FINAL)
        thinking = parts[0].strip()
        response = parts[1].strip() if len(parts) > 1 else ""
        return thinking, response
    return "", raw


def build_session(concurrency: int) -> requests.Session:
    """接続プールを concurrency に合わせた HTTP セッションを作る。

    Args:
        concurrency: 並列数。

    Returns:
        requests.Session。
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=concurrency, max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def generate_one(
    session: requests.Session,
    base_url: str,
    model: str,
    tokenizer,
    prompt: str,
    prefix: str,
    mode: str,
    effort: str,
    max_tokens: int,
    timeout: int,
    system_prompt: str,
) -> dict:
    """1 件生成してレコード dict を返す。

    Args:
        session: HTTP セッション。
        base_url: vLLM の `/v1` ベース URL。
        model: served model name。
        tokenizer: Harmony レンダリング用 tokenizer。
        prompt: 元プロンプト。
        prefix: 書き出しプレフィックス。
        mode: "reasoning" / "no_reasoning"。
        effort: reasoning_effort。
        max_tokens: 生成上限。
        timeout: リクエストタイムアウト秒。
        system_prompt: system メッセージ本文。

    Returns:
        レコード dict（filter 判定前）。失敗時は `error` に理由が入る。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = render_prompt(tokenizer, messages, mode, effort) + GEN_HEADER[mode] + prefix
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        # vLLM の既定（skip_special_tokens=True）は Harmony のチャネルマーカーを
        # デコードから除き、素の `assistantfinal` になる。analysis / final の分離には
        # special token の保持が必要。
        "skip_special_tokens": False,
    }
    try:
        resp = session.post(f"{base_url.rstrip('/')}/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
    except requests.RequestException as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    thinking, response = parse_completion(choice["text"], mode, prefix)
    return {
        "error": "",
        "thinking": sanitize(thinking),
        "response": sanitize(response),
        "finish_reason": choice.get("finish_reason", ""),
    }


def load_state(path: Path, deduper: dedup.Deduplicator) -> tuple[set[str], Counter, int]:
    """既存出力から resume 用の状態を復元する。

    済みキーに加え、(domain, mode) 別の通過数を数え直し、全レコードを出力順に
    deduper へ流して重複判定の index を再構築する。中断で末尾が書きかけの行は切り捨てる
    （次の追記が壊れた行に連結されるのを防ぐ）。

    Args:
        path: 出力 JSONL。
        deduper: 状態を復元する重複除去器。

    Returns:
        (済みキーの set, (domain, mode) 別通過数, 既存レコード数)。
    """
    seen: set[str] = set()
    kept: Counter = Counter()
    if not path.exists():
        return seen, kept, 0
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with path.open("r+b") as fh:
            fh.truncate(cut)
        print(f"[resume] truncated partial tail ({len(raw) - cut} bytes)", flush=True)
        raw = raw[:cut]
    total = 0
    for line in raw.decode("utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        domain, mode = rec.get("domain"), rec.get("mode")
        seen.add(f"{domain}|{mode}|{rec.get('seed_index')}")
        deduper.is_duplicate(domain, rec.get("prompt", ""), rec.get("response", ""))
        if rec.get("passed") and not rec.get("duplicate"):
            kept[(domain, mode)] += 1
    return seen, kept, total


def run(args: argparse.Namespace) -> dict:
    """生成・フィルタ・重複除去を実行し、統計を返す。

    Args:
        args: CLI 引数。

    Returns:
        統計 dict。
    """
    rng = random.Random(args.seed)
    excluded = {d for d in args.exclude_domains.split(",") if d}
    # 除外ドメインのシードは読まない（大規模ソースの読み込みを省く）
    pool = seeds.load_seed_pool({src["domain"] for src in seeds.SEED_SOURCES} - excluded)
    sampler = seeds.SeedPool(pool, rng)
    session = build_session(args.concurrency)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    domains = sorted(pool)
    if not domains:
        return {"total": 0, "kept": 0, "reasons": {}, "by_domain_mode": {}}

    # ドメイン別 quota（通過数）。小プール域は expected_pass 換算の上限で頭打ちになり、
    # 不足分は headroom を持つドメインへ再配分する
    share = args.target_kept // len(domains)
    cap_kept = {d: int(sampler.size(d) * args.max_reuse * args.expected_pass) for d in domains}
    quota = {d: min(share, cap_kept[d]) for d in domains}
    deficit = args.target_kept - sum(quota.values())
    while deficit > 0:
        room = [d for d in domains if quota[d] < cap_kept[d]]
        if not room:
            break
        add = deficit // len(room) or 1
        for d in room:
            quota[d] = min(cap_kept[d], quota[d] + add)
        deficit = args.target_kept - sum(quota.values())
    # --domain-quota で指定したドメインは自動配分を上書きする（プール上限より優先）
    for item in filter(None, args.domain_quota.split(",")):
        name, value = item.split("=")
        if name not in quota:
            raise ValueError(f"--domain-quota: unknown domain {name!r}")
        quota[name] = int(value)
    for d in domains:
        print(f"[quota] {d}: target kept {quota[d]} (pool {sampler.size(d)}, cap {cap_kept[d]})")
    if deficit:
        print(f"[quota] unreachable: {deficit} short (pool caps)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduper = dedup.Deduplicator(threshold=args.dedup_threshold)
    seen, kept_dm, resumed = load_state(out_path, deduper)

    stats: Counter = Counter()
    reason_counter: Counter = Counter()
    domain_mode: Counter = Counter()
    kept_by_domain: Counter = Counter()
    effort_counter: Counter = Counter()
    for (d, _), c in kept_dm.items():
        kept_by_domain[d] += c
    stats["total"] = resumed
    stats["kept"] = sum(kept_dm.values())
    if resumed:
        print(f"[resume] {resumed} records, kept {stats['kept']}", flush=True)
    next_report = (stats["total"] // args.chunk_size + 1) * args.chunk_size

    with out_path.open("a", encoding="utf-8") as fh:
        for domain in domains:
            half = quota[domain] // 2
            tasks = [(domain, "reasoning", half), (domain, "no_reasoning", quota[domain] - half)]
            for domain, mode, count in tasks:
                if count <= 0:
                    continue
                generated = 0
                cap = sampler.size(domain) * args.max_reuse
                while kept_dm[(domain, mode)] < count and generated < cap:
                    batch = min(args.concurrency, cap - generated)
                    candidates = []
                    for _ in range(batch):
                        prompt, seed_index, seed_source = sampler.sample(domain)
                        if domain == "japanese" and not seeds.is_japanese_text(prompt):
                            prompt += JAPANESE_USER_SUFFIX
                        key = f"{domain}|{mode}|{seed_index}"
                        if key in seen:
                            continue
                        seen.add(key)
                        effort = seeds.sample_effort(rng) if mode == "reasoning" else ""
                        candidates.append(
                            (prompt, seed_index, seed_source, effort, seeds.sample_prefix(domain, mode, rng, prompt))
                        )
                    if not candidates:
                        break
                    generated += len(candidates)
                    def limit(effort: str) -> int:
                        """effort ごとの生成上限を返す。

                        Args:
                            effort: reasoning_effort（no_reasoning では空）。

                        Returns:
                            max_tokens。
                        """
                        return seeds.EFFORT_MAX_TOKENS.get(effort, args.max_tokens) if effort else args.max_tokens

                    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
                        futures = [
                            ex.submit(
                                generate_one,
                                session,
                                args.base_url,
                                args.served_model_name,
                                tokenizer,
                                prompt,
                                prefix,
                                mode,
                                effort,
                                limit(effort),
                                args.timeout,
                                args.system_prompt,
                            )
                            for prompt, seed_index, seed_source, effort, prefix in candidates
                        ]
                        for (prompt, seed_index, seed_source, effort, prefix), fut in zip(candidates, futures):
                            result = fut.result()
                            rec = {
                                "prompt": prompt,
                                "prefix": prefix,
                                "mode": mode,
                                "reasoning_effort": effort,
                                "thinking": result.get("thinking", ""),
                                "response": result.get("response", ""),
                                "finish_reason": result.get("finish_reason", ""),
                                "error": result.get("error", ""),
                                "domain": domain,
                                "lang": filters.detect_language(prompt),
                                "seed_source": seed_source,
                                "seed_index": seed_index,
                            }
                            verdict = filters.evaluate(rec)
                            rec["passed"] = verdict.passed
                            rec["reasons"] = verdict.reasons
                            rec["duplicate"] = (
                                deduper.is_duplicate(domain, rec["prompt"], rec["response"]) or ""
                            )
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            stats["total"] += 1
                            domain_mode[(domain, mode)] += 1
                            if effort:
                                effort_counter[effort] += 1
                            if not verdict.passed:
                                for r in verdict.reasons:
                                    reason_counter[r.split(":")[0]] += 1
                            elif rec["duplicate"]:
                                reason_counter[rec["duplicate"]] += 1
                            else:
                                stats["kept"] += 1
                                kept_by_domain[domain] += 1
                                kept_dm[(domain, mode)] += 1
                    # バッチごとに flush し、中断時の損失を 1 バッチに抑える
                    fh.flush()
                    if stats["total"] >= next_report:
                        next_report = (stats["total"] // args.chunk_size + 1) * args.chunk_size
                        print(
                            f"[progress] {time.strftime('%Y-%m-%d %H:%M:%S')} total {stats['total']} "
                            f"kept {stats['kept']} ({domain}/{mode} {kept_dm[(domain, mode)]}/{count})",
                            flush=True,
                        )
        fh.flush()

    return {
        "total": stats["total"],
        "kept": stats["kept"],
        "reasons": dict(reason_counter),
        "by_domain": {d: kept_by_domain[d] for d in domains},
        "by_domain_mode": {f"{d}/{m}": c for (d, m), c in sorted(domain_mode.items())},
        "by_effort": dict(effort_counter),
    }


def main() -> None:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="gpt-oss-20b 自己蒸留データ生成")
    parser.add_argument("--target-kept", type=int, default=1000, help="通過数の目標")
    parser.add_argument(
        "--domain-quota", default="", help="ドメイン別の通過数（例: code=50000,english=25500）。自動配分を上書き"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="tokenizer 読み込み用の HF 模型 ID")
    parser.add_argument("--served-model-name", default="gpt-oss-20b", help="vLLM の served model name")
    parser.add_argument("--output", default="out/pilot.jsonl")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096, help="no_reasoning モードの生成上限")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=1000, help="flush・進捗表示の interval")
    parser.add_argument("--dedup-threshold", type=float, default=0.80)
    parser.add_argument("--max-reuse", type=int, default=10, help="1 シードの再利用上限（quota 上限算出用）")
    parser.add_argument("--expected-pass", type=float, default=0.5, help="quota 換算用の想定通過率")
    parser.add_argument("--exclude-domains", default="dialogue")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload-to-hf", default="")
    args = parser.parse_args()

    started = time.time()
    summary = run(args)
    summary["elapsed_sec"] = round(time.time() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.upload_to_hf:
        from datasets import Dataset  # noqa: PLC0415

        records = [
            json.loads(line)
            for line in Path(args.output).read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        Dataset.from_list(records).push_to_hub(args.upload_to_hf)
        print(f"uploaded: {args.upload_to_hf}")


if __name__ == "__main__":
    main()
