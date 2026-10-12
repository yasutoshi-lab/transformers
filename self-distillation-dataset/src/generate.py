"""google/gemma-4-26B-A4B-it 自身による自己蒸留データ生成パイプライン。

gpt-oss-20b 版（gpt-oss-self-distill ブランチ）の方式を Gemma 4 へ移植したもの。

設計方針:

- 教師は外部に置かない。生成モデル = 対象モデル自身（google/gemma-4-26B-A4B-it）。
  特権情報（参照解）を与えない素朴 on-policy 生成のみを使う（特権情報を与えると
  thinking モデルで fork suppression が起きる —
  wiki/sources/rethinking-on-policy-self-distillation-for-thinking-models）。
- reasoning あり / なしを同量生成する。Gemma 4 の思考は `<|channel>thought\\n…<channel|>`
  で囲まれる。chat template の `enable_thinking=True` は system 先頭に `<|think|>` を入れ、
  生成開始は `<|turn>model\\n` で終わる。`enable_thinking=False` は生成開始に空の思考
  `<|channel>thought\\n<channel|>` を置く。prefix は必ずこのヘッダの後ろに置く
  （gpt-oss 版の教訓: ヘッダ前に置くとモデルがヘッダを出し直し、継ぎ目が断片になる）。
- Gemma 4 には gpt-oss の reasoning_effort に当たる段階指定がないため、effort 混合は行わない。
- サンプリングはモデル同梱の generation_config（temperature 1.0 / top_p 0.95 / top_k 64）。
- 系列長は max-model-len 16384。生成上限はリクエストごとにプロンプト長から逆算して切り詰める。
- 途中切れは `finish_reason == "length"` で除去。言語混入・重複は規則ベースで除去。
- 目標は**通過数**。複数 vLLM エンドポイント（GPU ごとに 1 サーバー）へ round-robin で投げ、
  (domain, mode) の進捗が quota 比で最も遅れているものから順にシードを供給する。

実行例:

  # vLLM サーバー（ws3-arc、GPU ごとに 1 コンテナ）は README 参照
  python3 src/generate.py --target-kept 500000 \\
    --base-urls http://localhost:8110/v1,http://localhost:8111/v1 \\
    --concurrency 96 --output out/full.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from transformers import AutoTokenizer

import dedup
import filters
import seeds

MODES = ("reasoning", "no_reasoning")
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
# 生成本文に残りうる終端トークン（EOS として止まった場合は通常出力されない）
END_TOKENS = ("<turn|>", "<eos>")

# json.dumps は U+2028 / U+2029 / U+0085 / \x0b / \x1c-\x1e をエスケープしないため、
# これらを含むレコードは行読み取り（splitlines）で分断される。生成側で置換する。
LINE_SEPARATORS = dict.fromkeys([0x2028, 0x2029, 0x0085, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E], " ")

# japanese 域は域の定義が「日本語で答える」。seed が英語プロンプトでも、言語指示は隠れた
# system ではなくユーザープロンプト末尾に付けて record に残す（thinking が存在しない
# system 指示を引用すると、学習時に再現できない特権情報になるため）。
JAPANESE_USER_SUFFIX = "\n\n日本語で、簡単な説明を添えて答えてください。"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Think in English. "
    "Answer in the language of the user's request."
)

# 生成上限（mode 別）。実際の max_tokens は max_model_len - プロンプト長 で切り詰める
MODE_MAX_TOKENS = {"reasoning": 12288, "no_reasoning": 8192}
# 切り詰め後の生成上限がこれ未満なら、そのシードは生成しない（途中切れが確実なため）
MIN_GEN_TOKENS = 2048


def sanitize(text: str) -> str:
    """行区切りとして振る舞う文字を空白に置換する。

    Args:
        text: 対象テキスト。

    Returns:
        置換済み文字列。
    """
    return (text or "").translate(LINE_SEPARATORS)


def render_prompt(tokenizer, messages: list[dict], mode: str, prefix: str) -> str:
    """Gemma 4 形式のプロンプト文字列を、生成開始ヘッダ＋prefix まで組み立てる。

    Args:
        tokenizer: Gemma 4 の tokenizer（chat_template を持つ）。
        messages: chat messages（system / user）。
        mode: "reasoning" または "no_reasoning"。
        prefix: 書き出しプレフィックス。

    Returns:
        プロンプト文字列。reasoning は `<|turn>model\\n<|channel>thought\\n{prefix}`、
        no_reasoning は `<|turn>model\\n<|channel>thought\\n<channel|>{prefix}` で終わる。
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=(mode == "reasoning"),
        tokenize=False,
    )
    if mode == "reasoning":
        text += THOUGHT_OPEN
    return text + prefix


def parse_completion(raw: str, mode: str, prefix: str = "") -> tuple[str, str, bool]:
    """vLLM の completion 本文を thinking / response に分離する。

    Args:
        raw: 生成された生テキスト（special token を含む）。
        mode: "reasoning" または "no_reasoning"。
        prefix: プロンプト側に置いた書き出し。生成本文は prefix の続きなので先頭に戻す。

    Returns:
        (thinking, response, closed)。closed は reasoning で思考の閉じタグが出たか。
        reasoning なしモードでは thinking は空。
    """
    text = prefix + raw
    for tok in END_TOKENS:
        text = text.replace(tok, "")
    if mode == "reasoning":
        if THOUGHT_CLOSE not in text:
            return text.strip(), "", False
        thinking, response = text.split(THOUGHT_CLOSE, 1)
        return thinking.strip(), response.strip(), True
    return "", text.strip(), True


def build_session(pool_size: int) -> requests.Session:
    """接続プールを並列数に合わせた HTTP セッションを作る。

    Args:
        pool_size: 接続プールの大きさ。

    Returns:
        requests.Session。
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=pool_size, max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def generate_one(
    session: requests.Session,
    base_url: str,
    model: str,
    text: str,
    max_tokens: int,
    timeout: int,
    sampling: dict,
) -> dict:
    """1 件生成して結果 dict を返す。

    Args:
        session: HTTP セッション。
        base_url: vLLM の `/v1` ベース URL。
        model: served model name。
        text: レンダリング済みプロンプト。
        max_tokens: 生成上限。
        timeout: リクエストタイムアウト秒。
        sampling: temperature / top_p / top_k。

    Returns:
        {"error", "text", "finish_reason", "completion_tokens"}。失敗時は `error` に理由が入る。
    """
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": max_tokens,
        **sampling,
        # 思考の閉じタグ <channel|> は special token。既定（True）ではデコードから消え、
        # thinking と response を分離できない。
        "skip_special_tokens": False,
    }
    try:
        resp = session.post(f"{base_url.rstrip('/')}/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {
        "error": "",
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens", 0),
    }


def load_state(path: Path, deduper: dedup.Deduplicator) -> tuple[set[str], Counter, int]:
    """既存出力から resume 用の状態を復元する。

    済みキーに加え、(domain, mode) 別の通過数を数え直し、全レコードを出力順に
    deduper へ流して重複判定の index を再構築する。中断で末尾が書きかけの行は切り捨てる。

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
        if rec.get("passed"):
            deduper.is_duplicate(domain, rec.get("prompt", ""), rec.get("response", ""))
        if rec.get("passed") and not rec.get("duplicate"):
            kept[(domain, mode)] += 1
    return seen, kept, total


def parse_ratio(spec: str, domains: list[str]) -> dict[str, float]:
    """"math=0.2,code=0.3" 形式の構成比を解釈し、合計 1 に正規化する。

    Args:
        spec: 構成比の指定。空ならドメイン等分。
        domains: 対象ドメイン。

    Returns:
        domain -> 比率。

    Raises:
        ValueError: 指定に無いドメイン・プールに無いドメインがある場合。
    """
    if not spec:
        return {d: 1 / len(domains) for d in domains}
    ratio = {k.strip(): float(v) for k, v in (item.split("=") for item in spec.split(",") if item)}
    if set(ratio) != set(domains):
        raise ValueError(f"--domain-ratio {sorted(ratio)} does not match domains {sorted(domains)}")
    total = sum(ratio.values())
    return {d: ratio[d] / total for d in domains}


def compute_quota(
    target: int, domains: list[str], sizes: dict[str, int], expected_pass: float, ratio: dict[str, float]
) -> dict:
    """(domain, mode) 別の通過数 quota を決める。

    各シードは mode ごとに 1 回だけ使う（同一シード・同一 mode の再生成は近似重複になりやすい）。
    したがって (domain, mode) の上限は `プール件数 × 想定通過率`。構成比どおりに割り当て、
    上限で頭打ちになった不足分は headroom のある domain へ構成比に比例して再配分する。

    Args:
        target: 通過数の総目標。
        domains: 対象ドメイン。
        sizes: ドメイン別プール件数。
        expected_pass: 想定通過率。
        ratio: domain -> 構成比（合計 1）。

    Returns:
        (domain, mode) -> quota。
    """
    quota: dict = {}
    for mode in MODES:
        want = target // 2 if mode == "reasoning" else target - target // 2
        cap = {d: int(sizes[d] * expected_pass) for d in domains}
        q = {d: min(int(want * ratio[d]), cap[d]) for d in domains}
        deficit = want - sum(q.values())
        while deficit > 0:
            room = [d for d in domains if q[d] < cap[d]]
            if not room:
                break
            weight = sum(ratio[d] for d in room)
            for d in room:
                q[d] = min(cap[d], q[d] + max(int(deficit * ratio[d] / weight), 1))
            deficit = want - sum(q.values())
        for d in domains:
            quota[(d, mode)] = q[d]
        if deficit > 0:
            print(f"[quota] {mode}: unreachable by {deficit} at expected_pass={expected_pass}")
    return quota


class Scheduler:
    """(domain, mode) ごとのシード供給と進捗管理。

    quota に対して「通過数 + 実行中 × 想定通過率」が最も遅れている (domain, mode) から
    次のシードを出す。シードを使い切った (domain, mode) は除外し、mode 合計が目標に
    達するまでは他の domain が quota を超えて補う。

    Attributes:
        quota: (domain, mode) -> 通過数 quota。
        kept: (domain, mode) -> 通過数。
        inflight: (domain, mode) -> 実行中件数。
    """

    def __init__(self, pool, domains, quota, kept, seen, rng, expected_pass, mode_target):
        """初期化する。

        Args:
            pool: domain -> [(prompt, source)]。
            domains: 対象ドメイン。
            quota: (domain, mode) -> quota。
            kept: (domain, mode) -> 既存通過数（resume）。
            seen: 済みキー（resume）。
            rng: 乱数生成器。
            expected_pass: 想定通過率。
            mode_target: mode -> 通過数目標。
        """
        self.pool = pool
        self.quota = quota
        self.kept = kept
        self.seen = seen
        self.expected_pass = expected_pass
        self.mode_target = mode_target
        self.inflight: Counter = Counter()
        self.exhausted: set = set()
        # (domain, mode) ごとに独立した順序。mode 間でシードを奪い合わない
        self._order = {}
        self._cursor = {}
        for d in domains:
            for m in MODES:
                order = list(range(len(pool[d])))
                random.Random(f"{rng.random()}").shuffle(order)
                self._order[(d, m)] = order
                self._cursor[(d, m)] = 0
        self.keys = [(d, m) for d in domains for m in MODES]

    def mode_done(self, mode: str) -> bool:
        """mode の通過数が目標に届いたか（実行中の見込みを含む）。"""
        kept = sum(c for (_, m), c in self.kept.items() if m == mode)
        infl = sum(c for (_, m), c in self.inflight.items() if m == mode)
        return kept + infl * self.expected_pass >= self.mode_target[mode]

    def next(self):
        """次に生成する (domain, mode, seed_index) を返す。なければ None。"""
        while True:
            live = [k for k in self.keys if k not in self.exhausted and not self.mode_done(k[1])]
            if not live:
                return None

            def lag(k):
                q = max(self.quota.get(k, 0), 1)
                return (self.kept[k] + self.inflight[k] * self.expected_pass) / q

            key = min(live, key=lag)
            order, i = self._order[key], self._cursor[key]
            while i < len(order) and f"{key[0]}|{key[1]}|{order[i]}" in self.seen:
                i += 1
            if i >= len(order):
                self._cursor[key] = i
                self.exhausted.add(key)
                print(f"[exhausted] {key[0]}/{key[1]} kept {self.kept[key]}", flush=True)
                continue
            self._cursor[key] = i + 1
            idx = order[i]
            self.seen.add(f"{key[0]}|{key[1]}|{idx}")
            return key[0], key[1], idx


def run(args: argparse.Namespace) -> dict:
    """生成・フィルタ・重複除去を実行し、統計を返す。

    Args:
        args: CLI 引数。

    Returns:
        統計 dict。
    """
    rng = random.Random(args.seed)
    pool = seeds.load_seed_pool()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_urls = [u for u in args.base_urls.split(",") if u]
    workers = args.concurrency * len(base_urls)
    session = build_session(workers)
    url_cycle = itertools.cycle(base_urls)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}

    excluded = {d for d in args.exclude_domains.split(",") if d}
    domains = [d for d in sorted(pool) if d not in excluded]
    sizes = {d: len(pool[d]) for d in domains}
    ratio = parse_ratio(args.domain_ratio, domains)
    quota = compute_quota(args.target_kept, domains, sizes, args.expected_pass, ratio)
    for (d, m), q in sorted(quota.items()):
        print(f"[quota] {d}/{m}: target kept {q} (pool {sizes[d]})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduper = dedup.Deduplicator(threshold=args.dedup_threshold)
    seen, kept_dm, resumed = load_state(out_path, deduper)
    mode_target = {"reasoning": args.target_kept // 2, "no_reasoning": args.target_kept - args.target_kept // 2}
    sched = Scheduler(pool, domains, quota, kept_dm, seen, rng, args.expected_pass, mode_target)

    stats: Counter = Counter(total=resumed, kept=sum(kept_dm.values()))
    reason_counter: Counter = Counter()
    if resumed:
        print(f"[resume] {resumed} records, kept {stats['kept']}", flush=True)
    next_report = (stats["total"] // args.chunk_size + 1) * args.chunk_size
    started = time.time()
    lock = threading.Lock()

    def submit(ex):
        """シードを 1 件取り出して投入する。投入できなければ False。"""
        while True:
            item = sched.next()
            if item is None:
                return None
            domain, mode, idx = item
            prompt, source = pool[domain][idx]
            if domain == "japanese" and not seeds.is_japanese_text(prompt):
                prompt += JAPANESE_USER_SUFFIX
            prefix = seeds.sample_prefix(domain, mode, rng, prompt)
            # --system-prompt "" で system ターンを置かない（学習データに無い隠れ指示を排除する）
            messages = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []
            messages.append({"role": "user", "content": prompt})
            text = render_prompt(tokenizer, messages, mode, prefix)
            n_prompt = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            max_tokens = min(MODE_MAX_TOKENS[mode], args.max_model_len - n_prompt - 16)
            if max_tokens < MIN_GEN_TOKENS:
                stats["skipped_long_prompt"] += 1
                continue
            sched.inflight[(domain, mode)] += 1
            meta = {
                "prompt": prompt,
                "prefix": prefix,
                "mode": mode,
                "domain": domain,
                "seed_source": source,
                "seed_index": idx,
                "prompt_tokens": n_prompt,
            }
            fut = ex.submit(
                generate_one, session, next(url_cycle), args.served_model_name, text, max_tokens, args.timeout, sampling
            )
            return fut, meta

    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        pending = {}
        while True:
            while len(pending) < workers:
                item = submit(ex)
                if item is None:
                    break
                pending[item[0]] = item[1]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                meta = pending.pop(fut)
                result = fut.result()
                key = (meta["domain"], meta["mode"])
                sched.inflight[key] -= 1
                thinking, response, closed = parse_completion(result.get("text", ""), meta["mode"], meta["prefix"])
                rec = {
                    **meta,
                    "thinking": sanitize(thinking),
                    "response": sanitize(response),
                    "thinking_closed": closed,
                    "finish_reason": result.get("finish_reason", ""),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "error": result.get("error", ""),
                    "lang": filters.detect_language(meta["prompt"]),
                    "model": args.model,
                }
                verdict = filters.evaluate(rec)
                rec["passed"] = verdict.passed
                rec["reasons"] = verdict.reasons
                rec["duplicate"] = ""
                if verdict.passed:
                    rec["duplicate"] = deduper.is_duplicate(meta["domain"], rec["prompt"], rec["response"]) or ""
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["total"] += 1
                if not verdict.passed:
                    for r in verdict.reasons:
                        reason_counter[r.split(":")[0]] += 1
                elif rec["duplicate"]:
                    reason_counter[rec["duplicate"]] += 1
                else:
                    stats["kept"] += 1
                    kept_dm[key] += 1
            fh.flush()
            if stats["total"] >= next_report:
                next_report = (stats["total"] // args.chunk_size + 1) * args.chunk_size
                rate = (stats["total"] - resumed) / max(time.time() - started, 1)
                print(
                    f"[progress] {time.strftime('%Y-%m-%d %H:%M:%S')} total {stats['total']} "
                    f"kept {stats['kept']} ({stats['kept'] / max(stats['total'], 1):.0%}) "
                    f"{rate:.2f} rec/s | top reasons {dict(reason_counter.most_common(4))}",
                    flush=True,
                )

    return {
        "total": stats["total"],
        "kept": stats["kept"],
        "skipped_long_prompt": stats["skipped_long_prompt"],
        "reasons": dict(reason_counter),
        "kept_by_domain_mode": {f"{d}/{m}": c for (d, m), c in sorted(kept_dm.items())},
        "exhausted": sorted(f"{d}/{m}" for d, m in sched.exhausted),
    }


def main() -> None:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="gemma-4-26B-A4B-it 自己蒸留データ生成")
    parser.add_argument("--target-kept", type=int, default=1000, help="通過数の目標")
    parser.add_argument("--base-urls", default="http://localhost:8110/v1,http://localhost:8111/v1",
                        help="vLLM の /v1 URL（カンマ区切り。GPU ごとに 1 サーバー）")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it", help="tokenizer 読み込み用の HF モデル ID")
    parser.add_argument("--served-model-name", default="gemma-4-26b-a4b-it", help="vLLM の served model name")
    parser.add_argument("--output", default="out/pilot.jsonl")
    parser.add_argument("--concurrency", type=int, default=64, help="エンドポイントあたりの同時リクエスト数")
    parser.add_argument("--max-model-len", type=int, default=16384, help="vLLM の max-model-len と揃える")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--chunk-size", type=int, default=1000, help="進捗表示の interval")
    parser.add_argument("--dedup-threshold", type=float, default=0.80)
    parser.add_argument("--expected-pass", type=float, default=0.8, help="quota 換算用の想定通過率")
    parser.add_argument("--exclude-domains", default="")
    parser.add_argument(
        "--domain-ratio",
        default="math=0.2,japanese=0.1,english=0.1,general_knowledge=0.3,code=0.3",
        help="通過数の構成比（Curator 判断 2026-10-12）。空ならドメイン等分",
    )
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
