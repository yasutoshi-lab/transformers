"""評価結果と学習ログを集計し、レポート用の表（Markdown）と JSON を出力する.

Aggregate eval results and training logs into report tables.

出力内容:
    1. 条件別の指標（holdout perplexity / 自作 4 択 acc・acc_norm / JMMLU 全体・グループ別）
       4 択と JMMLU には問題単位のブートストラップによる 95% 信頼区間を付ける
    2. Base との差（同じ問題での正誤を対にしたブートストラップ。区間が 0 を跨がなければ差ありとみなす）
    3. CPT の量の曲線（train 使用率 × エポック → 学習トークン数・GPU 時間・各指標）
    4. コスト（GPU 時間 × ``--gpu-hour-price``。単価を渡さない場合は GPU 時間のみ）

使い方（eng-cpt/ 直下で実行）:
    python -m train.analyze_results --gpu-hour-price 2.5 --currency USD
"""

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "artifacts" / "eval"
RUNS_DIR = ROOT / "artifacts" / "runs"
QA_DIR = ROOT / "artifacts" / "qa"
N_BOOT = 2000
BOOT_SEED = 0

# 主実験（2×2）の条件名 → 評価 JSON 名
MAIN_CONDITIONS = {
    "Base": "base",
    "Base+CPT": "cpt-f100-ep3",
    "Base+SFT": "sft-base-ep2",
    "Base+CPT+SFT": "sft-cpt-f100-ep2",
}


def load_eval(name):
    """評価 JSON を読み込む（無ければ ``None``）.

    Args:
        name (str): 評価名（``artifacts/eval/<name>.json``）。

    Returns:
        dict | None: 評価結果。
    """
    path = EVAL_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def correctness(result, metric):
    """問題単位の正誤ベクトルを返す.

    Args:
        result (dict): 評価結果。
        metric (str): ``mcq`` / ``mcq_norm`` / ``jmmlu``。

    Returns:
        list[int]: 問題順に 1（正解）/ 0（不正解）。
    """
    if metric == "jmmlu":
        return [int(p["pred"] == p["answer"]) for p in result["jmmlu"]["predictions"]]
    key = "pred_norm" if metric == "mcq_norm" else "pred"
    return [int(p[key] == p["answer"]) for p in result["mcq"]["predictions"]]


def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    """平均の 95% ブートストラップ信頼区間を返す.

    Args:
        values (list[float]): 問題単位の値（正誤や正誤の差）。
        n_boot (int): リサンプリング回数。
        seed (int): 乱数シード。

    Returns:
        tuple[float, float, float]: ``(平均, 下限, 上限)``。
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def paired_delta(a, b):
    """同じ問題での正誤差（b − a）の平均と 95% 信頼区間を返す.

    Args:
        a (list[int]): 比較元の正誤ベクトル。
        b (list[int]): 比較先の正誤ベクトル（同じ問題順）。

    Returns:
        tuple[float, float, float]: ``(差の平均, 下限, 上限)``。

    Raises:
        ValueError: 問題数が一致しない場合。
    """
    if len(a) != len(b):
        raise ValueError("問題数が一致しません")
    return bootstrap_ci([y - x for x, y in zip(a, b)])


def pct(x):
    """割合を百分率の文字列にする.

    Args:
        x (float): 0〜1 の割合。

    Returns:
        str: 小数 1 桁の百分率。
    """
    return f"{x * 100:.1f}"


def fmt_ci(mean, lo, hi, signed=False):
    """平均と信頼区間を ``41.1 [37.6, 44.6]`` の形に整える.

    Args:
        mean (float): 平均。
        lo (float): 下限。
        hi (float): 上限。
        signed (bool): 符号を付けるか（差の表示用）。

    Returns:
        str: 整形済み文字列。
    """
    s = "+" if signed and mean >= 0 else ""
    return f"{s}{pct(mean)} [{pct(lo)}, {pct(hi)}]"


def run_summary(run):
    """学習 run の run_summary.json を読む（無ければ ``None``）.

    Args:
        run (str): run 名（``artifacts/runs/<run>``）。

    Returns:
        dict | None: 学習サマリ。
    """
    path = RUNS_DIR / run / "run_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def main_table(lines, out):
    """主実験（2×2）の指標表と Base との差の表を作る.

    Args:
        lines (list[str]): Markdown 行の追加先。
        out (dict): JSON 出力の追加先。

    Returns:
        None
    """
    base = load_eval("base")
    lines += ["## 主実験（2×2）", "",
              "| 条件 | holdout PPL | 自作4択 acc [95%CI] | acc_norm | JMMLU [95%CI] | JMMLU 工学系 | 数学系 | その他 |",
              "|---|---|---|---|---|---|---|---|"]
    rows = {}
    for label, name in MAIN_CONDITIONS.items():
        r = load_eval(name)
        if r is None:
            lines.append(f"| {label} | （未評価: {name}） | | | | | | |")
            continue
        acc = bootstrap_ci(correctness(r, "mcq"))
        jm = bootstrap_ci(correctness(r, "jmmlu"))
        g = r["jmmlu"]["by_group"]
        lines.append(f"| {label} | {r['holdout_ppl']['ppl']:.3f} | {fmt_ci(*acc)} | {pct(r['mcq']['acc_norm'])} | "
                     f"{fmt_ci(*jm)} | {pct(g['engineering']['acc'])} | {pct(g['math']['acc'])} | {pct(g['other']['acc'])} |")
        rows[label] = {"eval": name, "ppl": r["holdout_ppl"]["ppl"], "mcq_acc": acc, "mcq_acc_norm": r["mcq"]["acc_norm"],
                       "jmmlu_acc": jm, "jmmlu_by_group": {k: v["acc"] for k, v in g.items()}}
    out["main"] = rows

    lines += ["", "### Base との差（同一問題の対応ありブートストラップ、単位: ポイント）", "",
              "| 条件 | 自作4択 acc の差 [95%CI] | JMMLU の差 [95%CI] |", "|---|---|---|"]
    deltas = {}
    for label, name in list(MAIN_CONDITIONS.items())[1:]:
        r = load_eval(name)
        if r is None or base is None:
            continue
        d_mcq = paired_delta(correctness(base, "mcq"), correctness(r, "mcq"))
        d_jm = paired_delta(correctness(base, "jmmlu"), correctness(r, "jmmlu"))
        lines.append(f"| {label} | {fmt_ci(*d_mcq, signed=True)} | {fmt_ci(*d_jm, signed=True)} |")
        deltas[label] = {"mcq": d_mcq, "jmmlu": d_jm}
    sft_b, sft_c = load_eval(MAIN_CONDITIONS["Base+SFT"]), load_eval(MAIN_CONDITIONS["Base+CPT+SFT"])
    if sft_b and sft_c:
        d = paired_delta(correctness(sft_b, "mcq"), correctness(sft_c, "mcq"))
        lines.append(f"| （参考）Base+CPT+SFT − Base+SFT | {fmt_ci(*d, signed=True)} | "
                     f"{fmt_ci(*paired_delta(correctness(sft_b, 'jmmlu'), correctness(sft_c, 'jmmlu')), signed=True)} |")
        deltas["CPT+SFT_vs_SFT"] = {"mcq": d}
    out["deltas_vs_base"] = deltas


def curve_table(lines, out, price, currency):
    """CPT の量の曲線（使用率 × エポック）の表を作る.

    Args:
        lines (list[str]): Markdown 行の追加先。
        out (dict): JSON 出力の追加先。
        price (float | None): GPU 1 時間あたりの単価。
        currency (str): 通貨表記。

    Returns:
        None
    """
    base = load_eval("base")
    cost_col = f" | 学習コスト（{currency}）" if price else ""
    lines += ["", "## CPT の量の曲線", "",
              "累計 GPU 時間はエポック数に比例するとして按分した値（各 run は全エポックを通しで学習）。", "",
              f"| train 使用率 | エポック | 学習トークン（累計） | GPU 時間（累計）{cost_col} | holdout PPL | 自作4択 acc | Base との差 [95%CI] | JMMLU |",
              "|---|---|---|---|---|---|---|---|" + ("---|" if price else "")]
    curve = []
    for frac in (25, 50, 100):
        run = f"cpt-f{frac}"
        s = run_summary(run)
        if s is None:
            continue
        epochs = int(s["args"]["epochs"])
        for ep in range(1, epochs + 1):
            r = load_eval(f"{run}-ep{ep}")
            if r is None:
                continue
            tokens = s["data"]["tokens_per_epoch"] * ep
            hours = s["gpu_hours"] * ep / epochs
            d = paired_delta(correctness(base, "mcq"), correctness(r, "mcq")) if base else (0, 0, 0)
            cost = f" | {hours * price:.2f}" if price else ""
            lines.append(f"| {frac}% | {ep} | {tokens:,} | {hours:.3f}{cost} | {r['holdout_ppl']['ppl']:.3f} | "
                         f"{pct(r['mcq']['acc'])} | {fmt_ci(*d, signed=True)} | {pct(r['jmmlu']['acc'])} |")
            curve.append({"train_fraction": frac / 100, "epoch": ep, "tokens": tokens, "gpu_hours": hours,
                          "ppl": r["holdout_ppl"]["ppl"], "mcq_acc": r["mcq"]["acc"], "mcq_delta": d,
                          "jmmlu_acc": r["jmmlu"]["acc"]})
    out["cpt_curve"] = curve


def cost_table(lines, out, price, currency):
    """学習 run と QA 生成の資源使用量・コストの表を作る.

    Args:
        lines (list[str]): Markdown 行の追加先。
        out (dict): JSON 出力の追加先。
        price (float | None): GPU 1 時間あたりの単価。
        currency (str): 通貨表記。

    Returns:
        None
    """
    lines += ["", "## 学習の資源使用量" + (f"（単価 {price} {currency}/GPU時間）" if price else ""), "",
              "| run | モード | エポック | 1 エポックのトークン | 学習時間（分） | GPU 時間 | トークン/秒 | ピークメモリ (GiB)"
              + (f" | コスト（{currency}）" if price else "") + " |",
              "|---|---|---|---|---|---|---|---|" + ("---|" if price else "")]
    runs = {}
    for d in sorted(RUNS_DIR.glob("*/run_summary.json")):
        s = json.loads(d.read_text())
        name = d.parent.name
        cost = f" | {s['gpu_hours'] * price:.2f}" if price else ""
        lines.append(f"| {name} | {s['mode']} | {s['args']['epochs']:g} | {s['data']['tokens_per_epoch']:,} | "
                     f"{s['wall_seconds'] / 60:.1f} | {s['gpu_hours']:.3f} | {s['tokens_per_second']:,.0f} | "
                     f"{s['peak_memory_gib']:.1f}{cost} |")
        runs[name] = {k: s[k] for k in ("mode", "gpu_hours", "wall_seconds", "tokens_per_second", "peak_memory_gib")}
    out["runs"] = runs

    qa = QA_DIR / "qa_stats.json"
    if qa.exists():
        st = json.loads(qa.read_text())
        lines += ["", "## QA 生成（Gemma-4-26B-A4B-NVFP4 / vLLM / GPU1）", "",
                  "| タスク | 最終件数 | 入力トークン | 出力トークン | 計測済み生成時間（秒） |", "|---|---|---|---|---|"]
        for task, v in st.items():
            secs = sum(r["generate_seconds"] for r in v.get("runs", []))
            final = v["counts"].get(f"{task}_final", "")
            lines.append(f"| {task} | {final} | {v['usage']['prompt_tokens']:,} | {v['usage']['completion_tokens']:,} | "
                         f"{secs:.0f}（記録のある実行分のみ） |")
        out["qa_generation"] = st


def main():
    """全表を作り、Markdown と JSON を書き出す.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu-hour-price", type=float, default=None, help="GPU 1 時間あたりの単価（省略時はコスト列なし）")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "report")
    args = ap.parse_args()

    lines, out = ["# eng-cpt 実験結果", ""], {"gpu_hour_price": args.gpu_hour_price, "currency": args.currency}
    main_table(lines, out)
    curve_table(lines, out, args.gpu_hour_price, args.currency)
    cost_table(lines, out, args.gpu_hour_price, args.currency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
