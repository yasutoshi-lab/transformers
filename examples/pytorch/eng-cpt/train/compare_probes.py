"""生成検証（probe_generations）の結果を統合し、元データとの機械的な比較指標を付ける.

Merge probe generation results and attach objective, reproducible comparison metrics.

目的: 入力・元データ（根拠文）・4 条件の出力を並べ、出力が条件によってどう変化したかを、
主観的な判定を挟まずに確認できるようにする。指標はすべて表層的な一致で、正誤の判定ではない。

指標（出力ごと）:
    key_terms        根拠文・正答の要点に含まれ、入力（質問）には含まれない専門用語（漢字・カタカナの 2 字以上の
                     連続、英字・英字略語）のうち、出力に現れた語 / 現れなかった語。質問のオウム返しでは
                     上がらない「元データ固有の情報」を出力が含むかを見る
    key_term_recall  現れた語の割合
    numbers          出力中の数値のうち、根拠文に含まれるもの（supported）/ 含まれないもの（unsupported）
    char_bigram_f1   根拠文と出力の文字 2-gram の F1（表現の近さ）
    sim_to_base      Base の出力との文字 2-gram Jaccard 類似度（学習による出力の変化量。Base 自身は 1.0）
    n_chars / n_output_tokens / finish_reason
    repetition       出力中の文字 10-gram のうち、2 回以上現れるものの割合（同じ表現の繰り返し）

出力:
    <out>.jsonl      1 行 = 1 サンプル × 1 条件（元の記録 + metrics）
    <out>.csv        サンプル × 条件の指標一覧（表計算ソフトで比較する用）
    <out>.md         サンプルごとに 入力 / 元データ / 4 条件の出力と指標 を並べた一覧

使い方（eng-cpt/ 直下）:
    python -m train.compare_probes artifacts/probe/probe_generations.jsonl artifacts/probe/probe_train.jsonl \\
        artifacts/probe/probe_more.jsonl --out artifacts/probe/probe_compare
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


CONDITION_ORDER = ["Base", "Base+CPT", "Base+SFT", "Base+CPT+SFT"]
RE_TERM = re.compile(r"[一-龥々]{2,}|[ァ-ヴー・]{2,}|[A-Za-z][A-Za-z0-9]*")
RE_NUMBER = re.compile(r"\d+(?:\.\d+)?")
STOP_TERMS = {"場合", "方法", "ため", "こと", "もの", "など", "以上", "以下", "程度", "通常"}


def normalize(text):
    """比較用に NFKC 正規化し、空白を除く.

    Args:
        text (str): 対象テキスト。

    Returns:
        str: 正規化済みテキスト。
    """
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", text))


def key_terms(evidence, answer, question):
    """根拠文・正答から、質問に含まれない専門用語を抽出する（出現順・重複なし）.

    Args:
        evidence (str): 根拠文。
        answer (str): 正答の要点。
        question (str): 入力の質問（ここに含まれる語は除外する）。

    Returns:
        list[str]: 用語のリスト。
    """
    q = normalize(question)
    seen = []
    for t in RE_TERM.findall(normalize(answer) + "。" + normalize(evidence)):
        if t not in seen and t not in STOP_TERMS and t not in q:
            seen.append(t)
    return seen


def bigrams(text):
    """文字 2-gram の多重集合を返す.

    Args:
        text (str): 対象テキスト。

    Returns:
        collections.Counter: 2-gram の出現数。
    """
    s = normalize(text)
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def bigram_f1(ref, hyp):
    """文字 2-gram の F1 を返す.

    Args:
        ref (str): 参照（根拠文）。
        hyp (str): 出力。

    Returns:
        float: F1（どちらかが空なら 0.0）。
    """
    r, h = bigrams(ref), bigrams(hyp)
    overlap = sum((r & h).values())
    if not overlap:
        return 0.0
    p, rc = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * p * rc / (p + rc)


def bigram_jaccard(a, b):
    """文字 2-gram（集合）の Jaccard 類似度を返す.

    Args:
        a (str): テキスト A。
        b (str): テキスト B。

    Returns:
        float: Jaccard 係数（両方空なら 1.0）。
    """
    x, y = set(bigrams(a)), set(bigrams(b))
    return 1.0 if not x and not y else len(x & y) / len(x | y)


def repetition_ratio(text, n=10):
    """文字 n-gram のうち 2 回以上現れるものの割合を返す.

    Args:
        text (str): 対象テキスト。
        n (int): n-gram の長さ。

    Returns:
        float: 繰り返し率（n 文字未満なら 0.0）。
    """
    s = normalize(text)
    grams = Counter(s[i : i + n] for i in range(len(s) - n + 1))
    total = sum(grams.values())
    return sum(c for c in grams.values() if c > 1) / total if total else 0.0


def metrics(record, base_output):
    """1 出力の比較指標を計算する.

    Args:
        record (dict): probe_generations の 1 行。
        base_output (str): 同じサンプルの Base の出力。

    Returns:
        dict: 指標。
    """
    evidence, output = record["reference"]["evidence"], record["output"]
    terms = key_terms(evidence, record["reference"]["correct_choice"], record["input"]["question"])
    out_n = normalize(output)
    hit = [t for t in terms if t in out_n]
    ev_numbers = set(RE_NUMBER.findall(normalize(evidence) + normalize(record["reference"]["correct_choice"])))
    nums = RE_NUMBER.findall(out_n)
    return {
        "key_terms_found": hit,
        "key_terms_missing": [t for t in terms if t not in hit],
        "key_term_recall": round(len(hit) / len(terms), 3) if terms else None,
        "numbers_supported": sorted({n for n in nums if n in ev_numbers}),
        "numbers_unsupported": sorted({n for n in nums if n not in ev_numbers}),
        "char_bigram_f1": round(bigram_f1(evidence, output), 3),
        "sim_to_base": round(bigram_jaccard(base_output, output), 3),
        "n_chars": len(output),
        "n_output_tokens": record["n_output_tokens"],
        "finish_reason": record["finish_reason"],
        "repetition": round(repetition_ratio(output), 3),
    }


def main():
    """複数の probe 結果を統合して指標を付け、JSONL・CSV・Markdown を書き出す.

    Returns:
        None

    Raises:
        ValueError: 同じ sample_id が複数のファイルにまたがって重複する場合。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        for line in open(path):
            r = json.loads(line)
            r["probe_set"] = path.stem
            # 出典の範囲（qa_eval / train）を前に付けて、別セットの同名 ID を区別する
            r["sample_id"] = f"{r['source']['split']}-{r['sample_id']}"
            rows.append(r)
    by_sample = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], {})[r["condition"]] = r
    merged = []
    for sid, conds in by_sample.items():
        base = conds["Base"]["output"]
        for c in CONDITION_ORDER:
            r = conds[c]
            merged.append({**r, "metrics": metrics(r, base)})

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["probe_set", "sample_id", "domain", "aspect", "condition", "key_term_recall", "char_bigram_f1",
            "sim_to_base", "numbers_supported", "numbers_unsupported", "n_chars", "n_output_tokens", "finish_reason",
            "repetition", "key_terms_found", "key_terms_missing"]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in merged:
            m = r["metrics"]
            w.writerow([r["probe_set"], r["sample_id"], r["domain"], r["aspect"], r["condition"]]
                       + [" / ".join(m[k]) if isinstance(m[k], list) else m[k] for k in cols[5:]])

    lines = ["# 生成結果の比較（入力・元データ・4 条件の出力）", "",
             "指標は表層的な一致の機械計算で、正誤の判定ではない。用語 = 根拠文の専門用語のうち出力に現れた割合、"
             "F1 = 根拠文との文字 2-gram F1、Base類似 = Base の出力との文字 2-gram Jaccard、"
             "根拠外の数値 = 根拠文・正答にない数値。", ""]
    for sid, conds in by_sample.items():
        r0 = conds["Base"]
        src = r0["source"]
        lines += [f"## {sid}（{r0['domain']}・{r0['aspect']}）", "",
                  f"- 出典: {src['book']} p.{src['pages'][0]}-{src['pages'][1]}（{src['split']}。"
                  f"CPT データに含む: {src['in_cpt_data']}。SFT への包含: {src.get('sft_coverage', '評価用4択由来（SFT 生成元の範囲外）')}）",
                  f"- **入力（質問）**: {r0['input']['question']}",
                  f"- **元データ（根拠文）**: {r0['reference']['evidence']}",
                  f"- **正答の要点**: {r0['reference']['correct_choice']}", "",
                  "| 条件 | 用語 | F1 | Base類似 | 根拠外の数値 | 文字数 | 停止 | 反復 | 出力 |", "|---|---|---|---|---|---|---|---|---|"]
        for c in CONDITION_ORDER:
            r = next(x for x in merged if x["sample_id"] == sid and x["condition"] == c)
            m = r["metrics"]
            text = r["output"].replace("\n", " ").replace("|", "／")
            text = text if len(text) <= 220 else text[:220] + "…（以下略）"
            lines.append(f"| {c} | {m['key_term_recall']} | {m['char_bigram_f1']} | {m['sim_to_base']} | "
                         f"{', '.join(m['numbers_unsupported']) or '—'} | {m['n_chars']} | {m['finish_reason']} | "
                         f"{m['repetition']} | {text} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.with_suffix('.jsonl')} / .csv / .md ({len(merged)} records, {len(by_sample)} samples)")


if __name__ == "__main__":
    main()
