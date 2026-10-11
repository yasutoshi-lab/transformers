#!/usr/bin/env python
# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute generation-quality metrics from gen_eval_ranunculus.py outputs.

Metrics per (model, lang, mode), computed on token ids of the continuation:

- ``len``: mean continuation length in tokens (stops at EOS).
- ``early_eos``: share of continuations that emitted EOS before ``gen_len``.
- ``rep4``: mean within-sample repetition, 1 - unique 4-grams / total 4-grams.
- ``loop``: share of samples with rep4 > 0.5 (degenerate looping).
- ``distinct1`` / ``distinct2``: corpus-level unique n-grams / total n-grams.
- ``lang_ok``: share of letters in the target script (ja: kana/kanji, en: ASCII
  letters), averaged over samples.

Usage:

    python gen_metrics_ranunculus.py before=<dir> after=<dir> > metrics.md
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict


JA = re.compile(r"[぀-ヿ㐀-鿿]")
EN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]")


def ngrams(ids: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def rep(ids: list[int], n: int = 4) -> float:
    g = ngrams(ids, n)
    return 1 - len(set(g)) / len(g) if g else 0.0


def lang_share(text: str, lang: str) -> float:
    letters = LETTER.findall(text)
    if not letters:
        return 0.0
    pat = JA if lang == "ja" else EN
    return sum(1 for c in letters if pat.match(c)) / len(letters)


def summarize(path: str, gen_len: int = 256) -> dict:
    groups = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        groups[(r["lang"], r["mode"])].append(r)
    out = {}
    for key, rs in groups.items():
        lang = key[0]
        reps = [rep(r["ids"]) for r in rs]
        g1 = [g for r in rs for g in ngrams(r["ids"], 1)]
        g2 = [g for r in rs for g in ngrams(r["ids"], 2)]
        out[key] = {
            "n": len(rs),
            "len": sum(len(r["ids"]) for r in rs) / len(rs),
            "early_eos": sum(1 for r in rs if r["eos"] and len(r["ids"]) < gen_len) / len(rs),
            "rep4": sum(reps) / len(rs),
            "loop": sum(1 for x in reps if x > 0.5) / len(rs),
            "distinct1": len(set(g1)) / max(len(g1), 1),
            "distinct2": len(set(g2)) / max(len(g2), 1),
            "lang_ok": sum(lang_share(r["text"], lang) for r in rs) / len(rs),
        }
    return out


def main() -> None:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    res = {name: summarize(f"{d}/generations.jsonl") for name, d in runs.items()}
    dump = {name: {f"{k[0]}/{k[1]}": v for k, v in r.items()} for name, r in res.items()}
    for name, d in runs.items():
        rows = [json.loads(l) for l in open(f"{d}/short_prompts.jsonl", encoding="utf-8")]
        dump[name]["short_prompts"] = rows
    with open("gen_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    cols = ["n", "len", "early_eos", "rep4", "loop", "distinct1", "distinct2", "lang_ok"]
    print("| model | lang | mode | " + " | ".join(cols) + " |")
    print("|" + "---|" * (3 + len(cols)))
    for lang in ("ja", "en"):
        for mode in ("reference", "greedy", "sample"):
            for name, r in res.items():
                if mode == "reference" and name != next(iter(res)):
                    continue
                m = r.get((lang, mode))
                if m is None:
                    continue
                label = "（実テキスト）" if mode == "reference" else name
                vals = [f"{m[c]:.0f}" if c == "n" else f"{m[c]:.3f}" for c in cols]
                print(f"| {label} | {lang} | {mode} | " + " | ".join(vals) + " |")

    print()
    print("| model | lang | leading_eos | mean P(EOS) first | EOS rank 1 の割合 | greedy 平均長 | greedy が 5 tok 以内で終了 |")
    print("|---|---|---|---|---|---|---|")
    for name, d in runs.items():
        rows = [json.loads(l) for l in open(f"{d}/short_prompts.jsonl", encoding="utf-8")]
        for lang in ("ja", "en"):
            for lead in (False, True):
                rs = [r for r in rows if r["lang"] == lang and r["leading_eos"] == lead]
                print(f"| {name} | {lang} | {lead} | {sum(r['p_eos_first'] for r in rs) / len(rs):.4f} | "
                      f"{sum(1 for r in rs if r['eos_rank_first'] == 1) / len(rs):.2f} | "
                      f"{sum(r['greedy_len'] for r in rs) / len(rs):.1f} | "
                      f"{sum(1 for r in rs if r['greedy_len'] <= 5 and r['greedy_eos']) / len(rs):.2f} |")


if __name__ == "__main__":
    main()
