"""検証キャッシュ導入前に作った 4 択の自己検証結果を、成果物から事後復元する（1 回限りの移行用）.

``mcq_raw.jsonl`` にフィルタとシャッフルをかけた問題列のうち、``mcq_eval_v1.jsonl`` に残っている
問題は自己検証を通過し、残っていない問題は不合格だった（v1 はその順序を保った部分列）。
この対応から ``mcq_verify_cache.jsonl`` を作り、``--postprocess-only`` で v1 を LLM なしに再現できるようにする。

部分列として対応が取れない場合は、復元せずにエラーで止める。

使い方（eng-cpt/ 直下で実行）:
    python -m qagen.backfill_mcq_verify_cache
"""

import collections
import json
import random

from qagen.generate_qa import OUT_DIR, SHUFFLE_SEED, filter_mcq, read_raw, shuffle_choices, verify_key


def main():
    """フィルタ・シャッフル後の問題列と v1 を突き合わせ、検証キャッシュを書き出す.

    Returns:
        None

    Raises:
        RuntimeError: v1 が問題列の部分列になっていない、またはキャッシュが既に存在する場合。
    """
    cache_path = OUT_DIR / "mcq_verify_cache.jsonl"
    if cache_path.exists():
        raise RuntimeError(f"{cache_path} は既に存在します（上書きしません）")
    items, _ = read_raw(OUT_DIR / "mcq_raw.jsonl")
    kept = filter_mcq(items, collections.Counter())
    rng = random.Random(SHUFFLE_SEED)
    for it in kept:
        it["choices"], it["answer"] = shuffle_choices(it, rng)
    with open(OUT_DIR / "mcq_eval_v1.jsonl") as f:
        v1 = [json.loads(line) for line in f]

    j, records = 0, []
    for it in kept:
        passed = j < len(v1) and (it["question"], it["choices"], it["answer"]) == (
            v1[j]["question"], v1[j]["choices"], v1[j]["answer"])
        j += passed
        records.append({"key": verify_key(it), "id_hint": it["chunk_id"], "ok": passed, "backfilled": True})
    if j != len(v1):
        raise RuntimeError(f"v1 が部分列として対応しません（{j}/{len(v1)}）")
    with open(cache_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"backfilled {len(records)} entries (ok={j}, failed={len(records) - j}) -> {cache_path}")


if __name__ == "__main__":
    main()
