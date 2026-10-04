"""生成済み QA ファイルの LaTeX エスケープ破損を修復する（1 回限りの移行用）.

``generate_qa.fix_latex_escapes`` を組み込む前に生成したファイルでは、"\\frac" が
改ページ + "rac" になるなど LaTeX コマンドが壊れている。各 JSONL の全文字列に同じ復元処理をかけ、
修復前後の破損行数を表示する。処理は冪等（修復済みのファイルにかけても変化しない）。

使い方（eng-cpt/ 直下で実行）:
    python -m qagen.repair_escapes
"""

import json

from qagen.generate_qa import LATEX_CONTROL_RESTORE, OUT_DIR, fix_latex_escapes


TARGETS = ["sft_raw.jsonl", "sft.jsonl", "mcq_raw.jsonl", "mcq_eval_v1.jsonl", "mcq_eval.jsonl"]


def _strings(value):
    """dict / list を再帰的にたどり、含まれる文字列をすべて返す.

    Args:
        value (object): JSON 由来の値。

    Yields:
        str: 含まれる文字列。
    """
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def count_broken(rows):
    """制御文字（改ページ・バックスペース・タブ・復帰）を含む行の数を数える.

    Args:
        rows (list[dict]): JSONL の行。

    Returns:
        int: 壊れた行の数。
    """
    return sum(any(ch in s for s in _strings(r) for ch in LATEX_CONTROL_RESTORE) for r in rows)


def main():
    """対象ファイルを修復して上書きし、前後の破損行数を表示する.

    Returns:
        None
    """
    assert set(LATEX_CONTROL_RESTORE) == {"\x0c", "\x08", "\t", "\r"}
    for name in TARGETS:
        path = OUT_DIR / name
        if not path.exists():
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        before = count_broken(rows)
        fixed = [fix_latex_escapes(r) for r in rows]
        with open(path, "w") as f:
            for r in fixed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: rows={len(rows)} broken_before={before} broken_after={count_broken(fixed)}")


if __name__ == "__main__":
    main()
