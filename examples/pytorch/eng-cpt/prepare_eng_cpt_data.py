"""工学系教科書 OCR JSON から軽量 CPT 用コーパスを構築する前処理パイプライン.

Build a light continual-pretraining (CPT) corpus from OCR'd engineering textbooks.

入力は Gaia の ``raw/books/<title>.json``（``[{name, page, content}]`` 形式の OCR 結果）。
各ステージで落としたページ数・文字数・理由を ``stats.json`` に記録し、
前処理の知見レポートに使える数値を残す。

Stages:
    S1 文字正規化   : NFKC・簡体字→日本字体・LaTeX 丸数字・空白整理
    S2 行クリーニング: 柱（ランニングヘッダ）・ページ番号行・目次リーダ行の除去
    S3 ページ除去   : 奥付・クレジット・まえがき・目次・索引・参考文献・図主体・短すぎ
    S4 重複除去     : 完全一致ハッシュ + MinHash 近似重複
    S5 分割         : 書籍ごとに連続ブロックで train / qa_eval / holdout_ppl に分ける
    S6 文書化       : 連続ページを結合し、ページ跨ぎの文を接合して JSONL 出力

注意: 出力は著作物由来のため ``artifacts/`` 配下（git 追跡外）にのみ書き出す。
"""

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import opencc
from datasketch import MinHash, MinHashLSH


BOOKS = {
    "mechanical": [
        "機械加工学の基礎",
        "材料加工プロセス",
        "材料基礎工学",
        "溶接I",
        "板金工作法及びプレス加工法",
        "流体力学",
        "はじめての表面処理技術",
        "これでわかるプラスチック技術",
        "生産技術の実践手法がよく分かる本",
        "絵とき機械図面の読み方かき方",
    ],
    "electrical": [
        "電気基礎I",
        "電気基礎II",
        "電子回路",
        "通信技術",
        "電子情報技術",
        "電子製図",
        "電気電子実習I",
        "工業技術基礎",
    ],
    "aeronautical": [
        "プロが教える飛行機のメカニズム",
        "飛行機の仕組みパーフェクト辞典",
    ],
}

# OpenCC t2jp が日本の常用字体にしない文字の手動補正
CHAR_OVERRIDES = {"豔": "艶", "韌": "靭"}

RE_ISBN = re.compile(r"ISBN|C\d{4}\s*¥")
RE_COLOPHON = re.compile(r"定価|発行所|発行者|印刷所|検定済|無断で複写|無断転載|著作権法")
RE_CREDITS = re.compile(r"作成委員|監修委員|執筆者|著者略歴|著者紹介|編集委員|執筆協力|執筆者紹介")
RE_PREFACE = re.compile(r"^.{0,12}(まえがき|はしがき|序文|あとがき|刊行にあたって|監修の言葉)", re.MULTILINE)
RE_TOC_HEAD = re.compile(r"目\s*次|もくじ|CONTENTS", re.IGNORECASE)
RE_TOC_LINE = re.compile(r"(\.{3,}|…{2,}|·{3,}|・{3,}|—{2,}|-{3,})\s*\d{1,3}\s*$")
RE_INDEX_HEAD = re.compile(r"^\s*(索\s*引|さくいん|INDEX)", re.IGNORECASE)
RE_INDEX_LINE = re.compile(r"^[^|$]{1,30}\s[\d,\s—-]+$|^【.】$")
RE_REFS = re.compile(r"^\s*(引用文献|参考文献|参考図書|〈引用文献〉|〈参考文献〉)")
RE_PAGE_NUM = re.compile(r"^\s*[-—]?\s*\d{1,3}\s*[-—]?\s*$")
RE_TEXTCIRCLED = re.compile(r"\$?\\textcircled\{(\d{1,2})\}\$?")
RE_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", re.DOTALL)
RE_TABLE_ROW = re.compile(r"^.*\|.*$", re.MULTILINE)
RE_MATH_ENV = re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.DOTALL)
RE_HTML = re.compile(r"<[^>]+>")
RE_JA = re.compile(r"[぀-ヿ一-鿿]")
RE_HIRA = re.compile(r"[぀-ゟ]")
RE_CONT_HEAD = re.compile(r"^[\u3041-\u3093、。,.]")
SENT_END = tuple("。．.！？!?）」』】")


def to_jp_glyphs(text, converter, stats):
    """JIS X 0213 に無い漢字だけを簡体字とみなし日本の字体へ変換する.

    Args:
        text (str): NFKC 済みテキスト。
        converter (Callable[[str], str]): 1 文字を変換する関数（s2t→t2jp）。
        stats (collections.Counter): 変換件数・未変換件数を加算するカウンタ。

    Returns:
        str: 変換後のテキスト。正しい日本語の文字は一切変更しない。
    """
    out = []
    for ch in text:
        if "一" <= ch <= "鿿":
            try:
                ch.encode("euc_jis_2004")
            except UnicodeEncodeError:
                new = CHAR_OVERRIDES.get(converter(ch), converter(ch))
                try:
                    new.encode("euc_jis_2004")
                    stats["glyph_converted"] += 1
                    ch = new
                except UnicodeEncodeError:
                    stats["glyph_unresolved"] += 1
        out.append(ch)
    return "".join(out)


def normalize_page(text, converter, stats):
    """S1: ページ本文の文字レベル正規化を行う.

    Args:
        text (str): OCR の生テキスト。
        converter (Callable[[str], str]): 簡体字→日本字体の 1 文字変換関数。
        stats (collections.Counter): 変換件数の集計先。

    Returns:
        str: 正規化済みテキスト。
    """
    text = RE_TEXTCIRCLED.sub(lambda m: chr(0x2460 + int(m.group(1)) - 1), text)
    text = unicodedata.normalize("NFKC", text)
    text = to_jp_glyphs(text, converter, stats)
    text = re.sub(r"[ \t　]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def find_running_heads(pages):
    """S2 の前段: 書籍内で繰り返し現れるページ先頭/末尾行（柱）を検出する.

    Args:
        pages (list[str]): 正規化済みページ本文のリスト。

    Returns:
        set[str]: 柱とみなす行（数字を除いた正規形）の集合。
    """
    counter = collections.Counter()
    for text in pages:
        lines = [ln for ln in text.split("\n") if ln]
        for ln in lines[:2] + lines[-2:]:
            key = re.sub(r"\d+", "#", ln)
            if len(ln) <= 30:
                counter[key] += 1
    return {k for k, v in counter.items() if v >= 4}


def clean_lines(text, heads, stats):
    """S2: 柱・ページ番号行・目次リーダ行をページ本文から除去する.

    Args:
        text (str): 正規化済みページ本文。
        heads (set[str]): ``find_running_heads`` が返す柱の正規形集合。
        stats (collections.Counter): 除去行数の集計先。

    Returns:
        str: 行クリーニング後の本文。
    """
    lines = text.split("\n")
    kept = []
    nonblank_idx = [i for i, ln in enumerate(lines) if ln]
    edge = set(nonblank_idx[:2] + nonblank_idx[-2:])
    for i, ln in enumerate(lines):
        if RE_PAGE_NUM.match(ln) and ln:
            stats["line_page_number"] += 1
            continue
        if i in edge and re.sub(r"\d+", "#", ln) in heads:
            stats["line_running_head"] += 1
            continue
        kept.append(ln)
    joined = []
    for ln in kept:
        # 段組み OCR で文中改行された行（次行がひらがな・句読点始まり）を接合する
        if joined and joined[-1] and ln and len(joined[-1]) >= 10 and "|" not in ln \
                and not joined[-1].endswith(SENT_END) and RE_CONT_HEAD.match(ln):
            joined[-1] += ln
            stats["line_joined"] += 1
        else:
            joined.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(joined)).strip()


def prose_only(text):
    """数式・数式環境・表・HTML タグ・コード/英文行を除いた地の文だけを返す.

    Args:
        text (str): ページ本文。

    Returns:
        str: 言語比率の判定に使う地の文。
    """
    text = RE_MATH_ENV.sub("", RE_MATH.sub("", text))
    text = RE_HTML.sub("", RE_TABLE_ROW.sub("", text))
    keep = []
    for ln in text.split("\n"):
        body = re.sub(r"\s", "", ln)
        if body and (sum(ch.isascii() for ch in body) / len(body) > 0.8 or "\\" in ln):
            continue
        keep.append(ln)
    return "\n".join(keep)


def salvage_sentences(text):
    """図主体ページから、図ラベルを捨てて解説文（句点で終わる行）だけを救出する.

    Args:
        text (str): 図主体と判定されたページ本文。

    Returns:
        str: 救出した解説文。該当がなければ空文字列。
    """
    return "\n".join(ln for ln in text.split("\n") if len(ln) >= 20 and ln.endswith(SENT_END))


def classify_drop(text):
    """S3: ページを除去すべきか判定し、理由を返す.

    Args:
        text (str): 行クリーニング後のページ本文。

    Returns:
        str | None: 除去理由（ルール名）。残す場合は ``None``。
    """
    lines = [ln for ln in text.split("\n") if ln]
    head = "\n".join(lines[:3])
    n = len(lines)
    if len(text) < 50:
        return "empty"
    if RE_ISBN.search(text) or len(set(RE_COLOPHON.findall(text))) >= 2:
        return "colophon"
    if RE_CREDITS.search(text):
        return "credits"
    if RE_PREFACE.search(head):
        return "preface"
    plain = [ln for ln in lines if "|" not in ln]
    if RE_INDEX_HEAD.search(head) or (len(plain) >= 10 and sum(bool(RE_INDEX_LINE.match(ln)) for ln in plain) / len(plain) >= 0.5):
        return "index"
    if RE_TOC_HEAD.search(head) or (n >= 5 and sum(bool(RE_TOC_LINE.search(ln)) for ln in lines) / n >= 0.3):
        return "toc"
    if RE_REFS.search(head):
        return "references"
    chars = re.sub(r"\s", "", text)
    if len(chars) < 100:
        return "too_short"
    # 数式・表は工学知識の担い手なので、言語比率は「数式・表を除いた地の文」で測る
    prose = re.sub(r"\s", "", prose_only(text))
    if n >= 8 and sum(len(ln) for ln in lines if len(ln) > 20) < 150 and len(prose) < 300:
        return "figure_dominant"
    if len(prose) >= 50 and len(RE_JA.findall(prose)) / len(prose) < 0.3:
        return "low_japanese"
    if len(prose) >= 50 and len(RE_HIRA.findall(prose)) / len(prose) < 0.05:
        return "low_hiragana"
    return None


def minhash(text, num_perm=128, n=5):
    """文字 n-gram の MinHash を計算する.

    Args:
        text (str): 対象テキスト。
        num_perm (int): 置換数。
        n (int): 文字 n-gram の n。

    Returns:
        MinHash: 計算済み MinHash。
    """
    m = MinHash(num_perm=num_perm)
    s = re.sub(r"\s", "", text)
    for i in range(max(1, len(s) - n + 1)):
        m.update(s[i : i + n].encode())
    return m


def join_pages(texts):
    """S6: 連続ページを 1 文書に結合し、ページ跨ぎで途切れた文を接合する.

    Args:
        texts (list[str]): 連続するページ本文のリスト。

    Returns:
        str: 結合後の文書テキスト。
    """
    out = texts[0]
    for t in texts[1:]:
        sep = "\n\n" if out.rstrip().endswith(SENT_END) else ""
        out = out.rstrip() + sep + t.lstrip()
    return out


def split_of(idx, n, qa_range, ppl_range):
    """S5: 書籍内の位置からページの分割先を決める.

    Args:
        idx (int): 書籍内（除去後）のページ順位。
        n (int): 書籍内（除去後）のページ数。
        qa_range (tuple[float, float]): qa_eval ブロックの相対位置範囲。
        ppl_range (tuple[float, float]): holdout_ppl ブロックの相対位置範囲。

    Returns:
        str: ``train`` / ``qa_eval`` / ``holdout_ppl`` のいずれか。
    """
    r = idx / n
    if ppl_range[0] <= r < ppl_range[1]:
        return "holdout_ppl"
    if qa_range[0] <= r < qa_range[1]:
        return "qa_eval"
    return "train"


def main():
    """コマンドライン引数を解釈してパイプライン全体を実行する.

    Raises:
        FileNotFoundError: 入力 JSON が存在しない場合。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default="/home/ubuntu/Documents/project/Gaia-Personal/raw/books")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "artifacts" / "data"))
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s2t, t2jp = opencc.OpenCC("s2t"), opencc.OpenCC("t2jp")
    converter = lambda ch: t2jp.convert(s2t.convert(ch))  # noqa: E731

    stats = {"global": collections.Counter(), "books": {}}
    g = stats["global"]
    kept_pages = []  # (category, book, page_no, text)
    drop_samples = []

    for category, books in BOOKS.items():
        for book in books:
            raw = json.loads((Path(args.raw_dir) / f"{book}.json").read_text())
            bs = collections.Counter(pages_raw=len(raw), chars_raw=sum(len(p["content"] or "") for p in raw))
            norm = [(int(p["page"]), normalize_page(p["content"] or "", converter, g)) for p in raw]
            heads = find_running_heads([t for _, t in norm])
            for page_no, text in norm:
                text = clean_lines(text, heads, g)
                reason = classify_drop(text)
                if reason == "figure_dominant":
                    salvaged = salvage_sentences(text)
                    if len(salvaged) >= 50:
                        bs["salvaged_figure_pages"] += 1
                        reason, text = None, salvaged
                if reason:
                    bs[f"drop_{reason}"] += 1
                    drop_samples.append({"book": book, "page": page_no, "reason": reason, "head": text[:160]})
                    continue
                kept_pages.append((category, book, page_no, text))
            stats["books"][book] = bs

    # S4: 重複除去
    seen_hash, lsh = set(), MinHashLSH(threshold=args.dedup_threshold, num_perm=128)
    deduped = []
    for i, (category, book, page_no, text) in enumerate(kept_pages):
        h = hashlib.md5(re.sub(r"\s", "", text).encode()).hexdigest()
        if h in seen_hash:
            stats["books"][book]["drop_exact_dup"] += 1
            continue
        mh = minhash(text)
        dup = lsh.query(mh)
        if dup:
            stats["books"][book]["drop_near_dup"] += 1
            drop_samples.append({"book": book, "page": page_no, "reason": "near_dup", "head": text[:160], "dup_of": dup[0]})
            continue
        seen_hash.add(h)
        lsh.insert(f"{book}:{page_no}", mh)
        deduped.append((category, book, page_no, text))

    # S5/S6: 分割と文書化
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    tok = Tokenizer.from_file(hf_hub_download(args.tokenizer, "tokenizer.json"))
    by_book = collections.defaultdict(list)
    for row in deduped:
        by_book[row[1]].append(row)
    writers = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "qa_eval", "holdout_ppl")}
    split_tot = collections.Counter()
    for book, rows in by_book.items():
        n = len(rows)
        groups = []
        for idx, (category, _, page_no, text) in enumerate(rows):
            sp = split_of(idx, n, (0.30, 0.40), (0.60, 0.70))
            if groups and groups[-1]["split"] == sp and page_no - groups[-1]["pages"][-1] == 1:
                groups[-1]["pages"].append(page_no)
                groups[-1]["texts"].append(text)
            else:
                groups.append({"split": sp, "category": category, "pages": [page_no], "texts": [text]})
        for gi, grp in enumerate(groups):
            text = join_pages(grp["texts"])
            ntok = len(tok.encode(text, add_special_tokens=False).ids)
            rec = {
                "id": f"{book}-{gi:04d}",
                "book": book,
                "category": grp["category"],
                "split": grp["split"],
                "pages": [grp["pages"][0], grp["pages"][-1]],
                "n_chars": len(text),
                "n_tokens": ntok,
                "text": text,
            }
            writers[grp["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            bs = stats["books"][book]
            bs[f"{grp['split']}_pages"] += len(grp["pages"])
            bs[f"{grp['split']}_tokens"] += ntok
            bs[f"{grp['split']}_chars"] += len(text)
            split_tot[f"{grp['split']}_docs"] += 1
            split_tot[f"{grp['split']}_pages"] += len(grp["pages"])
            split_tot[f"{grp['split']}_tokens"] += ntok
            split_tot[f"{grp['split']}_chars"] += len(text)
    for w in writers.values():
        w.close()

    drops = collections.Counter()
    for bs in stats["books"].values():
        for k, v in bs.items():
            if k.startswith("drop_"):
                drops[k] += v
    summary = {
        "pages_raw": sum(b["pages_raw"] for b in stats["books"].values()),
        "chars_raw": sum(b["chars_raw"] for b in stats["books"].values()),
        "line_cleaning": {k: v for k, v in g.items()},
        "page_drops": dict(drops.most_common()),
        "splits": dict(split_tot),
    }
    json.dump({"summary": summary, "books": stats["books"]}, open(out / "stats.json", "w"), ensure_ascii=False, indent=1)
    with open(out / "drop_samples.jsonl", "w") as f:
        for d in drop_samples:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
