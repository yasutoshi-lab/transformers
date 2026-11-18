"""freesia-weak-cl B 案 段階B-1: 段階A を通ったペアを bge-m3 で埋め込む。旧ペアは前回の埋め込みを使い回す。

入力: ~/freesia-work/weak-cl/b/pairs_a.jsonl.gz、~/freesia-work/weak-cl/emb/（前回の埋め込みと ids）
出力: ~/freesia-work/weak-cl/b/emb/{lang}_{passage,query}.npy・{lang}_ids.txt（b/pairs_a の言語ごとの出現順）

埋め込みの計算は weakcl_embed.py と同じ（CLS プーリング・L2 正規化・fp16）。前回にない id だけを新しく計算する。
"""
import gzip, json, os, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weakcl_embed import MODEL, embed  # noqa: E402

D = os.path.expanduser("~/freesia-work/weak-cl")
OLD, E = f"{D}/emb", f"{D}/b/emb"


def main():
    os.makedirs(E, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    rows = {"ja": [], "en": []}
    for l in gzip.open(f"{D}/b/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["lang"]].append((r["id"], r["passage"], r["query"]))
    for lang, rs in rows.items():
        with open(f"{E}/{lang}_ids.txt", "w") as f:
            f.writelines(i + "\n" for i, _, _ in rs)
        old_pos = {x.strip(): n for n, x in enumerate(open(f"{OLD}/{lang}_ids.txt"))}
        src = [old_pos.get(i, -1) for i, _, _ in rs]
        new = [n for n, s in enumerate(src) if s < 0]
        reuse = np.array([(n, s) for n, s in enumerate(src) if s >= 0], dtype=np.int64)
        print(lang, "total", len(rs), "reuse", len(reuse), "new", len(new), flush=True)
        for kind, col in (("query", 2), ("passage", 1)):
            path = f"{E}/{lang}_{kind}.npy"
            if os.path.exists(path):
                continue
            out = np.zeros((len(rs), 1024), dtype=np.float16)
            out[reuse[:, 0]] = np.load(f"{OLD}/{lang}_{kind}.npy", mmap_mode="r")[reuse[:, 1]]
            out[new] = embed(tok, model, [rs[n][col] for n in new], kind)
            np.save(path + ".tmp.npy", out)
            os.replace(path + ".tmp.npy", path)
    print("EMBED_DONE", {k: len(v) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
