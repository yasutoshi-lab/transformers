"""freesia-weak-cl 後処理 段階B-1: bge-m3 でクエリと一節を埋め込んで保存する。

入力: ~/freesia-work/weak-cl/pairs_a.jsonl.gz
出力: ~/freesia-work/weak-cl/emb/{lang}_{passage,query}.npy（fp16・L2 正規化済み、pairs_a の言語ごとの出現順）
      ~/freesia-work/weak-cl/emb/{lang}_ids.txt（各行の id）

bge-m3 の密ベクトル（CLS プーリング・L2 正規化）を fp16 で計算する。途中で止まっても、保存済みのファイルは飛ばす。
"""
import gzip, json, os, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

D = os.path.expanduser("~/freesia-work/weak-cl")
E = f"{D}/emb"
MODEL = "BAAI/bge-m3"
MAXLEN = {"query": 128, "passage": 512}


def embed(tok, model, texts, kind):
    out = np.zeros((len(texts), 1024), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])[::-1]  # 長さ順にまとめて詰め物を減らす
    bs = 256 if kind == "query" else 64
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN[kind], return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()
        if (s // bs) % 2000 == 0:
            print(f"{time.strftime('%m-%d %H:%M')} {kind} {s}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    os.makedirs(E, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    rows = {"ja": [], "en": []}
    for l in gzip.open(f"{D}/pairs_a.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["lang"]].append((r["id"], r["passage"], r["query"]))
    for lang, rs in rows.items():
        with open(f"{E}/{lang}_ids.txt", "w") as f:
            f.writelines(i + "\n" for i, _, _ in rs)
        for kind, col in (("query", 2), ("passage", 1)):
            path = f"{E}/{lang}_{kind}.npy"
            if os.path.exists(path):
                continue
            print(lang, kind, len(rs), flush=True)
            v = embed(tok, model, [r[col] for r in rs], kind)
            np.save(path + ".tmp.npy", v)
            os.replace(path + ".tmp.npy", path)
    print("EMBED_DONE", {k: len(v) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
