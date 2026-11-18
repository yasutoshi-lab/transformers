"""freesia-supervised-cl: bge-m3 でプールの一節、または retrieval の生成クエリを埋め込む。

使い方: python supcl_embed.py pool            → sup-cl/emb/{lang}_pool.npy・{lang}_pool_ids.txt（pool_{lang} の並び）
        python supcl_embed.py query GEN.jsonl  → sup-cl/emb/{lang}_query_<GEN の名前>.npy・_ids.txt（retrieval の ok 行だけ）

weak-cl と同じ embed()（CLS プーリング・L2 正規化・fp16）を使う。保存済みのファイルは飛ばす。
"""
import gzip, json, os, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weakcl_embed import MODEL, embed  # noqa: E402

D = os.path.expanduser("~/freesia-work/sup-cl")
E = f"{D}/emb"


def save(path, ids, texts, kind, tok, model):
    if os.path.exists(path):
        print("skip", path, flush=True)
        return
    with open(path.replace(".npy", "_ids.txt"), "w") as f:
        f.writelines(i + "\n" for i in ids)
    print(path, len(texts), flush=True)
    v = embed(tok, model, texts, kind)
    np.save(path + ".tmp.npy", v)
    os.replace(path + ".tmp.npy", path)


def main():
    os.makedirs(E, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    if sys.argv[1] == "pool":
        for lang in ("ja", "en"):
            rs = [json.loads(l) for l in gzip.open(f"{D}/pool_{lang}.jsonl.gz", "rt")]
            save(f"{E}/{lang}_pool.npy", [r["pt_id"] for r in rs], [r["passage"] for r in rs], "passage", tok, model)
            del rs
    else:
        name = os.path.basename(sys.argv[2]).split(".")[0]
        rows = {"ja": [], "en": []}
        for l in open(sys.argv[2]):
            r = json.loads(l)
            if r["task"] == "retrieval" and r.get("query"):
                rows[r["lang"]].append(r)
        for lang, rs in rows.items():
            save(f"{E}/{lang}_query_{name}.npy", [r["id"] for r in rs], [r["query"] for r in rs], "query", tok, model)
    print("EMBED_DONE", flush=True)


if __name__ == "__main__":
    main()
