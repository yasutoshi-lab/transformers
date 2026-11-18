"""freesia-supervised-cl: retrieval の生成クエリごとに、同じ split のプールから難しい負例の候補を探す。

使い方: python supcl_mine.py GEN.jsonl OUT.jsonl
入力: sup-cl/emb/{lang}_pool.npy・_pool_ids.txt、sup-cl/emb/{lang}_query_<GEN の名前>.npy・_ids.txt、sup-cl/pool_{lang}.jsonl.gz
出力: OUT.jsonl（retrieval の行ごとに own_rank・own_score・cands=[[pt_id, score], ...]）

- 探す範囲はクエリと同じ split の文書だけ（validation のクエリの負例は validation の文書からしか取らない）
- 自分の一節の順位 own_rank は「同じ split のプールで自分より高いスコアの一節の数 + 1」
- 候補は自分以外の一節の 10〜50 位から、スコアが自分の一節の 0.95 倍未満で、本文が正例と異なり互いに重複しないものを上から CANDS 件
"""
import gzip, json, os, sys
import numpy as np, torch

D = os.path.expanduser("~/freesia-work/sup-cl")
E = f"{D}/emb"
LO, HI, CANDS, MARGIN, BS = 10, 50, 10, 0.95, 512


def main():
    gen, out_path = sys.argv[1], sys.argv[2]
    name = os.path.basename(gen).split(".")[0]
    rows = {}
    for l in open(gen):
        r = json.loads(l)
        if r["task"] == "retrieval" and r.get("query"):
            rows[r["id"]] = r
    out = open(out_path, "w")
    for lang in ("ja", "en"):
        pool_ids = [x.strip() for x in open(f"{E}/{lang}_pool_ids.txt")]
        meta = [json.loads(l) for l in gzip.open(f"{D}/pool_{lang}.jsonl.gz", "rt")]
        assert [m["pt_id"] for m in meta] == pool_ids, "プールと埋め込みの並びが一致しない"
        pos = {p: k for k, p in enumerate(pool_ids)}
        P = torch.from_numpy(np.load(f"{E}/{lang}_pool.npy")).cuda()
        qids = [x.strip() for x in open(f"{E}/{lang}_query_{name}_ids.txt")]
        Q = torch.from_numpy(np.load(f"{E}/{lang}_query_{name}.npy")).cuda()
        for split in ("validation", "test", "train"):
            idx = [k for k, m in enumerate(meta) if m["split"] == split]
            lo, hi = idx[0], idx[-1] + 1
            assert hi - lo == len(idx), "プールの split が連続していない"
            sel = [k for k, q in enumerate(qids) if rows[q]["split"] == split]
            for s in range(0, len(sel), BS):
                b = sel[s:s + BS]
                sc = (Q[b] @ P[lo:hi].T).float()
                own_k = torch.tensor([pos[rows[qids[k]]["pt_id"]] - lo for k in b], device="cuda")
                assert all(meta[lo + int(o)]["pt_id"] == rows[qids[k]]["pt_id"] for o, k in zip(own_k, b))
                own = sc.gather(1, own_k[:, None]).squeeze(1)
                rank = 1 + (sc > own[:, None]).sum(1)
                sc.scatter_(1, own_k[:, None], -1e4)  # 自分を候補から外す
                top_s, top_i = sc.topk(HI, dim=1)
                for j, k in enumerate(b):
                    r = rows[qids[k]]
                    ptext = meta[lo + int(own_k[j])]["passage"]
                    seen, cands = {ptext}, []
                    for t in range(LO - 1, HI):
                        v, i = float(top_s[j, t]), lo + int(top_i[j, t])
                        if v >= MARGIN * float(own[j]) or meta[i]["passage"] in seen:
                            continue
                        seen.add(meta[i]["passage"])
                        cands.append([meta[i]["pt_id"], round(v, 4)])
                        if len(cands) == CANDS:
                            break
                    out.write(json.dumps({"id": r["id"], "own_rank": int(rank[j]), "own_score": round(float(own[j]), 4),
                                          "cands": cands}) + "\n")
            print(lang, split, len(sel), flush=True)
        del P, Q
        torch.cuda.empty_cache()
    out.close()
    print("MINE_DONE", flush=True)


if __name__ == "__main__":
    main()
