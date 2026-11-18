"""freesia-weak-cl 後処理 段階B-2: 同じ言語の一節すべてを検索対象にしたときの、自分の一節の順位を求める。

入力: ~/freesia-work/weak-cl/emb/{lang}_{query,passage}.npy（weakcl_embed.py の出力）
出力: ~/freesia-work/weak-cl/emb/{lang}_rank.npy（int32、1 が最上位）・{lang}_own.npy（自分の一節とのスコア）

順位は「自分の一節より高いスコアの一節の数 + 1」。
"""
import os, sys, time
import numpy as np, torch

E = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/freesia-work/weak-cl/emb")  # B 案では b/emb を渡す
BS = 1024

for lang in ("ja", "en"):
    if os.path.exists(f"{E}/{lang}_rank.npy"):
        r = np.load(f"{E}/{lang}_rank.npy")
        print(lang, len(r), {k: round(float((r <= k).mean()), 4) for k in (1, 3, 10, 30, 100)}, "saved", flush=True)
        continue
    t0 = time.time()
    P = torch.from_numpy(np.load(f"{E}/{lang}_passage.npy")).cuda()
    Q = torch.from_numpy(np.load(f"{E}/{lang}_query.npy")).cuda()
    n = len(Q)
    # 行列積の fp16 出力と同じ丸めにそろえる
    own = torch.cat([(Q[s:s + 65536].float() * P[s:s + 65536].float()).sum(-1).half() for s in range(0, n, 65536)])
    rank = torch.empty(n, dtype=torch.int32, device="cuda")
    for s in range(0, n, BS):
        e = min(s + BS, n)
        sc = Q[s:e] @ P.T
        rank[s:e] = 1 + (sc > own[s:e, None]).sum(-1).int()
    np.save(f"{E}/{lang}_rank.npy", rank.cpu().numpy())
    np.save(f"{E}/{lang}_own.npy", own.float().cpu().numpy())
    r = rank.cpu().numpy()
    print(lang, n, {k: round(float((r <= k).mean()), 4) for k in (1, 3, 10, 30, 100)}, f"{time.time() - t0:.0f}s", flush=True)
    del P, Q
    torch.cuda.empty_cache()
print("RANK_DONE")
