"""Final causal-LM loss of freesia-300m-lm on the full validation and test token files (non-overlapping 1024 windows)."""
import json, math, os, sys
import numpy as np, torch
LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, LM)
from transformers import FreesiaForPreTraining
from run_freesia_pretrain import MODE_LM
P = os.path.expanduser("~/freesia-work/v2/packed")
model = FreesiaForPreTraining.from_pretrained(f"{LM}/artifacts/runs/freesia-300m-lm/final").cuda().eval()
res = {}
with torch.no_grad():
    for split in ("validation", "test"):
        for lang in ("ja", "en"):
            data = np.memmap(f"{P}/{lang}_{split}.bin", dtype=np.uint16, mode="r")
            n = len(data) // 1024
            tot, cnt = 0.0, 0
            for s in range(0, n, 8):
                b = min(8, n - s)
                ids = torch.from_numpy(np.asarray(data[s * 1024:(s + b) * 1024], dtype=np.int64).reshape(b, 1024)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(ids, labels=ids, mode=MODE_LM).loss
                tot += loss.item() * b * 1023; cnt += b * 1023
            res[f"{split}/{lang}"] = {"loss": round(tot / cnt, 4), "ppl": round(math.exp(tot / cnt), 2), "tokens": int(n * 1024)}
            print(split, lang, res[f"{split}/{lang}"], flush=True)
json.dump(res, open(f"{LM}/artifacts/runs/freesia-300m-lm/final_eval.json", "w"), indent=1)
print("EVAL_DONE")
