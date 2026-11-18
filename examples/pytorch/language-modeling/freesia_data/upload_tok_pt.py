import os, requests, time
from huggingface_hub import HfApi, get_token
r = "yasutoshi-lab/freesia-tokenizer-pt"; d = os.path.expanduser("~/freesia-work/v2/tokenizer-pt")
api = HfApi()
api.create_repo(r, repo_type="dataset", private=True, exist_ok=False)
assert api.dataset_info(r).private
api.upload_folder(repo_id=r, repo_type="dataset", folder_path=d, allow_patterns=["README.md", "ja/*.parquet", "en/*.parquet"])
print("FILES", sorted(api.list_repo_files(r, repo_type="dataset")), "private", api.dataset_info(r).private, flush=True)
H = {"Authorization": "Bearer " + get_token()}
for _ in range(40):
    out = {}
    for c in ["ja", "en"]:
        for s in ["train", "validation", "test"]:
            j = requests.get("https://datasets-server.huggingface.co/first-rows", params={"dataset": r, "config": c, "split": s}, headers=H, timeout=60).json()
            out[f"{c}/{s}"] = len(j["rows"]) if "rows" in j else j.get("error", "")[:120]
    if not any(isinstance(v, str) and "not ready" in v for v in out.values()):
        break
    time.sleep(60)
print("VIEWER", out)
print("TOKPT_UPLOADED")
