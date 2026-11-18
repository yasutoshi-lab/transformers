import os, sys, json, glob, requests, time
sys.path.insert(0, os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling"))
from huggingface_hub import HfApi, CommitOperationAdd, get_token
from transformers import AutoTokenizer
from train_freesia_tokenizer import evaluate
W = os.path.expanduser("~/freesia-work/v2"); api = HfApi(); H = {"Authorization": "Bearer " + get_token()}
# re-measure tokenizer compression on the new test split
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling/artifacts/tokenizer-v2/v32k"))
import pyarrow.parquet as pq, tempfile
paths = {}
for l in ("ja", "en"):
    tf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    for x in pq.read_table(f"{W}/tokenizer-pt/{l}/test-00000-of-00001.parquet", columns=["text"]).column(0).to_pylist():
        tf.write(x.replace("\n", " ").strip() + "\n")
    tf.close(); paths[l] = __import__("pathlib").Path(tf.name)
print("TEST_CPT", json.dumps(evaluate(tok._tokenizer, paths, max_lines=10**9)), flush=True)
for repo, d in (("yasutoshi-lab/freesia-pt", f"{W}/freesia-pt"), ("yasutoshi-lab/freesia-tokenizer-pt", f"{W}/tokenizer-pt")):
    assert api.dataset_info(repo).private
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=f"{d}/README.md")]
    for l in ("ja", "en"):
        ops.append(CommitOperationAdd(path_in_repo=f"{l}/test-00000-of-00001.parquet", path_or_fileobj=f"{d}/{l}/test-00000-of-00001.parquet"))
    api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message="Rebuild test split with stricter near-duplicate filter (J<0.5 vs train/validation)")
    print("COMMITTED", repo, api.dataset_info(repo).private, flush=True)
for repo in ("yasutoshi-lab/freesia-pt", "yasutoshi-lab/freesia-tokenizer-pt"):
    for _ in range(40):
        out = {}
        for c in ("ja", "en"):
            j = requests.get("https://datasets-server.huggingface.co/first-rows", params={"dataset": repo, "config": c, "split": "test"}, headers=H, timeout=60).json()
            out[c] = len(j["rows"]) if "rows" in j else j.get("error", "")[:100]
        if not any(isinstance(v, str) for v in out.values()):
            break
        time.sleep(60)
    print("VIEWER", repo, out, flush=True)
print("TEST_UPLOAD_DONE")
