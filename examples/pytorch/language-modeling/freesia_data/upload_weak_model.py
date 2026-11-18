import os, shutil
from huggingface_hub import HfApi
LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
src = f"{LM}/artifacts/runs/freesia-300m-weak/final"
stage = os.path.expanduser("~/freesia-work/v2/upload-weak"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
for f in ("config.json", "freesia_embed_meta.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
    os.symlink(f"{src}/{f}", f"{stage}/{f}")
shutil.copy(os.path.expanduser("~/freesia-work/v2/freesia-300m-causal-weak-README.md"), f"{stage}/README.md")
api = HfApi(); r = "yasutoshi-lab/Freesia-300M-Causal-Weak"
api.create_repo(r, private=True, exist_ok=False)
assert api.model_info(r).private
api.upload_folder(repo_id=r, folder_path=stage, commit_message="Upload Freesia-300M-Causal-Weak (stage 3, weakly supervised contrastive learning)")
print("FILES", sorted(api.list_repo_files(r)), "private", api.model_info(r).private)
print("WEAK_UPLOADED")
