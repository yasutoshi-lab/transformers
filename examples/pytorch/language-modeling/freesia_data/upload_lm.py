import os, shutil
from huggingface_hub import HfApi
LM = os.path.expanduser("~/python_project/transformers/examples/pytorch/language-modeling")
src = f"{LM}/artifacts/runs/freesia-300m-lm/final"
stage = os.path.expanduser("~/freesia-work/v2/upload-lm"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
    os.symlink(f"{src}/{f}", f"{stage}/{f}")
shutil.copy(os.path.expanduser("~/freesia-work/v2/freesia-300m-lm-README.md"), f"{stage}/README.md")
api = HfApi(); r = "yasutoshi-lab/freesia-300m-lm"
api.create_repo(r, private=True, exist_ok=False)
assert api.model_info(r).private
api.upload_folder(repo_id=r, folder_path=stage, commit_message="Upload Freesia-300M stage-2 causal LM (v2 data, tokenizer v32k)")
print("FILES", sorted(api.list_repo_files(r)), "private", api.model_info(r).private)
print("LM_UPLOADED")
