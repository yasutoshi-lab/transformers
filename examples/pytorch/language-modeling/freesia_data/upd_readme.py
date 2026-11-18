import os
from huggingface_hub import HfApi
api = HfApi(); r = "yasutoshi-lab/freesia-300m-lm"
assert api.model_info(r).private
api.upload_file(repo_id=r, path_or_fileobj=os.path.expanduser("~/freesia-work/v2/freesia-300m-lm-README.md"), path_in_repo="README.md", commit_message="Add final validation/test loss")
print("README_UPDATED", api.model_info(r).private)
