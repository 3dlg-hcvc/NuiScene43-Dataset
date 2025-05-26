import os
from huggingface_hub import snapshot_download

local_dir = 'data'
os.makedirs(local_dir, exist_ok=True)

snapshot_dir = snapshot_download(repo_id="3dlg-hcvc/NuiScene43", repo_type='dataset', local_dir=local_dir, allow_patterns=["nuiscene43/**"])
