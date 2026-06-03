#!/usr/bin/env python3
"""Download InternLM2-1.8B model - try multiple repo IDs"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# Try different repo IDs
repo_ids = [
    "internlm/InternLM2-1_8B",
    "internlm/InternLM2_1_8B",
    "internlm/InternLM2.5-1.8B-chat",
    "internlm/InternLM-1.8B",
]

for repo_id in repo_ids:
    print(f"\nTrying {repo_id}...")
    try:
        snapshot_download(
            repo_id,
            local_dir=f"/root/autodl-tmp/models/{repo_id.replace('/', '_')}",
            local_dir_use_symlinks=False,
        )
        print(f"SUCCESS: {repo_id}")
        break
    except Exception as e:
        print(f"FAILED: {e}")
