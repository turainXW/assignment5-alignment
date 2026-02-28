#!/usr/bin/env python3
"""Upload GRPO models to ModelScope."""

from modelscope.hub.api import HubApi

TOKEN = "ms-97ebb469-2b95-48b3-8606-ea14742c7b34"

models = [
    ("TURAINXW/grpo-qwen2.5-math-1.5b-step150", "data/trl_grpo_output/checkpoint-150"),
    ("TURAINXW/grpo-qwen2.5-math-1.5b-final", "data/trl_grpo_output/final_model"),
]

api = HubApi()
api.login(TOKEN)

for repo_id, local_path in models:
    print(f"\n{'='*60}")
    print(f"Uploading: {local_path} -> {repo_id}")
    print(f"{'='*60}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        commit_message=f"Upload model from {local_path}",
    )
    print(f"Done: {repo_id}")

print("\nAll uploads complete!")
