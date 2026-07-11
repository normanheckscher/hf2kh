#!/usr/bin/env python3
"""Audit hf-cache: flag anything present that ISN'T recorded as successfully
mirrored — these are stuck/failed downloads worth investigating or clearing."""
import json
import os
from huggingface_hub import scan_cache_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_HOME = "/mnt/leo-storage/KohakuHub/hf-cache"
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "mirrored_repos.json")

manifest = {}
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

mirrored_repo_ids = {key.split(":", 1)[1] for key in manifest}

info = scan_cache_dir(os.path.join(HF_HOME, "hub"))
print(f"Total cache size: {info.size_on_disk_str}\n")

if not info.repos:
    print("✅ Cache is empty — nothing to report.")
else:
    orphans = []
    for repo in info.repos:
        flag = "" if repo.repo_id in mirrored_repo_ids else "  ⚠️  NOT in manifest — stuck/failed?"
        print(f"{repo.size_on_disk_str:>10}  {repo.repo_id}{flag}")
        if not flag == "":
            orphans.append(repo)

    if orphans:
        total_orphan_size = sum(r.size_on_disk for r in orphans)
        print(f"\n⚠️  {len(orphans)} unaccounted-for repo(s) using "
              f"{total_orphan_size / 1e9:.2f} GB. These likely failed to "
              f"complete their mirror. Investigate, or clear with:")
        for r in orphans:
            print(f"   rm -rf {HF_HOME}/hub/models--{r.repo_id.replace('/', '--')}")
