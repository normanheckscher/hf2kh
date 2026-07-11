#!/usr/bin/env python3
"""
Standalone upload worker, always run as its OWN subprocess by mirror.py.

Why this exists: empirically, setting os.environ["HF_HUB_DISABLE_XET"] = "1"
AFTER huggingface_hub has already been imported in the same process does
NOT stop upload_folder() from attempting xet — it still tries to hit
KohakuHub's (nonexistent) /xet-write-token/ endpoint and crashes. Whatever
decides xet-on-vs-off for uploads in this installed version appears to be
frozen at huggingface_hub's import time, not read fresh per call. mirror.py
imports huggingface_hub once near the top of its own process (for the
download side, where xet SHOULD be on), so toggling the env var later in
that same process can't un-freeze the decision for the upload call.

The fix: do the upload in a brand-new Python process where
HF_HUB_DISABLE_XET=1 is set BEFORE huggingface_hub is imported at all, so
the decision is frozen correctly this time.

argv: repo_id repo_type local_path
env required (inherited from parent, with HF_HUB_DISABLE_XET forced to "1"
before this process's Python even starts): KOHAKU_ENDPOINT, KOHAKU_TOKEN

Prints "UPLOAD_OK" and exits 0 on success. Exits nonzero with the error on
stderr on failure. No retry logic here — mirror.py's caller retries by
re-invoking this whole subprocess.
"""
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"

if len(sys.argv) != 4:
    print("usage: _kohaku_upload_worker.py <repo_id> <repo_type> <local_path>", file=sys.stderr)
    sys.exit(2)

repo_id, repo_type, local_path = sys.argv[1], sys.argv[2], sys.argv[3]

from huggingface_hub import HfApi

api = HfApi(endpoint=os.environ["KOHAKU_ENDPOINT"], token=os.environ["KOHAKU_TOKEN"])

try:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=local_path)
except Exception as e:
    print(f"UPLOAD_FAILED: {e}", file=sys.stderr)
    sys.exit(1)

print("UPLOAD_OK")
