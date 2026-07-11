#!/usr/bin/env python3
"""Re-fetch small metadata files (README, LICENSE, chat templates) that an
earlier allowlist-based filter silently dropped, and add them to already-
mirrored KohakuHub repos WITHOUT re-touching the large weight files."""
import json
import os
import socket
import sys

def use_direct_dns(servers=("8.8.8.8", "8.8.4.4")):
    try:
        import dns.resolver
    except ImportError:
        return
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = list(servers)
    _orig = socket.getaddrinfo
    def _patched(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return _orig(str(resolver.resolve(host, "A")[0]), port, family, type, proto, flags)
        except Exception:
            return _orig(host, port, family, type, proto, flags)
    socket.getaddrinfo = _patched

use_direct_dns()

KOHAKU_ENDPOINT = os.environ["KOHAKU_ENDPOINT"]
KOHAKU_TOKEN = os.environ["KOHAKU_TOKEN"]
HF_TOKEN_REAL = os.environ["HF_TOKEN_REAL"]
HF_HOME = os.environ.get("MIRROR_HF_HOME", "/mnt/leo-storage/KohakuHub/hf-cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_DISABLE_XET"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "mirrored_repos.json")

from huggingface_hub import snapshot_download, HfApi

METADATA_PATTERNS = ["README.md", "*.md", "LICENSE*", "*.jinja", "USE_POLICY.md"]

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

os.environ["HF_TOKEN"] = HF_TOKEN_REAL
kohaku_api = HfApi(endpoint=KOHAKU_ENDPOINT, token=KOHAKU_TOKEN)

for key in manifest:
    repo_type, repo_id = key.split(":", 1)
    print(f"Backfilling metadata: {repo_id} ({repo_type})...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id, repo_type=repo_type,
            allow_patterns=METADATA_PATTERNS,
        )
        kohaku_api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=local_path)
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ⚠️  Skipped ({e})")
