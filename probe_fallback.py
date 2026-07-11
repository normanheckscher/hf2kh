#!/usr/bin/env python3
"""
One-shot probe: confirms whether KohakuHub's ?fallback=false query param
actually distinguishes "genuinely mirrored locally" from "proxied from
real HuggingFace because we don't have it" — and shows the exact JSON
shape of the tree/repo-info responses so check_quants.py's parsing can be
built against reality instead of a guess.

Run this once against your live KohakuHub instance:
    source .mirror_env
    ./venv/bin/python3 probe_fallback.py

Pass a repo you know you HAVE actually mirrored (e.g. one of the small
Qwen3 test models from early on) and one you know you have NOT, e.g.:
    ./venv/bin/python3 probe_fallback.py Qwen/Qwen3-0.6B unsloth/gemma-4-31B-it-qat-GGUF
"""
import json
import os
import sys

import httpx

KOHAKU_ENDPOINT = os.environ["KOHAKU_ENDPOINT"]
KOHAKU_TOKEN = os.environ["KOHAKU_TOKEN"]

DEFAULT_MIRRORED = "Qwen/Qwen3-0.6B"          # known-mirrored, per your manifest
DEFAULT_NOT_MIRRORED = "unsloth/gemma-4-31B-it-qat-GGUF"  # shown as false-DRIFT


def probe(repo_id, repo_type="model"):
    plural = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    headers = {"Authorization": f"Bearer {KOHAKU_TOKEN}"} if KOHAKU_TOKEN else {}

    print(f"\n{'='*70}\n{repo_id}  (type={repo_type})\n{'='*70}")

    for label, path in [
        ("repo-info, fallback=true (default)",  f"/api/{plural}/{repo_id}"),
        ("repo-info, fallback=false",           f"/api/{plural}/{repo_id}?fallback=false"),
        ("tree, fallback=true (default)",       f"/api/{plural}/{repo_id}/tree/main?recursive=true"),
        ("tree, fallback=false",                f"/api/{plural}/{repo_id}/tree/main?recursive=true&fallback=false"),
    ]:
        url = f"{KOHAKU_ENDPOINT}{path}"
        try:
            resp = httpx.get(url, headers=headers, timeout=15)
        except Exception as e:
            print(f"\n--- {label} ---\nREQUEST FAILED: {e}")
            continue
        print(f"\n--- {label} ---")
        print(f"GET {path}")
        print(f"status: {resp.status_code}")
        print(f"headers of interest: X-Error-Code={resp.headers.get('x-error-code')}  "
              f"X-Repo-Commit={resp.headers.get('x-repo-commit')}")
        body = resp.text
        try:
            parsed = resp.json()
            body = json.dumps(parsed, indent=2)[:1500]
        except Exception:
            body = body[:500]
        print(f"body (truncated to 1500 chars):\n{body}")


def main():
    args = sys.argv[1:]
    if not args:
        args = [DEFAULT_MIRRORED, DEFAULT_NOT_MIRRORED]
    for repo_id in args:
        probe(repo_id)

    print(f"\n{'='*70}")
    print("What to look for:")
    print("  - Does fallback=false produce a 404 / X-Error-Code: RepoNotFound")
    print("    for the repo you have NOT mirrored, while fallback=true (default)")
    print("    returns 200 with a full file listing for the same repo?")
    print("  - For the repo you HAVE mirrored, does fallback=false still return")
    print("    200, and what are the exact JSON key names for each file entry")
    print("    (path? rfilename? name?) in the tree response?")
    print("  - Does the repo-info response include a '_source' field when")
    print("    proxied, and is it absent/null for genuinely local repos?")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
