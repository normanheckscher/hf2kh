#!/usr/bin/env python3
"""
Standalone download worker, run as its own subprocess by mirror.py ONLY
after the normal in-process xet-enabled snapshot_download() has already
failed with the known hf_xet/DNS-bypass failure signature (see
XET_ERROR_MARKERS in mirror.py — e.g. "Unable to parse string as hex hash
value").

Why this has to be a subprocess: empirically (confirmed live, 2026-07-11),
setting os.environ["HF_HUB_DISABLE_XET"] = "1" AFTER huggingface_hub has
already been imported in the same process does NOT reliably stop
snapshot_download() from attempting xet — the retry still hit the exact
same "hex hash value" error on the very next attempt despite the env var
being flipped first. This is the same freeze-at-import behavior already
documented and worked around on the upload side
(_kohaku_upload_worker.py) — whatever decides xet-on-vs-off appears to be
frozen at huggingface_hub's import time, not read fresh per call.
mirror.py imports huggingface_hub once near the top of its own process
(needed for the first, xet-enabled attempt), so flipping the env var
later in that same process can't un-freeze the decision. The fix: retry
in a brand-new Python process where HF_HUB_DISABLE_XET=1 is set BEFORE
huggingface_hub is imported at all.

argv: repo_id repo_type json_extra_kwargs
  json_extra_kwargs is a JSON object merged into the snapshot_download()
  call (e.g. {"max_workers": 3, "ignore_patterns": [...]} or
  {"max_workers": 3, "allow_patterns": [...]}).

env required (inherited from parent, with HF_HUB_DISABLE_XET forced to
"1" before this process's Python even starts): HF_HOME, HF_TOKEN, and
whatever else the parent process already has set (KOHAKU_* not needed
here — this is the download-from-HF leg only).

Prints "DOWNLOAD_OK:<local_path>" on the last line and exits 0 on
success. Exits nonzero with the error on stderr on failure. No retry
logic here — mirror.py's caller retries by re-invoking this whole
subprocess.
"""
import json
import os
import socket
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"


def use_direct_dns(servers=("8.8.8.8", "8.8.4.4")):
    """Same DNS-bypass patch as mirror.py's top-level use_direct_dns().
    Duplicated here (rather than imported from mirror.py) so this worker
    has no dependency on mirror.py's module-level env-var requirements
    (KOHAKU_ENDPOINT etc.) that aren't needed for a pure download."""
    try:
        import dns.resolver
    except ImportError:
        return
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = list(servers)
    _original_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            answer = resolver.resolve(host, "A")
            ip = str(answer[0])
            return _original_getaddrinfo(ip, port, family, type, proto, flags)
        except Exception:
            return _original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _patched_getaddrinfo


use_direct_dns()

if len(sys.argv) != 4:
    print("usage: _hf_download_worker.py <repo_id> <repo_type> <json_extra_kwargs>", file=sys.stderr)
    sys.exit(2)

repo_id, repo_type, kwargs_json = sys.argv[1], sys.argv[2], sys.argv[3]
extra_kwargs = json.loads(kwargs_json)

from huggingface_hub import snapshot_download

try:
    local_path = snapshot_download(repo_id=repo_id, repo_type=repo_type, **extra_kwargs)
except Exception as e:
    print(f"DOWNLOAD_FAILED: {e}", file=sys.stderr)
    sys.exit(1)

print(f"DOWNLOAD_OK:{local_path}")
