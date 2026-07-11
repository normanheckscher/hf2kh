#!/usr/bin/env python3
"""
Mirror a HuggingFace repo into a self-hosted KohakuHub instance.

Modes:
  (default)  Mirror if never done before; skip if already in the manifest.
  --size     Report total repo size (no download, no modification).
  --check    Compare local recorded revision vs HF's live revision. Report only.
  --update   Like --check, but tags the old snapshot and re-mirrors if changed.
  --force    Re-mirror unconditionally, no revision check, no tagging.
  --include-original  Also grab original/*.pth alongside safetensors (off by default).
  --quant PATTERN  Only pull files matching PATTERN (substring or glob) plus
                   metadata/README/tokenizer/mmproj. Repeatable over time with
                   a different PATTERN to add another quant without re-pulling
                   the first.
  --list-quants    List files (name+size) in the repo, no download.
  --no-xet         Force-disable hf_xet for the DOWNLOAD-from-HF leg only.

hf_xet (download side): HF has moved large files (many GGUF quants, some
safetensors shards) to xet-only storage with no HTTP fallback. Install:
    ./venv/bin/python3 -m ensurepip --upgrade   # if venv has no pip
    ./venv/bin/python3 -m pip install hf_xet
Xet is on by default now (no longer force-disabled). --no-xet troubleshoots
the download leg only.
CAVEAT: the DNS-bypass patch below only patches Python's socket.getaddrinfo.
hf_xet is a compiled Rust extension with its own networking that does NOT
go through Python's socket module — if the DNS bypass exists because of
broken/hijacked DNS, xet transfers may still fail to resolve hosts even
though regular LFS transfers succeed. Observed failure signature: "Unable
to parse string as hex hash value" partway through a large file — this is
xet receiving a broken/error response and choking on it trying to parse a
content hash.

CONFIRMED LIVE (2026-07-11): simply flipping os.environ["HF_HUB_DISABLE_XET"]
mid-process and retrying in the SAME process does NOT work — the very next
attempt hit the identical error. This is the same freeze-at-import behavior
already known on the upload side (see below): whatever decides xet-on-vs-off
is frozen once huggingface_hub is imported, and mirror.py already imported it
at module load time for the first (xet-enabled) attempt. So the download side
now uses the same subprocess workaround as the upload side: once this
signature is seen, remaining attempts run in a brand-new subprocess
(_hf_download_worker.py) with HF_HUB_DISABLE_XET=1 set before THAT process's
Python even starts.

hf_xet (upload side, KohakuHub): confirmed by testing — KohakuHub does not
implement HF's xet-write-token API (404 on
/api/{repo_type}s/{repo_id}/xet-write-token/{revision}). ALSO confirmed by
testing: toggling HF_HUB_DISABLE_XET after huggingface_hub is already
imported in this process does NOT stop upload_folder() from trying xet —
whatever decides this is frozen at import time, not read fresh per call.
So the upload step below runs in a brand-new subprocess
(_kohaku_upload_worker.py) with HF_HUB_DISABLE_XET=1 set before THAT
process's Python even starts, guaranteeing xet is really off for it. Not
configurable — xet-to-KohakuHub does not work with this server, full stop.

Manifest schema (mirrored_repos.json), v2: each entry now tracks HOW it was
mirrored, not just THAT it was:
    "mode": "full" | "quant-selective"
    "quants_present": [patterns satisfied so far] or null
    "files": [relative filenames actually present when last uploaded]
Older entries (no "mode" key) predate --quant and are treated as "full".
A repo can be in the manifest as "mirrored" while only containing ONE quant
out of a dozen — asking for a full mirror on a quant-selective entry does
NOT silently report success, see mirror_repo() below. For a live check
against what's actually in KohakuHub (manifest is local bookkeeping and can
drift from reality), see check_quants.py.

Usage:
    ./mirror-model.sh <namespace/repo> [--type model|dataset|space] [--size|--check|--update|--force] [--include-original] [--quant PATTERN] [--list-quants] [--no-xet]
"""
import argparse
import fnmatch
import glob
import json
import os
import socket
import subprocess
import sys
import time

def use_direct_dns(servers=("8.8.8.8", "8.8.4.4")):
    try:
        import dns.resolver
    except ImportError:
        print("dnspython not installed — DNS bypass disabled.")
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
    print(f"DNS bypass active — resolving via {', '.join(servers)} directly.")

use_direct_dns()

KOHAKU_ENDPOINT = os.environ["KOHAKU_ENDPOINT"]
KOHAKU_API_DIRECT = os.environ["KOHAKU_API_DIRECT"]
KOHAKU_TOKEN = os.environ["KOHAKU_TOKEN"]
HF_TOKEN_REAL = os.environ["HF_TOKEN_REAL"]
HF_HOME = os.environ.get("MIRROR_HF_HOME", "/mnt/leo-storage/KohakuHub/hf-cache")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "mirrored_repos.json")
UPLOAD_WORKER = os.path.join(SCRIPT_DIR, "_kohaku_upload_worker.py")
DOWNLOAD_WORKER = os.path.join(SCRIPT_DIR, "_hf_download_worker.py")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

import httpx
from huggingface_hub import snapshot_download, HfApi, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError

ORIGINAL_FORMAT_EXCLUDE = ["original/*"]

# Failure-message substrings that indicate the hf_xet/DNS-bypass interaction
# documented above, rather than a generic network blip. Matched case-insensitively.
XET_ERROR_MARKERS = (
    "hex hash value",   # confirmed live signature — xet choking on a broken response
    "hf_xet",
    "xethub",
    "xetblob",
)


def looks_like_xet_error(msg):
    lowered = msg.lower()
    return any(marker in lowered for marker in XET_ERROR_MARKERS)

# Always grab these regardless of --quant filtering — small files that make
# the repo usable/documented (chat template, tokenizer, vision projector).
ALWAYS_INCLUDE_PATTERNS = [
    "*.json", "README*", "*.md", "LICENSE*", "tokenizer*", "*.jinja",
    "mmproj*", "USE_POLICY.md", "*.txt",
]


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def manifest_key(repo_id, repo_type):
    return f"{repo_type}:{repo_id}"


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def report_size(repo_id, repo_type="model", include_original=False, quant_pattern=None):
    hf_api = HfApi(token=HF_TOKEN_REAL)
    try:
        info = hf_api.repo_info(repo_id=repo_id, repo_type=repo_type,
                                 revision="main", files_metadata=True)
    except Exception as e:
        print(f"could not fetch metadata for {repo_id} — {e}")
        return None

    siblings = info.siblings or []
    if not include_original:
        siblings = [s for s in siblings if not s.rfilename.startswith("original/")]

    if quant_pattern:
        glob_pattern = quant_pattern if any(c in quant_pattern for c in "*?[") else f"*{quant_pattern}*"
        matched = [s for s in siblings if fnmatch.fnmatch(s.rfilename.lower(), glob_pattern.lower())]
        always = [s for s in siblings if any(fnmatch.fnmatch(s.rfilename.lower(), p.lower())
                                              for p in ALWAYS_INCLUDE_PATTERNS)]
        siblings = list({s.rfilename: s for s in (matched + always)}.values())
        note = f" (quant '{quant_pattern}' + metadata only, not full repo)"
        if not matched:
            print(f"quant pattern '{quant_pattern}' matched nothing for {repo_id} — sizing metadata only")
    else:
        note = "" if include_original else " (excludes original/*, use --include-original for full size)"

    total = sum((getattr(f, "size", None) or 0) for f in siblings)
    print(f"{human_size(total):>12}  {repo_id}  ({len(siblings)} files){note}")
    return total


def get_remote_files(repo_id, repo_type="model"):
    hf_api = HfApi(token=HF_TOKEN_REAL)
    info = hf_api.repo_info(repo_id=repo_id, repo_type=repo_type,
                             revision="main", files_metadata=True)
    return [(s.rfilename, getattr(s, "size", None) or 0) for s in (info.siblings or [])]


def list_quants(repo_id, repo_type="model"):
    try:
        files = get_remote_files(repo_id, repo_type)
    except Exception as e:
        print(f"could not fetch file listing for {repo_id} — {e}")
        return
    gguf_files = [(f, sz) for f, sz in files if f.endswith(".gguf")]
    other_large = [(f, sz) for f, sz in files if not f.endswith(".gguf") and sz > 1e9]
    if not gguf_files and not other_large:
        print(f"No .gguf or large (>1GB) files in {repo_id}. Full listing has {len(files)} entries.")
        return
    if gguf_files:
        print(f"GGUF quants in {repo_id}:")
        for fname, size in sorted(gguf_files, key=lambda x: x[1]):
            print(f"  {human_size(size):>10}  {fname}")
    if other_large:
        print(f"\nOther large (>1GB) files in {repo_id}:")
        for fname, size in sorted(other_large, key=lambda x: x[1]):
            print(f"  {human_size(size):>10}  {fname}")
    print("\nPass one of the names above (or a distinguishing substring) to --quant.")


def resolve_quant_files(repo_id, repo_type, quant_pattern):
    """Match quant_pattern against the live remote file list. Fails loud
    (not a silent empty download) if nothing matches."""
    files = get_remote_files(repo_id, repo_type)
    glob_pattern = quant_pattern if any(c in quant_pattern for c in "*?[") else f"*{quant_pattern}*"
    matches = [f for f, _ in files if fnmatch.fnmatch(f.lower(), glob_pattern.lower())]
    if not matches:
        print(f"No files in {repo_id} match quant pattern '{quant_pattern}'. Run --list-quants first.")
        sys.exit(1)
    sizes = {f: sz for f, sz in files}
    total = sum(sizes.get(f, 0) for f in matches)
    print(f"  Quant '{quant_pattern}' resolves to {len(matches)} file(s), {human_size(total)}: {', '.join(matches)}")
    return matches


def resolve_schema(spec, schema):
    if "$ref" in schema:
        node = spec
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return node
    return schema


def discover_org_create_route(spec):
    out = []
    for path, methods in spec.get("paths", {}).items():
        if "org" not in path.lower() or "{" in path:
            continue
        if "post" in methods:
            out.append((path, methods["post"]))
    return out


def build_payload(spec, op, namespace):
    content = op.get("requestBody", {}).get("content", {})
    schema = resolve_schema(spec, content.get("application/json", {}).get("schema", {}))
    props, required = schema.get("properties", {}), schema.get("required", [])
    payload = {}
    for field in required:
        ftype = props.get(field, {}).get("type", "string")
        if any(k in field.lower() for k in ("name", "id", "login", "org", "slug")):
            payload[field] = namespace
        elif ftype == "boolean":
            payload[field] = False
        elif ftype in ("integer", "number"):
            payload[field] = 0
        else:
            payload[field] = ""
    return payload


def ensure_namespace_exists(namespace, token):
    print(f"Checking namespace '{namespace}'...")
    try:
        spec = httpx.get(f"{KOHAKU_API_DIRECT}/openapi.json", timeout=10).json()
    except Exception as e:
        print(f"  Couldn't fetch OpenAPI spec: {e}")
        return
    for path, op in discover_org_create_route(spec):
        payload = build_payload(spec, op, namespace)
        try:
            resp = httpx.post(f"{KOHAKU_ENDPOINT}{path}",
                               headers={"Authorization": f"Bearer {token}"},
                               json=payload, timeout=10)
        except Exception as e:
            print(f"  {e}")
            continue
        if resp.status_code in (200, 201, 409):
            print(f"  Namespace ready ({resp.status_code})")
            return
    print("  Could not confirm namespace (may already exist — continuing).")


def clear_stale_locks(repo_id):
    safe_name = repo_id.replace("/", "--")
    pattern = os.path.join(HF_HOME, "hub", f"*{safe_name}*", "**", "*.lock")
    for f in glob.glob(pattern, recursive=True):
        try:
            os.remove(f)
            print(f"  Cleared stale lock: {f}")
        except OSError:
            pass

def clear_stale_incomplete_files(repo_id):
    """huggingface_hub's resume mechanism creates a fresh randomly-suffixed
    .incomplete file on every retry instead of reusing the existing one
    (huggingface/huggingface_hub#4196) — old ones are dead weight, not
    resumable progress. Clear before starting."""
    safe_name = repo_id.replace("/", "--")
    pattern = os.path.join(HF_HOME, "hub", f"*{safe_name}*", "**", "*.incomplete")
    freed, count = 0, 0
    for f in glob.glob(pattern, recursive=True):
        try:
            freed += os.path.getsize(f)
            os.remove(f)
            count += 1
        except OSError:
            pass
    if count:
        print(f"  Cleared {count} orphaned .incomplete file(s), freed {freed / 1e9:.2f} GB")


def cleanup_local_cache(repo_id, repo_type="model"):
    try:
        cache_info = scan_cache_dir(os.path.join(HF_HOME, "hub"))
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.repo_type == repo_type:
                revisions = {rev.commit_hash for rev in repo.revisions}
                delete_strategy = cache_info.delete_revisions(*revisions)
                print(f"  Freeing {delete_strategy.expected_freed_size_str} from local cache for {repo_id}...")
                delete_strategy.execute()
                return
        print(f"  No local cache entry found for {repo_id} (nothing to clean).")
    except Exception as e:
        print(f"  Cache cleanup skipped: {e}")


def get_live_hf_revision(repo_id, repo_type="model"):
    hf_api = HfApi(token=HF_TOKEN_REAL)
    info = hf_api.repo_info(repo_id=repo_id, repo_type=repo_type, revision="main")
    return info.sha


def tag_current_snapshot(repo_id, repo_type, old_revision, kohaku_api):
    tag_name = f"archived-{old_revision[:12]}"
    try:
        kohaku_api.create_tag(repo_id=repo_id, repo_type=repo_type, tag=tag_name)
        print(f"  Tagged previous snapshot as '{tag_name}'")
        return tag_name
    except Exception as e:
        print(f"  Could not create tag '{tag_name}': {e}")
        return None


def check_for_update(repo_id, repo_type="model"):
    manifest = load_manifest()
    key = manifest_key(repo_id, repo_type)
    try:
        remote_rev = get_live_hf_revision(repo_id, repo_type)
    except Exception as e:
        print(f"  Couldn't check live HF revision for {repo_id}: {e}")
        return "check-failed", None, None
    if key not in manifest:
        return "never-mirrored", None, remote_rev
    local_rev = manifest[key]["revision"]
    if local_rev == remote_rev:
        return "up-to-date", local_rev, remote_rev
    return "update-available", local_rev, remote_rev


def entry_mode(entry):
    """Old manifest entries predate mode/quants_present/files — they were
    all full-repo mirrors, so treat a missing key as 'full'."""
    return entry.get("mode", "full")


def record_mirror(manifest, key, revision, local_path, quant_pattern, old_revision_to_tag):
    entry = manifest.get(key, {})
    existing_mode = entry_mode(entry) if entry else None
    existing_quants = set(entry.get("quants_present") or []) if existing_mode == "quant-selective" else set()
    existing_files = set(entry.get("files") or [])

    new_files = set()
    for root, _, files in os.walk(local_path):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), local_path)
            new_files.add(rel)

    if quant_pattern:
        mode = "full" if existing_mode == "full" else "quant-selective"
        quants_present = sorted(existing_quants | {quant_pattern}) if mode == "quant-selective" else None
    else:
        mode = "full"
        quants_present = None

    history = entry.get("history", [])
    if old_revision_to_tag:
        history.append(old_revision_to_tag)

    manifest[key] = {
        "mirrored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "revision": revision,
        "history": history,
        "mode": mode,
        "quants_present": quants_present,
        "files": sorted(existing_files | new_files),
    }
    save_manifest(manifest)


def upload_via_subprocess(repo_id, repo_type, local_path):
    """Run the upload in a fresh Python process with HF_HUB_DISABLE_XET=1
    set before that process's huggingface_hub import happens. See the
    module docstring for why this has to be a subprocess rather than an
    in-process env-var toggle."""
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    result = subprocess.run(
        [sys.executable, UPLOAD_WORKER, repo_id, repo_type, local_path],
        env=env, capture_output=True, text=True,
    )
    ok = result.returncode == 0 and "UPLOAD_OK" in result.stdout
    return ok, result.stdout, result.stderr


def download_via_subprocess(repo_id, repo_type, download_kwargs):
    """Run the download in a fresh Python process with HF_HUB_DISABLE_XET=1
    set before that process's huggingface_hub import happens — the same
    subprocess workaround already used on the upload side (see
    upload_via_subprocess). Confirmed live (2026-07-11) that toggling the
    env var in THIS already-running process does not reliably re-decide
    xet-on-vs-off once huggingface_hub has already been imported here."""
    extra = {k: v for k, v in download_kwargs.items() if k not in ("repo_id", "repo_type")}
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    result = subprocess.run(
        [sys.executable, DOWNLOAD_WORKER, repo_id, repo_type, json.dumps(extra)],
        env=env, capture_output=True, text=True,
    )
    local_path = None
    for line in result.stdout.splitlines():
        if line.startswith("DOWNLOAD_OK:"):
            local_path = line[len("DOWNLOAD_OK:"):]
    ok = result.returncode == 0 and local_path is not None
    return ok, local_path, result.stdout, result.stderr


def mirror_repo(repo_id, repo_type="model", max_retries=5, retry_delay=10,
                force=False, check_only=False, do_update=False, size_only=False,
                include_original=False, quant_pattern=None, list_quants_only=False,
                no_xet=False):
    if list_quants_only:
        list_quants(repo_id, repo_type)
        return

    if size_only:
        report_size(repo_id, repo_type, include_original=include_original, quant_pattern=quant_pattern)
        return

    if no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Xet downloads disabled for this run (--no-xet).")
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        try:
            import hf_xet  # noqa: F401
        except ImportError:
            print("hf_xet not installed — large xet-only files will fail. "
                  "Run: ./venv/bin/python3 -m pip install hf_xet")

    manifest = load_manifest()
    key = manifest_key(repo_id, repo_type)

    if check_only:
        status, local_rev, remote_rev = check_for_update(repo_id, repo_type)
        if status == "never-mirrored":
            print(f"NEW {repo_id}: not yet mirrored (remote revision: {remote_rev[:12]})")
        elif status == "up-to-date":
            print(f"OK {repo_id}: up to date (revision {local_rev[:12]})")
        elif status == "update-available":
            print(f"UPDATE AVAILABLE {repo_id}: local={local_rev[:12]} remote={remote_rev[:12]}")
        else:
            print(f"? {repo_id}: could not determine status")
        if key in manifest:
            entry = manifest[key]
            if entry_mode(entry) == "quant-selective":
                have = ', '.join(entry.get('quants_present') or []) or '(none recorded)'
                print(f"   quant-selective mirror — have: {have}")
        return

    old_revision_to_tag = None
    if do_update and key in manifest and not force:
        status, local_rev, remote_rev = check_for_update(repo_id, repo_type)
        if status == "up-to-date":
            print(f"{repo_id}: already up to date. Nothing to do.")
            return
        elif status == "update-available":
            old_revision_to_tag = local_rev
        elif status == "check-failed":
            print(f"Could not verify update status for {repo_id} — aborting to be safe.")
            return
    elif key in manifest and not force and not do_update:
        entry = manifest[key]
        mode = entry_mode(entry)
        if quant_pattern is None:
            if mode == "full":
                print(f"Already mirrored (full): {repo_id} (on {entry['mirrored_at']}). Skipping.")
                return
            else:
                have = ', '.join(entry.get('quants_present') or []) or '(none recorded)'
                print(f"{repo_id} only has a quant-selective mirror so far (have: {have}).")
                print("NOT treating that as a full mirror. Pass --quant <tag> to add a "
                      "specific quant, or --force to pull the whole repo.")
                return
        else:
            if mode == "full":
                print(f"{repo_id} is already a full mirror — quant '{quant_pattern}' is already included. Skipping.")
                return
            if quant_pattern in (entry.get("quants_present") or []):
                print(f"Already have quant '{quant_pattern}' for {repo_id} (on {entry['mirrored_at']}). Skipping.")
                return
            print(f"{repo_id} already has quant(s) [{', '.join(entry.get('quants_present') or [])}] — "
                  f"adding '{quant_pattern}' without re-pulling those.")

    namespace = repo_id.split("/")[0]
    ensure_namespace_exists(namespace, KOHAKU_TOKEN)
    clear_stale_locks(repo_id)
    clear_stale_incomplete_files(repo_id)
    kohaku_api = HfApi(endpoint=KOHAKU_ENDPOINT, token=KOHAKU_TOKEN)

    if old_revision_to_tag:
        tag_current_snapshot(repo_id, repo_type, old_revision_to_tag, kohaku_api)

    os.environ["HF_TOKEN"] = HF_TOKEN_REAL

    download_kwargs = {"repo_id": repo_id, "repo_type": repo_type, "max_workers": 3}
    if quant_pattern:
        matched_files = resolve_quant_files(repo_id, repo_type, quant_pattern)
        download_kwargs["allow_patterns"] = ALWAYS_INCLUDE_PATTERNS + matched_files
        print(f"\nDownloading {repo_id} from HuggingFace (quant-selective: '{quant_pattern}')...")
    else:
        if not include_original:
            download_kwargs["ignore_patterns"] = ORIGINAL_FORMAT_EXCLUDE
            print("  (excluding original/* duplicate-format folder — pass --include-original to keep it too)")
        print(f"\nDownloading {repo_id} from HuggingFace (full repo)...")

    local_path = None
    xet_auto_disabled = False
    for attempt in range(1, max_retries + 1):
        try:
            if xet_auto_disabled:
                print(f"  (attempt {attempt}: running in a fresh xet-disabled subprocess — "
                      f"no live progress bar for this leg)")
                ok, local_path, out, err = download_via_subprocess(repo_id, repo_type, download_kwargs)
                if not ok:
                    tail = (err.strip() or out.strip() or "download subprocess failed with no output")
                    raise RuntimeError(tail[-800:])
            else:
                local_path = snapshot_download(**download_kwargs)
            break
        except KeyboardInterrupt:
            print("\nInterrupted — re-run to resume.")
            sys.exit(130)
        except Exception as e:
            msg = str(e)
            print(f"Download attempt {attempt}/{max_retries} failed: {msg}")
            if no_xet and "hf_xet" in msg.lower():
                print("   (You passed --no-xet, but this file may require it — re-run without --no-xet.)")
            elif not no_xet and not xet_auto_disabled and looks_like_xet_error(msg):
                print("   Signature matches the known hf_xet/DNS-bypass failure mode. Flipping "
                      "the env var in this process alone doesn't reliably work (same "
                      "freeze-at-import behavior as the upload side) — switching remaining "
                      "attempts to a fresh subprocess with xet disabled before import.")
                xet_auto_disabled = True
                continue
            if attempt == max_retries:
                sys.exit(1)
            time.sleep(retry_delay)

    if xet_auto_disabled:
        print(f"Downloaded to: {local_path}  (xet was disabled via subprocess after a failed attempt)")
    else:
        print(f"Downloaded to: {local_path}")
    print(f"\nMirroring into KohakuHub as {repo_id}...")

    # Upload in a fresh subprocess with xet forced off BEFORE that process's
    # huggingface_hub import — see upload_via_subprocess() and the module
    # docstring for why an in-process env toggle doesn't work here.
    for attempt in range(1, max_retries + 1):
        try:
            ok, out, err = upload_via_subprocess(repo_id, repo_type, local_path)
        except KeyboardInterrupt:
            print("\nInterrupted — re-run to resume. Manifest/cache untouched.")
            sys.exit(130)
        if ok:
            print(f"\nMirror complete: {KOHAKU_ENDPOINT}/{repo_id}")
            new_revision = os.path.basename(local_path)
            record_mirror(manifest, key, new_revision, local_path, quant_pattern, old_revision_to_tag)
            cleanup_local_cache(repo_id, repo_type)
            return
        print(f"Upload attempt {attempt}/{max_retries} failed:")
        if out.strip():
            print(out.strip()[-1000:])
        if err.strip():
            print(err.strip()[-1000:])
        if attempt == max_retries:
            sys.exit(1)
        time.sleep(retry_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror a HuggingFace repo into a self-hosted KohakuHub instance.",
        epilog=(
            "Examples:\n"
            "  ./mirror-model.sh unsloth/Qwen3.6-27B-GGUF --list-quants\n"
            "  ./mirror-model.sh unsloth/Qwen3.6-27B-GGUF --quant UD-Q6_K_XL\n"
            "  ./mirror-model.sh Qwen/Qwen3.6-27B --check\n"
            "  ./mirror-model.sh Qwen/Qwen3.6-27B --size\n"
            "\n"
            "Batch runner for whole queue files (models.txt, models-gguf.txt,\n"
            "datasets.txt): ./mirror-batch.sh --help\n"
            "\n"
            "Full notes on hf_xet quirks and the manifest schema are in the\n"
            "module docstring at the top of mirror.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo_id", help="HF repo id, e.g. Qwen/Qwen3.6-27B or unsloth/Qwen3.6-27B-GGUF")
    parser.add_argument("--type", default="model", choices=["model", "dataset", "space"],
                         help="Repo type on HF/KohakuHub (default: model)")
    parser.add_argument("--force", action="store_true",
                         help="Re-mirror unconditionally, ignoring the manifest and revision checks")
    parser.add_argument("--check", action="store_true",
                         help="Report status only (never-mirrored / up-to-date / update-available); no download")
    parser.add_argument("--update", action="store_true",
                         help="Like --check, but tags the old snapshot and re-mirrors if HF has changed")
    parser.add_argument("--size", action="store_true",
                         help="Report total size only (respects --quant if given); no download")
    parser.add_argument("--include-original", action="store_true",
                         help="Also pull original/*.pth alongside safetensors (excluded by default)")
    parser.add_argument("--quant", default=None, metavar="PATTERN",
                         help="Only pull files matching PATTERN (substring or glob) plus metadata "
                              "(README/config/tokenizer/mmproj). Run again later with a different "
                              "PATTERN to add another quant without re-pulling the first.")
    parser.add_argument("--list-quants", action="store_true",
                         help="List every file in the repo with its size, no download. "
                              "Use this first to find the exact PATTERN for --quant.")
    parser.add_argument("--no-xet", action="store_true",
                         help="Force-disable hf_xet for the DOWNLOAD-from-HF leg only "
                              "(troubleshooting; the upload-to-KohakuHub leg always disables "
                              "xet regardless of this flag)")
    args = parser.parse_args()
    mirror_repo(args.repo_id, repo_type=args.type, force=args.force,
                check_only=args.check, do_update=args.update, size_only=args.size,
                include_original=args.include_original, quant_pattern=args.quant,
                list_quants_only=args.list_quants, no_xet=args.no_xet)
