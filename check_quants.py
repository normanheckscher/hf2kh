#!/usr/bin/env python3
"""
Reconcile "what do we WANT" (queue files: models.txt, models-gguf.txt,
datasets.txt) against "what does the manifest THINK we have"
(mirrored_repos.json) against "what's ACTUALLY sitting in KohakuHub"
(live query). Three separate sources on purpose — the manifest is local
bookkeeping written by mirror.py and can drift from reality (a
partially-failed upload, a manual delete on the KohakuHub side, editing
the manifest by hand, etc.). Don't trust it as ground truth by itself.

IMPORTANT — why this doesn't use huggingface_hub for the Kohaku side:
KohakuHub has an "External Source Fallback" feature — when a repo isn't
stored locally, it transparently proxies the listing/metadata from real
HuggingFace.co and serves it as a normal-looking response. The standard
huggingface_hub client (list_repo_files/repo_info) has no way to disable
this, so it can't tell "genuinely mirrored here" apart from "KohakuHub is
just relaying what's real on HuggingFace." That made every earlier version
of this script report false DRIFT/OK for repos that were never actually
mirrored (confirmed against a live instance — see probe_fallback.py).
The fix: talk to KohakuHub's tree/repo-info endpoints directly over HTTP
with `?fallback=false` appended, which forces a genuine 404 if the repo
isn't really stored on this instance.

Handles all three queue file formats:
  - models-gguf.txt : "RepoID · quant  # notes"  (quant-selective check)
  - models.txt / datasets.txt : "org/repo  # notes"  (full-repo check)
Repo type (model vs dataset) is inferred from the filename ("dataset" in
the name -> dataset, else model) — override with --type if a file doesn't
follow that convention.

Verdicts:
    OK       — manifest and KohakuHub agree the thing is present
    MISSING  — neither manifest nor KohakuHub has it; needs mirroring
    PARTIAL  — present, but only as a quant-selective mirror where a full
               mirror was expected (models.txt / datasets.txt entries)
    DRIFT    — manifest and KohakuHub DISAGREE (the case this script exists
               to catch — one of your two records of "what we have" is wrong)
    UNKNOWN  — couldn't reach KohakuHub to verify (network/auth issue) —
               falls back to manifest-only, flagged so you know it's unverified

--fix rewrites mirrored_repos.json to match KohakuHub (treated as ground
truth over the manifest): DRIFT rows where Kohaku has it get added/updated
in the manifest; DRIFT rows where Kohaku does NOT have it get removed from
the manifest. OK/PARTIAL/UNKNOWN rows are left untouched. A timestamped
backup of the old manifest is written before any change.

--discover tries to enumerate every repo actually sitting on KohakuHub
under the namespaces your queue files reference (also with fallback=false),
and flags any repo that exists on the hub but isn't listed in any queue
file at all (manual uploads, leftovers, etc.). Repos listed (even
commented-out) in models-non-v100.txt are treated as accounted-for too,
since that file is intentionally excluded from mirroring, not forgotten.
This depends on KohakuHub's author-filtered list endpoint responding the
same way to fallback=false as the tree/repo-info endpoints do — plausible
given it's documented as a generic query param, but only directly
confirmed for tree/repo-info so far. If listing isn't usable, --discover
says so and exits cleanly. Because a --discover run (plus the normal
per-repo table) can be a lot of terminal output, it is ALWAYS also written
to a timestamped report file automatically — pass --output to control
where.

--output/-o writes a full copy of everything this run prints to the given
path (in addition to printing it normally). If you pass --discover without
--output, a report file is created for you automatically (printed at the
end) so you don't lose the output.

Usage:
    ./venv/bin/python3 check_quants.py                      # check the 3 default queue files
    ./venv/bin/python3 check_quants.py models.txt            # check just one file
    ./venv/bin/python3 check_quants.py --fix                 # also correct the manifest
    ./venv/bin/python3 check_quants.py --discover            # also look for untracked repos on Kohaku (auto-saves a report)
    ./venv/bin/python3 check_quants.py --output report.txt   # save this run's full output to a file
    ./venv/bin/python3 check_quants.py datasets.txt --type dataset
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import socket
import sys
import time

import httpx


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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "mirrored_repos.json")
DEFAULT_QUEUE_FILES = ["models.txt", "models-gguf.txt", "datasets.txt"]
# Not mirrored (on purpose — FP8/NVFP4/MLX/etc. aren't V100-viable), but
# still "accounted for" — --discover shouldn't call these orphans just
# because they're outside the 3 files that actually get reconciled.
CONTEXT_ONLY_FILES = ["models-non-v100.txt"]

REPO_TYPE_PLURAL = {"model": "models", "dataset": "datasets", "space": "spaces"}


class Tee:
    """Duplicates everything written to stdout into a file as well, so a
    run's full output can be saved without losing the live terminal view."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _kohaku_headers():
    return {"Authorization": f"Bearer {KOHAKU_TOKEN}"} if KOHAKU_TOKEN else {}


def infer_repo_type(path):
    return "dataset" if "dataset" in os.path.basename(path).lower() else "model"


def parse_queue(path, repo_type):
    """Returns (active, archived) lists of dicts:
    {repo_id, quant (or None for full-repo lines), repo_type}
    Comment (#) lines are "archived" — reported separately, not silently
    skipped, so you know they exist but aren't expected to be mirrored yet."""
    active, archived = [], []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            is_comment = bool(re.match(r"^\s*#", line))
            if is_comment:
                # Strip the leading "#" that marks the whole line as
                # archived, THEN strip any trailing "# note" the same way
                # active lines get theirs stripped. Without this second
                # pass, an archived line like "# org/repo   # some note"
                # kept the trailing note text (and its spaces) attached,
                # which made the "no spaces allowed" sanity check below
                # reject it — silently dropping it from `archived` instead
                # of recording it, so --discover would flag a repo that's
                # actually already listed (just commented) as a false
                # "orphan".
                stripped = re.sub(r"^\s*#\s?", "", line)
                stripped = re.sub(r"#.*$", "", stripped).strip()
            else:
                stripped = re.sub(r"#.*$", "", line).strip()
            if not stripped:
                continue

            if "·" in stripped:
                parts = [p.strip() for p in stripped.split("·", 1)]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                repo_id, quant = parts[0], parts[1].split()[0] if parts[1] else ""
                if not quant or "/" not in repo_id or " " in repo_id:
                    continue
            else:
                # plain "org/repo" line — only treat as a real entry if it
                # actually looks like one (avoids false positives on stray
                # comment text with a slash in it)
                if "/" not in stripped or " " in stripped:
                    continue
                repo_id, quant = stripped, None

            row = {"repo_id": repo_id, "quant": quant, "repo_type": repo_type}
            (archived if is_comment else active).append(row)
    return active, archived


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    if os.path.exists(MANIFEST_PATH):
        backup = MANIFEST_PATH + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(MANIFEST_PATH, backup)
        print(f"  (backed up old manifest to {os.path.basename(backup)})")
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, MANIFEST_PATH)


def manifest_key(repo_type, repo_id):
    return f"{repo_type}:{repo_id}"


def entry_mode(entry):
    return entry.get("mode", "full")


def manifest_status(manifest, repo_type, repo_id, quant):
    entry = manifest.get(manifest_key(repo_type, repo_id))
    if not entry:
        return "missing", None
    mode = entry_mode(entry)
    if quant is None:
        # full-repo line: anything in the manifest counts as present, but
        # flag it if it's only a quant-selective mirror, not a real full one
        return "have", ("partial" if mode != "full" else None)
    if mode == "full":
        return "have", None
    return ("have" if quant in (entry.get("quants_present") or []) else "missing"), None


_kohaku_file_cache = {}


def kohaku_files(repo_id, repo_type):
    """Direct HTTP call to KohakuHub's tree endpoint with fallback=false —
    see the module docstring for why this can't go through huggingface_hub.
    Confirmed against a live instance: fallback=false still returns 200
    with the real file list for a genuinely-mirrored repo, and a clean 404
    (X-Error-Code: RepoNotFound) for one that was never mirrored, even
    though that same repo is real on HuggingFace and would 200 without the
    flag. Tree entries use a "path" key (not "rfilename") and a "type" key
    ("file" vs "directory")."""
    cache_key = (repo_type, repo_id)
    if cache_key in _kohaku_file_cache:
        return _kohaku_file_cache[cache_key]
    plural = REPO_TYPE_PLURAL.get(repo_type, repo_type + "s")
    url = f"{KOHAKU_ENDPOINT}/api/{plural}/{repo_id}/tree/main?recursive=true&fallback=false"
    try:
        resp = httpx.get(url, headers=_kohaku_headers(), timeout=20)
    except Exception as e:
        _kohaku_file_cache[cache_key] = ("error", str(e))
        return _kohaku_file_cache[cache_key]

    if resp.status_code == 404:
        _kohaku_file_cache[cache_key] = ("not-found", [])
    elif resp.status_code == 200:
        try:
            data = resp.json()
            files = [item["path"] for item in data
                     if isinstance(item, dict) and item.get("type") == "file" and item.get("path")]
            _kohaku_file_cache[cache_key] = ("ok", files)
        except Exception as e:
            _kohaku_file_cache[cache_key] = ("error", f"couldn't parse tree response: {e}")
    else:
        _kohaku_file_cache[cache_key] = ("error", f"HTTP {resp.status_code}")
    return _kohaku_file_cache[cache_key]


def kohaku_status(repo_id, repo_type, quant):
    status, payload = kohaku_files(repo_id, repo_type)
    if status == "not-found":
        return "missing"
    if status == "error":
        return "unknown"
    if quant is None:
        return "have" if payload else "missing"
    glob_pattern = quant if any(c in quant for c in "*?[") else f"*{quant}*"
    match = any(fnmatch.fnmatch(f.lower(), glob_pattern.lower()) for f in payload)
    return "have" if match else "missing"


def get_kohaku_revision(repo_id, repo_type):
    plural = REPO_TYPE_PLURAL.get(repo_type, repo_type + "s")
    url = f"{KOHAKU_ENDPOINT}/api/{plural}/{repo_id}?fallback=false"
    try:
        resp = httpx.get(url, headers=_kohaku_headers(), timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json().get("sha")
    except Exception:
        return None


def reconcile(manifest, active):
    rows = []
    for row in active:
        repo_id, quant, repo_type = row["repo_id"], row["quant"], row["repo_type"]
        m, note = manifest_status(manifest, repo_type, repo_id, quant)
        k = kohaku_status(repo_id, repo_type, quant)
        if k == "unknown":
            verdict = "UNKNOWN"
        elif m == k:
            if m == "have":
                verdict = "PARTIAL" if note == "partial" else "OK"
            else:
                verdict = "MISSING"
        else:
            verdict = "DRIFT"
        rows.append({"verdict": verdict, "repo_id": repo_id, "quant": quant,
                      "repo_type": repo_type, "manifest": m, "kohaku": k})
    return rows


def apply_fix(manifest, rows):
    changed = 0
    for r in rows:
        if r["verdict"] != "DRIFT":
            continue
        key = manifest_key(r["repo_type"], r["repo_id"])
        if r["kohaku"] == "have":
            # Kohaku has it, manifest doesn't (correctly) reflect that — add/update.
            entry = manifest.get(key, {})
            revision = get_kohaku_revision(r["repo_id"], r["repo_type"]) or entry.get("revision", "unknown")
            if r["quant"] is None:
                entry["mode"] = "full"
                entry["quants_present"] = None
            else:
                if entry_mode(entry) != "full":
                    entry["mode"] = "quant-selective"
                    present = set(entry.get("quants_present") or [])
                    present.add(r["quant"])
                    entry["quants_present"] = sorted(present)
            entry["revision"] = revision
            entry["mirrored_at"] = entry.get("mirrored_at", time.strftime("%Y-%m-%d %H:%M:%S") + " (reconciled from KohakuHub, no prior local record)")
            entry.setdefault("history", [])
            entry.setdefault("files", [])
            manifest[key] = entry
            changed += 1
            print(f"  fixed: {key} -> now recorded as present"
                  + (f" (quant {r['quant']})" if r["quant"] else " (full)"))
        else:
            # Manifest claims we have it, Kohaku says no — manifest is stale.
            entry = manifest.get(key)
            if not entry:
                continue
            if r["quant"] is None or entry_mode(entry) == "full":
                del manifest[key]
                changed += 1
                print(f"  fixed: {key} -> removed (not actually on KohakuHub)")
            else:
                present = set(entry.get("quants_present") or [])
                present.discard(r["quant"])
                if present:
                    entry["quants_present"] = sorted(present)
                    manifest[key] = entry
                else:
                    del manifest[key]
                changed += 1
                print(f"  fixed: {key} -> removed quant '{r['quant']}' (not actually on KohakuHub)")
    return changed


def discover_orphans(known_repo_ids):
    namespaces = sorted({rid.split("/")[0] for rid in known_repo_ids if "/" in rid})
    print(f"\n--discover: probing {len(namespaces)} namespace(s) on KohakuHub for repos "
          f"not listed in any queue file...")
    print("(Uses the same fallback=false trick as the main check. Confirmed working for "
          "single-repo lookups; the author-filtered list endpoint is assumed to honor it "
          "the same way but hasn't been separately confirmed.)")

    found_any_support = False
    orphans = []
    for repo_type in ("model", "dataset"):
        plural = REPO_TYPE_PLURAL[repo_type]
        for ns in namespaces:
            url = f"{KOHAKU_ENDPOINT}/api/{plural}?author={ns}&fallback=false"
            try:
                resp = httpx.get(url, headers=_kohaku_headers(), timeout=20)
            except Exception as e:
                print(f"  GET {plural}?author={ns} failed: {e}")
                continue
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                print(f"  GET {plural}?author={ns} -> HTTP {resp.status_code}")
                continue
            try:
                data = resp.json()
                results = data if isinstance(data, list) else data.get("models") or data.get("datasets") or []
            except Exception as e:
                print(f"  couldn't parse listing for {plural}?author={ns}: {e}")
                continue
            found_any_support = True
            for r in results:
                if not isinstance(r, dict):
                    continue
                rid = r.get("id") or r.get("modelId") or r.get("full_id")
                if rid and rid not in known_repo_ids:
                    orphans.append((repo_type, rid))

    if not found_any_support:
        print("  Could not list repos by author on this KohakuHub instance (or none of your "
              "namespaces returned anything) — --discover isn't usable here. Falling back to "
              "per-queue-file checks only (already done above).")
        return

    if orphans:
        print(f"\n  {len(orphans)} repo(s) on KohakuHub NOT referenced in any queue file:")
        for repo_type, rid in sorted(set(orphans)):
            print(f"    ({repo_type}) {rid}")
    else:
        print("  No orphan repos found — everything on KohakuHub is accounted for "
              "in a queue file.")


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile queue files vs. mirrored_repos.json vs. live KohakuHub state.",
        epilog=(
            "Examples:\n"
            "  ./check-quants.sh                          # check all 3 default queue files\n"
            "  ./check-quants.sh models-gguf.txt           # check just one\n"
            "  ./check-quants.sh --fix                     # also correct the manifest to match Kohaku\n"
            "  ./check-quants.sh --discover                # also look for untracked repos on Kohaku (auto-saves a report)\n"
            "  ./check-quants.sh --output report.txt       # save this run's full output to a file\n"
            "\n"
            "Verdicts: OK, MISSING, PARTIAL (full mirror expected, only a quant-selective\n"
            "one found), DRIFT (manifest and Kohaku disagree — investigate first), UNKNOWN\n"
            "(couldn't reach Kohaku to verify).\n"
            "\n"
            "Kohaku-side checks use ?fallback=false directly over HTTP (not huggingface_hub)\n"
            "to avoid KohakuHub's external-source fallback reporting repos as 'present'\n"
            "just because they're real on HuggingFace, even if never mirrored here.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("queue_files", nargs="*", default=None,
                         help="Queue file(s) to check (default: models.txt, models-gguf.txt, "
                              "datasets.txt in this script's directory)")
    parser.add_argument("--type", default=None, choices=["model", "dataset"],
                         help="Force repo type for ALL files passed, instead of inferring per "
                              "file from the filename (files with 'dataset' in the name default "
                              "to dataset, everything else defaults to model)")
    parser.add_argument("--fix", action="store_true",
                         help="Rewrite mirrored_repos.json so DRIFT rows match KohakuHub "
                              "(treated as ground truth). A timestamped backup is written first. "
                              "OK/PARTIAL/UNKNOWN rows are never touched.")
    parser.add_argument("--discover", action="store_true",
                         help="Also try to enumerate every repo on KohakuHub under the "
                              "namespaces your queue files reference, and flag any repo that "
                              "exists on the hub but isn't in any queue file. Best-effort — "
                              "depends on your KohakuHub instance supporting author-based repo "
                              "listing, which may not be the case. Because this run's output "
                              "can get long, it's auto-saved to a report file if --output isn't given.")
    parser.add_argument("--output", "-o", default=None, metavar="PATH",
                         help="Also write this run's full output to PATH (in addition to "
                              "printing it). If --discover is passed without --output, a "
                              "timestamped report file is created automatically.")
    args = parser.parse_args()

    output_path = args.output
    if not output_path and args.discover:
        output_path = os.path.join(SCRIPT_DIR, f"discover-report-{time.strftime('%Y%m%d-%H%M%S')}.txt")

    report_file = None
    real_stdout = sys.stdout
    if output_path:
        report_file = open(output_path, "w")
        sys.stdout = Tee(real_stdout, report_file)

    try:
        queue_paths = args.queue_files or [os.path.join(SCRIPT_DIR, f) for f in DEFAULT_QUEUE_FILES]
        queue_paths = [p if os.path.isabs(p) or os.path.exists(p) else os.path.join(SCRIPT_DIR, p)
                       for p in queue_paths]

        manifest = load_manifest()

        all_rows = []
        all_archived = []
        known_repo_ids = set()

        for path in queue_paths:
            if not os.path.exists(path):
                print(f"Skipping {path} — file not found.")
                continue
            repo_type = args.type or infer_repo_type(path)
            active, archived = parse_queue(path, repo_type)
            known_repo_ids.update(r["repo_id"] for r in active + archived)
            print(f"Checking {len(active)} active entr{'y' if len(active)==1 else 'ies'} "
                  f"from {os.path.basename(path)} (type={repo_type}, "
                  f"{len(archived)} archived/commented entries skipped for now)...")
            rows = reconcile(manifest, active)
            all_rows.extend(rows)
            all_archived.extend((os.path.basename(path), r) for r in archived)

        if args.discover:
            for fname in CONTEXT_ONLY_FILES:
                ctx_path = os.path.join(SCRIPT_DIR, fname)
                if os.path.exists(ctx_path):
                    ctx_type = infer_repo_type(ctx_path)
                    ctx_active, ctx_archived = parse_queue(ctx_path, ctx_type)
                    known_repo_ids.update(r["repo_id"] for r in ctx_active + ctx_archived)

        order = {"DRIFT": 0, "UNKNOWN": 1, "MISSING": 2, "PARTIAL": 3, "OK": 4}
        all_rows.sort(key=lambda r: order[r["verdict"]])

        icon = {"OK": "✅", "MISSING": "⬜", "PARTIAL": "\U0001F7E1",
                "DRIFT": "⚠️ ", "UNKNOWN": "❓"}

        print()
        for r in all_rows:
            quant_disp = r["quant"] or "(full repo)"
            print(f"{icon[r['verdict']]} {r['verdict']:8} {r['repo_id']:55} {quant_disp:20} "
                  f"manifest={r['manifest']:8} kohaku={r['kohaku']}")

        n_drift = sum(1 for r in all_rows if r["verdict"] == "DRIFT")
        n_missing = sum(1 for r in all_rows if r["verdict"] == "MISSING")
        n_partial = sum(1 for r in all_rows if r["verdict"] == "PARTIAL")
        n_unknown = sum(1 for r in all_rows if r["verdict"] == "UNKNOWN")
        print(f"\n{len(all_rows)} checked across {len(queue_paths)} file(s) — {n_missing} missing, "
              f"{n_partial} partial-only, {n_drift} DRIFT (manifest and KohakuHub disagree — "
              f"investigate these first), {n_unknown} unverifiable this run.")

        if all_archived:
            print(f"\n--- {len(all_archived)} archived/commented entries across all files (not "
                  f"expected to be mirrored) ---")
            for fname, r in all_archived:
                quant_disp = f" · {r['quant']}" if r["quant"] else ""
                print(f"   [{fname}] # {r['repo_id']}{quant_disp}")

        if args.fix:
            print("\n--fix: reconciling manifest to match KohakuHub...")
            changed = apply_fix(manifest, all_rows)
            if changed:
                save_manifest(manifest)
                print(f"  {changed} manifest entr{'y' if changed==1 else 'ies'} corrected.")
            else:
                print("  Nothing to fix — no DRIFT rows.")

        if args.discover:
            discover_orphans(known_repo_ids)
    finally:
        if report_file:
            sys.stdout = real_stdout
            report_file.close()
            print(f"\nFull output saved to: {output_path}")


if __name__ == "__main__":
    main()
