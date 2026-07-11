# KohakuHub Mirror Tools

Mirrors HuggingFace repos (or just specific GGUF quants) into our
self-hosted KohakuHub instance.

## One-time setup

```
./venv/bin/python3 -m ensurepip --upgrade   # only if venv has no pip
./venv/bin/python3 -m pip install hf_xet
```

## Everyday use

**Look at a repo before downloading anything:**
```
./mirror-model.sh unsloth/Qwen3.6-27B-GGUF --list-quants
```

**Pull one quant** (repeat later with a different tag to add another, without re-pulling the first):
```
./mirror-model.sh unsloth/Qwen3.6-27B-GGUF --quant UD-Q6_K_XL
```

**Pull a whole (non-GGUF) repo:**
```
./mirror-model.sh Qwen/Qwen3.6-27B
```

**Run everything in the queue files, in the background:**
```
nohup bash -c './mirror-batch.sh models.txt && ./mirror-batch.sh models-gguf.txt && ./mirror-batch.sh datasets.txt --type dataset' > mirror-run.log 2>&1 &
tail -f mirror-run.log
```
Don't run `models-non-v100.txt` — that one's parked on purpose, nothing in
it should be downloaded.

**Check status / progress:**
```
./project-status.sh          # containers, disk, manifest, active queue
./check-quants.sh            # do models.txt, models-gguf.txt, datasets.txt, the
                              # manifest, and KohakuHub all agree?
./check-quants.sh --fix      # same check, then correct the manifest to match
                              # KohakuHub (backs up mirrored_repos.json first)
./check-quants.sh --discover # also flag repos sitting on KohakuHub that aren't
                              # in any queue file (best-effort — depends on your
                              # KohakuHub instance supporting author-based
                              # repo listing). Output is long, so it's auto-saved
                              # to a timestamped report file as well as printed.
./check-quants.sh -o out.txt # save any run's full output to a file you name
```

Note: KohakuHub has an "external source fallback" feature that transparently
proxies HuggingFace.co for any repo not stored locally — its Models/Datasets
browse pages will show far more repos than you've actually mirrored. `check-quants.sh`
talks to KohakuHub directly with `?fallback=false` so it isn't fooled by this;
plain browsing the web UI can still be misleading about what's really local.

## Queue files

| File | What it is |
|---|---|
| `models.txt` | Full repos to mirror as-is (safetensors/AWQ) |
| `models-gguf.txt` | One quant per line: `repo_id · quant_tag` |
| `models-non-v100.txt` | Parked — do NOT mirror, kept for reference only |
| `datasets.txt` | Datasets — run with `--type dataset` |

Lines starting with `#` (including whole commented-out Tier 4 entries) are
skipped automatically.

## Scripts

| Script | Purpose |
|---|---|
| `mirror-model.sh` | Mirror one repo (or one quant of one repo) |
| `mirror-batch.sh` | Run every line in a queue file through `mirror-model.sh` |
| `check-quants.sh` | Reconcile wanted (all 3 queue files) vs. manifest vs. actual KohakuHub contents. `--fix` corrects the manifest, `--discover` looks for untracked repos and auto-saves a report, `--output`/`-o` saves any run to a file you name. |
| `check_cache.py` | Flag local downloads that aren't in the manifest (stuck/failed) |
| `backfill-metadata.sh` | Re-fetch README/LICENSE/etc. for already-mirrored repos |
| `total_size.sh` | Total size of a queue file's contents |
| `project-status.sh` | One-shot health check: containers, disk, manifest, queues |

## More detail

Every script above (except the no-argument ones) has a full `--help`:
```
./mirror-model.sh --help
./mirror-batch.sh --help
./check-quants.sh --help
```
