#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        cat <<'EOF'
Usage: ./mirror-batch.sh [queue_file] [flags...]

Runs every line in a queue file through mirror-model.sh, one repo at a time.
If queue_file is omitted, defaults to models.txt.

Queue file formats (auto-detected per line):
  one repo_id per line              e.g. models.txt
      Qwen/Qwen3.6-27B               # optional trailing comment

  repo_id MIDDLEDOT quant per line  e.g. models-gguf.txt
      unsloth/Qwen3.6-27B-GGUF (middle dot) UD-Q6_K_XL

Blank lines and lines starting with # are skipped entirely. A line with a
middle dot but no quant after it is skipped with a warning (never silently
pulls every quant in the repo).

Any flags you pass are forwarded as-is to mirror-model.sh for EVERY line:
  --type dataset|model|space   set repo type for the whole file
  --check                      dry-run status report, no downloads
  --update                     re-mirror only what changed upstream
  --force                      re-mirror everything, ignore the manifest
  --size                       report sizes only, no downloads
  --include-original           also pull original/*.pth
  --no-xet                     force-disable xet for the download leg

Examples:
  ./mirror-batch.sh                              models.txt, no extra flags
  ./mirror-batch.sh models-gguf.txt               the GGUF quant queue
  ./mirror-batch.sh datasets.txt --type dataset   the dataset queue
  ./mirror-batch.sh models.txt --check            whats new/changed, no downloads
  ./mirror-batch.sh models.txt --size             total size, no downloads

Per-repo flag reference: run mirror-model.sh --help
EOF
        exit 0
    fi
done

MODELS_FILE="$SCRIPT_DIR/models.txt"
EXTRA_ARGS=()

args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    if [[ "$arg" == "--type" ]]; then
        EXTRA_ARGS+=("$arg" "${args[$((i+1))]}")
        i=$((i+2))
    elif [[ "$arg" == --* ]]; then
        EXTRA_ARGS+=("$arg")
        i=$((i+1))
    else
        MODELS_FILE="$arg"
        i=$((i+1))
    fi
done

echo "Using queue file: $MODELS_FILE"

while IFS= read -r raw_line; do
    [[ -z "$raw_line" || "$raw_line" =~ ^[[:space:]]*# ]] && continue
    line="$(echo "$raw_line" | sed 's/#.*//' | xargs)"
    [[ -z "$line" ]] && continue

    if [[ "$line" == *"·"* ]]; then
        repo_id="$(echo "$line" | awk -F'·' '{print $1}' | xargs)"
        quant="$(echo "$line" | awk -F'·' '{print $2}' | xargs)"
        [[ -z "$repo_id" ]] && continue
        echo "=== $repo_id (quant: ${quant:-none-given}) ==="
        if [[ -n "$quant" ]]; then
            "$SCRIPT_DIR/mirror-model.sh" "$repo_id" --quant "$quant" "${EXTRA_ARGS[@]}"
        else
            echo "No quant tag after separator, skipping. Fix the line in $MODELS_FILE."
            continue
        fi
    else
        repo_id="$line"
        echo "=== $repo_id ==="
        "$SCRIPT_DIR/mirror-model.sh" "$repo_id" "${EXTRA_ARGS[@]}"
    fi
    exit_code=$?

    if [ $exit_code -eq 130 ]; then
        echo ""
        echo "Interrupted by Ctrl-C, stopping the whole batch."
        echo "Re-run this same command to pick up from here."
        exit 130
    elif [ $exit_code -ne 0 ]; then
        echo "Failed: $repo_id (exit $exit_code), continuing to next"
    fi
done < "$MODELS_FILE"
