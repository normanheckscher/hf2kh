#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

source "$SCRIPT_DIR/.mirror_env"

REPO_ID="${1:?Usage: ./mirror-model.sh <namespace/repo> [--type dataset|space]}"
shift

BW_DOWN="${MIRROR_BW_DOWN_KBPS:-0}"
BW_UP="${MIRROR_BW_UP_KBPS:-0}"

TRICKLE_ARGS=()
[ "$BW_DOWN" -gt 0 ] && TRICKLE_ARGS+=(-d "$BW_DOWN")
[ "$BW_UP" -gt 0 ] && TRICKLE_ARGS+=(-u "$BW_UP")

if [ ${#TRICKLE_ARGS[@]} -gt 0 ]; then
    trickle "${TRICKLE_ARGS[@]}" "$VENV_PYTHON" "$SCRIPT_DIR/mirror.py" "$REPO_ID" "$@"
else
    "$VENV_PYTHON" "$SCRIPT_DIR/mirror.py" "$REPO_ID" "$@"
fi
