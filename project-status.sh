#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "KohakuHub project status — $(date)"
echo "======================================================================"

echo -e "\n--- Docker containers ---"
docker ps --format "table {{.Names}}\t{{.Status}}"

echo -e "\n--- Confirmed mirrored (manifest) ---"
cat mirrored_repos.json 2>/dev/null || echo "(empty/missing)"

echo -e "\n--- Active queue: models.txt (uncommented lines) ---"
grep -v '^\s*#' models.txt | grep -v '^\s*$' || true

echo -e "\n--- Active queue: models-gguf.txt (uncommented lines) ---"
grep -v '^\s*#' models-gguf.txt | grep -v '^\s*$' || true

echo -e "\n--- Disk usage ---"
du -sh /mnt/leo-storage/KohakuHub/hub-meta/minio-data 2>/dev/null || true
du -sh /mnt/leo-storage/KohakuHub/hf-cache 2>/dev/null || true
df -h /mnt/leo-storage

echo -e "\n--- Total size of active queues (this takes a minute) ---"
./total_size.sh 2>&1 | tail -10
