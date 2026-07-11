#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    FILES=("$SCRIPT_DIR/models.txt" "$SCRIPT_DIR/models-gguf.txt")
fi

REPORT="/tmp/size-report-$$.txt"
> "$REPORT"

for FILE in "${FILES[@]}"; do
    echo "--- Sizing $FILE ---"
    "$SCRIPT_DIR/mirror-batch.sh" "$FILE" --size 2>&1 | tee -a "$REPORT"
done

echo ""
echo "=== Grand total across ${FILES[*]} ==="
grep -oE '^\s*[0-9.]+ (B|KB|MB|GB|TB)' "$REPORT" | awk '
{
    val=$1; unit=$2
    if (unit=="B") bytes=val
    else if (unit=="KB") bytes=val*1024
    else if (unit=="MB") bytes=val*1024*1024
    else if (unit=="GB") bytes=val*1024*1024*1024
    else if (unit=="TB") bytes=val*1024*1024*1024*1024
    sum+=bytes
}
END {
    gb = sum/1024/1024/1024
    tb = gb/1024
    printf "Total: %.1f GB (%.3f TB)\n", gb, tb
}'
rm -f "$REPORT"
