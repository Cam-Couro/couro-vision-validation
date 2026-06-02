#!/bin/bash
# Extract ASPset video archives and delete the .tar.gz files to save 22GB disk
set -euo pipefail
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/aspset510

echo "[$(date '+%H:%M:%S')] Free disk before:"
df -h . | tail -1

for f in aspset510_v1_trainval-videos.tar.gz aspset510_v1_test-videos.tar.gz; do
  if [ -f "$f" ]; then
    SIZE=$(stat -f %z "$f")
    EXPECTED_MIN=$(if [[ "$f" == *trainval* ]]; then echo 22000000000; else echo 1280000000; fi)
    if [ "$SIZE" -lt "$EXPECTED_MIN" ]; then
      echo "[$(date '+%H:%M:%S')] SKIP $f: incomplete ($SIZE bytes)"
      continue
    fi
    echo "[$(date '+%H:%M:%S')] Extracting $f..."
    tar xzf "$f"
    echo "[$(date '+%H:%M:%S')] Deleting $f to free disk..."
    rm "$f"
  fi
done

echo ""
echo "[$(date '+%H:%M:%S')] Video inventory:"
find ASPset-510 -name "*.mkv" -o -name "*.mp4" 2>/dev/null | wc -l
echo "videos extracted"
echo ""
echo "Free disk after:"
df -h . | tail -1
