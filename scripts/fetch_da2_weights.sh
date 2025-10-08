#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints
FILE="checkpoints/depth_anything_v2_vits.pth"
URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"

if [ ! -f "$FILE" ]; then
  echo "Fetching Depth Anything v2 Small weights..."
  curl -L --retry 5 --retry-delay 3 -o "$FILE" "$URL"
  ls -lh "$FILE"
fi
