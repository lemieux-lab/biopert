#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <outdir>"
  exit 1
fi

command -v hf >/dev/null || { echo "hf CLI not found — run 'conda activate biopert' first"; exit 1; }

TAHOE_DIR="$1/Tahoe-100M"
mkdir -p "$TAHOE_DIR"

HF_XET_HIGH_PERFORMANCE=1 hf download tahoebio/Tahoe-100M \
    --repo-type dataset \
    --local-dir "$TAHOE_DIR" \
    --token "$HF_TOKEN"

echo "Tahoe-100M has been downloaded successfully to:"
echo "  $TAHOE_DIR"