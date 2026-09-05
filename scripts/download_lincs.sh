#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <outdir>"
  exit 1
fi

LINCS_DIR="$1/LINCS_beta"
mkdir -p "$LINCS_DIR"
cd "$LINCS_DIR"

wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_all_n3026460x12328.gctx

wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt

wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/README.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/"LINCS2020 Release Metadata Field Definitions.xlsx"

echo "LINCS Beta Level 3 has been downloaded successfully to:"
echo "  $LINCS_DIR"
