#!/bin/bash

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <outdir> <dataset (lincs|tahoe)>"
  exit 1
fi

OUTDIR="$1"
DATASET="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generate fingerprints using RDKit ..."
python "$SCRIPT_DIR/generate_fingerprints.py" \
  --outdir "$OUTDIR" \
  --dataset "$DATASET"

echo "Computing CheMeleon embeddings ..."
python "$SCRIPT_DIR/get_chemeleon_embeddings.py" \
  --outdir "$OUTDIR" \
  --dataset "$DATASET"

MODELS=(
  "DeepChem/ChemBERTa-77M-MLM"
  "ibm/MoLFormer-XL-both-10pct"
  "zjunlp/MolGen-large"
  "unikei/bert-base-smiles"
  "unimolv1/84m"
)

for MODEL in "${MODELS[@]}"; do
  echo "Computing embeddings for $MODEL ..."
  python "$SCRIPT_DIR/get_pretrained_molec_embeddings.py" \
    --pretrained_name_or_path "$MODEL" \
    --outdir "$OUTDIR" \
    --dataset "$DATASET"
done

echo "Computing random baseline embeddings ..."
python "$SCRIPT_DIR/generate_random_embeddings.py" \
  --outdir "$OUTDIR" \
  --dataset "$DATASET"
