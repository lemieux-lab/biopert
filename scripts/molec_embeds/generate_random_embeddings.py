import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py_common"))
from dataset_paths import dataset_paths  # noqa: E402
from parquet_io import save_embeds_parquet  # noqa: E402


def smiles_seed(smiles: str, global_seed: int) -> int:
    """Derive a reproducible integer seed from a SMILES string and a global seed."""
    h = hashlib.md5((smiles + str(global_seed)).encode()).hexdigest()
    return int(h, 16) % (2**31)


def random_embedding(smiles: str, n: int, global_seed: int) -> np.ndarray:
    rng = np.random.default_rng(smiles_seed(smiles, global_seed))
    return rng.standard_normal(n).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate random embeddings for a dataset's molecules.")
    parser.add_argument("--outdir", required=True, help="BIOPERT_OUTDIR: base directory for all pipeline data.")
    parser.add_argument("--dataset", required=True, choices=["lincs", "tahoe"],
                        help="Dataset whose SMILES to embed.")
    parser.add_argument("--n", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    args = parser.parse_args()

    paths = dataset_paths(args.outdir, args.dataset)
    df = pd.read_csv(paths["smiles_csv"])
    print(f"Embedding {len(df)} unique molecules...")

    df["embedding"] = [random_embedding(s, args.n, args.seed) for s in df["smiles"]]

    outdir = paths["molec_embeds_dir"] / f"random_{args.n}"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "embeds.parquet"
    save_embeds_parquet(df, outfile)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
