import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import MolFromSmiles


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py_common"))
from dataset_paths import dataset_paths  # noqa: E402
from parquet_io import save_embeds_parquet  # noqa: E402


ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def load_model(device: torch.device, ckpt_path: Path) -> MPNN:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        print(f"Downloading CheMeleon weights to {ckpt_path} ...")
        urlretrieve(ZENODO_URL, ckpt_path)
    chemeleon_mp = torch.load(ckpt_path, weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
    mp.load_state_dict(chemeleon_mp["state_dict"])
    model = MPNN(
        message_passing=mp,
        agg=nn.MeanAggregation(),
        predictor=RegressionFFN(input_dim=mp.output_dim),
    )
    model.eval()
    model.to(device)
    return model


def embed(model: MPNN, smiles_list: list[str], batch_size: int, device: torch.device) -> np.ndarray:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        mols = [MolFromSmiles(s) for s in batch_smiles]
        if any(m is None for m in mols):
            bad = [s for s, m in zip(batch_smiles, mols) if m is None]
            raise ValueError(f"Invalid SMILES: {bad}")
        bmg = BatchMolGraph([featurizer(m) for m in mols])
        bmg.to(device=device)
        with torch.no_grad():
            embeddings.append(model.fingerprint(bmg).numpy(force=True))
    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compute CheMeleon embeddings from SMILES.")
    parser.add_argument("--outdir", required=True, help="BIOPERT_OUTDIR: base directory for all pipeline data.")
    parser.add_argument("--dataset", required=True, choices=["lincs", "tahoe"],
                        help="Dataset whose SMILES to embed.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt_path", type=str, default=str(Path.home() / ".chemprop" / "chemeleon_mp.pt"),
                        help="Path to the CheMeleon checkpoint; downloaded here if missing.")
    args = parser.parse_args()

    paths = dataset_paths(args.outdir, args.dataset)
    df = pd.read_csv(paths["smiles_csv"])
    print(f"Embedding {len(df)} unique molecules...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device, Path(args.ckpt_path))
    vecs = embed(model, df["smiles"].tolist(), args.batch_size, device)
    print(f"Embedding shape: {vecs.shape}")

    out = pd.DataFrame({
        "drug": df["drug"].values,
        "smiles": df["smiles"].values,
        "embedding": list(vecs),
    })

    model_outdir = paths["molec_embeds_dir"] / "chemeleon"
    model_outdir.mkdir(parents=True, exist_ok=True)
    save_embeds_parquet(out, model_outdir / "embeds.parquet")
    print(f"Saved to {model_outdir / 'embeds.parquet'}")


if __name__ == "__main__":
    main()
