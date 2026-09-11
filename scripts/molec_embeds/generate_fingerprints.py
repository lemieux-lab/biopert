import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py_common"))
from dataset_paths import dataset_paths  # noqa: E402
from parquet_io import save_embeds_parquet  # noqa: E402


def generate_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    valid_rows = []
    maccs_fps = []
    ecfp6_fps = {512: [], 1024: [], 2048: []}
    rdkit_fps = {512: [], 1024: [], 2048: []}
    atompair_fps = {512: [], 1024: [], 2048: []}

    print("Generating fingerprints...")

    mol_list = []
    for index, row in df.iterrows():
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)

        # Handle invalid SMILES
        if mol is None:
            print(f"Warning: Invalid SMILES at row {index}: {smiles}")
            continue

        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_fps.append(fp_maccs.ToBitString())
        mol_list.append(mol)
        valid_rows.append(index)

    for nBits in (512, 1024, 2048):
        ecfp6_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=nBits)
        rdk_gen   = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nBits)
        ap_gen    = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)

        for mol in mol_list:
            ecfp6_fps[nBits].append(ecfp6_gen.GetFingerprint(mol).ToBitString())
            rdkit_fps[nBits].append(rdk_gen.GetFingerprint(mol).ToBitString())
            atompair_fps[nBits].append(ap_gen.GetFingerprint(mol).ToBitString())

    df = df.loc[valid_rows, :].copy()
    df["MACCS_166"] = maccs_fps
    for nBits in (512, 1024, 2048):
        df[f"ECFP6_{nBits}"] = ecfp6_fps[nBits]
        df[f"RDKit_{nBits}"] = rdkit_fps[nBits]
        df[f"AtomPair_{nBits}"] = atompair_fps[nBits]

    return df


def fingerprint_columns() -> list[str]:
    return ["MACCS_166"] + [
        f"{kind}_{nBits}"
        for nBits in (512, 1024, 2048)
        for kind in ("ECFP6", "RDKit", "AtomPair")
    ]


def main():
    parser = argparse.ArgumentParser(description="Compute fingerprints of molecules with rdkit.")
    parser.add_argument("--outdir", required=True, help="BIOPERT_OUTDIR: base directory for all pipeline data.")
    parser.add_argument("--dataset", required=True, choices=["lincs", "tahoe"],
                        help="Dataset whose SMILES to fingerprint.")
    args = parser.parse_args()

    paths = dataset_paths(args.outdir, args.dataset)
    df = pd.read_csv(paths["smiles_csv"])
    print(f"Fingerprinting {len(df)} unique molecules...")

    df = generate_fingerprints(df)

    for col in fingerprint_columns():
        embeds = df[["drug", "smiles"]].copy()
        embeds["embedding"] = df[col].apply(lambda x: np.array([np.float32(e) for e in x]))

        model_outdir = paths["molec_embeds_dir"] / col
        model_outdir.mkdir(parents=True, exist_ok=True)
        outfile = model_outdir / "embeds.parquet"
        save_embeds_parquet(embeds, outfile)
        print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
