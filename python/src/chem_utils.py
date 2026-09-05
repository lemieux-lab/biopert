import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina


def smiles_to_fingerprints(
    smiles_list: list[str], radius: int = 3, fp_size: int = 2048,
) -> tuple[list[ExplicitBitVect], list[str]]:
    """
    Convert SMILES strings to ECFP fingerprints, skipping invalid entries.

    Args:
        smiles_list: list of SMILES strings
        radius     : Morgan radius (default 3 = ECFP6)
        fp_size    : fingerprint bit size (default 2048)

    Returns:
        fps         : list of RDKit ExplicitBitVect fingerprints
        valid_smiles: SMILES successfully parsed (same order as fps)
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps, valid_smiles = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(mfpgen.GetFingerprint(mol))
            valid_smiles.append(s)
        else:
            print(f"Warning: failed to get fingerprint for SMILES: {s}")
    return fps, valid_smiles


def tanimoto_distance(smiles_i: str, smiles_j: str) -> float | None:
    """
    Compute Tanimoto distance (1 - Tanimoto similarity) between two SMILES strings.
    Returns None if either SMILES is invalid.
    """
    fps, _ = smiles_to_fingerprints([smiles_i, smiles_j])
    if len(fps) < 2:
        return None
    return float(1 - DataStructs.TanimotoSimilarity(fps[0], fps[1]))


def butina_split(
    unique_smiles: list[str],
    cutoff: float = 0.4,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split SMILES into train/val/test using Butina clustering on Tanimoto distance.
    Whole clusters are assigned to a single split to prevent similar SMILES
    from leaking across splits. 

    Args:
        unique_smiles : list of unique SMILES strings
        cutoff        : Tanimoto distance cutoff for clustering (default 0.4)
        val_frac      : fraction of SMILES for validation (default 0.1)
        test_frac     : fraction of SMILES for testing (default 0.1)
        seed          : random seed for shuffling before clustering

    Returns:
        train_smiles, val_smiles, test_smiles
    """
    assert val_frac >= 0
    assert test_frac >= 0
    assert val_frac + test_frac < 1.0
    
    train_frac = 1.0 - val_frac - test_frac

    unique_smiles_copy = unique_smiles.copy()
    random.seed(seed)
    random.shuffle(unique_smiles_copy)

    fps, valid_smiles = smiles_to_fingerprints(unique_smiles_copy)
    n = len(fps)
    if n < len(unique_smiles):
        print(f"Warning: {len(unique_smiles) - n} SMILES could not be parsed and were excluded from the split")

    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    train_idx, val_idx, test_idx = [], [], []
    for cluster in clusters:
        n_seen = len(train_idx) + len(val_idx) + len(test_idx) + len(cluster)
        if len(train_idx) < train_frac * n_seen:
            train_idx.extend(cluster)
        elif len(val_idx) < val_frac * n_seen:
            val_idx.extend(cluster)
        else:
            test_idx.extend(cluster)

    train_smiles = [valid_smiles[i] for i in train_idx]
    val_smiles   = [valid_smiles[i] for i in val_idx]
    test_smiles  = [valid_smiles[i] for i in test_idx]

    print(
        f"Clusters : {len(clusters)}\n"
        f"Train    : {len(train_smiles)} ({100 * len(train_smiles) / n:.1f}%)\n"
        f"Val      : {len(val_smiles)}   ({100 * len(val_smiles)   / n:.1f}%)\n"
        f"Test     : {len(test_smiles)}  ({100 * len(test_smiles)  / n:.1f}%)"
    )
    return train_smiles, val_smiles, test_smiles


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    """
    Load embeddings from a parquet file with 'smiles' and 'embedding' columns.
    The 'embedding' column contains raw float32 bytes.
    Returns a dict mapping smiles -> float32 embedding array.
    """
    df = pd.read_parquet(path)
    return {row.smiles: np.frombuffer(row.embedding, dtype=np.float32) for _, row in df.iterrows()}