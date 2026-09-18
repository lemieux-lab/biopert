"""
Preprocessing helpers for results.ipynb.

Public API
----------
load_results(biopert_outdir, lincs_beta_dir=None,
             force_recompute=False, use_cache=False)
    Returns a LoadedResults dataclass with all tables ready for analysis.
configure(biopert_outdir, lincs_beta_dir=None)
    Sets the module-level paths (BASELINES_DIR, LINCS_CELLINFO_PATH,
    REPRO_CSV, ...) without loading any results. Called automatically by
    load_results(); call directly if you need those paths beforehand (e.g.
    build_run_cache.py, which never calls load_results() itself).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py_common"))
from dataset_paths import dataset_paths

# ── Paths ─────────────────────────────────────────────────────────────────────

BIOPERT_OUTDIR: str | None = None
LINCS_BETA_DIR: str | None = None
BASELINES_DIR: str | None = None
BASELINES_CELLLINE_DIR: str | None = None
LINCS_CELLINFO_PATH: str | None = None
REPRO_CSV: dict[str, str] = {}


def repro_csv_path(dataset: str, name: str = "repro_delta_inter") -> str:
    paths = dataset_paths(BIOPERT_OUTDIR, dataset)
    jld2_stem = Path(paths["jld2_path"]).stem
    return str(paths["repro_dir"] / f"repro_{jld2_stem}" / f"{name}.csv")


def configure(biopert_outdir: str, lincs_beta_dir: str | None = None) -> None:
    global BIOPERT_OUTDIR, LINCS_BETA_DIR
    global BASELINES_DIR, BASELINES_CELLLINE_DIR, LINCS_CELLINFO_PATH
    BIOPERT_OUTDIR = biopert_outdir
    LINCS_BETA_DIR = lincs_beta_dir
    BASELINES_DIR = os.path.join(BIOPERT_OUTDIR, "baselines")
    BASELINES_CELLLINE_DIR = os.path.join(BIOPERT_OUTDIR, "baselines_cellline")
    LINCS_CELLINFO_PATH = os.path.join(lincs_beta_dir, "cellinfo_beta.txt") if lincs_beta_dir else None
    REPRO_CSV.clear()
    REPRO_CSV.update({ds: repro_csv_path(ds) for ds in ("lincs", "tahoe")})


MAX_TRIALS_PER_EXP = 50  # sweep run_cap; enforced so no exp_key exceeds it after backfill

_RUN_STATS_CACHE = "cache_run_stats.parquet"
_CACHE_KEY_FILE = "cache_key.txt"
_PER_RUN_CACHE_FILE = "run_stats_cache.parquet"  # written inside each run dir

# ── Experiment metadata ───────────────────────────────────────────────────────

# exp_key -> (dataset, category, display_label)
# category: ablation | ref_cell | ref_handling | embedding | repro | baseline | holdout | dose_encoding
EXPERIMENT_META: dict[str, tuple[str, str, str]] = {
    # Tahoe
    "tahoe_default_config": ("Tahoe", "ablation", "default"),
    "tahoe_cellline_holdout": ("Tahoe", "holdout", "cell-line holdout"),
    "tahoe_delta_ref_only": ("Tahoe", "ablation", "delta_ref_only"),
    "tahoe_random_512": ("Tahoe", "ablation", "random_512"),
    "tahoe_default_landmark_genes": ("Tahoe", "ablation", "landmark_genes"),
    # Cellosaurus identities verified against final_data/cell_line_metadata.parquet
    # (cell_name <-> Cell_ID_Cellosaur). Accession kept in the label so the sweep /
    # config / run-dir names stay greppable from any figure or stats table.
    "tahoe_CVCL_0131": ("Tahoe", "ref_cell", "A-172 (CVCL_0131)"),
    "tahoe_CVCL_0218": ("Tahoe", "ref_cell", "COLO 205 (CVCL_0218)"),
    "tahoe_CVCL_0480": ("Tahoe", "ref_cell", "PANC-1 (CVCL_0480)"),
    "tahoe_bert-base-smiles": ("Tahoe", "embedding", "bert-base-smiles"),
    "tahoe_CheMeleon": ("Tahoe", "embedding", "CheMeleon"),
    "tahoe_ChemBERTa-5M-MLM": ("Tahoe", "embedding", "ChemBERTa-5M-MLM"),
    "tahoe_ChemBERTa-5M-MTR": ("Tahoe", "embedding", "ChemBERTa-5M-MTR"),
    "tahoe_ChemBERTa-77M-MLM": ("Tahoe", "embedding", "ChemBERTa-77M-MLM"),
    "tahoe_ChemBERTa-100M-MLM": ("Tahoe", "embedding", "ChemBERTa-100M-MLM"),
    "tahoe_MoLFormer-XL-both-10pct": ("Tahoe", "embedding", "MoLFormer-XL"),
    "tahoe_MolGen-large": ("Tahoe", "embedding", "MolGen-large"),
    "tahoe_Morgan_512": ("Tahoe", "embedding", "Morgan_512"),
    "tahoe_Morgan_1024": ("Tahoe", "embedding", "Morgan_1024"),
    "tahoe_Morgan_2048": ("Tahoe", "embedding", "Morgan_2048"),
    "tahoe_unimolv1_84m": ("Tahoe", "embedding", "UniMolv1-84M"),
    "tahoe_unimolv2_84m": ("Tahoe", "embedding", "UniMolv2-84M"),
    "tahoe_unimolv2_164m": ("Tahoe", "embedding", "UniMolv2-164M"),
    "tahoe_unimolv2_310m": ("Tahoe", "embedding", "UniMolv2-310M"),
    "tahoe_unimolv2_570m": ("Tahoe", "embedding", "UniMolv2-570M"),
    "tahoe_unimolv2_1.1B": ("Tahoe", "embedding", "UniMolv2-1.1B"),
    # LINCS
    "lincs_default_config": ("LINCS", "ablation", "default"),
    "lincs_cellline_holdout": ("LINCS", "holdout", "cell-line holdout"),
    "lincs_delta_ref_only": ("LINCS", "ablation", "delta_ref_only"),
    "lincs_random_512": ("LINCS", "ablation", "random_512"),
    "lincs_MCF7": ("LINCS", "ref_cell", "MCF7"),
    "lincs_PC3": ("LINCS", "ref_cell", "PC3"),
    "lincs_average_ref": ("LINCS", "ref_handling", "average_ref"),
    "lincs_resample_ref": ("LINCS", "ref_handling", "resample_ref"),
    "lincs_bert-base-smiles": ("LINCS", "embedding", "bert-base-smiles"),
    "lincs_CheMeleon": ("LINCS", "embedding", "CheMeleon"),
    "lincs_ChemBERTa-5M-MLM": ("LINCS", "embedding", "ChemBERTa-5M-MLM"),
    "lincs_ChemBERTa-5M-MTR": ("LINCS", "embedding", "ChemBERTa-5M-MTR"),
    "lincs_ChemBERTa-77M-MLM": ("LINCS", "embedding", "ChemBERTa-77M-MLM"),
    "lincs_ChemBERTa-100M-MLM": ("LINCS", "embedding", "ChemBERTa-100M-MLM"),
    "lincs_MoLFormer-XL-both-10pct": ("LINCS", "embedding", "MoLFormer-XL"),
    "lincs_MolGen-large": ("LINCS", "embedding", "MolGen-large"),
    "lincs_Morgan_512": ("LINCS", "embedding", "Morgan_512"),
    "lincs_Morgan_1024": ("LINCS", "embedding", "Morgan_1024"),
    "lincs_Morgan_2048": ("LINCS", "embedding", "Morgan_2048"),
    "lincs_unimolv1_84m": ("LINCS", "embedding", "UniMolv1-84M"),
    "lincs_unimolv2_84m": ("LINCS", "embedding", "UniMolv2-84M"),
    "lincs_unimolv2_164m": ("LINCS", "embedding", "UniMolv2-164M"),
    "lincs_unimolv2_310m": ("LINCS", "embedding", "UniMolv2-310M"),
    "lincs_unimolv2_570m": ("LINCS", "embedding", "UniMolv2-570M"),
    "lincs_unimolv2_1.1B": ("LINCS", "embedding", "UniMolv2-1.1B"),
    "lincs_default_repro_0": ("LINCS", "repro", "repro_0.0"),
    "lincs_default_repro_0.1": ("LINCS", "repro", "repro_0.1"),
    "lincs_default_repro_0.2": ("LINCS", "repro", "repro_0.2"),
    "lincs_default_repro_0.3": ("LINCS", "repro", "repro_0.3"),
    "lincs_default_repro_0.4": ("LINCS", "repro", "repro_0.4"),
    "lincs_default_repro_0.5": ("LINCS", "repro", "repro_0.5"),
}

# Dose/time-encoding arms: the same embeddings as above, but with dose (and, on
# LINCS, exposure time) exposed explicitly to the model. Registered here because
# runs whose exp_key is absent from EXPERIMENT_META are silently dropped.
for _emb, _label in [
    ("Morgan_1024", "Morgan_1024"),
    ("CheMeleon", "CheMeleon"),
    ("unimolv2_1.1B", "UniMolv2-1.1B"),
    ("random_512", "random_512"),
]:
    for _ds, _dsname in [("tahoe", "Tahoe"), ("lincs", "LINCS")]:
        for _enc in ["concat", "gate", "onehot"]:
            EXPERIMENT_META[f"{_ds}_{_emb}_dose_{_enc}"] = (
                _dsname,
                "dose_encoding",
                f"{_label} + dose ({_enc})",
            )

# Gate-only dose/time encoding for the remaining embeddings (gate outperformed
# concat/onehot in the first batch above, so the rest of the roster gets gate only).
for _emb, _label in [
    ("bert-base-smiles", "bert-base-smiles"),
    ("ChemBERTa-5M-MLM", "ChemBERTa-5M-MLM"),
    ("ChemBERTa-5M-MTR", "ChemBERTa-5M-MTR"),
    ("ChemBERTa-77M-MLM", "ChemBERTa-77M-MLM"),
    ("ChemBERTa-100M-MLM", "ChemBERTa-100M-MLM"),
    ("MoLFormer-XL-both-10pct", "MoLFormer-XL"),
    ("MolGen-large", "MolGen-large"),
    ("Morgan_512", "Morgan_512"),
    ("Morgan_2048", "Morgan_2048"),
    ("unimolv1_84m", "UniMolv1-84M"),
    ("unimolv2_84m", "UniMolv2-84M"),
    ("unimolv2_164m", "UniMolv2-164M"),
    ("unimolv2_310m", "UniMolv2-310M"),
    ("unimolv2_570m", "UniMolv2-570M"),
]:
    for _ds, _dsname in [("tahoe", "Tahoe"), ("lincs", "LINCS")]:
        EXPERIMENT_META[f"{_ds}_{_emb}_dose_gate"] = (
            _dsname,
            "dose_encoding",
            f"{_label} + dose (gate)",
        )

EMBED_TYPE: dict[str, str] = {
    "CheMeleon": "Molecular graph MPNN",
    "bert-base-smiles": "SMILES transformer",
    "ChemBERTa-5M-MLM": "SMILES transformer",
    "ChemBERTa-5M-MTR": "SMILES transformer",
    "ChemBERTa-77M-MLM": "SMILES transformer",
    "ChemBERTa-100M-MLM": "SMILES transformer",
    "MoLFormer-XL": "SMILES transformer",
    "MolGen-large": "SMILES transformer",
    "Morgan_512": "Morgan fingerprint",
    "Morgan_1024": "Morgan fingerprint",
    "Morgan_2048": "Morgan fingerprint",
    "UniMolv1-84M": "3D geometry (UniMol)",
    "UniMolv2-84M": "3D geometry (UniMol)",
    "UniMolv2-164M": "3D geometry (UniMol)",
    "UniMolv2-310M": "3D geometry (UniMol)",
    "UniMolv2-570M": "3D geometry (UniMol)",
    "UniMolv2-1.1B": "3D geometry (UniMol)",
}

UNIMOL_PARAMS: dict[str, float] = {
    "UniMolv2-84M": 84e6,
    "UniMolv2-164M": 164e6,
    "UniMolv2-310M": 310e6,
    "UniMolv2-570M": 570e6,
    "UniMolv2-1.1B": 1100e6,
}

# Native (pre-PCA) dimensionality of each compound embedding, measured directly
# from its parquet's `embedding` column (byte length / 4, since Float32 — see
# predict_profiles.jl's `reinterpret(Float32, e)`). Identical between Tahoe and
# LINCS (each embedding is generated once per compound, not per dataset).
EMBED_NATIVE_DIM: dict[str, int] = {
    "CheMeleon": 2048,
    "bert-base-smiles": 768,
    "ChemBERTa-5M-MLM": 384,
    "ChemBERTa-5M-MTR": 384,
    "ChemBERTa-77M-MLM": 384,
    "ChemBERTa-100M-MLM": 768,
    "MoLFormer-XL": 768,
    "MolGen-large": 1024,
    "Morgan_512": 512,
    "Morgan_1024": 1024,
    "Morgan_2048": 2048,
    "random_512": 512,
    "UniMolv1-84M": 512,
    "UniMolv2-84M": 768,
    "UniMolv2-164M": 768,
    "UniMolv2-310M": 1024,
    "UniMolv2-570M": 1536,
    "UniMolv2-1.1B": 1536,
}

BASELINE_LABELS = {"delta_ref": "delta_ref", "mean_delta": "mean train delta"}

# ── Internal constants ────────────────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{6})$")

_METRIC_COLS = ["pearson", "spearman", "l2", "cosine"]
_METRIC_COLS_UMI = ["total_cells", "total_umis", "pearson", "spearman", "l2", "cosine"]

_NUMERIC_SUMMARY_COLS = [
    "n_pca_expr",
    "n_pca_molec",
    "input_dim",
    "output_dim",
    "batch_size",
    "weight_decay",
    "n_epochs",
    "lr",
    "n_train",
    "n_val",
    "n_test",
    "selected_epoch",
    "test_pearson",
    "val_pearson",
    "train_pearson",
    "test_spearman",
    "val_spearman",
    "train_spearman",
    "test_l2",
    "test_cosine_sim",
]

# ── Low-level parsers ─────────────────────────────────────────────────────────

def _parse_run_dir(dirname: str) -> tuple[str | None, str | None]:
    m = _TIMESTAMP_RE.search(dirname)
    if not m:
        return None, None
    ts = m.group(1)
    prefix = dirname[: m.start()]
    if prefix.startswith("filtered_pseudobulks_alpha_10000_tahoe_"):
        exp_key = "tahoe_" + prefix.removeprefix("filtered_pseudobulks_alpha_10000_tahoe_")
    elif prefix.startswith("filtered_lincs_lincs_"):
        exp_key = "lincs_" + prefix.removeprefix("filtered_lincs_lincs_")
    else:
        return None, None
    return exp_key, ts


def _read_perfs_csv(path: str, has_umi: bool = False) -> pd.DataFrame:
    """Parse a per-obs CSV whose ``drug`` column may contain commas.

    Layout: cell_line, drug, smiles, dose, time, [total_cells, total_umis,]
            pearson, spearman, l2, cosine

    ``drug`` is the only comma-bearing field. Older files wrote it unquoted (Julia
    Symbol columns bypassed CSV.jl quoting), so a drug name like
    "Dapagliflozin ((2S)-1,2-propanediol, hydrate)" splits into extra tokens.
    No SMILES contains a comma, so ``cell_line`` is at a fixed offset from the
    left and everything from ``smiles`` onward is at a fixed offset from the
    right. ``drug`` is whatever is left in between, rejoined.

    Works for both the old (unquoted) and current (quoted) formats.
    """
    if has_umi:
        # cell_line, drug, smiles..., dose, time, total_cells, total_umis, pearson, spearman, l2, cosine
        n_trail = 6  # dose + time + 4 metrics + 2 umi = 8 total; last 6 are metrics+umi
        trail_names = ["total_cells", "total_umis", "pearson", "spearman", "l2", "cosine"]
        n_suffix = 8  # dose, time, total_cells, total_umis, pearson, spearman, l2, cosine
    else:
        # cell_line, drug, smiles..., dose, time, pearson, spearman, l2, cosine
        n_trail = 4
        trail_names = ["pearson", "spearman", "l2", "cosine"]
        n_suffix = 6  # dose, time, pearson, spearman, l2, cosine

    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")[1:]  # skip header
    if lines and not lines[-1]:
        lines = lines[:-1]

    cell_lines, drugs, smiles, doses, times = [], [], [], [], []
    metrics = []
    for line in lines:
        parts = line.split(b",")
        cell_lines.append(parts[0].decode())
        # smiles sits immediately before dose; drug is everything between it and cell_line
        drugs.append(b",".join(parts[1:-(n_suffix + 1)]).decode().strip('"'))
        smiles.append(parts[-(n_suffix + 1)].decode())
        doses.append(parts[-n_suffix].decode())
        times.append(parts[-n_suffix + 1].decode())
        metrics.append(parts[-n_trail:])

    arr = np.array(metrics, dtype=np.float64)
    df = pd.DataFrame(arr, columns=trail_names)
    df["cell_line"] = cell_lines
    df["drug"] = drugs
    df["smiles"] = smiles
    df["dose"] = doses
    df["time"] = times
    return df


def _read_perfs_csv_metrics_only(path: str, has_umi: bool = False) -> np.ndarray:
    """Parse only the trailing metric columns from a per-obs CSV.

    Returns a 2D float64 array of shape (n_obs, n_metrics). Faster than
    _read_perfs_csv because it skips decoding cell_line/drug/dose/time.
    """
    n_trail = 6 if has_umi else 4

    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")[1:]
    if lines and not lines[-1]:
        lines = lines[:-1]
    rows = [line.split(b",")[-n_trail:] for line in lines]
    return np.array(rows, dtype=np.float64)


def _read_summary(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ── Table builders ────────────────────────────────────────────────────────────

def _build_runs_df() -> pd.DataFrame:
    records = []
    # Run dirs live under each dataset's prediction_dir (both datasets currently
    # share the same one), not directly under BIOPERT_OUTDIR.
    prediction_dirs = {dataset_paths(BIOPERT_OUTDIR, ds)["prediction_dir"] for ds in ("lincs", "tahoe")}
    for prediction_dir in sorted(prediction_dirs):
        if not os.path.isdir(prediction_dir):
            continue
        for dirname in sorted(os.listdir(prediction_dir)):
            path = os.path.join(prediction_dir, dirname)
            if not os.path.isdir(path):
                continue
            exp_key, ts = _parse_run_dir(dirname)
            if exp_key is None or exp_key not in EXPERIMENT_META:
                continue
            summary_path = os.path.join(path, "summary.toml")
            if not os.path.exists(summary_path):
                continue
            try:
                s = _read_summary(summary_path)
            except Exception:
                continue
            dataset, category, label = EXPERIMENT_META[exp_key]
            hyperband_stopped = os.path.exists(os.path.join(path, "hyperband_completed.toml"))
            records.append(
                dict(
                    run_dir=os.path.relpath(path, BIOPERT_OUTDIR),
                    hyperband_stopped=hyperband_stopped,
                    exp_key=exp_key,
                    timestamp=ts,
                    dataset=dataset,
                    category=category,
                    label=label,
                    n_train=s.get("n_train"),
                    n_val=s.get("n_val"),
                    n_test=s.get("n_test"),
                    train_pearson=s.get("train_pearson"),
                    val_pearson=s.get("val_pearson"),
                    test_pearson=s.get("test_pearson"),
                    train_spearman=s.get("train_spearman"),
                    val_spearman=s.get("val_spearman"),
                    test_spearman=s.get("test_spearman"),
                    test_l2=s.get("test_l2"),
                    test_cosine_sim=s.get("test_cosine_sim"),
                    selected_epoch=s.get("selected_epoch"),
                    n_epochs=s.get("n_epochs"),
                    lr=s.get("lr"),
                    batch_size=s.get("batch_size"),
                    weight_decay=s.get("weight_decay"),
                    hidden_layers=s.get("hidden_layers"),
                    n_pca_expr=s.get("n_pca_expr"),
                    n_pca_molec=s.get("n_pca_molec"),
                    input_dim=s.get("input_dim"),
                    output_dim=s.get("output_dim"),
                    model_type=s.get("model_type", "mlp"),
                    use_delta_ref=s.get("use_delta_ref", True),
                    use_untrt_target=s.get("use_untrt_target", True),
                    use_molec_embed=s.get("use_molec_embed", False),
                    average_ref=s.get("average_ref", False),
                    resample_ref=s.get("resample_ref", False),
                    holdout_cell_lines=s.get("holdout_cell_lines", False),
                    landmark_genes_only=s.get("landmark_genes_only", False),
                )
            )
    df = pd.DataFrame(records)
    for col in _NUMERIC_SUMMARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not df.empty:
        df = df.sort_values("timestamp").groupby("exp_key", group_keys=False).head(MAX_TRIALS_PER_EXP)
    return df


def _build_best_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df.copy()
    eligible = runs_df.dropna(subset=["val_spearman"])
    if eligible.empty:
        return runs_df.iloc[0:0].copy()
    return eligible.loc[eligible.groupby("exp_key")["val_spearman"].idxmax()].reset_index(drop=True)


_TEST_KEY_COLS = ["cell_line", "smiles", "dose", "time"]


def _test_obs_keys(csv_path: str, has_umi: bool) -> set[tuple]:
    """Return the set of (cell_line, smiles, dose, time) keys present in a test per-obs CSV."""
    df = _read_perfs_csv(csv_path, has_umi=has_umi)
    return set(map(tuple, df[_TEST_KEY_COLS].itertuples(index=False, name=None)))


_INTERSECTION_EXCLUDED_CATEGORIES = {"ref_cell", "holdout"}


def _compute_test_key_intersection(runs_df: pd.DataFrame) -> dict[str, set[tuple]]:
    """For each dataset, intersect the test-set observation keys across every exp_key
    that shares the default reference cell line and the compound-only butina split.

    Different exp_keys can drop different test observations in `build_obs` (e.g. a
    compound missing a molecular embedding, or missing a delta_ref match in the
    reference cell line) even though they share the same pinned compound split. Any
    cross-config comparison of test-set metrics must therefore be restricted to
    observations present in every config, or the comparison is confounded by which
    rows happened to survive each config's filtering. One representative run per
    exp_key is enough: every run of a given exp_key shares the same test-set
    membership (it's determined by the config, not the hyperparameters).

    `ref_cell` (alternate reference cell line, e.g. MCF7/PC3/CVCL_*) and `holdout`
    (cellline_holdout's 2x2 union split) exp_keys are excluded: their test-set keys
    are drawn from a structurally different population (different reference cell
    line's delta_ref availability, or a different pinned split entirely), not a
    filtered subset of the default-reference/compound-only-split population, so
    intersecting them in would conflate two different experimental designs.
    """
    reps = runs_df[~runs_df["category"].isin(_INTERSECTION_EXCLUDED_CATEGORIES)].drop_duplicates(subset=["exp_key"])
    intersections: dict[str, set[tuple]] = {}
    for dataset, group in reps.groupby("dataset"):
        has_umi = dataset == "Tahoe"
        key_sets = []
        for _, row in group.iterrows():
            csv_path = os.path.join(BIOPERT_OUTDIR, row["run_dir"], "test_perfs_per_obs.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                key_sets.append(_test_obs_keys(csv_path, has_umi))
            except Exception as e:
                warnings.warn(f"skipping {csv_path} for test-key intersection: {e}")
        intersections[dataset] = set.intersection(*key_sets) if key_sets else set()
    return intersections


def _compute_run_stats_single(
    run_path: str,
    run_dir: str,
    exp_key: str,
    dataset: str,
    category: str,
    label: str,
    test_key_intersection: set[tuple],
) -> pd.DataFrame | None:
    """Read per-obs CSVs for one run and return per-split summary statistics.

    The test split is restricted to `test_key_intersection` (observations present
    in every exp_key's test set for this dataset) before computing mean/std, so
    test-split stats are directly comparable across configs. Train/val splits use
    every observation, unrestricted. Runs in `_INTERSECTION_EXCLUDED_CATEGORIES`
    (different reference cell line or a different pinned split entirely) are exempt
    and keep their full, unmasked test set — `test_key_intersection` is drawn only
    from the default-reference/compound-only-split population, so masking against
    it would incorrectly drop most/all of an excluded run's test rows.

    Result is cached as <run_dir>/run_stats_cache.parquet (3 rows). On
    subsequent calls the cache is returned directly without re-reading CSVs.
    """
    cache_path = os.path.join(run_path, _PER_RUN_CACHE_FILE)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    has_umi = dataset == "Tahoe"
    metric_cols = _METRIC_COLS_UMI if has_umi else _METRIC_COLS
    restrict_test = category not in _INTERSECTION_EXCLUDED_CATEGORIES

    frames = []
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(run_path, f"{split}_perfs_per_obs.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            if split == "test" and restrict_test:
                df = _read_perfs_csv(csv_path, has_umi=has_umi)
                keys = df[_TEST_KEY_COLS].itertuples(index=False, name=None)
                mask = [k in test_key_intersection for k in keys]
                arr = df.loc[mask, metric_cols].to_numpy(dtype=np.float64)
            else:
                arr = _read_perfs_csv_metrics_only(csv_path, has_umi=has_umi)
        except Exception as e:
            warnings.warn(f"skipping {csv_path}: {e}")
            continue
        row = {"run_dir": run_dir, "exp_key": exp_key, "dataset": dataset, "category": category, "label": label, "split": split}
        for j, c in enumerate(metric_cols):
            row[f"{c}_mean"] = arr[:, j].mean()
            row[f"{c}_std"] = arr[:, j].std()
        frames.append(row)

    if not frames:
        return None
    result = pd.DataFrame(frames)
    result.to_parquet(cache_path, index=False)
    return result


def _build_run_stats(runs_df: pd.DataFrame, n_workers: int = 8) -> pd.DataFrame:
    from tqdm.auto import tqdm
    from concurrent.futures import ProcessPoolExecutor

    test_key_intersections = _compute_test_key_intersection(runs_df)
    for dataset, keys in test_key_intersections.items():
        print(f"Test-set intersection ({dataset}): {len(keys):,} observations shared across all exp_keys")

    tasks = [
        (
            os.path.join(BIOPERT_OUTDIR, row["run_dir"]),
            row["run_dir"],
            row["exp_key"],
            row["dataset"],
            row["category"],
            row["label"],
            test_key_intersections[row["dataset"]],
        )
        for _, row in runs_df.iterrows()
    ]

    cached = sum(os.path.exists(os.path.join(BIOPERT_OUTDIR, d, _PER_RUN_CACHE_FILE)) for d in runs_df["run_dir"])
    print(f"Building run_stats: {cached} cached, {len(runs_df) - cached} to parse " f"({n_workers} workers)")

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for result in tqdm(ex.map(_compute_run_stats_single, *zip(*tasks)), total=len(tasks), unit="run"):
            if result is not None:
                frames.append(result)

    return pd.concat(frames, ignore_index=True)


def _build_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for dirname in sorted(os.listdir(BASELINES_DIR)):
        bpath = os.path.join(BASELINES_DIR, dirname)
        if not os.path.isdir(bpath):
            continue
        parts = dirname.split("_", 1)
        if len(parts) != 2:
            continue
        ds_raw, baseline = parts
        dataset = "LINCS" if ds_raw == "lincs" else "Tahoe"
        csv_path = os.path.join(bpath, "test_perfs_per_obs.csv")
        if not os.path.exists(csv_path):
            continue
        # Tahoe per-obs CSVs carry the two UMI columns (total_cells, total_umis)
        # ahead of the metrics; reading them as has_umi=False would shift dose/time
        # onto those UMI values (the metrics are the last 4 cols either way, so
        # summary stats are unaffected — but the join keys would be garbage).
        df = _read_perfs_csv(csv_path, has_umi=(dataset == "Tahoe"))
        df["baseline"] = baseline
        df["dataset"] = dataset
        records.append(df)
    baselines_df = pd.concat(records, ignore_index=True)
    baselines_df["label"] = baselines_df["baseline"].map(BASELINE_LABELS)
    baseline_summary = baselines_df.groupby(["dataset", "label"])[["pearson", "spearman"]].mean().reset_index()
    return baselines_df, baseline_summary


# ── Public result container ───────────────────────────────────────────────────

@dataclass
class LoadedResults:
    runs_df: pd.DataFrame
    best_runs: pd.DataFrame
    run_stats: pd.DataFrame  # per-run, per-split summary stats (mean/std of metrics)
    baselines_df: pd.DataFrame
    baseline_summary: pd.DataFrame

    def load_perobs_single(self, exp_key: str, split: str = "test") -> pd.DataFrame:
        """Return raw per-obs rows for the best run of one experiment."""
        row = self.best_runs[self.best_runs["exp_key"] == exp_key]
        if row.empty:
            raise ValueError(f"exp_key not found: {exp_key}")
        row = row.iloc[0]
        path = os.path.join(BIOPERT_OUTDIR, row["run_dir"], f"{split}_perfs_per_obs.csv")
        df = _read_perfs_csv(path, has_umi=row["dataset"] == "Tahoe")
        df["exp_key"] = row["exp_key"]
        df["dataset"] = row["dataset"]
        df["category"] = row["category"]
        df["label"] = row["label"]
        df["split"] = split
        return df

    def load_perobs_multi(self, exp_keys: list[str], split: str = "test") -> pd.DataFrame:
        """Return raw per-obs rows for the best run of each listed experiment, concatenated."""
        frames = []
        for exp_key in exp_keys:
            try:
                frames.append(self.load_perobs_single(exp_key, split=split))
            except (ValueError, FileNotFoundError):
                warnings.warn(f"skipping {exp_key} ({split}): file not found or exp unknown")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_quadrants(self, exp_key: str, split: str = "test") -> pd.DataFrame:
        """Label each held-out observation by what the model had actually seen in training.

        Membership is read from the run's own train split, so the labels reflect what
        the model was really trained on rather than what a config claims. Adds
        `seen_cell_line`, `seen_compound`, and `quadrant`.

        Quadrants (see the cell-line holdout experiment):
          Q1 seen cell line, seen compound    — never appears in val/test
          Q2 seen cell line, unseen compound  — the setting the other sweeps measure
          Q3 unseen cell line, seen compound  — the cell-line axis, isolated
          Q4 unseen cell line, unseen compound
        """
        train = self.load_perobs_single(exp_key, split="train")
        seen_cl = set(train["cell_line"])
        seen_smi = set(train["smiles"])

        df = self.load_perobs_single(exp_key, split=split)
        df["seen_cell_line"] = df["cell_line"].isin(seen_cl)
        df["seen_compound"] = df["smiles"].isin(seen_smi)
        df["quadrant"] = np.select(
            [
                df["seen_cell_line"] & df["seen_compound"],
                df["seen_cell_line"] & ~df["seen_compound"],
                ~df["seen_cell_line"] & df["seen_compound"],
            ],
            ["Q1", "Q2", "Q3"],
            default="Q4",
        )
        return df

    def quadrant_summary(
        self, exp_key: str, split: str = "test", metric: str = "spearman"
    ) -> pd.DataFrame:
        """Per-quadrant micro and macro averages of `metric`.

        micro = mean over observations. macro = mean over cell lines of the per-cell-line
        mean. They diverge when one cell line dominates a quadrant: in LINCS, JURKAT is
        ~47% of the held-out-cell-line test observations, so only the macro average keeps
        it from carrying the headline number.
        """
        df = self.load_quadrants(exp_key, split=split)
        per_cl = df.groupby(["quadrant", "cell_line"])[metric].mean()
        out = pd.DataFrame(
            {
                "n_obs": df.groupby("quadrant")[metric].size(),
                "n_cell_lines": df.groupby("quadrant")["cell_line"].nunique(),
                "n_compounds": df.groupby("quadrant")["smiles"].nunique(),
                f"micro_{metric}": df.groupby("quadrant")[metric].mean(),
                f"macro_{metric}": per_cl.groupby("quadrant").mean(),
            }
        )
        return out.sort_index()

    def load_quadrant_baselines(
        self,
        exp_key: str,
        split: str = "test",
        metric: str = "spearman",
        baselines: tuple[str, ...] = ("delta_ref", "mean_delta"),
    ) -> pd.DataFrame:
        """Quadrant-labelled per-obs rows with baseline metrics attached, one column each.

        Baselines come from BASELINES_CELLLINE_DIR — the recompute on the cell-line
        holdout split. Using the compound-only baselines here would compare against a
        different split entirely.

        Pairing is **positional**, not a key merge. `(cell_line, smiles, dose, time)` is
        not unique: distinct drug names share a canonical SMILES (e.g. `(S)-Crizotinib`
        and `crizotinib`; two BRD ids for the same structure), so a merge on those keys
        silently fans those observations out. The baseline CSVs are emitted from the same
        split in the same order, which is asserted below rather than assumed.
        """
        ds_raw = exp_key.split("_", 1)[0]
        if ds_raw not in ("tahoe", "lincs"):
            raise ValueError(f"cannot infer dataset prefix from exp_key: {exp_key!r}")

        df = self.load_quadrants(exp_key, split=split).reset_index(drop=True)
        key = ["cell_line", "smiles", "dose", "time"]

        for b in baselines:
            path = os.path.join(BASELINES_CELLLINE_DIR, f"{ds_raw}_{b}", f"{split}_perfs_per_obs.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            bdf = _read_perfs_csv(path, has_umi=(ds_raw == "tahoe")).reset_index(drop=True)
            if len(bdf) != len(df) or not bdf[key].equals(df[key]):
                raise ValueError(
                    f"baseline {b!r} is not row-aligned with {exp_key} {split} "
                    f"({len(bdf)} vs {len(df)} rows); refusing to pair positionally"
                )
            df[f"{metric}__{b}"] = bdf[metric].to_numpy()
        return df

    def quadrant_vs_baseline(
        self,
        exp_key: str,
        split: str = "test",
        metric: str = "spearman",
        baselines: tuple[str, ...] = ("delta_ref", "mean_delta"),
    ) -> pd.DataFrame:
        """Per-quadrant model-vs-baseline comparison, micro and macro, with paired tests.

        One row per (quadrant, baseline). `*_diff` columns are model minus baseline, so
        positive means the model wins. (Named `diff`, not `delta`, to keep them clear of
        the `mean_delta` baseline and of the delta profiles themselves.)

        Two paired Wilcoxon signed-rank tests, because they answer different questions:
          `p_obs` pairs observations (n in the thousands — near-anything reaches
          significance, so read `micro_delta` for the size of the effect);
          `p_cellline` pairs per-cell-line means (n = the number of held-out cell lines,
          7 or 20 here — low-powered but the honest unit of replication, since
          observations within a cell line are not independent).

        Baselines that are all-NaN for a quadrant (`zero` has no rank variance, so its
        Spearman is undefined) yield NaN rather than propagating into the model column.
        """
        from scipy.stats import wilcoxon

        df = self.load_quadrant_baselines(exp_key, split=split, metric=metric, baselines=baselines)

        rows = []
        for quad, g in df.groupby("quadrant"):
            per_cl_model = g.groupby("cell_line")[metric].mean()
            for b in baselines:
                bcol = f"{metric}__{b}"
                paired = g[[metric, bcol, "cell_line"]].dropna()
                per_cl_base = g.groupby("cell_line")[bcol].mean()
                cl = pd.concat([per_cl_model, per_cl_base], axis=1).dropna()

                def _p(a, b_):
                    if len(a) < 1 or np.allclose(np.asarray(a) - np.asarray(b_), 0):
                        return np.nan
                    try:
                        return wilcoxon(a, b_).pvalue
                    except ValueError:
                        return np.nan

                rows.append(
                    {
                        "quadrant": quad,
                        "baseline": b,
                        "n_obs": len(g),
                        "n_obs_paired": len(paired),
                        "n_cell_lines": g["cell_line"].nunique(),
                        "micro_model": g[metric].mean(),
                        "micro_baseline": g[bcol].mean(),
                        "micro_diff": g[metric].mean() - g[bcol].mean(),
                        "macro_model": per_cl_model.mean(),
                        "macro_baseline": per_cl_base.mean(),
                        "macro_diff": per_cl_model.mean() - per_cl_base.mean(),
                        "n_cl_wins": int((cl.iloc[:, 0] > cl.iloc[:, 1]).sum()),
                        "p_obs": _p(paired[metric], paired[bcol]),
                        "p_cellline": _p(cl.iloc[:, 0], cl.iloc[:, 1]),
                    }
                )
        return pd.DataFrame(rows).sort_values(["quadrant", "baseline"]).reset_index(drop=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def load_results(
    biopert_outdir: str,
    lincs_beta_dir: str | None = None,
    force_recompute: bool = False,
    use_cache: bool = False,
    n_workers: int = 8,
) -> LoadedResults:
    """
    Load all sweep results and return a LoadedResults object.

    Parameters
    ----------
    biopert_outdir:
        Pipeline output directory (see README.md#output-directory-biopert_outdir).
    lincs_beta_dir:
        Raw LINCS_beta download directory (see configure()). Optional: only
        LINCS_CELLINFO_PATH depends on it.
    force_recompute:
        If True, delete the run_stats cache and re-parse every per-obs CSV.
    use_cache:
        If True, load the run_stats cache without checking for new runs.
        Ignored when force_recompute=True.
    n_workers:
        Number of parallel processes for the cold-start CSV parse (default 8).
    """
    configure(biopert_outdir, lincs_beta_dir)

    runs_df = _build_runs_df()
    best_runs = _build_best_runs(runs_df)
    print(f"Completed runs: {len(runs_df)}  |  Experiments with ≥1 run: {len(best_runs)} / {len(EXPERIMENT_META)}")

    if force_recompute:
        if os.path.exists(_RUN_STATS_CACHE):
            os.remove(_RUN_STATS_CACHE)
        for run_dir in runs_df["run_dir"]:
            p = os.path.join(BIOPERT_OUTDIR, run_dir, _PER_RUN_CACHE_FILE)
            if os.path.exists(p):
                os.remove(p)

    cache_key = "|".join(sorted(runs_df["run_dir"].tolist()))
    cache_exists = os.path.exists(_RUN_STATS_CACHE) and os.path.exists(_CACHE_KEY_FILE)
    cache_hit = not force_recompute and cache_exists and (use_cache or open(_CACHE_KEY_FILE).read().strip() == cache_key)

    if cache_hit:
        print(f"Loading run_stats from cache ({_RUN_STATS_CACHE})...")
        run_stats = pd.read_parquet(_RUN_STATS_CACHE)
    else:
        run_stats = _build_run_stats(runs_df, n_workers=n_workers)
        run_stats.to_parquet(_RUN_STATS_CACHE, index=False)
        open(_CACHE_KEY_FILE, "w").write(cache_key)
        print(f"Cache updated: {len(run_stats):,} rows → {_RUN_STATS_CACHE}")

    baselines_df, baseline_summary = _build_baselines()

    return LoadedResults(
        runs_df=runs_df,
        best_runs=best_runs,
        run_stats=run_stats,
        baselines_df=baselines_df,
        baseline_summary=baseline_summary,
    )
