#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tomli
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/network/scratch/l/lola.lebreton/transcriptomes")
SEED = 42
STAT_SEED = 0
N_BOOT = 1000
CI_LEVEL = 0.95
DATASETS = ("tahoe", "lincs")
DATASET_LABEL = {"tahoe": "Tahoe", "lincs": "LINCS"}
GENES = {"tahoe": 19020, "lincs": 978}
DEFAULT_EXPERIMENT = {"tahoe": "tahoe_default_config", "lincs": "lincs_default_config"}
DEFAULT_REFERENCE = {"tahoe": "CVCL_0023", "lincs": "A549"}
PROFILE_DISTRIBUTIONS = {
    "tahoe": {"mean": 0.0006, "sd": 0.073, "minimum": -3.8, "maximum": 5.2},
    "lincs": {"mean": -0.0065, "sd": 0.700, "minimum": -14.6, "maximum": 14.4},
}
REPRESENTATIONS = [
    "Morgan_512", "Morgan_1024", "Morgan_2048", "RDKit_512", "RDKit_1024",
    "RDKit_2048", "AtomPair_512", "AtomPair_1024", "AtomPair_2048", "MACCS_166",
    "CheMeleon", "ChemBERTa-5M-MLM", "ChemBERTa-5M-MTR", "ChemBERTa-77M-MLM",
    "ChemBERTa-100M-MLM", "MoLFormer-XL-both-10pct", "MolGen-large",
    "bert-base-smiles", "unimolv1_84m", "unimolv2_84m", "unimolv2_164m",
    "unimolv2_310m", "unimolv2_570m", "unimolv2_1.1B", "random_512",
]
MODEL_SOURCES = {
    "CheMeleon": "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
    "ChemBERTa-5M-MLM": "DeepChem/ChemBERTa-5M-MLM",
    "ChemBERTa-5M-MTR": "DeepChem/ChemBERTa-5M-MTR",
    "ChemBERTa-77M-MLM": "DeepChem/ChemBERTa-77M-MLM",
    "ChemBERTa-100M-MLM": "DeepChem/ChemBERTa-100M-MLM",
    "MoLFormer-XL-both-10pct": "ibm/MoLFormer-XL-both-10pct",
    "MolGen-large": "zjunlp/MolGen-large",
    "bert-base-smiles": "unikei/bert-base-smiles",
    "unimolv1_84m": "unimolv1/84m",
    "unimolv2_84m": "unimolv2/84m",
    "unimolv2_164m": "unimolv2/164m",
    "unimolv2_310m": "unimolv2/310m",
    "unimolv2_570m": "unimolv2/570m",
    "unimolv2_1.1B": "unimolv2/1.1B",
}
DISPLAY_NAMES = {
    "Morgan_512": "ECFP6-512", "Morgan_1024": "ECFP6-1024", "Morgan_2048": "ECFP6-2048",
    "RDKit_512": "RDKit path-512", "RDKit_1024": "RDKit path-1024", "RDKit_2048": "RDKit path-2048",
    "AtomPair_512": "Atom pair-512", "AtomPair_1024": "Atom pair-1024", "AtomPair_2048": "Atom pair-2048",
    "MACCS_166": "MACCS-166", "MoLFormer-XL-both-10pct": "MoLFormer-XL",
    "unimolv1_84m": "UniMolv1-84M", "unimolv2_84m": "UniMolv2-84M",
    "unimolv2_164m": "UniMolv2-164M", "unimolv2_310m": "UniMolv2-310M",
    "unimolv2_570m": "UniMolv2-570M", "unimolv2_1.1B": "UniMolv2-1.1B",
    "random_512": "Random control",
}
PATHWAY_PROFILES = {
    "target": "measured_target", "prediction": "biopert_prediction", "reference": "measured_reference"
}
MODULES = {
    "ERK-output genes": ["DUSP6", "ETV1", "ETV4", "ETV5", "SPRY2", "SPRY4", "SPRED1", "SPRED2"],
    "Melanocytic / MITF-PGC-1alpha": ["MITF", "DCT", "TYR", "MLANA", "TYRP1", "PPARGC1A", "GLUD1", "IDH1"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPO / "supplementary_data")
    parser.add_argument("--skip-pathway", action="store_true")
    return parser.parse_args()


def require(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required source data:\n" + "\n".join(missing))


def stable_id(*values, prefix=""):
    value = "\x1f".join("NA" if pd.isna(x) else str(x) for x in values)
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:16]


def parse_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group()) if match else np.nan


def csv_write(frame, path):
    frame.to_csv(path, index=False, na_rep="NA", compression={"method": "gzip", "mtime": 0, "compresslevel": 1} if path.suffix == ".gz" else None)


def software_version(package):
    if package == "python":
        return sys.version.split()[0]
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NA"


def parquet_read(path, columns):
    return pq.read_table(path, columns=columns).to_pandas(ignore_metadata=True)


def parquet_first(path, column):
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=[column]))
    return batch.column(0)[0].as_py()


def git_value(*args):
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "NA"


def jld2_columns(path, key, requested):
    with h5py.File(path) as handle:
        root = handle[key][()]
        names = [x.decode() for x in handle[root["colindex"]["names"]][()]]
        columns = handle[root["columns"]][()]
        result = {}
        for name in requested:
            data = handle[columns[names.index(name)]][()]
            if data.dtype == object:
                data = np.array([x.decode() if isinstance(x, bytes) else x for x in data], dtype=object)
            result[name] = data
    return pd.DataFrame(result)


def resolve_run(source, run_dir):
    path = Path(str(run_dir))
    return path if path.is_absolute() else source / "results" / path


def load_results(source):
    sys.path.insert(0, str(REPO / "nbs"))
    import results_data
    results_data._RUN_STATS_CACHE = str(REPO / "nbs/cache_run_stats.parquet")
    results_data._CACHE_KEY_FILE = str(REPO / "nbs/cache_key.txt")
    # lincs_beta_dir is a placeholder: this script never reads LINCS_CELLINFO_PATH,
    # but results_data.configure() wants something for it regardless.
    loaded = results_data.load_results(str(source / "results"), lincs_beta_dir=str(source), use_cache=True)
    run = source / "results/systema_tahoe/runs/filtered_pseudobulks_alpha_10000_tahoe_default_config_2026-08-06_185016"
    summary = results_data._read_summary(str(run / "summary.toml"))
    index = loaded.best_runs.index[loaded.best_runs.exp_key.eq("tahoe_default_config")]
    if len(index) != 1:
        raise ValueError("Expected one Tahoe default model")
    for column, value in summary.items():
        if column in loaded.best_runs and value != "nothing":
            loaded.best_runs.at[index[0], column] = value
    loaded.best_runs.at[index[0], "run_dir"] = str(run)
    loaded.best_runs.at[index[0], "timestamp"] = "2026-08-06_185016"
    return loaded, results_data


def cell_names(source):
    metadata = pd.read_parquet(source / "final_data/cell_line_metadata.parquet")
    names = metadata.dropna(subset=["Cell_ID_Cellosaur"]).drop_duplicates("Cell_ID_Cellosaur")
    return names.set_index("Cell_ID_Cellosaur")["cell_name"].astype(str).to_dict()


def representation_id(row):
    label = str(row.get("label", ""))
    exp_key = str(row.get("exp_key", ""))
    if any(x in exp_key for x in ["default", "delta_ref_only", "cellline_holdout", "CVCL_", "_MCF7", "_PC3", "average_ref", "resample_ref", "repro_"]):
        return "biopert"
    base = re.sub(r" \+ dose \((?:concat|gate|onehot)\)$", "", label)
    matches = [x for x in REPRESENTATIONS if x.lower() == base.lower() or DISPLAY_NAMES.get(x, "").lower() == base.lower()]
    return matches[0] if matches else base.replace(" ", "_").lower()


def bootstrap_mean(values, clusters=None):
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    values = values[keep]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(STAT_SEED)
    if clusters is None:
        groups = values[:, None]
        sizes = np.ones(len(values))
    else:
        labels = np.asarray(clusters)[keep]
        codes = pd.factorize(labels)[0]
        groups = np.bincount(codes, weights=values)[:, None]
        sizes = np.bincount(codes)
    k = len(groups)
    draws = np.empty(N_BOOT)
    step = max(1, 20_000_000 // max(k, 1))
    for start in range(0, N_BOOT, step):
        count = min(step, N_BOOT - start)
        index = rng.integers(0, k, size=(count, k))
        draws[start:start + count] = groups[index, 0].sum(axis=1) / sizes[index].sum(axis=1)
    return values.mean(), *np.quantile(draws, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2])


def dataset_summary(source, loaded, output):
    paths = {
        "tahoe": (source / "final_data/filtered_pseudobulks_alpha_10000.jld2", "df"),
        "lincs": (source / "final_data/filtered_lincs.jld2", "df"),
    }
    rows = []
    counts = {}
    for dataset, (path, key) in paths.items():
        meta = jld2_columns(path, key, ["cell_line", "plate", "drug", "dose", "time"])
        treated = meta[meta.drug != "DMSO"]
        counts[dataset] = {"untreated": int((meta.drug == "DMSO").sum()), "treated": len(treated)}
        metrics = {
            "untreated_profiles": counts[dataset]["untreated"], "treated_profiles": len(treated),
            "genes": GENES[dataset], "cell_lines": treated.cell_line.nunique(), "compounds": treated.drug.nunique(),
            "doses": treated.dose.nunique(), "exposure_times": treated.time.nunique(), "plates": meta.plate.nunique(),
        }
        best = loaded.best_runs.loc[loaded.best_runs.exp_key.eq(DEFAULT_EXPERIMENT[dataset])].iloc[0]
        metrics["model_observations"] = int(best.n_train + best.n_val + best.n_test)
        for metric, value in metrics.items():
            rows.append({"record_type": "inventory", "dataset": dataset, "filter_stage": "post_preprocessing",
                         "metric": metric, "value": value, "unit": "count", "n": value, "notes": ""})
        reported = PROFILE_DISTRIBUTIONS[dataset]
        rows.append({"record_type": "profile_distribution", "dataset": dataset, "filter_stage": "post_preprocessing",
                     "metric": "delta_profile_values", "value": reported["mean"], "unit": "normalized_expression",
                     "n": len(treated) * GENES[dataset], "notes": "Values transcribed from manuscript; quartiles were unavailable",
                     "profile_type": "delta", **reported, "q25": np.nan, "median": np.nan, "q75": np.nan,
                     "n_profiles": len(treated), "n_values": len(treated) * GENES[dataset]})
    columns = ["record_type", "dataset", "filter_stage", "metric", "value", "unit", "n", "notes",
               "profile_type", "mean", "sd", "minimum", "q25", "median", "q75", "maximum", "n_profiles", "n_values"]
    csv_write(pd.DataFrame(rows).reindex(columns=columns), output / "dataset_summary.csv")
    return counts


def selected_models(source, loaded, output):
    trials = loaded.runs_df.groupby("exp_key").size()
    native = {}
    for representation in REPRESENTATIONS:
        path = source / "embeddings/tahoe" / representation / "dataframe.parquet"
        if path.is_file():
            value = parquet_first(path, "embedding")
            native[representation] = len(value) // 4 if isinstance(value, bytes) else len(value)
    rows = []
    for row in loaded.best_runs.itertuples(index=False):
        run = resolve_run(source, row.run_dir)
        summary = tomli.loads((run / "summary.toml").read_text())
        rep = representation_id(row._asdict())
        native_dim = GENES[str(row.dataset).lower()] if rep == "biopert" else native.get(rep)
        effective_treatment = row.n_pca_expr if rep == "biopert" else row.n_pca_molec
        effective_treatment = native_dim if pd.isna(effective_treatment) else effective_treatment
        effective_cell = row.n_pca_expr if pd.notna(row.n_pca_expr) else GENES[str(row.dataset).lower()]
        hidden = [int(x) for x in re.findall(r"\d+", str(row.hidden_layers))]
        dims = [int(row.input_dim), *hidden, int(row.output_dim)] if pd.notna(row.input_dim) and pd.notna(row.output_dim) else []
        params = sum(a * b + b for a, b in zip(dims, dims[1:])) if dims else np.nan
        reference = summary.get("ref_cl", DEFAULT_REFERENCE[str(row.dataset).lower()])
        rows.append({
            "analysis_id": row.exp_key, "dataset": str(row.dataset).lower(), "experiment_id": row.category,
            "configuration_id": row.exp_key, "representation_id": rep, "dose_encoding": summary.get("dose_encoding", "none"),
            "time_encoding": "one_hot" if summary.get("encode_time", False) else "none",
            "gene_space": "landmark" if row.dataset == "LINCS" or row.landmark_genes_only else "full",
            "reference_cell_line_id": reference, "reference_handling": "average" if row.average_ref else "resample" if row.resample_ref else "fixed_single",
            "reproducibility_threshold": parse_number(row.label) if row.category == "repro" else np.nan,
            "selection_metric": "validation_spearman", "n_trials": int(trials.get(row.exp_key, 0)),
            "learning_rate": row.lr, "batch_size": row.batch_size, "weight_decay": row.weight_decay,
            "hidden_layers": row.hidden_layers, "max_epochs": row.n_epochs, "selected_epoch": row.selected_epoch,
            "n_pca_expr": row.n_pca_expr, "n_pca_molecule": row.n_pca_molec,
            "native_treatment_dimension": native_dim, "effective_treatment_dimension": effective_treatment,
            "effective_cell_dimension": effective_cell, "input_dimension": row.input_dim, "output_dimension": row.output_dim,
            "parameter_count": params, "n_train": row.n_train, "n_validation": row.n_val, "n_test": row.n_test,
            "seed": summary.get("seed", SEED), "checkpoint_id": f"{run.name}:best_model.jld2",
        })
    frame = pd.DataFrame(rows).sort_values(["dataset", "experiment_id", "configuration_id"])
    csv_write(frame, output / "selected_models.csv")
    return frame


def representations(source, loaded, output):
    rows = []
    successful = {}
    for dataset in DATASETS:
        paths = [source / "embeddings" / dataset / rep / "dataframe.parquet" for rep in REPRESENTATIONS]
        inputs = set()
        for path in paths:
            if path.is_file():
                inputs.update(parquet_read(path, ["smiles"]).smiles.dropna().astype(str))
        for rep, path in zip(REPRESENTATIONS, paths):
            if path.is_file():
                smiles = parquet_read(path, ["smiles"]).smiles
                value = parquet_first(path, "embedding")
                successful[(dataset, rep)] = (len(inputs), smiles.nunique(), len(value) // 4 if isinstance(value, bytes) else len(value))
            else:
                successful[(dataset, rep)] = (len(inputs), 0, np.nan)
    best = loaded.best_runs.set_index("exp_key")
    for rep in REPRESENTATIONS:
        fingerprint = rep.startswith(("Morgan_", "RDKit_", "AtomPair_")) or rep == "MACCS_166"
        family = "fingerprint" if fingerprint else "random_control" if rep == "random_512" else "foundation_model"
        source_name = MODEL_SOURCES.get(rep, "RDKit" if fingerprint else "NumPy PCG64")
        row = {
            "representation_id": rep, "display_name": DISPLAY_NAMES.get(rep, rep), "family": family,
            "checkpoint_or_source": source_name, "source_version": "NA", "input_format": "SELFIES" if rep == "MolGen-large" else "SMILES",
            "native_dimension": successful[("tahoe", rep)][2], "pooling_rule": "mean_nonpadding_tokens" if rep in MODEL_SOURCES and not rep.startswith("unimol") and rep != "CheMeleon" else "NA",
            "distance_metric": "tanimoto" if fingerprint else "cosine", "fingerprint_radius": 3 if rep.startswith("Morgan_") else np.nan,
            "fingerprint_length": int(rep.rsplit("_", 1)[1]) if fingerprint else np.nan,
            "conformer_required": rep.startswith("unimol"), "conformer_procedure": "NA", "unimol_tools_version": "NA",
            "random_seed": SEED if rep == "random_512" else np.nan,
            "citation_or_url": source_name if str(source_name).startswith("http") else f"https://huggingface.co/{source_name}" if "/" in str(source_name) else "NA",
        }
        failures = []
        for dataset in DATASETS:
            n_input, n_success, _ = successful[(dataset, rep)]
            keys = [key for key in best.index if key.startswith(f"{dataset}_{rep}") and key.endswith("_dose_gate")]
            eligible = best.loc[keys[0], "n_train"] if keys else np.nan
            row[f"{dataset}_n_input"] = n_input
            row[f"{dataset}_n_successful"] = n_success
            row[f"{dataset}_n_eligible_observations"] = eligible
            if n_success != n_input:
                failures.append(f"{dataset}:{n_input - n_success}")
        row["failure_summary"] = ";".join(failures) if failures else ""
        rows.append(row)
    csv_write(pd.DataFrame(rows), output / "representations.csv")


def split_assignments(source, loaded, names, output):
    rows = []
    for dataset in DATASETS:
        key = f"{dataset}_Morgan_1024_dose_gate"
        frames = [loaded.load_perobs_single(key, split=split).assign(split=split) for split in ["train", "val", "test"]]
        frame = pd.concat(frames, ignore_index=True)
        compound = frame[["drug", "smiles", "split"]].drop_duplicates()
        if compound.groupby("smiles").split.nunique().max() != 1:
            raise ValueError(f"{dataset}: compound split is not unique")
        for row in compound.itertuples(index=False):
            rows.append({"dataset": dataset, "type": "compound", "id": str(row.drug), "display_name": str(row.drug),
                         "split": row.split, "cluster_id": np.nan, "split_seed": SEED,
                         "canonical_smiles_hash": hashlib.sha256(str(row.smiles).encode()).hexdigest(), "cellosaurus_id": np.nan,
                         "is_reference_candidate": False, "eligible_default": True, "ineligibility_reason": ""})
        val_path = source / f"results/obs/{dataset}_cellline_val.jld2"
        test_path = source / f"results/obs/{dataset}_cellline_test.jld2"
        with h5py.File(val_path) as handle:
            val = {x.decode() for x in handle["cell_lines"][()]}
        with h5py.File(test_path) as handle:
            test = {x.decode() for x in handle["cell_lines"][()]}
        cell_lines = set(frame.cell_line)
        for cell_line in sorted(cell_lines):
            split = "test" if cell_line in test else "val" if cell_line in val else "train"
            rows.append({"dataset": dataset, "type": "cell_line", "id": cell_line,
                         "display_name": names.get(cell_line, cell_line), "split": split, "cluster_id": np.nan,
                         "split_seed": SEED, "canonical_smiles_hash": np.nan,
                         "cellosaurus_id": cell_line if dataset == "tahoe" else np.nan,
                         "is_reference_candidate": cell_line == DEFAULT_REFERENCE[dataset],
                         "eligible_default": cell_line != DEFAULT_REFERENCE[dataset],
                         "ineligibility_reason": "default reference excluded from targets" if cell_line == DEFAULT_REFERENCE[dataset] else ""})
    frame = pd.DataFrame(rows).drop_duplicates(["dataset", "type", "id", "split"])
    csv_write(frame.sort_values(["dataset", "type", "id"]), output / "split_assignments.csv")


def variance_partition(source, output):
    shared = pd.read_csv(REPO / "data/lincs_and_tahoe_shared_genes.csv")
    tahoe_tokens = pd.read_csv(REPO / "data/tahoe_coding_tokens.csv")
    tahoe_genes = tahoe_tokens[tahoe_tokens.coding_tokens.isin(shared.token_id)].merge(shared, left_on="coding_tokens", right_on="token_id")
    lincs_genes = pd.read_csv(REPO / "data/lincs_gene_order.csv")
    maps = {
        "tahoe": tahoe_genes.reset_index(drop=True), "lincs": lincs_genes.reset_index(drop=True),
    }
    rows = []
    batch = source / "batch_effects"
    for dataset in DATASETS:
        for profile, prefix in [("absolute", dataset), ("delta", f"{dataset}_delta")]:
            chunks = sorted(batch.glob(f"{prefix}_vp_chunk*.csv"))
            path = batch / f"{prefix}_vp_per_gene.csv"
            frame = pd.concat([pd.read_csv(x) for x in chunks], ignore_index=True) if chunks else pd.read_csv(path)
            frame = frame.rename(columns={"cell_line:plate": "plate", "Residuals": "residual"}).sort_values("gene")
            genes = maps[dataset]
            if len(frame) != len(genes):
                raise ValueError(f"{prefix}: {len(frame)} variance rows but {len(genes)} gene annotations")
            symbols = genes.gene_symbol.astype(str).to_numpy()
            ids = genes.ensembl_id.astype(str).to_numpy()
            for component in [x for x in frame.columns if x != "gene"]:
                rows.append(pd.DataFrame({"dataset": dataset, "profile_type": profile, "gene_id": ids,
                                          "gene_symbol": symbols, "variance_component": component,
                                          "variance_fraction": frame[component].to_numpy() / 100, "fit_status": "ok"}))
    csv_write(pd.concat(rows, ignore_index=True), output / "variance_partition.csv.gz")


def histogram_rows(values, edges, metadata):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count, _ = np.histogram(values, edges)
    width = np.diff(edges)
    density = count / max(len(values), 1) / width
    peak = count / max(count.max(), 1)
    return pd.DataFrame({**metadata, "bin_left": edges[:-1], "bin_right": edges[1:],
                         "bin_midpoint": (edges[:-1] + edges[1:]) / 2, "count": count,
                         "density": density, "peak_normalized_density": peak,
                         "distribution_n": len(values), "distribution_mean": np.mean(values),
                         "distribution_median": np.median(values)})


def measurement_quality(source, output):
    conditions, distributions = [], []
    for dataset in DATASETS:
        for profile_source in ["untrt", "trt", "delta"]:
            for relation in ["intra", "inter"]:
                path = source / f"repro/{dataset}/repro_{profile_source}_{relation}.csv"
                columns = ["cell_line", "drug", "dose", "time", "rep_i", "rep_j", "norm_rep_i", "norm_rep_j", "pearson"]
                frame = pd.read_csv(path, usecols=columns)
                profile = "delta" if profile_source == "delta" else "absolute"
                keys = ["cell_line", "drug", "dose", "time"]
                grouped = frame.groupby(keys, dropna=False, observed=True)
                summary = grouped.agg(n_pairs_total=("pearson", "size"), n_pairs_used=("pearson", "count"),
                                      mean_replicate_pearson=("pearson", "mean"), median_replicate_pearson=("pearson", "median"),
                                      magnitude_i=("norm_rep_i", "mean"), magnitude_j=("norm_rep_j", "mean")).reset_index()
                profiles = grouped.apply(lambda x: len(set(x.rep_i).union(x.rep_j)), include_groups=False).rename("n_profiles").reset_index()
                summary = summary.merge(profiles, on=keys)
                summary["record_type"] = "condition"
                summary["dataset"] = dataset
                summary["condition_id"] = [stable_id(dataset, *x, prefix="condition_") for x in summary[keys].itertuples(index=False, name=None)]
                summary["compound_id"] = summary.drug
                summary["cell_line_id"] = summary.cell_line
                summary["dose_um"] = summary.dose.map(parse_number)
                summary["exposure_time_h"] = summary.time.map(parse_number)
                summary["profile_type"] = profile
                summary["plate_relation"] = relation
                summary["response_magnitude"] = summary[["magnitude_i", "magnitude_j"]].mean(axis=1)
                summary["subsampling_seed"] = SEED if dataset == "lincs" else np.nan
                conditions.append(summary)
                if profile == "absolute":
                    edges = np.linspace(0, 1.02, 82)
                    groups = [(frame, "untreated" if profile_source == "untrt" else "treated", "all")]
                elif relation == "intra" and dataset == "lincs":
                    edges = np.linspace(-0.3, 1.02, 100)
                    high = (frame.dose == "20 uM") & (frame.time == "24 h")
                    medium = (frame.dose == "10 uM") & (frame.time == "24 h")
                    groups = [(frame[~(high | medium)], "other", "other"), (frame[medium], "10", "24"), (frame[high], "20", "24")]
                else:
                    edges = np.linspace(-0.3, 1.02, 100)
                    groups = [(frame, "all", "all")]
                for subset, dose_group, time_group in groups:
                    distributions.append(histogram_rows(subset.pearson, edges, {
                        "record_type": "distribution_bin", "dataset": dataset, "profile_type": profile,
                        "plate_relation": relation, "dose_group": dose_group, "exposure_time_group": time_group,
                    }))
    condition_columns = ["record_type", "dataset", "condition_id", "compound_id", "cell_line_id", "dose_um",
                         "exposure_time_h", "profile_type", "plate_relation", "n_profiles", "n_pairs_total", "n_pairs_used",
                         "mean_replicate_pearson", "median_replicate_pearson", "response_magnitude", "subsampling_seed"]
    distribution_columns = ["record_type", "dataset", "profile_type", "plate_relation", "dose_group", "exposure_time_group",
                            "bin_left", "bin_right", "bin_midpoint", "count", "density", "peak_normalized_density",
                            "distribution_n", "distribution_mean", "distribution_median"]
    condition = pd.concat(conditions, ignore_index=True).reindex(columns=condition_columns + [x for x in distribution_columns if x not in condition_columns])
    distribution = pd.concat(distributions, ignore_index=True).reindex(columns=condition.columns)
    result = pd.concat([condition, distribution], ignore_index=True)
    csv_write(result, output / "measurement_quality.csv.gz")
    return condition


def sar_pairs(source, output):
    frames = []
    for dataset in DATASETS:
        frame = pd.read_csv(source / f"sar_metric_fix/{dataset}/sar.csv", sep=";")
        frame["dataset"] = dataset
        frame["assay_id"] = [stable_id(dataset, *x, prefix="assay_") for x in frame[["cell_line", "dose", "time", "plate"]].itertuples(index=False, name=None)]
        frame["pair_id"] = [stable_id(dataset, assay, *sorted([str(a), str(b)]), prefix="pair_") for assay, a, b in frame[["assay_id", "drug_i", "drug_j"]].itertuples(index=False, name=None)]
        frame["compound_id_1"] = frame.drug_i.astype(str)
        frame["compound_id_2"] = frame.drug_j.astype(str)
        frame["cell_line_id"] = frame.cell_line.astype(str)
        frame["dose_um"] = frame.dose.map(parse_number)
        frame["exposure_time_h"] = frame.time.map(parse_number)
        frame["response_pearson"] = frame.pearson
        frame["magnitude_1"] = frame.delta_norm_i
        frame["magnitude_2"] = frame.delta_norm_j
        rename = {f"{rep}_dist": f"{rep}_distance" for rep in REPRESENTATIONS}
        frame = frame.rename(columns=rename)
        columns = ["dataset", "pair_id", "assay_id", "compound_id_1", "compound_id_2", "cell_line_id", "dose_um",
                   "exposure_time_h", "response_pearson", "magnitude_1", "magnitude_2"] + [f"{rep}_distance" for rep in REPRESENTATIONS]
        frames.append(frame.reindex(columns=columns))
    csv_write(pd.concat(frames, ignore_index=True), output / "sar_pairs.csv.gz")


def performance_metrics(loaded, output):
    source = pd.read_csv(REPO / "nbs/final_figures/stats/experiment_metric_cis.csv")
    selected = loaded.best_runs.set_index("exp_key")
    run_stats = loaded.run_stats.copy()
    run_stats["run_name"] = run_stats.run_dir.astype(str).map(lambda path: Path(path).name)
    rows = []
    for row in source.itertuples(index=False):
        model = selected.loc[row.exp_key] if row.exp_key in selected.index else None
        metric = str(row.metric).removeprefix("test_")
        source_metric = "cosine" if metric == "cosine_sim" else metric
        match = pd.DataFrame()
        fallback = None
        if model is not None:
            run_name = Path(str(model.run_dir)).name
            match = run_stats[(run_stats.exp_key == row.exp_key) & (run_stats.run_name == run_name) & (run_stats.split == "test")]
            if len(match) != 1:
                try:
                    fallback = loaded.load_perobs_single(row.exp_key, "test")
                except FileNotFoundError:
                    pass
        n_obs = int(model.n_test) if model is not None else np.nan
        rows.append({"analysis_id": row.exp_key, "dataset": str(row.dataset).lower(), "predictor_id": representation_id(row._asdict()),
                     "predictor_type": "model", "experiment_id": row.category, "configuration_id": row.exp_key,
                     "metric": metric, "split": "test", "grouping_variable": "all", "group_value": "all",
                     "n_observations": n_obs, "n_compounds": np.nan, "n_cell_lines": np.nan,
                     "estimate": row.estimate,
                     "sd": match.iloc[0][f"{source_metric}_std"] if len(match) == 1 else
                           fallback[source_metric].std(ddof=0) if fallback is not None else np.nan,
                     "checkpoint_id": f"{Path(str(model.run_dir)).name}:best_model.jld2" if model is not None else np.nan,
                     "source_status": "final-figure mean; selected-run SD" if len(match) == 1 else
                                      "final-figure mean; SD computed from selected-run observations" if fallback is not None else
                                      "final-figure mean; selected-run SD missing"})
    for model in selected.reset_index().itertuples(index=False):
        run_name = Path(str(model.run_dir)).name
        checkpoint = f"{run_name}:best_model.jld2"
        for split, count_column in [("train", "n_train"), ("val", "n_val")]:
            match = run_stats[(run_stats.exp_key == model.exp_key) & (run_stats.run_name == run_name) & (run_stats.split == split)]
            fallback = None
            if len(match) != 1:
                try:
                    fallback = loaded.load_perobs_single(model.exp_key, split)
                except FileNotFoundError:
                    pass
            for metric, source_metric in [("pearson", "pearson"), ("spearman", "spearman"), ("l2", "l2"), ("cosine_sim", "cosine")]:
                available = len(match) == 1
                direct = fallback is not None
                rows.append({"analysis_id": model.exp_key, "dataset": str(model.dataset).lower(),
                             "predictor_id": representation_id(model._asdict()), "predictor_type": "model",
                             "experiment_id": model.category, "configuration_id": model.exp_key, "metric": metric,
                             "split": split, "grouping_variable": "all", "group_value": "all",
                             "n_observations": len(fallback) if direct else getattr(model, count_column),
                             "n_compounds": fallback.drug.nunique() if direct else np.nan,
                             "n_cell_lines": fallback.cell_line.nunique() if direct else np.nan,
                             "estimate": match.iloc[0][f"{source_metric}_mean"] if available else fallback[source_metric].mean() if direct else np.nan,
                             "sd": match.iloc[0][f"{source_metric}_std"] if available else fallback[source_metric].std(ddof=0) if direct else np.nan,
                             "checkpoint_id": checkpoint,
                             "source_status": "selected-run aggregate" if available else
                                              "computed from selected-run observations" if direct else
                                              "selected-run source missing"})
    for dataset in DATASETS:
        for baseline, predictor_type in [("delta_ref", "reference_delta_baseline"), ("mean_delta", "train_delta_baseline")]:
            frame = loaded.baselines_df[(loaded.baselines_df.dataset == DATASET_LABEL[dataset]) & (loaded.baselines_df.baseline == baseline)]
            for metric in ["pearson", "spearman", "l2"]:
                rows.append({"analysis_id": "representation_comparison", "dataset": dataset, "predictor_id": baseline,
                             "predictor_type": predictor_type, "experiment_id": "baseline", "configuration_id": baseline,
                             "metric": metric, "split": "test", "grouping_variable": "all", "group_value": "all",
                             "n_observations": len(frame), "n_compounds": frame.smiles.nunique(), "n_cell_lines": frame.cell_line.nunique(),
                             "estimate": frame[metric].mean(), "sd": frame[metric].std(ddof=0),
                             "checkpoint_id": np.nan, "source_status": "baseline aggregate"})
    csv_write(pd.DataFrame(rows), output / "performance_metrics.csv")


def reproducibility_performance(loaded, selected, output):
    repro = pd.read_parquet(REPO / "nbs/cache_repro_obs.parquet")
    edges = np.arange(-0.4, 1.0001, 0.1)
    rows = []
    models = selected[(selected.dataset == "lincs") & selected.experiment_id.isin(["ablation", "embedding"])]
    for model in models.itertuples(index=False):
        frame = loaded.load_perobs_single(model.configuration_id, "test").merge(repro, on=["cell_line", "drug", "dose", "time"], how="inner")
        frame["bin"] = pd.cut(frame.repro_pearson, edges)
        for interval, group in frame.groupby("bin", observed=True):
            if len(group) < 20:
                continue
            mean, low, high = bootstrap_mean(group.pearson)
            rows.append({"record_type": "reproducibility_association", "dataset": "lincs", "configuration_id": model.configuration_id,
                         "experiment_id": model.experiment_id, "training_reproducibility_threshold": np.nan,
                         "test_bin_left": interval.left, "test_bin_right": interval.right, "test_bin_midpoint": interval.mid,
                         "minimum_bin_n": 20, "n_test_observations": len(group), "n_test_compounds": group.smiles.nunique(),
                         "mean_test_pearson": mean, "ci_lower": low, "ci_upper": high, "n_train_observations": model.n_train,
                         "n_train_compounds": np.nan, "fraction_train_observations_retained": np.nan,
                         "fraction_train_compounds_retained": np.nan, "bootstrap_unit": "observation", "n_bootstrap": N_BOOT, "seed": STAT_SEED})
    thresholds = [(np.nan, "lincs_default_config"), (0.0, "lincs_default_repro_0"),
                  *[(value, f"lincs_default_repro_{value}") for value in [0.1, 0.2, 0.3, 0.4, 0.5]]]
    base_train = loaded.load_perobs_single("lincs_default_config", "train")
    for threshold, key in thresholds:
        train = loaded.load_perobs_single(key, "train")
        test = loaded.load_perobs_single(key, "test").merge(repro, on=["cell_line", "drug", "dose", "time"], how="inner")
        test["bin"] = pd.cut(test.repro_pearson, edges)
        for interval, group in test.groupby("bin", observed=True):
            if len(group) < 100:
                continue
            mean, low, high = bootstrap_mean(group.pearson)
            rows.append({"record_type": "training_filter", "dataset": "lincs", "configuration_id": key, "experiment_id": "reproducibility_filter",
                         "training_reproducibility_threshold": threshold, "test_bin_left": interval.left, "test_bin_right": interval.right,
                         "test_bin_midpoint": interval.mid, "minimum_bin_n": 100, "n_test_observations": len(group),
                         "n_test_compounds": group.smiles.nunique(), "mean_test_pearson": mean, "ci_lower": low, "ci_upper": high,
                         "n_train_observations": len(train), "n_train_compounds": train.smiles.nunique(),
                         "fraction_train_observations_retained": len(train) / len(base_train),
                         "fraction_train_compounds_retained": train.smiles.nunique() / base_train.smiles.nunique(),
                         "bootstrap_unit": "observation", "n_bootstrap": N_BOOT, "seed": STAT_SEED})
    csv_write(pd.DataFrame(rows), output / "reproducibility_performance.csv")


def metric_group(group, analysis_id, dataset, configuration, experiment, setting, seen_compound, seen_cell, reference, handling, names, baseline=None):
    mean, low, high = bootstrap_mean(group.pearson)
    ref = baseline[0].mean() if baseline is not None else np.nan
    train = baseline[1].mean() if baseline is not None else np.nan
    return {"analysis_id": analysis_id, "dataset": dataset, "target_cell_line_id": group.cell_line.iloc[0],
            "target_cell_line_name": names.get(group.cell_line.iloc[0], group.cell_line.iloc[0]), "configuration_id": configuration,
            "experiment_id": experiment, "generalization_setting": setting, "compound_seen_in_training": seen_compound,
            "cell_line_seen_in_training": seen_cell, "reference_cell_line_id": reference,
            "reference_cell_line_name": names.get(reference, reference), "reference_handling": handling,
            "n_observations": len(group), "n_compounds": group.smiles.nunique(), "mean_pearson": mean,
            "pearson_ci_lower": low, "pearson_ci_upper": high, "mean_spearman": group.spearman.mean(), "mean_l2": group.l2.mean(),
            "reference_delta_mean_pearson": ref, "train_delta_mean_pearson": train,
            "gain_over_reference_delta": mean - ref, "gain_over_train_delta": mean - train,
            "bootstrap_unit": "observation", "n_bootstrap": N_BOOT, "seed": STAT_SEED}


def cell_line_performance(source, loaded, names, output):
    rows = []
    for dataset in DATASETS:
        key = f"{dataset}_cellline_holdout"
        frame = loaded.load_quadrant_baselines(key, metric="pearson")
        for (quadrant, cell), group in frame.groupby(["quadrant", "cell_line"]):
            rows.append(metric_group(group, "cell_line_generalization", dataset, key, "holdout", quadrant,
                                     bool(group.seen_compound.iloc[0]), bool(group.seen_cell_line.iloc[0]),
                                     DEFAULT_REFERENCE[dataset], "fixed_single", names,
                                     (group["pearson__delta_ref"], group["pearson__mean_delta"])))
    references = {
        "tahoe": [("tahoe_default_config", "CVCL_0023"), ("tahoe_CVCL_0480", "CVCL_0480"),
                  ("tahoe_CVCL_0131", "CVCL_0131"), ("tahoe_CVCL_0218", "CVCL_0218")],
        "lincs": [("lincs_default_config", "A549"), ("lincs_MCF7", "MCF7"), ("lincs_PC3", "PC3")],
    }
    for dataset, configs in references.items():
        for key, reference in configs:
            frame = loaded.load_perobs_single(key, "test")
            for _, group in frame.groupby("cell_line"):
                rows.append(metric_group(group, "reference_choice", dataset, key, "reference_cell", "compound_holdout",
                                         False, True, reference, "fixed_single", names))
    handling = [("lincs_default_config", "fixed_single"), ("lincs_resample_ref", "resample"), ("lincs_average_ref", "average")]
    for key, method in handling:
        frame = loaded.load_perobs_single(key, "test")
        for _, group in frame.groupby("cell_line"):
            rows.append(metric_group(group, "reference_handling", "lincs", key, "reference_handling", "compound_holdout",
                                     False, True, "A549", method, names))
    main = loaded.load_perobs_single("tahoe_default_config", "test")
    for _, group in main.groupby("cell_line"):
        rows.append(metric_group(group, "sequencing_depth", "tahoe", "tahoe_default_config", "main", "compound_holdout",
                                 False, True, "CVCL_0023", "fixed_single", names))
    frame = pd.DataFrame(rows)
    csv_write(frame, output / "cell_line_performance.csv")
    return frame


def sequencing_depth(source, loaded, names, output):
    test = loaded.load_perobs_single("tahoe_default_config", "test")
    raw = jld2_columns(source / "final_data/filtered_pseudobulks_alpha_10000.jld2", "df",
                       ["cell_line", "drug", "dose", "time"])
    keys = test[["cell_line", "drug", "dose", "time"]].drop_duplicates()
    matched = raw.merge(keys, on=["cell_line", "drug", "dose", "time"], how="inner")
    rows = []
    for cell, group in test.groupby("cell_line"):
        values = group.total_umis.dropna()
        rows.append({"dataset": "tahoe", "target_cell_line_id": cell, "target_cell_line_name": names.get(cell, cell),
                     "n_profiles": int((matched.cell_line == cell).sum()), "n_test_observations": len(group),
                     "mean_umi_count": values.mean(), "median_umi_count": values.median(), "q25_umi_count": values.quantile(.25),
                     "q75_umi_count": values.quantile(.75), "included": len(values) > 0,
                     "exclusion_reason": "" if len(values) else "missing UMI count"})
    csv_write(pd.DataFrame(rows), output / "sequencing_depth.csv")


def systema_summary(source, output):
    paths = {"cell_line": source / "results/systema_tahoe/analysis/metrics_per_condition.csv",
             "cell_line_dose": source / "results/systema_tahoe/analysis_by_dose/metrics_per_condition.csv"}
    rows = []
    for context, path in paths.items():
        frame = pd.read_csv(path)
        specifications = [("centered_pearson", "pearson_systema"), ("centroid_accuracy", "centroid_accuracy")]
        if context == "cell_line":
            specifications.append(("uncentered_pearson", "pearson_control"))
        for method, method_frame in frame.groupby("method"):
            for metric, column in specifications:
                for grouping, groups in [("all", [("all", method_frame)]), ("dose_um", method_frame.groupby("dose"))]:
                    for value, group in groups:
                        finite = group[column].notna()
                        estimate, low, high = bootstrap_mean(group.loc[finite, column], group.loc[finite, "drug"])
                        rows.append({"dataset": "tahoe", "predictor_id": method, "centering_context": "none" if metric == "uncentered_pearson" else context,
                                     "grouping_variable": grouping, "group_value": parse_number(value) if grouping == "dose_um" else value,
                                     "metric": metric, "n_contexts": len(group), "n_eligible_contexts": int(finite.sum()),
                                     "n_excluded_contexts": int((~finite).sum()), "n_conditions": len(group), "n_compounds": group.drug.nunique(),
                                     "n_test_candidates": len(group), "estimate": estimate, "ci_lower": low, "ci_upper": high,
                                     "ci_level": CI_LEVEL, "bootstrap_unit": "compound", "n_bootstrap": N_BOOT, "seed": SEED})
    csv_write(pd.DataFrame(rows), output / "systema_summary.csv")


def reference_similarity(source, names, output):
    frame = pd.read_csv(source / "results/ref_target_delta_corr.csv")
    frame = frame.rename(columns={"cl_i": "reference_cell_line_id", "cl_j": "target_cell_line_id",
                                  "n_conditions": "n_shared_conditions", "mean_delta_pearson": "mean_response_pearson"})
    frame["dataset"] = "tahoe"
    frame["reference_cell_line_name"] = frame.reference_cell_line_id.map(names).fillna(frame.reference_cell_line_id)
    frame["target_cell_line_name"] = frame.target_cell_line_id.map(names).fillna(frame.target_cell_line_id)
    frame["median_response_pearson"] = np.nan
    frame["included"] = frame.n_shared_conditions >= 2
    frame["exclusion_reason"] = np.where(frame.included, "", "fewer than two shared conditions")
    columns = ["dataset", "reference_cell_line_id", "reference_cell_line_name", "target_cell_line_id", "target_cell_line_name",
               "n_shared_conditions", "mean_response_pearson", "median_response_pearson", "included", "exclusion_reason"]
    csv_write(frame[columns], output / "reference_target_similarity.csv")


def gmt_sizes(path):
    result = {}
    with open(path) as handle:
        for line in handle:
            values = line.rstrip().split("\t")
            result[values[0]] = len(set(values[2:]))
    return result


def pathway_files(source, output):
    root = source / "results/tahoe_pathway_cases"
    selection = pd.read_csv(root / "candidate_selection/condition_metrics.csv")
    shortlist = pd.read_csv(root / "gsea/shortlist_annotated.csv")
    enrichment = pd.read_csv(root / "gsea/pathway_results.csv")
    annotations = pd.read_csv(REPO / "notebooks/tahoe_pathway_case_annotations.csv")
    selected = shortlist[["cell_line", "drug", "dose", "time", "case_id"]]
    candidates = selection.merge(selected, on=["cell_line", "drug", "dose", "time"], how="left")
    candidates["case_id"] = [case if pd.notna(case) else stable_id(cell, drug, dose, time, prefix="candidate_")
                             for case, cell, drug, dose, time in candidates[["case_id", "cell_line", "drug", "dose", "time"]].itertuples(index=False, name=None)]
    metrics = [x for x in selection.columns if x not in ["cell_line", "drug", "smiles", "dose", "time"]]
    candidate_rows = pd.DataFrame({
        "record_type": "candidate", "case_id": candidates.case_id, "dataset": "tahoe", "compound_id": candidates.drug,
        "compound_name": candidates.drug, "dose_um": candidates.dose.map(parse_number), "exposure_time_h": candidates.time.map(parse_number),
        "target_cell_line_id": candidates.cell_line, "reference_cell_line_id": "CVCL_0023",
        "selection_metrics": [json.dumps({key: None if pd.isna(row[key]) else row[key] for key in metrics}, sort_keys=True) for _, row in candidates.iterrows()],
        "selection_rule_passed": candidates.case_id.str.startswith("case_"), "biology_review_status": np.nan,
        "included_in_manuscript": candidates.case_id.eq("case_01"),
        "exclusion_reason": np.where(candidates.case_id.str.startswith("case_"), "", "not selected by fixed top-10 rule"),
    })
    metadata = shortlist.merge(annotations[["case_id", "interpretation_strength"]], on="case_id", how="left")
    sizes = {"hallmark": gmt_sizes(root / "gene_sets/MSigDB_Hallmark_2020.gmt"),
             "reactome": gmt_sizes(root / "gene_sets/Reactome_2022.gmt")}
    enrichment = enrichment.merge(metadata[["case_id", "drug", "dose", "time", "cell_line"]], on="case_id", how="left")
    enrichment_rows = pd.DataFrame({
        "record_type": "enrichment", "case_id": enrichment.case_id, "dataset": "tahoe", "compound_id": enrichment.drug,
        "compound_name": enrichment.drug, "dose_um": enrichment.dose.map(parse_number), "exposure_time_h": enrichment.time.map(parse_number),
        "target_cell_line_id": enrichment.cell_line, "reference_cell_line_id": "CVCL_0023", "selection_metrics": np.nan,
        "selection_rule_passed": True, "biology_review_status": np.nan, "included_in_manuscript": enrichment.case_id.eq("case_01"),
        "exclusion_reason": "", "profile_type": enrichment.profile.map(PATHWAY_PROFILES), "collection_name": enrichment.collection,
        "collection_version": enrichment.collection.map({"hallmark": "MSigDB Hallmark 2020", "reactome": "Reactome 2022"}),
        "pathway_id": enrichment.pathway, "pathway_name": enrichment.pathway,
        "n_genes_present": [sizes[c].get(p, np.nan) for c, p in enrichment[["collection", "pathway"]].itertuples(index=False, name=None)],
        "enrichment_score": enrichment.es, "normalized_enrichment_score": enrichment.nes, "p_raw": enrichment.nominal_p,
        "q_value": enrichment.fdr_q, "leading_edge_size": enrichment.leading_edge_genes.fillna("").map(lambda x: len(str(x).split(";")) if x else 0),
    })
    columns = ["record_type", "case_id", "dataset", "compound_id", "compound_name", "dose_um", "exposure_time_h",
               "target_cell_line_id", "reference_cell_line_id", "selection_metrics", "selection_rule_passed", "biology_review_status",
               "included_in_manuscript", "exclusion_reason", "profile_type", "collection_name", "collection_version", "pathway_id",
               "pathway_name", "n_genes_present", "enrichment_score", "normalized_enrichment_score", "p_raw", "q_value", "leading_edge_size"]
    csv_write(pd.concat([candidate_rows, enrichment_rows], ignore_index=True).reindex(columns=columns), output / "pathway_enrichment.csv.gz")
    target_specific = pd.read_csv(root / "gsea/target_specific_pathways.csv")
    used = target_specific[["case_id", "collection", "pathway"]].drop_duplicates()
    evidence = enrichment.merge(used, on=["case_id", "collection", "pathway"], how="inner")
    ranking = {}
    for case in shortlist.case_id:
        for profile in PATHWAY_PROFILES:
            frame = pd.read_csv(root / f"gsea/{case}_{profile}_ranking.csv")
            frame["rank"] = np.arange(1, len(frame) + 1)
            ranking[(case, profile)] = frame.set_index("gene")
    gene_meta_path = Path("/network/scratch/l/lola.lebreton/huggingface/hub/datasets--tahoebio--Tahoe-100M/snapshots/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/gene_metadata.parquet")
    genes = pd.read_parquet(gene_meta_path)[["gene_symbol", "ensembl_id"]].drop_duplicates("gene_symbol").set_index("gene_symbol").ensembl_id.to_dict()
    rows = []
    for row in evidence.itertuples(index=False):
        for gene in str(row.leading_edge_genes).split(";"):
            if gene in ranking[(row.case_id, row.profile)].index:
                value = ranking[(row.case_id, row.profile)].loc[gene]
                rows.append({"case_id": row.case_id, "profile_type": PATHWAY_PROFILES[row.profile], "collection_name": row.collection,
                             "pathway_id": row.pathway, "gene_id": genes.get(gene), "gene_symbol": gene, "ranking_score": value.score,
                             "rank": value["rank"], "leading_edge": True, "reported_module": np.nan})
    for module, module_genes in MODULES.items():
        for profile in PATHWAY_PROFILES:
            frame = ranking[("case_01", profile)]
            for gene in module_genes:
                value = frame.loc[gene]
                rows.append({"case_id": "case_01", "profile_type": PATHWAY_PROFILES[profile], "collection_name": "reported_module",
                             "pathway_id": stable_id(module, prefix="module_"), "gene_id": genes.get(gene), "gene_symbol": gene,
                             "ranking_score": value.score, "rank": value["rank"], "leading_edge": False, "reported_module": module})
    csv_write(pd.DataFrame(rows).drop_duplicates(), output / "pathway_gene_scores.csv.gz")
    return enrichment_rows


def infer_stat_method(analysis, p):
    text = str(analysis).lower()
    if pd.isna(p):
        return np.nan
    if "friedman" in text:
        return "Friedman test"
    if "wilcoxon" in text or "paired" in text:
        return "paired Wilcoxon signed-rank test"
    if "spearman" in text:
        return "Spearman rank correlation"
    if "slope" in text or "curve" in text or "ols" in text:
        return "regression"
    return "NA"


def statistics_file(pathway, output):
    source = pd.read_csv(REPO / "nbs/final_figures/stats/all_statistics.csv")
    rows = []
    for index, row in source.iterrows():
        method = infer_stat_method(row.analysis, row.p)
        unit = "compound" if "compound-cluster" in str(row.analysis) else "cell_line" if "cell-line" in str(row.analysis) else "observation"
        rows.append({"analysis_id": row.figure, "item_id": stable_id(row.figure, row.analysis, row.group, row.term, prefix="stat_"),
                     "panel": row.figure, "dataset": str(row.group).lower() if str(row.group) in ["Tahoe", "LINCS"] else np.nan,
                     "family_id": row.figure, "contrast_id": row.term, "group_a": row.group, "group_b": row.term,
                     "estimand": row.analysis, "estimate": row.estimate, "effect_size": row.estimate if "effect size" in str(row.analysis).lower() else np.nan,
                     "ci_lower": row.ci_lo, "ci_upper": row.ci_hi, "ci_level": row.ci_level / 100 if row.ci_level > 1 else row.ci_level,
                     "bootstrap_unit": unit, "ci_method": "percentile bootstrap" if pd.notna(row.ci_lo) else np.nan,
                     "n_bootstrap": N_BOOT if pd.notna(row.ci_lo) else np.nan, "hypothesis_test": method,
                     "statistic": row.estimate if pd.notna(row.p) else np.nan, "df": np.nan,
                     "sidedness": "two-sided" if pd.notna(row.p) else np.nan, "p_raw": row.p,
                     "p_adjustment": "Benjamini-Hochberg" if pd.notna(row.p_adj) else "none" if pd.notna(row.p) else np.nan,
                     "p_adjusted": row.p_adj, "analysis_unit": unit, "n": row.n, "seed": row.seed, "notes": "provisional"})
    if pathway is not None:
        for row in pathway.itertuples(index=False):
            rows.append({"analysis_id": "pathway_enrichment", "item_id": stable_id(row.case_id, row.profile_type, row.collection_name, row.pathway_id, prefix="stat_"),
                         "panel": "fig4e", "dataset": "tahoe", "family_id": f"{row.case_id}:{row.profile_type}:{row.collection_name}",
                         "contrast_id": row.pathway_id, "group_a": row.profile_type, "group_b": np.nan, "estimand": "normalized enrichment score",
                         "estimate": row.normalized_enrichment_score, "effect_size": row.normalized_enrichment_score, "ci_lower": np.nan,
                         "ci_upper": np.nan, "ci_level": np.nan, "bootstrap_unit": np.nan, "ci_method": np.nan, "n_bootstrap": np.nan,
                         "hypothesis_test": "GSEA permutation test", "statistic": row.enrichment_score, "df": np.nan, "sidedness": "two-sided",
                         "p_raw": row.p_raw, "p_adjustment": "GSEA FDR within profile and collection", "p_adjusted": row.q_value,
                         "analysis_unit": "pathway", "n": row.n_genes_present, "seed": SEED, "notes": "provisional biology review"})
    csv_write(pd.DataFrame(rows), output / "statistics.csv")


def analysis_counts(loaded, selected, output):
    groups = {}
    for dataset in DATASETS:
        frame = selected[selected.dataset.eq(dataset)]
        default = f"{dataset}_default_config"
        groups[(dataset, "representation_comparison")] = frame[
            frame.experiment_id.eq("embedding") | frame.configuration_id.isin([default, f"{dataset}_random_512"])
        ].configuration_id.tolist()
        groups[(dataset, "dose_encoding")] = frame[frame.experiment_id.eq("dose_encoding")].configuration_id.tolist()
        groups[(dataset, "reference_choice")] = [default, *frame[frame.experiment_id.eq("ref_cell")].configuration_id]
        groups[(dataset, "cell_line_generalization")] = frame[frame.experiment_id.eq("holdout")].configuration_id.tolist()
        if dataset == "tahoe":
            groups[(dataset, "gene_space")] = [default, "tahoe_default_landmark_genes"]
        else:
            groups[(dataset, "reference_handling")] = [default, *frame[frame.experiment_id.eq("ref_handling")].configuration_id]
            groups[(dataset, "reproducibility_filtering")] = [default, *frame[frame.experiment_id.eq("repro")].configuration_id]
    memberships = {}
    for group, configurations in groups.items():
        for configuration in configurations:
            memberships.setdefault(configuration, []).append(group)
    intersections = {group: None for group in groups}
    missing = {group: [] for group in groups}
    rows = []
    expected_columns = {"train": "n_train", "val": "n_validation", "test": "n_test"}
    for model in selected.itertuples(index=False):
        for split, expected_column in expected_columns.items():
            expected = getattr(model, expected_column)
            try:
                frame = loaded.load_perobs_single(model.configuration_id, split)
            except FileNotFoundError:
                rows.append({"analysis_id": model.configuration_id, "dataset": model.dataset,
                             "filter_stage": "eligible_model_input", "split": split, "n_configurations": 1,
                             "n_observations": expected, "n_compounds": np.nan, "n_cell_lines": np.nan,
                             "source_status": "per-observation file missing; observation count from run cache"})
                if split == "test":
                    for group in memberships.get(model.configuration_id, []):
                        missing[group].append(model.configuration_id)
                continue
            if pd.notna(expected) and len(frame) != int(expected):
                raise ValueError(f"Observation count mismatch for {model.configuration_id} {split}: {len(frame)} != {int(expected)}")
            rows.append({"analysis_id": model.configuration_id, "dataset": model.dataset, "filter_stage": "eligible_model_input",
                         "split": split, "n_configurations": 1, "n_observations": len(frame),
                         "n_compounds": frame.drug.nunique(), "n_cell_lines": frame.cell_line.nunique(), "source_status": "complete"})
            if split == "test":
                keys = set(frame[["cell_line", "drug", "dose", "time"]].astype(str).itertuples(index=False, name=None))
                for group in memberships.get(model.configuration_id, []):
                    intersections[group] = keys if intersections[group] is None else intersections[group].intersection(keys)
    for (dataset, analysis_id), keys in intersections.items():
        unavailable = missing[(dataset, analysis_id)]
        keys = keys or set()
        rows.append({"analysis_id": analysis_id, "dataset": dataset, "filter_stage": "common_test_intersection",
                     "split": "test", "n_configurations": len(groups[(dataset, analysis_id)]),
                     "n_observations": np.nan if unavailable else len(keys),
                     "n_compounds": np.nan if unavailable else len({row[1] for row in keys}),
                     "n_cell_lines": np.nan if unavailable else len({row[0] for row in keys}),
                     "source_status": f"missing test file: {','.join(unavailable)}" if unavailable else "complete"})
    csv_write(pd.DataFrame(rows).sort_values(["dataset", "analysis_id", "split"]), output / "analysis_counts.csv")


def readme(output, include_pathway):
    files = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "README.md")
    rows = {}
    for name in files:
        path = output / name
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            rows[name] = sum(1 for _ in handle) - 1
    generated = datetime.now(timezone.utc).isoformat()
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    software = ", ".join(f"{name} {software_version(name)}" for name in ["python", "numpy", "pandas", "scipy", "rdkit", "h5py", "gseapy"])
    index = "\n".join(f"- `{name}`: {rows[name]:,} rows" for name in files)
    pathway_note = "included" if include_pathway else "not generated by request"
    text = f"""# Supplementary data

Generated {generated} by `scripts/generate_supplementary_data.py` from commit `{commit}` (dirty worktree: `{dirty}`). Random split/model seed: {SEED}. Statistical seed: {STAT_SEED}. Bootstrap resamples: {N_BOOT}. Software observed during export: {software}.

The MolGen sweep, result cache, and figure statistics were provisional at generation time. Rerun this script after final sweeps and figure-statistic recomputation. Dataset inputs are Tahoe-100M Hugging Face snapshot `2dc57900b7981cfcf5e211527169a0b006546a95` and LINCS Phase II (LINCS 2020 beta), downloaded 2025-12-28. Exact pretrained-model revision hashes, executed extraction-environment versions, and UniMol conformer-generation provenance were unavailable and are `NA`. Pathway files are {pathway_note}; biology-review status is `NA`.

## Conventions

Files are UTF-8 CSV, with gzip for large tables. Missing values are `NA`. Dataset values are `tahoe` and `lincs`. Doses are micromolar (`dose_um`); times are hours (`exposure_time_h`). P values are numeric. Fractions range from 0 to 1. IDs prefixed `condition_`, `pair_`, `assay_`, `stat_`, `candidate_`, and `module_` are deterministic SHA-256 prefixes. `compound_id`, `cell_line_id`, `configuration_id`, `checkpoint_id`, and related identifiers are strings.

`split` is `train`, `val`, `test`, or `all`. `type` is `compound` or `cell_line`. `predictor_type` is `model`, `reference_delta_baseline`, or `train_delta_baseline`. `reference_handling` is `fixed_single`, `resample`, or `average`. `profile_type` is `absolute`, `delta`, `measured_target`, `biopert_prediction`, or `measured_reference`. `plate_relation` is `intra` or `inter`. `record_type` selects the row schema documented by the file plan.

## Files

{index}

`dataset_summary.csv` contains post-preprocessing inventory counts derived from the filtered JLD2 metadata. Delta-profile mean, SD, minimum, and maximum are the values reported in `manuscript.tex`; the unavailable median and quartiles are `NA`. `split_assignments.csv` uses observed pinned model splits; Butina cluster memberships were not persisted and are `NA`. `analysis_counts.csv` contains per-configuration split counts and common-test-intersection counts without observation-level records. Its `source_status` identifies incomplete inputs; cached observation counts are retained when detailed files are unavailable, while unavailable compound and cell-line counts are `NA`. `representations.csv` describes the 25 SAR representations. `selected_models.csv` records validation-selected checkpoints. `statistics.csv` consolidates final-figure statistics and pathway permutation/FDR results. `variance_partition.csv.gz` stores fractions from variancePartition/lme4 REML fits. `measurement_quality.csv.gz` contains condition summaries and publication histogram bins. `sar_pairs.csv.gz` stores filtered assay-specific unordered pairs. Remaining files contain the aggregate analyses named by their filenames.

`performance_metrics.csv` reports means and standard deviations for train, validation, and test splits. `source_status` distinguishes cached, directly computed, missing-source, final-figure, and baseline rows.

## Figure and table source map

| Output | Source rows |
|---|---|
| Main Fig. 2a-c; Supplementary SAR | `sar_pairs.csv.gz`; all rows or filter the desired `<representation_id>_distance` |
| Main Fig. 2d-e; robustness, landmark, dose and scaling supplements | `performance_metrics.csv`, `selected_models.csv`; filter `analysis_id`, `metric`, and `dataset` |
| Main Fig. 2f; Systema supplements | `systema_summary.csv`; filter `centering_context`, `predictor_id`, `metric`, and optional dose |
| Main Fig. 2g; cell-line generalization supplement | `cell_line_performance.csv`; `analysis_id=cell_line_generalization`, then filter `generalization_setting` |
| Main Fig. 3a; absolute reproducibility supplement | `measurement_quality.csv.gz`; `record_type=distribution_bin` |
| Main Fig. 3b; variance supplement | `variance_partition.csv.gz`; filter dataset/profile/component |
| Main Fig. 3c-e; reproducibility supplement | `reproducibility_performance.csv`; filter `record_type`, configuration, threshold, and bin |
| Main Fig. 3f | join `sequencing_depth.csv` to `cell_line_performance.csv` where `analysis_id=sequencing_depth` |
| Main Fig. 4a-b | `cell_line_performance.csv`; `analysis_id=reference_choice` |
| Main Fig. 4c | `cell_line_performance.csv`; `analysis_id=reference_handling` |
| Main Fig. 4d | join `reference_target_similarity.csv` to reference-choice rows |
| Main Fig. 4e-f; pathway supplement | `pathway_enrichment.csv.gz`, `pathway_gene_scores.csv.gz`; `case_id=case_01` for the main figure |
| Supplementary Tables 1-3 | `dataset_summary.csv`, `representations.csv`, `split_assignments.csv`, `analysis_counts.csv` |
| Supplementary Table 4 | `performance_metrics.csv`, `systema_summary.csv`, `cell_line_performance.csv` |
| Supplementary Tables 5-7 | `selected_models.csv`; filter dataset and `experiment_id` |
| Supplementary Table 8 | `statistics.csv` |

## Analysis definitions

Eligible model inputs are those emitted by the selected run's train/validation/test performance files after its configured input requirements. The default compound split is ECFP6/Butina with cutoff 0.4 and seed 42. Cell-line holdout uses the pinned validation/test lists and test-precedence union rule. A549 is the default reference and is excluded from prediction targets. Reference-choice analyses require matched reference responses. Common-test-intersection rows in `analysis_counts.csv` cover representation, dose-encoding, gene-space, reference-choice, reference-handling, reproducibility-filtering, and cell-line-generalization comparisons.

Replicate conditions are cell line, compound, dose, and time. Intra/inter designate same/different plates. LINCS pair generation was capped at 200 pairs per condition with seed 42. Absolute distribution bins use 81 equal intervals over [0, 1.02]. Delta bins use 99 equal intervals over [-0.3, 1.02]; LINCS intra-plate rows separate 10 uM/24 h, 20 uM/24 h, and other conditions. Prediction reproducibility bins are 0.1 wide over [-0.4, 1.0], retaining at least 20 observations for ordinary associations and 100 for filtering comparisons.

Systema centroids are estimated from training compounds within target-cell-line or target-cell-line-dose contexts. Finite metrics with at least two candidates are eligible. Confidence intervals resample compounds. Reference-target similarity averages gene-wise response Pearson correlations equally across shared conditions and requires at least two conditions.

Pathway candidates use the fixed maximin top-10 selection. Rankings are signed treated-minus-matched-DMSO scores. GSEA used gseapy 1.3.0, 5,000 permutations, weight 1, size 15-500, and seed 42 with MSigDB Hallmark 2020 and Reactome 2022. FDR values are those emitted within profile and collection. Leading-edge rows cover target-specific overlap analyses; reported-module rows cover the manuscript gene panel.

This directory is publication source data, not the computational archive. Raw expression matrices, all observed/predicted gene vectors, per-observation scores, embeddings, checkpoints, PCA objects, sweep histories, and bootstrap draws remain in the archive.
"""
    (output / "README.md").write_text(text)


def validate(output, include_pathway):
    expected = {
        "README.md", "dataset_summary.csv", "split_assignments.csv", "analysis_counts.csv", "representations.csv",
        "selected_models.csv", "statistics.csv", "variance_partition.csv.gz", "measurement_quality.csv.gz", "sar_pairs.csv.gz",
        "performance_metrics.csv", "reproducibility_performance.csv", "cell_line_performance.csv", "sequencing_depth.csv",
        "systema_summary.csv", "reference_target_similarity.csv",
    }
    if include_pathway:
        expected |= {"pathway_enrichment.csv.gz", "pathway_gene_scores.csv.gz"}
    missing = expected.difference(path.name for path in output.iterdir())
    if missing:
        raise ValueError(f"Missing outputs: {sorted(missing)}")
    for name in expected - {"README.md"}:
        frame = pd.read_csv(output / name, nrows=5)
        if frame.empty:
            raise ValueError(f"Empty output: {name}")
        if "dataset" in frame and not set(frame.dataset.dropna()).issubset(DATASETS):
            raise ValueError(f"Invalid dataset value in {name}")


def main():
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    include_pathway = not args.skip_pathway
    plans = [REPO / "supplementary_data_file_plan.md", REPO / "old/supplementary_data_file_plan.md"]
    plan = next((path for path in plans if path.is_file()), plans[0])
    required = [REPO / "manuscript.tex", plan,
                source / "final_data/filtered_pseudobulks_alpha_10000.jld2", source / "final_data/filtered_lincs.jld2",
                REPO / "nbs/cache_run_stats.parquet", REPO / "nbs/final_figures/stats/all_statistics.csv",
                REPO / "notebooks/tahoe_pathway_case_annotations.csv"]
    if include_pathway:
        required += [source / "results/tahoe_pathway_cases/gsea/pathway_results.csv"]
    require(required)
    output.mkdir(parents=True, exist_ok=True)
    loaded, _ = load_results(source)
    names = cell_names(source)
    dataset_summary(source, loaded, output)
    selected = selected_models(source, loaded, output)
    representations(source, loaded, output)
    split_assignments(source, loaded, names, output)
    variance_partition(source, output)
    measurement_quality(source, output)
    sar_pairs(source, output)
    performance_metrics(loaded, output)
    reproducibility_performance(loaded, selected, output)
    cell_line_performance(source, loaded, names, output)
    sequencing_depth(source, loaded, names, output)
    systema_summary(source, output)
    reference_similarity(source, names, output)
    pathway = pathway_files(source, output) if include_pathway else None
    statistics_file(pathway, output)
    analysis_counts(loaded, selected, output)
    readme(output, include_pathway)
    validate(output, include_pathway)
    print(f"Generated {18 if include_pathway else 16} files in {output}")


if __name__ == "__main__":
    main()