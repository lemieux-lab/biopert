import argparse
import io
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py_common"))
from dataset_paths import dataset_paths  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_REF_CL = {"lincs": "A549", "tahoe": "CVCL_0023"}


def load_pseudobulk_df(jld2_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise RuntimeError("Could not find a `julia` executable on PATH.")
    julia_project = str(Path(__file__).resolve().parents[2] / "julia")
    os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", julia_project)
    from juliacall import Main as jl

    jl.seval("using JLD2, DataFrames, CSV")
    load_fn = jl.seval(
        """
        function _pseudobulk_df_to_python(path)
            df = JLD2.load(path, "df")
            expr = permutedims(reduce(hcat, df.expr))
            meta_df = select(df, Not(:expr))
            # Stringify every column (some, e.g. combination-drug doses, aren't
            # plain AbstractString and would otherwise print unquoted commas,
            # like "[10.0, 5.0]", that break CSV tokenization on read-back).
            for c in names(meta_df)
                meta_df[!, c] = string.(meta_df[!, c])
            end
            buf = IOBuffer()
            CSV.write(buf, meta_df; quotestrings=true)
            return String(take!(buf)), expr
        end
        """
    )
    meta_csv, expr_jl = load_fn(str(jld2_path))
    meta_df = pd.read_csv(io.StringIO(str(meta_csv)))
    expr = np.asarray(expr_jl, dtype=np.float32)
    logging.info("  meta_df: %s rows  |  expr: %s genes × %s samples", len(meta_df), expr.shape[1], expr.shape[0])
    return meta_df, expr


def compute_delta_ref(
    meta_all: pd.DataFrame,
    expr_all: np.ndarray,
    ref_cl: str,
    average: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    logging.info("Computing per-plate mean DMSO profiles …")
    dmso_mask = meta_all["drug"] == "DMSO"
    dmso_meta = meta_all[dmso_mask]
    plate_dmso: dict[str, np.ndarray] = {}
    for plate, grp in dmso_meta.groupby("plate"):
        plate_dmso[plate] = expr_all[grp.index.values].mean(axis=0)
    logging.info("  %s plates with DMSO controls", len(plate_dmso))

    # Filter to treated samples on the ref cell line
    trt_ref_mask = (~dmso_mask) & (meta_all["cell_line"] == ref_cl) & meta_all["plate"].isin(plate_dmso)
    trt_ref = meta_all[trt_ref_mask].copy()
    logging.info("  %s treated samples for ref cell line %s", len(trt_ref), ref_cl)

    # Compute delta profiles
    logging.info("Computing delta profiles (treated − plate mean DMSO) …")
    idxs = trt_ref.index.values
    delta_expr = expr_all[idxs] - np.stack([plate_dmso[p] for p in trt_ref["plate"]])
    logging.info("  delta_expr: %s samples × %s genes", delta_expr.shape[0], delta_expr.shape[1])

    meta_df = trt_ref.reset_index(drop=True)
    return average_delta_ref(meta_df, delta_expr, ref_cl, average)


def average_delta_ref(
    meta_df: pd.DataFrame,
    delta_expr: np.ndarray,
    ref_cl: str,
    average: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_genes = delta_expr.shape[1]
    if average:
        logging.info("Averaging replicates per (drug, dose, time) …")
        group_cols = ["drug", "smiles", "dose", "time"]
        groups = meta_df.groupby(group_cols, sort=False)
        avg_delta = np.empty((len(groups), n_genes), dtype=np.float32)
        avg_meta_rows = []
        for j, (key, grp) in enumerate(groups):
            avg_delta[j] = delta_expr[grp.index.values].mean(axis=0)
            avg_meta_rows.append(dict(zip(group_cols, key), cell_line=ref_cl))
        delta_expr = avg_delta
        meta_df = pd.DataFrame(avg_meta_rows).reset_index(drop=True)

    logging.info("  %s delta profiles  (%s genes)", len(meta_df), n_genes)
    return meta_df, delta_expr.astype(np.float32)


def build_delta_ref(dataset: str, jld2_path: Path, ref_cl: str, average: bool) -> tuple[pd.DataFrame, np.ndarray]:
    logging.info("Loading %s (via embedded Julia) …", jld2_path)
    meta_all, expr_all = load_pseudobulk_df(jld2_path)

    if dataset == "tahoe":
        # Restrict Tahoe genes to the landmark genes shared with LINCS
        logging.info("Loading landmark gene mask …")
        df_tokens = pd.read_csv(jld2_path.parent / "tahoe_coding_tokens.csv")
        df_shared = pd.read_csv("data/lincs_and_tahoe_shared_genes.csv")
        shared_tokens = set(df_shared["token_id"])
        landmark_mask = df_tokens["coding_tokens"].isin(shared_tokens).to_numpy()
        logging.info("  Landmark genes: %s / %s", int(landmark_mask.sum()), len(landmark_mask))
        expr_all = expr_all[:, landmark_mask]

    return compute_delta_ref(meta_all, expr_all, ref_cl, average)


def main():
    parser = argparse.ArgumentParser(description="UMAP of LINCS or Tahoe delta-ref profiles for the reference cell line.")
    parser.add_argument("dataset", choices=["lincs", "tahoe"], help="Which dataset to use.")
    parser.add_argument("outdir", type=Path, help="BIOPERT_OUTDIR: base directory for all pipeline data.")
    parser.add_argument("--ref_cl", default=None, help="Reference cell line (default: A549 for lincs, CVCL_0023 for tahoe).")
    parser.add_argument("--average", action="store_true", help="Average replicates per (drug, dose, time) before UMAP.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15).")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1).")
    parser.add_argument("--metric", default="cosine", help="UMAP metric (default: cosine).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    ref_cl = args.ref_cl or DEFAULT_REF_CL[args.dataset]
    jld2_path = dataset_paths(args.outdir, args.dataset)["jld2_path"]
    output_dir = Path(args.outdir) / "delta_ref_umap" / args.dataset
    os.makedirs(output_dir, exist_ok=True)

    meta_df, delta_expr = build_delta_ref(args.dataset, jld2_path, ref_cl, args.average)

    # Fit UMAP
    logging.info(
        "Fitting UMAP (n=%s, metric=%s, n_neighbors=%s, min_dist=%s) …",
        len(meta_df),
        args.metric,
        args.n_neighbors,
        args.min_dist,
    )
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        verbose=True,
    )
    embedding = reducer.fit_transform(delta_expr)

    avg_suffix = "" if args.average else "_unaveraged"
    umap_dir = output_dir / f"{ref_cl}_{args.n_neighbors}_{args.min_dist}{avg_suffix}"
    os.makedirs(umap_dir, exist_ok=True)

    model_path = umap_dir / "umap_model.pkl"
    emb_path = umap_dir / "umap_embedding.npy"
    with open(model_path, "wb") as fh:
        pickle.dump(reducer, fh)
    np.save(str(emb_path), embedding)
    logging.info("Saved UMAP model     → %s", model_path)
    logging.info("Saved UMAP embedding → %s", emb_path)

    # Single-color scatter
    logging.info("Generating scatter plot …")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.4, linewidths=0, color="steelblue", rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    avg_label = " (averaged)" if args.average else ""
    ax.set_title(
        f"{args.dataset.upper()} delta-ref profiles — {ref_cl}{avg_label}\n"
        f"n={len(meta_df):,}  |  metric={args.metric}  |  "
        f"n_neighbors={args.n_neighbors}  min_dist={args.min_dist}"
    )
    fig.tight_layout()
    plot_path = umap_dir / "umap_uncolored.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info("Saved plot → %s", plot_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
