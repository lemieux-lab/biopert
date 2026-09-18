#!/usr/bin/env python3
"""Run preranked GSEA for shortlisted Tahoe reference/target/prediction profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gseapy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROFILE_TYPES = ("reference", "target", "prediction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("selection_dir", type=Path)
    parser.add_argument("gene_metadata", type=Path)
    parser.add_argument("cell_line_metadata", type=Path)
    parser.add_argument("hallmark_gmt", type=Path)
    parser.add_argument("reactome_gmt", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size", type=int, default=500)
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")


def annotate_shortlist(shortlist: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_parquet(metadata_path)
    required = {
        "Cell_ID_Cellosaur",
        "cell_name",
        "Organ",
        "Driver_Gene_Symbol",
        "Driver_ProtEffect_or_CdnaEffect",
        "Driver_Mech_InferDM",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Cell-line metadata is missing: {sorted(missing)}")

    def unique_join(values: pd.Series) -> str:
        clean = sorted({str(value) for value in values.dropna() if str(value).strip()})
        return ";".join(clean)

    cell_lines = (
        metadata.groupby("Cell_ID_Cellosaur", as_index=False)
        .agg(
            cell_name=("cell_name", unique_join),
            organ=("Organ", unique_join),
            driver_genes=("Driver_Gene_Symbol", unique_join),
            driver_effects=("Driver_ProtEffect_or_CdnaEffect", unique_join),
            driver_mechanisms=("Driver_Mech_InferDM", unique_join),
        )
        .rename(columns={"Cell_ID_Cellosaur": "cell_line"})
    )
    annotated = shortlist.merge(cell_lines, on="cell_line", how="left", validate="many_to_one")
    if annotated["cell_name"].isna().any():
        missing_ids = annotated.loc[annotated["cell_name"].isna(), "cell_line"].tolist()
        raise ValueError(f"Missing cell-line annotations: {missing_ids}")
    return annotated


def map_genes(profiles: pd.DataFrame, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_parquet(metadata_path)[["token_id", "gene_symbol", "ensembl_id"]]
    metadata = metadata.drop_duplicates("token_id")
    mapped = profiles.merge(metadata, on="token_id", how="left", validate="one_to_one")
    missing = int(mapped["gene_symbol"].isna().sum())
    mapped["gene_symbol"] = mapped["gene_symbol"].astype("string").str.strip().str.upper()
    mapped = mapped.loc[mapped["gene_symbol"].notna() & mapped["gene_symbol"].ne("")].copy()
    report = {
        "n_input_tokens": int(len(profiles)),
        "n_mapped_tokens": int(len(mapped)),
        "n_unmapped_tokens": missing,
        "n_unique_symbols": int(mapped["gene_symbol"].nunique()),
        "n_duplicate_symbol_rows": int(mapped["gene_symbol"].duplicated(keep=False).sum()),
    }
    return mapped, report


def collapse_ranking(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    ranking = frame[["gene_symbol", value_column]].dropna().copy()
    ranking.columns = ["gene", "score"]
    ranking["abs_score"] = ranking["score"].abs()
    ranking = ranking.sort_values(["gene", "abs_score", "score"], ascending=[True, False, False])
    ranking = ranking.drop_duplicates("gene", keep="first").drop(columns="abs_score")
    # Break exact expression ties deterministically without changing meaningful ranks.
    scale = max(float(ranking["score"].abs().max()), 1.0)
    ranking["score"] += np.linspace(1.0, -1.0, len(ranking)) * scale * 1e-12
    ranking = ranking.sort_values(["score", "gene"], ascending=[False, True]).reset_index(drop=True)
    if not np.isfinite(ranking["score"]).all():
        raise ValueError(f"Non-finite values in {value_column}")
    if ranking["gene"].duplicated().any():
        raise ValueError(f"Duplicate genes remain in {value_column}")
    return ranking


def run_prerank(
    ranking: pd.DataFrame,
    gene_sets: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    result = gseapy.prerank(
        rnk=ranking,
        gene_sets=str(gene_sets),
        min_size=args.min_size,
        max_size=args.max_size,
        permutation_num=args.permutations,
        weight=1.0,
        ascending=False,
        threads=args.threads,
        outdir=None,
        seed=args.seed,
        no_plot=True,
        verbose=False,
    ).res2d
    return result.rename(
        columns={
            "Term": "pathway",
            "ES": "es",
            "NES": "nes",
            "NOM p-val": "nominal_p",
            "FDR q-val": "fdr_q",
            "FWER p-val": "fwer_p",
            "Tag %": "tag_fraction",
            "Gene %": "gene_fraction",
            "Lead_genes": "leading_edge_genes",
        }
    )


def pathway_comparison(case_results: pd.DataFrame, collection: str) -> tuple[dict, pd.DataFrame]:
    values = case_results.pivot(
        index="pathway",
        columns="profile",
        values=["nes", "fdr_q", "leading_edge_genes"],
    )
    values.columns = [f"{metric}_{profile}" for metric, profile in values.columns]
    values = values.dropna(subset=["nes_reference", "nes_target", "nes_prediction"]).reset_index()

    reference_error = (values["nes_reference"] - values["nes_target"]).abs()
    prediction_error = (values["nes_prediction"] - values["nes_target"]).abs()
    target_significant = values["fdr_q_target"] < 0.05
    reference_sign = np.sign(values["nes_reference"]) == np.sign(values["nes_target"])
    prediction_sign = np.sign(values["nes_prediction"]) == np.sign(values["nes_target"])
    target_specific = target_significant & (
        (~reference_sign) | (reference_error >= 1.0)
    )

    summary = {
        "collection": collection,
        "n_pathways": int(len(values)),
        "target_reference_nes_spearman": float(spearmanr(values["nes_target"], values["nes_reference"]).statistic),
        "target_prediction_nes_spearman": float(spearmanr(values["nes_target"], values["nes_prediction"]).statistic),
        "target_reference_nes_mae": float(reference_error.mean()),
        "target_prediction_nes_mae": float(prediction_error.mean()),
        "n_target_significant": int(target_significant.sum()),
        "reference_sign_agreement_target_significant": float(reference_sign[target_significant].mean()) if target_significant.any() else np.nan,
        "prediction_sign_agreement_target_significant": float(prediction_sign[target_significant].mean()) if target_significant.any() else np.nan,
        "n_target_specific": int(target_specific.sum()),
        "n_target_specific_closer_prediction": int((target_specific & (prediction_error < reference_error)).sum()),
    }

    comparison = values.loc[target_specific].copy()
    comparison["collection"] = collection
    comparison["reference_nes_error"] = reference_error[target_specific].to_numpy()
    comparison["prediction_nes_error"] = prediction_error[target_specific].to_numpy()
    comparison["nes_error_reduction"] = (
        comparison["reference_nes_error"] - comparison["prediction_nes_error"]
    )
    comparison["prediction_closer"] = comparison["nes_error_reduction"] > 0
    comparison = comparison.sort_values(
        ["prediction_closer", "nes_error_reduction", "fdr_q_target"],
        ascending=[False, False, True],
    )
    return summary, comparison


def main() -> None:
    args = parse_args()
    selection_files = [
        args.selection_dir / "shortlist.csv",
        args.selection_dir / "shortlist_profiles.csv",
        args.selection_dir / "manifest.toml",
    ]
    require_files(
        selection_files
        + [args.gene_metadata, args.cell_line_metadata, args.hallmark_gmt, args.reactome_gmt]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shortlist = pd.read_csv(selection_files[0])
    profiles = pd.read_csv(selection_files[1])
    expected_cases = {f"case_{i:02d}" for i in range(1, 11)}
    if set(shortlist["case_id"]) != expected_cases:
        raise ValueError(f"Expected cases {sorted(expected_cases)}, found {sorted(shortlist['case_id'])}")
    shortlist = annotate_shortlist(shortlist, args.cell_line_metadata)
    shortlist.to_csv(args.output_dir / "shortlist_annotated.csv", index=False)

    profiles, mapping_report = map_genes(profiles, args.gene_metadata)
    libraries = {"hallmark": args.hallmark_gmt, "reactome": args.reactome_gmt}
    result_frames = []
    summary_rows = []
    comparison_frames = []

    for case_id in shortlist.sort_values("objective_rank")["case_id"]:
        case_frames = []
        for profile in PROFILE_TYPES:
            column = f"{case_id}_{profile}"
            ranking = collapse_ranking(profiles, column)
            ranking.to_csv(args.output_dir / f"{case_id}_{profile}_ranking.csv", index=False)
            for collection, gene_sets in libraries.items():
                result = run_prerank(ranking, gene_sets, args)
                result.insert(0, "collection", collection)
                result.insert(0, "profile", profile)
                result.insert(0, "case_id", case_id)
                result_frames.append(result)
                case_frames.append(result)

        case_results = pd.concat(case_frames, ignore_index=True)
        for collection in libraries:
            collection_results = case_results.loc[case_results["collection"].eq(collection)]
            summary, comparison = pathway_comparison(collection_results, collection)
            summary["case_id"] = case_id
            summary_rows.append(summary)
            comparison.insert(0, "case_id", case_id)
            comparison_frames.append(comparison)

    pathway_results = pd.concat(result_frames, ignore_index=True)
    pathway_results.to_csv(args.output_dir / "pathway_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "case_pathway_summary.csv", index=False)
    pd.concat(comparison_frames, ignore_index=True).to_csv(
        args.output_dir / "target_specific_pathways.csv", index=False
    )

    manifest = {
        "gseapy_version": gseapy.__version__,
        "expression_space": "treated-minus-matched-DMSO log-normalized pseudobulk delta",
        "profile_types": list(PROFILE_TYPES),
        "gene_mapping": mapping_report,
        "gsea": {
            "permutations": args.permutations,
            "weight": 1.0,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "threads": args.threads,
            "libraries": {name: str(path.resolve()) for name, path in libraries.items()},
        },
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
