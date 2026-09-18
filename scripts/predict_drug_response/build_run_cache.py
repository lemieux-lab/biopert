"""
Build the splitstats per-run cache for a single completed run directory.
Called at the end of training by predict_profiles.jl.

Usage: python build_run_cache.py <run_dir> [<biopert_outdir>]

<biopert_outdir> defaults to the parent directory of <run_dir>; the
predict_target_delta_profiles.jl caller always passes it explicitly instead
(run_dir now lives under <outdir>/<prediction_dir>/..., not directly under
<outdir>), so the default only matters for manual invocation with a <run_dir>
that isn't already inside the right BIOPERT_OUTDIR.
"""
import os
import sys

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nbs"))

import results_data
from results_data import (
    EXPERIMENT_META,
    _PER_RUN_CACHE_FILE,
    _build_runs_df,
    _compute_run_stats_single,
    _compute_test_key_intersection,
    _parse_run_dir,
)


def build_cache(run_dir: str, biopert_outdir: str | None = None) -> None:
    run_path = os.path.abspath(run_dir)
    results_data.configure(biopert_outdir or os.path.dirname(run_path))
    run_name = os.path.basename(run_path)
    cache_path = os.path.join(run_path, _PER_RUN_CACHE_FILE)

    if os.path.exists(cache_path):
        print(f"Cache already exists, skipping: {cache_path}")
        return

    exp_key, _ = _parse_run_dir(run_name)
    if exp_key is None or exp_key not in EXPERIMENT_META:
        print(f"Warning: unrecognised run dir, skipping cache: {run_name}")
        return

    dataset, category, label = EXPERIMENT_META[exp_key]
    intersections = _compute_test_key_intersection(_build_runs_df())
    test_keys = intersections.get(dataset, set())
    if category not in {"ref_cell", "holdout"} and not test_keys:
        raise RuntimeError(f"No shared test observations found for {dataset}")

    result = _compute_run_stats_single(
        run_path,
        run_name,
        exp_key,
        dataset,
        category,
        label,
        test_keys,
    )
    if result is not None:
        print(f"Cache written: {cache_path} ({len(result)} rows)")
    else:
        print(f"Warning: no data found for {run_name}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <run_dir> [<biopert_outdir>]")
        sys.exit(1)
    build_cache(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)