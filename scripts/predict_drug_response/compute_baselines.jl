"""
Compute naive baselines on the exact train/val/test splits used by the sweeps.

Three baselines per dataset:
  - delta_ref  : predict the reference cell line's (averaged) delta for every sample
  - mean_delta : predict the mean delta_target across the training set
  - zero       : predict zero (no drug effect)

Splits are loaded from pinned butina obs files — the same files all sweep configs
point to — so the train/val/test membership is identical to what the sweeps see.
Observations whose (drug, dose, time) has no matching reference-cell-line profile
are dropped by build_obs, exactly as they are for the sweeps (which also call
build_obs with use_delta_ref=true), so the delta_ref/mean_delta/zero baselines all
share that same membership.

Outputs (one sub-directory per dataset × baseline):
  <output_dir>/{tahoe,lincs}_{delta_ref,mean_delta,zero}/{train,val,test}_perfs_per_obs.csv

Passing --{tahoe,lincs}_cellline_{val,test}_obs additionally holds out whole cell
lines, matching a config with holdout_cell_lines = true. Baselines must be recomputed
in that case: mean_delta is defined over the training set, which changes.

Usage:
  julia --project=julia scripts/predict_drug_response/compute_baselines.jl <output_dir> \\
    --tahoe_jld2   <path>  --tahoe_val_obs  <path>  --tahoe_test_obs  <path> \\
    --lincs_jld2   <path>  --lincs_val_obs  <path>  --lincs_test_obs  <path> \\
    [--tahoe_cellline_val_obs <path>] [--tahoe_cellline_test_obs <path>] \\
    [--lincs_cellline_val_obs <path>] [--lincs_cellline_test_obs <path>] \\
    [--ref_cl_tahoe CVCL_0023] [--ref_cl_lincs A549]
"""

using ArgParse, JLD2, CSV, DataFrames, Statistics
using Biopert


# Split helpers (load_split_smiles, load_split_cell_lines, apply_pinned_split)
# and build_obs come from Biopert, so this script and predict_target_delta_profiles.jl
# cannot disagree about what "train" means or how delta_ref is computed.


# ── Baseline predictions ──────────────────────────────────────────────────────

function predict_mean_delta(obs::Obs, mean_vec::Vector{Float32})::Matrix{Float32}
    repeat(mean_vec, 1, nrow(obs.meta_df))
end

function predict_zero(obs::Obs)::Matrix{Float32}
    zeros(Float32, size(obs.avg_delta_target_exprs, 1), nrow(obs.meta_df))
end


# ── Per-dataset runner ────────────────────────────────────────────────────────

function run_dataset(
    label       :: String,
    jld2_path   :: String,
    val_obs_path:: String,
    test_obs_path::String,
    ref_cl      :: Symbol,
    output_dir  :: String;
    cellline_val_obs_path  :: Union{Nothing, String} = nothing,
    cellline_test_obs_path :: Union{Nothing, String} = nothing,
)
    @info "=== $label ==="

    df       = load(jld2_path, "df")
    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)

    # average_delta_ref=true: the delta_ref baseline predicts the *mean* reference-
    # cell-line delta for a (drug, dose, time), not a single resampled replicate —
    # a deterministic baseline needs the average, not a random draw.
    obs = build_obs(
        ref_cl, untrt_df, trt_df;
        use_delta_ref     = true,
        average_delta_ref = true,
        return_ref_pool   = false,
    )

    val_cl  = isnothing(cellline_val_obs_path)  ? Set{Symbol}() : load_split_cell_lines(cellline_val_obs_path)
    test_cl = isnothing(cellline_test_obs_path) ? Set{Symbol}() : load_split_cell_lines(cellline_test_obs_path)

    train_obs, val_obs, test_obs = apply_pinned_split(
        obs;
        val_smiles  = load_split_smiles(val_obs_path),
        test_smiles = load_split_smiles(test_obs_path),
        val_cl      = val_cl,
        test_cl     = test_cl,
    )

    mean_vec = vec(mean(train_obs.avg_delta_target_exprs; dims=2))

    for (name, Ŷ_fn) in [
        ("delta_ref",  obs -> obs.delta_ref_exprs),
        ("mean_delta", obs -> predict_mean_delta(obs, mean_vec)),
        ("zero",       obs -> predict_zero(obs)),
    ]
        out = joinpath(output_dir, "$(label)_$(name)")
        mkpath(out)
        for (split, split_obs) in [("train", train_obs), ("val", val_obs), ("test", test_obs)]
            Ŷ = Ŷ_fn(split_obs)
            metrics_per_obs(
                split_obs.meta_df, split_obs.avg_delta_target_exprs, Ŷ,
                joinpath(out, "$(split)_perfs_per_obs.csv"),
            )
        end
        @info "[$label/$name] saved to $out"
    end
end


# ── Argument parsing + entry point ────────────────────────────────────────────

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "output_dir"
            help     = "Directory where baseline CSVs are written"
            arg_type = String
        "--tahoe_jld2"
            help     = "Path to Tahoe JLD2 file"
            arg_type = String
            required = true
        "--tahoe_val_obs"
            help     = "Pinned Tahoe val obs file"
            arg_type = String
            required = true
        "--tahoe_test_obs"
            help     = "Pinned Tahoe test obs file"
            arg_type = String
            required = true
        "--lincs_jld2"
            help     = "Path to LINCS JLD2 file"
            arg_type = String
            required = true
        "--lincs_val_obs"
            help     = "Pinned LINCS val obs file"
            arg_type = String
            required = true
        "--lincs_test_obs"
            help     = "Pinned LINCS test obs file"
            arg_type = String
            required = true
        "--ref_cl_tahoe"
            help     = "Reference cell line for Tahoe"
            arg_type = String
            default  = "CVCL_0023"
        "--ref_cl_lincs"
            help     = "Reference cell line for LINCS"
            arg_type = String
            default  = "A549"
        "--tahoe_cellline_val_obs"
            help     = "Pinned Tahoe val cell-line file (enables the cell-line holdout axis)"
            arg_type = String
            default  = nothing
        "--tahoe_cellline_test_obs"
            help     = "Pinned Tahoe test cell-line file"
            arg_type = String
            default  = nothing
        "--lincs_cellline_val_obs"
            help     = "Pinned LINCS val cell-line file (enables the cell-line holdout axis)"
            arg_type = String
            default  = nothing
        "--lincs_cellline_test_obs"
            help     = "Pinned LINCS test cell-line file"
            arg_type = String
            default  = nothing
    end
    return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_cli()

    run_dataset(
        "tahoe",
        args["tahoe_jld2"],
        args["tahoe_val_obs"],
        args["tahoe_test_obs"],
        Symbol(args["ref_cl_tahoe"]),
        args["output_dir"];
        cellline_val_obs_path  = args["tahoe_cellline_val_obs"],
        cellline_test_obs_path = args["tahoe_cellline_test_obs"],
    )

    run_dataset(
        "lincs",
        args["lincs_jld2"],
        args["lincs_val_obs"],
        args["lincs_test_obs"],
        Symbol(args["ref_cl_lincs"]),
        args["output_dir"];
        cellline_val_obs_path  = args["lincs_cellline_val_obs"],
        cellline_test_obs_path = args["lincs_cellline_test_obs"],
    )
end
