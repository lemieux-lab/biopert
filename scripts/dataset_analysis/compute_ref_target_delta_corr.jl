using ArgParse, CSV, DataFrames, JLD2, Statistics
using Biopert

# Reference cell lines used in the ref-cell ablation sweeps for each dataset
# (see EXPERIMENT_META in scripts/nbs/results_data.py), included even when not
# in the test split so their pairwise correlations are always available.
const REF_CELLS = Dict(
    "tahoe" => [:CVCL_0023, :CVCL_0131, :CVCL_0218, :CVCL_0480],
    "lincs" => [:A549, :MCF7, :PC3],
)


# Like Metrics.pearson_corr, but returns 0.0 (rather than NaN) for zero-variance
# inputs, so a handful of degenerate conditions don't poison the pairwise mean.
function pearson(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Float64
    (std(x) == 0.0 || std(y) == 0.0) && return 0.0
    return cor(x, y)
end


function compute_all_pairs_corr(
    delta_mean     :: DataFrame,
    cell_lines     :: Vector{Symbol};
    min_conditions :: Int = 2,
) :: DataFrame

    # Build index: cell_line => Dict{(drug, dose, time) => expr vector}
    cl_index = Dict{Symbol, Dict{Tuple{Symbol,Symbol,Symbol}, Vector{Float32}}}()
    for row in eachrow(delta_mean)
        cl  = row.cell_line
        key = (row.drug, row.dose, row.time)
        d   = get!(cl_index, cl, Dict{Tuple{Symbol,Symbol,Symbol}, Vector{Float32}}())
        d[key] = row.expr
    end

    n   = length(cell_lines)
    out = DataFrame(
        cl_i               = Symbol[],
        cl_j               = Symbol[],
        mean_delta_pearson = Float64[],
        n_conditions       = Int[],
    )

    # All ordered pairs (i, j) — includes both directions
    for i in 1:n, j in 1:n
        i == j && continue
        cl_i = cell_lines[i]
        cl_j = cell_lines[j]

        idx_i = get(cl_index, cl_i, nothing)
        idx_j = get(cl_index, cl_j, nothing)
        (isnothing(idx_i) || isnothing(idx_j)) && continue

        shared = intersect(keys(idx_i), keys(idx_j))
        length(shared) < min_conditions && continue

        corrs = [pearson(idx_i[k], idx_j[k]) for k in shared]
        push!(out, (cl_i, cl_j, mean(corrs), length(shared)))
    end

    return out
end


function run(dataset::String, jld2_path::String, test_obs_path::String, output_path::String)
    haskey(REF_CELLS, dataset) || error("Unknown dataset \"$dataset\"; expected one of: $(join(keys(REF_CELLS), ", "))")

    # Load raw data
    @info "Loading df from $jld2_path"
    df = load(jld2_path, "df")
    @info "Loaded df: $(nrow(df)) rows, $(length(unique(df.cell_line))) cell lines"

    # Load test split to get the target cell lines
    @info "Loading test obs from $test_obs_path"
    test_obs  = load(test_obs_path, "obs")
    test_cls  = unique(test_obs.meta_df.cell_line)
    @info "Test split: $(length(test_cls)) unique cell lines"

    # Build deltas — mirrors predict_profiles.jl
    @info "Building delta profiles..."
    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)
    delta    = build_delta_df(untrt_df, trt_df)
    @info "Delta: $(nrow(delta)) rows, $(length(unique(delta.cell_line))) cell lines"

    # Average delta per (cell_line, drug, dose, time) — covers both target mean and average_ref
    @info "Averaging delta per (cell_line, drug, dose, time)..."
    delta_mean = average_expr_df(delta, [:cell_line, :drug, :smiles, :dose, :time])
    @info "delta_mean: $(nrow(delta_mean)) rows"

    # All cell lines present in the averaged delta that also appear in the test split,
    # plus this dataset's reference cell lines (they may not all be in the test split).
    ref_cells = REF_CELLS[dataset]
    delta_cls = unique(delta_mean.cell_line)
    cell_lines = unique(vcat(test_cls, intersect(ref_cells, delta_cls)))
    @info "Computing all-pairs correlations for $(length(cell_lines)) cell lines " *
          "($(length(cell_lines)^2 - length(cell_lines)) ordered pairs)..."

    result = compute_all_pairs_corr(delta_mean, cell_lines)
    @info "Done: $(nrow(result)) pairs computed"

    # Summary
    r = result.mean_delta_pearson
    @info "mean_delta_pearson — mean=$(round(mean(r),digits=3)), " *
          "median=$(round(median(r),digits=3)), std=$(round(std(r),digits=3)), " *
          "min=$(round(minimum(r),digits=3)), max=$(round(maximum(r),digits=3))"

    mkpath(dirname(output_path))
    CSV.write(output_path, result)
    @info "Wrote $(nrow(result)) rows to $output_path"
end


function build_argument_parser()
    s = ArgParseSettings(
        description = "Compute all-pairs cross-CL delta profile correlations for LINCS or Tahoe"
    )
    @add_arg_table s begin
        "dataset"
            arg_type = String
            help     = "\"lincs\" or \"tahoe\""
        "jld2_path"
            arg_type = String
            help     = "Path to the dataset's preprocessed .jld2 (e.g. filtered_lincs.jld2 or pseudobulks_alpha_10000.jld2)"
        "test_obs_path"
            arg_type = String
            help     = "Path to butina test obs JLD2 (to determine target cell lines)"
        "output_path"
            arg_type = String
            help     = "Output CSV path"
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(build_argument_parser())
    run(args["dataset"], args["jld2_path"], args["test_obs_path"], args["output_path"])
end