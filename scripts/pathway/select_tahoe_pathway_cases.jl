"""Rank Tahoe cases for target-context pathway analysis.

The script aligns the pinned A549-reference test observations with the saved
BioPert reproduction, computes reference/target/prediction diagnostics for all
test conditions, applies fixed percentile criteria, and exports the top cases
with their gene-level profiles.

Usage:
    julia --project=julia scripts/pathway/select_tahoe_pathway_cases.jl \
        TEST_OBS TEST_PREDICTIONS CODING_TOKENS OUTPUT_DIR [--n_cases 10]
"""

using ArgParse, CSV, DataFrames, JLD2, LinearAlgebra, Statistics, TOML
using Biopert: cosine_similarity, l2_dist, pearson_corr

const KEYS = [:cell_line, :drug, :dose, :time]

row_key(row) = Tuple(string(row[column]) for column in KEYS)

function safe_pearson(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    (iszero(std(x)) || iszero(std(y))) && return NaN
    return pearson_corr(x, y)
end

function index_rows(meta::DataFrame, label::String)
    index = Dict{Tuple, Int}()
    for (i, row) in enumerate(eachrow(meta))
        key = row_key(row)
        haskey(index, key) && error("Duplicate $label condition: $key")
        index[key] = i
    end
    return index
end

function align_observations(obs, prediction_meta::DataFrame)
    obs.delta_ref_exprs === nothing && error("Test observations do not contain delta_ref_exprs")
    obs_index = index_rows(obs.meta_df, "observation")
    order = Int[]
    for row in eachrow(prediction_meta)
        key = row_key(row)
        haskey(obs_index, key) || error("Prediction condition absent from test observations: $key")
        push!(order, obs_index[key])
    end
    length(order) == length(obs_index) || error(
        "Prediction/test condition counts differ: $(length(order)) vs $(length(obs_index))",
    )
    return order
end

function maximum_column_difference(a::Matrix{Float32}, b::Matrix{Float32}, order::Vector{Int})
    size(a) == size(b) || error("Expression matrices differ in shape: $(size(a)) vs $(size(b))")
    maximum_difference = 0f0
    for (j, obs_j) in enumerate(order)
        maximum_difference = max(
            maximum_difference,
            maximum(abs, @view(a[:, j]) .- @view(b[:, obs_j])),
        )
    end
    return maximum_difference
end

function compute_metrics(
    meta::DataFrame,
    target::Matrix{Float32},
    prediction::Matrix{Float32},
    reference::Matrix{Float32},
)
    out = select(meta, intersect([:cell_line, :drug, :smiles, :dose, :time], Symbol.(names(meta))))
    columns = Dict{Symbol, Vector{Float64}}(
        name => Vector{Float64}(undef, nrow(meta)) for name in [
            :reference_target_pearson,
            :prediction_target_pearson,
            :reference_target_cosine,
            :prediction_target_cosine,
            :adaptation_pearson,
            :adaptation_cosine,
            :reference_target_l2,
            :prediction_target_l2,
            :target_norm,
            :reference_norm,
            :prediction_norm,
            :pearson_gain,
            :fractional_l2_reduction,
        ]
    )

    for i in 1:nrow(meta)
        y = @view target[:, i]
        yhat = @view prediction[:, i]
        ref = @view reference[:, i]
        target_shift = y .- ref
        predicted_shift = yhat .- ref

        ref_r = safe_pearson(ref, y)
        pred_r = safe_pearson(yhat, y)
        ref_l2 = l2_dist(ref, y)
        pred_l2 = l2_dist(yhat, y)

        columns[:reference_target_pearson][i] = ref_r
        columns[:prediction_target_pearson][i] = pred_r
        columns[:reference_target_cosine][i] = cosine_similarity(ref, y)
        columns[:prediction_target_cosine][i] = cosine_similarity(yhat, y)
        columns[:adaptation_pearson][i] = safe_pearson(predicted_shift, target_shift)
        columns[:adaptation_cosine][i] = cosine_similarity(predicted_shift, target_shift)
        columns[:reference_target_l2][i] = ref_l2
        columns[:prediction_target_l2][i] = pred_l2
        columns[:target_norm][i] = norm(y)
        columns[:reference_norm][i] = norm(ref)
        columns[:prediction_norm][i] = norm(yhat)
        columns[:pearson_gain][i] = pred_r - ref_r
        columns[:fractional_l2_reduction][i] = iszero(ref_l2) ? NaN : 1 - pred_l2 / ref_l2
    end

    for (name, values) in columns
        out[!, name] = values
    end
    return out
end

function percentile_scores(values::AbstractVector{<:Real}; higher_is_better::Bool)
    n = length(values)
    n > 1 || error("At least two cases are required for percentile ranking")
    order = sortperm(values)
    scores = Vector{Float64}(undef, n)
    first = 1
    while first <= n
        last = first
        while last < n && values[order[last + 1]] == values[order[first]]
            last += 1
        end
        average_rank = ((first - 1) + (last - 1)) / 2
        score = average_rank / (n - 1)
        for position in first:last
            scores[order[position]] = higher_is_better ? score : 1 - score
        end
        first = last + 1
    end
    return scores
end

function rank_candidates(metrics::DataFrame, n_cases::Int)
    all(isfinite, metrics.reference_target_pearson) || error("Non-finite reference correlations")
    all(isfinite, metrics.prediction_target_pearson) || error("Non-finite prediction correlations")
    all(isfinite, metrics.target_norm) || error("Non-finite target norms")

    thresholds = (
        target_norm_median = median(metrics.target_norm),
        reference_target_pearson_q10 = quantile(metrics.reference_target_pearson, 0.10),
        prediction_target_pearson_q90 = quantile(metrics.prediction_target_pearson, 0.90),
    )
    eligible = filter(row -> row.target_norm >= thresholds.target_norm_median, metrics)
    eligible[!, :divergence_percentile] = percentile_scores(
        eligible.reference_target_pearson; higher_is_better = false,
    )
    eligible[!, :prediction_percentile] = percentile_scores(
        eligible.prediction_target_pearson; higher_is_better = true,
    )
    eligible[!, :gain_percentile] = percentile_scores(
        eligible.pearson_gain; higher_is_better = true,
    )
    eligible[!, :maximin_score] = min.(
        eligible.divergence_percentile,
        eligible.prediction_percentile,
        eligible.gain_percentile,
    )
    sort!(eligible, [
        order(:maximin_score, rev = true),
        order(:pearson_gain, rev = true),
        order(:prediction_target_pearson, rev = true),
        order(:reference_target_pearson),
        order(:target_norm, rev = true),
    ])
    eligible[!, :objective_rank] = 1:nrow(eligible)
    nrow(eligible) >= n_cases || error("Only $(nrow(eligible)) magnitude-eligible cases")
    shortlist = first(eligible, n_cases)
    shortlist[!, :case_id] = ["case_$(lpad(i, 2, '0'))" for i in 1:n_cases]
    return eligible, shortlist, thresholds
end

function export_profiles(
    path::String,
    shortlist::DataFrame,
    metrics::DataFrame,
    target::Matrix{Float32},
    prediction::Matrix{Float32},
    reference::Matrix{Float32},
    coding_tokens_path::String,
)
    tokens = CSV.read(coding_tokens_path, DataFrame)
    nrow(tokens) == size(target, 1) || error(
        "Coding-token count $(nrow(tokens)) does not match $(size(target, 1)) genes",
    )
    names(tokens) == ["coding_tokens"] || error("Unexpected coding-token column: $(names(tokens))")
    profiles = DataFrame(gene_index = 1:size(target, 1), token_id = tokens.coding_tokens)
    metric_index = index_rows(metrics, "metric")

    for row in eachrow(shortlist)
        i = metric_index[row_key(row)]
        prefix = row.case_id
        profiles[!, "$(prefix)_reference"] = reference[:, i]
        profiles[!, "$(prefix)_target"] = target[:, i]
        profiles[!, "$(prefix)_prediction"] = prediction[:, i]
    end
    CSV.write(path, profiles)
end

function run(
    test_obs_path::String,
    predictions_path::String,
    coding_tokens_path::String,
    output_dir::String;
    n_cases::Int,
)
    isfile(test_obs_path) || error("Test observation file not found: $test_obs_path")
    isfile(predictions_path) || error("Prediction file not found: $predictions_path")
    isfile(coding_tokens_path) || error("Coding-token file not found: $coding_tokens_path")

    obs = load(test_obs_path, "obs")
    target = Matrix{Float32}(load(predictions_path, "Y"))
    prediction = Matrix{Float32}(load(predictions_path, "Ŷ"))
    meta = DataFrame(load(predictions_path, "meta"))
    size(target) == size(prediction) || error("Target and prediction matrices differ in shape")
    size(target, 2) == nrow(meta) || error("Prediction metadata is not column-aligned")

    order = align_observations(obs, meta)
    max_y_difference = maximum_column_difference(target, obs.avg_delta_target_exprs, order)
    max_y_difference <= 1f-6 || error(
        "Pinned and reproduced targets differ (maximum absolute difference=$max_y_difference)",
    )
    reference = obs.delta_ref_exprs[:, order]

    metrics = compute_metrics(meta, target, prediction, reference)
    eligible, shortlist, thresholds = rank_candidates(metrics, n_cases)
    n_strict = count(eachrow(metrics)) do row
        row.target_norm >= thresholds.target_norm_median &&
        row.reference_target_pearson <= thresholds.reference_target_pearson_q10 &&
        row.prediction_target_pearson >= thresholds.prediction_target_pearson_q90
    end

    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "condition_metrics.csv"), metrics)
    CSV.write(joinpath(output_dir, "eligible_candidates.csv"), eligible)
    CSV.write(joinpath(output_dir, "shortlist.csv"), shortlist)
    export_profiles(
        joinpath(output_dir, "shortlist_profiles.csv"),
        shortlist,
        metrics,
        target,
        prediction,
        reference,
        coding_tokens_path,
    )

    manifest = Dict(
        "test_obs_path" => abspath(test_obs_path),
        "predictions_path" => abspath(predictions_path),
        "coding_tokens_path" => abspath(coding_tokens_path),
        "n_test_conditions" => nrow(metrics),
        "n_magnitude_eligible_conditions" => nrow(eligible),
        "n_original_strict_conditions" => n_strict,
        "n_shortlisted_conditions" => nrow(shortlist),
        "n_genes" => size(target, 1),
        "reference_cell_line" => "CVCL_0023",
        "expression_space" => "treated-minus-matched-DMSO log-normalized pseudobulk delta",
        "selection" => Dict(
            "target_norm_minimum" => thresholds.target_norm_median,
            "original_reference_target_pearson_q10" => thresholds.reference_target_pearson_q10,
            "original_prediction_target_pearson_q90" => thresholds.prediction_target_pearson_q90,
            "score" => "minimum of divergence, prediction, and gain percentiles among magnitude-eligible cases",
            "percentile_ties" => "average empirical rank",
            "ranking" => [
                "maximin score descending",
                "pearson_gain descending",
                "prediction_target_pearson descending",
                "reference_target_pearson ascending",
                "target_norm descending",
            ],
        ),
        "validation" => Dict("maximum_target_alignment_difference" => max_y_difference),
    )
    open(joinpath(output_dir, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    @info "Wrote $(nrow(eligible)) magnitude-eligible conditions and $n_cases shortlisted cases to $output_dir"
end

function build_parser()
    settings = ArgParseSettings(description = "Select Tahoe pathway case studies")
    @add_arg_table settings begin
        "test_obs_path"
            arg_type = String
        "predictions_path"
            arg_type = String
        "coding_tokens_path"
            arg_type = String
        "output_dir"
            arg_type = String
        "--n_cases"
            arg_type = Int
            default = 10
    end
    return settings
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(build_parser())
    run(
        args["test_obs_path"],
        args["predictions_path"],
        args["coding_tokens_path"],
        args["output_dir"];
        n_cases = args["n_cases"],
    )
end
