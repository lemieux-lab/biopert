"""
Evaluate saved BioPert predictions with Systema.

The input directory must contain `train_predictions.jld2` and
`test_predictions.jld2`, each with gene-by-condition matrices `Y` and `Ŷ` plus
aligned `meta`. The analysis uses log-normalized pseudobulk deltas directly.

The training response mean is the perturbed-centroid reference. It is estimated
separately within each context, then subtracted from both observations and
predictions. Centroid accuracy compares test conditions only within context.

Usage:
    julia --project=julia scripts/predict_drug_response/eval_systema.jl RUN_DIR OUTPUT_DIR \
        [--context_cols cell_line] [--condition_cols drug,dose,time] [--top_k 20]
"""

using ArgParse, CSV, DataFrames, JLD2, LinearAlgebra, Statistics, TOML
using Biopert


parse_columns(value::String) = Symbol.(filter(!isempty, strip.(split(value, ","))))

row_key(meta::DataFrame, i::Int, columns::Vector{Symbol}) =
    Tuple(meta[i, column] for column in columns)


function load_bundle(path::String; load_predictions::Bool = true)
    isfile(path) || error("Prediction file not found: $path")
    required = load_predictions ? ["Y", "Ŷ", "meta"] : ["Y", "meta"]
    available = jldopen(path, "r") do file
        Set(string.(keys(file)))
    end
    missing_keys = filter(key -> key ∉ available, required)
    isempty(missing_keys) || error("$path is missing keys: $(join(missing_keys, ", "))")

    Y = load(path, "Y")
    Yhat = load_predictions ? load(path, "Ŷ") : nothing
    meta = load(path, "meta")
    !load_predictions || size(Y) == size(Yhat) || error("Y and Ŷ differ in shape in $path")
    size(Y, 2) == nrow(meta) || error("Expression columns and metadata rows differ in $path")
    all(isfinite, Y) || error("Y contains non-finite values in $path")
    !load_predictions || all(isfinite, Yhat) || error("Ŷ contains non-finite values in $path")
    predictions = load_predictions ? Matrix{Float32}(Yhat) : nothing
    return Matrix{Float32}(Y), predictions, DataFrame(meta)
end


function validate_columns(meta::DataFrame, columns::Vector{Symbol}, label::String)
    isempty(columns) && error("$label must contain at least one column")
    available = Set(Symbol.(names(meta)))
    missing_columns = filter(column -> column ∉ available, columns)
    isempty(missing_columns) || error("Missing $label columns: $(join(missing_columns, ", "))")
end


function group_indices(meta::DataFrame, columns::Vector{Symbol})
    groups = Dict{Tuple, Vector{Int}}()
    for i in 1:nrow(meta)
        push!(get!(groups, row_key(meta, i, columns), Int[]), i)
    end
    return groups
end


function context_means(meta::DataFrame, Y::Matrix{Float32}, context_cols::Vector{Symbol})
    return Dict(
        key => vec(mean(@view(Y[:, indices]); dims = 2))
        for (key, indices) in group_indices(meta, context_cols)
    )
end


function reference_matrix(
    meta::DataFrame,
    means::Dict,
    context_cols::Vector{Symbol},
    n_genes::Int,
)
    reference = Matrix{Float32}(undef, n_genes, nrow(meta))
    for i in 1:nrow(meta)
        key = row_key(meta, i, context_cols)
        haskey(means, key) || error("No training perturbed centroid for context $key")
        reference[:, i] = means[key]
    end
    return reference
end


function safe_pearson(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    (iszero(std(x)) || iszero(std(y))) && return NaN
    return pearson_corr(x, y)
end


function safe_cosine(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    (iszero(norm(x)) || iszero(norm(y))) && return NaN
    return cosine_similarity(x, y)
end


function profile_metrics(
    y::AbstractVector{T},
    yhat::AbstractVector{T},
    top::AbstractVector{Int},
) where {T <: AbstractFloat}
    return (
        pearson = safe_pearson(y, yhat),
        pearson_topk = safe_pearson(y[top], yhat[top]),
        cosine = safe_cosine(y, yhat),
        rmse = sqrt(mse(y, yhat)),
    )
end


function assert_unique_conditions(
    meta::DataFrame,
    context_cols::Vector{Symbol},
    condition_cols::Vector{Symbol},
)
    columns = unique(vcat(context_cols, condition_cols))
    groups = group_indices(meta, columns)
    duplicates = filter(pair -> length(last(pair)) > 1, collect(groups))
    isempty(duplicates) || error(
        "Centroid accuracy requires one row per context-condition; found " *
        "$(length(duplicates)) duplicated key(s)",
    )
end


function centroid_accuracy(
    meta::DataFrame,
    Y::Matrix{Float32},
    Yhat::Matrix{Float32},
    context_cols::Vector{Symbol},
)
    scores = fill(NaN, nrow(meta))
    for indices in values(group_indices(meta, context_cols))
        length(indices) < 2 && continue
        truth = Y[:, indices]
        pred = Yhat[:, indices]
        distances = max.(
            vec(sum(abs2, pred; dims = 1)) .+
            vec(sum(abs2, truth; dims = 1))' .-
            2f0 .* (pred' * truth),
            0f0,
        )
        for (local_i, global_i) in enumerate(indices)
            self_distance = distances[local_i, local_i]
            scores[global_i] = count(
                j -> j != local_i && distances[local_i, j] > self_distance,
                eachindex(indices),
            ) / (length(indices) - 1)
        end
    end
    return scores
end


function output_metadata(meta::DataFrame, columns::Vector{Symbol})
    out = select(meta, columns)
    for column in names(out)
        out[!, column] = string.(out[!, column])
    end
    return out
end


function systematic_variation(
    split::String,
    meta::DataFrame,
    Y::Matrix{Float32},
    means::Dict,
    context_cols::Vector{Symbol},
    id_cols::Vector{Symbol},
)
    reference = reference_matrix(meta, means, context_cols, size(Y, 1))
    out = output_metadata(meta, id_cols)
    out[!, :split] = fill(split, nrow(meta))
    out[!, :cosine_to_perturbed_centroid] = [
        safe_cosine(@view(Y[:, i]), @view(reference[:, i])) for i in 1:nrow(meta)
    ]
    out[!, :response_norm] = [norm(@view(Y[:, i])) for i in 1:nrow(meta)]
    out[!, :specific_norm] = [norm(@view(Y[:, i]) .- @view(reference[:, i])) for i in 1:nrow(meta)]
    return out
end


function condition_metrics(
    meta::DataFrame,
    Y::Matrix{Float32},
    Yhat::Matrix{Float32},
    reference::Matrix{Float32},
    id_cols::Vector{Symbol},
    context_cols::Vector{Symbol},
    top_k::Int,
)
    frames = DataFrame[]
    k = min(top_k, size(Y, 1))
    top_indices = [partialsortperm(abs.(@view(Y[:, i])), 1:k; rev = true) for i in 1:nrow(meta)]
    for (method, predictions) in [
        ("biopert", Yhat),
        ("perturbed_mean", reference),
    ]
        out = output_metadata(meta, id_cols)
        out[!, :method] = fill(method, nrow(meta))
        control = NamedTuple[]
        systema = NamedTuple[]
        for i in 1:nrow(meta)
            top = top_indices[i]
            push!(control, profile_metrics(@view(Y[:, i]), @view(predictions[:, i]), top))
            push!(systema, profile_metrics(
                @view(Y[:, i]) .- @view(reference[:, i]),
                @view(predictions[:, i]) .- @view(reference[:, i]),
                top,
            ))
        end
        for name in propertynames(first(control))
            out[!, Symbol("$(name)_control")] = getproperty.(control, name)
            out[!, Symbol("$(name)_systema")] = getproperty.(systema, name)
        end
        out[!, :centroid_accuracy] = centroid_accuracy(meta, Y, predictions, context_cols)
        push!(frames, out)
    end
    return vcat(frames...)
end


function summarize_metrics(per_condition::DataFrame)
    metric_cols = filter(
        column -> column ∉ [:method] && eltype(per_condition[!, column]) <: AbstractFloat,
        Symbol.(names(per_condition)),
    )
    rows = NamedTuple[]
    for group in groupby(per_condition, :method), metric in metric_cols
        values = Float64.(group[!, metric])
        finite = filter(isfinite, values)
        push!(rows, (
            method = first(group.method),
            metric = string(metric),
            n = length(values),
            n_finite = length(finite),
            mean = isempty(finite) ? NaN : mean(finite),
            median = isempty(finite) ? NaN : median(finite),
            q25 = isempty(finite) ? NaN : quantile(finite, 0.25),
            q75 = isempty(finite) ? NaN : quantile(finite, 0.75),
        ))
    end
    return DataFrame(rows)
end


function summarize_systematic_variation(systematic::DataFrame)
    metrics = [:cosine_to_perturbed_centroid, :response_norm, :specific_norm]
    rows = NamedTuple[]
    for group in groupby(systematic, :split), metric in metrics
        raw = Float64.(group[!, metric])
        values = filter(isfinite, raw)
        isempty(values) && error("No finite values for $metric in split $(first(group.split))")
        push!(rows, (
            split = first(group.split),
            metric = string(metric),
            n = length(raw),
            n_finite = length(values),
            mean = mean(values),
            median = median(values),
            q25 = quantile(values, 0.25),
            q75 = quantile(values, 0.75),
        ))
    end
    return DataFrame(rows)
end


function run(
    run_dir::String,
    output_dir::String;
    context_cols::Vector{Symbol},
    condition_cols::Vector{Symbol},
    top_k::Int,
)
    top_k > 1 || error("top_k must be greater than one")
    train_path = joinpath(run_dir, "train_predictions.jld2")
    test_path = joinpath(run_dir, "test_predictions.jld2")
    Y_train, _, train_meta = load_bundle(train_path; load_predictions = false)
    Y_test, Yhat_test, test_meta = load_bundle(test_path)
    size(Y_train, 1) == size(Y_test, 1) || error("Train and test gene dimensions differ")

    id_cols = unique(vcat(context_cols, condition_cols))
    for meta in [train_meta, test_meta]
        validate_columns(meta, context_cols, "context")
        validate_columns(meta, condition_cols, "condition")
    end
    assert_unique_conditions(test_meta, context_cols, condition_cols)

    means = context_means(train_meta, Y_train, context_cols)
    reference_test = reference_matrix(test_meta, means, context_cols, size(Y_test, 1))
    systematic = vcat(
        systematic_variation("train", train_meta, Y_train, means, context_cols, id_cols),
        systematic_variation("test", test_meta, Y_test, means, context_cols, id_cols),
    )
    per_condition = condition_metrics(
        test_meta, Y_test, Yhat_test, reference_test, id_cols, context_cols, top_k,
    )
    summary = summarize_metrics(per_condition)
    systematic_summary = summarize_systematic_variation(systematic)

    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "systematic_variation.csv"), systematic)
    CSV.write(joinpath(output_dir, "systematic_variation_summary.csv"), systematic_summary)
    CSV.write(joinpath(output_dir, "metrics_per_condition.csv"), per_condition)
    CSV.write(joinpath(output_dir, "summary.csv"), summary)
    open(joinpath(output_dir, "manifest.toml"), "w") do io
        TOML.print(io, Dict(
            "run_dir" => abspath(run_dir),
            "train_predictions" => abspath(train_path),
            "test_predictions" => abspath(test_path),
            "context_columns" => string.(context_cols),
            "condition_columns" => string.(condition_cols),
            "top_k" => top_k,
            "n_genes" => size(Y_test, 1),
            "n_train" => size(Y_train, 2),
            "n_test" => size(Y_test, 2),
            "perturbed_centroid_source" => "training observations only",
            "expression_space" => "log-normalized pseudobulk delta",
        ))
    end
    @info "Systema outputs written to $output_dir"
end


function build_argument_parser()
    settings = ArgParseSettings(description = "Evaluate saved BioPert predictions with Systema")
    @add_arg_table settings begin
        "run_dir"
            arg_type = String
            help = "Run directory containing train/test prediction JLD2 files"
        "output_dir"
            arg_type = String
            help = "Directory for Systema CSVs and manifest"
        "--context_cols"
            arg_type = String
            default = "cell_line"
            help = "Comma-separated context columns"
        "--condition_cols"
            arg_type = String
            default = "drug,dose,time"
            help = "Comma-separated perturbation-condition columns"
        "--top_k"
            arg_type = Int
            default = 20
            help = "Number of largest observed effects for top-k Pearson"
    end
    return settings
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(build_argument_parser())
    run(
        args["run_dir"],
        args["output_dir"];
        context_cols = parse_columns(args["context_cols"]),
        condition_cols = parse_columns(args["condition_cols"]),
        top_k = args["top_k"],
    )
end