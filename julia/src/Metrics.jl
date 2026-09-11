module Metrics

export mse, rmse, pearson_corr, spearman_corr, cosine_similarity, l2_dist
export average_metrics, metrics_per_obs

using CSV, DataFrames, LinearAlgebra, Statistics, StatsBase


# ── Metrics ──────────────────────────────────────────────────────────────────

function mse(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    mean((ŷ .- y) .^ 2)
end

function rmse(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    sqrt(mean((ŷ .- y) .^ 2))
end

function pearson_corr(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    cor(y, ŷ)
end

function spearman_corr(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    corspearman(y, ŷ)
end

function l2_dist(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    norm(y - ŷ)
end

function cosine_similarity(
    y::AbstractVector{T}, ŷ::AbstractVector{T},
) where {T <: AbstractFloat}
    dot(y, ŷ) / (norm(y) * norm(ŷ) + eps(T))
end


# ── Wrappers ─────────────────────────────────────────────────────────────────

function average_rmse(
    Y::Matrix{T}, Ŷ::Matrix{T},
) where {T <: AbstractFloat}
    average_metric(Y, Ŷ, rmse)
end

function average_pearson_corr(
    Y::Matrix{T}, Ŷ::Matrix{T},
) where {T <: AbstractFloat}
    average_metric(Y, Ŷ, pearson_corr)
end

function average_spearman_corr(
    Y::Matrix{T}, Ŷ::Matrix{T},
) where {T <: AbstractFloat}
    average_metric(Y, Ŷ, spearman_corr)
end

function average_l2_dist(
    Y::Matrix{T}, Ŷ::Matrix{T},
) where {T <: AbstractFloat}
    average_metric(Y, Ŷ, l2_dist)
end

function average_cosine_similarity(
    Y::Matrix{T}, Ŷ::Matrix{T},
) where {T <: AbstractFloat}
    average_metric(Y, Ŷ, cosine_similarity)
end


# ── Aggregate metrics ───────────────────────────────────────────────────────

# Average value of metric f computed column-wise between Y and Ŷ
average_metric(Y::Matrix{T}, Ŷ::Matrix{T}, f::Function) where {T <: AbstractFloat} =
    mean(f(y, ŷ) for (y, ŷ) in zip(eachcol(Y), eachcol(Ŷ)))


function average_metrics(Y::Matrix{T}, Ŷ::Matrix{T}; digits::Int = 2) where {T <: AbstractFloat}
    return (
        avg_rmse       = round(average_rmse(Y, Ŷ);              digits=digits),
        avg_pearson    = round(average_pearson_corr(Y, Ŷ);      digits=digits),
        avg_spearman   = round(average_spearman_corr(Y, Ŷ);     digits=digits),
        avg_l2         = round(average_l2_dist(Y, Ŷ);           digits=digits),
        avg_cosine_sim = round(average_cosine_similarity(Y, Ŷ); digits=digits),
    )
end


function metrics_per_obs(
    meta_df::DataFrame, Y::Matrix{T}, Ŷ::Matrix{T}, outpath::String;
    digits::Int = 2,
) where {T <: AbstractFloat}
    n_obs = size(Y, 2)
    @assert size(Ŷ, 2) == n_obs
    @assert nrow(meta_df) == n_obs "meta_df must have one row per observation. " *
        "Got nrow(meta_df) = $(nrow(meta_df)) vs n_obs = $n_obs."

    rmses     = Vector{Float64}(undef, n_obs)
    pearsons  = Vector{Float64}(undef, n_obs)
    spearmans = Vector{Float64}(undef, n_obs)
    l2s       = Vector{Float64}(undef, n_obs)
    cosines   = Vector{Float64}(undef, n_obs)

    for j in 1:n_obs
        y = @view Y[:, j]
        ŷ = @view Ŷ[:, j]
        rmses[j]     = rmse(y, ŷ)
        pearsons[j]  = pearson_corr(y, ŷ)
        spearmans[j] = spearman_corr(y, ŷ)
        l2s[j]       = l2_dist(y, ŷ)
        cosines[j]   = cosine_similarity(y, ŷ)
    end

    results = copy(meta_df)
    results[!, :rmse]     = round.(rmses;     digits=digits)
    results[!, :pearson]  = round.(pearsons;  digits=digits)
    results[!, :spearman] = round.(spearmans; digits=digits)
    results[!, :l2]       = round.(l2s;       digits=digits)
    results[!, :cosine]   = round.(cosines;   digits=digits)

    # CSV.jl quotes String fields containing the delimiter but leaves Symbol fields
    # unquoted. Some drug names contain commas (e.g. "Dapagliflozin ((2S)-1,2-propanediol,
    # hydrate)"), which would shift every column to their right.
    for c in names(results)
        if eltype(results[!, c]) === Symbol
            results[!, c] = string.(results[!, c])
        end
    end

    CSV.write(outpath, results)
end


end