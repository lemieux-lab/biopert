module Repro

export pairwise_stats, filter_by_repro

using Combinatorics, DataFrames, LinearAlgebra
using ProgressMeter, Statistics, StatsBase


"""
    pairwise_stats(df; plate_mode, max_pairs)

Compute pairwise reproducibility statistics (Pearson, Spearman) for every pair
of profiles sharing the same condition (`cell_line`, `drug`, `dose`, `time`).
`plate_mode` selects which pairs within a condition are kept: `:intra` keeps
only same-plate pairs, `:inter` (default) keeps only cross-plate pairs.

`df` must have columns `cell_line`, `drug`, `dose`, `time`, `plate`, `sample`,
and `expr` (e.g. `filtered_lincs.jld2` or `pseudobulks_alpha_<alpha>.jld2`).

If a condition has more than `max_pairs` eligible pairs, `max_pairs` of them
are sampled without replacement; `max_pairs=nothing` (default) keeps all pairs.

Returns a DataFrame with one row per retained pair, with columns `cell_line`,
`drug`, `dose`, `time`, `plate_i`, `plate_j`, `rep_i`, `rep_j`, `norm_rep_i`,
`norm_rep_j`, `pearson`, and `spearman`.
"""
function pairwise_stats(df::DataFrame;
                        plate_mode::Symbol=:inter,
                        max_pairs::Union{Int, Nothing}=nothing)
    plate_mode in (:intra, :inter) ||
        error("plate_mode must be :intra or :inter (got $(plate_mode))")

    condition_cols = [:cell_line, :drug, :dose, :time]
    groups = groupby(df, condition_cols)

    cell_line  = eltype(df.cell_line)[]
    drug       = eltype(df.drug)[]
    dose       = eltype(df.dose)[]
    time       = eltype(df.time)[]
    plate_i    = eltype(df.plate)[]
    plate_j    = eltype(df.plate)[]
    rep_i      = eltype(df.sample)[]
    rep_j      = eltype(df.sample)[]
    norm_rep_i = Float64[]
    norm_rep_j = Float64[]
    pearson    = Float64[]
    spearman   = Float64[]

    @showprogress desc = "Reproducibility ($(plate_mode))..." for gdf in groups
        idxs = parentindices(gdf)[1]
        length(idxs) < 2 && continue

        same_plate = plate_mode == :intra
        pairs = [(i, j) for (i, j) in combinations(idxs, 2)
                 if (df.plate[i] == df.plate[j]) == same_plate]

        if !isnothing(max_pairs) && length(pairs) > max_pairs
            pairs = sample(pairs, max_pairs; replace=false)
        end

        for (i, j) in pairs
            push!(cell_line, gdf.cell_line[1])
            push!(drug, gdf.drug[1])
            push!(dose, gdf.dose[1])
            push!(time, gdf.time[1])
            push!(plate_i, df.plate[i])
            push!(plate_j, df.plate[j])
            push!(rep_i, df.sample[i])
            push!(rep_j, df.sample[j])
            push!(norm_rep_i, norm(df.expr[i]))
            push!(norm_rep_j, norm(df.expr[j]))
            push!(pearson, cor(df.expr[i], df.expr[j]))
            push!(spearman, corspearman(df.expr[i], df.expr[j]))
        end
    end

    return DataFrame(;
        cell_line, drug, dose, time, plate_i, plate_j,
        rep_i, rep_j, norm_rep_i, norm_rep_j, pearson, spearman,
    )
end


function filter_by_repro(df::DataFrame;
                        plate_mode::Symbol=:inter,
                        max_pairs::Union{Int, Nothing}=nothing,
                        criteria::Symbol=:pearson,
                        threshold::Float64=0.7)
    criteria in (:pearson, :spearman) ||
        error("criteria must be :pearson or :spearman (got $(criteria))")
    
    ps = pairwise_stats(df; plate_mode=plate_mode, max_pairs=max_pairs)

    group_cols = [:cell_line, :drug, :dose, :time]
    agg = combine(groupby(ps, group_cols), criteria => minimum => :min_repro)

    reproducible_conditions = Set(Tuple(row[k] for k in group_cols)
                   for row in eachrow(agg) if row.min_repro >= threshold)

    filtered_df = filter(row -> Tuple(row[k] for k in group_cols) in reproducible_conditions, df)
    @info "filter_by_repro: kept $(nrow(filtered_df))/$(nrow(df)) rows (min $(criteria) ≥ $(threshold))"
    return filtered_df
end


end