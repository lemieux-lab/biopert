module Observations

export Obs
export subset_obs, split_obs, apply_pinned_split
export concatenate_inputs, average_expr, build_delta_df, build_obs

using DataFrames, Statistics
using ..Splits


const UNTRT_AVG_GROUP_COLS = [:cell_line, :plate, :time]
const DELTA_AVG_GROUP_COLS = [:cell_line, :drug, :dose, :time]


struct Obs
    meta_df           :: DataFrame
    avg_untrt_targets :: Matrix{Float32}                          # (n_genes × n_obs)
    avg_delta_targets :: Matrix{Float32}                          # (n_genes × n_obs)
    # Each delta_ref_pools entry holds the reference cell line's delta profiles that share
    # the (drug, dose, time) of the corresponding meta_df row
    delta_ref_pools   :: Union{Vector{Matrix{Float32}}, Nothing}  # ((n_genes x n_replicates) x n_obs)
    molec_embeds      :: Union{Matrix{Float32}, Nothing}          # (n_embed_dims × n_obs)
    times             :: Union{Matrix{Float32}, Nothing}
    doses             :: Union{Matrix{Float32}, Nothing}
end

Obs(meta_df, avg_untrt_targets, avg_delta_targets;
    delta_ref_pools=nothing, molec_embeds=nothing, times=nothing, doses=nothing) =
    Obs(meta_df, avg_untrt_targets, avg_delta_targets, delta_ref_pools, molec_embeds, times, doses)


function subset_obs(obs::Obs, idxs::AbstractVector{Int})
    slice(m) = isnothing(m) ? nothing : m[:, idxs]
    return Obs(
        obs.meta_df[idxs, :],
        obs.avg_untrt_targets[:, idxs],
        obs.avg_delta_targets[:, idxs],
        isnothing(obs.delta_ref_pools) ? nothing : obs.delta_ref_pools[idxs],
        slice(obs.molec_embeds),
        slice(obs.times),
        slice(obs.doses),
    )
end


function split_obs(
    obs::Obs;
    cutoff::Float64      = 0.4,
    val_frac::Float64    = 0.1,
    test_frac::Float64   = 0.1,
    split_seed::Int      = 42
)
    train_smiles, val_smiles, test_smiles = split_smiles(
        obs.meta_df.smiles;
        cutoff       = cutoff,
        val_frac     = val_frac,
        test_frac    = test_frac,
        seed         = split_seed,
    )

    train_idx = findall(row -> row.smiles in train_smiles, eachrow(obs.meta_df))
    val_idx   = findall(row -> row.smiles in val_smiles,   eachrow(obs.meta_df))
    test_idx  = findall(row -> row.smiles in test_smiles,  eachrow(obs.meta_df))

    return subset_obs(obs, train_idx), subset_obs(obs, val_idx), subset_obs(obs, test_idx)
end


function apply_pinned_split(
    obs::Obs;
    val_smiles::Set{String}  = Set{String}(),
    test_smiles::Set{String} = Set{String}(),
    val_cl::Set{Symbol}      = Set{Symbol}(),
    test_cl::Set{Symbol}     = Set{Symbol}(),
)
    smiles     = [string(strip(s)) for s in obs.meta_df.smiles]
    cell_lines = Symbol.(obs.meta_df.cell_line)
    n          = length(smiles)

    train_idx = Int[]
    val_idx   = Int[]
    test_idx  = Int[]

    for i in 1:n
        if cell_lines[i] in test_cl || smiles[i] in test_smiles
            push!(test_idx, i)
        elseif cell_lines[i] in val_cl || smiles[i] in val_smiles
            push!(val_idx, i)
        else
            push!(train_idx, i)
        end
    end

    @info "Split sizes (pinned) — train: $(length(train_idx)), val: $(length(val_idx)), test: $(length(test_idx))"
    return subset_obs(obs, train_idx), subset_obs(obs, val_idx), subset_obs(obs, test_idx)
end


function concatenate_inputs(obs::Obs)::Matrix{Float32}
    # Pick one random reference cell line delta profile per pool.
    # Placed first so callers that resample delta_ref_pools per epoch can
    # overwrite these rows in-place via X[1:delta_ref_n_genes, :]
    delta_ref = isnothing(obs.delta_ref_pools) ? nothing :
        reduce(hcat, [P[:, rand(1:size(P, 2))] for P in obs.delta_ref_pools])
    parts = filter(!isnothing, [delta_ref, obs.avg_untrt_targets, obs.molec_embeds,
                                obs.times, obs.doses])
    return vcat(parts...)
end


function average_expr(df::DataFrame, group_cols::Vector{Symbol})
    groups = groupby(df, group_cols)

    # Holds the per-group mean, but stays named :expr so downstream code can
    # treat it like any other expr column
    col_specs = Pair{Symbol, AbstractVector}[
        c => [first(g[!, c]) for g in groups] for c in group_cols
    ]
    push!(col_specs, :expr => [vec(mean(reduce(hcat, g.expr), dims=2)) for g in groups])

    return DataFrame(col_specs...)
end


function build_delta_df(untrt_df::DataFrame, trt_df::DataFrame)
    # DMSO profiles are averaged per (cell line, plate, time) before subtracting from each matching treated profile
    avg_untrt_df = average_expr(untrt_df, UNTRT_AVG_GROUP_COLS)
    # Indexed by (cell_line, plate, time) for quick look up
    untrt_index = Dict(
        (row.cell_line, row.plate, row.time) => row.expr
        for row in eachrow(avg_untrt_df)
    )

    # Tracks (cell_line, plate, time) keys already warned about, so a missing
    # DMSO match warns once per key rather than once per treated row
    warned_keys = Set{Tuple{Symbol,Symbol,Symbol}}()

    matched_idxs = Int[]
    delta = Vector{Float32}[]

    for (i, row) in enumerate(eachrow(trt_df))
        key = (row.cell_line, row.plate, row.time)
        avg_untrt_expr = get(untrt_index, key, nothing)
        if isnothing(avg_untrt_expr)
            if key ∉ warned_keys
                @warn "No DMSO match found for cell_line=$(row.cell_line), plate=$(row.plate), time=$(row.time) — skipping"
                push!(warned_keys, key)
            end
            continue
        end
        push!(matched_idxs, i)
        push!(delta, row.expr .- avg_untrt_expr)
    end

    meta_df = select(trt_df[matched_idxs, :], Not(:expr))
    return hcat(meta_df, DataFrame(expr = delta))
end


# Looks up `key` in `index` and applies `xform`; falls back to an NaN-filled
# array of `fallback_size` on a miss, so one bad lookup degrades a single
# observation instead of aborting the whole build.
function lookup_or_nan(xform, index::AbstractDict, key, fallback_size::Tuple)
    val = get(index, key, nothing)
    return isnothing(val) ? fill(NaN32, fallback_size) : xform(val)
end

function build_obs(
    ref_cl::Symbol,
    untrt_df::DataFrame,
    trt_df::DataFrame;
    use_delta_ref::Bool = true,
    smiles_to_embeds::Union{Dict{String, Vector{Float32}}, Nothing} = nothing,
)
    untrt_target_df = filter(row -> row.cell_line != ref_cl, untrt_df)
    # Untreated target profiles are averaged per cell line across the entire dataset
    avg_untrt_target_df = average_expr(untrt_target_df, [:cell_line])
    # Indexed by cell_line for quick look up
    untrt_target_index = Dict(row.cell_line => row.expr for row in eachrow(avg_untrt_target_df))

    trt_target_df = filter(row -> row.cell_line != ref_cl, trt_df)

    delta_target_df = build_delta_df(untrt_target_df, trt_target_df)
    # Delta target profiles are averaged per (cell line, treatment) across the entire dataset
    avg_delta_target_df = average_expr(delta_target_df, DELTA_AVG_GROUP_COLS)
    avg_delta_targets = reduce(hcat, avg_delta_target_df.expr)

    meta_df = select(avg_delta_target_df, Not(:expr))

    # One averaged target untreated profile per observation, matched to meta_df's cell_line
    avg_untrt_targets = reduce(hcat, [untrt_target_index[cl] for cl in meta_df.cell_line])

    delta_ref_pools = nothing
    if use_delta_ref
        untrt_ref_df = filter(row -> row.cell_line == ref_cl, untrt_df)
        trt_ref_df = filter(row -> row.cell_line == ref_cl, trt_df)
        delta_ref_df = build_delta_df(untrt_ref_df, trt_ref_df)

        # Indexed by (drug, dose, time) for quick look up
        delta_ref_index = Dict{Tuple{Symbol,Symbol,Symbol}, Vector{Vector{Float32}}}()
        for row in eachrow(delta_ref_df)
            key = (row.drug, row.dose, row.time)
            push!(get!(delta_ref_index, key, Vector{Float32}[]), row.expr)
        end
        isempty(delta_ref_index) && error(
            "No treated profiles found for reference cell line $ref_cl; " *
            "cannot build delta_ref_pools"
        )

        # One replicate pool per observation, matched to meta_df's (drug, dose, time)
        n_genes = length(first(first(values(delta_ref_index))))
        delta_ref_pools = [
            lookup_or_nan(v -> reduce(hcat, v), delta_ref_index,
                          (row.drug, row.dose, row.time), (n_genes, 1))
            for row in eachrow(meta_df)
        ]
    end

    molec_embeds = nothing
    if !isnothing(smiles_to_embeds)
        isempty(smiles_to_embeds) && error("smiles_to_embeds is empty; cannot build molec_embeds")
        n_embed_dims = length(first(values(smiles_to_embeds)))
        molec_embeds = reduce(hcat, [
            lookup_or_nan(identity, smiles_to_embeds, s, (n_embed_dims,))
            for s in meta_df.smiles
        ])
    end

    # Drop observations left with missing (NaN-filled) values from a failed
    # delta_ref_pools or molec_embeds look up
    valid_idxs = filter(1:nrow(meta_df)) do i
        (isnothing(molec_embeds)   || !any(isnan, view(molec_embeds, :, i))) &&
        (isnothing(delta_ref_pools) || !any(isnan, delta_ref_pools[i]))
    end

    return Obs(
        meta_df[valid_idxs, :],
        avg_untrt_targets[:, valid_idxs],
        avg_delta_targets[:, valid_idxs];
        delta_ref_pools = isnothing(delta_ref_pools) ? nothing : delta_ref_pools[valid_idxs],
        molec_embeds    = isnothing(molec_embeds)    ? nothing : molec_embeds[:, valid_idxs],
    )
end


end
