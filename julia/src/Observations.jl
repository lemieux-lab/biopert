module Observations

export Obs
export obs_signature
export average_expr_df, build_delta_df, build_time_one_hot
export concatenate_inputs, build_obs
export subset_obs, split_obs, apply_pinned_split

using DataFrames, Random, SHA, Statistics
using ..DoseUtils, ..Splits


struct Obs
    meta_df            :: DataFrame

    # Treatment representations
    delta_ref_exprs    :: Union{Matrix{Float32}, Nothing}   # (n_genes × n_obs)
    molec_embeds       :: Union{Matrix{Float32}, Nothing}   # (n_embed_dims × n_obs)
    time_feats         :: Union{Matrix{Float32}, Nothing}   # (n_time_feats × n_obs)
    dose_feats         :: Union{Matrix{Float32}, Nothing}   # (n_dose_feats × n_obs)

    # Target cell line representation
    avg_untrt_target_exprs :: Matrix{Float32}               # (n_genes × n_obs)

    # Ground-truth
    avg_delta_target_exprs :: Matrix{Float32}               # (n_genes × n_obs)
end


function obs_signature(obs::Obs)
    cols = intersect([:cell_line, :drug, :smiles, :dose, :time], Symbol.(names(obs.meta_df)))
    rows = [join((string(row[c]) for c in cols), "|") for row in eachrow(obs.meta_df)]
    return bytes2hex(sha256(join(sort(rows), "\n")))
end


# ── Data prep ─────────────────────────────────────────────────────────────────

function average_expr_df(df::DataFrame, group_cols::Vector{Symbol})
    groups = groupby(df, group_cols)

    # Holds the per-group mean, but stays named :expr so downstream code can
    # treat it like any other expr column.
    col_specs = Pair{Symbol, AbstractVector}[
        c => [first(g[!, c]) for g in groups] for c in group_cols
    ]
    push!(col_specs, :expr => [vec(mean(reduce(hcat, g.expr), dims=2)) for g in groups])

    return DataFrame(col_specs...)
end


function build_delta_df(untrt_df::DataFrame, trt_df::DataFrame)
    # DMSO profiles are averaged per (cell line, plate, time) before subtracting
    # from each matching treated profile
    avg_untrt_df = average_expr_df(untrt_df, [:cell_line, :plate, :time])
    # Indexed by (cell_line, plate, time) for quick look up
    untrt_index = Dict(
        (row.cell_line, row.plate, row.time) => row.expr
        for row in eachrow(avg_untrt_df)
    )

    # Tracks (cell_line, plate, time) keys already warned about, so a missing
    # DMSO match warns once per key rather than once per treated row.
    warned_keys = Set{Tuple{Symbol,Symbol,Symbol}}()

    matched_idxs = Int[]
    delta = Vector{Float32}[]

    for (i, row) in enumerate(eachrow(trt_df))
        key = (row.cell_line, row.plate, row.time)
        avg_untrt_expr = get(untrt_index, key, nothing)
        if isnothing(avg_untrt_expr)
            if key ∉ warned_keys
                @warn "No DMSO match found for cell_line=$(row.cell_line), plate=$(row.plate), " *
                    "time=$(row.time) — skipping"
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


# Returns nothing when there's only one time level (Tahoe has a single timepoint).
function build_time_one_hot(time_list::Vector{Symbol})::Union{Matrix{Float32}, Nothing}
    time_levels = unique(time_list)
    length(time_levels) <= 1 && return nothing

    T = zeros(Float32, length(time_levels), length(time_list))
    for (col, t) in zip(eachcol(T), time_list)
        col[findfirst(==(t), time_levels)] = 1f0
    end
    return T
end


# ── Build Obs ─────────────────────────────────────────────────────────────────

function lookup_or_nan(xform, index::AbstractDict, key, fallback_size::Tuple)
    val = get(index, key, nothing)
    return isnothing(val) ? fill(NaN32, fallback_size) : xform(val)
end


function drop_unmatched(
    keep_idxs::Vector{Int}, total::Int, reason::String,
    meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool,
)
    length(keep_idxs) == total &&
        return meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool

    @warn "Dropping $(total - length(keep_idxs)) observation(s) with no matching $reason."
    meta_df                = meta_df[keep_idxs, :]
    avg_delta_target_exprs = avg_delta_target_exprs[:, keep_idxs]
    avg_untrt_target_exprs = avg_untrt_target_exprs[:, keep_idxs]
    delta_ref_exprs        = delta_ref_exprs === nothing ? nothing : delta_ref_exprs[:, keep_idxs]
    ref_pool               = isempty(ref_pool) ? ref_pool : ref_pool[keep_idxs]
    return meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool
end


function concatenate_inputs(obs::Obs)::Matrix{Float32}
    parts = filter(!isnothing, [obs.delta_ref_exprs, obs.molec_embeds,
                                obs.time_feats, obs.dose_feats,
                                obs.avg_untrt_target_exprs])
    return vcat(parts...)
end


function build_obs(
    ref_cl::Symbol,
    untrt_df::DataFrame,
    trt_df::DataFrame;
    use_delta_ref::Bool = true,
    return_ref_pool::Bool = true,
    average_delta_ref::Bool = false,
    smiles_to_embeds::Union{Dict{String, Vector{Float32}}, Nothing} = nothing,
    dose_encoding::String = "gate",
    seed::Int = 42,
)
    if return_ref_pool
        use_delta_ref || error("return_ref_pool=true requires use_delta_ref=true.")
        average_delta_ref &&
            error("return_ref_pool=true is incompatible with average_delta_ref=true.")
    end

    untrt_ref_df = filter(row -> row.cell_line == ref_cl, untrt_df)
    trt_ref_df   = filter(row -> row.cell_line == ref_cl, trt_df)
    delta_ref_df = build_delta_df(untrt_ref_df, trt_ref_df)

    untrt_target_df = filter(row -> row.cell_line != ref_cl, untrt_df)
    trt_target_df   = filter(row -> row.cell_line != ref_cl, trt_df)
    delta_target_df = build_delta_df(untrt_target_df, trt_target_df)

    # Untreated target profiles are averaged per cell line across the entire dataset
    avg_untrt_target_df = average_expr_df(untrt_target_df, [:cell_line])
    untrt_index         = Dict(row.cell_line => row.expr for row in eachrow(avg_untrt_target_df))

    # Delta target profiles are averaged per (cell line, treatment) across the entire
    # dataset; this is the ground truth.
    avg_delta_target_df = average_expr_df(
        delta_target_df, [:cell_line, :drug, :smiles, :dose, :time])
    nrow(avg_delta_target_df) == 0 &&
        error("No observations could be built — delta_target_df is empty.")

    meta_df                = select(avg_delta_target_df, Not(:expr))
    avg_delta_target_exprs = reduce(hcat, avg_delta_target_df.expr) |> Matrix{Float32}
    n_genes                = size(avg_delta_target_exprs, 1)

    if average_delta_ref
        delta_ref_df = average_expr_df(delta_ref_df, [:cell_line, :drug, :smiles, :dose, :time])
    end

    # Untreated target profiles, aligned to meta_df by cell_line.
    avg_untrt_target_exprs = reduce(
        hcat, [lookup_or_nan(identity, untrt_index, cl, (n_genes,)) for cl in meta_df.cell_line],
    ) |> Matrix{Float32}

    # Pre-index delta_ref_df by (drug, dose, time) to avoid O(N) DataFrame scans
    # inside the per-observation loop. Several replicate rows can share a key
    # (unless average_delta_ref=true), which is what makes a resamplable ref_pool.
    delta_ref_exprs = nothing
    ref_pool        = Vector{Vector{Vector{Float32}}}()
    if use_delta_ref
        ref_index = Dict{Tuple{Symbol,Symbol,Symbol}, Vector{Vector{Float32}}}()
        for r in eachrow(delta_ref_df)
            key = (r.drug, r.dose, r.time)
            push!(get!(ref_index, key, Vector{Float32}[]), r.expr)
        end

        ref_pool = [
            get(ref_index, (row.drug, row.dose, row.time), Vector{Float32}[])
            for row in eachrow(meta_df)
        ]

        # Drop observations whose (drug, dose, time) was never applied to the
        # reference cell line.
        keep_idxs = findall(!isempty, ref_pool)
        meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool =
            drop_unmatched(
                keep_idxs, length(ref_pool), "reference-cell-line (drug, dose, time) profile",
                meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool,
            )

        rng = MersenneTwister(seed)
        delta_ref_exprs = reduce(
            hcat, [p[rand(rng, 1:length(p))] for p in ref_pool],
        ) |> Matrix{Float32}
    end

    molec_embeds = nothing
    time_feats   = nothing
    dose_feats   = nothing
    if smiles_to_embeds !== nothing
        has_embed = [haskey(smiles_to_embeds, strip(String(s))) for s in meta_df.smiles]

        # Drop observations whose compound has no molecular embedding.
        keep_idxs = findall(has_embed)
        meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool =
            drop_unmatched(
                keep_idxs, length(has_embed), "molecular embedding",
                meta_df, avg_delta_target_exprs, avg_untrt_target_exprs, delta_ref_exprs, ref_pool,
            )

        molec_embeds = reduce(
            hcat,
            [smiles_to_embeds[strip(String(s))] for s in meta_df.smiles],
        ) |> Matrix{Float32}

        # Time and dose encoding
        time_feats = build_time_one_hot(meta_df.time)
        dose_feats = build_dose_feats(meta_df.dose; encoding = dose_encoding)
    end

    obs = Obs(
        meta_df,
        delta_ref_exprs,
        molec_embeds,
        time_feats,
        dose_feats,
        avg_untrt_target_exprs,
        avg_delta_target_exprs,
    )

    return return_ref_pool ? (obs, ref_pool) : obs
end


# ── Split obs ─────────────────────────────────────────────────────────────────

function subset_obs(obs::Obs, idxs::AbstractVector{Int})
    slice(m) = m !== nothing ? m[:, idxs] : nothing
    return Obs(
        obs.meta_df[idxs, :],
        slice(obs.delta_ref_exprs),
        slice(obs.molec_embeds),
        slice(obs.time_feats),
        slice(obs.dose_feats),
        slice(obs.avg_untrt_target_exprs),
        obs.avg_delta_target_exprs[:, idxs],
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

    train_idxs = findall(row -> row.smiles in train_smiles, eachrow(obs.meta_df))
    val_idxs   = findall(row -> row.smiles in val_smiles,   eachrow(obs.meta_df))
    test_idxs  = findall(row -> row.smiles in test_smiles,  eachrow(obs.meta_df))

    @info "Split sizes — train: $(length(train_idxs)), val: $(length(val_idxs)), " *
        "test: $(length(test_idxs))"
    return subset_obs(obs, train_idxs), subset_obs(obs, val_idxs), subset_obs(obs, test_idxs)
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

    train_idxs = Int[]
    val_idxs   = Int[]
    test_idxs  = Int[]

    for i in 1:n
        if cell_lines[i] in test_cl || smiles[i] in test_smiles
            push!(test_idxs, i)
        elseif cell_lines[i] in val_cl || smiles[i] in val_smiles
            push!(val_idxs, i)
        else
            push!(train_idxs, i)
        end
    end

    @info "Split sizes (pinned) — train: $(length(train_idxs)), val: $(length(val_idxs)), " *
        "test: $(length(test_idxs))"
    return subset_obs(obs, train_idxs), subset_obs(obs, val_idxs), subset_obs(obs, test_idxs)
end


end
