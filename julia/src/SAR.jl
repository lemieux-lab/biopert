module SAR  # Structure-Activity Relationship

export build_sar_table

using Combinatorics, DataFrames, LinearAlgebra, Parquet2, ProgressMeter
using PythonCall, Random, Statistics
using ..Metrics


const BIOPERT_ROOT = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
const CHEM_FILE = normpath(joinpath(BIOPERT_ROOT, "python", "src", "chem_utils.py"))
const chem = Ref{Py}()

function __init__()
    isfile(CHEM_FILE) || error("Python module not found: $CHEM_FILE")
    py_dir = dirname(CHEM_FILE)
    sys    = pyimport("sys")
    pycontains(sys.path, py_dir) || sys.path.insert(0, py_dir)
    chem[] = pyimport("chem_utils")
end


function load_embeddings(path::String)::Dict{String, Vector{Float32}}
    df = DataFrame(Parquet2.readfile(path); copycols=false)
    return Dict(
        s => Vector{Float32}(reinterpret(Float32, e)) for (s, e) in zip(df.smiles, df.embedding)
    )
end


function tanimoto_distance(smiles_i::String, smiles_j::String)
    result = chem[].tanimoto_distance(smiles_i, smiles_j)
    pyisinstance(result, pybuiltins.float) ? pyconvert(Float32, result) : missing
end


# Tanimoto distance between two binary (0/1) fingerprint vectors:
# `1 - |A∩B| / |A∪B|`, with `|A∪B| = |A| + |B| - |A∩B|`. The appropriate metric
# for binary fingerprints — unlike Euclidean (which reduces to `sqrt(Hamming)`
# here), it normalises by molecule size. Returns `0f0` when both vectors are
# all-zero (empty union).
function tanimoto_dist_vec(a::Vector{Float32}, b::Vector{Float32})
    inter = dot(a, b)
    union = sum(a) + sum(b) - inter
    return union == 0 ? 0f0 : Float32(1 - inter / union)
end


# Cosine distance for continuous embeddings: `1 - cos(a, b)`, oriented so that
# larger means "more chemically different", matching `tanimoto_dist_vec`'s
# direction. Returns `missing` if either vector has zero norm.
function cosine_dist_vec(a::Vector{Float32}, b::Vector{Float32})
    denom = norm(a) * norm(b)
    return denom == 0 ? missing : Float32(1 - dot(a, b) / denom)
end


# Chemical distance between the embeddings of two SMILES `si`/`sj` looked up in
# `smiles_to_embeds`. Dispatches to `tanimoto_dist_vec` when `binary` is true
# (fingerprints) or `cosine_dist_vec` otherwise (continuous embeddings) —
# `binary` is precomputed once per representation by the caller via `is_binary`,
# rather than re-detected per pair. Returns `missing` if either SMILES is
# absent from `smiles_to_embeds`.
function embedding_dist(smiles_to_embeds::Dict, si::String, sj::String, binary::Bool)
    emb_i = get(smiles_to_embeds, si, nothing)
    emb_j = get(smiles_to_embeds, sj, nothing)
    (isnothing(emb_i) || isnothing(emb_j)) && return missing
    return binary ? tanimoto_dist_vec(emb_i, emb_j) : cosine_dist_vec(emb_i, emb_j)
end


# Return `true` if every embedding vector in `smiles_to_embeds` holds only 0/1
# values, i.e. `smiles_to_embeds` is a binary fingerprint representation rather
# than a continuous one. Detected from the vectors themselves rather than a
# hardcoded name list, so a newly added fingerprint family gets the right
# metric without touching this code.
function is_binary(smiles_to_embeds::Dict{String, Vector{Float32}})
    for v in values(smiles_to_embeds)
        all(x -> x == 0f0 || x == 1f0, v) || return false
    end
    return true
end


"""
    build_sar_table(delta_df; embeddings_dir=nothing) -> DataFrame

Build a table: one row per compound pair, comparing chemical distance against
biological (delta profile) similarity.

Compounds are paired within the same `(cell_line, dose, time, plate)` group in
`delta_df` (each SMILES deduplicated to a single random row per group first).
For every pair, the biological side reports Pearson/Spearman correlation, cosine
similarity, L2 distance, and MSE between the two delta profiles.

When `embeddings_dir` is `nothing` (the default), every pair gets a single
`ecfp6_2048_dist` column (RDKit-computed ECFP6, 2048 bits). When `embeddings_dir`
is given, it should instead contain one subdirectory per molecular representation,
each with an `embeds.parquet`; one chemical-distance column is added per
representation found there, using the metric appropriate to its type, and
`ecfp6_2048_dist` is not computed (redundant with an `ECFP6_2048` representation,
if present in `embeddings_dir`). Larger means more chemically different.
"""
function build_sar_table(
    delta_df::DataFrame;
    embeddings_dir::Union{String, Nothing}=nothing,
)
    # Load all embeddings upfront
    all_embeddings = if isnothing(embeddings_dir)
        Dict{String, Dict{String, Vector{Float32}}}()
    else
        Dict(name => load_embeddings(joinpath(embeddings_dir, name, "embeds.parquet"))
             for name in readdir(embeddings_dir)
             if isfile(joinpath(embeddings_dir, name, "embeds.parquet")))
    end

    # Metric choice is a property of the representation, so resolve it once per
    # representation rather than once per pair.
    is_binary_emb = Dict(
        name => is_binary(smiles_to_embeds) for (name, smiles_to_embeds) in all_embeddings
    )
    for name in sort(collect(keys(all_embeddings)))
        metric = is_binary_emb[name] ? "tanimoto" : "cosine"
        @info "SAR distance metric" representation=name metric=metric
    end

    rows = []
    groups = groupby(delta_df, [:cell_line, :dose, :time, :plate])

    @showprogress desc="SAR analysis..." for gdf in groups
        cell_line = gdf.cell_line[1]
        dose      = gdf.dose[1]
        time      = gdf.time[1]
        plate     = gdf.plate[1]

        # Deduplicate by SMILES so each compound appears once, keeping a random row
        # per SMILES.
        gdf_u = unique(gdf[shuffle(1:nrow(gdf)), :], :smiles)
        n     = nrow(gdf_u)
        n < 2 && continue

        pairs = collect(combinations(1:n, 2))

        for (i, j) in pairs
            si = gdf_u.smiles[i]
            sj = gdf_u.smiles[j]

            if isnothing(embeddings_dir)
                ecfp6_2048_dist = tanimoto_distance(si, sj)
                ismissing(ecfp6_2048_dist) && continue
            end

            y_i = gdf_u.expr[i]
            y_j = gdf_u.expr[j]

            base = (
                cell_line    = cell_line,
                dose         = dose,
                time         = time,
                plate        = plate,
                drug_i       = gdf_u.drug[i],
                drug_j       = gdf_u.drug[j],
                smiles_i     = si,
                smiles_j     = sj,
                delta_norm_i = norm(y_i),
                delta_norm_j = norm(y_j),
                pearson      = pearson_corr(y_i, y_j),
                spearman     = spearman_corr(y_i, y_j),
                cosine       = cosine_similarity(y_i, y_j),
                l2           = l2_dist(y_i, y_j),
                mse          = mse(y_i, y_j),
            )

            row = if isnothing(embeddings_dir)
                merge(base, (ecfp6_2048_dist = ecfp6_2048_dist,))
            else
                emb_dists = NamedTuple(
                    Symbol(name, "_dist") =>
                        embedding_dist(smiles_to_embeds, si, sj, is_binary_emb[name])
                    for (name, smiles_to_embeds) in all_embeddings
                )
                merge(base, emb_dists)
            end

            push!(rows, row)
        end
    end
    return DataFrame(rows)
end


end