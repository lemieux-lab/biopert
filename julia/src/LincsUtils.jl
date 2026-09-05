module LincsUtils

export Lincs

using CSV, DataFrames, HDF5, JLD2, ProgressMeter
import Base.Threads.@spawn


struct Lincs
    expr::Matrix{Float32}
    gene::DataFrame
    compound::DataFrame
    inst::DataFrame
end


function load_cached(fn::String)
    d = load(fn)
    return d["lincs"]
end


function psortperm!(v, data, lo::Int=1, hi::Int=length(v))
    if lo >= hi
        return v
    end
    if hi - lo < 100_000
        sort!(view(v, lo:hi), alg=MergeSort, order=Base.Order.Perm(Base.Order.Forward, data))
        return v
    end
    mid = (lo + hi) >>> 1
    half = @spawn psortperm!(v, data, lo, mid)
    psortperm!(v, data, mid+1, hi)
    wait(half)
    temp = v[lo:mid]
    i, k, j = 1, lo, mid+1
    @inbounds while k < j <= hi
        @inbounds if data[v[j]] < data[temp[i]]
            v[k] = v[j]; j += 1
        else
            v[k] = temp[i]; i += 1
        end
        k += 1
    end
    @inbounds while k < j
        v[k] = temp[i]; k += 1; i += 1
    end
    return v
end

psortperm(data) = psortperm!(collect(1:length(data)), data)


function Lincs(prefix::String, gctx::String, out_fn::String)
    isfile(out_fn) && return load_cached(out_fn)

    println("Parsing from LINCS files...")
    f = h5open(prefix * gctx)
    lincs = try
        parse_lincs(f, prefix, out_fn)
    finally
        close(f)
    end
    return lincs
end


function parse_lincs(f, prefix::String, out_fn::String)
    expr = f["0/DATA/0/matrix"]
    exprGene_id = Symbol.(f["0/META/ROW/id"][:])

    println("Loading compound annotations...")
    compound_df = CSV.File(prefix * "compoundinfo_beta.txt",
                           delim="\t", types=String, missingstring=nothing, pool=false, ntasks=1) |> DataFrame
    gdf = groupby(compound_df, [:pert_id, :canonical_smiles, :inchi_key])
    compound_df = combine(gdf, :cmap_name => (x -> first(x)) => :first_name)
    compound_df[!, :pert_id] = Symbol.(compound_df.pert_id)

    println("Loading gene and sample annotations...")
    gene_df = CSV.File(prefix * "geneinfo_beta.txt",
                       delim="\t", types=String, missingstring=nothing, pool=false) |> DataFrame
    gene_df.gene_id       = Symbol.(gene_df.gene_id)
    gene_df.gene_type     = Symbol.(gene_df.gene_type)
    gene_df.src           = Symbol.(gene_df.src)
    gene_df.feature_space = Symbol.(gene_df.feature_space)

    inst_df = CSV.File(prefix * "instinfo_beta.txt",
                       delim='\t', types=String, missingstring=nothing, pool=false) |> DataFrame
    @Threads.threads for col in names(inst_df)
        inst_df[!, col] = Symbol.(inst_df[!, col])
    end

    expr_id = Symbol.(f["0/META/COL/id"][:])
    @assert sort(expr_id) == sort(inst_df.sample_id) "expression column ids and instance sample_ids don't match"
    e2s = psortperm(expr_id)
    i2s = psortperm(inst_df.sample_id)
    s2e = psortperm(e2s)
    i2e = i2s[s2e]
    inst_df = inst_df[i2e, :]

    println("Subsetting landmark genes...")
    lm_gene_df = filter(row -> row.feature_space == :landmark, gene_df)
    lm_id  = lm_gene_df.gene_id
    exprGene_row = Dict(id => i for (i, id) in enumerate(exprGene_id))
    lm_row = [exprGene_row[sym] for sym in lm_id]

    chunk_size = 32_000
    _, ninst = size(expr)
    nlm = length(lm_row)
    final = Matrix{Float32}(undef, (nlm, ninst))

    function load!(final, expr, lm_row, r::UnitRange{T}) where T <: Integer
        slab = expr[:, r]
        final[:, r] = slab[lm_row, :]
    end

    starts = collect(1:chunk_size:ninst)
    pbar = Progress(length(starts); desc="Subsetting landmark genes", showspeed=true)
    pbar_lock = Threads.SpinLock()
    @Threads.threads for start in starts
        r = start:min(start + chunk_size - 1, ninst)
        load!(final, expr, lm_row, r)
        lock(pbar_lock) do
            next!(pbar)
        end
    end
    finish!(pbar)

    lincs = Lincs(final, lm_gene_df, compound_df, inst_df)
    jldsave(out_fn; lincs)
    return lincs
end


end
