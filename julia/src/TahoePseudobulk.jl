module TahoePseudobulk

export build_sample_to_dose, build_pseudobulks, log_normalize!

using DataFrames, ProgressMeter
using ..TahoeUtils


function build_sample_to_dose(sample_meta::DataFrame)::Dict{Symbol, Symbol}
    sample_meta.sample = Symbol.(sample_meta.sample)
    rx = r"\('([^']+)',\s*([\d\.eE+-]+),\s*'([^']+)'\)"
    sample_to_dose = Dict{Symbol, Symbol}()
    for row in eachrow(sample_meta)
        m = match(rx, row.drugname_drugconc)
        isnothing(m) && continue
        sample_to_dose[row.sample] = Symbol(string(m.captures[2], " ", m.captures[3]))
    end
    return sample_to_dose
end


function build_pseudobulks(
    data_dir       :: String,
    coding_tokens  :: Vector{Int},
    sample_to_dose :: Dict{Symbol, Symbol},
)::DataFrame
    files = readdir(data_dir; join=true)
    p = Progress(length(files); desc="Build pseudobulks", showspeed=true)
    key_to_row = Dict{Tuple{Symbol, Symbol}, Int}()
    df = DataFrame(
        cell_line   = Symbol[],
        sample      = Symbol[],
        plate       = Symbol[],
        drug        = Symbol[],
        smiles      = String[],
        dose        = Symbol[],
        total_cells = Int[],
        total_umis  = Int[],
        expr        = Vector{Float32}[],
    )

    for (i, path) in enumerate(files)
        fp = TahoeUtils.FromParquet(path)
        fp = TahoeUtils.FromParquet(fp.s[coding_tokens, :], fp.df)

        cell_lines  = fp.df.cell_line
        samples     = fp.df.sample
        plates      = fp.df.plate
        drugs       = fp.df.drug
        smiles_list = fp.df.smiles
        doses       = getindex.(Ref(sample_to_dose), samples)

        # 1 sample = 1 well = 1 treatment (drug x dose x 24 h) on a MOSAIC tumor
        # Group columns by (cell_line, sample)
        key_to_col_idxs = Dict{Tuple{Symbol, Symbol}, Vector{Int}}()
        @inbounds for j in eachindex(cell_lines)
            key = (cell_lines[j], samples[j])
            push!(get!(key_to_col_idxs, key, Int[]), j)
        end

        # Build pseudobulks
        for (key, col_idxs) in key_to_col_idxs
            n_cells   = length(col_idxs)
            bulk      = vec(sum(fp.s[:, col_idxs]; dims=2))
            file_umis = Int(sum(bulk))

            if !haskey(key_to_row, key)
                j0 = col_idxs[1]
                push!(df, (
                    cell_line   = cell_lines[j0],
                    sample      = samples[j0],
                    plate       = plates[j0],
                    drug        = drugs[j0],
                    smiles      = String(smiles_list[j0]),
                    dose        = doses[j0],
                    total_cells = n_cells,
                    total_umis  = file_umis,
                    expr        = bulk,
                ))
                key_to_row[key] = nrow(df)
            else
                row = key_to_row[key]
                df.expr[row]        .+= bulk
                df.total_cells[row]  += n_cells
                df.total_umis[row]   += file_umis
            end
        end

        fp = nothing
        (i % 10 == 0) && GC.gc(false)
        next!(p)
    end

    finish!(p)
    return df
end


function log_normalize!(df::DataFrame; α::Int=10000, digits::Int=2)
    # Cast total_umis to Float32 (it was Int64)
    df[!, :total_umis] = Float32.(df[!, :total_umis])
    for i in eachindex(df.expr)
        v  = df.expr[i]
        s  = sum(v)
        if s == 0
            @warn "log_normalize!: encountered a zero-sum vector at row $i; returning zeros."
            df.expr[i]       = zeros(eltype(v), length(v))
            df.total_umis[i] = 0f0
        else
            sf               = α / s
            df.expr[i]       = log1p.(eltype(v)(sf) .* v)
            df.total_umis[i] = round(Float32(s); digits=digits)
        end
    end
    return df
end


end
