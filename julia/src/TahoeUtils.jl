module TahoeUtils

export FromParquet

using CxxWrap, SparseArrays


const BIOPERT_ROOT = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
const LIBARROW_WRAP = normpath(joinpath(BIOPERT_ROOT, "cxx", "ArrowWrap", "build", "libarrow_wrap.so"))

function __init__()
    if !isfile(LIBARROW_WRAP)
        error("Shared library not found: $(LIBARROW_WRAP). Run scripts/setup_tahoe_deps.sh first.")
    end
    CxxWrap.@wrapmodule(() -> LIBARROW_WRAP, :define_julia_module)
end


struct FromParquet
    s  :: SparseMatrixCSC{Float32, Int32}
    df :: NamedTuple{
        (:drug, :sample, :barcode, :cell_line, :moa, :smiles, :pubchem, :plate),
        Tuple{Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol},
              Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}
    }
end


function FromParquet(path::String)::FromParquet
    r  = ParquetReaderWrapper(path)
    n  = 62712  # Number of genes in Tahoe
    m, nb = get_sizes(r)

    colptr      = Vector{Int32}(undef,   m + 1)
    rowval      = Vector{Int32}(undef,   nb)
    nzval       = Vector{Float32}(undef, nb)
    colptr[end] = nb + 1
    load_expr(r, colptr, rowval, nzval)

    # Column indices match the Tahoe parquet schema
    drug      = Vector{Symbol}(undef, m); load_syms(r, drug,      2)
    sample    = Vector{Symbol}(undef, m); load_syms(r, sample,    3)
    barcode   = Vector{Symbol}(undef, m); load_syms(r, barcode,   4)
    cell_line = Vector{Symbol}(undef, m); load_syms(r, cell_line, 5)
    moa       = Vector{Symbol}(undef, m); load_syms(r, moa,       6)
    smiles    = Vector{Symbol}(undef, m); load_syms(r, smiles,    7)
    pubchem   = Vector{Symbol}(undef, m); load_syms(r, pubchem,   8)
    plate     = Vector{Symbol}(undef, m); load_syms(r, plate,     9)

    return FromParquet(
        SparseMatrixCSC{Float32, Int32}(n, m, colptr, rowval, nzval),
        (
            drug      = drug,
            sample    = sample,
            barcode   = barcode,
            cell_line = cell_line,
            moa       = moa,
            smiles    = smiles,
            pubchem   = pubchem,
            plate     = plate,
        )
    )
end


end