module Splits

export split_smiles, split_cell_lines, load_split_smiles, load_split_cell_lines

using DataFrames, JLD2, PythonCall, Random


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


function split_smiles(
    smiles_list::Vector{String};
    cutoff::Float64       = 0.4,
    val_frac::Float64     = 0.1,
    test_frac::Float64    = 0.1,
    seed::Int             = 42,
)
    unique_smiles = unique(smiles_list)
    train, val, test = chem[].butina_split(
        unique_smiles,
        cutoff    = cutoff,
        val_frac  = val_frac,
        test_frac = test_frac,
        seed      = seed
    )
    return (
        Set(pyconvert(Vector{String}, train)),
        Set(pyconvert(Vector{String}, val)),
        Set(pyconvert(Vector{String}, test)),
    )
end


# Partition cell lines into train/val/test sets. `cell_lines` is the per-observation
# cell line vector (one entry per observation), so observation counts can be derived.
function split_cell_lines(
    cell_lines::Vector{Symbol};
    val_frac::Float64       = 0.1,
    test_frac::Float64      = 0.1,
    seed::Int               = 42,
    min_obs::Int            = 0,
    exclude::Vector{Symbol} = Symbol[],
)
    @assert val_frac >= 0 && test_frac >= 0
    @assert val_frac + test_frac < 1.0

    counts = Dict{Symbol, Int}()
    for cl in cell_lines
        counts[cl] = get(counts, cl, 0) + 1
    end

    excluded = Set(exclude)
    # sort() before shuffle: Dict iteration order is not stable across sessions
    eligible = sort([cl for (cl, n) in counts if n >= min_obs && cl ∉ excluded])

    shuffled = shuffle(MersenneTwister(seed), eligible)
    n        = length(shuffled)
    n_val    = round(Int, val_frac  * n)
    n_test   = round(Int, test_frac * n)

    val_cl   = Set(shuffled[1 : n_val])
    test_cl  = Set(shuffled[n_val+1 : n_val+n_test])
    train_cl = setdiff(Set(keys(counts)), union(val_cl, test_cl))

    n_ineligible = length(counts) - n
    @info "Cell-line split — eligible: $n (of $(length(counts)), $n_ineligible ineligible → train), " *
          "train: $(length(train_cl)), val: $(length(val_cl)), test: $(length(test_cl))"
    return train_cl, val_cl, test_cl
end


# `path` points to a saved `Obs` covering only one subset (train, val, or test).
function load_split_smiles(path::String)::Set{String}
    data = load(path)
    haskey(data, "obs") || error("Split file must contain key \"obs\": $path")
    return Set(string.(strip.(data["obs"].meta_df.smiles)))
end


# `path` points to a file holding the cell lines assigned to one subset (train, val,
# or test).
function load_split_cell_lines(path::String)::Set{Symbol}
    data = load(path)
    haskey(data, "cell_lines") || error("Cell-line split file must contain key \"cell_lines\": $path")
    return Set(Symbol.(data["cell_lines"]))
end


end