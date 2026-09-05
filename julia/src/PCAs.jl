module PCAs

using JLD2, PythonCall


# External Python package: scikit-learn
const sklearn_decomp = Ref{Py}()

function __init__()
    sklearn_decomp[] = pyimport("sklearn.decomposition")
end


function fit(expr_train::Matrix{Float32}, n_components::Int; random_state::Int = 42)
    pca = sklearn_decomp[].PCA(n_components = n_components, random_state = random_state)
    pca.fit(pylist(eachcol(expr_train)))
    return pca
end


function parameters(pca)
    return (
        mean = pyconvert(Vector{Float32}, pca.mean_.copy()),
        components = pyconvert(Matrix{Float32}, pca.components_.copy()),
    )
end


function transform(pca, expr::Matrix{Float32})::Matrix{Float32}
    if size(expr, 2) == 0
        n_components = pyconvert(Int, pca.n_components_)
        return Matrix{Float32}(undef, n_components, 0)
    end
    result = pca.transform(pylist(eachcol(expr)))
    return pyconvert(Matrix{Float32}, result.copy())'
end


function fit_transform_returning_model(
    train::Matrix{Float32},
    val::Matrix{Float32},
    test::Matrix{Float32},
    n_components::Int,
    random_state::Int = 42,
)
    pca = fit(train, n_components; random_state = random_state)
    return pca, transform(pca, train), transform(pca, val), transform(pca, test)
end


function load_cached_transforms(
    pca_file::Union{Nothing, String},
    return_parameters::Bool,
)
    (isnothing(pca_file) || !isfile(pca_file)) && return nothing

    @info "Loading cached PCA transforms from $pca_file"
    data = JLD2.load(pca_file)
    if !return_parameters
        return data["train"], data["val"], data["test"]
    end
    if all(haskey(data, key) for key in ["mean", "components"])
        params = (mean = data["mean"], components = data["components"])
        return params, data["train"], data["val"], data["test"]
    end
    @warn "PCA cache lacks fitted parameters; recomputing" pca_file
    return nothing
end


# Fit a PCA on `train`, transform all three splits, and (if `pca_file` is
# given) save the transforms and fitted parameters for later reuse.
function fit_and_transform(
    train::Matrix{Float32},
    val::Matrix{Float32},
    test::Matrix{Float32},
    n_components::Int,
    pca_file::Union{Nothing, String},
    random_state::Int,
    return_parameters::Bool,
)
    if isnothing(pca_file)
        @info "Fitting PCA (results will not be saved in a file)"
    else
        @info "Fitting PCA and saving transforms to $pca_file"
    end

    pca     = fit(train, n_components; random_state = random_state)
    params  = parameters(pca)
    train_t = transform(pca, train)
    val_t   = transform(pca, val)
    test_t  = transform(pca, test)
    !isnothing(pca_file) && JLD2.save(
        pca_file,
        "mean", params.mean,
        "components", params.components,
        "train", train_t,
        "val", val_t,
        "test", test_t,
    )
    return return_parameters ? (params, train_t, val_t, test_t) : (train_t, val_t, test_t)
end


function load_or_fit_transform(
    train::Matrix{Float32},
    val::Matrix{Float32},
    test::Matrix{Float32},
    n_components::Int,
    pca_file::Union{Nothing, String} = nothing,
    random_state::Int = 42;
    return_parameters::Bool = false,
)
    cached = load_cached_transforms(pca_file, return_parameters)
    !isnothing(cached) && return cached

    return fit_and_transform(
        train, val, test, n_components, pca_file, random_state, return_parameters,
    )
end


end