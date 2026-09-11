module PCAs

using JLD2, PythonCall, SHA
using ..Observations: Obs, obs_signature


# External Python package: scikit-learn
const sklearn_decomp = Ref{Py}()

function __init__()
    sklearn_decomp[] = pyimport("sklearn.decomposition")
end


# Fit on train data.
function fit(expr_train::Matrix{Float32}, n_components::Int; random_state::Int = 42)
    pca = sklearn_decomp[].PCA(n_components = n_components, random_state = random_state)
    pca.fit(pylist(eachcol(expr_train)))
    return pca
end


function extract_parameters(pca)
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
    # pca.transform is scikit-learn's own PCA.transform (not this Julia function),
    # reached via PythonCall attribute access; it returns (n_samples, n_components),
    # so transpose to match this codebase's (n_components, n_samples) convention.
    result = pca.transform(pylist(eachcol(expr)))
    return pyconvert(Matrix{Float32}, result.copy())'
end


function fit_and_transform_returning_model(
    train::Matrix{Float32},
    val::Matrix{Float32},
    test::Matrix{Float32},
    n_components::Int,
    random_state::Int = 42,
)
    pca = fit(train, n_components; random_state = random_state)
    return pca, transform(pca, train), transform(pca, val), transform(pca, test)
end


function build_cache_file_path(
    pca_dir::Union{Nothing, String},
    label::String,
    n_components::Int,
    jld2_path::String,
    train_obs::Obs,
    val_obs::Obs,
    test_obs::Obs;
    extra = (),
)
    pca_dir === nothing && return nothing
    mkpath(pca_dir)
    payload = join(string.([
        label,
        n_components,
        jld2_path,
        size(train_obs.avg_delta_target_exprs),
        size(val_obs.avg_delta_target_exprs),
        size(test_obs.avg_delta_target_exprs),
        obs_signature(train_obs),
        obs_signature(val_obs),
        obs_signature(test_obs),
        extra...,
    ]), "\0")
    digest = bytes2hex(sha256(payload))
    return joinpath(pca_dir, "pca_$(label)_$(digest).jld2")
end


function load_cache(cache_file::String, return_parameters::Bool)
    data = JLD2.load(cache_file)
    if return_parameters
        if all(haskey(data, key) for key in ["mean", "components"])
            params = (mean = data["mean"], components = data["components"])
            return params, data["train"], data["val"], data["test"]
        end
        @warn "PCA cache lacks fitted parameters; recomputing" cache_file
        return nothing
    end
    return data["train"], data["val"], data["test"]
end


function load_or_fit_and_transform(
    train::Matrix{Float32},
    val::Matrix{Float32},
    test::Matrix{Float32},
    n_components::Int,
    cache_file::Union{Nothing, String} = nothing,
    random_state::Int = 42;
    return_parameters::Bool = false,
)
    if cache_file !== nothing && isfile(cache_file)
        @info "PCA: loading cached transforms from $cache_file"
        cached = load_cache(cache_file, return_parameters)
        cached !== nothing && return cached
    end

    if cache_file === nothing
        @info "PCA: fit and transform"
    else
        @info "PCA: fit, transform, and save to $cache_file"
    end
    pca     = fit(train, n_components; random_state = random_state)
    params  = extract_parameters(pca)
    train_t = transform(pca, train)
    val_t   = transform(pca, val)
    test_t  = transform(pca, test)
    cache_file !== nothing && JLD2.save(
        cache_file,
        "mean", params.mean,
        "components", params.components,
        "train", train_t,
        "val", val_t,
        "test", test_t,
    )
    return return_parameters ? (params, train_t, val_t, test_t) : (train_t, val_t, test_t)
end


end