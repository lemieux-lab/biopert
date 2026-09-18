using ArgParse, CSV, DataFrames, Dates, Flux, JLD2, ParameterSchedulers
using Parquet2, PythonCall, Random, SHA, Statistics, TOML
using Biopert
using Biopert.PCAs  # Not exported by Biopert itself, to avoid conflicts with other
                     # packages' own PCA-related exports (e.g. MultivariateStats).

const wandb = pyimport("wandb")


const _pybuiltins = pyimport("builtins")
function _pyval(v::Py)
    pyisinstance(v, _pybuiltins.bool)  && return pyconvert(Bool,    v)
    pyisinstance(v, _pybuiltins.int)   && return pyconvert(Int,     v)
    pyisinstance(v, _pybuiltins.float) && return pyconvert(Float64, v)
    pyisinstance(v, _pybuiltins.list)  && return [_pyval(item) for item in v]
    return pyconvert(String, _pybuiltins.str(v))
end


function configured_path(config::Dict, key::String)
    path = get(config, key, nothing)
    path === nothing && return nothing
    path_s = string(strip(path))
    isempty(path_s) && return nothing
    lowercase(path_s) in ("none", "nothing", "false") && return nothing
    return path_s
end


function cleanup_extra_checkpoints(run_dir::String, keep_path::Union{Nothing, String})
    keep_abs = keep_path === nothing ? nothing : abspath(keep_path)
    for name in readdir(run_dir)
        path = joinpath(run_dir, name)
        isfile(path) || continue
        endswith(name, ".jld2") || continue
        occursin("checkpoint", lowercase(name)) || startswith(lowercase(name), "best_model") || continue
        keep_abs !== nothing && abspath(path) == keep_abs && continue
        rm(path; force = true)
        @info "Removed extra checkpoint $path"
    end
end


function main(config_file::String, outdir::String, dataset::String)
    paths     = dataset_paths(outdir, dataset)
    jld2_path = paths.jld2_path

    # ── Run dir ──────────────────────────────────────────────────────────────────
    timestamp       = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    config_basename = splitext(basename(config_file))[1]
    jld2_basename   = splitext(basename(jld2_path))[1]
    run_dir = joinpath(paths.prediction_dir, "$(jld2_basename)_$(config_basename)_$(timestamp)")
    mkpath(run_dir)

    # ── Config ──────────────────────────────────────────────────────────────────
    config = TOML.parsefile(config_file)

    # Parameters
    seed                               = get(config, "seed", 42)
    split_seed                         = get(config, "split_seed", seed)
    Random.seed!(seed)
    # Data
    ref_cl                             = Symbol(config["ref_cl"])
    average_ref                        = get(config, "average_ref", false)
    resample_ref                       = get(config, "resample_ref", false)
    use_delta_ref                      = get(config, "use_delta_ref", true)
    # If molec_embed_file is provided, molec embeds will be used as input.
    molec_embed_file                   = get(config, "molec_embed_file", nothing)
    obs_dir                            = configured_path(config, "obs_dir")
    dose_encoding                      = get(config, "dose_encoding", "none")
    encode_time                        = get(config, "encode_time", false)
    # Data filtering
    landmark_genes_only                = get(config, "landmark_genes_only", false)
    # Data split (only Butina/compound-scaffold splitting is currently supported)
    val_frac                           = Float64(get(config,     "val_frac",  0.1))
    test_frac                          = Float64(get(config,     "test_frac", 0.1))
    val_obs_path                       = configured_path(config, "val_obs_path")
    test_obs_path                      = configured_path(config, "test_obs_path")
    # Cell-line holdout (second split axis; disabled by default)
    holdout_cell_lines                 = get(config,             "holdout_cell_lines", false)
    cellline_val_obs_path              = configured_path(config, "cellline_val_obs_path")
    cellline_test_obs_path             = configured_path(config, "cellline_test_obs_path")
    # Train data filtering
    repro_delta_inter_path             = get(config,         "repro_delta_inter_path", nothing)
    repro_delta_inter_threshold        = Float64(get(config, "repro_delta_inter_threshold", 0.1))
    # PCA
    # TODO: it should be two different parameters for delta_ref_pca_dim and untrt_target_pca_dim.
    n_pca_expr                         = get(config, "n_pca_expr", nothing)
    n_pca_molec                        = get(config, "n_pca_molec", nothing)
    use_pca_cache                      = get(config, "use_pca_cache", false)
    pca_dir                            = use_pca_cache ? paths.pca_dir : nothing
    # Model and training
    model_type                         = get(config, "model_type", "mlp")
    hidden_layers                      = config["hidden_layers"]
    dropout_arr                        = get(config, "dropout_arr", nothing)
    batch_size                         = config["batch_size"]
    n_epochs                           = config["n_epochs"]
    lr                                 = config["lr"]
    warmup_steps                       = get(config, "warmup_steps", nothing)
    weight_decay                       = config["weight_decay"]
    loss_name                          = get(config, "loss_name", "rmse")
    save_predictions                   = get(config, "save_predictions", false)

    # Check that there is no incompatibility between parameter values.
    if resample_ref
        model_type == "mlp" ||
            error("resample_ref=true is only supported for model_type=\"mlp\" (got \"$model_type\").")
        use_delta_ref       || error("resample_ref=true requires use_delta_ref=true.")
        average_ref         && error("resample_ref=true is incompatible with average_ref=true.")
    end

    if molec_embed_file !== nothing && isfile(molec_embed_file)
        dose_encoding == "gate" && model_type != "mlp" &&
            error("dose_encoding=\"gate\" is only supported for model_type=\"mlp\" (got \"$model_type\").")
    end

    # ── Initialize WandB run ───────────────────────────────────────────────────────
    wandb.init(
        project = "biopert",
        config  = pydict(config),
        mode    = config["wandb_mode"],
        name    = config_basename,
        dir     = run_dir,
    )
    wandb_params = Dict{String, Any}(pyconvert(String, k) => _pyval(v) for (k, v) in wandb.config.items())

    # Log input data path to WandB
    wandb.config.update(pydict(Dict("jld2_path" => jld2_path)))

    # Override hyperparameters with sweep values (wandb_params merges TOML config + sweep overrides)
    n_pca_expr    = get(wandb_params, "n_pca_expr",    n_pca_expr)
    n_pca_molec   = get(wandb_params, "n_pca_molec",   n_pca_molec)
    n_pca_expr    = (n_pca_expr == 0)  ? nothing :      n_pca_expr
    n_pca_molec   = (n_pca_molec == 0) ? nothing :     n_pca_molec
    hidden_layers = get(wandb_params, "hidden_layers", hidden_layers)
    dropout_arr   = get(wandb_params, "dropout_arr",   dropout_arr)
    batch_size    = get(wandb_params, "batch_size",    batch_size)
    n_epochs      = get(wandb_params, "n_epochs",      n_epochs)
    lr            = get(wandb_params, "lr",            lr)
    weight_decay  = get(wandb_params, "weight_decay",  weight_decay)
    loss_name     = get(wandb_params, "loss_name",     loss_name)
    resample_ref  = get(wandb_params, "resample_ref",  resample_ref)

    if isa(hidden_layers, String)
        hidden_layers = parse.(Int, split(strip(hidden_layers, ['[', ']']), ","))
    end
    if isa(dropout_arr, String)
        dropout_arr = parse.(Float64, split(strip(dropout_arr, ['[', ']']), ","))
    end

    # ── Load data ──────────────────────────────────────────────────────────────────
    df = load(jld2_path, "df")

    # Optionally restrict to landmark genes
    if landmark_genes_only
        df_tahoe_coding = CSV.read(
            joinpath(dirname(jld2_path), "tahoe_coding_tokens.csv"), DataFrame)
        df_shared       = CSV.read("data/lincs_and_tahoe_shared_genes.csv", DataFrame)
        shared_tokens   = Set(df_shared.token_id)
        mask            = [token in shared_tokens for token in df_tahoe_coding.coding_tokens]
        df.expr         = [expr[mask] for expr in df.expr]
    end

    # ── Molecular structure embeddings ──────────────────────────────────────────────────────────────────

    molec_embed_model = nothing
    smiles_to_embeds  = nothing

    if molec_embed_file !== nothing && isfile(molec_embed_file)
        @info "Loading molecular embeddings from $molec_embed_file"
        df_emb = Parquet2.Dataset(molec_embed_file) |> DataFrame

        smiles_to_embeds = Dict{String, Vector{Float32}}(
            string(strip(s)) => Vector{Float32}(reinterpret(Float32, e))
            for (s, e) in zip(df_emb.smiles, df_emb.embedding)
        )

        molec_embed_model = basename(dirname(molec_embed_file))

        embed_dim = length(first(values(smiles_to_embeds)))
        @info "Loaded embeddings for $(length(smiles_to_embeds)) compounds. Dim: $embed_dim"
    end

    # ── Build observations ──────────────────────────────────────────────────────────────────

    obs_params = (
        jld2_path, ref_cl, average_ref, use_delta_ref,
        molec_embed_file, landmark_genes_only, dose_encoding, seed,
    )
    # Only perturb the cache key when resampling is on, so existing (non-resample)
    # caches stay valid. Resample caches additionally store the reference pool.
    resample_ref && (obs_params = (obs_params..., :resample_ref))
    obs_cache_file = isnothing(obs_dir) ? nothing : joinpath(obs_dir, "obs_hash_$(hash(obs_params)).jld2")

    # ref_pool_full: candidate reference replicates per observation (raw gene space),
    # aligned to the full Obs columns via meta_df._obs_id. nothing unless resample_ref.
    obs, ref_pool_full = if !isnothing(obs_cache_file) && isfile(obs_cache_file)
        @info "Loading cached obs from $obs_cache_file"
        data = load(obs_cache_file)
        data["obs"], (resample_ref ? data["ref_pool"] : nothing)
    else
        untrt_df = filter(row -> row.drug == :DMSO, df)
        trt_df   = filter(row -> row.drug != :DMSO, df)

        # dose_encoding/time features are built internally by build_obs (only when
        # smiles_to_embeds is given), so there is no separate post-hoc attachment step.
        _obs, _ref_pool = if resample_ref
            build_obs(
                ref_cl, untrt_df, trt_df;
                use_delta_ref     = use_delta_ref,
                average_delta_ref = average_ref,
                smiles_to_embeds  = smiles_to_embeds,
                dose_encoding     = dose_encoding,
                seed              = seed,
                return_ref_pool   = true,
            )
        else
            build_obs(
                ref_cl, untrt_df, trt_df;
                use_delta_ref     = use_delta_ref,
                average_delta_ref = average_ref,
                smiles_to_embeds  = smiles_to_embeds,
                dose_encoding     = dose_encoding,
                seed              = seed,
                return_ref_pool   = false,
            ), nothing
        end

        if !isnothing(obs_cache_file)
            mkpath(obs_dir)
            if resample_ref
                JLD2.save(obs_cache_file, "obs", _obs, "ref_pool", _ref_pool)
            else
                JLD2.save(obs_cache_file, "obs", _obs)
            end
            @info "Saved obs to $obs_cache_file"
        end

        _obs, _ref_pool
    end

    if dose_encoding != "none"
        n_dose = obs.dose_feats === nothing ? 0 : size(obs.dose_feats, 1)
        n_time = obs.time_feats === nothing ? 0 : size(obs.time_feats, 1)
        @info "Dose/time encoding \"$dose_encoding\": $n_dose dose feature(s), $n_time time feature(s)"
    end

    # ── Data split ──────────────────────────────────────────────────────────────────

    # Pinned split files define held-out SMILES membership. Features are always
    # sliced from the current run's Obs so configs with different inputs remain valid.
    train_obs, val_obs, test_obs =
        if !isnothing(test_obs_path) || !isnothing(val_obs_path)
            isnothing(test_obs_path) && error("val_obs_path is set but test_obs_path is missing.")
            isfile(test_obs_path) || error("Configured test_obs_path does not exist: $test_obs_path")
            if val_frac != 0
                isnothing(val_obs_path) && error("val_frac=$val_frac but val_obs_path is missing.")
                isfile(val_obs_path) || error("Configured val_obs_path does not exist: $val_obs_path")
            end

            @info "Loading pinned val/test split membership from files"
            test_cmpd = load_split_smiles(test_obs_path)
            val_cmpd  = val_frac == 0 ? Set{String}() : load_split_smiles(val_obs_path)

            # Optional second split axis: hold out whole cell lines as well as compounds.
            val_cl, test_cl = if holdout_cell_lines
                isnothing(cellline_test_obs_path) &&
                    error("holdout_cell_lines is true but cellline_test_obs_path is missing.")
                isfile(cellline_test_obs_path) ||
                    error("Configured cellline_test_obs_path does not exist: $cellline_test_obs_path")
                _test_cl = load_split_cell_lines(cellline_test_obs_path)
                _val_cl  = if val_frac == 0
                    Set{Symbol}()
                else
                    isnothing(cellline_val_obs_path) && error(
                        "holdout_cell_lines is true and val_frac=$val_frac but " *
                        "cellline_val_obs_path is missing.")
                    isfile(cellline_val_obs_path) ||
                        error("Configured cellline_val_obs_path does not exist: $cellline_val_obs_path")
                    load_split_cell_lines(cellline_val_obs_path)
                end
                @info "Holding out cell lines — val: $(length(_val_cl)), test: $(length(_test_cl))"
                _val_cl, _test_cl
            else
                Set{Symbol}(), Set{Symbol}()
            end

            train_obs_loaded, val_obs_loaded, test_obs_loaded = apply_pinned_split(
                obs;
                val_smiles  = val_cmpd,
                test_smiles = test_cmpd,
                val_cl      = val_cl,
                test_cl     = test_cl,
            )

            if nrow(test_obs_loaded.meta_df) == 0 || (val_frac != 0 && nrow(val_obs_loaded.meta_df) == 0)
                error("Pinned split files produced an empty validation or test split for the " *
                      "current config.")
            end
            train_obs_loaded, val_obs_loaded, test_obs_loaded

        else
            _train_obs, _val_obs, _test_obs = split_obs(
                obs;
                val_frac     = val_frac,
                test_frac    = test_frac,
                split_seed   = split_seed,
            )
            # Save val and test splits
            if val_frac != 0
                JLD2.save(joinpath(run_dir, "val_obs.jld2"), "obs", _val_obs)
            end
            JLD2.save(joinpath(run_dir, "test_obs.jld2"), "obs", _test_obs)
            @info "Saved val_obs and test_obs to $run_dir"
            _train_obs, _val_obs, _test_obs
        end

    # Filter train set per delta inter-plate reproducibility
    if !isnothing(repro_delta_inter_path) && isfile(repro_delta_inter_path)
        repro_df      = dropmissing(
            CSV.read(repro_delta_inter_path, DataFrame), [:cell_line, :drug, :dose, :time])
        mean_repro    = combine(
            groupby(repro_df, [:cell_line, :drug, :dose, :time]),
            :pearson => mean => :mean_pearson,
        )
        above_thresh  = coalesce.(mean_repro.mean_pearson .>= repro_delta_inter_threshold, false)
        repro_keys    = Set(zip(
            string.(mean_repro[above_thresh, :cell_line]),
            string.(mean_repro[above_thresh, :drug]),
            string.(mean_repro[above_thresh, :dose]),
            string.(mean_repro[above_thresh, :time]),
        ))
        keep_idx      = findall(row -> (string(row.cell_line), string(row.drug),
                                        string(row.dose),      string(row.time)) in repro_keys,
                                eachrow(train_obs.meta_df))
        train_obs     = subset_obs(train_obs, keep_idx)
        @info "After repro filter (mean pearson >= $repro_delta_inter_threshold) — " *
            "train: $(nrow(train_obs.meta_df)) obs"
    end

    # Align the reference resampling pool to the (possibly repro-filtered) training
    # columns via meta_df._obs_id, then drop the helper column so downstream artifacts
    # (per-obs CSVs, saved splits) are byte-identical to a non-resample run.
    train_raw_pool = nothing
    if resample_ref
        train_raw_pool = ref_pool_full[Int.(train_obs.meta_df._obs_id)]
        for these_obs in (train_obs, val_obs, test_obs)
            :_obs_id in propertynames(these_obs.meta_df) && select!(these_obs.meta_df, Not(:_obs_id))
        end
    end

    n_train = nrow(train_obs.meta_df)
    n_val   = nrow(val_obs.meta_df)
    n_test  = nrow(test_obs.meta_df)

    # ── PCA ──────────────────────────────────────────────────────────────────
    # Each PCA is fit on the training split and applied to val/test. 

    # Per-epoch reference resampling pool, in the final delta_ref feature space
    # (post-PCA if applicable). Built below when resample_ref; nothing otherwise.
    train_ref_pool = nothing
    pca_parameters = Dict{String, NamedTuple}()
    # Stack a pool of replicate vectors into a (genes × k) matrix. The `init` keeps
    # reduce from short-circuiting to a bare Vector when k == 1 (single replicate).
    pool_to_matrix(p) = reduce(hcat, p; init = Matrix{Float32}(undef, length(first(p)), 0))

    # Rebuild with one PCA-transformed field, keeping every other field as-is.
    with_pca_delta_ref(o, delta_ref_exprs) = Obs(
        o.meta_df, delta_ref_exprs, o.molec_embeds, o.time_feats, o.dose_feats,
        o.avg_untrt_target_exprs, o.avg_delta_target_exprs,
    )
    with_pca_untrt_target(o, avg_untrt_target_exprs) = Obs(
        o.meta_df, o.delta_ref_exprs, o.molec_embeds, o.time_feats, o.dose_feats,
        avg_untrt_target_exprs, o.avg_delta_target_exprs,
    )

    if n_pca_expr !== nothing
        train_delta_ref_exprs, val_delta_ref_exprs, test_delta_ref_exprs =
            if train_obs.delta_ref_exprs === nothing
                nothing, nothing, nothing
            elseif resample_ref
                # Fit the delta_ref PCA and retain the model so the resampling pool can
                # be projected through the same frozen transform. Bypasses the transform
                # cache (the model is required and a cached transform would not carry it).
                pca_ref, tr_dr, va_dr, te_dr = PCAs.fit_and_transform_returning_model(
                    train_obs.delta_ref_exprs, val_obs.delta_ref_exprs,
                    test_obs.delta_ref_exprs, n_pca_expr, seed)
                pca_parameters["delta_ref_exprs"] = PCAs.extract_parameters(pca_ref)
                # Project each training obs's raw replicate pool; memoize by
                # (drug, dose, time) since the pool depends only on that key.
                memo           = Dict{Tuple, Matrix{Float32}}()
                train_ref_pool = Vector{Matrix{Float32}}(undef, length(train_raw_pool))
                for i in eachindex(train_raw_pool)
                    key = (train_obs.meta_df.drug[i], train_obs.meta_df.dose[i], train_obs.meta_df.time[i])
                    train_ref_pool[i] = get!(memo, key) do
                        PCAs.transform(pca_ref, pool_to_matrix(train_raw_pool[i]))
                    end
                end
                tr_dr, va_dr, te_dr
            else
                params, train_t, val_t, test_t = PCAs.load_or_fit_and_transform(
                    train_obs.delta_ref_exprs, val_obs.delta_ref_exprs, test_obs.delta_ref_exprs,
                    n_pca_expr,
                    PCAs.build_cache_file_path(
                        pca_dir, "delta_ref_exprs", n_pca_expr, jld2_path,
                        train_obs, val_obs, test_obs;
                        extra = (ref_cl, average_ref, landmark_genes_only, seed),
                    ),
                    seed;
                    return_parameters = true,
                )
                pca_parameters["delta_ref_exprs"] = params
                train_t, val_t, test_t
            end

        train_obs = with_pca_delta_ref(train_obs, train_delta_ref_exprs)
        val_obs   = with_pca_delta_ref(val_obs,   val_delta_ref_exprs)
        test_obs  = with_pca_delta_ref(test_obs,  test_delta_ref_exprs)
    end

    # No delta_ref PCA: the resampling pool stays in raw gene space, matching the
    # raw delta_ref block of X.
    if resample_ref && n_pca_expr === nothing
        train_ref_pool = [pool_to_matrix(p) for p in train_raw_pool]
    end

    if n_pca_expr !== nothing
        params, train_untrt_target_exprs, val_untrt_target_exprs, test_untrt_target_exprs =
            PCAs.load_or_fit_and_transform(
                train_obs.avg_untrt_target_exprs, val_obs.avg_untrt_target_exprs,
                test_obs.avg_untrt_target_exprs, n_pca_expr,
                PCAs.build_cache_file_path(
                    pca_dir, "avg_untrt_target_exprs", n_pca_expr, jld2_path,
                    train_obs, val_obs, test_obs;
                    extra = (ref_cl, average_ref, landmark_genes_only, seed),
                ),
                seed;
                return_parameters = true,
            )
        pca_parameters["avg_untrt_target_exprs"] = params

        train_obs = with_pca_untrt_target(train_obs, train_untrt_target_exprs)
        val_obs   = with_pca_untrt_target(val_obs,   val_untrt_target_exprs)
        test_obs  = with_pca_untrt_target(test_obs,  test_untrt_target_exprs)
    end

    if n_pca_molec !== nothing && train_obs.molec_embeds !== nothing
        train_molec_embeds, val_molec_embeds, test_molec_embeds =
        begin
            params, train_t, val_t, test_t = PCAs.load_or_fit_and_transform(
                train_obs.molec_embeds, val_obs.molec_embeds, test_obs.molec_embeds,
                n_pca_molec,
                PCAs.build_cache_file_path(
                    pca_dir, "molec_embeds", n_pca_molec, jld2_path,
                    train_obs, val_obs, test_obs;
                    extra = (molec_embed_file, seed),
                ),
                seed;
                return_parameters = true,
            )
            pca_parameters["molec_embeds"] = params
            train_t, val_t, test_t
        end

        with_pca_molec(o, molec_embeds) = Obs(
            o.meta_df, o.delta_ref_exprs, molec_embeds, o.time_feats, o.dose_feats,
            o.avg_untrt_target_exprs, o.avg_delta_target_exprs,
        )
        train_obs = with_pca_molec(train_obs, train_molec_embeds)
        val_obs   = with_pca_molec(val_obs,   val_molec_embeds)
        test_obs  = with_pca_molec(test_obs,  test_molec_embeds)
    end

    if !isempty(pca_parameters)
        pca_path = joinpath(run_dir, "pca_parameters.jld2")
        jldopen(pca_path, "w") do file
            for (name, params) in pca_parameters
                file["$name/mean"] = params.mean
                file["$name/components"] = params.components
            end
        end
        @info "Saved fitted PCA parameters to $pca_path"
    end

    # ── Model ──────────────────────────────────────────────────────────────────

    train_X    = concatenate_inputs(train_obs)
    input_dim  = size(train_X, 1)
    output_dim = size(train_obs.avg_delta_target_exprs, 1)
    @info "Input size: $input_dim - Output size: $output_dim"

    wandb.config.update(pydict(Dict(
        "molec_embed_model" => isnothing(molec_embed_model) ? "none" : molec_embed_model,
        "train_samples"     => n_train,
        "val_samples"       => n_val,
        "test_samples"      => n_test,
        "input_features"    => input_dim,
        "output_dim"        => output_dim,
        "model_type"        => model_type,
        "timestamp"         => timestamp,
    )))

    best_epoch = nothing

    if model_type == "mlp"
        mlp_gpu = create_mlp(
            input_dim,
            hidden_layers,
            output_dim;
            dropout_arr = dropout_arr,
        )

        # Wrap the MLP so the (post-PCA) embedding block is gated by log-dose.
        # concatenate_inputs order is delta_ref_exprs, molec_embeds, time_feats,
        # dose_feats, avg_untrt_target_exprs, so the embedding block and the dose row
        # are located from the leading block sizes.
        if dose_encoding == "gate"
            nrows(m) = isnothing(m) ? 0 : size(m, 1)
            embed_lo = nrows(train_obs.delta_ref_exprs) + 1
            embed_hi = embed_lo + nrows(train_obs.molec_embeds) - 1
            # First row of dose_feats holds log1p(dose µM).
            dose_row = embed_hi + nrows(train_obs.time_feats) + 1

            embed_hi >= embed_lo ||
                error("dose_encoding=\"gate\" requires a non-empty molecular embedding block.")
            # An off-by-one here would gate on a time column instead of dose, silently.
            dose_row <= input_dim && train_X[dose_row, :] == train_obs.dose_feats[1, :] ||
                error("Doser gate row $dose_row does not line up with the log-dose feature row.")

            mlp_gpu = DoserGate(mlp_gpu, embed_lo:embed_hi, dose_row)
            @info "Doser gate on embedding rows $embed_lo:$embed_hi, log-dose row $dose_row."
        end

        mlp_gpu = mlp_gpu |> gpu

        param_count = sum(length, Flux.trainables(mlp_gpu))
        @info "Model number of parameters: $param_count"

        wandb.config.update(pydict(Dict("model_param_count" => param_count)))

        total_steps  = max(1, round(Int, ceil(n_train / batch_size * n_epochs)))
        if isnothing(warmup_steps)
            warmup_steps = round(Int, ceil(n_train / batch_size) * 10)
        end
        warmup_steps = max(1, warmup_steps)

        warmup      = Triangle( λ0 = 1e-8, λ1 = lr,      period = 2 * warmup_steps)
        decay       = CosAnneal(λ0 = lr,   λ1 = lr / 10, period = total_steps)
        lr_schedule = Sequence([warmup, decay], [warmup_steps])

        opt_state = Flux.setup(Flux.AdamW(lr, (0.9, 0.999), weight_decay), mlp_gpu)
        checkpoint_path = n_val > 0 ? joinpath(run_dir, "best_model.jld2") : nothing

        @time train_mlp!(
            train_obs,
            val_obs,
            mlp_gpu,
            batch_size,
            n_epochs,
            opt_state;
            loss_name         = loss_name,
            wandb             = wandb,
            checkpoint_path   = checkpoint_path,
            checkpoint_metric = "val_avg_spearman",
            lr_schedule       = lr_schedule,
            ref_pool          = resample_ref ? train_ref_pool : nothing,
            resample_seed     = seed,
        )

        if checkpoint_path !== nothing && isfile(checkpoint_path)
            @info "Loading best validation-Spearman checkpoint for final test reporting"
            mlp_gpu = load_best_checkpoint(checkpoint_path)
            best_epoch = load(checkpoint_path, "epoch")
        end
        cleanup_extra_checkpoints(run_dir, checkpoint_path)

        @info "Running inference on Train/Val/Test"
        Ŷ_train = predict_mlp(train_obs, mlp_gpu, batch_size)
        Ŷ_val   = predict_mlp(val_obs,   mlp_gpu, batch_size)
        Ŷ_test  = predict_mlp(test_obs,  mlp_gpu, batch_size)

    else
        @info "Training classical ML model: $model_type"
        model   = train_classical(model_type, train_obs)
        Ŷ_train = predict_classical(model, model_type, train_obs)
        Ŷ_val   = predict_classical(model, model_type, val_obs)
        Ŷ_test  = predict_classical(model, model_type, test_obs)
    end

    # ── Post-training evaluation ──────────────────────────────────────────────────────────────────

    Y_train = collect(train_obs.avg_delta_target_exprs)
    Y_val   = collect(val_obs.avg_delta_target_exprs)
    Y_test  = collect(test_obs.avg_delta_target_exprs)

    train_eval = average_metrics(Y_train, Ŷ_train; digits = 6)
    test_eval  = average_metrics(Y_test,  Ŷ_test;  digits = 6)

    if val_frac != 0
        val_eval = average_metrics(Y_val, Ŷ_val; digits = 6)
        @info "Train Pearson: $(train_eval.avg_pearson), Val Pearson: $(val_eval.avg_pearson), " *
            "Test Pearson: $(test_eval.avg_pearson)"
    else
        val_eval = nothing
        @info "Train Pearson: $(train_eval.avg_pearson), Val Pearson: N/A (val_frac=0), " *
            "Test Pearson: $(test_eval.avg_pearson)"
    end

    wandb.log(pydict(Dict(
        "val_avg_pearson"              => isnothing(val_eval) ? NaN : val_eval.avg_pearson,
        "val_avg_spearman"             => isnothing(val_eval) ? NaN : val_eval.avg_spearman,
        "test_avg_pearson"             => test_eval.avg_pearson,
        "test_avg_spearman"            => test_eval.avg_spearman,
        "selected_train_avg_pearson"    => train_eval.avg_pearson,
        "selected_train_avg_spearman"   => train_eval.avg_spearman,
        "selected_val_avg_pearson"      => isnothing(val_eval) ? NaN : val_eval.avg_pearson,
        "selected_val_avg_spearman"     => isnothing(val_eval) ? NaN : val_eval.avg_spearman,
        "selected_test_avg_pearson"     => test_eval.avg_pearson,
        "selected_test_avg_spearman"    => test_eval.avg_spearman,
        "selected_test_avg_l2"          => test_eval.avg_l2,
        "selected_test_avg_cosine_sim"  => test_eval.avg_cosine_sim,
        "selected_epoch"                => isnothing(best_epoch) ? n_epochs : best_epoch,
    )))

    metrics_per_obs(train_obs.meta_df, Y_train, Ŷ_train, joinpath(run_dir, "train_perfs_per_obs.csv"))
    if val_frac != 0
        metrics_per_obs(val_obs.meta_df, Y_val, Ŷ_val, joinpath(run_dir, "val_perfs_per_obs.csv"))
    end
    metrics_per_obs(test_obs.meta_df,  Y_test,  Ŷ_test,  joinpath(run_dir, "test_perfs_per_obs.csv"))

    # ── Save predictions ──────────────────────────────────────────────────────────────────
    # Disabled by default to save disk space

    if save_predictions
        for (split_name, these_obs, Y, Ŷ) in [
            ("train", train_obs, Y_train, Ŷ_train),
            ("val",   val_obs,   Y_val,   Ŷ_val),
            ("test",  test_obs,  Y_test,  Ŷ_test),
        ]
            JLD2.save(
                joinpath(run_dir, "$(split_name)_predictions.jld2"),
                "Y",    Y,
                "Ŷ",    Ŷ,
                "meta", these_obs.meta_df,
            )
        end
    end

    # ── Run summary ──────────────────────────────────────────────────────────────────

    summary = Dict(
        "jld2_path"           => jld2_path,
        "seed"                => seed,
        "split_seed"          => split_seed,
        "ref_cl"              => string(ref_cl),
        "average_ref"         => average_ref,
        "resample_ref"        => resample_ref,
        # Recorded explicitly: these were previously absent from summary.toml, so
        # downstream readers had to assume defaults (and assumed them wrongly).
        "use_delta_ref"       => use_delta_ref,
        "use_molec_embed"     => molec_embed_file !== nothing && isfile(molec_embed_file),
        "dose_encoding"       => dose_encoding,
        # encode_time is currently unused: build_obs derives time_feats automatically
        # whenever smiles_to_embeds is given, with no separate toggle.
        "encode_time"         => encode_time,
        "landmark_genes_only" => landmark_genes_only,
        "val_obs_path"        => isnothing(val_obs_path)  ? "nothing" : val_obs_path,
        "test_obs_path"       => isnothing(test_obs_path) ? "nothing" : test_obs_path,
        "holdout_cell_lines"  => holdout_cell_lines,
        "cellline_val_obs_path"  => isnothing(cellline_val_obs_path)  ? "nothing" : cellline_val_obs_path,
        "cellline_test_obs_path" => isnothing(cellline_test_obs_path) ? "nothing" : cellline_test_obs_path,
        "val_frac"            => val_frac,
        "test_frac"           => test_frac,
        "n_train"             => n_train,
        "n_val"               => n_val,
        "n_test"              => n_test,
        "n_pca_expr"          => isnothing(n_pca_expr)  ? "nothing" : n_pca_expr,
        "n_pca_molec"         => isnothing(n_pca_molec) ? "nothing" : n_pca_molec,
        "pca_dir"             => isnothing(pca_dir)     ? "nothing" : pca_dir,
        "pca_cache_enabled"   => !isnothing(pca_dir),
        "selected_epoch"      => isnothing(best_epoch) ? n_epochs : best_epoch,
        "input_dim"           => input_dim,
        "hidden_layers"       => string(hidden_layers),
        "output_dim"          => output_dim,
        "dropout_arr"         => isnothing(dropout_arr) ? "nothing" : string(dropout_arr),
        "batch_size"          => batch_size,
        "n_epochs"            => n_epochs,
        "lr"                  => lr,
        "weight_decay"        => weight_decay,
        "loss_name"           => loss_name,
        "save_predictions"    => save_predictions,
        "train_pearson"       => train_eval.avg_pearson,
        "val_pearson"         => isnothing(val_eval) ? "N/A" : val_eval.avg_pearson,
        "test_pearson"        => test_eval.avg_pearson,
        "train_spearman"      => train_eval.avg_spearman,
        "val_spearman"        => isnothing(val_eval) ? "N/A" : val_eval.avg_spearman,
        "test_spearman"       => test_eval.avg_spearman,
        "train_l2"            => train_eval.avg_l2,
        "val_l2"              => isnothing(val_eval) ? "N/A" : val_eval.avg_l2,
        "test_l2"             => test_eval.avg_l2,
        "train_cosine_sim"    => train_eval.avg_cosine_sim,
        "val_cosine_sim"      => isnothing(val_eval) ? "N/A" : val_eval.avg_cosine_sim,
        "test_cosine_sim"     => test_eval.avg_cosine_sim,
    )
    open(joinpath(run_dir, "summary.toml"), "w") do io
        TOML.print(io, summary)
    end
    @info "Saved summary to $run_dir"

    cache_script = joinpath("scripts", "predict_drug_response", "build_run_cache.py")
    run(`$(PythonCall.python_executable_path()) $cache_script $run_dir $outdir`)

    wandb.finish()
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "config_file"
            help     = "Path to the TOML configuration file"
            arg_type = String
        "outdir"
            help     = "BIOPERT_OUTDIR: base directory for all pipeline data (see configs/default_paths.toml)."
            arg_type = String
        "dataset"
            help     = "Dataset name (\"lincs\" or \"tahoe\"); resolves jld2_path and pca_dir from configs/default_paths.toml."
            arg_type = String
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args   = parse_args(parser)
    main(args["config_file"], args["outdir"], args["dataset"])
end