module Models

export DoserGate
export create_mlp, train_mlp!, load_best_checkpoint, predict_mlp
export train_classical, predict_classical

using CUDA, cuDNN, DataFrames, GLMNet, Flux, JLD2
using ProgressMeter, Random, Statistics, XGBoost
using ..Metrics, ..Observations


# ── Doser gate ──────────────────────────────────────────────────────────────────

"""
    DoserGate(net, embed_rows, dose_row)

CPA-style multiplicative dose gate wrapping an inner network. Scales rows
`embed_rows` (the molecular embedding block) by

    s = σ(u * β + b) − σ(b)

where `u = log1p(dose µM)` is read from row `dose_row`. Since `u ≥ 0`, `s` is
non-negative, so the gate attenuates the embedding toward zero at zero dose
rather than flipping its sign (`− σ(b)` pins `s = 0` there). That's why the
`"gate"` encoding emits `log1p` while other encodings emit `log10`.

Unlike CPA, `β`/`b` are shared across compounds rather than learned per-compound,
since per-compound parameters can't generalize to the held-out compounds of a
butina split. They sit after the molecular PCA, inside the model, and are
learned jointly with the network by backprop.

CPA reference: Lotfollahi, M., Klimovskaia Susmelj, A., De Donno, C., et al. (2023).
Predicting Cellular Responses to Complex Perturbations in High-Throughput Screens.
Molecular Systems Biology, 19(6), e11517. https://doi.org/10.15252/msb.202211517
"""
# β/b are type-parameterized, not `Vector{Float32}`: `Flux.@layer` moves them to
# GPU as `CuArray`, which a concrete `Vector{Float32}` field couldn't hold.
struct DoserGate{N, A}
    net        :: N
    embed_rows :: UnitRange{Int}
    dose_row   :: Int
    β          :: A
    b          :: A
end

# β = 1 (CPA's init): spans the usable range of s without saturating (LINCS:
# 0.000/0.167/0.417/0.495 at 1e-4/1/10/200 µM; Tahoe: 0.012/0.100/0.357).
# Smaller β only compresses this further.
#
# Caveat: log1p barely separates sub-µM doses, so the gate is nearly flat there.
# That's inherent to CPA's transform, not this layer; β is learnable and can
# sharpen the response.
DoserGate(net, embed_rows::UnitRange{Int}, dose_row::Int) =
    DoserGate(net, embed_rows, dose_row, Float32[1.0], Float32[0.0])

Flux.@layer DoserGate trainable=(net, β, b)

function (m::DoserGate)(x)
    u = x[m.dose_row:m.dose_row, :]  # log1p(dose µM), ≥ 0
    s = σ.(u .* m.β .+ m.b) .- σ.(m.b)

    lo, hi = m.embed_rows.start, m.embed_rows.stop
    return m.net(vcat(x[1:lo-1, :], x[m.embed_rows, :] .* s, x[hi+1:end, :]))
end


# ── Losses ──────────────────────────────────────────────────────────────────

mse_loss(Y, Ŷ)  = Flux.mse(Ŷ, Y)
rmse_loss(Y, Ŷ) = sqrt(mean((Ŷ .- Y) .^ 2))
mape_loss(Y, Ŷ) = mean(abs.((Ŷ .- Y) ./ (Y .+ 1f-8))) * 100

const LOSS_DICT = Dict{String, Function}(
    "mse"  => mse_loss,
    "rmse" => rmse_loss,
    "mape" => mape_loss,
)

get_loss(loss_name) =
    get(LOSS_DICT, String(loss_name)) do
        error("Unknown loss_name=$loss_name. Choose one of: $(collect(keys(LOSS_DICT))).")
    end


# ── Training performance metrics ──────────────────────────────────────────────────────────────────

# Whether higher or lower is better for each metric that can be used to select
# the checkpointed model.
const CHECKPOINT_METRIC_DIRECTIONS = Dict(
    "val_avg_pearson"    => :max,
    "val_avg_spearman"   => :max,
    "val_avg_cosine_sim" => :max,
    "val_avg_l2"         => :min,
)

is_better(val_score, best_val_score, direction::Symbol) =
    direction == :max ? val_score > best_val_score : val_score < best_val_score


# ── Multilayer perceptron ──────────────────────────────────────────────────────────────────

function create_mlp(
    input_dim     :: Int,
    hidden_layers :: Vector{Int},
    output_dim    :: Int;
    activation_f  = relu,
    dropout_arr   :: Union{Nothing, Vector{Float64}} = nothing,
)
    @assert !isempty(hidden_layers) "Hidden_layers must be non-empty."
    in_dims      = [input_dim; hidden_layers[1:end-1]]
    dense_layers = [Dense(i => j, activation_f) for (i, j) in zip(in_dims, hidden_layers)]

    if isnothing(dropout_arr)
        net_layers = dense_layers
    else
        @assert length(dropout_arr) == length(dense_layers) "Dropout_arr must match number of " *
            "hidden layers."
        drop_layers = [Dropout(d) for d in dropout_arr]
        net_layers  = [layer for pair in zip(dense_layers, drop_layers) for layer in pair]
    end

    return Chain(net_layers..., Dense(hidden_layers[end] => output_dim, identity))
end


function predict_mlp(obs::Obs, mlp_gpu, batch_size::Int)
    testmode!(mlp_gpu)
    output_dim = size(obs.avg_delta_target_exprs, 1)
    n          = nrow(obs.meta_df)
    Ŷ          = Matrix{Float32}(undef, output_dim, n)
    X          = concatenate_inputs(obs)
    col        = 1
    for batch_idxs in Iterators.partition(1:n, batch_size)
        x   = gpu(X[:, batch_idxs])
        ŷb  = cpu(mlp_gpu(x))
        bsz = length(batch_idxs)
        Ŷ[:, col:col+bsz-1] .= ŷb
        col += bsz
    end
    return Ŷ
end


function train_mlp!(
    train_obs       :: Obs,
    val_obs         :: Obs,
    mlp_gpu,
    batch_size      :: Int,
    n_epochs        :: Int,
    opt_state;
    ref_pool          :: Union{Nothing, Vector{Matrix{Float32}}} = nothing,
    resample_seed     :: Int = 42,
    lr_schedule       = nothing,
    loss_name         = "rmse",
    eval_every        = 10,
    checkpoint_metric :: String = "val_avg_spearman",
    checkpoint_path   :: Union{Nothing, String} = nothing,
    wandb             = nothing,
)
    loss      = get_loss(loss_name)
    n_train   = nrow(train_obs.meta_df)
    prog      = Progress(n_epochs; desc = "Training", showspeed = true)

    direction = get(CHECKPOINT_METRIC_DIRECTIONS, checkpoint_metric) do
        error("Unknown checkpoint_metric=$checkpoint_metric")
    end
    best_val_score = direction == :max ? -Inf : Inf
    step           = 0  # global batch step, 0-indexed to match ParameterSchedulers

    # Pre-transfer training data to GPU if it fits in VRAM,
    # otherwise fall back to per-batch CPU→GPU transfer.
    X_train = concatenate_inputs(train_obs)
    X_train_gpu, Y_train_gpu = try
        gpu(X_train), gpu(train_obs.avg_delta_target_exprs)
    catch e
        @warn "Could not pre-load training data to GPU ($e). Falling back to per-batch transfer."
        nothing, nothing
    end

    X_train_gpu !== nothing && @info "Training data pre-loaded to GPU."

    # Each epoch, redraw ref_pool[i]'s column for training obs i (training-time
    # augmentation). Val/test are never resampled.
    resample = ref_pool !== nothing
    d_ref    = 0
    ref_rng  = MersenneTwister(resample_seed)
    if resample
        train_obs.delta_ref_exprs === nothing &&
            error("ref_pool given but train_obs has no delta_ref_exprs.")
        length(ref_pool) == n_train ||
            error("ref_pool length $(length(ref_pool)) != n_train $n_train.")
        d_ref = size(train_obs.delta_ref_exprs, 1)
        all(p -> size(p, 1) == d_ref, ref_pool) ||
            error("ref_pool feature dim mismatch with delta_ref block ($d_ref).")
        @info "Per-epoch reference resampling enabled " *
            "($(n_train) train obs, $d_ref delta_ref features)."
    end

    for epoch in 1:n_epochs
        trainmode!(mlp_gpu)

        if resample
            @inbounds for i in 1:n_train
                P = ref_pool[i]
                X_train[1:d_ref, i] .= @view P[:, rand(ref_rng, 1:size(P, 2))]
            end
            X_train_gpu !== nothing && (X_train_gpu[1:d_ref, :] .= gpu(X_train[1:d_ref, :]))
        end

        for batch_idxs in Iterators.partition(shuffle(1:n_train), batch_size)
            if lr_schedule !== nothing
                Flux.adjust!(opt_state, lr_schedule(step))
            end
            step += 1

            if X_train_gpu !== nothing
                x = X_train_gpu[:, batch_idxs]
                y = Y_train_gpu[:, batch_idxs]
            else
                x = gpu(X_train[:, batch_idxs])
                y = gpu(train_obs.avg_delta_target_exprs[:, batch_idxs])
            end
            grads = Flux.gradient(mlp_gpu) do m
                loss(y, m(x))
            end
            Flux.update!(opt_state, mlp_gpu, grads[1])
        end

        # LR for logging (value used at the last batch of this epoch)
        current_lr = lr_schedule !== nothing ? lr_schedule(step - 1) : NaN

        log_data = Dict(
            "epoch"         => epoch,
            "learning_rate" => current_lr,
        )

        # Evaluate every eval_every epochs
        if epoch % eval_every == 0
            Ŷ_train    = predict_mlp(train_obs, mlp_gpu, batch_size)
            train_loss = loss(train_obs.avg_delta_target_exprs, Ŷ_train)
            merge!(log_data, Dict("train_loss" => train_loss))

            if nrow(val_obs.meta_df) > 0
                Ŷ_val        = predict_mlp(val_obs, mlp_gpu, batch_size)
                eval_metrics = average_metrics(val_obs.avg_delta_target_exprs, Ŷ_val; digits=6)
                merge!(log_data, Dict(
                    "val_avg_rmse"       => eval_metrics.avg_rmse,
                    "val_avg_pearson"    => eval_metrics.avg_pearson,
                    "val_avg_spearman"   => eval_metrics.avg_spearman,
                    "val_avg_l2"         => eval_metrics.avg_l2,
                    "val_avg_cosine_sim" => eval_metrics.avg_cosine_sim,
                ))

                # Checkpoint when the selected validation metric improves
                val_score = get(log_data, checkpoint_metric) do
                    error("Unknown checkpoint_metric=$checkpoint_metric")
                end
                if checkpoint_path !== nothing && is_better(val_score, best_val_score, direction)
                    best_val_score = val_score
                    mlp_cpu = cpu(mlp_gpu)
                    jldsave(checkpoint_path; model = mlp_cpu, epoch = epoch,
                            checkpoint_metric = checkpoint_metric,
                            checkpoint_score  = val_score,
                            val_avg_rmse      = eval_metrics.avg_rmse,
                            val_avg_pearson   = eval_metrics.avg_pearson,
                            val_avg_spearman  = eval_metrics.avg_spearman,
                            val_avg_l2        = eval_metrics.avg_l2,
                            val_avg_cosine_sim = eval_metrics.avg_cosine_sim,
                    )
                    @info "Checkpoint saved at epoch $epoch " *
                        "($checkpoint_metric=$(round(val_score; digits=6)))"
                end
            end
        end

        # Return fragmented GPU memory to the pool after each eval block to prevent
        # monotonic per-epoch slowdown from CUDA allocator fragmentation.
        if epoch % eval_every == 0
            GC.gc(false)
            CUDA.reclaim()
        end

        wandb !== nothing && wandb.log(log_data)
        next!(prog)
    end

    return nothing
end


function load_best_checkpoint(path::String)
    data = JLD2.load(path)
    return gpu(data["model"])
end


# ── Other classical ML models ──────────────────────────────────────────────────────────────────

function train_classical(model_type::String, train_obs::Obs; config = Dict())
    # Classical models expect (n_obs × n_features) and (n_obs × n_genes)
    X_train_T = concatenate_inputs(train_obs)'
    Y_train_T = train_obs.avg_delta_target_exprs'

    if model_type == "lasso" || model_type == "ridge"
        # Lasso: alpha=1.0, Ridge: alpha=0.0
        # GLMNet uses 'alpha' for ElasticNet mixing, 'lambda' for penalty strength.
        glm_alpha = (model_type == "lasso") ? 1.0 : 0.0
        return glmnet(X_train_T, Y_train_T, family="mgaussian", alpha=glm_alpha)

    elseif model_type == "xgboost"
        num_round = get(config, "xgb_rounds", 50)
        params = [
            "eta"       => get(config, "xgb_eta", 0.1),
            "max_depth" => get(config, "xgb_max_depth", 6),
            "objective" => "reg:squarederror",
        ]
        # XGBoost natively supports single target — train one booster per gene.
        n_targets = size(Y_train_T, 2)
        models    = Vector{Any}(undef, n_targets)
        @info "Training XGBoost models for $n_targets targets..."
        @showprogress for i in 1:n_targets
            models[i] = xgboost(X_train_T, num_round; label=Y_train_T[:, i], param=params)
        end
        return models
    else
        error("Unknown model_type=$model_type (expected \"lasso\", \"ridge\", or \"xgboost\").")
    end
end


function predict_classical(model, model_type::String, obs::Obs)
    # Classical models expect (n_obs × n_features); predictions returned as (n_genes × n_obs)
    X_T = concatenate_inputs(obs)'

    if model_type == "lasso" || model_type == "ridge"
        # GLMNet predict returns (n_obs, n_genes, n_lambdas) — select last lambda (smallest penalty)
        preds = GLMNet.predict(model, X_T)
        return Matrix{Float32}(preds[:, :, end]')

    elseif model_type == "xgboost"
        n_obs     = size(X_T, 1)
        n_targets = length(model)
        Ŷ_T = zeros(Float32, n_obs, n_targets)
        for (i, booster) in enumerate(model)
            Ŷ_T[:, i] .= XGBoost.predict(booster, X_T)
        end
        return Matrix{Float32}(Ŷ_T')  # (n_genes × n_obs)
    else
        error("Unknown model_type=$model_type (expected \"lasso\", \"ridge\", or \"xgboost\").")
    end
end


end