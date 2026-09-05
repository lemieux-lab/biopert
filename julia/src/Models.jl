module Models

export create_mlp, train_mlp!, load_best_checkpoint, predict_mlp

using CUDA, DataFrames, Flux, JLD2
using ProgressMeter, Random, Statistics
using ..Metrics, ..Observations


# Whether higher or lower is better for each metric that can be used to select
# the checkpointed model
const CHECKPOINT_METRIC_DIRECTIONS = Dict(
    "val_avg_pearson"    => :max,
    "val_avg_spearman"   => :max,
    "val_avg_cosine_sim" => :max,
    "val_avg_l2"         => :min,
)

is_better(val_score, best_val_score, direction::Symbol) =
    direction == :max ? val_score > best_val_score : val_score < best_val_score


batch_rmse_loss(Y, Ŷ) = sqrt(mean((Ŷ .- Y) .^ 2))


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
        @assert length(dropout_arr) == length(dense_layers) "Dropout_arr must match number of hidden layers."
        drop_layers = [Dropout(d) for d in dropout_arr]
        net_layers  = [layer for pair in zip(dense_layers, drop_layers) for layer in pair]
    end

    return Chain(net_layers..., Dense(hidden_layers[end] => output_dim, identity))
end


function train_mlp!(train_obs::Obs, val_obs::Obs, mlp_gpu, opt_state,
                    n_epochs::Int, batch_size::Int;
                    resample_ref      :: Bool = false,
                    resample_seed     :: Int = 42,
                    lr_schedule       = nothing,
                    eval_every        = 10,
                    checkpoint_path   :: Union{Nothing, String} = nothing,
                    checkpoint_metric :: String = "val_avg_spearman",
                    wandb             = nothing,
)
    checkpoint_direction = get(CHECKPOINT_METRIC_DIRECTIONS, checkpoint_metric) do
        error("Unknown checkpoint_metric=$checkpoint_metric. Must be one of " *
              "$(join(keys(CHECKPOINT_METRIC_DIRECTIONS), ", "))")
    end

    n_train = nrow(train_obs.meta_df)
    prog    = Progress(n_epochs; desc = "Training", showspeed = true)

    best_val_score = checkpoint_direction == :max ? -Inf : Inf
    step           = 0  # global batch step, 0-indexed to match ParameterSchedulers

    # Pre-transfer training data to GPU if it fits in VRAM,
    # otherwise fall back to per-batch CPU→GPU transfer
    X_train = concatenate_inputs(train_obs)
    Y_train = train_obs.avg_delta_targets
    X_train_gpu, Y_train_gpu = try
        gpu(X_train), gpu(Y_train)
    catch e
        @warn "Could not pre-load training data to GPU ($e). Falling back to per-batch transfer."
        nothing, nothing
    end
    !isnothing(X_train_gpu) && @info "Training data pre-loaded to GPU."

    delta_ref_n_genes = 0
    ref_rng           = MersenneTwister(resample_seed)
    ref_pool          = train_obs.delta_ref_pools
    if resample_ref
        isnothing(ref_pool) && error("resample_ref=true but train_obs has no delta_ref_pools.")
        delta_ref_n_genes = size(ref_pool[1], 1)
        @info "Per-epoch reference cell line delta profile resampling enabled."
    end

    for epoch in 1:n_epochs
        trainmode!(mlp_gpu)

        if resample_ref
            # Each epoch, redraw one reference cell line delta profile per training observation
            # from its `delta_ref_pools` candidates. Val/test are never resampled.
            @inbounds for i in 1:n_train
                P = ref_pool[i]
                X_train[1:delta_ref_n_genes, i] .= @view P[:, rand(ref_rng, 1:size(P, 2))]
            end
            !isnothing(X_train_gpu) && (X_train_gpu[1:delta_ref_n_genes, :] .= gpu(X_train[1:delta_ref_n_genes, :]))
        end

        for batch_idxs in Iterators.partition(shuffle(1:n_train), batch_size)
            if !isnothing(lr_schedule)
                Flux.adjust!(opt_state, lr_schedule(step))
            end
            step += 1

            if !isnothing(X_train_gpu)
                x = X_train_gpu[:, batch_idxs]
                y = Y_train_gpu[:, batch_idxs]
            else
                x = gpu(X_train[:, batch_idxs])
                y = gpu(train_obs.avg_delta_targets[:, batch_idxs])
            end
            grads = Flux.gradient(mlp_gpu) do m
                batch_rmse_loss(y, m(x))
            end
            Flux.update!(opt_state, mlp_gpu, grads[1])
        end

        # LR for logging (value used at the last batch of this epoch)
        current_lr = !isnothing(lr_schedule) ? lr_schedule(step - 1) : NaN

        log_data = Dict(
            "epoch"         => epoch,
            "learning_rate" => current_lr,
        )

        # Evaluate every eval_every epochs
        if epoch % eval_every == 0
            Y_train    = train_obs.avg_delta_targets
            Ŷ_train    = predict_mlp(train_obs, mlp_gpu, batch_size)
            train_loss = batch_rmse_loss(Y_train, Ŷ_train)
            merge!(log_data, Dict("train_loss" => train_loss))

            if nrow(val_obs.meta_df) > 0
                Y_val        = val_obs.avg_delta_targets
                Ŷ_val        = predict_mlp(val_obs, mlp_gpu, batch_size)
                eval_metrics = average_metrics(Y_val, Ŷ_val; digits=6)
                merge!(log_data, Dict(
                    "val_avg_rmse"       => eval_metrics.avg_rmse,
                    "val_avg_pearson"    => eval_metrics.avg_pearson,
                    "val_avg_spearman"   => eval_metrics.avg_spearman,
                    "val_avg_l2"         => eval_metrics.avg_l2,
                    "val_avg_cosine_sim" => eval_metrics.avg_cosine_sim,
                ))

                # Checkpoint when the selected validation metric improves
                val_score = log_data[checkpoint_metric]

                if !isnothing(checkpoint_path) && is_better(val_score, best_val_score, checkpoint_direction)
                    best_val_score = val_score
                    mlp_cpu = cpu(mlp_gpu)
                    jldsave(checkpoint_path; model = mlp_cpu, epoch = epoch,
                            checkpoint_metric  = checkpoint_metric,
                            checkpoint_score   = val_score,
                            val_avg_rmse       = eval_metrics.avg_rmse,
                            val_avg_pearson    = eval_metrics.avg_pearson,
                            val_avg_spearman   = eval_metrics.avg_spearman,
                            val_avg_l2         = eval_metrics.avg_l2,
                            val_avg_cosine_sim = eval_metrics.avg_cosine_sim)
                    @info "Checkpoint saved at epoch $epoch ($checkpoint_metric=$(round(val_score; digits=6)))"
                end
            end
        end

        # Return fragmented GPU memory to the pool after each eval block to prevent
        # monotonic per-epoch slowdown from CUDA allocator fragmentation
        if epoch % eval_every == 0
            GC.gc(false)
            CUDA.reclaim()
        end

        !isnothing(wandb) && wandb.log(log_data)
        next!(prog)
    end

    return nothing
end


function load_best_checkpoint(path::String)
    data = JLD2.load(path)
    return gpu(data["model"])
end


function predict_mlp(obs::Obs, mlp_gpu, batch_size::Int)
    testmode!(mlp_gpu)
    output_dim = size(obs.avg_delta_targets, 1)
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


end