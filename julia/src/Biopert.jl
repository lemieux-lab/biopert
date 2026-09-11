module Biopert

include("Paths.jl")
using .Paths
export dataset_paths

include("LincsUtils.jl")
using .LincsUtils
export Lincs

include("TahoeUtils.jl")
using .TahoeUtils
export FromParquet

include("TahoePseudobulk.jl")
using .TahoePseudobulk
export build_sample_to_dose, build_pseudobulks, log_normalize!

include("DoseUtils.jl")
using .DoseUtils
export build_dose_feats

include("Splits.jl")
using .Splits
export split_smiles, split_cell_lines, load_split_smiles, load_split_cell_lines

include("Observations.jl")
using .Observations
export Obs
export obs_signature
export average_expr_df, build_delta_df, build_time_one_hot
export concatenate_inputs, build_obs
export subset_obs, split_obs, apply_pinned_split

include("PCAs.jl")
using .PCAs

include("Metrics.jl")
using .Metrics
export pearson_corr, spearman_corr, cosine_similarity, l2_dist, rmse
export average_metrics, metrics_per_obs

include("Models.jl")
using .Models
export DoserGate
export create_mlp, train_mlp!, load_best_checkpoint, predict_mlp
export train_classical, predict_classical

include("Repro.jl")
using .Repro
export pairwise_stats, filter_by_repro

include("SAR.jl")
using .SAR
export build_sar_table

end
