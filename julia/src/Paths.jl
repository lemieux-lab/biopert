module Paths

using TOML

export dataset_paths

const BIOPERT_ROOT = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
const CONFIG_FILE = normpath(joinpath(BIOPERT_ROOT, "configs", "default_paths.toml"))

"""
    dataset_paths(outdir::String, dataset::String) -> NamedTuple

Resolve the default paths for `dataset` ("lincs" or "tahoe") under `outdir`
(BIOPERT_OUTDIR), as configured in `configs/default_paths.toml`.
"""
function dataset_paths(outdir::String, dataset::String)
    config = TOML.parsefile(CONFIG_FILE)
    haskey(config, dataset) || error("Unknown dataset \"$dataset\"; expected one of: $(join(keys(config), ", "))")
    entry = config[dataset]
    return (
        jld2_path        = joinpath(outdir, entry["jld2_path"]),
        smiles_csv       = joinpath(outdir, entry["smiles_csv"]),
        molec_embeds_dir = joinpath(outdir, entry["molec_embeds_dir"]),
        repro_dir        = joinpath(outdir, entry["repro_dir"]),
        sar_dir          = joinpath(outdir, entry["sar_dir"]),
        pca_dir          = joinpath(outdir, entry["pca_dir"]),
        predictions_dir  = joinpath(outdir, entry["predictions_dir"]),
    )
end

end
