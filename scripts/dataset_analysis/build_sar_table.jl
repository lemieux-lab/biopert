using ArgParse, CSV, DataFrames, JLD2
using Biopert


function main(outdir::String, dataset::String;
            max_pairs::Union{Int, Nothing}=nothing,
            repro_criteria::Symbol=:pearson,
            repro_threshold::Float64=0.7)
    paths      = Biopert.dataset_paths(outdir, dataset)
    input_name = replace(basename(paths.jld2_path), r"\.jld2$" => "")
    mkpath(paths.sar_dir)

    df = load(paths.jld2_path, "df")

    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)
    delta_df = build_delta_df(untrt_df, trt_df)

    delta_high_repro_df = filter_by_repro(delta_df; plate_mode=:inter, max_pairs=max_pairs,
                                          criteria=repro_criteria, threshold=repro_threshold)

    embeddings_dir = isdir(paths.molec_embeds_dir) ? paths.molec_embeds_dir : nothing
    results = build_sar_table(delta_high_repro_df; embeddings_dir=embeddings_dir)
    max_pairs_suffix = max_pairs === nothing ? "" : "_max_pairs_$(max_pairs)"
    sar_path = joinpath(
        paths.sar_dir,
        "sar_$(input_name)_$(repro_criteria)_$(repro_threshold)$(max_pairs_suffix).csv",
    )
    CSV.write(sar_path, results, delim=';')  # use ";" delimiter because some SMILES contain ","
    @info "SAR table saved" path=sar_path
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "outdir"
            help     = "BIOPERT_OUTDIR: base directory for all pipeline data (see configs/default_paths.toml)."
            arg_type = String
        "dataset"
            help     = "Dataset to build the SAR table for: \"lincs\" or \"tahoe\"."
            arg_type = String
            range_tester = x -> x in ("lincs", "tahoe")
        "--max_pairs"
            help     = "Max pairs to sample per condition when computing reproducibility " *
                       "(default: keep all pairs)."
            arg_type = Int
        "--repro_criteria"
            help     = "Reproducibility criteria: \"pearson\" or \"spearman\"."
            arg_type = String
            default  = "pearson"
        "--repro_threshold"
            help     = "Minimum delta-profile reproducibility for a condition to be kept."
            arg_type = Float64
            default  = 0.7
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args   = parse_args(parser)

    main(args["outdir"], args["dataset"];
         max_pairs=args["max_pairs"], repro_criteria=Symbol(args["repro_criteria"]),
         repro_threshold=args["repro_threshold"])
end
