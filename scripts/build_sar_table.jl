using ArgParse, CSV, DataFrames, JLD2
using Biopert


function main(jld2_path::String, outdir::String;
            embeddings_dir::Union{String, Nothing}=nothing,
            max_pairs::Union{Int, Nothing}=nothing,
            repro_criteria::Symbol=:pearson,
            repro_threshold::Float64=0.7)
    input_name = replace(basename(jld2_path), r"\.jld2$" => "")
    mkpath(outdir)

    df = load(jld2_path, "df")

    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)
    delta_df = build_delta_df(untrt_df, trt_df)

    delta_high_repro_df = filter_by_repro(delta_df; plate_mode=:inter, max_pairs=max_pairs,
                                          criteria=repro_criteria, threshold=repro_threshold)

    results = build_sar_table(delta_high_repro_df; embeddings_dir=embeddings_dir)
    sar_path = joinpath(outdir, "sar_$(input_name).csv")
    CSV.write(sar_path, results, delim=';')  # use ";" delimiter because some SMILES contain ","
    @info "SAR table saved" path=sar_path
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "jld2_path"
            help     = "Path to a preprocessed .jld2 file (must contain a DataFrame under key \"df\")"
            arg_type = String
        "outdir"
            help     = "Path to output directory; sar.csv is written here"
            arg_type = String
        "--embeddings_dir"
            help     = "Directory with one subdirectory per molecular representation, each " *
                       "containing a dataframe.parquet (default: skip chemical-embedding " *
                       "distance columns)"
            arg_type = String
        "--max_pairs"
            help     = "Max pairs to sample per condition when computing reproducibility " *
                       "(default: keep all pairs)"
            arg_type = Int
        "--repro_criteria"
            help     = "Reproducibility criteria: \"pearson\" or \"spearman\""
            arg_type = String
            default  = "pearson"
        "--repro_threshold"
            help     = "Minimum delta-profile reproducibility for a condition to be kept"
            arg_type = Float64
            default  = 0.7
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args   = parse_args(parser)

    main(args["jld2_path"], args["outdir"];
         embeddings_dir=args["embeddings_dir"],
         max_pairs=args["max_pairs"], repro_criteria=Symbol(args["repro_criteria"]),
         repro_threshold=args["repro_threshold"])
end
