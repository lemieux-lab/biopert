using ArgParse, CSV, DataFrames, JLD2
using Biopert


function main(outdir::String, dataset::String;
              max_pairs::Union{Int, Nothing}=nothing)
    paths      = Biopert.dataset_paths(outdir, dataset)
    input_name = replace(basename(paths.jld2_path), r"\.jld2$" => "")
    repro_dir  = joinpath(paths.repro_dir, "repro_$(input_name)")
    mkpath(repro_dir)

    df = load(paths.jld2_path, "df")

    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)
    delta_df = build_delta_df(untrt_df, trt_df)

    @info "Assessing intra-plate reproducibility..."
    ps_untrt_intra = pairwise_stats(untrt_df; plate_mode=:intra, max_pairs=max_pairs)
    ps_trt_intra   = pairwise_stats(trt_df;   plate_mode=:intra, max_pairs=max_pairs)
    ps_delta_intra = pairwise_stats(delta_df; plate_mode=:intra, max_pairs=max_pairs)

    @info "Assessing inter-plate reproducibility..."
    ps_untrt_inter = pairwise_stats(untrt_df; plate_mode=:inter, max_pairs=max_pairs)
    ps_trt_inter   = pairwise_stats(trt_df;   plate_mode=:inter, max_pairs=max_pairs)
    ps_delta_inter = pairwise_stats(delta_df; plate_mode=:inter, max_pairs=max_pairs)

    max_pairs_suffix = max_pairs === nothing ? "" : "_max_pairs_$(max_pairs)"
    for (name, repro_df) in [
        ("untrt_intra", ps_untrt_intra), ("untrt_inter", ps_untrt_inter),
        ("trt_intra",   ps_trt_intra),   ("trt_inter",   ps_trt_inter),
        ("delta_intra", ps_delta_intra), ("delta_inter", ps_delta_inter),
    ]
        outfile = "repro_$(name)$(max_pairs_suffix).csv"
        CSV.write(joinpath(repro_dir, outfile), repro_df)
        @info "Saved $(outfile)"
    end

    @info "Pairwise reproducibility statistics computed and saved to $(repro_dir)"
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "outdir"
            help     = "BIOPERT_OUTDIR: base directory for all pipeline data (see configs/default_paths.toml)."
            arg_type = String
        "dataset"
            help     = "Dataset to assess: \"lincs\" or \"tahoe\"."
            arg_type = String
            range_tester = x -> x in ("lincs", "tahoe")
        "--max_pairs"
            help     = "Cap on replicate pairs assessed per condition (default: no cap)."
            arg_type = Int
            default  = nothing
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args   = parse_args(parser)

    main(args["outdir"], args["dataset"]; max_pairs=args["max_pairs"])
end