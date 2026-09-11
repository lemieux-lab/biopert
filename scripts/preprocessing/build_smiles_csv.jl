using ArgParse, CSV, DataFrames, JLD2
using Biopert


function main(outdir::String, dataset::String)
    paths = Biopert.dataset_paths(outdir, dataset)
    mkpath(dirname(paths.smiles_csv))

    df = load(paths.jld2_path, "df")

    compounds = filter(row -> row.drug != :DMSO, df)[:, [:drug, :smiles]]
    compounds.drug = string.(compounds.drug)
    compounds.smiles = strip.(string.(compounds.smiles))
    unique!(compounds)

    CSV.write(paths.smiles_csv, compounds)
    @info "$(paths.smiles_csv) saved" n_compounds = nrow(compounds)
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "outdir"
            help     = "BIOPERT_OUTDIR: base directory for all pipeline data (see configs/default_paths.toml)."
            arg_type = String
        "dataset"
            help     = "Dataset to extract compounds from: \"lincs\" or \"tahoe\"."
            arg_type = String
            range_tester = x -> x in ("lincs", "tahoe")
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args   = parse_args(parser)
    main(args["outdir"], args["dataset"])
end
