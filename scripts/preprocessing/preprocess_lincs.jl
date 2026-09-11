using ArgParse, DataFrames, JLD2
using Biopert


function main(lincs_dir::String, outdir::String)
    paths = Biopert.dataset_paths(outdir, "lincs")
    mkpath(dirname(paths.jld2_path))

    # Extract LINCS L1000 level 3 expression profiles (landmark genes only)
    lm_file = joinpath(lincs_dir, "lincs_beta_landmark_genes.jld2")
    lm_data = Biopert.Lincs(joinpath(lincs_dir, ""), "level3_beta_all_n3026460x12328.gctx", lm_file)

    # Add original index before any filtering, so it can still be used to
    # match rows of lm_data.inst back to columns of lm_data.expr after filtering.
    lm_data.inst.orig_idx = 1:nrow(lm_data.inst)

    data = filter(row ->
        (row.pert_type === :ctl_vehicle && row.pert_id === :DMSO) ||
        row.pert_type === :trt_cp,
        lm_data.inst
    )

    data = filter(row -> row.qc_pass === Symbol("1"), data)

    # (det_plate, pert_itime, cell_iname) combinations for which a DMSO profile exists
    dmso_triplets = unique(
        data[data.pert_id .=== :DMSO, [:det_plate, :pert_itime, :cell_iname]]
    )

    # Keep only rows (DMSO and trt_cp) whose (det_plate, pert_itime, cell_iname) has a DMSO control
    data = innerjoin(data, dmso_triplets, on = [:det_plate, :pert_itime, :cell_iname])

    data_with_smiles = leftjoin(data, lm_data.compound[:, [:pert_id, :canonical_smiles]], on = :pert_id)

    data_with_smiles = filter(row ->
        row.pert_id === :DMSO ||
        let smi = coalesce(row.canonical_smiles, "")
            smi != "" && smi != "restricted"
        end,
        data_with_smiles
    )

    # DMSO controls have no meaningful dose; give them a placeholder so they
    # aren't dropped by the missing/empty-metadata filter below.
    data_with_smiles.pert_idose[data_with_smiles.pert_id .=== :DMSO] .= Symbol("NA")

    required_cols = [:cell_iname, :pert_id, :pert_idose, :pert_itime, :det_plate, :sample_id]
    dropmissing!(data_with_smiles, required_cols)
    data_with_smiles = filter(row -> all(col -> row[col] != Symbol(""), required_cols), data_with_smiles)

    df = DataFrame(
        cell_line = Symbol.(data_with_smiles.cell_iname),
        sample    = Symbol.(data_with_smiles.sample_id),
        plate     = Symbol.(data_with_smiles.det_plate),
        drug      = Symbol.(data_with_smiles.pert_id),
        smiles    = coalesce.(data_with_smiles.canonical_smiles, ""),
        dose      = Symbol.(data_with_smiles.pert_idose),
        time      = Symbol.(data_with_smiles.pert_itime),
        expr      = [lm_data.expr[:, i] for i in data_with_smiles.orig_idx]
    )

    @save paths.jld2_path df
    @info "$(paths.jld2_path) saved"
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "lincs_dir"
            help     = "Path to LINCS_beta directory."
            arg_type = String
        "outdir"
            help     = "BIOPERT_OUTDIR: base directory for all pipeline data (see configs/default_paths.toml)."
            arg_type = String
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args = parse_args(parser)
    main(args["lincs_dir"], args["outdir"])
end