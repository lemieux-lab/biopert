using ArgParse, CSV, DataFrames, JLD2, Parquet2, Statistics
using Biopert


function main(tahoe_dir::String, outdir::String; cell_thresh::Int=50, umi_thresh::Int=35_000, α::Int=10000)
    mkpath(outdir)

    sample_meta = DataFrame(Parquet2.Dataset(tahoe_dir * "/metadata/sample_metadata.parquet"))
    gene_exp_meta = DataFrame(Parquet2.Dataset(tahoe_dir * "/metadata/gene_metadata.parquet"))

    # Get coding gene tokens
    # Downloaded from https://www.genenames.org/download/statistics-and-files/
    df_coding            = CSV.read("data/protein-coding_gene.txt", DataFrame; delim='\t')
    coding_genes         = Set(df_coding.symbol)
    coding_gene_exp_meta = filter(row -> row.gene_symbol in coding_genes, gene_exp_meta)
    coding_tokens        = coding_gene_exp_meta.token_id        
    coding_tokens        = collect(skipmissing(coding_tokens))
    CSV.write(joinpath(outdir, "tahoe_coding_tokens.csv"), DataFrame(coding_tokens = coding_tokens))

    # Create dict sample to dose 
    sample_to_dose = Biopert.build_sample_to_dose(sample_meta)

    # One pseudobulk per (:cell_line, :sample)
    raw_data = tahoe_dir * "/data"
    df = Biopert.build_pseudobulks(raw_data, coding_tokens, sample_to_dose)

    # Rename DMSO_TF -> DMSO and add time column for consistency with LINCS dataset
    df.drug = replace(df.drug, Symbol("DMSO_TF") => :DMSO)
    df.time = fill(Symbol("24 h"), nrow(df))

    # Add missing SMILES
    rows_with_missing_smiles = filter(row -> row.drug != Symbol("DMSO") && row.smiles == "", df)
    drugs_with_missing_smiles = unique(rows_with_missing_smiles.drug)
    println("Drugs with missing SMILES: $drugs_with_missing_smiles")
    # Two drugs have a missing SMILES: Sacubitril/Valsartan and Verteporfin

    # Sacubitril / Valsartan (PubChem CID: 24755620)
    sacubitril_smiles = "CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)[C@@H](C(C)C)C(=O)O.CCOC(=O)[C@H](C)C[C@@H](CC1=CC=C(C=C1)C2=CC=CC=C2)NC(=O)CCC(=O)O"
    mask = df.drug .== Symbol("Sacubitril/Valsartan")
    df[mask, :smiles] .= sacubitril_smiles

    # Verteporfin (PubChem CID: 168430535)
    verteporfin_smiles = "CC1=C(C2=CC3=C(C(=C(N3)C=C4[C@]5([C@H](C(=CC=C5C(=CC6=C(C(=C(N6)C=C1N2)C=C)C)N4)C(=O)OC)C(=O)OC)C)C)CCC(=O)OC)CCC(=O)O.CC1=C(C2=CC3=C(C(=C(N3)C=C4[C@]5([C@H](C(=CC=C5C(=CC6=C(C(=C(N6)C=C1N2)C=C)C)N4)C(=O)O)C(=O)OC)C)C)CCC(=O)O)CCC(=O)OC"
    mask = df.drug .== :Verteporfin
    df[mask, :smiles] .= verteporfin_smiles
    
    @assert nrow(df) == nrow(unique(df, Not(:expr))) "Sanity check failed: df contains duplicate rows"

    @info "Tahoe pseudobulks before filtering: $(nrow(df)) total"
    outfile = joinpath(outdir, "pseudobulks.jld2")
    @save outfile df
    @info "$outfile saved"

    # ── Filter ────────────────────────────────────────────────────────────────
    df.qc_pass = (df.total_cells .>= cell_thresh) .& (df.total_umis .>= umi_thresh)

    counts = combine(groupby(df, :cell_line),
        :qc_pass => (x -> 100 * (1 - mean(x))) => :pct_fail)

    high_fail_cls = Set(counts.cell_line[counts.pct_fail .> 90.0])
    if !isempty(high_fail_cls)
        @info "Cell lines removed (>90% QC failure):" collect(high_fail_cls)
    end

    filter!(row -> row.qc_pass && row.cell_line ∉ high_fail_cls, df)

    @info "Tahoe pseudobulks after filtering: $(nrow(df)) total"

    # ── Log-normalize ──────────────────────────────────────────────────────────
    Biopert.log_normalize!(df; α=α)

    @info "Tahoe pseudobulks after log-normalization: $(nrow(df)) total"
    outfile = joinpath(outdir, "pseudobulks_alpha_$(α).jld2")
    @save outfile df
    @info "$outfile saved"
end

function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "tahoe_dir"
            help = "Path to Tahoe-100M directory"
            arg_type = String
        "outdir"
            help = "Path to output directory for tahoe_coding_tokens.csv and Tahoe pseudobulks"
            arg_type = String
        "--cell_thresh"
            help = "Minimum number of cells per pseudobulk"
            arg_type = Int
            default = 50
        "--umi_thresh"
            help = "Minimum number of UMIs per pseudobulk"
            arg_type = Int
            default = 35_000
        "--alpha"
            help = "Alpha value for log-normalization"
            arg_type = Int
            default = 10000
    end
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    parser = build_argument_parser()
    args = parse_args(parser)
    main(args["tahoe_dir"], args["outdir"];
        cell_thresh = args["cell_thresh"],
        umi_thresh  = args["umi_thresh"],
        α           = args["alpha"])
end