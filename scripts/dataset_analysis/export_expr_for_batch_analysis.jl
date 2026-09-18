"""
Export an expression matrix + design metadata to NPY/CSV, so the batch-effect
variance decomposition (variancePartition, see scripts/analyze_batch_variance.R)
can run without re-loading the multi-GB JLD2 every time. Handles LINCS and Tahoe,
absolute and delta profiles.

Unlike `repro_*.csv`, which stores one row per *pair* of replicate wells, this
exports one row per *profile*. Pairwise reproducibility cancels any offset shared
by a plate, so it is blind to exactly the directional batch shifts we want to
measure; the profile matrix is not.

The two datasets have opposite plate designs, which changes what is identifiable:

  * LINCS — `plate` is ~100% nested in `cell_line` (99.96% of plates hold one
    cell line), so the two are only separable via cell lines assayed on several
    plates. `--min_plates_per_cl` drops the rest and `--max_plates_per_cl`
    balances what remains; downstream, plate must be fitted as `cell_line:plate`.
  * Tahoe — fully crossed: all 48 cell lines appear on all 14 plates. Nothing
    needs dropping and plate can be fitted as an ordinary crossed term.

Subsampling (`--max_per_plate`) keeps the mixed models tractable. Sampling is seeded.

Usage:
    julia --project=julia scripts/dataset_analysis/export_expr_for_batch_analysis.jl <jld2_path> <output_dir> \\
        --prefix lincs|tahoe [--min_plates_per_cl INT] [--max_plates_per_cl INT] \\
        [--max_per_plate INT] [--shared_genes] [--seed INT] [--drop_dmso]

Outputs (in output_dir):
    <prefix>_expr.npy        — Float32 (n_genes, n_samples), column order == meta rows
    <prefix>_expr_meta.csv   — cell_line, plate, drug, dose, time, sample per column
"""

using ArgParse, CSV, DataFrames, JLD2, NPZ, Random


function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "jld2_path"
            arg_type = String
        "output_dir"
            arg_type = String
        "--prefix"
            arg_type = String
            required = true
            help     = "Output file prefix, e.g. `lincs` or `tahoe`."
        "--min_plates_per_cl"
            arg_type = Int
            default  = 0
            help     = "Drop cell lines seen on fewer plates than this (LINCS: 4; Tahoe: 0, already crossed)."
        "--max_plates_per_cl"
            arg_type = Int
            default  = 0
            help     = "Sample this many plates per cell line (0 = keep all)."
        "--max_per_plate"
            arg_type = Int
            default  = 0
            help     = "Sample this many profiles per plate (0 = keep all)."
        "--shared_genes"
            action   = :store_true
            help     = "Subset to the LINCS/Tahoe shared gene list (Tahoe: 19020 -> 965 genes)."
        "--seed"
            arg_type = Int
            default  = 0
            help     = "RNG seed for plate/profile subsampling."
        "--drop_dmso"
            action   = :store_true
            help     = "Exclude vehicle-control wells (kept by default; delta files have none)."
    end
    args = parse_args(s)
    mkpath(args["output_dir"])
    rng = MersenneTwister(args["seed"])

    @info "Loading $(args["jld2_path"]) ..."
    # Raw-profile files store the frame under "df"; delta files under "delta".
    df = jldopen(args["jld2_path"], "r") do f
        haskey(f, "df") ? read(f, "df") : read(f, "delta")
    end
    @info "Loaded $(nrow(df)) profiles."

    if args["drop_dmso"]
        df = filter(row -> row.drug != :DMSO, df)
        @info "Dropped vehicle controls -> $(nrow(df)) profiles."
    end

    # ── Stratified subsample ──────────────────────────────────────────────────
    if args["min_plates_per_cl"] > 1
        pc = combine(groupby(df, :cell_line), :plate => (x -> length(unique(x))) => :np)
        keep_cl = Set(pc.cell_line[pc.np .>= args["min_plates_per_cl"]])
        df = filter(row -> row.cell_line in keep_cl, df)
        @info "Kept $(length(keep_cl)) cell lines with >= $(args["min_plates_per_cl"]) plates -> $(nrow(df)) profiles."
    end
    if args["max_plates_per_cl"] > 0
        keep = Set{Symbol}()
        for g in groupby(df, :cell_line)
            pl = unique(g.plate)
            n = min(length(pl), args["max_plates_per_cl"])
            union!(keep, pl[randperm(rng, length(pl))[1:n]])
        end
        df = filter(row -> row.plate in keep, df)
        @info "Capped at $(args["max_plates_per_cl"]) plates/cell line -> $(nrow(df)) profiles."
    end
    if args["max_per_plate"] > 0
        parts = DataFrame[]
        for g in groupby(df, :plate)
            n = min(nrow(g), args["max_per_plate"])
            push!(parts, n == nrow(g) ? DataFrame(g) : DataFrame(g)[randperm(rng, nrow(g))[1:n], :])
        end
        df = vcat(parts...)
        @info "Capped at $(args["max_per_plate"])/plate -> $(nrow(df)) profiles."
    end

    # ── Optional shared-gene subset, so Tahoe is comparable to LINCS ──────────
    gene_mask = trues(length(df.expr[1]))
    if args["shared_genes"]
        tokens_path = joinpath(dirname(args["jld2_path"]), "tahoe_coding_tokens.csv")
        tokens = CSV.read(tokens_path, DataFrame)
        shared = Set(CSV.read("data/lincs_and_tahoe_shared_genes.csv", DataFrame).token_id)
        gene_mask = [t in shared for t in tokens.coding_tokens]
        length(gene_mask) == length(df.expr[1]) ||
            error("gene mask ($(length(gene_mask))) != expr length ($(length(df.expr[1])))")
        @info "Shared-gene subset: $(sum(gene_mask)) / $(length(gene_mask)) genes."
    end

    # ── Write matrix (genes x samples) and the matching metadata rows ──────────
    n_genes, n_samples = sum(gene_mask), nrow(df)
    @info "Building $(n_genes) x $(n_samples) matrix..."
    X = Matrix{Float32}(undef, n_genes, n_samples)
    for (j, v) in enumerate(df.expr)
        X[:, j] = v[gene_mask]
    end

    npzwrite(joinpath(args["output_dir"], "$(args["prefix"])_expr.npy"), X)
    meta_df = DataFrame(
        sample    = string.(df.sample),
        cell_line = string.(df.cell_line),
        plate     = string.(df.plate),
        drug      = string.(df.drug),
        dose      = string.(df.dose),
        time      = string.(df.time),
    )
    CSV.write(joinpath(args["output_dir"], "$(args["prefix"])_expr_meta.csv"), meta_df)
    @info "Wrote $(n_genes)x$(n_samples) matrix and metadata to $(args["output_dir"])"
end

main()