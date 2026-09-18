using ArgParse, CSV, DataFrames, JLD2, TOML
using Biopert


function configured_path(config::Dict, key::String)
    path = get(config, key, nothing)
    path === nothing && return nothing
    path_s = strip(String(path))
    isempty(path_s) && return nothing
    lowercase(path_s) in ("none", "nothing", "false") && return nothing
    return path_s
end


to_sym_or_nothing(x) =
    x === nothing                  ? nothing :
    x isa Vector{Symbol}           ? x :
    x isa Vector{<:AbstractString} ? Symbol.(x) :
    x isa AbstractString           ? [Symbol(x)] :
    error("Expected Vector, String, or nothing, got $(typeof(x))")


function create_split_obs(jld2_path::String, config_file::String; cellline_only::Bool = false)
    df     = load(jld2_path, "df")
    config = TOML.parsefile(config_file)

    seed                 = get(config, "seed", 42)
    split_seed           = get(config, "split_seed", seed)
    ref_cl               = Symbol(config["ref_cl"])
    average_ref          = get(config, "average_ref", false)
    use_delta_ref        = get(config, "use_delta_ref", true)
    landmark_genes_only  = get(config, "landmark_genes_only", false)
    chosen_times         = to_sym_or_nothing(get(config, "chosen_times", nothing))
    chosen_doses         = to_sym_or_nothing(get(config, "chosen_doses", nothing))
    val_frac             = Float64(get(config, "val_frac", 0.1))
    test_frac            = Float64(get(config, "test_frac", 0.1))
    val_obs_path         = configured_path(config, "val_obs_path")
    test_obs_path        = configured_path(config, "test_obs_path")
    # Cell-line holdout axis
    holdout_cell_lines     = get(config, "holdout_cell_lines", false)
    cellline_val_frac      = Float64(get(config, "cellline_val_frac",  val_frac))
    cellline_test_frac     = Float64(get(config, "cellline_test_frac", test_frac))
    cellline_min_obs       = Int(get(config, "cellline_min_obs", 0))
    # An empty TOML array parses as Vector{Any}, so build the Vector{Symbol} explicitly.
    cellline_exclude       = Symbol[Symbol(x) for x in get(config, "cellline_exclude", [])]
    cellline_val_obs_path  = configured_path(config, "cellline_val_obs_path")
    cellline_test_obs_path = configured_path(config, "cellline_test_obs_path")

    isnothing(test_obs_path) && error("test_obs_path must be set in $config_file")
    val_frac != 0 && isnothing(val_obs_path) && error("val_obs_path must be set in $config_file")

    if holdout_cell_lines
        isnothing(cellline_test_obs_path) &&
            error("holdout_cell_lines is true but cellline_test_obs_path is not set in $config_file")
        cellline_val_frac != 0 && isnothing(cellline_val_obs_path) &&
            error("holdout_cell_lines is true but cellline_val_obs_path is not set in $config_file")
    end

    if !isnothing(chosen_times)
        filter!(row -> row.time in chosen_times, df)
    end
    if !isnothing(chosen_doses)
        filter!(row -> row.drug === :DMSO || row.dose in chosen_doses, df)
    end
    # Optionally restrict to landmark genes
    if landmark_genes_only
        df_tahoe_coding = CSV.read(
            joinpath(dirname(jld2_path), "tahoe_coding_tokens.csv"), DataFrame)
        df_shared       = CSV.read("data/lincs_and_tahoe_shared_genes.csv", DataFrame)
        shared_tokens   = Set(df_shared.token_id)
        mask            = [token in shared_tokens for token in df_tahoe_coding.coding_tokens]
        df.expr         = [expr[mask] for expr in df.expr]
    end

    untrt_df = filter(row -> row.drug == :DMSO, df)
    trt_df   = filter(row -> row.drug != :DMSO, df)

    obs = build_obs(
        ref_cl, untrt_df, trt_df;
        use_delta_ref     = use_delta_ref,
        average_delta_ref = average_ref,
        seed              = seed,
        return_ref_pool   = false,
    )

    if cellline_only
        println("--cellline_only: leaving the pinned compound split files untouched")
        println("  val:  $(val_frac == 0 ? "none" : val_obs_path)")
        println("  test: $test_obs_path")
    else
        _, val_obs, test_obs = split_obs(
            obs;
            val_frac   = val_frac,
            test_frac  = test_frac,
            split_seed = split_seed,
        )

        mkpath(dirname(test_obs_path))
        JLD2.save(test_obs_path, "obs", test_obs)
        if val_frac != 0
            mkpath(dirname(val_obs_path))
            JLD2.save(val_obs_path, "obs", val_obs)
        end

        println("Saved split obs for $config_file")
        println("  val:  $(val_frac == 0 ? "none" : val_obs_path)")
        println("  test: $test_obs_path")
    end

    if holdout_cell_lines
        train_cl, val_cl, test_cl = split_cell_lines(
            Symbol.(obs.meta_df.cell_line);
            val_frac  = cellline_val_frac,
            test_frac = cellline_test_frac,
            seed      = split_seed,
            min_obs   = cellline_min_obs,
            exclude   = cellline_exclude,
        )

        mkpath(dirname(cellline_test_obs_path))
        JLD2.save(cellline_test_obs_path, "cell_lines", sort(collect(test_cl)))
        if cellline_val_frac != 0
            mkpath(dirname(cellline_val_obs_path))
            JLD2.save(cellline_val_obs_path, "cell_lines", sort(collect(val_cl)))
        end

        println("Saved cell-line split for $config_file")
        println("  train ($(length(train_cl))): $(join(sort(collect(train_cl)), ", "))")
        println("  val   ($(length(val_cl))): $(join(sort(collect(val_cl)), ", "))")
        println("  test  ($(length(test_cl))): $(join(sort(collect(test_cl)), ", "))")
    end
end


function build_argument_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "jld2_path"
            help     = "Path to the final-data JLD2 file"
            arg_type = String
        "config_file"
            help     = "Config containing split settings and output paths"
            arg_type = String
        "--cellline_only"
            help     = "Write only the cell-line split files; leave the pinned compound split files untouched"
            action   = :store_true
    end
    return s
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(build_argument_parser())
    create_split_obs(args["jld2_path"], args["config_file"]; cellline_only = args["cellline_only"])
end