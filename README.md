# BioPert

## Python environment

Python tooling lives in a conda env defined by
[`environment.yml`](environment.yml). Recreate it with:

```bash
conda env create -f environment.yml
conda activate biopert
```

## Downloading LINCS Beta Level 3

Download the LINCS2020 Level 3 beta dataset with:

```bash
./scripts/download_lincs.sh <outdir>
```

`<outdir>` is the directory the data will be downloaded into; the script
creates an `LINCS_beta` subdirectory there and fetches:

- `level3_beta_all_n3026460x12328.gctx` — Level 3 expression data
- `cellinfo_beta.txt`, `compoundinfo_beta.txt`, `geneinfo_beta.txt`, `instinfo_beta.txt` — metadata
- `README.txt`, `LINCS2020 Release Metadata Field Definitions.xlsx` — documentation

## Downloading Tahoe-100M

Downloading the [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
dataset requires a Hugging Face access token:

1. Create a read-access token at https://huggingface.co/settings/tokens.
2. Export it in your shell before running the script:

   ```bash
   export HF_TOKEN=<your-hugging-face-token>
   ```

Then download the dataset with:

```bash
./scripts/download_tahoe.sh <outdir>
```

`<outdir>` is the directory the data will be downloaded into; the script
creates a `Tahoe-100M` subdirectory there and mirrors the full dataset repo
into it.

Make sure the [Python environment](#python-environment) is active first,
since the script needs the `hf` CLI.

## Tahoe-100M build dependencies

Reading Tahoe-100M parquet files from Julia efficiently requires building
`libcxxwrap-julia`, Apache Arrow (C++), and the `ArrowWrap` CxxWrap module.
Set these up with:

```bash
conda activate biopert
./scripts/setup_tahoe_deps.sh
```

The script requires the `biopert` conda env to be active (it exits with an
error otherwise), and assumes `cmake` and `wget` are already available on
your `PATH`.

## Julia package

Julia code lives under [`julia/`](julia/) as the `Biopert` package. It bundles
Julia submodules; instantiate its dependencies with:

```bash
julia --project=julia -e 'import Pkg; Pkg.instantiate()'
```

## Python calls from Julia (PythonCall)

Some Julia code (e.g. [`julia/src/Splits.jl`](julia/src/Splits.jl)) calls into
Python via `PythonCall.jl`, which needs a Python built with a *shared*
libpython (`.so`), not a static `libpython.a`. Cluster-provided Python
modules sometimes ship only a static build, which `PythonCall` cannot load.

To point `PythonCall` at the project's own [`biopert` conda env](#python-environment)
(which does ship a shared libpython), source the setup script before starting
Julia:

```bash
source scripts/setup_python_env.sh
julia --project=julia
```

The script must be *sourced*, not executed, so the environment variables it
sets (`JULIA_CONDAPKG_BACKEND`, `JULIA_PYTHONCALL_EXE`, `JULIA_PYTHONCALL_LIB`)
reach your shell. It creates the `biopert` conda env from `environment.yml`
if it doesn't exist yet.

## LINCS Preprocessing

Once the raw data is downloaded, extract and filter the landmark-gene expression
profiles with:

```bash
julia --project=julia scripts/preprocess_lincs.jl <lincs_dir> <outdir>
```

Make sure the [Python environment is configured for PythonCall](#python-calls-from-julia-pythoncall)
first (`source scripts/setup_python_env.sh`), since every `Biopert` script
triggers Python module loading at startup.

- `<lincs_dir>` is the `LINCS_beta` directory produced by `download_lincs.sh`
  (i.e. `<outdir>/LINCS_beta` from the download step).
- `<outdir>` is where the preprocessed output is written.

The result is written to `<outdir>/filtered_lincs.jld2` as a `DataFrame` with
columns `cell_line`, `sample`, `plate`, `drug`, `smiles`, `dose`, `time`, and
`expr` (the landmark-gene expression vector for that sample).

## Tahoe-100M Preprocessing

Once the raw data is downloaded and the [Tahoe-100M build dependencies](#tahoe-100m-build-dependencies)
are set up, build pseudobulk expression profiles with:

```bash
julia --project=julia scripts/preprocess_tahoe.jl <tahoe_dir> <outdir> [--cell_thresh N] [--umi_thresh N] [--alpha N]
```

Make sure the [Python environment is configured for PythonCall](#python-calls-from-julia-pythoncall)
first (`source scripts/setup_python_env.sh`), since every `Biopert` script
triggers Python module loading at startup.

- `<tahoe_dir>` is the `Tahoe-100M` directory produced by `download_tahoe.sh`
  (i.e. `<outdir>/Tahoe-100M` from the download step).
- `<outdir>` is where the preprocessed output is written.
- `--cell_thresh` (default `50`) and `--umi_thresh` (default `35000`) set the
  minimum cell/UMI counts a pseudobulk must have to pass QC.
- `--alpha` (default `10000`) is the scale factor used for log-normalization.

The script:

1. Loads sample and gene metadata, and restricts to protein-coding genes
   (from `data/protein-coding_gene.txt`), writing the resulting gene tokens
   to `<outdir>/tahoe_coding_tokens.csv`.
2. Builds one pseudobulk per (cell line, sample) by summing single-cell
   expression profiles, saving the raw result to `<outdir>/pseudobulks.jld2`.
3. Filters out pseudobulks failing the cell/UMI thresholds, and cell lines
   with a QC failure rate above 90%.
4. Log-normalizes the filtered pseudobulks and writes the final result to
   `<outdir>/pseudobulks_alpha_<alpha>.jld2`.

## Reproducibility statistics

Once a preprocessed dataset is available (`filtered_lincs.jld2` from
[LINCS Preprocessing](#lincs-preprocessing), or `pseudobulks_alpha_<alpha>.jld2`
from [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), compute pairwise
reproducibility statistics (Pearson, Spearman) between condition-matched
profiles with:

```bash
julia --project=julia scripts/compute_repro_stats.jl <jld2_path> <outdir> [--max_pairs N]
```

Make sure the [Python environment is configured for PythonCall](#python-calls-from-julia-pythoncall)
first (`source scripts/setup_python_env.sh`), since every `Biopert` script
triggers Python module loading at startup.

- `<jld2_path>` is the preprocessed `.jld2` file to assess.
- `<outdir>` is the base output directory (see below for where files actually land).
- `--max_pairs` caps the number of profile pairs assessed per condition
  (`cell_line`, `drug`, `dose`, `time`); by default there is no cap. Profiles
  sharing a condition are condition-matched, not necessarily true replicates —
  in LINCS, for example, they can come from different projects. For LINCS,
  we suggest setting `--max_pairs 200` to keep runtime tractable, given how
  many condition-matched pairs a condition can have.

The script assesses untreated, treated, and delta profiles, each at both
intra-plate and inter-plate `plate_mode`, writing six files to
`<outdir>/repro_stats_<input_basename>/` (where `<input_basename>` is the name
of `<jld2_path>` without its `.jld2` extension, e.g. `filtered_lincs` or
`pseudobulks_alpha_10000`): `repro_untrt_intra.csv`, `repro_untrt_inter.csv`,
`repro_trt_intra.csv`, `repro_trt_inter.csv`, `repro_delta_intra.csv`, and
`repro_delta_inter.csv`.

## SAR table

Once a preprocessed dataset is available (see [LINCS Preprocessing](#lincs-preprocessing)
or [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), build a
structure-activity relationship (SAR) table — one row per compound pair,
comparing chemical distance against biological (delta profile) similarity —
with:

```bash
julia --project=julia scripts/build_sar_table.jl <jld2_path> <outdir> [--embeddings_dir DIR] [--max_pairs N] [--repro_criteria pearson|spearman] [--repro_threshold N]
```

Make sure the [Python environment is configured for PythonCall](#python-calls-from-julia-pythoncall)
first (`source scripts/setup_python_env.sh`), since every `Biopert` script
triggers Python module loading at startup.

- `<jld2_path>` is the preprocessed `.jld2` file to assess.
- `<outdir>` is the output directory; the table is written to
  `<outdir>/sar_<input_basename>.csv` (where `<input_basename>` is the name
  of `<jld2_path>` without its `.jld2` extension).
- `--embeddings_dir`, if given, should contain one subdirectory per molecular
  representation, each with a `dataframe.parquet`; one additional
  chemical-distance column is added per representation. By default, no
  embedding-based distance columns are added.
- `--max_pairs` caps the number of profile pairs assessed per condition when
  computing reproducibility; by default there is no cap. **For LINCS, we
  suggest setting `--max_pairs 200`** to keep runtime tractable, given how
  many condition-matched pairs a condition can have.
- `--repro_criteria` (default `pearson`) and `--repro_threshold` (default
  `0.7`) control which delta profiles are kept: only conditions (`cell_line`,
  `drug`, `dose`, `time`) whose minimum inter-plate `repro_criteria` is at
  least `repro_threshold` are included in the SAR table.
