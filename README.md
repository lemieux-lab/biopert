# BioPert

## Python environment

Python tooling lives in a conda env defined by
[`environment.yml`](environment.yml). Recreate it with:

```bash
conda env create -f environment.yml
conda activate biopert
which python  # should print .../envs/biopert/bin/python
```

**Troubleshooting:** if `which python` does not point into `envs/biopert/`, something
else on your `$PATH` (another conda env, `pyenv` shims, etc.) is taking priority over
the environment `conda activate` just switched to. `conda activate` only removes PATH
entries it added itself in the current shell, so a foreign tool that got onto `$PATH`
some other way can keep shadowing it. Check `echo $PATH` for entries ahead of
`envs/biopert/bin` and fix your shell startup files accordingly.

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
source scripts/setup/setup_python_env.sh
julia --project=julia
```

The script must be *sourced*, not executed, so the environment variables it
sets (`JULIA_CONDAPKG_BACKEND`, `JULIA_PYTHONCALL_EXE`, `JULIA_PYTHONCALL_LIB`)
reach your shell. It creates the `biopert` conda env from `environment.yml`
if it doesn't exist yet.

## Output directory (BIOPERT_OUTDIR)

Most scripts past the download step take a single `<outdir>` — referred to
below as `BIOPERT_OUTDIR` — plus a `<dataset>` argument (`lincs` or `tahoe`),
and derive every other path (preprocessed `.jld2` file, SMILES CSV, molecule
embeddings, reproducibility stats, SAR table, predictions) automatically. The
mapping from dataset name to relative path is defined in
[`configs/default_paths.toml`](configs/default_paths.toml):

```toml
[lincs]
jld2_path        = "preprocessed_data/filtered_lincs.jld2"
smiles_csv       = "smiles/smiles_filtered_lincs.csv"
molec_embeds_dir = "molec_embeds/lincs"
repro_dir        = "repro"
sar_dir          = "sar"
predictions_dir  = "target_delta_profiles_predictions"
```

(and similarly under `[tahoe]`). We suggest exporting `BIOPERT_OUTDIR` once
and reusing it for every command below:

```bash
export BIOPERT_OUTDIR=/path/to/your/data
```

Raw downloads (`LINCS_beta/`, `Tahoe-100M/`) are the exception: they're large,
one-time downloads, so the download and preprocessing scripts still take an
explicit path to them — you may want to keep those somewhere shared, separate
from any one `BIOPERT_OUTDIR`.

## Downloading LINCS Beta Level 3

Download the LINCS2020 Level 3 beta dataset with:

```bash
./scripts/setup/download_lincs.sh <outdir>
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

Then, make sure the [Python environment](#python-environment) is active first
(the script needs the `hf` CLI), and download the dataset with:

```bash
./scripts/setup/download_tahoe.sh <outdir>
```

`<outdir>` is the directory the data will be downloaded into; the script
creates a `Tahoe-100M` subdirectory there and mirrors the full dataset repo
into it.

## Tahoe-100M build dependencies

Reading Tahoe-100M parquet files from Julia efficiently requires building
`libcxxwrap-julia`, Apache Arrow (C++), and the `ArrowWrap` CxxWrap module.
Make sure the `biopert` conda env is active first (the script exits with an
error otherwise) and that `cmake` and `wget` are already available on your
`PATH`, then set these up with:

```bash
conda activate biopert
./scripts/setup/setup_tahoe_deps.sh
```

## LINCS Preprocessing

Once the raw data is downloaded, make sure the [Python environment is
configured for PythonCall](#python-calls-from-julia-pythoncall) first
(`source scripts/setup/setup_python_env.sh`), then extract and filter the
landmark-gene expression profiles with:

```bash
julia --project=julia scripts/preprocessing/preprocess_lincs.jl <lincs_dir> $BIOPERT_OUTDIR
```

- `<lincs_dir>` is the `LINCS_beta` directory produced by `download_lincs.sh`.

The result is written to `$BIOPERT_OUTDIR/preprocessed_data/filtered_lincs.jld2`
as a `DataFrame` with columns `cell_line`, `sample`, `plate`, `drug`, `smiles`,
`dose`, `time`, and `expr` (the landmark-gene expression vector for that sample).

## Tahoe-100M Preprocessing

Once the raw data is downloaded and the [Tahoe-100M build dependencies](#tahoe-100m-build-dependencies)
are set up, make sure the [Python environment is configured for
PythonCall](#python-calls-from-julia-pythoncall) first
(`source scripts/setup/setup_python_env.sh`), then build pseudobulk expression
profiles with:

```bash
julia --project=julia scripts/preprocessing/preprocess_tahoe.jl <tahoe_dir> $BIOPERT_OUTDIR [--cell_thresh N] [--umi_thresh N] [--alpha N]
```

- `<tahoe_dir>` is the `Tahoe-100M` directory produced by `download_tahoe.sh`.
- `--cell_thresh` (default `50`) and `--umi_thresh` (default `35000`) set the
  minimum cell/UMI counts a pseudobulk must have to pass QC.
- `--alpha` (default `10000`) is the scale factor used for log-normalization.
  The default config in `configs/default_paths.toml` assumes `--alpha 10000`;
  if you use a different value, downstream scripts won't find the result
  automatically unless you update the config to match.

The script:

1. Loads sample and gene metadata, and restricts to protein-coding genes
   (from `data/protein-coding_gene.txt`), writing the resulting gene tokens
   to `$BIOPERT_OUTDIR/preprocessed_data/tahoe_coding_tokens.csv`.
2. Builds one pseudobulk per (cell line, sample) by summing single-cell
   expression profiles, saving the raw result to
   `$BIOPERT_OUTDIR/preprocessed_data/pseudobulks.jld2`.
3. Filters out pseudobulks failing the cell/UMI thresholds, and cell lines
   with a QC failure rate above 90%.
4. Log-normalizes the filtered pseudobulks and writes the final result to
   `$BIOPERT_OUTDIR/preprocessed_data/pseudobulks_alpha_<alpha>.jld2`.

## Molecular embeddings

Molecule embeddings are computed from a CSV of unique (drug, SMILES) pairs, extracted
from a preprocessed dataset. "Embedding" here covers both learned/dense vectors
(CheMeleon, pretrained models) and classic bit-vector fingerprints (RDKit) — every
method below produces one fixed-length vector per compound, saved the same way.

### 1. Extract compound SMILES

Once a preprocessed dataset is available (see [LINCS Preprocessing](#lincs-preprocessing)
or [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), extract the unique
compounds it contains with:

```bash
julia --project=julia scripts/preprocessing/build_smiles_csv.jl $BIOPERT_OUTDIR <dataset>
```

- `<dataset>` is `lincs` or `tahoe`.

The result is written to the dataset's `smiles_csv` path from
`configs/default_paths.toml` (e.g.
`$BIOPERT_OUTDIR/smiles/smiles_filtered_lincs.csv` for LINCS), with columns
`drug` and `smiles`.

Run this once for LINCS and once for Tahoe-100M.

### 2. Compute CheMeleon embeddings

Encodes each compound as a dense vector using the pretrained
[CheMeleon](https://zenodo.org/records/15460715) message-passing model. Weights
are downloaded automatically on first use and cached at `--ckpt_path` (default:
`~/.chemprop/chemeleon_mp.pt`). GPU is used automatically if available.

```bash
conda activate biopert
python scripts/molec_embeds/get_chemeleon_embeddings.py \
    --outdir $BIOPERT_OUTDIR \
    --dataset <dataset> \
    [--batch_size INT]     # default: 256
    [--ckpt_path PATH]     # default: ~/.chemprop/chemeleon_mp.pt
```

- `<dataset>` is `lincs` or `tahoe`.

**Output:** `$BIOPERT_OUTDIR/molec_embeds/<dataset>/chemeleon/embeds.parquet`,
with columns `drug`, `smiles`, and `embedding`.

### 3. Compute pretrained-model embeddings

Encodes each compound as a dense vector using a pretrained HuggingFace or
UniMol model. For HuggingFace models whose tokenizer vocabulary is SELFIES
rather than SMILES (e.g. `zjunlp/MolGen-large`), SMILES are converted to
SELFIES before tokenization; molecules that fail to convert are skipped.
GPU is used automatically if available.

```bash
conda activate biopert
python scripts/molec_embeds/get_pretrained_molec_embeddings.py \
    --pretrained_name_or_path <model_name> \
    --outdir $BIOPERT_OUTDIR \
    --dataset <dataset> \
    [--batch_size INT]    # default: 32
    [--max_length INT]    # default: 512
    [--device auto|cuda|cpu]  # default: auto
```

- `<dataset>` is `lincs` or `tahoe`.
- `<model_name>` can be any HuggingFace model identifier (e.g.
  `DeepChem/ChemBERTa-77M-MLM`, `ibm/MoLFormer-XL-both-10pct`,
  `zjunlp/MolGen-large`, `unikei/bert-base-smiles`), or a UniMol identifier of the form
  `<model_name>/<model_size>` (e.g. `unimolv1/84m`), which is routed to
  `unimol_tools` instead.

**Output:** `$BIOPERT_OUTDIR/molec_embeds/<dataset>/<model_name>/embeds.parquet`,
with columns `drug`, `smiles`, and `embedding` (also saved as a HuggingFace
`Dataset` in the same directory).

### 4. Compute RDKit fingerprints

Computes classic bit-vector fingerprints for each compound with RDKit: MACCS
(166 bits) and, at 512/1024/2048 bits each, ECFP6 (Morgan, radius 3), RDKit,
and AtomPair. Each family is saved as its own embedding.

```bash
conda activate biopert
python scripts/molec_embeds/generate_fingerprints.py \
    --outdir $BIOPERT_OUTDIR \
    --dataset <dataset>
```

- `<dataset>` is `lincs` or `tahoe`.

**Output:** one `$BIOPERT_OUTDIR/molec_embeds/<dataset>/<fingerprint>/embeds.parquet`
per fingerprint family (`MACCS_166`, `ECFP6_512`, `ECFP6_1024`, `ECFP6_2048`,
`RDKit_512`, `RDKit_1024`, `RDKit_2048`, `AtomPair_512`, `AtomPair_1024`,
`AtomPair_2048`), with columns `drug`, `smiles`, and `embedding`.

### 5. Generate random baseline embeddings

Generates a random Gaussian vector per compound, for use as an uninformative
baseline against the CheMeleon/pretrained-model embeddings above. Each
molecule's vector is seeded from its SMILES string and `--seed`, so it stays
fixed across runs regardless of row order.

```bash
conda activate biopert
python scripts/molec_embeds/generate_random_embeddings.py \
    --outdir $BIOPERT_OUTDIR \
    --dataset <dataset> \
    [--n INT]     # embedding dimension, default: 512
    [--seed INT]  # global random seed, default: 42
```

- `<dataset>` is `lincs` or `tahoe`.

**Output:** `$BIOPERT_OUTDIR/molec_embeds/<dataset>/random_<n>/embeds.parquet`,
with columns `drug`, `smiles`, and `embedding`.

### Run everything at once

To run steps 2-5 above (fingerprints, CheMeleon, every pretrained model, and
the random baseline) in one go, with default arguments throughout:

```bash
conda activate biopert
scripts/molec_embeds/run_all_molec_embeds.sh $BIOPERT_OUTDIR <dataset>
```

- `<dataset>` is `lincs` or `tahoe`.
- Step 1 (extract compound SMILES) is a prerequisite and must be run
  separately first.

## Predict target delta profiles

Once a preprocessed dataset is available (see [LINCS Preprocessing](#lincs-preprocessing)
or [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), make sure the
[Python environment is configured for
PythonCall](#python-calls-from-julia-pythoncall) first
(`source scripts/setup/setup_python_env.sh`) — this script also logs to
[Weights & Biases](https://wandb.ai) via the `wandb` Python package, so the
active Python environment must have it installed and, unless `wandb_mode =
"disabled"`, be logged in (`wandb login`) — then train a model to predict
each target cell line's delta expression profile from a reference cell line's
matched delta profile and/or the target cell line's average untreated
profile, with:

```bash
julia --project=julia scripts/predict_drug_response/predict_target_delta_profiles.jl <config_file> <outdir> <dataset>
```

- `<outdir>` is `$BIOPERT_OUTDIR` (see [Output directory](#output-directory-biopert_outdir)).
  Results are written under it (see below).
- `<dataset>` is `"lincs"` or `"tahoe"`. Together with `<outdir>`, it resolves
  the preprocessed `.jld2` file and, when `use_pca_cache` is set, `pca_dir`
  from `configs/default_paths.toml`.
- `<config_file>` is a TOML file (see
  [`configs/tahoe_minimal_config.toml`](configs/tahoe_minimal_config.toml) for
  a minimal example) with the following keys:
  - `ref_cl` — the reference cell line symbol. Required.
  - `wandb_mode` — `"online"`, `"offline"`, or `"disabled"`. Required.
  - `hidden_layers` — array of hidden layer sizes for the MLP. Required.
  - `batch_size`, `n_epochs`, `lr`, `weight_decay` — training hyperparameters. Required.
  - `use_delta_ref` (default `true`) — whether to include the reference cell
    line's delta profile as an input feature.
  - `molec_embed_file` (default none) — path to a molecular-embedding
    parquet file; if set, molecular embeddings are used as input.
  - `dose_encoding` (default `"none"`) — `"none"`, `"concat"`, `"gate"`, or
    `"onehot"`; requires `molec_embed_file` to be set.
  - `n_pca_expr` (default none) — PCA-reduce the delta-ref and
    untreated-target expression inputs to this many components. Both share
    this one dimensionality (fit as two separate transforms, same target
    size) — they can't currently be set independently (see the `TODO` above
    this option's parsing in the script).
  - `n_pca_molec` (default none) — PCA-reduce the molecular-embedding
    inputs to this many components.
  - `use_pca_cache` (default `false`) — if `true`, cache fitted PCA transforms
    under `pca_dir` across runs (loading them back if already computed);
    requires `pca_dir` to be set. If `false`, PCA is recomputed every run.
  - `pca_dir` (default none) — directory to cache fitted PCA transforms in;
    only used when `use_pca_cache` is `true`.
  - `model_type` (default `"mlp"`) — `"mlp"`, `"lasso"`, `"ridge"`, or `"xgboost"`.
  - Everything else (`seed`, `split_seed`, `average_ref`, `resample_ref`,
    `landmark_genes_only`, `val_frac`/`test_frac`, pinned/cell-line-holdout
    split paths, `repro_delta_inter_path`/`threshold`, `dropout_arr`,
    `warmup_steps`, `loss_name`, `save_predictions`, ...) is optional; see the
    top of `main` in the script for the full list and defaults.

Results are written to `<outdir>/<jld2_basename>_<config_basename>_<timestamp>/`
(e.g. `pseudobulks_alpha_10000_tahoe_minimal_config_2026-01-01_120000/`):
- `best_model.jld2` — the checkpoint with the highest validation Spearman correlation.
- `test_perfs_per_obs.csv` — per-observation test-set metrics.
- `summary.toml` — hyperparameters and aggregate test-set metrics (Pearson,
  Spearman, cosine similarity, L2, RMSE).

## Reproducibility statistics

Once a preprocessed dataset is available (`filtered_lincs.jld2` from
[LINCS Preprocessing](#lincs-preprocessing), or `pseudobulks_alpha_<alpha>.jld2`
from [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), make sure the
[Python environment is configured for
PythonCall](#python-calls-from-julia-pythoncall) first
(`source scripts/setup/setup_python_env.sh`), then compute pairwise
reproducibility statistics (Pearson, Spearman) between condition-matched
profiles with:

```bash
julia --project=julia scripts/dataset_analysis/compute_repro_stats.jl $BIOPERT_OUTDIR <dataset> [--max_pairs N]
```

- `<dataset>` is `lincs` or `tahoe`.
- `--max_pairs` caps the number of profile pairs assessed per condition
  (`cell_line`, `drug`, `dose`, `time`); by default there is no cap. Profiles
  sharing a condition are condition-matched, not necessarily true replicates —
  in LINCS, for example, they can come from different projects. For LINCS,
  we suggest setting `--max_pairs 200` to keep runtime tractable, given how
  many condition-matched pairs a condition can have.

The script assesses untreated, treated, and delta profiles, each at both
intra-plate and inter-plate `plate_mode`, writing six files to
`$BIOPERT_OUTDIR/repro/repro_<input_basename>/` (where `<input_basename>` is
the name of the dataset's preprocessed `.jld2` file without its extension,
e.g. `filtered_lincs` or `pseudobulks_alpha_10000`): `repro_untrt_intra.csv`,
`repro_untrt_inter.csv`, `repro_trt_intra.csv`, `repro_trt_inter.csv`,
`repro_delta_intra.csv`, and `repro_delta_inter.csv`.

## SAR table

Once a preprocessed dataset is available (see [LINCS Preprocessing](#lincs-preprocessing)
or [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), make sure the
[Python environment is configured for
PythonCall](#python-calls-from-julia-pythoncall) first
(`source scripts/setup/setup_python_env.sh`), then build a
structure-activity relationship (SAR) table — one row per compound pair,
comparing chemical distance against biological (delta profile) similarity —
with:

```bash
julia --project=julia scripts/dataset_analysis/build_sar_table.jl $BIOPERT_OUTDIR <dataset> [--max_pairs N] [--repro_criteria pearson|spearman] [--repro_threshold N]
```

- `<dataset>` is `lincs` or `tahoe`. The table is written to
  `$BIOPERT_OUTDIR/sar/sar_<input_basename>.csv` (where `<input_basename>` is
  the name of the dataset's preprocessed `.jld2` file without its extension).
- If `$BIOPERT_OUTDIR/molec_embeds/<dataset>/` exists (see [Molecular
  embeddings](#molecular-embeddings)), one chemical-distance column is added
  per molecular representation found there; otherwise embedding-based
  distance columns are skipped.
- `--max_pairs` caps the number of profile pairs assessed per condition when
  computing reproducibility; by default there is no cap. **For LINCS, we
  suggest setting `--max_pairs 200`** to keep runtime tractable, given how
  many condition-matched pairs a condition can have.
- `--repro_criteria` (default `pearson`) and `--repro_threshold` (default
  `0.7`) control which delta profiles are kept: only conditions (`cell_line`,
  `drug`, `dose`, `time`) whose minimum inter-plate `repro_criteria` is at
  least `repro_threshold` are included in the SAR table.

## Delta-ref UMAP

Once a preprocessed dataset is available (see [LINCS Preprocessing](#lincs-preprocessing)
or [Tahoe-100M Preprocessing](#tahoe-100m-preprocessing)), make sure the
[Python environment](#python-environment) is active (for `juliacall`) and a
working `julia` executable is on `PATH` with
[`julia/Project.toml`'s dependencies instantiated](#julia-package), then
compute a UMAP of delta profiles (treated − matched plate DMSO mean) on the
reference cell line with:

```bash
conda activate biopert
python scripts/dataset_analysis/delta_ref_umap.py <dataset> $BIOPERT_OUTDIR [--ref_cl CL] [--average] [--n_neighbors N] [--min_dist F] [--metric M] [--seed N]
```

- `<dataset>` is `lincs` or `tahoe`. The preprocessed `.jld2` is located via
  `configs/default_paths.toml`, the same way every other script in this repo
  resolves dataset paths.
- The script reads the preprocessed JLD2 (a DataFrame with one row per
  `(cell_line, sample)`) through an embedded Julia runtime (`juliacall`), since
  JLD2 serializes DataFrames using Julia-specific encoding that plain HDF5
  readers can't parse.
- `--ref_cl` defaults to `A549` for LINCS and `CVCL_0023` for Tahoe.
- `--average` averages replicates per `(drug, dose, time)` before fitting UMAP;
  by default all replicates are kept separate.
- `--n_neighbors`, `--min_dist`, `--metric`, and `--seed` control the UMAP fit
  (defaults: `15`, `0.1`, `cosine`, `42`).
- Outputs are written to
  `$BIOPERT_OUTDIR/delta_ref_umap/<dataset>/<ref_cl>_<n_neighbors>_<min_dist>[_unaveraged]/`:
  `umap_model.pkl` (fitted `umap.UMAP` object), `umap_embedding.npy` (2-D
  coordinates), and `umap_uncolored.png` (scatter plot).
