#!/usr/bin/env Rscript
# Variance decomposition of batch effects using variancePartition (Hoffman & Schadt,
# BMC Bioinformatics 2016), the standard tool for attributing expression variance to
# experimental factors. Replaces an earlier in-house estimator so that the reported
# components come from the reference implementation (lme4 REML) rather than a
# reimplementation.
#
# Reads the matrices written by scripts/export_expr_for_batch_analysis.jl and writes
#   <prefix>_vp_per_gene.csv  one row per gene, % variance per factor
#   <prefix>_vp_summary.csv   median / IQR across genes per factor
#
# Usage:
#   Rscript scripts/analyze_batch_variance.R <dir> <prefix> <design> [n_cores] \
#          [max_genes] [chunk_id] [n_chunks]
#     design:    "nested"  -> (1|cell_line) + (1|cell_line:plate)   [LINCS]
#                "crossed" -> (1|cell_line) + (1|plate)             [Tahoe]
#     max_genes: fit only the first N genes (0 = all). For smoke tests and timing.
#     chunk_id / n_chunks: fit only this chunk's slice of genes and write it to
#          <prefix>_vp_chunk<id>.csv. Genes are independent, so a SLURM array can
#          run the chunks concurrently; merge them afterwards with --merge.

# Bioconductor packages live outside the conda env (see project memory): conda's R
# toolchain cannot link on this Debian host, so they were built against the system
# compilers and installed to a separate library. Override with BIOPERT_R_LIBS if
# your build lives elsewhere.
.libPaths(c(Sys.getenv("BIOPERT_R_LIBS", "/network/scratch/l/lola.lebreton/R_libs"), .libPaths()))
suppressPackageStartupMessages({
  library(variancePartition); library(BiocParallel); library(arrow)
})

# variancePartition warns unless the response columns and metadata rows carry
# matching names; the export guarantees they are in the same order, so name them.


args    <- commandArgs(trailingOnly = TRUE)
dir     <- args[1]; prefix <- args[2]; design <- args[3]
n_cores   <- if (length(args) >= 4) as.integer(args[4]) else 4L
max_genes <- if (length(args) >= 5) as.integer(args[5]) else 0L
chunk_id  <- if (length(args) >= 6) as.integer(args[6]) else 0L
n_chunks  <- if (length(args) >= 7) as.integer(args[7]) else 0L
# Minimum plausible seconds/gene; guards against the aborted-optimiser failure.
min_s_per_gene <- if (length(args) >= 8) as.numeric(args[8]) else 0

meta <- read.csv(file.path(dir, paste0(prefix, "_expr_meta.csv")), stringsAsFactors = FALSE)
expr <- arrow::read_feather(file.path(dir, paste0(prefix, "_expr.feather")))
expr <- as.matrix(expr)                       # genes x samples
stopifnot(ncol(expr) == nrow(meta))
# NB: `sample` is NOT unique in Tahoe — one sample is a treatment well profiled
# across all 48 cell lines, so only (cell_line, sample) identifies a row. Use a
# positional id, which is unique for both datasets and keeps the export's order.
ids <- paste0("p", seq_len(nrow(meta)))
rownames(meta) <- colnames(expr) <- ids
rownames(expr) <- paste0("g", seq_len(nrow(expr)) - 1L)

# Missing dose (vehicle wells) / time become an explicit level, matching the export.
for (c in c("cell_line", "plate", "drug", "dose", "time")) {
  meta[[c]][is.na(meta[[c]]) | meta[[c]] == ""] <- "__none__"
  meta[[c]] <- factor(meta[[c]])
}

# Drop factors with a single level (Tahoe has one time point).
terms <- c("cell_line", "plate", "drug", "dose", "time")
terms <- terms[vapply(terms, function(t) nlevels(meta[[t]]) >= 2, logical(1))]

# Plate is ~100% nested in cell line on LINCS, so it enters as the interaction;
# Tahoe is fully crossed and plate enters directly.
plate_term <- if (design == "nested") "(1|cell_line:plate)" else "(1|plate)"
others     <- setdiff(terms, c("cell_line", "plate"))
form <- as.formula(paste("~ (1|cell_line) +", plate_term,
                         if (length(others)) paste("+", paste0("(1|", others, ")", collapse = " + ")) else ""))
if (max_genes > 0L && max_genes < nrow(expr)) {
  expr <- expr[seq_len(max_genes), , drop = FALSE]
  cat("SUBSET: fitting only the first", max_genes, "genes\n")
}

# Chunking for SLURM array jobs. Genes are fitted independently, so slicing the
# matrix by row is exact — each chunk sees the full profile matrix and the same
# model, only fewer responses. Contiguous slices keep gene ids trivially recoverable.
gene_offset <- 0L
if (n_chunks > 0L) {
  stopifnot(chunk_id >= 1L, chunk_id <= n_chunks)
  bounds <- round(seq(0, nrow(expr), length.out = n_chunks + 1L))
  lo <- bounds[chunk_id] + 1L; hi <- bounds[chunk_id + 1L]
  if (lo > hi) { cat("Chunk", chunk_id, "is empty; nothing to do.\n"); quit(status = 0) }
  gene_offset <- lo - 1L
  expr <- expr[lo:hi, , drop = FALSE]
  cat(sprintf("CHUNK %d/%d: genes %d-%d (%d genes)\n",
              chunk_id, n_chunks, lo - 1L, hi - 1L, nrow(expr)))
}

cat("Formula:", deparse(form), "\n")
cat("Design:", nrow(meta), "profiles x", nrow(expr), "genes |", n_cores, "cores\n")
cat("Levels:", paste(sprintf("%s=%d", terms,
    vapply(terms, function(t) nlevels(meta[[t]]), integer(1))), collapse = " "), "\n")

# Pass BPPARAM explicitly. Relying on register() alone did not parallelise —
# the job ran single-threaded at 100% of one core with 8 allocated.
bp <- MulticoreParam(workers = n_cores, progressbar = FALSE)
register(bp)
t0 <- Sys.time()
vp <- fitExtractVarPartModel(expr, form, meta, BPPARAM = bp)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("Fitted %d genes in %.1f min (%.2f s/gene)\n",
            nrow(expr), elapsed, elapsed * 60 / nrow(expr)))
vp <- as.data.frame(vp)

# Variance fractions must form a valid partition of [0,1] for every gene.
stopifnot(all(vp >= -1e-8), all(vp <= 1 + 1e-8))
stopifnot(all(abs(rowSums(vp) - 1) < 1e-6))

# Degenerate-fit guard. On this cluster's Sapphire Rapids `cn-m*` nodes, lmer's
# optimiser terminated immediately and returned nonsense while still producing a
# structurally valid partition, so the checks above do not catch it. The reliable
# signature is speed: a genuine fit of this model costs seconds to a minute per
# gene, whereas the aborted fits returned 14-30x faster. min_s_per_gene is passed
# in per dataset (0 disables) — set it from a known-good run on a healthy node.
sec_per_gene <- elapsed * 60 / nrow(expr)
if (min_s_per_gene > 0 && sec_per_gene < min_s_per_gene) {
  cat(sprintf("\nFATAL: %.2f s/gene is below the %.2f s/gene floor for this dataset.\n",
              sec_per_gene, min_s_per_gene))
  cat("lmer almost certainly aborted without converging (observed on cn-m* nodes).\n")
  cat("Nothing written. Re-run on a different node architecture.\n")
  quit(status = 2)
}
# Per-gene single-factor concentration. This was previously fatal, but it fires on
# ordinary per-gene convergence failures (uniformly 6-12% on lincs_delta, on healthy
# nodes at normal speed), not on node corruption — the speed floor above is the
# actual corruption defence. Reported, never fatal; filtering is a notebook decision.
frac_degenerate <- mean(rowSums(vp) - apply(vp, 1, max) < 1e-3)
cat(sprintf("Genes with one factor at ~100%%: %.1f%%\n", 100 * frac_degenerate))

# A chunk writes only its own slice; the summary is meaningless per chunk, so it
# is skipped and computed once at merge time.
if (n_chunks > 0L) {
  out <- file.path(dir, sprintf("%s_vp_chunk%03d.csv", prefix, chunk_id))
  write.csv(cbind(gene = gene_offset + seq_len(nrow(vp)) - 1L, vp * 100),
            out, row.names = FALSE)
  cat("Wrote", out, "\n")
  quit(status = 0)
}

suffix <- if (max_genes > 0L) paste0("_subset", max_genes) else ""
write.csv(cbind(gene = seq_len(nrow(vp)) - 1L, vp * 100),
          file.path(dir, paste0(prefix, "_vp_per_gene", suffix, ".csv")), row.names = FALSE)
summ <- data.frame(
  factor = colnames(vp),
  median = apply(vp * 100, 2, median),
  q25    = apply(vp * 100, 2, quantile, 0.25),
  q75    = apply(vp * 100, 2, quantile, 0.75))
write.csv(summ, file.path(dir, paste0(prefix, "_vp_summary", suffix, ".csv")), row.names = FALSE)
print(summ, row.names = FALSE)
cat("Wrote", prefix, "variancePartition outputs to", dir, "\n")