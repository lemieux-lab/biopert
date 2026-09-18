"""
Statistical annotations for the CAP submission figures.

All uncertainty in these figures reflects **sampling variability over the finite
test set**, not model-training variance: each experiment is a single (val-best)
model, so we cannot estimate initialization/seed variance. What we *can* estimate
is how confident we are in a reported test metric given the finite set of test
conditions. Two consequences shape every helper here:

* Every plotted metric is a *mean over test conditions* of a per-observation
  score (``pearson`` / ``spearman`` / ``l2`` / ``cosine``). Its confidence
  interval is a **cluster/percentile bootstrap over test conditions**
  (:func:`bootstrap_ci_mean`).
* The test set is **pinned and shared** across experiments within a dataset, so
  model-vs-model and model-vs-baseline comparisons are **paired** on the test
  conditions — far more powerful than unpaired tests. We report a paired
  bootstrap CI on the mean difference plus a Wilcoxon signed-rank p-value
  (:func:`paired_delta`), Benjamini-Hochberg-corrected across the comparisons in
  a figure (:func:`bh_fdr`).

For the correlation panels we report the Spearman ρ already shown, now with a
bootstrap CI and its analytic p-value (:func:`spearman_ci`). For "does the
reference cell line matter?" we run a Friedman test blocked by target cell line
(:func:`friedman_blocked`) — a paired omnibus across references.

Everything is deterministic given a fixed ``seed`` so re-running the notebook
reproduces identical intervals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata

# Keys that uniquely identify a test condition across experiments of one dataset.
OBS_KEYS = ["cell_line", "drug", "dose", "time"]

# One bootstrap budget for every figure analysis.
DEFAULT_N_BOOT = 1000

# Memory budget for the vectorised bootstrap: process resamples in chunks so the
# index matrix never exceeds ~2e7 entries at once (a few hundred MB).
_MAX_CELLS = 20_000_000


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _boot_means(values: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Bootstrap distribution of the mean, computed in memory-safe chunks."""
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, int(_MAX_CELLS // max(n, 1)))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        means[done:done + b] = values[idx].mean(axis=1)
        done += b
    return means


def _boot_cluster_means(sums: np.ndarray, counts: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Bootstrap distribution of the pooled mean, resampling whole clusters.

    The mean of a pooled cluster resample is ``sum(cluster sums) / sum(cluster
    sizes)``, so only per-cluster totals are needed. That makes this
    O(n_boot x n_clusters) instead of O(n_boot x n) and exactly equal to
    concatenating the resampled clusters and averaging.
    """
    k = sums.size
    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, int(_MAX_CELLS // max(k, 1)))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(b, k))
        means[done:done + b] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        done += b
    return means


def bootstrap_ci_mean(values, n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0,
                      seed: int = 0, cluster_by=None):
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(mean, lo, hi)``. NaNs are dropped. ``n_boot=1000`` is the
    project-wide bootstrap budget.

    ``cluster_by`` — optional group labels, one per element of ``values``. When
    given, **whole groups are resampled with replacement** instead of individual
    rows, so the interval no longer assumes rows are independent. Use it when
    observations share a source of variation: test conditions in the same cell
    line share that line's difficulty, and treating them as independent makes the
    interval far too narrow (LINCS default mean Pearson: +-0.0024 i.i.d. vs
    +-0.015 clustered by cell line). The point estimate is unchanged either way;
    only the interval widens. Paired differences are much less affected, since
    differencing cancels the shared per-condition component.

    Default ``cluster_by=None`` performs the i.i.d. row bootstrap.
    """
    if cluster_by is None:
        values = _clean(values)
        if values.size == 0:
            return (np.nan, np.nan, np.nan)
        rng = np.random.default_rng(seed)
        means = _boot_means(values, n_boot, rng)
    else:
        values = np.asarray(values, dtype=np.float64).ravel()
        labels = np.asarray(cluster_by).ravel()
        if labels.size != values.size:
            raise ValueError(f"cluster_by has {labels.size} labels for {values.size} values")
        keep = np.isfinite(values)
        values, labels = values[keep], labels[keep]
        if values.size == 0:
            return (np.nan, np.nan, np.nan)
        _, inv = np.unique(labels, return_inverse=True)
        k = int(inv.max()) + 1
        sums = np.bincount(inv, weights=values, minlength=k)
        counts = np.bincount(inv, minlength=k).astype(np.float64)
        rng = np.random.default_rng(seed)
        means = _boot_cluster_means(sums, counts, n_boot, rng)
    alpha = (100.0 - ci) / 2.0
    return float(values.mean()), float(np.percentile(means, alpha)), \
        float(np.percentile(means, 100.0 - alpha))


def paired_delta(df_a, df_b, metric: str = "pearson", keys=OBS_KEYS,
                 n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0, seed: int = 0):
    """Paired comparison of two experiments on their shared test conditions.

    Aligns ``df_a`` and ``df_b`` on ``keys`` (mean-aggregating any duplicate
    key so the join is one-to-one), then reports the mean of the paired
    difference ``a - b`` with a bootstrap CI, and a two-sided Wilcoxon
    signed-rank p-value on the paired differences.

    Returns a dict: ``mean_diff, lo, hi, p, n_pairs`` (mean_a/mean_b too).
    """
    a = df_a.groupby(keys, observed=True)[metric].mean()
    b = df_b.groupby(keys, observed=True)[metric].mean()
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    d = (joined["a"] - joined["b"]).to_numpy(dtype=np.float64)
    out = {"n_pairs": int(d.size), "mean_a": float(joined["a"].mean()),
           "mean_b": float(joined["b"].mean()), "mean_diff": float(d.mean()),
           "lo": np.nan, "hi": np.nan, "p": np.nan}
    if d.size == 0:
        return out
    rng = np.random.default_rng(seed)
    boot = _boot_means(d, n_boot, rng)
    alpha = (100.0 - ci) / 2.0
    out["lo"] = float(np.percentile(boot, alpha))
    out["hi"] = float(np.percentile(boot, 100.0 - alpha))
    # Wilcoxon is undefined if every difference is zero; guard it.
    if np.any(d != 0):
        try:
            out["p"] = float(stats.wilcoxon(d, zero_method="wilcox",
                                            alternative="two-sided").pvalue)
        except ValueError:
            out["p"] = np.nan
    return out


def spearman_ci(x, y, n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0, seed: int = 0):
    """Spearman ρ with an analytic p-value and a bootstrap CI over point pairs.

    Returns a dict: ``rho, p, lo, hi, n``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    out = {"rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan, "n": int(n)}
    if n < 3:
        return out
    res = stats.spearmanr(x, y)
    out["rho"], out["p"] = float(res.statistic), float(res.pvalue)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size:
        alpha = (100.0 - ci) / 2.0
        out["lo"] = float(np.percentile(rhos, alpha))
        out["hi"] = float(np.percentile(rhos, 100.0 - alpha))
    return out


def linfit_ci(x, y, n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0, seed: int = 0):
    """Slope of a least-squares line with bootstrap CI + Pearson p-value.

    For the "bigger encoder does not help" scaling panel: ``x`` is typically
    log10(parameters). Returns ``slope, lo, hi, p, r, n``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    out = {"slope": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
           "r": np.nan, "n": int(n)}
    if n < 3:
        return out
    slope, intercept = np.polyfit(x, y, 1)
    pr = stats.pearsonr(x, y)
    out["slope"], out["r"], out["p"] = float(slope), float(pr.statistic), \
        float(pr.pvalue)
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
    alpha = (100.0 - ci) / 2.0
    out["lo"] = float(np.percentile(slopes, alpha))
    out["hi"] = float(np.percentile(slopes, 100.0 - alpha))
    return out


def partial_spearman(x, y, covar, n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0,
                     seed: int = 0):
    """Spearman partial correlation of ``x`` and ``y`` controlling for ``covar``.

    Rank-transforms every variable, regresses the ranks of ``x`` and ``y`` on the
    ranks of the covariate(s), and correlates the residuals — i.e. the monotone
    association between x and y with the covariate's (monotone) effect removed.
    ``covar`` may be 1-D or a 2-D array (multiple covariates). Returns a dict
    ``rho, p, lo, hi, n``. The analytic p-value (t-test, df = n − 2 − k) and the
    row bootstrap CI both assume independent rows; when rows are clustered (e.g.
    repeated references per target cell line) prefer :func:`cluster_ols` for the
    non-independence-aware version.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(covar, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    x, y, C = x[m], y[m], C[m]
    n, k = x.size, C.shape[1]
    out = {"rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan, "n": int(n)}
    if n < k + 3:
        return out

    def _pr(xx, yy, CC):
        A = np.column_stack([np.ones(xx.size)]
                            + [rankdata(CC[:, j]) for j in range(CC.shape[1])])
        rx = rankdata(xx); ry = rankdata(yy)
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        sx, sy = ex.std(), ey.std()
        if sx == 0 or sy == 0:
            return np.nan
        return float(np.corrcoef(ex, ey)[0, 1])

    r = _pr(x, y, C)
    out["rho"] = r
    df = n - 2 - k
    if np.isfinite(r) and df > 0 and abs(r) < 1:
        t = r * np.sqrt(df / (1.0 - r * r))
        out["p"] = float(2 * stats.t.sf(abs(t), df))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = _pr(x[idx], y[idx], C[idx])
        if np.isfinite(v):
            boot.append(v)
    if boot:
        boot = np.asarray(boot)
        alpha = (100.0 - ci) / 2.0
        out["lo"] = float(np.percentile(boot, alpha))
        out["hi"] = float(np.percentile(boot, 100.0 - alpha))
    return out


def cluster_ols(y, X, groups, names=None, n_boot: int = DEFAULT_N_BOOT, ci: float = 95.0,
                seed: int = 0):
    """Multiple OLS regression with a cluster bootstrap over ``groups``.

    Fits ``y ~ 1 + X`` and gets a CI + two-sided bootstrap p-value for every
    coefficient by resampling whole clusters (``groups``) with replacement — the
    dependency-free stand-in for a random-intercept mixed model when observations
    are nested (here: the several references measured on each target cell line).
    Standardize the columns of ``X`` beforehand for comparable coefficients.
    Returns ``{name: {coef, lo, hi, p}, ...}`` plus ``_n`` and ``_n_groups``.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    groups = np.asarray(groups)
    A = np.column_stack([np.ones(y.size), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([gidx[g] for g in samp])
        try:
            boots.append(np.linalg.lstsq(A[rows], y[rows], rcond=None)[0])
        except np.linalg.LinAlgError:
            pass
    boots = np.asarray(boots)
    labels = ["intercept"] + (list(names) if names is not None
                              else [f"x{i}" for i in range(X.shape[1])])
    alpha = (100.0 - ci) / 2.0
    res = {"_n": int(y.size), "_n_groups": int(uniq.size)}
    for i, lab in enumerate(labels):
        col = boots[:, i]
        frac = float(np.mean(col > 0))
        # Two-sided bootstrap p, floored at the resolution of the bootstrap
        # (~1/n_boot): with every replicate on one side we can only say
        # p < 1/n_boot, not p ≈ 0.
        p = max(2 * min(frac, 1.0 - frac), 1.0 / col.size)
        res[lab] = {"coef": float(beta[i]),
                    "lo": float(np.percentile(col, alpha)),
                    "hi": float(np.percentile(col, 100.0 - alpha)),
                    "p": float(min(1.0, p))}
    return res


def friedman_blocked(wide, treatments=None):
    """Friedman test across ``treatments`` (columns) blocked by row.

    ``wide`` is a DataFrame indexed by block (here: target cell line) with one
    column per treatment (here: reference choice). Rows with any missing
    treatment are dropped so every block is complete. Also returns Kendall's W
    (effect size in [0, 1]). Returns ``stat, p, n_blocks, k, W``.
    """
    if treatments is not None:
        wide = wide[list(treatments)]
    complete = wide.dropna(axis=0, how="any")
    n_blocks, k = complete.shape
    out = {"stat": np.nan, "p": np.nan, "n_blocks": int(n_blocks),
           "k": int(k), "W": np.nan}
    if n_blocks < 2 or k < 3:
        return out
    cols = [complete[c].to_numpy(dtype=np.float64) for c in complete.columns]
    res = stats.friedmanchisquare(*cols)
    out["stat"], out["p"] = float(res.statistic), float(res.pvalue)
    out["W"] = float(res.statistic / (n_blocks * (k - 1)))  # Kendall's W
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (same order as input).

    NaNs pass through as NaN and are excluded from the ranking.
    """
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = order.size
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        val = p[order[rank]] * m / (rank + 1)
        prev = min(prev, val)
        adj[rank] = prev
    out[order] = adj
    return out


def fmt_p(p) -> str:
    """Compact p-value string for figure annotations.

    Guards the float-underflow case (a p-value that rounds to 0.0 would render
    as the misleading "0e+00") by reporting the floating-point floor instead.
    """
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p == 0.0:
        return "p < 1e-300"
    if p < 1e-3:
        return f"p = {p:.0e}"
    return f"p = {p:.3f}"


def stars(p) -> str:
    """Significance marker for an (already-corrected) p-value."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."
