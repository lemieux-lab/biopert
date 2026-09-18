"""Nature-family figure styling shared by ``final_figures.ipynb``.

Provides the global rcParams (``set_nature_style``), page-width constants, the
fixed colour palette, per-axes styling (``style_ax`` / ``style_hbar_ax``),
automatic panel lettering (``label_panels`` / ``panel_label``) and PDF+PNG
export (``save_figure``). Spec: columns 89 / 120 / 183 mm, max height 247 mm,
sans-serif 5–7 pt, line weights ≥ 0.25 pt, embedded fonts, coolwarm for
continuous maps.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Page geometry (Nature) ─────────────────────────────────────────────────────
MM = 1.0 / 25.4
SINGLE_COL = 89 * MM  # 3.50 in
INTER_COL = 120 * MM  # ~4.72 in
DOUBLE_COL = 183 * MM  # 7.20 in
MAX_HEIGHT = 247 * MM  # full page, leaving room for the caption

# ── Colour palette (held consistent across every figure) ────────────────────────
BLUE = "#6687ed"  # default (reference-delta + untreated target); pulled from the
# mandated `coolwarm` map (coolwarm(0.14), avoiding its near-navy
# extreme) so the categorical blue still reads as part of the same
# continuous coolwarm family used elsewhere.
LBLUE = "#67b2f8"  # secondary / lighter series within a family (e.g. validation)
DBLUE = "#003f7f"  # reference-delta family (copy-ref-delta baseline, landmark_genes)
RED = "#d62728"  # emphasis: thresholds, the train/overfitting curve, fitted trends
ORANGE = "#e07b00"  # structure / FM embedding cluster
GRAY = "#888888"  # null references (random_512, train-mean) — dashed when a line

# Continuous palette mandated for any sequential / diverging map.
CONTINUOUS_CMAP = "coolwarm"

# category -> colour, consistent across all figures.
CATEGORY_COLORS = {
    "default": BLUE,
    "reference_delta": DBLUE,  # landmark_genes, copy-reference-delta baseline
    "embedding": ORANGE,  # structure / FM encoders
    "null": GRAY,  # random_512 control, train-mean baseline
    "emphasis": RED,
    "secondary": LBLUE,
}


# Bar colour grouped by *model type* (used in the representation ranking).
# Soft tints matching results.ipynb's TYPE_COLORS (default = BLUE, random = GRAY).
MODEL_TYPE_COLORS = {
    "Morgan fingerprint": "#f1ca82",  # warm sand
    "SMILES transformer": "#9ecdb1",  # sage
    "3D geometry (UniMol)": "#bba6e2",  # lavender
    "Molecular graph MPNN": "#e098a4",  # rose
}

# x-axis (bottom spine) weight — matches results.ipynb (heavier than other spines).
XAXIS_LW = 1.5

# Shared marker / error-bar kwargs (identical across panels).
SCATTER_KW = dict(edgecolor="white", linewidth=0.3, zorder=3)
ERR_KW = dict(fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5, capthick=0, zorder=2)
BAR_ERR_KW = dict(ecolor="#333333", elinewidth=0.6, capsize=1.5)


@contextmanager
def no_constrained_layout():
    """Build figures with constrained-layout OFF — needed for the manual
    gridspec/colorbar layouts, which otherwise hit a matplotlib bug under the
    inline backend. Set the rcParam (not the ``constrained_layout=`` kwarg,
    which is unreliable when the rcParam default is True)."""
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        yield


def category_color(category: str, label: str) -> str:
    """Return the palette colour for a ``results.csv`` (category, label) row.

    Mirrors the category -> colour rule in ``figure_plan.md`` so every figure
    colours a representation the same way.
    """
    if label == "default":
        return CATEGORY_COLORS["default"]
    if label in ("landmark_genes",) or label == "delta_ref":
        return CATEGORY_COLORS["reference_delta"]
    if label in ("random_512",) or label == "mean train delta":
        return CATEGORY_COLORS["null"]
    if category == "embedding":
        return CATEGORY_COLORS["embedding"]
    if category == "ref_cell":
        return CATEGORY_COLORS["default"]
    return GRAY


# ── Global style ────────────────────────────────────────────────────────────────
def set_nature_style(report: bool = True) -> None:
    """Apply Nature-family rcParams. Call once at the top of the notebook.

    Fonts embed as TrueType (``pdf.fonttype = 42``). Arial-likes are listed first
    for forward-compat, but DejaVu Sans is the only embeddable sans-serif on this
    host (covers ρ, Δ, ≈, ≥) and is what the PDFs actually use.
    """
    mpl.rcParams.update(
        {
            # Fonts — sans-serif, single small size across panels (5–7 pt).
            # Arial-likes first (forward-compatible); DejaVu Sans is the guaranteed
            # embeddable fallback actually used on this host.
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Arimo", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.titlesize": 8,
            "axes.titleweight": "regular",
            # Line / spine weights — thin but >= 0.25 pt and visible at final size.
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelpad": 2.0,
            "axes.titlepad": 3.0,
            # Legend — no frame, compact.
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.3,
            # Colour / export.
            "image.cmap": CONTINUOUS_CMAP,
            "axes.unicode_minus": False,  # ASCII minus -> always has a glyph
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 600,  # raster preview only; PDF is vector
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType, keep text editable
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Layout — keep panels from colliding.
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.04,
            "figure.constrained_layout.w_pad": 0.04,
            "figure.constrained_layout.hspace": 0.06,
            "figure.constrained_layout.wspace": 0.06,
        }
    )

    if report:
        from matplotlib.font_manager import findfont, FontProperties

        resolved = findfont(FontProperties(family=mpl.rcParams["font.sans-serif"]))
        print(f"Nature style set. Sans-serif font in use: " f"{os.path.basename(resolved)}")


# ── Per-axes styling ────────────────────────────────────────────────────────────
def style_ax(ax, grid: str = "y") -> None:
    """Shared panel style matching results.ipynb: drop top/right/left spines,
    heavier x-axis, no left ticks, faint reference grid.

    ``grid`` is ``"y"`` (default), ``"x"``, ``"both"`` or ``"none"``.
    """
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(XAXIS_LW)
    ax.tick_params(left=False, length=2.5, width=0.6)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.4, color=GRAY, alpha=0.6)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, which="major", linestyle=":", linewidth=0.4, color=GRAY, alpha=0.6)
    ax.set_axisbelow(True)


def style_hbar_ax(ax) -> None:
    """Style for horizontal bar panels: heavier x-axis, no y-axis line/ticks,
    faint x grid (category labels stay, but there is no y spine)."""
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(XAXIS_LW)
    ax.tick_params(length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, which="major", linestyle=":", linewidth=0.4, color=GRAY, alpha=0.6)
    ax.set_axisbelow(True)


# ── Panel lettering ─────────────────────────────────────────────────────────────
def panel_label(ax, letter: str, dx: float = -22.0, dy: float = 4.0, fontsize: float = 11.0, **kw) -> None:
    """Stamp one lower-case bold panel letter at the top-left of ``ax``.

    Anchored to the axes' top-left corner (axes fraction ``(0, 1)``) and
    nudged by ``(dx, dy)`` *points*, so the offset is identical regardless of
    panel size. Increase ``|dx|`` for panels with wide y tick labels.
    """
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        annotation_clip=False,
        **kw,
    )


def label_panels(axes: Iterable, letters: Sequence[str] | None = None, start: int = 0, **kw) -> None:
    """Automatically stamp a, b, c, … on a sequence of panels.

    ``axes`` is any iterable of Axes (e.g. a flattened ``plt.subplots`` array
    or an explicit list defining panel order). Hidden axes are skipped.
    Extra kwargs (``dx``, ``dy``, ``fontsize``) pass through to ``panel_label``.
    """
    axes = list(np.ravel(list(axes)))
    if letters is None:
        letters = [chr(ord("a") + i) for i in range(start, start + len(axes))]
    for ax, letter in zip(axes, letters):
        if ax is None or not ax.get_visible():
            continue
        panel_label(ax, letter, **kw)


# ── Trend annotation helper ─────────────────────────────────────────────────────
def add_trend(ax, x, y, color: str = RED, lw: float = 1.0, zorder: int = 4):
    """Draw a least-squares trend line spanning the x-range and return slope/intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    slope, intercept = np.polyfit(x[m], y[m], 1)
    xs = np.array([x[m].min(), x[m].max()])
    ax.plot(xs, slope * xs + intercept, color=color, lw=lw, zorder=zorder)
    return slope, intercept


# ── Export ───────────────────────────────────────────────────────────────────────
FINAL_DIR = "final_figures"


def save_figure(fig, name: str, formats: Sequence[str] = ("pdf", "png"), directory: str = FINAL_DIR) -> None:
    """Export ``fig`` as a vector PDF (and PNG preview) into ``final_figures/``."""
    os.makedirs(directory, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(directory, f"{name}.{ext}"))
    print(f"saved {directory}/{name}.{{{','.join(formats)}}}")
