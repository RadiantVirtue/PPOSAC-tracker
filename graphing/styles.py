"""Shared visual style: colors, figure helpers, axis formatters."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

C_SUCCESS      = "#E07B39"   # orange  — success group
C_FAILURE      = "#5B7FBF"   # blue    — failure group
C_OPPOSITION   = "#6B3FA0"   # purple  — opposition score
C_ACTIVATION   = "#2E8B57"   # green   — activation separation
C_COSINE       = "#9B2335"   # crimson — cosine distance
C_RSA_FIGHTING = "#C0392B"
C_RSA_RESOURCE = "#2980B9"
C_RSA_CRAFTING = "#27AE60"
C_RSA_HOUSING  = "#F39C12"

# Ordered palette for achievement lines in zoomed plots (cycles if > len)
ACHIEVEMENT_COLORS = [
    "#E07B39", "#5B7FBF", "#6B3FA0", "#2E8B57",
    "#9B2335", "#C0392B", "#2980B9", "#27AE60",
    "#F39C12", "#8E44AD", "#16A085", "#D35400",
]

# RSA group → color, for consistent coloring across plots
RSA_GROUP_COLORS: dict[str, str] = {
    "fighting": C_RSA_FIGHTING,
    "resource":  C_RSA_RESOURCE,
    "crafting":  C_RSA_CRAFTING,
    "housing":   C_RSA_HOUSING,
}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def apply_base_style(ax) -> None:
    """Grid, spine cleanup common to all plots."""
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fmt_millions(ax) -> None:
    """Format x-axis ticks as '1M', '500K', etc."""
    def _fmt(x, pos):
        if abs(x) >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if abs(x) >= 1_000:
            return f"{x / 1_000:.0f}K"
        return str(int(x))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))


def shaded_line(ax, x, y, yerr=None, color="#333333",
                label=None, marker=None, markersize=6,
                marker_every=20, alpha_fill=0.2) -> None:
    """Plot a line with optional shaded ±1 std band.

    Args:
        x:           1-D array of x values
        y:           1-D array of y values (may contain NaN)
        yerr:        1-D array of std values; if None no band is drawn
        color:       line and fill color
        label:       legend label
        marker:      matplotlib marker string or None
        markersize:  marker size in points
        marker_every: stride for placing markers
        alpha_fill:  opacity of std band
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if not valid.any():
        return

    ax.plot(x[valid], y[valid], color=color, label=label,
            marker=marker, markersize=markersize,
            markevery=marker_every, linewidth=1.6)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        ax.fill_between(x[valid], y[valid] - yerr[valid], y[valid] + yerr[valid],
                        color=color, alpha=alpha_fill)


def save_fig(fig, path: str, dpi: int = 150) -> str:
    """Save figure to path, creating parent directories as needed. Returns path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  saved {path}")
    return path
