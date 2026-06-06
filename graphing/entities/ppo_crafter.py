"""Entity-specific plots for ppo_crafter.

Called automatically by `python -m graphing.run --entity ppo_crafter`.
Produces:
    - rdm_heatmap.pdf      Cosine-dissimilarity RDM for the final checkpoint
    - rsa_group_bars.pdf   Bar chart of RSA group alignments at the final checkpoint
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from graphing.loader import RunData, load_rdm_artifact, RSA_METRIC_PREFIX
from graphing.styles import (
    apply_base_style, save_fig, RSA_GROUP_COLORS,
)


def plot(run_data: RunData, output_dir: str, dpi: int = 150) -> None:
    """Generate all ppo_crafter entity-specific plots."""
    _plot_rdm_heatmap(run_data, output_dir, dpi)
    _plot_rsa_group_bars(run_data, output_dir, dpi)


# ---------------------------------------------------------------------------
# RDM heatmap — final checkpoint
# ---------------------------------------------------------------------------

def _plot_rdm_heatmap(run_data: RunData, output_dir: str, dpi: int) -> str | None:
    """Heatmap of the cosine-dissimilarity RDM from the last available checkpoint."""
    rdm_data = load_rdm_artifact(run_data.run_id)
    if rdm_data is None:
        print("  (rdm heatmap skipped: no RDM artifacts found)")
        return None

    rdm    = np.array(rdm_data["rdm"], dtype=float)
    labels = rdm_data["labels"]
    step   = rdm_data["step"]
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(rdm, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="Cosine dissimilarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Achievement Activation RDM — step {step:,}")

    fig.tight_layout()
    return save_fig(fig, os.path.join(output_dir, "rdm_heatmap.pdf"), dpi)


# ---------------------------------------------------------------------------
# RSA group alignment bar chart — final checkpoint
# ---------------------------------------------------------------------------

def _plot_rsa_group_bars(run_data: RunData, output_dir: str, dpi: int) -> str | None:
    """Bar chart of RSA alignment (Spearman ρ) per group at the last checkpoint."""
    if not run_data.rsa_groups:
        print("  (rsa group bars skipped: no RSA groups found)")
        return None

    # Pick value at the last step that is finite for each group
    group_values: dict[str, float] = {}
    for group in run_data.rsa_groups:
        key = f"{RSA_METRIC_PREFIX}{group}"
        vals = np.array(run_data.metrics.get(key, []), dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            group_values[group] = float(finite[-1])

    if not group_values:
        return None

    groups = sorted(group_values.keys())
    values = [group_values[g] for g in groups]
    colors = [RSA_GROUP_COLORS.get(g, "#555555") for g in groups]

    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.2), 4))
    bars = ax.bar(groups, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ax.set_ylabel("RSA alignment (Spearman ρ)")
    ax.set_title("RSA Group Alignment — final checkpoint")
    ax.set_ylim(-1.0, 1.0)
    apply_base_style(ax)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * np.sign(val),
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    return save_fig(fig, os.path.join(output_dir, "rsa_group_bars.pdf"), dpi)
