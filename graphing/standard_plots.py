"""Hard-coded standard metric timeseries plots.

All functions take (run_data: RunData, output_dir: str, dpi: int = 150)
and return the saved file path (or None if no data).

These plots require no entity-specific knowledge; they work for any entity
that has run the standard pipeline.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from graphing.loader import RunData, RSA_METRIC_PREFIX
from graphing.styles import (
    apply_base_style, fmt_millions, shaded_line, save_fig,
    C_SUCCESS, C_FAILURE, C_OPPOSITION, C_ACTIVATION, C_COSINE,
    RSA_GROUP_COLORS, ACHIEVEMENT_COLORS,
)


def _get(run_data: RunData, key: str) -> np.ndarray:
    """Return metric values as float array (NaN where absent)."""
    return np.array(run_data.metrics.get(key, [float("nan")] * len(run_data.steps)))


def _steps(run_data: RunData) -> np.ndarray:
    return np.array(run_data.steps, dtype=float)


def _zero_line(ax) -> None:
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_opposition(run_data: RunData, output_dir: str, dpi: int = 150) -> str | None:
    """Opposition score (cosine similarity between success/failure gradients) over training."""
    x = _steps(run_data)
    y = _get(run_data, "opposition_score")
    if not np.isfinite(y).any():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    shaded_line(ax, x, y, color=C_OPPOSITION, label="Opposition score")
    _zero_line(ax)
    ax.set_ylabel("Cosine similarity (success vs failure gradients)")
    ax.set_xlabel("Training step")
    ax.set_title("Opposition Score")
    fmt_millions(ax)
    apply_base_style(ax)
    ax.legend(fontsize=11)
    return save_fig(fig, os.path.join(output_dir, "opposition_score.pdf"), dpi)


def plot_coherence(run_data: RunData, output_dir: str, dpi: int = 150) -> str | None:
    """Gradient coherence for success and failure groups over training."""
    x = _steps(run_data)
    y_s = _get(run_data, "coherence_success")
    y_f = _get(run_data, "coherence_failure")
    if not np.isfinite(y_s).any() and not np.isfinite(y_f).any():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    shaded_line(ax, x, y_s, color=C_SUCCESS, label="Success group")
    shaded_line(ax, x, y_f, color=C_FAILURE, label="Failure group")
    ax.set_ylabel("Average pairwise cosine similarity")
    ax.set_xlabel("Training step")
    ax.set_title("Gradient Coherence")
    fmt_millions(ax)
    apply_base_style(ax)
    ax.legend(fontsize=11)
    return save_fig(fig, os.path.join(output_dir, "coherence.pdf"), dpi)


def plot_gradient_magnitude(run_data: RunData, output_dir: str, dpi: int = 150) -> str | None:
    """L2 norm of mean gradient for success and failure groups over training."""
    x = _steps(run_data)
    y_s = _get(run_data, "gradient_magnitude_success")
    y_f = _get(run_data, "gradient_magnitude_failure")
    if not np.isfinite(y_s).any() and not np.isfinite(y_f).any():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    shaded_line(ax, x, y_s, color=C_SUCCESS, label="Success group")
    shaded_line(ax, x, y_f, color=C_FAILURE, label="Failure group")
    ax.set_ylabel("L2 norm of mean gradient")
    ax.set_xlabel("Training step")
    ax.set_title("Gradient Magnitude")
    fmt_millions(ax)
    apply_base_style(ax)
    ax.legend(fontsize=11)
    return save_fig(fig, os.path.join(output_dir, "gradient_magnitude.pdf"), dpi)


def plot_activation(run_data: RunData, output_dir: str, dpi: int = 150) -> str | None:
    """Activation space structure: separation and cosine distance side by side."""
    x = _steps(run_data)
    y_sep = _get(run_data, "activation_separation")
    y_cos = _get(run_data, "activation_cosine_distance")
    if not np.isfinite(y_sep).any() and not np.isfinite(y_cos).any():
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

    shaded_line(ax1, x, y_sep, color=C_ACTIVATION, label="Euclidean separation")
    ax1.set_ylabel("Euclidean distance")
    ax1.set_xlabel("Training step")
    ax1.set_title("Activation Centroid Separation")
    fmt_millions(ax1)
    apply_base_style(ax1)
    ax1.legend(fontsize=11)

    shaded_line(ax2, x, y_cos, color=C_COSINE, label="Cosine distance")
    ax2.set_ylabel("Cosine distance (1 − cos sim)")
    ax2.set_xlabel("Training step")
    ax2.set_title("Activation Cosine Distance")
    fmt_millions(ax2)
    apply_base_style(ax2)
    ax2.legend(fontsize=11)

    fig.tight_layout()
    return save_fig(fig, os.path.join(output_dir, "activation.pdf"), dpi)


def plot_rsa_groups(run_data: RunData, output_dir: str, dpi: int = 150) -> list[str]:
    """One PDF per RSA group, each showing Spearman ρ alignment over training."""
    x = _steps(run_data)
    saved = []
    for group in run_data.rsa_groups:
        key = f"{RSA_METRIC_PREFIX}{group}"
        y = _get(run_data, key)
        if not np.isfinite(y).any():
            continue

        color = RSA_GROUP_COLORS.get(group, "#555555")
        fig, ax = plt.subplots(figsize=(10, 4))
        shaded_line(ax, x, y, color=color, label=group.capitalize())
        _zero_line(ax)
        ax.set_ylabel("RSA alignment (Spearman ρ)")
        ax.set_xlabel("Training step")
        ax.set_title(f"RSA Alignment — {group.capitalize()} group")
        fmt_millions(ax)
        apply_base_style(ax)
        ax.legend(fontsize=11)
        path = save_fig(fig, os.path.join(output_dir, f"rsa_{group}.pdf"), dpi)
        saved.append(path)
    return saved


def plot_all(run_data: RunData, output_dir: str, dpi: int = 150) -> list[str]:
    """Generate all standard metric plots. Returns list of saved paths."""
    saved = []
    for fn in (plot_opposition, plot_coherence, plot_gradient_magnitude, plot_activation):
        p = fn(run_data, output_dir, dpi)
        if p:
            saved.append(p)
    saved.extend(plot_rsa_groups(run_data, output_dir, dpi))
    return saved
