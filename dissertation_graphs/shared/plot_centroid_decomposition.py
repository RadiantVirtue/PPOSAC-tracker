"""dissertation_graphs/shared/plot_centroid_decomposition.py

Centroid Decomposition: tracks how far the success (μ⁺) and failure (μ⁻)
activation centroids have each drifted from the untrained baseline μ₀
(pooled centroid at step ~50k).

  d⁺_t = ||μ⁺_t − μ₀||    (Euclidean drift, success)
  d⁻_t = ||μ⁻_t − μ₀||    (Euclidean drift, failure)

Output:
  GRAPHS/Dissertation graphs/A.2/centroid_decomposition.pdf
    Two-panel figure: PPO (left), Rainbow (right).

Data:
  ablation_results/centroid_decomposition/ppo/seed_N.json
  ablation_results/centroid_decomposition/rainbow/seed_N.json

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/shared/plot_centroid_decomposition.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    C_ORANGE_LIGHT,
    C_COHFAIL,
    METRIC_MARKERS,
)

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs", "A.2")
_DATA_ROOT   = os.path.join(_ROOT, "ablation_results", "centroid_decomposition")



def _load_algo(algo: str) -> dict[int, list[dict]]:
    """Return {seed_id: [records...]} for the given algorithm."""
    algo_dir = os.path.join(_DATA_ROOT, algo)
    if not os.path.isdir(algo_dir):
        return {}
    out = {}
    for fname in sorted(os.listdir(algo_dir)):
        if not fname.endswith(".json"):
            continue
        seed_str = os.path.splitext(fname)[0]
        try:
            sid = int(seed_str.replace("seed_", ""))
        except ValueError:
            continue
        with open(os.path.join(algo_dir, fname)) as f:
            out[sid] = json.load(f)
    return out


def _avg_field(seed_data: dict[int, list[dict]], field: str,
               step_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    arrs = []
    for records in seed_data.values():
        steps = np.array([r["step"] for r in records], dtype=float)
        vals  = np.array([r[field]  for r in records], dtype=float)
        interp = np.interp(step_grid, steps, vals,
                           left=np.nan, right=np.nan)
        in_range = (step_grid >= steps[0]) & (step_grid <= steps[-1])
        interp[~in_range] = np.nan
        arrs.append(interp)
    if not arrs:
        nan = np.full(len(step_grid), np.nan)
        return nan, nan, 0
    mat = np.vstack(arrs)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), len(arrs)


def _step_grid(seed_data: dict, n: int = 200) -> np.ndarray:
    all_steps = [r["step"] for records in seed_data.values() for r in records]
    if not all_steps:
        return np.linspace(0, 3_000_000, n)
    return np.linspace(min(all_steps), max(all_steps), n)



def _draw_panel(ax, seed_data: dict, title: str):
    if not seed_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        ax.set_title(title)
        return 0

    grid = _step_grid(seed_data)
    mk_suc, ms_suc = METRIC_MARKERS["coherence_success"]   # square
    mk_fai, ms_fai = METRIC_MARKERS["coherence_failure"]   # triangle

    for field, color, mk, ms, label in [
        ("d_plus",  C_ORANGE_LIGHT, mk_suc, ms_suc, r"$d^+$ (success drift)"),
        ("d_minus", C_COHFAIL,      mk_fai, ms_fai, r"$d^-$ (failure drift)"),
    ]:
        mean, std, n = _avg_field(seed_data, field, grid)
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax, grid[fin], mean[fin], std[fin],
                              color=color, label=label,
                              marker=mk, markersize=ms)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(r"$\|\mu_t - \mu_0\|$ (Euclidean)")
    _rq_fmt_millions(ax)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return n


def plot(ppo_root: str = "", rainbow_root: str = "",
         _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    ppo_data = _load_algo("ppo")
    rbw_data = _load_algo("rainbow")

    if not ppo_data and not rbw_data:
        print("WARNING: no centroid decomposition data found — skipping")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

    n_ppo = _draw_panel(axes[0], ppo_data,    "PPO")
    n_rbw = _draw_panel(axes[1], rbw_data, "Rainbow")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    _add_seed_note(fig, max(n_ppo, n_rbw))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(_GRAPHS_ROOT, exist_ok=True)
    path = os.path.join(_GRAPHS_ROOT, "centroid_decomposition.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return [path]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    plot(dpi=args.dpi)


if __name__ == "__main__":
    main()
