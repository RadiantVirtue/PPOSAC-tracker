"""dissertation_graphs/shared/plot_activation_comparison.py

Figs 7.29 / 7.30: separate PPO and Rainbow panels for activation
separation (Euclidean) and activation cosine distance.

Four output PDFs (all in shared/output/):
  ppo_euclidean_separation.pdf       PPO activation_separation      Y = 0 – 6
  rainbow_euclidean_separation.pdf   Rainbow activation_separation  Y = 0 – 1.5
  ppo_cosine_distance.pdf            PPO activation_cosine_distance Y = 0 – 0.06
  rainbow_cosine_distance.pdf        Rainbow activation_cosine_distance Y = 0 – 0.06
  (plus legacy combined filenames for run_all.py backward compat)

Each: mean ± 1 std across 5 seeds.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/shared/plot_activation_comparison.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_experiment_root
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _avg_metric_across_seeds,
    _build_step_grid,
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    C_INDIGO,
    C_VIOLET,
    PPO_CLUSTERS,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs")



def _load_checkpoint_results(seed_dir: str, algorithm: str) -> list:
    log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
    if not os.path.isdir(log_dir):
        return []
    results = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            rec = load_analysis_results(fpath)
        except Exception:
            continue
        label = label_from_path(os.path.splitext(fname)[0])
        results.append((label, rec))
    results.sort(key=lambda x: (x[1].get("episode", 0) or 0,
                                 x[1].get("global_step", 0) or 0))
    return results


def _load_all_seed_results(experiment_root: str, algorithm: str) -> dict:
    out = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if not m:
            continue
        seed_id = int(m.group(1))
        out[seed_id] = _load_checkpoint_results(
            os.path.join(experiment_root, name), algorithm
        )
    return out



def _save_fig(fig, out_dir: str, fname: str, dpi: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _single_panel(results, step_grid, ach_steps, field, color, ylabel, title,
                  y_lo, y_hi, out_dir, fname, dpi, clusters=None) -> str:
    mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
    fin = np.isfinite(mean)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if fin.any():
        _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                          color=color, label=ylabel)
    ax.set_ylim(y_lo, y_hi)
    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=clusters)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper left", ncol=2)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    ax.set_ylabel(ylabel)
    _rq_fmt_millions(ax)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, n)
    return _save_fig(fig, out_dir, fname, dpi)



def plot(ppo_root: str, rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    ppo_results = _load_all_seed_results(ppo_root, "ppo")
    rbw_results = _load_all_seed_results(rainbow_root, "rainbow")

    if not ppo_results:
        print("ERROR: no PPO seed data found.")
        return []
    if not rbw_results:
        print("ERROR: no Rainbow seed data found.")
        return []

    ppo_grid = _build_step_grid(ppo_results)
    rbw_grid = _build_step_grid(rbw_results)
    ppo_ach = _ach_steps_from_all_results(ppo_results)
    rbw_ach = _ach_steps_from_all_results(rbw_results)
    out_dir = os.path.join(_GRAPHS_ROOT, "7.5.1")
    paths = []

    paths.append(_single_panel(
        ppo_results, ppo_grid, ppo_ach,
        "activation_separation", C_INDIGO, "Euclidean separation",
        "PPO — Activation Separation (Euclidean)",
        0.0, 6.0, out_dir, "ppo_euclidean_separation.pdf", dpi,
        clusters=PPO_CLUSTERS,
    ))

    paths.append(_single_panel(
        rbw_results, rbw_grid, rbw_ach,
        "activation_separation", C_INDIGO, "Euclidean separation",
        "Rainbow — Activation Separation (Euclidean)",
        0.0, 1.5, out_dir, "rainbow_euclidean_separation.pdf", dpi,
        clusters=RAINBOW_CLUSTERS,
    ))

    paths.append(_single_panel(
        ppo_results, ppo_grid, ppo_ach,
        "activation_cosine_distance", C_VIOLET, "Cosine distance",
        "PPO — Activation Cosine Distance",
        0.0, 0.06, out_dir, "ppo_cosine_distance.pdf", dpi,
        clusters=PPO_CLUSTERS,
    ))

    paths.append(_single_panel(
        rbw_results, rbw_grid, rbw_ach,
        "activation_cosine_distance", C_VIOLET, "Cosine distance",
        "Rainbow — Activation Cosine Distance",
        0.0, 0.06, out_dir, "rainbow_cosine_distance.pdf", dpi,
        clusters=RAINBOW_CLUSTERS,
    ))

    return [p for p in paths if p]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output"),
                        help="Output directory (default: shared/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = (os.path.join(_ROOT, args.ppo_root)
                if not os.path.isabs(args.ppo_root) else args.ppo_root)
    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(ppo_root, rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
