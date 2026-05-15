"""dissertation_graphs/shared/plot_rq1_cross_algorithm.py

RQ1 dissertation graph: PPO vs Rainbow opposition score, both algorithms
on the same axes (mean +/- 1 std across 5 seeds each).

Colour convention:
  PPO     -- _C_UNIFORM (#2c7bb6, dark blue)  consistent with PPO RQ graphs
  Rainbow -- C_RED      (#d62020, red)          consistent with Rainbow RQ graphs

Usage:
    python dissertation_graphs/shared/plot_rq1_cross_algorithm.py \\
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
    C_RED,
    _C_UNIFORM,
    METRIC_MARKERS,
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



def plot(ppo_root: str, rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150):
    ppo_results = _load_all_seed_results(ppo_root, "ppo")
    rbw_results = _load_all_seed_results(rainbow_root, "rainbow")

    if not ppo_results:
        print("ERROR: no PPO seed data found.")
        return None
    if not rbw_results:
        print("ERROR: no Rainbow seed data found.")
        return None

    # Build unified step grid spanning both algorithms
    combined = {f"ppo_{k}": v for k, v in ppo_results.items()}
    combined.update({f"rbw_{k}": v for k, v in rbw_results.items()})
    step_grid = _build_step_grid(combined)

    ppo_mean, ppo_std, ppo_n = _avg_metric_across_seeds(
        ppo_results, "opposition_score", step_grid)
    rbw_mean, rbw_std, rbw_n = _avg_metric_across_seeds(
        rbw_results, "opposition_score", step_grid)

    mk_ppo, ms_ppo = METRIC_MARKERS["g_uniform"]        # diamond — PPO
    mk_rbw, ms_rbw = METRIC_MARKERS["opposition_score"]  # circle  — Rainbow

    fig, ax = plt.subplots(figsize=(11, 4.5))

    _make_shaded_line(ax, step_grid, ppo_mean, ppo_std,
                      color=_C_UNIFORM, label="PPO",
                      marker=mk_ppo, markersize=ms_ppo)
    _make_shaded_line(ax, step_grid, rbw_mean, rbw_std,
                      color=C_RED, label="Rainbow",
                      marker=mk_rbw, markersize=ms_rbw)

    ax.axhline(0.0, color="grey", linestyle=":", lw=0.8, alpha=0.5)
    ax.axhline(1.0, color="grey", linestyle=":", lw=0.8, alpha=0.3)
    ax.set_ylabel("Opposition Score  (cosine similarity)")
    _rq_fmt_millions(ax)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, ppo_n)
    out_dir = os.path.join(_GRAPHS_ROOT, "7.2.1")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rq1_cross_algorithm_opposition.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True,
                        help="Path to PPO experiment root (e.g. ppo_experiment_root)")
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--out_dir", default=os.path.join(_HERE, "output"),
                        help="Directory to save the PNG (default: shared/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = os.path.join(_ROOT, args.ppo_root) if not os.path.isabs(args.ppo_root) else args.ppo_root
    rbw_root = os.path.join(_ROOT, args.rainbow_root) if not os.path.isabs(args.rainbow_root) else args.rainbow_root
    plot(ppo_root, rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
