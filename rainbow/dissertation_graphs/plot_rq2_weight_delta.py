"""dissertation_graphs/rainbow/plot_rq2_weight_delta.py

RQ2 dissertation graph: cos(G_variant, delta_theta) weight-delta validation
for Rainbow, averaged across 5 seeds.

Output: GRAPHS/Dissertation graphs/A.1/weight_delta_rainbow.pdf

Colours (consistent with shared/graphing.py):
  G_uniform -- _C_UNIFORM (#2c7bb6)
  G_IS      -- _C_IS      (#7b2d8b)
  G_reward  -- _C_REWARD  (#c0507a)

Usage:
    python dissertation_graphs/rainbow/plot_rq2_weight_delta.py \\
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
    _rq_fmt_millions,
    _add_seed_note,
    _C_UNIFORM,
    _C_IS,
    _C_REWARD,
    METRIC_MARKERS,
    MARKER_EVERY,
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
        out[int(m.group(1))] = _load_checkpoint_results(
            os.path.join(experiment_root, name), algorithm
        )
    return out


def plot(rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150):
    results = _load_all_seed_results(rainbow_root, "rainbow")
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return None

    step_grid = _build_step_grid(results)

    mk_u, ms_u = METRIC_MARKERS["g_uniform"]
    mk_i, ms_i = METRIC_MARKERS["g_is"]
    mk_r, ms_r = METRIC_MARKERS["g_reward"]

    series = [
        ("cos_uniform_success_delta", "cos(G_uniform, Δθ)", _C_UNIFORM, "-",   mk_u, ms_u),
        ("cos_is_success_delta",      "cos(G_IS, Δθ)",      _C_IS,      "--",  mk_i, ms_i),
        ("cos_reward_success_delta",  "cos(G_reward, Δθ)",  _C_REWARD,  "-.",  mk_r, ms_r),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_seeds = None

    for key, label, color, ls, mk, ms in series:
        mean, std, n = _avg_metric_across_seeds(results, key, step_grid)
        fin = np.isfinite(mean)
        if not fin.any():
            print(f"  WARNING: no data for {key} — skipping")
            continue
        if n_seeds is None:
            n_seeds = n
        ax.plot(step_grid[fin], mean[fin], color=color, lw=2.0, linestyle=ls,
                marker=mk if MARKER_EVERY else None, markersize=ms, markevery=MARKER_EVERY or 1, label=label)
        ax.fill_between(step_grid[fin],
                        (mean - std)[fin], (mean + std)[fin],
                        color=color, alpha=0.15)

    ax.axhline(0.0, color="grey", linestyle="-", lw=1.0, alpha=0.5)
    ax.set_ylabel("cos(G_variant, Δθ)")
    _rq_fmt_millions(ax)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, n_seeds or 0)
    out_dir = os.path.join(_GRAPHS_ROOT, "A.1")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "weight_delta_rainbow.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(rbw_root, dpi=args.dpi)


if __name__ == "__main__":
    main()
