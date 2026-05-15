"""dissertation_graphs/rainbow/plot_cos_is_reward.py

Combined cos(G_IS, G_reward) figure: success and failure groups on shared axes.

Output: rainbow/output/cos_is_reward_rainbow.pdf
  X: step 0–3M
  Y: −0.5 to +1.0
  Two lines: success (purple), failure (pink)

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_cos_is_reward.py \\
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
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _add_seed_note,
    _C_IS,
    _C_REWARD,
    METRIC_MARKERS,
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
        try:
            rec = load_analysis_results(os.path.join(log_dir, fname))
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


def plot(rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> str | None:
    results = _load_all_seed_results(rainbow_root, "rainbow")
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return None

    step_grid = _build_step_grid(results)
    ach_steps = _ach_steps_from_all_results(results)

    mean_s, std_s, n_s = _avg_metric_across_seeds(results, "cos_is_reward_success", step_grid)
    mean_f, std_f, n_f = _avg_metric_across_seeds(results, "cos_is_reward_failure", step_grid)

    fin_s = np.isfinite(mean_s)
    fin_f = np.isfinite(mean_f)
    if not fin_s.any() and not fin_f.any():
        print("  skip cos_is_reward - no finite data")
        return None

    mk_i, ms_i = METRIC_MARKERS["g_is"]
    mk_r, ms_r = METRIC_MARKERS["g_reward"]

    fig, ax = plt.subplots(figsize=(12, 5.0))

    if fin_s.any():
        _make_shaded_line(ax, step_grid[fin_s], mean_s[fin_s], std_s[fin_s],
                          color=_C_IS, label="cos(G_IS, G_reward) success",
                          marker=mk_i, markersize=ms_i)
    if fin_f.any():
        _make_shaded_line(ax, step_grid[fin_f], mean_f[fin_f], std_f[fin_f],
                          color="#c8a000", label="cos(G_IS, G_reward) failure",
                          marker=mk_r, markersize=ms_r)

    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ax.axhline(1.0, color="grey", lw=0.6, alpha=0.3, linestyle=":")
    ax.set_ylim(-0.5, 1.0)

    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=RAINBOW_CLUSTERS)
    _rq_fmt_millions(ax)
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14,
                     loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), ncol=4)

    _add_seed_note(fig, n_s or n_f)
    out_dir = os.path.join(_GRAPHS_ROOT, "7.3")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cos_is_reward_rainbow.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--out_dir", default=os.path.join(_HERE, "output"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    root = (os.path.join(_ROOT, args.rainbow_root)
            if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
