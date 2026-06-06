"""dissertation_graphs/rainbow/plot_mor_ratio_vs_return.py

Fig 7.10 - Rainbow MoR magnitude ratio (r>0 / r=0) over training.

Single panel, log y-axis.  Shows mean ± 1 std across seeds plus individual
seed lines, with a dashed reference line at 9× and achievement markers.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_mor_ratio_vs_return.py \\
        --rainbow_root rainbow_experiment_root
"""
from __future__ import annotations

import argparse
import math
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
    _build_step_grid,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _rq_fmt_millions,
    _make_shaded_line,
    _add_seed_note,
    _parse_step,
    C_RED,
    seed_color,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_ALGORITHM = "rainbow"
_REFERENCE_RATIO = 9.0
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



def _extract_ratio_series(checkpoint_results: list) -> list[tuple[int, float]]:
    pairs = []
    for label, record in checkpoint_results:
        step = _parse_step(label)
        if step is None:
            continue
        mor = record.get("moment_of_reward") or {}
        mag_pos = mor.get("gradient_magnitude_positive")
        mag_neu = mor.get("gradient_magnitude_neutral")
        if (mag_pos is not None and mag_neu is not None
                and math.isfinite(float(mag_pos))
                and math.isfinite(float(mag_neu))
                and float(mag_neu) > 0):
            pairs.append((step, float(mag_pos) / float(mag_neu)))
    return pairs



def plot(rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> str | None:
    all_seed_results = _load_all_seed_results(rainbow_root, _ALGORITHM)
    if not all_seed_results:
        print("ERROR: no Rainbow seed data found.")
        return None

    step_grid = _build_step_grid(all_seed_results)
    ach_steps = _ach_steps_from_all_results(all_seed_results)

    ratio_arrs = []
    seed_series = []

    for seed_id, checkpoint_results in sorted(all_seed_results.items()):
        pairs = _extract_ratio_series(checkpoint_results)
        if len(pairs) < 2:
            print(f"  seed {seed_id}: insufficient ratio data ({len(pairs)} points)")
            continue
        xs, ys = zip(*sorted(pairs))
        interp = np.interp(step_grid, xs, ys, left=np.nan, right=np.nan)
        ratio_arrs.append(interp)
        seed_series.append((seed_id, interp))

    if not ratio_arrs:
        print("ERROR: no ratio data to plot.")
        return None

    ratio_mat  = np.vstack(ratio_arrs)
    ratio_mean = np.nanmean(ratio_mat, axis=0)
    ratio_std  = np.nanstd(ratio_mat,  axis=0)
    n = ratio_mat.shape[0]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Per-seed lines
    for i, (seed_id, interp) in enumerate(seed_series):
        fin = np.isfinite(interp) & (interp > 0)
        ax.plot(step_grid[fin], interp[fin],
                color=seed_color(i), lw=0.8, alpha=0.45,
                label=f"seed {seed_id}")

    # Mean ± 1 std
    fin = np.isfinite(ratio_mean) & (ratio_mean > 0)
    _make_shaded_line(ax, step_grid[fin], ratio_mean[fin], ratio_std[fin],
                      C_RED, label="Mean ± 1 std", lw=2.0)

    # Reference line at 33×
    ax.axhline(_REFERENCE_RATIO, color="grey", linestyle="--", lw=1.2,
               alpha=0.8, label=f"{_REFERENCE_RATIO:.0f}× reference")

    ax.set_yscale("log")
    ax.set_ylabel("Grad Magnitude Ratio  r>0 / r=0  (log scale)")
    _rq_fmt_millions(ax)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=RAINBOW_CLUSTERS)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.28), ncol=4)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.5), ncol=4)

    _add_seed_note(fig, n)
    out_dir = os.path.join(_GRAPHS_ROOT, "7.4.1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mor_ratio_rainbow.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output"),
                        help="Output directory (default: rainbow/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
