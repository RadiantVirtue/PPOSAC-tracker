"""dissertation_graphs/ppo/plot_all_metrics.py

Per-metric timeseries for PPO.

Produces exactly 4 PDFs (the only ones referenced in the dissertation):
  GRAPHS/Dissertation graphs/7.2.2/coherence_success_ppo.pdf
  GRAPHS/Dissertation graphs/7.2.2/coherence_failure_ppo.pdf
  GRAPHS/Dissertation graphs/7.2.3/activation_separation_ppo.pdf
  GRAPHS/Dissertation graphs/7.2.3/activation_cosine_dist_ppo.pdf

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ppo/plot_all_metrics.py --ppo_root ppo_experiment_root
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
    _build_step_grid,
    _avg_metric_across_seeds,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _rq_fmt_millions,
    _make_shaded_line,
    _add_seed_note,
    C_ORANGE_LIGHT,
    C_COHFAIL,
    C_INDIGO,
    C_VIOLET,
    METRIC_MARKERS,
    MARKER_EVERY,
    PPO_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_ALGORITHM = "ppo"

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs")

_SCALAR_METRICS = [
    # (field_key, colour, marker_key, y_label, output_stem, subdir)
    ("coherence_success",          C_ORANGE_LIGHT, "coherence_success", "Coherence (success group)",    "coherence_success_ppo",   "7.2.2"),
    ("coherence_failure",          C_COHFAIL,      "coherence_failure", "Coherence (failure group)",    "coherence_failure_ppo",   "7.2.2"),
    ("activation_separation",      C_INDIGO,       None,                "Activation Separation",        "activation_separation_ppo", "7.2.3"),
    ("activation_cosine_distance", C_VIOLET,       None,                "Activation Cosine Distance",   "activation_cosine_dist_ppo", "7.2.3"),
]


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


def _plot_scalar(results, step_grid, ach_steps, field, color, marker_key,
                 ylabel, stem, out_dir, dpi) -> str | None:
    mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
    fin = np.isfinite(mean)
    if not fin.any():
        print(f"  skip {field!r} — no finite data")
        return None

    marker, ms = METRIC_MARKERS.get(marker_key, (None, 6)) if marker_key else (None, 6)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin], color,
                      label=ylabel, marker=marker, markersize=ms)
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")

    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=PPO_CLUSTERS)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    _rq_fmt_millions(ax)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, n)
    return _save_fig(fig, out_dir, f"{stem}.pdf", dpi)


def plot(ppo_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list:
    results = _load_all_seed_results(ppo_root, _ALGORITHM)
    if not results:
        print("ERROR: no PPO seed data found.")
        return []

    step_grid = _build_step_grid(results)
    ach_steps = _ach_steps_from_all_results(results)
    paths = []

    for field, color, marker_key, ylabel, stem, subdir in _SCALAR_METRICS:
        out_dir = os.path.join(_GRAPHS_ROOT, subdir)
        p = _plot_scalar(results, step_grid, ach_steps,
                         field, color, marker_key, ylabel, stem, out_dir, dpi)
        if p:
            paths.append(p)

    print(f"  Total: {len(paths)} graphs saved.")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    ppo_root = (os.path.join(_ROOT, args.ppo_root)
                if not os.path.isabs(args.ppo_root) else args.ppo_root)
    plot(ppo_root, dpi=args.dpi)


if __name__ == "__main__":
    main()
