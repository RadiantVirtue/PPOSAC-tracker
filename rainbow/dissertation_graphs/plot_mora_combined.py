"""dissertation_graphs/rainbow/plot_mora_combined.py

Two combined MORA figures (full training run: 0–3M steps):

  GRAPHS/Dissertation graphs/7.4.1/mora_grad_mag_all_rainbow.pdf
    Three lines: gradient_magnitude for r>0, r=0, r<0 subgroups.

  GRAPHS/Dissertation graphs/7.4.2/mora_opp_joint_panel.pdf
    Two lines: opp(r>0, failure) and opp(r=0, failure).
    Y: −0.6 to +1.0.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_mora_combined.py \\
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
    _avg_mora_series,
    _build_step_grid,
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _C_POS_MOR,
    _C_NEU_MOR,
    _C_NEG_MOR,
    METRIC_MARKERS,
    MARKER_EVERY,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_START_STEP = 0
_GRAPHS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "..", "GRAPHS", "Dissertation graphs")


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


def _restricted_step_grid(results: dict, start: int = _START_STEP, n: int = 200) -> np.ndarray:
    """Step grid clamped to [start, max_step]."""
    full = _build_step_grid(results)
    end = float(full[-1]) if full.size else 3_000_000.0
    return np.linspace(start, end, n)


def _save_fig(fig, out_dir: str, fname: str, dpi: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _finish(ax, ach_handles=None):
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.18), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), ncol=4)


def plot(rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    results = _load_all_seed_results(rainbow_root, "rainbow")
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return []

    step_grid = _restricted_step_grid(results)
    ach_steps = _ach_steps_from_all_results(results)
    paths = []

    mean_pos, std_pos, n_pos = _avg_mora_series(
        results, "gradient_magnitude_positive", step_grid)
    mean_neu, std_neu, n_neu = _avg_mora_series(
        results, "gradient_magnitude_neutral", step_grid)
    mean_neg, std_neg, n_neg = _avg_mora_series(
        results, "gradient_magnitude_negative", step_grid)

    mk_pos, ms_pos = METRIC_MARKERS["mora_pos"]
    mk_neu, ms_neu = METRIC_MARKERS["mora_neu"]
    mk_neg, ms_neg = METRIC_MARKERS["mora_neg"]

    fig_mag, ax_mag = plt.subplots(figsize=(12, 4.5))
    for mean, std, n, color, label, mk, ms in [
        (mean_pos, std_pos, n_pos, _C_POS_MOR, "r > 0", mk_pos, ms_pos),
        (mean_neg, std_neg, n_neg, _C_NEG_MOR, "r < 0", mk_neg, ms_neg),
        (mean_neu, std_neu, n_neu, _C_NEU_MOR, "r = 0", mk_neu, ms_neu),
    ]:
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax_mag, step_grid[fin], mean[fin], std[fin],
                              color=color, label=label, marker=mk, markersize=ms)

    ax_mag.set_yscale("log")
    ax_mag.set_ylabel("Gradient magnitude (log scale)")
    ach_handles_mag = _add_all_achievement_markers(ax_mag, ach_steps, clusters=RAINBOW_CLUSTERS)
    _finish(ax_mag, ach_handles=ach_handles_mag)
    _add_seed_note(fig_mag, n_pos)
    out_741 = os.path.normpath(os.path.join(_GRAPHS_ROOT, "7.4.1"))
    paths.append(_save_fig(fig_mag, out_741, "mora_grad_mag_all_rainbow.pdf", dpi))

    mean_pf, std_pf, n_pf = _avg_mora_series(
        results, "opp_pos_vs_failure", step_grid)
    mean_nf, std_nf, n_nf = _avg_mora_series(
        results, "opp_neutral_vs_failure", step_grid)

    fig_opp, ax_opp = plt.subplots(figsize=(12, 4.5))
    for mean, std, n, color, label, mk, ms in [
        (mean_nf, std_nf, n_nf, _C_NEU_MOR, "opp(r=0, failure)", mk_neu, ms_neu),
        (mean_pf, std_pf, n_pf, _C_POS_MOR, "opp(r>0, failure)", mk_pos, ms_pos),
    ]:
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax_opp, step_grid[fin], mean[fin], std[fin],
                              color=color, label=label, marker=mk, markersize=ms)

    ax_opp.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ax_opp.set_ylim(-0.6, 1.0)
    ax_opp.set_ylabel("Cosine similarity")
    ach_handles_opp = _add_all_achievement_markers(ax_opp, ach_steps, clusters=RAINBOW_CLUSTERS)
    _finish(ax_opp, ach_handles=ach_handles_opp)
    _add_seed_note(fig_opp, n_pf)
    out_742 = os.path.normpath(os.path.join(_GRAPHS_ROOT, "7.4.2"))
    paths.append(_save_fig(fig_opp, out_742, "mora_opp_joint_panel.pdf", dpi))

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    root = (os.path.join(_ROOT, args.rainbow_root)
            if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(root, dpi=args.dpi)


if __name__ == "__main__":
    main()
