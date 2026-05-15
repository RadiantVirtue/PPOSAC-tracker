"""dissertation_graphs/ablations/ablation_fixed_threshold.py

Ablation 2: Fixed-Threshold Partitioning vs Percentile Partitioning.

Outputs:
  GRAPHS/Ablation graphs/Rainbow/abl_fixed_threshold_mora_opp.pdf
  GRAPHS/Ablation graphs/Rainbow/abl_fixed_threshold_mora_bar.pdf

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ablations/ablation_fixed_threshold.py \\
        --rainbow_root rainbow_experiment_root \\
        --rainbow_v2_root rainbow_v2
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
    _avg_mora_series,
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    C_RED,
    C_COHFAIL,
    _C_UNIFORM,
    C_ORANGE_LIGHT,
    METRIC_MARKERS,
    MARKER_EVERY,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Ablation graphs", "Rainbow")
_MORA_STEPS  = [1_850_000, 2_300_000, 2_750_000]
_MORA_LABELS = ["1.85M", "2.30M", "2.75M"]
_START_STEP  = 1_000_000


def _load_all_seed_results(experiment_root: str, algorithm: str = "rainbow") -> dict:
    out = {}
    if not os.path.isdir(experiment_root):
        return out
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if not m:
            continue
        seed_dir = os.path.join(experiment_root, name)
        log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
        if not os.path.isdir(log_dir):
            continue
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
        out[int(m.group(1))] = results
    return out


def _mora_ratio_at_step(seed_results: list, target_step: int):
    best_dist = float("inf")
    best = None
    for label, rec in seed_results:
        m = re.search(r"\d+", label)
        if not m:
            continue
        step = int(m.group())
        mor = rec.get("moment_of_reward") or {}
        pos = mor.get("gradient_magnitude_positive")
        neu = mor.get("gradient_magnitude_neutral")
        if pos is None or neu is None:
            continue
        try:
            fp, fn = float(pos), float(neu)
        except (TypeError, ValueError):
            continue
        if fn == 0:
            continue
        dist = abs(step - target_step)
        if dist < best_dist:
            best_dist = dist
            best = fp / fn
    return best


def plot_mora_opp(pct_results: dict, fixed_results: dict, out_dir: str, dpi: int) -> str:
    """abl_fixed_threshold_mora_opp.pdf — opp(r>0, failure) comparison."""
    if not pct_results and not fixed_results:
        print("WARNING: no data for fixed-threshold opp plot — skipping")
        return ""

    # Build step grids
    all_results = {**pct_results}
    step_grid = _build_step_grid(all_results)
    step_grid = step_grid[step_grid >= _START_STEP] if step_grid.size else np.linspace(_START_STEP, 3e6, 200)

    mk_pos, ms_pos = METRIC_MARKERS["mora_pos"]
    mk_neu, ms_neu = METRIC_MARKERS["mora_neu"]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    if pct_results:
        mean, std, n = _avg_mora_series(pct_results, "opp_pos_vs_failure", step_grid)
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                              color=C_RED, label=f"Percentile ({n} seeds)",
                              marker=mk_pos, markersize=ms_pos)

    if fixed_results:
        # Fixed threshold — single seed, just plot the line without std
        mean_f, std_f, n_f = _avg_mora_series(fixed_results, "opp_pos_vs_failure", step_grid)
        fin_f = np.isfinite(mean_f)
        if fin_f.any():
            ax.plot(step_grid[fin_f], mean_f[fin_f], color=C_COHFAIL, lw=2.0,
                    marker=mk_neu if MARKER_EVERY else None, markersize=ms_neu, markevery=MARKER_EVERY or 1,
                    label="Fixed threshold (EPS>11.9, 1 seed)")

    ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(-0.6, 1.0)
    ach_handles = _add_all_achievement_markers(
        ax, _ach_steps_from_all_results(pct_results), clusters=RAINBOW_CLUSTERS)
    ax.set_ylabel("opp(r>0, failure)")
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, len(pct_results))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "abl_fixed_threshold_mora_opp.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_mora_bar(pct_results: dict, fixed_seed_results: list, out_dir: str, dpi: int) -> str:
    """abl_fixed_threshold_mora_bar.pdf — grouped bar chart."""
    x = np.arange(len(_MORA_STEPS))
    bar_w = 0.35

    pct_means, pct_stds = [], []
    fix_vals = []
    for tgt in _MORA_STEPS:
        vals = [_mora_ratio_at_step(sr, tgt)
                for sr in pct_results.values()
                if _mora_ratio_at_step(sr, tgt) is not None]
        pct_means.append(np.mean(vals) if vals else np.nan)
        pct_stds.append(np.std(vals) if vals else np.nan)
        fv = _mora_ratio_at_step(fixed_seed_results, tgt)
        fix_vals.append(fv if fv is not None else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)
    ax.bar(x - bar_w / 2, pct_means, bar_w, yerr=pct_stds, capsize=4,
           color=_C_UNIFORM, label=f"Percentile ({len(pct_results)} seeds)", zorder=3)
    ax.bar(x + bar_w / 2, fix_vals, bar_w,
           color=C_ORANGE_LIGHT, label="Fixed threshold (1 seed)", zorder=3)
    ax.axhline(9.2, color="grey", lw=0.8, ls="--", label="Reference ~9.2×", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(_MORA_LABELS)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MORA ratio (r>0 / r=0 gradient magnitude)")
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    _add_seed_note(fig, len(pct_results))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "abl_fixed_threshold_mora_bar.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot(rainbow_root: str, rainbow_v2_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    pct_results = _load_all_seed_results(rainbow_root)

    fixed_dir = os.path.join(rainbow_v2_root, "fixed_threshold")
    fixed_seed1_dir = os.path.join(fixed_dir, "seed_1")
    fixed_results_dict = _load_all_seed_results(fixed_dir)
    # For bar chart, use seed_1 directly
    fixed_seed1_results = list(fixed_results_dict.values())[0] if fixed_results_dict else []

    paths = []
    p = plot_mora_opp(pct_results, fixed_results_dict, _GRAPHS_ROOT, dpi)
    if p:
        paths.append(p)
    p = plot_mora_bar(pct_results, fixed_seed1_results, _GRAPHS_ROOT, dpi)
    if p:
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--rainbow_v2_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    rbw = (os.path.join(_ROOT, args.rainbow_root)
           if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    rbw2 = (os.path.join(_ROOT, args.rainbow_v2_root)
            if not os.path.isabs(args.rainbow_v2_root) else args.rainbow_v2_root)
    plot(rbw, rbw2, dpi=args.dpi)


if __name__ == "__main__":
    main()
