"""dissertation_graphs/ablations/ablation_scalar_dqn.py

Ablation 1: Scalar DQN vs Rainbow DQN — MORA ratio over training.

Output: GRAPHS/Ablation graphs/Rainbow/abl_scalar_dqn_mora_ratio.pdf

Two lines:
  Rainbow (5-seed mean±std)  — C_RED, solid, marker v
  Scalar DQN (1 seed)        — C_COHFAIL, dashed, marker X

Metric: gradient_magnitude_positive / gradient_magnitude_neutral (log scale), steps 1M–3M.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ablations/ablation_scalar_dqn.py \\
        --rainbow_root rainbow_experiment_root \\
        --scalar_json scalar_ablation_results/scalar_ablation_result.json
"""
from __future__ import annotations

import argparse
import json
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
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _parse_step,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    C_RED,
    C_COHFAIL,
    METRIC_MARKERS,
    MARKER_EVERY,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Ablation graphs", "Rainbow")
_START_STEP = 1_000_000


def _load_all_seed_results(experiment_root: str, algorithm: str = "rainbow") -> dict:
    out = {}
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


def _extract_mora_ratio(checkpoint_results: list) -> list[tuple[int, float]]:
    pairs = []
    for label, rec in checkpoint_results:
        step = _parse_step(label)
        if step is None or step < _START_STEP:
            continue
        mor = rec.get("moment_of_reward") or {}
        pos = mor.get("gradient_magnitude_positive")
        neu = mor.get("gradient_magnitude_neutral")
        if pos is None or neu is None:
            continue
        try:
            fp, fn = float(pos), float(neu)
        except (TypeError, ValueError):
            continue
        if fn <= 0 or not math.isfinite(fp) or not math.isfinite(fn):
            continue
        pairs.append((step, fp / fn))
    return pairs


def _avg_ratio(all_seed_results: dict, step_grid: np.ndarray):
    arrs = []
    for results in all_seed_results.values():
        pairs = _extract_mora_ratio(results)
        if len(pairs) < 2:
            continue
        xs, ys = zip(*sorted(pairs))
        arrs.append(np.interp(step_grid, xs, ys, left=np.nan, right=np.nan))
    if not arrs:
        nan = np.full(len(step_grid), np.nan)
        return nan, nan, 0
    mat = np.vstack(arrs)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), len(arrs)


def _scalar_ratio_series(scalar_json: str) -> list[tuple[int, float]]:
    if not os.path.isfile(scalar_json):
        return []
    try:
        with open(scalar_json, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    pairs = []
    for rec in (data if isinstance(data, list) else [data]):
        step = rec.get("global_step") or rec.get("step")
        if step is None or int(step) < _START_STEP:
            continue
        mor = rec.get("moment_of_reward") or {}
        pos = mor.get("gradient_magnitude_positive")
        neu = mor.get("gradient_magnitude_neutral")
        if pos is None or neu is None:
            continue
        try:
            fp, fn = float(pos), float(neu)
        except (TypeError, ValueError):
            continue
        if fn <= 0 or not math.isfinite(fp) or not math.isfinite(fn):
            continue
        pairs.append((int(step), fp / fn))
    return sorted(pairs)


def plot(rainbow_root: str, scalar_json: str, _out_dir_ignored: str = "", dpi: int = 150) -> str:
    all_results = _load_all_seed_results(rainbow_root)
    if not all_results:
        print("WARNING: no Rainbow data for scalar DQN ablation — skipping")
        return ""

    step_grid_full = _build_step_grid(all_results)
    step_grid = step_grid_full[step_grid_full >= _START_STEP]
    if len(step_grid) < 2:
        step_grid = np.linspace(_START_STEP, 3_000_000, 200)

    rbw_mean, rbw_std, n = _avg_ratio(all_results, step_grid)
    scalar_pairs = _scalar_ratio_series(scalar_json)

    mk_pos, ms_pos = METRIC_MARKERS["mora_pos"]
    mk_neu, ms_neu = METRIC_MARKERS["mora_neu"]

    fig, ax = plt.subplots(figsize=(12, 5))

    fin = np.isfinite(rbw_mean) & (rbw_mean > 0)
    if fin.any():
        _make_shaded_line(ax, step_grid[fin], rbw_mean[fin], rbw_std[fin],
                          color=C_RED, label=f"Rainbow ({n} seeds, mean±std)",
                          marker=mk_pos, markersize=ms_pos)

    if scalar_pairs:
        xs, ys = zip(*scalar_pairs)
        ax.plot(xs, ys, color=C_COHFAIL, lw=2.0, ls="--",
                marker=mk_neu if MARKER_EVERY else None, markersize=ms_neu, markevery=MARKER_EVERY or 1,
                label="Scalar DQN (1 seed)")

    ax.set_yscale("log")
    ax.axhline(9.2, color="grey", lw=0.8, ls="--", alpha=0.7, label="Reference ~9.2×")
    ach_handles = _add_all_achievement_markers(
        ax, _ach_steps_from_all_results(all_results), clusters=RAINBOW_CLUSTERS)
    ax.set_ylabel("MORA ratio (r>0 / r=0 gradient magnitude, log scale)")
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, n)
    os.makedirs(_GRAPHS_ROOT, exist_ok=True)
    path = os.path.join(_GRAPHS_ROOT, "abl_scalar_dqn_mora_ratio.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--scalar_json", default=os.path.join(
        _ROOT, "scalar_ablation_results", "scalar_ablation_result.json"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    rbw = (os.path.join(_ROOT, args.rainbow_root)
           if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(rbw, args.scalar_json, dpi=args.dpi)


if __name__ == "__main__":
    main()
