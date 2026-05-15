"""dissertation_graphs/ablations/ablation_ppo_cross_seed.py

Ablation 3: PPO Cross-Seed Gradient Similarity — individual seed lines.

Outputs:
  GRAPHS/Ablation graphs/PPO/abl_ppo_cross_seed_opposition.pdf
  GRAPHS/Ablation graphs/PPO/abl_ppo_cross_seed_coherence.pdf

Each plot shows 5 individual seed lines (tab10 palette), no mean band.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ablations/ablation_ppo_cross_seed.py \\
        --ppo_root ppo_experiment_root
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
    _rq_fmt_millions,
    _add_seed_note,
    _parse_step,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    seed_color,
    PPO_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Ablation graphs", "PPO")


def _load_all_seed_results(experiment_root: str, algorithm: str = "ppo") -> dict:
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


def _per_seed_series(all_results: dict, field: str) -> dict[int, tuple[list, list]]:
    out = {}
    for seed_id, checkpoint_results in all_results.items():
        steps, vals = [], []
        for label, rec in checkpoint_results:
            step = _parse_step(label)
            if step is None:
                continue
            v = rec.get(field)
            if v is not None:
                try:
                    steps.append(step)
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if steps:
            paired = sorted(zip(steps, vals))
            steps, vals = zip(*paired)
            out[seed_id] = (list(steps), list(vals))
    return out


def _plot_per_seed(all_results: dict, field: str, ylabel: str, fname: str, dpi: int) -> str:
    series = _per_seed_series(all_results, field)
    if not series:
        print(f"  skip {field!r} — no per-seed data")
        return ""

    step_grid = _build_step_grid(all_results)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (seed_id, (steps, vals)) in enumerate(sorted(series.items())):
        color = seed_color(i)
        interp = np.interp(step_grid, steps, vals, left=np.nan, right=np.nan)
        fin = np.isfinite(interp)
        ax.plot(step_grid[fin], interp[fin],
                color=color, lw=1.5, alpha=0.85, label=f"seed {seed_id}")

    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ach_handles = _add_all_achievement_markers(
        ax, _ach_steps_from_all_results(all_results), clusters=PPO_CLUSTERS)
    ax.set_ylabel(ylabel)
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=5)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, len(series))
    os.makedirs(_GRAPHS_ROOT, exist_ok=True)
    path = os.path.join(_GRAPHS_ROOT, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot(ppo_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    all_results = _load_all_seed_results(ppo_root)
    if not all_results:
        print("WARNING: no PPO data for cross-seed ablation — skipping")
        return []

    paths = []
    p = _plot_per_seed(all_results, "opposition_score",
                       "Opposition Score", "abl_ppo_cross_seed_opposition.pdf", dpi)
    if p:
        paths.append(p)

    p = _plot_per_seed(all_results, "coherence_success",
                       "Coherence (success group)", "abl_ppo_cross_seed_coherence.pdf", dpi)
    if p:
        paths.append(p)

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    ppo = (os.path.join(_ROOT, args.ppo_root)
           if not os.path.isabs(args.ppo_root) else args.ppo_root)
    plot(ppo, dpi=args.dpi)


if __name__ == "__main__":
    main()
