"""dissertation_graphs/ablations/ablation_growing_rsa.py

Ablation 4: Growing-Stimulus RSA — frozen vs growing design for Fighting and Crafting groups.

Output: GRAPHS/Ablation graphs/Rainbow/abl_growing_rsa_rainbow.pdf

Four lines:
  Frozen  Fighting  (solid   C_RSA_FIGHTING, marker o)
  Frozen  Crafting  (solid   C_RSA_CRAFTING, marker ^)
  Growing Fighting  (dashed  C_RSA_FIGHTING, marker o)
  Growing Crafting  (dashed  C_RSA_CRAFTING, marker ^)

Data: rainbow_root (frozen, main experiment) + rainbow_v2_root/frozen_rsa/ (growing variant).

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ablations/ablation_growing_rsa.py \\
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
    _avg_metric_across_seeds,
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    C_RSA_FIGHTING,
    C_RSA_CRAFTING,
    METRIC_MARKERS,
    RAINBOW_CLUSTERS,
    MARKER_EVERY,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Ablation graphs", "Rainbow")


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


def plot(rainbow_root: str, rainbow_v2_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> str:
    frozen_results = _load_all_seed_results(rainbow_root)
    growing_root = os.path.join(rainbow_v2_root, "frozen_rsa")
    growing_results = _load_all_seed_results(growing_root)

    if not frozen_results and not growing_results:
        print("WARNING: no data for growing RSA ablation — skipping")
        return ""

    mk_fight, ms_fight = METRIC_MARKERS["rsa_fighting"]
    mk_craft, ms_craft = METRIC_MARKERS["rsa_crafting"]

    # Four distinct colours — one per line
    _C_FROZEN_FIGHT  = C_RSA_FIGHTING   # red
    _C_GROWING_FIGHT = "#f08030"         # orange
    _C_FROZEN_CRAFT  = C_RSA_CRAFTING   # indigo
    _C_GROWING_CRAFT = "#20a050"         # green

    fig, ax = plt.subplots(figsize=(12, 4.5))

    lines = [
        (frozen_results,  "rsa_alignment_fighting", _C_FROZEN_FIGHT,  mk_fight, ms_fight, "Fighting (frozen)",  "-"),
        (frozen_results,  "rsa_alignment_crafting", _C_FROZEN_CRAFT,  mk_craft, ms_craft, "Crafting (frozen)",  "-"),
        (growing_results, "rsa_alignment_fighting", _C_GROWING_FIGHT, mk_fight, ms_fight, "Fighting (growing)", "--"),
        (growing_results, "rsa_alignment_crafting", _C_GROWING_CRAFT, mk_craft, ms_craft, "Crafting (growing)", "--"),
    ]

    for results, field, color, mk, ms, label, ls in lines:
        if not results:
            continue
        step_grid = _build_step_grid(results)
        mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                              color=color, label=label,
                              linestyle=ls, marker=mk, markersize=ms)

    ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    _rsa_ach = _ach_steps_from_all_results(frozen_results or growing_results)
    ach_handles = _add_all_achievement_markers(ax, _rsa_ach, clusters=RAINBOW_CLUSTERS)
    ax.set_ylabel("RSA Alignment (Spearman ρ)")
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=4)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, max(len(frozen_results), len(growing_results)))
    os.makedirs(_GRAPHS_ROOT, exist_ok=True)
    path = os.path.join(_GRAPHS_ROOT, "abl_growing_rsa_rainbow.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


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
