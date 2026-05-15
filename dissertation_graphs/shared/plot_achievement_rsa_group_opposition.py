"""dissertation_graphs/shared/plot_achievement_rsa_group_opposition.py

Achievement-centred opposition score delta, grouped by RSA functional group.

Two output PDFs (one per algorithm):
  shared/output/achievement_rsa_group_opposition_ppo.pdf
  shared/output/achievement_rsa_group_opposition_rainbow.pdf

Each panel shows 4 lines (Fighting, Resource, Crafting, Housing).  Each line
is the group mean across non-cluster achievements that belong to that RSA group,
with ±1 std shading.  "Non-cluster" means the achievement does not share a
clump_window=2% cluster with any other achievement at its mean global step.

Nine data points at offsets: -200k, -150k, -100k, -50k, 0, +50k, +100k, +150k, +200k.
Y fixed at [-0.5, +0.5].  Dashed reference line at y=0.

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/shared/plot_achievement_rsa_group_opposition.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_v2
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _ach_steps_from_results,
    _ach_steps_from_all_results,
    _parse_step,
    _cluster_achievements,
    C_RSA_FIGHTING,
    C_RSA_RESOURCE,
    C_RSA_CRAFTING,
    C_RSA_HOUSING,
    METRIC_MARKERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs")

# RSA group membership - achievement names → group label
_ACH_TO_GROUPS: dict[str, list[str]] = {
    "defeat_zombie":      ["Fighting"],
    "defeat_skeleton":    ["Fighting"],
    "make_wood_sword":    ["Fighting", "Crafting"],
    "make_stone_sword":   ["Fighting", "Crafting"],
    "make_iron_sword":    ["Fighting", "Crafting"],
    "collect_wood":       ["Resource"],
    "collect_stone":      ["Resource"],
    "collect_iron":       ["Resource"],
    "collect_coal":       ["Resource"],
    "make_wood_pickaxe":  ["Resource", "Crafting"],
    "make_stone_pickaxe": ["Resource", "Crafting"],
    "make_iron_pickaxe":  ["Resource", "Crafting"],
    "place_furnace":      ["Crafting", "Housing"],
    "place_table":        ["Housing"],
    "place_stone":        ["Housing"],
    "wake_up":            ["Housing"],
}

_GROUPS = ["Fighting", "Resource", "Crafting", "Housing"]
_GROUP_COLORS = {
    "Fighting": C_RSA_FIGHTING,
    "Resource": C_RSA_RESOURCE,
    "Crafting": C_RSA_CRAFTING,
    "Housing":  C_RSA_HOUSING,
}
_GROUP_MARKERS = {
    "Fighting": METRIC_MARKERS["rsa_fighting"][0],
    "Resource": METRIC_MARKERS["rsa_resource"][0],
    "Crafting": METRIC_MARKERS["rsa_crafting"][0],
    "Housing":  METRIC_MARKERS["rsa_housing"][0],
}

_OFFSETS = [-200_000, -150_000, -100_000, -50_000, 0, 50_000, 100_000, 150_000, 200_000]
_CENTER_IDX = 4  # index of offset=0 in _OFFSETS



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



def _extract_series(seed_results: list, field: str) -> list[tuple[int, float]]:
    pairs = []
    for label, rec in seed_results:
        step = _parse_step(label)
        if step is None:
            continue
        val = rec.get(field)
        if val is not None:
            try:
                pairs.append((step, float(val)))
            except (TypeError, ValueError):
                pass
    pairs.sort()
    return pairs


def _compute_9pts(seed_results: list, center_step: float, field: str) -> np.ndarray | None:
    pairs = _extract_series(seed_results, field)
    if len(pairs) < 2:
        return None
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    abs_offsets = np.array([center_step + o for o in _OFFSETS], dtype=float)
    raw = np.interp(abs_offsets, xs, ys, left=np.nan, right=np.nan)
    in_range = (abs_offsets >= xs[0]) & (abs_offsets <= xs[-1])
    if int(in_range.sum()) < 5:
        return None
    center_val = raw[_CENTER_IDX]
    if not np.isfinite(center_val):
        return None
    return raw - center_val



def _non_cluster_achievements(all_seed_results: dict) -> set[str]:
    """Return the set of achievements that are NOT clumped with any other at 2% threshold."""
    mean_steps = _ach_steps_from_all_results(all_seed_results)
    if not mean_steps:
        return set()
    vals = list(mean_steps.values())
    x_range = max(vals) - min(vals)
    clump_window = max(1, int(0.02 * x_range))
    groups = _cluster_achievements(
        {a: int(s) for a, s in mean_steps.items()},
        clump_window=clump_window,
    )
    return {g[0][0] for g in groups if len(g) == 1}



def _group_means(
    all_seed_results: dict,
    group: str,
    non_cluster: set[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean ± std of opposition_score delta across all eligible achievements in group.

    Returns (means_9, stds_9, n_achievements).
    """
    eligible = [a for a in _ACH_TO_GROUPS
                if group in _ACH_TO_GROUPS[a] and a in non_cluster]
    if not eligible:
        nan9 = np.full(9, np.nan)
        return nan9, nan9, 0

    ach_means: list[np.ndarray] = []
    for ach in eligible:
        seed_arrs = []
        for seed_results in all_seed_results.values():
            ach_steps = _ach_steps_from_results(seed_results)
            center = ach_steps.get(ach)
            if center is None:
                continue
            pts = _compute_9pts(seed_results, center, "opposition_score")
            if pts is not None:
                seed_arrs.append(pts)
        if seed_arrs:
            mat = np.vstack(seed_arrs)
            ach_means.append(np.nanmean(mat, axis=0))

    if not ach_means:
        nan9 = np.full(9, np.nan)
        return nan9, nan9, 0

    all_mat = np.vstack(ach_means)
    return np.nanmean(all_mat, axis=0), np.nanstd(all_mat, axis=0), len(ach_means)



def _save_fig(fig, out_dir: str, fname: str, dpi: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _x_formatter(x, _):
    if x == 0:
        return "0"
    k = int(x / 1_000)
    return f"+{k}k" if k > 0 else f"{k}k"


def _plot_panel(all_seed_results: dict, algorithm: str, dpi: int) -> str:
    non_cluster = _non_cluster_achievements(all_seed_results)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    offsets_arr = np.array(_OFFSETS)

    has_data = False
    for group in _GROUPS:
        means, stds, n_ach = _group_means(all_seed_results, group, non_cluster)
        fin = np.isfinite(means)
        if not fin.any():
            continue
        has_data = True
        color = _GROUP_COLORS[group]
        mk = _GROUP_MARKERS[group]
        ax.fill_between(
            offsets_arr[fin],
            (means - stds)[fin], (means + stds)[fin],
            alpha=0.2, color=color,
        )
        ax.plot(
            offsets_arr[fin], means[fin], f"{mk}-",
            color=color, lw=2.0, markersize=7,
            label=f"{group}  ({n_ach} achievements)",
        )

    if not has_data:
        print(f"  WARNING: no data for {algorithm} RSA group panel - skipping")
        plt.close(fig)
        return ""

    ax.axhline(0.0, color="#888888", lw=0.8, alpha=0.5, linestyle="--")
    ax.axvline(0, color="#555555", lw=1.4, ls="--", zorder=2, label="milestone step")
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(_OFFSETS)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_x_formatter))
    ax.set_xlabel("Steps relative to milestone")
    ax.set_ylabel("Opposition score delta (normalised)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

    fname = f"achievement_rsa_group_opposition_{algorithm}.pdf"
    out_dir = os.path.join(_GRAPHS_ROOT, "7.5.3")
    return _save_fig(fig, out_dir, fname, dpi)



def plot(ppo_root: str, rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    paths = []

    ppo_results = _load_all_seed_results(ppo_root, "ppo")
    if ppo_results:
        p = _plot_panel(ppo_results, "ppo", dpi)
        if p:
            paths.append(p)
    else:
        print("WARNING: no PPO seed data found.")

    rbw_results = _load_all_seed_results(rainbow_root, "rainbow")
    if rbw_results:
        p = _plot_panel(rbw_results, "rainbow", dpi)
        if p:
            paths.append(p)
    else:
        print("WARNING: no Rainbow seed data found.")

    return paths



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output"),
                        help="Output directory (default: shared/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = (os.path.join(_ROOT, args.ppo_root)
                if not os.path.isabs(args.ppo_root) else args.ppo_root)
    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(ppo_root, rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
