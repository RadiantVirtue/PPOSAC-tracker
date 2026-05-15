"""dissertation_graphs/shared/plot_make_stone_sword_case_study.py

make_stone_sword case study: opposition_score, coherence_success, coherence_failure
centred at the make_stone_sword milestone step.

Two output PDFs (one per algorithm):
  shared/output/make_stone_sword_case_study_ppo.pdf
  shared/output/make_stone_sword_case_study_rainbow.pdf

Each panel shows three lines:
  opposition_score   (C_RED)
  coherence_success  (C_ORANGE_LIGHT)
  coherence_failure  (C_ORANGE_DARK)

Nine data points at offsets: -200k, -150k, -100k, -50k, 0, +50k, +100k, +150k, +200k.
Per-seed normalisation: subtract value at offset=0 then average across seeds.
Y fixed at [-0.5, +0.5].  Dashed reference line at y=0.
Nearby achievements within ±200k are annotated (clump_window = 10 000 steps).

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/shared/plot_make_stone_sword_case_study.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_v2
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _ach_steps_from_results,
    _parse_step,
    _cluster_achievements,
    _add_seed_note,
    C_RED,
    C_ORANGE_LIGHT,
    C_COHFAIL,
    METRIC_MARKERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs")

_FOCUS_ACH = "make_stone_sword"
_OFFSETS = [-200_000, -150_000, -100_000, -50_000, 0, 50_000, 100_000, 150_000, 200_000]
_CENTER_IDX = 4
_CLUMP_WINDOW = 10_000



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


def _aggregate_field(all_seed_results: dict, field: str) -> tuple[np.ndarray, np.ndarray, int]:
    arrs = []
    for seed_results in all_seed_results.values():
        ach_steps = _ach_steps_from_results(seed_results)
        center = ach_steps.get(_FOCUS_ACH)
        if center is None:
            continue
        pts = _compute_9pts(seed_results, center, field)
        if pts is not None:
            arrs.append(pts)
    if not arrs:
        nan9 = np.full(9, np.nan)
        return nan9, nan9, 0
    mat = np.vstack(arrs)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), len(arrs)



def _gather_nearby_rel_steps(all_seed_results: dict) -> dict[str, float]:
    from collections import defaultdict as _dd
    ach_rel: dict[str, list[float]] = _dd(list)
    for seed_results in all_seed_results.values():
        ach_steps = _ach_steps_from_results(seed_results)
        center = ach_steps.get(_FOCUS_ACH)
        if center is None:
            continue
        for ach, step in ach_steps.items():
            if ach == _FOCUS_ACH:
                continue
            rel = step - center
            if -200_000 <= rel <= 200_000:
                ach_rel[ach].append(rel)
    return {a: float(np.mean(v)) for a, v in ach_rel.items()}


def _add_nearby_markers(ax, nearby_rel_steps: dict) -> tuple[list, list[str]]:
    if not nearby_rel_steps:
        return [], []
    _GRAY = "#888888"
    ach_int = {a: int(s) for a, s in nearby_rel_steps.items()}
    groups = _cluster_achievements(ach_int, clump_window=_CLUMP_WINDOW)
    cluster_idx = 0
    for group in groups:
        mid = float(np.median([nearby_rel_steps.get(a, 0) for a, _ in group]))
        ax.axvline(mid, color=_GRAY, linestyle=":", linewidth=0.8, alpha=0.55, zorder=1)
        if len(group) > 1:
            cluster_idx += 1
            ax.text(mid, 1.0, f"C{cluster_idx}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9, color="#555555", ha="center", va="bottom", zorder=5)
    return [], []



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


def _plot_panel(all_seed_results: dict, algorithm: str, _out_dir_ignored: str, dpi: int) -> str:
    m_opp, s_opp, n_opp = _aggregate_field(all_seed_results, "opposition_score")
    m_cs,  s_cs,  n_cs  = _aggregate_field(all_seed_results, "coherence_success")
    m_cf,  s_cf,  n_cf  = _aggregate_field(all_seed_results, "coherence_failure")

    if not (np.isfinite(m_opp).any() or np.isfinite(m_cs).any() or np.isfinite(m_cf).any()):
        print(f"  WARNING: no data for {algorithm} make_stone_sword case study - skipping")
        return ""

    nearby = _gather_nearby_rel_steps(all_seed_results)
    offsets_arr = np.array(_OFFSETS)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for means, stds, n, color, marker, label in [
        (m_opp, s_opp, n_opp, C_RED,          "o", "Opposition score"),
        (m_cs,  s_cs,  n_cs,  C_ORANGE_LIGHT, "s", "Coherence (success)"),
        (m_cf,  s_cf,  n_cf,  C_COHFAIL,      "^", "Coherence (failure)"),
    ]:
        fin = np.isfinite(means)
        if not fin.any():
            continue
        ax.fill_between(offsets_arr[fin],
                        (means - stds)[fin], (means + stds)[fin],
                        alpha=0.2, color=color)
        ax.plot(offsets_arr[fin], means[fin], f"{marker}-",
                color=color, lw=2.0, markersize=7, label=label)

    ax.axhline(0.0, color="#888888", lw=0.8, alpha=0.5, linestyle="--")
    ax.axvline(0, color="#555555", lw=1.4, ls="--", zorder=2)

    _add_nearby_markers(ax, nearby)
    focus_handle = mlines.Line2D([], [], color="#555555", linestyle="--", linewidth=1.4,
                                 label="make stone sword")

    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(_OFFSETS)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_x_formatter))
    ax.set_xlabel("Steps relative to make_stone_sword")
    ax.set_ylabel("Metric delta (normalised)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    existing_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=existing_handles + [focus_handle], fontsize=14,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

    _add_seed_note(fig, n_opp or n_cs or n_cf)
    fname = f"make_stone_sword_case_study_{algorithm}.pdf"
    out_dir = os.path.join(_GRAPHS_ROOT, "7.5.4")
    return _save_fig(fig, out_dir, fname, dpi)



def plot(ppo_root: str, rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    paths = []

    ppo_results = _load_all_seed_results(ppo_root, "ppo")
    if ppo_results:
        p = _plot_panel(ppo_results, "ppo", "", dpi)
        if p:
            paths.append(p)
    else:
        print("WARNING: no PPO seed data found.")

    rbw_results = _load_all_seed_results(rainbow_root, "rainbow")
    if rbw_results:
        p = _plot_panel(rbw_results, "rainbow", "", dpi)
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
