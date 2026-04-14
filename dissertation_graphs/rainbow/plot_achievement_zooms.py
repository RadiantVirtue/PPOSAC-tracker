"""dissertation_graphs/rainbow/plot_achievement_zooms.py

Achievement-centred zoom charts for Rainbow.

For each achievement that appears in the milestone data, produces one PNG per
key metric — centred on the mean unlock step, showing a +/- ZOOM_WINDOW-step
window averaged across seeds that reached it.

Key metrics:
  opposition_score, coherence_success, coherence_failure,
  gradient_magnitude_success, gradient_magnitude_failure

Output: rainbow/output/zooms/<achievement>_<metric>.png
  e.g.  rainbow/output/zooms/eat_cow_opposition_score.png

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_achievement_zooms.py --rainbow_root rainbow_experiment_root
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    load_all_data,
    _make_shaded_line,
    C_RED, C_ORANGE_LIGHT, C_ORANGE_DARK,
    C_YELLOW_LIGHT, C_YELLOW_DARK,
)

_ALGORITHM = "rainbow"
_ZOOM_WINDOW = 200_000   # steps either side of unlock

# (field_key, colour, y_label, filename_stem)
_ZOOM_METRICS = [
    ("opposition_score",           C_RED,          "Opposition Score",             "opposition_score"),
    ("coherence_success",          C_ORANGE_LIGHT, "Coherence (success group)",    "coherence_success"),
    ("coherence_failure",          C_ORANGE_DARK,  "Coherence (failure group)",    "coherence_failure"),
    ("gradient_magnitude_success", C_YELLOW_LIGHT, "Gradient Magnitude (success)", "grad_mag_success"),
    ("gradient_magnitude_failure", C_YELLOW_DARK,  "Gradient Magnitude (failure)", "grad_mag_failure"),
]

_REL_GRID = np.linspace(-_ZOOM_WINDOW, _ZOOM_WINDOW, 201)


# ── Episode → step conversion ─────────────────────────────────────────────────

def _build_ep_step_maps(periodic: dict) -> dict[int, tuple]:
    """Return {seed_id: (sorted_episodes_arr, sorted_steps_arr)} from periodic data."""
    raw: dict[int, list] = {}
    for step, entries in periodic.items():
        for seed_id, rec in entries:
            ep = rec.get("episode")
            if ep is None:
                continue
            try:
                raw.setdefault(seed_id, []).append((float(ep), float(step)))
            except (TypeError, ValueError):
                pass
    out = {}
    for sid, pairs in raw.items():
        pairs.sort()
        eps, steps = zip(*pairs)
        out[sid] = (np.array(eps), np.array(steps))
    return out


def _milestone_unlock_steps(milestone_entries: list,
                             ep_step_maps: dict) -> dict[int, float]:
    """Return {seed_id: unlock_step} for a single achievement's milestone entries."""
    result = {}
    for seed_id, rec in milestone_entries:
        ep_m = rec.get("episode")
        if ep_m is None:
            continue
        maps = ep_step_maps.get(seed_id)
        if maps is None:
            continue
        eps_arr, steps_arr = maps
        unlock_step = float(np.interp(float(ep_m), eps_arr, steps_arr))
        result[seed_id] = unlock_step
    return result


# ── Per-seed metric extraction ────────────────────────────────────────────────

def _seed_metric_series(periodic: dict, seed_id: int,
                        field: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, values) for a given seed and scalar field."""
    pts = []
    for step, entries in periodic.items():
        for sid, rec in entries:
            if sid != seed_id:
                continue
            val = rec.get(field)
            if val is not None:
                try:
                    pts.append((float(step), float(val)))
                except (TypeError, ValueError):
                    pass
    if not pts:
        return None
    pts.sort()
    steps, vals = zip(*pts)
    return np.array(steps), np.array(vals)


# ── Zoom computation ──────────────────────────────────────────────────────────

def _compute_zoom(periodic: dict, unlock_steps: dict[int, float],
                  field: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (mean, std, n) on _REL_GRID for a single achievement/metric pair."""
    traces = []
    for seed_id, unlock_step in unlock_steps.items():
        series = _seed_metric_series(periodic, seed_id, field)
        if series is None:
            continue
        steps_arr, vals_arr = series
        mask = ((steps_arr >= unlock_step - _ZOOM_WINDOW) &
                (steps_arr <= unlock_step + _ZOOM_WINDOW))
        if mask.sum() < 2:
            continue
        rel_steps = steps_arr[mask] - unlock_step
        rel_vals  = vals_arr[mask]
        # Interpolate to common relative grid; NaN outside data range
        trace = np.interp(_REL_GRID, rel_steps, rel_vals,
                          left=np.nan, right=np.nan)
        traces.append(trace)

    if not traces:
        return None
    mat  = np.array(traces)          # (n_seeds, n_grid)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    n    = len(traces)
    fin  = np.isfinite(mean)
    if not fin.any():
        return None
    return mean, std, n


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_zoom(ach_name: str, field: str, color: str, ylabel: str,
               mean: np.ndarray, std: np.ndarray, n: int,
               out_dir: str, dpi: int, stem: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))

    fin = np.isfinite(mean)
    _make_shaded_line(ax, _REL_GRID[fin], mean[fin], std[fin], color,
                      label=f"mean +/- 1 std  (n={n} seeds)")
    ax.axvline(0.0, color="gold", lw=1.5, linestyle="--", label="Achievement unlock")
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k")
    )
    ax.set_xlabel(f"Steps relative to achievement unlock  (+/- {_ZOOM_WINDOW//1000}k)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Rainbow — {ach_name.replace('_', ' ')}  |  {ylabel}\n"
        f"+/-{_ZOOM_WINDOW // 1000}k-step window, mean +/- 1 std ({n} seeds)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ach_name}_{stem}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Public entry point ────────────────────────────────────────────────────────

def plot(rainbow_root: str, out_dir: str, dpi: int = 150) -> list:
    data = load_all_data(rainbow_root, _ALGORITHM)
    periodic  = data["periodic"]   # {step: [(seed_id, rec), ...]}
    milestone = data["milestone"]  # {ach_name: [(seed_id, rec), ...]}

    if not milestone:
        print("ERROR: no Rainbow milestone data found.")
        return []

    ep_step_maps = _build_ep_step_maps(periodic)
    paths = []

    for ach_name, entries in sorted(milestone.items()):
        unlock_steps = _milestone_unlock_steps(entries, ep_step_maps)
        if not unlock_steps:
            print(f"  skip {ach_name} — no episode→step mapping available")
            continue

        for field, color, ylabel, stem in _ZOOM_METRICS:
            result = _compute_zoom(periodic, unlock_steps, field)
            if result is None:
                print(f"  skip {ach_name}/{field} — insufficient data in zoom window")
                continue
            mean, std, n = result
            p = _plot_zoom(ach_name, field, color, ylabel,
                           mean, std, n, out_dir, dpi, stem)
            paths.append(p)

    print(f"\n  Total: {len(paths)} zoom graphs saved to {out_dir}")
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output", "zooms"),
                        help="Output directory (default: rainbow/output/zooms/)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--window", type=int, default=_ZOOM_WINDOW,
                        help="Steps either side of unlock (default: 200000)")
    args = parser.parse_args()

    global _ZOOM_WINDOW, _REL_GRID
    _ZOOM_WINDOW = args.window
    _REL_GRID = np.linspace(-_ZOOM_WINDOW, _ZOOM_WINDOW, 201)

    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
