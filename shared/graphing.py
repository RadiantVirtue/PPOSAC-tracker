"""shared/graphing.py — PPO / Rainbow analysis visualisation.

Usage:
    python shared/graphing.py <experiment_root> [--dpi 150] [--algorithm ppo|rainbow]

Loads all seed_N/analysis_logs/{algorithm}/*.json files, averages metrics by
global step (NOT episode count), and writes PNGs to <experiment_root>/graphs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ── Constants ─────────────────────────────────────────────────────────────────

ACHIEVEMENT_ORDER = [
    # Tier 1
    "collect_wood", "collect_sapling", "collect_drink", "eat_plant", "wake_up",
    # Tier 2
    "place_table", "collect_stone", "defeat_zombie", "eat_cow", "defeat_skeleton",
    "make_wood_pickaxe", "make_wood_sword",
    # Tier 3
    "collect_coal", "place_stone", "place_furnace", "make_stone_pickaxe", "make_stone_sword",
    # Tier 4
    "collect_iron", "collect_diamond", "make_iron_pickaxe", "make_iron_sword", "place_plant",
]

ACHIEVEMENT_TIERS: dict[str, int] = {
    a: (1 if i < 5 else 2 if i < 12 else 3 if i < 17 else 4)
    for i, a in enumerate(ACHIEVEMENT_ORDER)
}

TIER_COLORS = {1: "#2ca02c", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}

# ── Rainbow metric colours ─────────────────────────────────────────────────────
# Red     → opposition score
# Orange  → coherence  (light = success, dark = failure)
# Yellow  → gradient magnitude  (light = success, dark = failure)
# (green skipped)
# Indigo  → activation separation
# Violet  → activation cosine distance
C_RED          = "#d62020"   # opposition score
C_ORANGE_LIGHT = "#ff8c00"   # coherence success
C_ORANGE_DARK  = "#b85c00"   # coherence failure
C_YELLOW_LIGHT = "#c8a000"   # gradient magnitude success
C_YELLOW_DARK  = "#806000"   # gradient magnitude failure
C_INDIGO       = "#3a3acc"   # activation separation
C_VIOLET       = "#8020c0"   # activation cosine distance
C_TEAL         = "#008080"   # RSA alignment / other axes

# Colours for per-seed lines (tab10 palette)
_TAB10 = plt.get_cmap("tab10")


def seed_color(seed_idx: int):
    return _TAB10(seed_idx % 10)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_step(reason: str) -> int | None:
    # Match "step" (case-insensitive) followed by digits, optionally separated by commas
    m = re.search(r"step\s*([\d,]+)", reason, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def _parse_achievement(reason: str) -> str | None:
    # Format 1 (from reason field):  "collect_coal @ ep601"
    m = re.match(r"^(.+?)\s*@\s*ep\d+$", reason)
    if m:
        return m.group(1).strip()
    # Format 2 (from filename stem): "milestone_first_collect_coal_ep601"
    m = re.match(r"milestone_first_(.+)_ep\d+$", reason)
    return m.group(1) if m else None


def _is_milestone(reason: str, filename: str = "") -> bool:
    # reason field uses "collect_coal @ ep601" for milestones
    if re.search(r"@\s*ep\d+", reason):
        return True
    # Fall back to filename pattern
    return "milestone_first_" in filename


# ── Data loading ───────────────────────────────────────────────────────────────

def _find_seed_dirs(root: str) -> list[tuple[int, str]]:
    """Return sorted list of (seed_int, path) for seed_N subdirectories."""
    results = []
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return []
    for name in sorted(entries):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        m = re.match(r"seed_(\d+)$", name)
        if m:
            results.append((int(m.group(1)), path))
    return results or [(0, root)]


def load_all_data(experiment_root: str, algorithm: str = "ppo") -> dict:
    """Load all analysis JSONs from all seeds for the given algorithm.

    Returns:
        {
          "periodic":  {step_int: [(seed_idx, record), ...]},
          "milestone": {ach_name: [(seed_idx, record), ...]},
          "seed_ids":  [seed_int, ...],
        }
    """
    periodic: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    milestone: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    seed_ids: list[int] = []

    seed_dirs = _find_seed_dirs(experiment_root)

    for seed_id, seed_path in seed_dirs:
        seed_ids.append(seed_id)
        log_dir = os.path.join(seed_path, "analysis_logs", algorithm)
        if not os.path.isdir(log_dir):
            continue
        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(log_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    record = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            reason = record.get("reason", fname.replace(".json", ""))

            if _is_milestone(reason, fname):
                ach = _parse_achievement(reason) or _parse_achievement(fname.replace(".json", ""))
                if ach:
                    milestone[ach].append((seed_id, record))
            else:
                step = _parse_step(reason)
                if step is not None:
                    periodic[step].append((seed_id, record))

    return {
        "periodic": dict(periodic),
        "milestone": dict(milestone),
        "seed_ids": sorted(set(seed_ids)),
    }


# ── Scalar extraction helpers ─────────────────────────────────────────────────

def _get(record: dict, key: str):
    """Get a scalar value from a record; returns None if missing or non-finite."""
    if "." in key:
        parts = key.split(".", 1)
        sub = record.get(parts[0])
        if not isinstance(sub, dict):
            return None
        return _get(sub, parts[1])
    v = record.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


_SCALAR_METRICS = [
    "opposition_score",
    "coherence_success",
    "coherence_failure",
    "activation_separation",
    "activation_cosine_distance",
    "gradient_magnitude_success",
    "gradient_magnitude_failure",
    "rsa_alignment_fighting",
    "rsa_alignment_resource",
    "rsa_alignment_crafting",
    "rsa_alignment_housing",
    "n_success",
    "n_failure",
    "threshold_lower",
    "threshold_upper",
    "episode",
    "cluster_stats.n_clusters",
    "cluster_stats.noise_fraction",
    # Weight-delta validation metrics (Rainbow only; None for PPO and first checkpoint)
    "cos_uniform_success_delta",
    "cos_is_success_delta",
    "cos_reward_success_delta",
    "cos_uniform_failure_delta",
    "cos_is_failure_delta",
    "cos_reward_failure_delta",
]


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_periodic(periodic: dict[int, list]) -> list[dict]:
    """Return sorted list of per-step aggregated dicts.

    Each dict has:
      step, n_seeds,
      <metric>_mean, <metric>_std, <metric>_values  for each scalar metric.
    """
    rows = []
    for step in sorted(periodic.keys()):
        entries = periodic[step]
        row: dict = {"step": step, "n_seeds": len(entries)}
        for metric in _SCALAR_METRICS:
            vals = [_get(rec, metric) for _, rec in entries]
            vals = [v for v in vals if v is not None]
            row[f"{metric}_values"] = vals
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def aggregate_milestone(milestone: dict[str, list]) -> dict[str, dict]:
    """Return per-achievement aggregated dicts."""
    result = {}
    for ach, entries in milestone.items():
        row: dict = {"n_seeds": len(entries)}
        for metric in _SCALAR_METRICS:
            vals = [_get(rec, metric) for _, rec in entries]
            vals = [v for v in vals if v is not None]
            row[f"{metric}_values"] = vals
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        result[ach] = row
    return result


def milestone_steps(
    periodic: dict[int, list],
    milestone_data: dict[str, list],
) -> dict[str, float]:
    """Map achievement name -> mean global step of unlock (approximate).

    Strategy: for each seed that has a milestone record, find the periodic step
    whose episode count is closest to the milestone's episode count.  Average
    those step estimates across seeds.
    """
    # Build per-seed episode->step lookup from periodic data
    seed_ep_to_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for step, entries in periodic.items():
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is not None:
                seed_ep_to_step[seed_id].append((int(ep), step))
    # Sort each seed's list by episode
    for seed_id in seed_ep_to_step:
        seed_ep_to_step[seed_id].sort()

    result: dict[str, float] = {}
    for ach, entries in milestone_data.items():
        step_estimates = []
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is None:
                continue
            ep = int(ep)
            pairs = seed_ep_to_step.get(seed_id, [])
            if not pairs:
                continue
            # Closest by episode distance
            best_step = min(pairs, key=lambda x: abs(x[0] - ep))[1]
            step_estimates.append(best_step)
        if step_estimates:
            result[ach] = float(np.mean(step_estimates))
    return result


# ── Per-seed series ────────────────────────────────────────────────────────────

def _per_seed_series(
    periodic: dict[int, list],
    seed_ids: list[int],
    metric: str,
) -> dict[int, tuple[list[int], list[float]]]:
    """Return {seed_id: (steps, values)} for a scalar metric."""
    out: dict[int, tuple[list, list]] = {s: ([], []) for s in seed_ids}
    for step in sorted(periodic.keys()):
        for seed_id, rec in periodic[step]:
            v = _get(rec, metric)
            if v is not None:
                out[seed_id][0].append(step)
                out[seed_id][1].append(v)
    return out


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _make_shaded_line(
    ax,
    steps: list,
    means: list,
    stds: list,
    color,
    label: str,
    alpha: float = 0.2,
    lw: float = 2.0,
    zorder: int = 3,
):
    xs = np.array(steps, dtype=float)
    ys = np.array([v if v is not None else np.nan for v in means])
    errs = np.array([v if v is not None else 0.0 for v in stds])
    ax.plot(xs, ys, color=color, label=label, linewidth=lw, zorder=zorder)
    mask = np.isfinite(ys)
    if mask.any():
        ax.fill_between(
            xs[mask], (ys - errs)[mask], (ys + errs)[mask],
            color=color, alpha=alpha, zorder=zorder - 1,
        )


def _smooth(values: list, window: int = 7) -> np.ndarray:
    """Rolling mean with edge-aware averaging (no scipy needed)."""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        chunk = arr[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        out[i] = np.mean(finite) if len(finite) else np.nan
    return out


def _add_tier_completion_lines(ax, ach_steps: dict[str, float]) -> list:
    """Draw one vertical dashed line per tier at the step of its last unlock.

    Returns a list of legend handles (one per tier present) so callers can
    include them in ax.legend().
    """
    import matplotlib.lines as mlines

    tier_last: dict[int, float] = {}
    for ach, step in ach_steps.items():
        t = ACHIEVEMENT_TIERS.get(ach, 1)
        if t not in tier_last or step > tier_last[t]:
            tier_last[t] = step

    handles = []
    for tier in sorted(tier_last):
        c = TIER_COLORS[tier]
        ax.axvline(tier_last[tier], color=c, linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2)
        handles.append(
            mlines.Line2D([], [], color=c, linestyle="--", linewidth=1.2,
                          label=f"Tier {tier} complete")
        )
    return handles


def _apply_xaxis_millions(ax):
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
    ax.set_xlabel("Global Training Step")


def _save_fig(fig, out_dir: str, filename: str, dpi: int = 150):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _seed_color_map(seed_ids: list[int]) -> dict[int, tuple]:
    return {s: seed_color(i) for i, s in enumerate(seed_ids)}


# ── Generic periodic line-plot factory ────────────────────────────────────────

def _plot_periodic_single(
    agg: list[dict],
    _periodic: dict,
    seed_ids: list[int],
    metric: str,
    title: str,
    ylabel: str,
    out_dir: str,
    filename: str,
    ach_steps: dict[str, float],
    dpi: int,
    avg_color="black",
    hline: float | None = None,
):
    steps = [r["step"] for r in agg]
    means = [r[f"{metric}_mean"] for r in agg]
    stds = [r[f"{metric}_std"] for r in agg]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    _make_shaded_line(ax, steps, means, stds, avg_color,
                      f"Mean \u00b1 1 std ({len(seed_ids)} seeds)")

    if hline is not None:
        ax.axhline(hline, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    tier_handles = _add_tier_completion_lines(ax, ach_steps) if ach_steps else []
    metric_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=metric_handles + tier_handles, loc="upper left", fontsize=8)

    _apply_xaxis_millions(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _save_fig(fig, out_dir, filename, dpi)


def _plot_periodic_dual(
    agg: list[dict],
    _periodic: dict,
    seed_ids: list[int],
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    color_a,
    color_b,
    title: str,
    ylabel: str,
    out_dir: str,
    filename: str,
    ach_steps: dict[str, float],
    dpi: int,
):
    steps = [r["step"] for r in agg]
    means_a = [r[f"{metric_a}_mean"] for r in agg]
    stds_a = [r[f"{metric_a}_std"] for r in agg]
    means_b = [r[f"{metric_b}_mean"] for r in agg]
    stds_b = [r[f"{metric_b}_std"] for r in agg]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    _make_shaded_line(ax, steps, means_a, stds_a, color_a, label_a)
    _make_shaded_line(ax, steps, means_b, stds_b, color_b, label_b)

    tier_handles = _add_tier_completion_lines(ax, ach_steps) if ach_steps else []
    metric_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=metric_handles + tier_handles, loc="upper left", fontsize=8)

    _apply_xaxis_millions(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _save_fig(fig, out_dir, filename, dpi)


# ── Return-log helpers ────────────────────────────────────────────────────────

def _load_return_log(seed_dir: str, algorithm: str = "ppo") -> list[float]:
    """Load per-episode returns from {algorithm}returnlog.txt (one float per line)."""
    path = os.path.join(seed_dir, "logs", f"{algorithm}returnlog.txt")
    if not os.path.exists(path):
        return []
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    return vals


def _milestone_episodes_per_seed(
    milestone_data: dict[str, list],
) -> dict[int, dict[str, int]]:
    """Return {seed_id: {ach_name: episode_count}} directly from milestone records."""
    result: dict[int, dict[str, int]] = defaultdict(dict)
    for ach, entries in milestone_data.items():
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is not None:
                result[seed_id][ach] = int(ep)
    return result


def _cluster_achievements(
    ach_episodes: dict[str, int],
    clump_window: int = 300,
) -> list[list[tuple[str, int]]]:
    """Group achievements within clump_window episodes of each other.

    Returns a list of groups; each group is a list of (ach_name, episode).
    Grouping is based on distance from the first item in each group.
    """
    sorted_achs = sorted(ach_episodes.items(), key=lambda x: x[1])
    groups: list[list] = []
    current: list = []
    group_start = None
    for ach, ep in sorted_achs:
        if not current:
            current = [(ach, ep)]
            group_start = ep
        elif ep - group_start <= clump_window:
            current.append((ach, ep))
        else:
            groups.append(current)
            current = [(ach, ep)]
            group_start = ep
    if current:
        groups.append(current)
    return groups


# ── Return vs metric dual-axis graphs ─────────────────────────────────────────

_RETURN_VS_METRIC_SPECS = [
    ("opposition_score",            "Opposition Score",         C_RED,          0.0),
    ("coherence_success",           "Coherence (Success)",      C_ORANGE_LIGHT, None),
    ("coherence_failure",           "Coherence (Failure)",      C_ORANGE_DARK,  None),
    ("activation_separation",       "Activation Separation",    C_INDIGO,       None),
    ("activation_cosine_distance",  "Activation Cosine Dist.",  C_VIOLET,       None),
    ("rsa_alignment_resource",      "RSA Align (Resource, ρ)",  C_TEAL,         0.0),
    ("gradient_magnitude_success",  "Grad Mag (Success)",       C_YELLOW_LIGHT, None),
    ("gradient_magnitude_failure",  "Grad Mag (Failure)",       C_YELLOW_DARK,  None),
]

_RETURN_SMOOTH = 200  # smoothing window for return values


def plot_return_vs_metrics(
    periodic: dict[int, list],
    seed_ids: list[int],
    milestone_data: dict[str, list],
    experiment_root: str,
    out_dir: str,
    dpi: int,
    algorithm: str = "ppo",
):
    """Dual y-axis: smoothed return (left) vs analysis metric (right), per seed.

    X-axis is global training step.  Achievement markers: isolated achievements
    get a tier-coloured vline; clumped achievements (within 2% of training range)
    share one grey vline.  All are described in the legend.
    """
    per_seed_ach_eps = _milestone_episodes_per_seed(milestone_data)

    # Locate seed directories
    seed_dirs: dict[int, str] = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if m:
            seed_dirs[int(m.group(1))] = os.path.join(experiment_root, name)

    for seed_id in seed_ids:
        seed_dir = seed_dirs.get(seed_id)
        if not seed_dir:
            continue

        returns = _load_return_log(seed_dir, algorithm)
        if not returns:
            print(f"  No return log for seed {seed_id}, skipping return-vs-metric graphs")
            continue

        # Build episode→step mapping for this seed from periodic data
        ep_step: list[tuple[int, int]] = []
        for step, entries in periodic.items():
            for sid, rec in entries:
                if sid == seed_id:
                    ep = _get(rec, "episode")
                    if ep is not None:
                        ep_step.append((int(ep), int(step)))
        ep_step.sort()
        if not ep_step:
            print(f"  No periodic data for seed {seed_id}, skipping return-vs-metric graphs")
            continue

        ep_arr  = np.array([p[0] for p in ep_step], dtype=float)
        st_arr  = np.array([p[1] for p in ep_step], dtype=float)

        # Convert return log from episode-indexed to step-indexed
        smooth_ret  = _smooth(returns, window=_RETURN_SMOOTH)
        n_eps       = len(smooth_ret)
        ep_indices  = np.arange(1, n_eps + 1, dtype=float)
        ep_min, ep_max = ep_arr[0], ep_arr[-1]
        valid_mask  = (ep_indices >= ep_min) & (ep_indices <= ep_max)
        ep_valid    = ep_indices[valid_mask]
        ret_valid   = smooth_ret[valid_mask]
        step_valid  = np.interp(ep_valid, ep_arr, st_arr)

        # Convert achievement episode positions to step positions
        ach_eps = per_seed_ach_eps.get(seed_id, {})
        ach_steps_seed = {}
        for ach, ep in ach_eps.items():
            if ep_arr[0] <= ep <= ep_arr[-1]:
                ach_steps_seed[ach] = float(np.interp(ep, ep_arr, st_arr))
        groups = _cluster_achievements(
            {a: int(s) for a, s in ach_steps_seed.items()},
            clump_window=int((st_arr[-1] - st_arr[0]) * 0.02),
        )

        seed_out = os.path.join(out_dir, f"seed_{seed_id}")

        for metric_key, metric_label, metric_color, hline in _RETURN_VS_METRIC_SPECS:
            # Gather metric values at their exact step positions
            metric_steps_list, metric_vals = [], []
            for step in sorted(periodic.keys()):
                for sid, rec in periodic[step]:
                    if sid == seed_id:
                        v = _get(rec, metric_key)
                        if v is not None:
                            metric_steps_list.append(float(step))
                            metric_vals.append(float(v))
            if not metric_vals:
                continue

            fig, ax1 = plt.subplots(figsize=(13, 5))
            ax2 = ax1.twinx()

            # ── Left axis: smoothed return (step x-axis) ────────────────────
            ax1.plot(step_valid, ret_valid, color="#888888", linewidth=1.2,
                     alpha=0.75, label=f"Return (smoothed, w={_RETURN_SMOOTH})", zorder=2)
            ax1.set_ylabel("Episode Return", color="#555555", fontsize=10)
            ax1.tick_params(axis="y", colors="#555555")
            ax1.spines["left"].set_color("#888888")

            # ── Right axis: metric (step x-axis, smoothed) ──────────────────
            smooth_metric = _smooth(metric_vals, window=_SMOOTH_WINDOW)
            ax2.plot(metric_steps_list, smooth_metric, color=metric_color, linewidth=2.0,
                     label=f"{metric_label} (smoothed, w={_SMOOTH_WINDOW})", zorder=3)
            if hline is not None:
                ax2.axhline(hline, color=metric_color, linestyle=":",
                            linewidth=0.8, alpha=0.5)
            ax2.set_ylabel(metric_label, color=metric_color, fontsize=10)
            ax2.tick_params(axis="y", colors=metric_color)
            ax2.spines["right"].set_color(metric_color)
            ax2.spines["left"].set_visible(False)

            # ── Achievement markers ─────────────────────────────────────────
            import matplotlib.lines as mlines

            ach_legend_handles = []
            cluster_notes = []

            for i, group in enumerate(groups):
                mid_step = float(np.median([ach_steps_seed.get(a, 0) for a, _ in group]))

                if len(group) == 1:
                    ach, _ = group[0]
                    tier = ACHIEVEMENT_TIERS.get(ach, 1)
                    c = TIER_COLORS[tier]
                    short_label = ach.replace("_", " ")
                else:
                    c = "#666666"
                    short_label = f"Cluster {i + 1}"
                    names = ",  ".join(a.replace("_", " ") for a, _ in group)
                    cluster_notes.append(f"Cluster {i + 1}:  {names}")

                ax1.axvline(mid_step, color=c, linestyle=":", linewidth=1.0,
                            alpha=0.7, zorder=1)
                ach_legend_handles.append(
                    mlines.Line2D([], [], color=c, linestyle=":", linewidth=1.0,
                                  label=short_label)
                )

            # ── Combined legend (short labels only) ─────────────────────────
            h1, _ = ax1.get_legend_handles_labels()
            h2, _ = ax2.get_legend_handles_labels()
            ax1.legend(handles=h1 + h2 + ach_legend_handles,
                       loc="upper left", fontsize=8, framealpha=0.85)

            ax1.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
            )
            ax1.set_xlabel("Global Training Step")
            ax1.set_title(
                f"PPO Seed {seed_id} — Return vs {metric_label} Over Training"
            )

            # ── Cluster notes below the plot ────────────────────────────────
            if cluster_notes:
                note_text = "\n".join(cluster_notes)
                fig.text(
                    0.01, -0.04, note_text,
                    fontsize=7, color="#444444",
                    verticalalignment="top", horizontalalignment="left",
                    wrap=True,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#cccccc", alpha=0.9),
                )

            safe = metric_key.replace(".", "_")
            _save_fig(fig, seed_out, f"ppo_seed{seed_id}_return_vs_{safe}.png", dpi)


def plot_return_vs_metrics_averaged(
    agg: list[dict],
    periodic: dict[int, list],
    seed_ids: list[int],
    milestone_data: dict[str, list],
    experiment_root: str,
    out_dir: str,
    dpi: int,
    algorithm: str = "ppo",
):
    """Dual y-axis averaged across all seeds: mean return (left) vs mean metric (right).

    X-axis is global training step.  For each seed the smoothed return log is
    converted from episode-indexed to step-indexed by linear interpolation using
    that seed's periodic (episode, step) pairs, then all seeds are resampled onto
    the common periodic step grid and averaged.

    Secondary x-axis annotations mark every 5 000 episodes (mean step across seeds).
    """
    import matplotlib.lines as mlines

    # ── Locate seed directories ────────────────────────────────────────────────
    seed_dirs: dict[int, str] = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if m:
            seed_dirs[int(m.group(1))] = os.path.join(experiment_root, name)

    # ── Common step grid (the 61 periodic checkpoints) ────────────────────────
    step_grid = np.array(sorted(periodic.keys()), dtype=float)

    # ── Per-seed episode→step interpolation, resampled onto step_grid ─────────
    return_at_grid: list[np.ndarray] = []
    # (seed_id, [(episode, step), ...]) — also used for mapping file and 5k marks
    all_ep_step_pairs: list[tuple[int, list[tuple[int, int]]]] = []

    for seed_id in seed_ids:
        sd = seed_dirs.get(seed_id)
        if not sd:
            continue
        raw = _load_return_log(sd, algorithm)
        if not raw:
            continue

        # Build this seed's (episode, step) pairs from periodic data
        ep_step: list[tuple[int, int]] = []
        for step, entries in periodic.items():
            for sid, rec in entries:
                if sid == seed_id:
                    ep = _get(rec, "episode")
                    if ep is not None:
                        ep_step.append((int(ep), int(step)))
        ep_step.sort()
        if not ep_step:
            continue
        all_ep_step_pairs.append((seed_id, ep_step))

        ep_arr   = np.array([p[0] for p in ep_step], dtype=float)
        step_arr = np.array([p[1] for p in ep_step], dtype=float)

        smooth_ret = _smooth(raw, window=_RETURN_SMOOTH)
        n_eps = len(smooth_ret)
        ep_indices = np.arange(1, n_eps + 1, dtype=float)
        ep_min, ep_max = ep_arr[0], ep_arr[-1]
        valid = (ep_indices >= ep_min) & (ep_indices <= ep_max)
        ep_valid   = ep_indices[valid]
        ret_valid  = smooth_ret[valid]
        step_valid = np.interp(ep_valid, ep_arr, step_arr)

        resampled = np.interp(step_grid, step_valid, ret_valid,
                              left=np.nan, right=np.nan)
        return_at_grid.append(resampled)

    if not return_at_grid:
        print("  No return logs found — skipping averaged return-vs-metric graphs")
        return

    mat = np.array(return_at_grid)
    avg_return = np.nanmean(mat, axis=0)
    std_return = np.nanstd(mat, axis=0)
    n_seeds_ret = len(return_at_grid)

    # ── Save episode→step mapping to text file ────────────────────────────────
    avg_out = os.path.join(out_dir, "averaged")
    os.makedirs(avg_out, exist_ok=True)
    mapping_path = os.path.join(avg_out, "episode_step_mapping.txt")
    with open(mapping_path, "w") as _f:
        _f.write("Episode-to-Step Mapping (from periodic checkpoint data)\n")
        _f.write("=" * 58 + "\n\n")
        for _sid, _pairs in all_ep_step_pairs:
            _f.write(f"Seed {_sid}:\n")
            _f.write(f"  {'Episode':>10}  {'Global Step':>12}\n")
            _f.write(f"  {'-'*10}  {'-'*12}\n")
            for _ep, _st in _pairs:
                _f.write(f"  {_ep:>10}  {_st:>12}\n")
            _f.write("\n")
        if all_ep_step_pairs:
            _max_ep = max(pairs[-1][0] for _, pairs in all_ep_step_pairs)
            _f.write("Mean step at every 5 000 episodes (averaged across seeds):\n")
            _f.write(f"  {'Episode':>10}  {'Mean Step':>12}  {'N Seeds':>7}\n")
            _f.write(f"  {'-'*10}  {'-'*12}  {'-'*7}\n")
            for _ep_t in range(5000, _max_ep + 1, 5000):
                _ests = []
                for _, _pairs in all_ep_step_pairs:
                    _ea = np.array([p[0] for p in _pairs], dtype=float)
                    _sa = np.array([p[1] for p in _pairs], dtype=float)
                    if _ea[0] <= _ep_t <= _ea[-1]:
                        _ests.append(float(np.interp(_ep_t, _ea, _sa)))
                if _ests:
                    _f.write(f"  {_ep_t:>10}  {np.mean(_ests):>12.0f}  {len(_ests):>7}\n")
    print(f"  Saved {mapping_path}")

    # ── Every-5k-episode tick positions (mean step across seeds) ──────────────
    # Build average episode→step mapping across seeds
    if all_ep_step_pairs:
        max_ep = max(pairs[-1][0] for _, pairs in all_ep_step_pairs)
        ep5k_marks: list[tuple[int, float]] = []
        for ep_target in range(5000, max_ep + 1, 5000):
            step_estimates = []
            for _, ep_step in all_ep_step_pairs:
                ep_arr = np.array([p[0] for p in ep_step], dtype=float)
                st_arr = np.array([p[1] for p in ep_step], dtype=float)
                if ep_arr[0] <= ep_target <= ep_arr[-1]:
                    step_estimates.append(float(np.interp(ep_target, ep_arr, st_arr)))
            if step_estimates:
                ep5k_marks.append((ep_target, float(np.mean(step_estimates))))
    else:
        ep5k_marks = []

    # ── Mean achievement steps for markers ────────────────────────────────────
    # Build a lookup: seed_id -> (ep_arr, st_arr) for fast access
    seed_ep_st: dict[int, tuple] = {
        sid: (np.array([p[0] for p in pairs], dtype=float),
              np.array([p[1] for p in pairs], dtype=float))
        for sid, pairs in all_ep_step_pairs
    }
    ach_steps_mean = {}
    for ach, entries in milestone_data.items():
        step_ests = []
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is None or seed_id not in seed_ep_st:
                continue
            ep_arr, st_arr = seed_ep_st[seed_id]
            if ep_arr[0] <= int(ep) <= ep_arr[-1]:
                step_ests.append(float(np.interp(int(ep), ep_arr, st_arr)))
        if step_ests:
            ach_steps_mean[ach] = float(np.mean(step_ests))

    groups = _cluster_achievements(
        {a: int(s) for a, s in ach_steps_mean.items()},
        clump_window=int((step_grid[-1] - step_grid[0]) * 0.02),  # 2% of total range
    )

    # ── Metric series from agg (step-indexed, already averaged) ───────────────
    agg_steps = np.array([r["step"] for r in agg], dtype=float)

    for metric_key, metric_label, metric_color, hline in _RETURN_VS_METRIC_SPECS:
        m_vals = np.array(
            [r[f"{metric_key}_mean"] if r[f"{metric_key}_mean"] is not None else np.nan
             for r in agg]
        )
        if np.all(np.isnan(m_vals)):
            continue

        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax2 = ax1.twinx()

        # ── Left: mean return ± std ────────────────────────────────────────────
        valid = np.isfinite(avg_return)
        ax1.plot(step_grid[valid], avg_return[valid], color="#888888",
                 linewidth=1.4, alpha=0.85, zorder=2,
                 label=f"Return — mean ± 1 std ({n_seeds_ret} seeds, w={_RETURN_SMOOTH})")
        ax1.fill_between(step_grid[valid],
                         (avg_return - std_return)[valid],
                         (avg_return + std_return)[valid],
                         color="#888888", alpha=0.15, zorder=1)
        ax1.set_ylabel("Episode Return", color="#555555", fontsize=10)
        ax1.tick_params(axis="y", colors="#555555")
        ax1.spines["left"].set_color("#888888")

        # ── Right: averaged metric (smoothed) ─────────────────────────────────
        smooth_m = _smooth(list(m_vals), window=_SMOOTH_WINDOW)
        valid_m = np.isfinite(smooth_m)
        ax2.plot(agg_steps[valid_m], smooth_m[valid_m], color=metric_color,
                 linewidth=2.0, zorder=3,
                 label=f"{metric_label} (mean, smoothed w={_SMOOTH_WINDOW}, {n_seeds_ret} seeds)")
        if hline is not None:
            ax2.axhline(hline, color=metric_color, linestyle=":",
                        linewidth=0.8, alpha=0.5)
        ax2.set_ylabel(metric_label, color=metric_color, fontsize=10)
        ax2.tick_params(axis="y", colors=metric_color)
        ax2.spines["right"].set_color(metric_color)
        ax2.spines["left"].set_visible(False)

        # ── Achievement markers ────────────────────────────────────────────────
        ach_legend_handles = []
        cluster_notes = []
        for i, group in enumerate(groups):
            mid_step = float(np.median([ach_steps_mean.get(a, 0) for a, _ in group]))
            if len(group) == 1:
                ach, _ = group[0]
                tier = ACHIEVEMENT_TIERS.get(ach, 1)
                c = TIER_COLORS[tier]
                short_label = ach.replace("_", " ")
            else:
                c = "#666666"
                short_label = f"Cluster {i + 1}"
                names = ",  ".join(a.replace("_", " ") for a, _ in group)
                cluster_notes.append(f"Cluster {i + 1}:  {names}")
            ax1.axvline(mid_step, color=c, linestyle=":", linewidth=1.0,
                        alpha=0.7, zorder=1)
            ach_legend_handles.append(
                mlines.Line2D([], [], color=c, linestyle=":", linewidth=1.0,
                              label=short_label)
            )

        # ── Every-5k-episode x-axis annotations ───────────────────────────────
        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        ax_top.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", length=4, labelsize=7, colors="#aaaaaa",
                           direction="in", pad=-14)
        if ep5k_marks:
            tick_steps = [s for _, s in ep5k_marks]
            tick_labels = [f"{ep // 1000}k ep" for ep, _ in ep5k_marks]
            ax_top.set_xticks(tick_steps)
            ax_top.set_xticklabels(tick_labels, rotation=90)

        # ── Legend, labels, title ──────────────────────────────────────────────
        h1, _ = ax1.get_legend_handles_labels()
        h2, _ = ax2.get_legend_handles_labels()
        ax1.legend(handles=h1 + h2 + ach_legend_handles,
                   loc="upper left", fontsize=8, framealpha=0.85)

        if cluster_notes:
            fig.text(0.01, -0.04, "\n".join(cluster_notes),
                     fontsize=7, color="#444444",
                     verticalalignment="top", horizontalalignment="left",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                               edgecolor="#cccccc", alpha=0.9))

        ax1.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax1.set_xlabel("Global Training Step")
        ax1.set_title(
            f"PPO (Averaged, {n_seeds_ret} Seeds) — Return vs {metric_label} Over Training"
        )

        safe = metric_key.replace(".", "_")
        _save_fig(fig, avg_out, f"ppo_averaged_return_vs_{safe}.png", dpi)


# ── Per-seed individual graphs ────────────────────────────────────────────────

def _milestone_steps_per_seed(
    periodic: dict[int, list],
    milestone_data: dict[str, list],
) -> dict[int, dict[str, int]]:
    """Return {seed_id: {ach_name: nearest_step}} using each seed's own data."""
    seed_ep_to_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for step, entries in periodic.items():
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is not None:
                seed_ep_to_step[seed_id].append((int(ep), step))
    for sid in seed_ep_to_step:
        seed_ep_to_step[sid].sort()

    result: dict[int, dict[str, int]] = defaultdict(dict)
    for ach, entries in milestone_data.items():
        for seed_id, rec in entries:
            ep = _get(rec, "episode")
            if ep is None:
                continue
            pairs = seed_ep_to_step.get(seed_id, [])
            if not pairs:
                continue
            best_step = min(pairs, key=lambda x: abs(x[0] - int(ep)))[1]
            result[seed_id][ach] = best_step
    return result


# Per-seed metric specs: (metric_key, ylabel, title_suffix, color, hline, subdir_key)
_PER_SEED_SPECS = [
    ("opposition_score",           "Opposition Score",          "Gradient Opposition Score",          C_RED,          0.0,  "gradient_signal"),
    ("coherence_success",          "Coherence",                 "Gradient Coherence (Success Group)",  C_ORANGE_LIGHT, None, "gradient_signal"),
    ("coherence_failure",          "Coherence",                 "Gradient Coherence (Failure Group)",  C_ORANGE_DARK,  None, "gradient_signal"),
    ("gradient_magnitude_success", "Gradient L2 Magnitude",    "Gradient Magnitude (Success Group)",  C_YELLOW_LIGHT, None, "gradient_signal"),
    ("gradient_magnitude_failure", "Gradient L2 Magnitude",    "Gradient Magnitude (Failure Group)",  C_YELLOW_DARK,  None, "gradient_signal"),
    ("activation_separation",      "Euclidean Distance",        "Activation Euclidean Separation",     C_INDIGO,       None, "activation_space"),
    ("activation_cosine_distance", "Cosine Distance",           "Activation Cosine Distance",          C_VIOLET,       None, "activation_space"),
    ("rsa_alignment_resource",     "Spearman \u03c1",           "RSA Alignment (Resource)",            C_TEAL,         0.0,  "rsa"),
    ("cluster_stats.n_clusters",   "Cluster Count",             "Number of Activation Clusters",       "teal",         None, "clustering"),
    ("cluster_stats.noise_fraction","Noise Fraction",           "Activation Cluster Noise Fraction",   "brown",        None, "clustering"),
]

# Weight-delta colour scheme: three shades of blue-purple per group
C_DELTA_UNIFORM_S  = "#1a6bbf"   # uniform — success
C_DELTA_IS_S       = "#7b2d8b"   # IS-weighted — success
C_DELTA_REWARD_S   = "#c0507a"   # reward-weighted — success
C_DELTA_UNIFORM_F  = "#5e9ecf"   # uniform — failure (lighter)
C_DELTA_IS_F       = "#b566c8"   # IS-weighted — failure (lighter)
C_DELTA_REWARD_F   = "#e08aa6"   # reward-weighted — failure (lighter)

_SMOOTH_WINDOW = 3


def plot_all_per_seed_graphs(
    periodic: dict[int, list],
    seed_ids: list[int],
    milestone_data: dict[str, list],
    graphs_root: str,
    dpi: int,
):
    """One graph per (seed, metric): raw line + smoothed line + labeled achievement markers."""
    per_seed_ach = _milestone_steps_per_seed(periodic, milestone_data)

    for seed_id in seed_ids:
        seed_ach_steps = per_seed_ach.get(seed_id, {})
        seed_out = os.path.join(graphs_root, f"seed_{seed_id}")
        os.makedirs(seed_out, exist_ok=True)

        for metric, ylabel, title_suffix, color, hline, _subdir in _PER_SEED_SPECS:
            # Collect this seed's data for this metric
            steps, raw_vals = [], []
            for step in sorted(periodic.keys()):
                for sid, rec in periodic[step]:
                    if sid == seed_id:
                        v = _get(rec, metric)
                        if v is not None:
                            steps.append(step)
                            raw_vals.append(v)

            if not steps:
                continue

            smoothed = _smooth(raw_vals, window=_SMOOTH_WINDOW)
            xs = np.array(steps, dtype=float)

            fig, ax = plt.subplots(figsize=(11, 4.5))

            # Raw: thin, transparent
            ax.plot(xs, raw_vals, color=color, linewidth=0.8, alpha=0.3,
                    label="Raw", zorder=2)
            # Smoothed: solid, thicker
            ax.plot(xs, smoothed, color=color, linewidth=2.0, alpha=0.9,
                    label=f"Smoothed (window={_SMOOTH_WINDOW})", zorder=3)

            if hline is not None:
                ax.axhline(hline, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

            tier_handles = _add_tier_completion_lines(ax, seed_ach_steps) if seed_ach_steps else []
            h, _ = ax.get_legend_handles_labels()
            ax.legend(handles=h + tier_handles, loc="lower right", fontsize=8)

            _apply_xaxis_millions(ax)
            ax.set_ylabel(ylabel)
            ax.set_title(f"PPO Seed {seed_id} — {title_suffix} Over Training")

            safe_metric = metric.replace(".", "_").replace("/", "_")
            _save_fig(fig, seed_out, f"ppo_seed{seed_id}_{safe_metric}.png", dpi)


# ── Per-seed opposition detail ─────────────────────────────────────────────────

def plot_opposition_per_seed_detail(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    """One graph per seed: raw + smoothed (window=3) + cross-seed average."""
    # Build step->mean lookup from aggregate data
    avg_by_step = {r["step"]: r["opposition_score_mean"] for r in agg
                   if r["opposition_score_mean"] is not None}
    cmap = _seed_color_map(seed_ids)

    for seed_id in seed_ids:
        steps, raw_vals = [], []
        for step in sorted(periodic.keys()):
            for sid, rec in periodic[step]:
                if sid == seed_id:
                    v = _get(rec, "opposition_score")
                    if v is not None:
                        steps.append(step)
                        raw_vals.append(v)

        if not steps:
            continue

        smoothed = _smooth(raw_vals, window=3)
        xs = np.array(steps, dtype=float)

        # Average line at same x-points
        avg_ys = np.array([avg_by_step.get(s, np.nan) for s in steps])

        fig, ax = plt.subplots(figsize=(11, 4.5))

        seed_c = cmap[seed_id]
        ax.plot(xs, raw_vals, color=seed_c, linewidth=0.8, alpha=0.35,
                label="Raw", zorder=2)
        ax.plot(xs, smoothed, color=seed_c, linewidth=2.0, alpha=0.95,
                label="Smoothed (window=3)", zorder=3)
        ax.plot(xs, avg_ys, color="black", linewidth=1.5, linestyle="--", alpha=0.7,
                label=f"Average ({len(seed_ids)} seeds)", zorder=4)

        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

        tier_handles = _add_tier_completion_lines(ax, ach_steps) if ach_steps else []
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h + tier_handles, loc="upper left", fontsize=8)

        _apply_xaxis_millions(ax)
        ax.set_ylabel("Opposition Score")
        ax.set_title(
            f"PPO Seed {seed_id} — Opposition Score: Raw vs Smoothed vs Average\n"
            "(lower = better gradient distinction between success/failure)"
        )

        seed_out = os.path.join(out_dir, f"seed_{seed_id}")
        _save_fig(fig, seed_out, f"ppo_seed{seed_id}_opposition_detail.png", dpi)


# ── Periodic line plots ────────────────────────────────────────────────────────

def plot_opposition_score(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_single(
        agg, periodic, seed_ids,
        metric="opposition_score",
        title="PPO — Gradient Opposition Score Over Training\n"
              "(lower = better distinction between success/failure gradients)",
        ylabel="Opposition Score",
        out_dir=out_dir, filename="ppo_periodic_opposition_score.png",
        ach_steps=ach_steps, dpi=dpi, hline=0.0, avg_color=C_RED,
    )


def plot_coherence(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_dual(
        agg, periodic, seed_ids,
        metric_a="coherence_success", metric_b="coherence_failure",
        label_a="Success group", label_b="Failure group",
        color_a=C_ORANGE_LIGHT, color_b=C_ORANGE_DARK,
        title="PPO — Within-Group Gradient Coherence Over Training",
        ylabel="Coherence (cosine alignment)",
        out_dir=out_dir, filename="ppo_periodic_coherence_combined.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_gradient_magnitude(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_dual(
        agg, periodic, seed_ids,
        metric_a="gradient_magnitude_success", metric_b="gradient_magnitude_failure",
        label_a="Success group", label_b="Failure group",
        color_a=C_YELLOW_LIGHT, color_b=C_YELLOW_DARK,
        title="PPO — Gradient Magnitude Over Training (Success vs Failure)",
        ylabel="Gradient L2 Magnitude",
        out_dir=out_dir, filename="ppo_periodic_gradient_magnitude_combined.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_activation_separation(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_single(
        agg, periodic, seed_ids,
        metric="activation_separation",
        title="PPO — Activation Euclidean Separation Over Training\n"
              "(Euclidean distance between success/failure activation centroids)",
        ylabel="Euclidean Distance",
        out_dir=out_dir, filename="ppo_periodic_activation_separation.png",
        ach_steps=ach_steps, dpi=dpi, avg_color=C_INDIGO,
    )


def plot_activation_cosine(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_single(
        agg, periodic, seed_ids,
        metric="activation_cosine_distance",
        title="PPO — Activation Cosine Distance Over Training\n"
              "(cosine dissimilarity between success/failure activation centroids)",
        ylabel="Cosine Distance",
        out_dir=out_dir, filename="ppo_periodic_activation_cosine_distance.png",
        ach_steps=ach_steps, dpi=dpi, avg_color=C_VIOLET,
    )


def plot_activation_combined(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    steps = [r["step"] for r in agg]
    means_sep = [r["activation_separation_mean"] for r in agg]
    stds_sep = [r["activation_separation_std"] for r in agg]
    means_cos = [r["activation_cosine_distance_mean"] for r in agg]
    stds_cos = [r["activation_cosine_distance_std"] for r in agg]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    _make_shaded_line(ax1, steps, means_sep, stds_sep, C_INDIGO,
                      f"Mean \u00b1 1 std ({len(seed_ids)} seeds)")
    _make_shaded_line(ax2, steps, means_cos, stds_cos, C_VIOLET,
                      f"Mean \u00b1 1 std ({len(seed_ids)} seeds)")

    tier_handles = _add_tier_completion_lines(ax1, ach_steps) if ach_steps else []
    if ach_steps:
        _add_tier_completion_lines(ax2, ach_steps)

    h1, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + tier_handles, loc="upper left", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("Euclidean Distance")
    ax1.set_title("PPO — Activation Centroid Separation Over Training")
    ax2.set_ylabel("Cosine Distance")
    _apply_xaxis_millions(ax2)

    _save_fig(fig, out_dir, "ppo_periodic_activation_combined.png", dpi)


def plot_rsa_alignment(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    """Plot 4 RSA alignment curves (Fighting / Resource / Crafting / Housing) on one axes."""
    _GROUP_COLORS = {
        "rsa_alignment_fighting": "#d62728",   # red
        "rsa_alignment_resource": "#2ca02c",   # green
        "rsa_alignment_crafting": "#ff7f0e",   # orange
        "rsa_alignment_housing":  "#1f77b4",   # blue
    }
    _GROUP_LABELS = {
        "rsa_alignment_fighting": "Fighting",
        "rsa_alignment_resource": "Resource",
        "rsa_alignment_crafting": "Crafting",
        "rsa_alignment_housing":  "Housing",
    }

    # Only keep steps where at least one group has valid data
    valid_agg = [
        r for r in agg
        if any(r.get(f"{m}_mean") is not None for m in _GROUP_COLORS)
    ]
    if not valid_agg:
        print("  Skipping rsa_alignment plot (no valid data)")
        return

    steps = [r["step"] for r in valid_agg]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    for metric, color in _GROUP_COLORS.items():
        means = [r.get(f"{metric}_mean") for r in valid_agg]
        stds  = [r.get(f"{metric}_std", 0) or 0 for r in valid_agg]
        # Replace None with nan for plotting
        means_arr = np.array([m if m is not None else np.nan for m in means], dtype=float)
        stds_arr  = np.array(stds, dtype=float)
        _make_shaded_line(ax, steps, means_arr.tolist(), stds_arr.tolist(),
                          color, _GROUP_LABELS[metric])

    tier_handles = _add_tier_completion_lines(ax, ach_steps) if ach_steps else []
    metric_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=metric_handles + tier_handles, loc="upper left", fontsize=8)

    _apply_xaxis_millions(ax)
    ax.set_ylabel("Spearman \u03c1")
    ax.set_title(
        "RSA Alignment (Spearman \u03c1 vs Ground-Truth RDM) Over Training\n"
        "(Fighting / Resource / Crafting / Housing — items may appear in multiple groups)"
    )
    _save_fig(fig, out_dir, "ppo_periodic_rsa_alignment.png", dpi)


def plot_n_success_failure(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_dual(
        agg, periodic, seed_ids,
        metric_a="n_success", metric_b="n_failure",
        label_a="Success episodes", label_b="Failure episodes",
        color_a="#1f77b4", color_b="#d62728",
        title="PPO — Partition Size Over Training",
        ylabel="Episode Count",
        out_dir=out_dir, filename="ppo_periodic_n_success_failure.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_cluster_n_clusters(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_single(
        agg, periodic, seed_ids,
        metric="cluster_stats.n_clusters",
        title="PPO — Number of Activation Clusters Over Training\n(HDBSCAN)",
        ylabel="Cluster Count",
        out_dir=out_dir, filename="ppo_periodic_cluster_n_clusters.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_cluster_noise_fraction(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_single(
        agg, periodic, seed_ids,
        metric="cluster_stats.noise_fraction",
        title="PPO — Fraction of Noise Points in Activation Clustering Over Training",
        ylabel="Noise Fraction",
        out_dir=out_dir, filename="ppo_periodic_cluster_noise_fraction.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_episode_count(periodic, seed_ids, out_dir, dpi):
    """Per-seed episode count vs step — NOT averaged."""
    cmap = _seed_color_map(seed_ids)
    per_seed = _per_seed_series(periodic, seed_ids, "episode")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for seed_id, (steps, vals) in per_seed.items():
        if steps:
            ax.plot(steps, vals, color=cmap[seed_id], linewidth=1.2,
                    label=f"Seed {seed_id}")

    _apply_xaxis_millions(ax)
    ax.set_ylabel("Episode Count")
    ax.set_title("PPO — Episode Count vs Global Step (per seed)\n"
                 "(divergence shows different episode rates across seeds)")
    ax.legend(loc="upper left")
    _save_fig(fig, out_dir, "ppo_periodic_episode_count.png", dpi)


def plot_threshold_bounds(agg, periodic, seed_ids, ach_steps, out_dir, dpi):
    _plot_periodic_dual(
        agg, periodic, seed_ids,
        metric_a="threshold_lower", metric_b="threshold_upper",
        label_a="Lower bound (25th pct)", label_b="Upper bound (75th pct)",
        color_a="#1f77b4", color_b="#d62728",
        title="PPO — Dynamic Success Threshold Bounds Over Training\n"
              "(EPS score percentile thresholds defining success/failure partition)",
        ylabel="EPS Score",
        out_dir=out_dir, filename="ppo_periodic_threshold_bounds.png",
        ach_steps=ach_steps, dpi=dpi,
    )


def plot_summary_dashboard(agg, out_dir, dpi):
    """3×3 summary dashboard with averaged lines + shading only (no per-seed)."""
    steps = np.array([r["step"] for r in agg], dtype=float)

    panels = [
        ("opposition_score",              "Opposition Score",         "black",     None),
        ("activation_separation",         "Activation Eucl. Sep.",    "steelblue", None),
        ("activation_cosine_distance",    "Activation Cos. Dist.",    "darkorange", None),
        ("rsa_alignment_resource",        "RSA Align (Resource)",     "purple",    0.0),
        ("gradient_magnitude_success",    "Grad Mag (Success)",       "#1f77b4",   None),
        ("gradient_magnitude_failure",    "Grad Mag (Failure)",       "#d62728",   None),
        ("cluster_stats.n_clusters",      "N Clusters",               "teal",      None),
        ("cluster_stats.noise_fraction",  "Noise Fraction",           "brown",     None),
        ("n_success",                     "N Success/Failure",        "#1f77b4",   None),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False)
    axes_flat = axes.flatten()

    for i, (metric, ylabel, color, hline) in enumerate(panels):
        ax = axes_flat[i]
        means = np.array([r[f"{metric}_mean"] if r[f"{metric}_mean"] is not None else np.nan
                          for r in agg])
        stds = np.array([r[f"{metric}_std"] for r in agg])

        if metric == "n_success":
            # Special: overlay n_failure too
            means_f = np.array([r["n_failure_mean"] if r["n_failure_mean"] is not None else np.nan
                                 for r in agg])
            stds_f = np.array([r["n_failure_std"] for r in agg])
            ax.plot(steps, means, color="#1f77b4", linewidth=1.5, label="Success")
            ax.fill_between(steps, means - stds, means + stds, color="#1f77b4", alpha=0.2)
            ax.plot(steps, means_f, color="#d62728", linewidth=1.5, label="Failure")
            ax.fill_between(steps, means_f - stds_f, means_f + stds_f, color="#d62728", alpha=0.2)
            ax.legend(fontsize=7)
        else:
            ax.plot(steps, means, color=color, linewidth=1.5)
            ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)

        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=0.8)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle="--")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("PPO — Analysis Summary Dashboard (Mean ± 1 std across seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, out_dir, "ppo_periodic_summary_dashboard.png", dpi)


# ── Milestone bar charts ───────────────────────────────────────────────────────

def _plot_milestone_bar(
    mil_agg: dict[str, dict],
    metric: str,
    title: str,
    xlabel: str,
    out_dir: str,
    filename: str,
    dpi: int,
):
    # Collect achievements in ACHIEVEMENT_ORDER order, only those with data
    present = []
    for ach in ACHIEVEMENT_ORDER:
        row = mil_agg.get(ach)
        if row is None:
            continue
        v = row.get(f"{metric}_mean")
        if v is None:
            continue
        present.append(ach)

    if not present:
        print(f"  Skipping {filename} (no milestone data for {metric})")
        return

    # Also add any achievements in the data not in ACHIEVEMENT_ORDER
    for ach in sorted(mil_agg.keys()):
        if ach not in present:
            row = mil_agg[ach]
            v = row.get(f"{metric}_mean")
            if v is not None:
                present.append(ach)

    # Reverse so Tier 1 is at top
    ordered = list(reversed(present))
    means = [mil_agg[a][f"{metric}_mean"] for a in ordered]
    stds = [mil_agg[a][f"{metric}_std"] for a in ordered]
    colors = [TIER_COLORS.get(ACHIEVEMENT_TIERS.get(a, 1), "#aaaaaa") for a in ordered]
    labels = [a.replace("_", " ") for a in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, len(ordered) * 0.45)))
    y_pos = np.arange(len(ordered))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, height=0.65,
                   capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Tier legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=f"Tier {t}")
        for t in sorted(TIER_COLORS)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    _save_fig(fig, out_dir, filename, dpi)


def plot_all_milestone_bars(mil_agg, metric_dirs: dict[str, str], dpi):
    """metric_dirs maps metric key -> output subdirectory path."""
    specs = [
        ("opposition_score",           "Opposition Score",                    "PPO — Opposition Score at First Achievement Unlock"),
        ("activation_separation",      "Activation Euclidean Separation",     "PPO — Activation Separation at First Achievement Unlock"),
        ("activation_cosine_distance", "Activation Cosine Distance",          "PPO — Activation Cosine Distance at First Achievement Unlock"),
        ("coherence_success",          "Coherence (Success group)",           "PPO — Gradient Coherence (Success) at First Achievement Unlock"),
        ("coherence_failure",          "Coherence (Failure group)",           "PPO — Gradient Coherence (Failure) at First Achievement Unlock"),
        ("rsa_alignment_fighting",     "RSA Align \u03c1 (Fighting)",         "RSA Alignment — Fighting at First Achievement Unlock"),
        ("rsa_alignment_resource",     "RSA Align \u03c1 (Resource)",         "RSA Alignment — Resource at First Achievement Unlock"),
        ("rsa_alignment_crafting",     "RSA Align \u03c1 (Crafting)",         "RSA Alignment — Crafting at First Achievement Unlock"),
        ("rsa_alignment_housing",      "RSA Align \u03c1 (Housing)",          "RSA Alignment — Housing at First Achievement Unlock"),
        ("episode",                    "Episode Count at Unlock",             "PPO — Episode Count at First Achievement Unlock"),
    ]
    for metric, xlabel, title in specs:
        subdir = metric_dirs.get(metric, metric_dirs.get("_default", ""))
        fname = f"ppo_milestone_{metric}.png"
        _plot_milestone_bar(mil_agg, metric, title, xlabel, subdir, fname, dpi)


# ── Snapshot helpers ───────────────────────────────────────────────────────────

def _select_snapshot_steps(periodic: dict[int, list]) -> list[int]:
    """Pick first, middle, and last periodic steps with rsa_n_stimuli >= 2."""
    valid = sorted(
        step for step, entries in periodic.items()
        if any(_get(rec, "rsa_n_stimuli") is not None and
               (_get(rec, "rsa_n_stimuli") or 0) >= 2
               for _, rec in entries)
    )
    if not valid:
        return []
    if len(valid) <= 3:
        return valid
    return [valid[0], valid[len(valid) // 2], valid[-1]]


# ── RDM heatmaps ──────────────────────────────────────────────────────────────

def _plot_rdm(rdm: list[list[float]], labels: list[str], title: str, out_path: str, dpi: int):
    n = len(labels)
    mat = np.array(rdm)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.8), max(4, n * 0.8)))
    im = ax.imshow(mat, vmin=0, vmax=mat.max() if mat.max() > 0 else 1,
                   cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine Dissimilarity", fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rdm_snapshots(periodic: dict[int, list], out_dir: str, dpi: int):
    snap_steps = _select_snapshot_steps(periodic)
    if not snap_steps:
        print("  No RDM snapshot steps found")
        return

    for step in snap_steps:
        entries = periodic.get(step, [])
        label_sets: list[tuple[tuple, list[list]]] = []  # (labels_tuple, [rdm, ...])

        for seed_id, rec in entries:
            rdm = rec.get("rsa_rdm")
            labels = rec.get("rsa_labels")
            if not rdm or not labels:
                continue
            ep = int(_get(rec, "episode") or 0)
            out_path = os.path.join(
                out_dir, f"ppo_rdm_seed{seed_id}_step{step:07d}.png"
            )
            _plot_rdm(
                rdm, labels,
                f"PPO Seed {seed_id} — RSA Representational Dissimilarity Matrix\n"
                f"Step {step:,} (ep {ep:,})",
                out_path, dpi,
            )
            label_sets.append((tuple(labels), rdm))

        # Averaged RDM — only if 2+ seeds have identical label sets
        from collections import Counter
        label_counts = Counter(ls for ls, _ in label_sets)
        for label_tuple, count in label_counts.items():
            if count < 2:
                continue
            matching_rdms = [rdm for ls, rdm in label_sets if ls == label_tuple]
            avg_rdm = np.mean(matching_rdms, axis=0).tolist()
            out_path = os.path.join(out_dir, f"ppo_rdm_averaged_step{step:07d}.png")
            _plot_rdm(
                avg_rdm, list(label_tuple),
                f"PPO (Averaged across {count} seeds) — RSA RDM at Step {step:,}",
                out_path, dpi,
            )
            break  # Only one averaged RDM per step


# ── Cluster composition charts ─────────────────────────────────────────────────

def plot_cluster_snapshots(periodic: dict[int, list], out_dir: str, dpi: int):
    snap_steps = _select_snapshot_steps(periodic)
    if not snap_steps:
        return

    for step in snap_steps:
        entries = periodic.get(step, [])
        for seed_id, rec in entries:
            clusters = rec.get("cluster_stats", {}).get("clusters")
            if not clusters:
                continue
            ep = int(_get(rec, "episode") or 0)

            # Sort by size descending, cap at 30
            sorted_clusters = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)
            sorted_clusters = sorted_clusters[:30]

            sizes = [c.get("size", 0) for c in sorted_clusters]
            succ = [c.get("success_count", 0) for c in sorted_clusters]
            fail = [c.get("failure_count", 0) for c in sorted_clusters]
            y = np.arange(len(sorted_clusters))

            fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_clusters) * 0.3)))
            ax.barh(y, succ, color="#1f77b4", height=0.7, label="Success")
            ax.barh(y, fail, left=succ, color="#d62728", height=0.7, label="Failure")
            ax.set_yticks(y)
            ax.set_yticklabels([f"Cluster {i+1}" for i in range(len(y))], fontsize=7)
            ax.set_xlabel("Episode Count")
            ax.set_title(
                f"PPO Seed {seed_id} — Cluster Composition at Step {step:,} (ep {ep:,})\n"
                f"(top {len(sorted_clusters)} clusters by size, sorted descending)"
            )
            ax.legend(loc="lower right", fontsize=8)
            out_path = os.path.join(out_dir, f"ppo_clusters_seed{seed_id}_step{step:07d}.png")
            _save_fig(fig, out_dir, f"ppo_clusters_seed{seed_id}_step{step:07d}.png", dpi)


# ── Survival-event zoomed plots ────────────────────────────────────────────────

# Achievements that relate to combat / survival difficulty
_SURVIVAL_ACHIEVEMENTS = [
    "defeat_zombie",
    "defeat_skeleton",
    "make_stone_sword",
    "make_iron_sword",
    "collect_iron",
]

_ZOOM_HALF      = 200_000   # ±200 k steps around each event
_ZOOM_METRICS   = [         # (metric_key, label, right-axis colour)
    ("opposition_score",  "Opposition Score",    C_RED),
    ("coherence_failure", "Coherence (Failure)", C_ORANGE_DARK),
]


def plot_survival_zoom(
    agg: list[dict],
    periodic: dict[int, list],
    seed_ids: list[int],
    ach_steps: dict[str, float],
    experiment_root: str,
    out_dir: str,
    dpi: int,
    milestone_data: dict | None = None,
    algorithm: str = "ppo",
):
    """Zoomed dual-axis plots around each survival achievement first unlock.

    X-axis is steps *relative to each seed's own unlock* of the achievement,
    so x=0 always means "the moment of first unlock" for every seed.  Seeds
    that never unlocked the achievement are excluded from that plot.

    For every survival achievement present in ach_steps, produces one graph per
    zoom-metric (opposition_score, coherence_failure).  Each graph shows:
      • Left axis  : mean return ± std (relative-step-indexed, smoothed)
      • Right axis : per-seed metric lines (thin, α=0.25) + bold mean + shading
      • Gold dashed vline at x=0 (the unlock moment)
    Window = ±_ZOOM_HALF steps around unlock.
    """
    import matplotlib.lines as mlines
    import matplotlib.ticker as ticker

    target_achs = [a for a in _SURVIVAL_ACHIEVEMENTS if a in ach_steps]
    if not target_achs:
        print("  No survival achievement data; skipping survival zoom plots")
        return

    # Per-seed unlock steps: {seed_id: {ach_name: step}}
    per_seed_ach: dict[int, dict[str, int]] = {}
    if milestone_data:
        per_seed_ach = _milestone_steps_per_seed(periodic, milestone_data)

    # ── Build step-indexed return arrays per seed (absolute steps) ─────────────
    seed_dirs: dict[int, str] = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if m:
            seed_dirs[int(m.group(1))] = os.path.join(experiment_root, name)

    seed_step_return: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed_id in seed_ids:
        seed_dir = seed_dirs.get(seed_id)
        if not seed_dir:
            continue
        returns = _load_return_log(seed_dir, algorithm)
        if not returns:
            continue
        ep_step: list[tuple[int, int]] = []
        for step, entries in periodic.items():
            for sid, rec in entries:
                if sid == seed_id:
                    ep = _get(rec, "episode")
                    if ep is not None:
                        ep_step.append((int(ep), int(step)))
        ep_step.sort()
        if not ep_step:
            continue
        ep_arr = np.array([p[0] for p in ep_step], dtype=float)
        st_arr = np.array([p[1] for p in ep_step], dtype=float)
        raw_ret = np.array(returns, dtype=float)
        n_eps = len(raw_ret)
        ep_idx = np.arange(1, n_eps + 1, dtype=float)
        valid = (ep_idx >= ep_arr[0]) & (ep_idx <= ep_arr[-1])
        st_valid  = np.interp(ep_idx[valid], ep_arr, st_arr)
        ret_valid = raw_ret[valid]
        seed_step_return[seed_id] = (st_valid, ret_valid)

    os.makedirs(out_dir, exist_ok=True)
    cmap = _seed_color_map(seed_ids)

    rel_grid = np.linspace(-_ZOOM_HALF, _ZOOM_HALF, 300)

    def _fmt_rel(x, _):
        if x == 0:
            return "0"
        return f"{x / 1_000:+.0f}k"

    for ach in target_achs:
        # Seeds that actually unlocked this achievement (required for alignment)
        seeds_with_unlock = [
            sid for sid in seed_ids
            if per_seed_ach.get(sid, {}).get(ach) is not None
        ]
        if not seeds_with_unlock:
            # Fall back: no per-seed data, skip relative plot
            print(f"  Skipping relative zoom for {ach}: no per-seed unlock steps")
            continue

        n_seeds_unlock = len(seeds_with_unlock)

        # ── Return arrays shifted to relative steps ────────────────────────────
        ret_arrs = []
        for seed_id in seeds_with_unlock:
            seed_unlock = per_seed_ach[seed_id][ach]
            if seed_id not in seed_step_return:
                continue
            st_arr, ret_arr = seed_step_return[seed_id]
            rel_st = st_arr - seed_unlock
            mask = (rel_st >= -_ZOOM_HALF) & (rel_st <= _ZOOM_HALF)
            if mask.sum() < 2:
                continue
            interp = np.interp(rel_grid, rel_st[mask], ret_arr[mask],
                               left=np.nan, right=np.nan)
            ret_arrs.append(interp)

        for metric_key, metric_label, metric_color in _ZOOM_METRICS:
            fig, ax1 = plt.subplots(figsize=(11, 4.5))
            ax2 = ax1.twinx()

            # ── Left: averaged return (relative steps) ─────────────────────────
            if ret_arrs:
                ret_mat = np.vstack(ret_arrs)
                avg_ret = np.nanmean(ret_mat, axis=0)
                std_ret = np.nanstd(ret_mat, axis=0)
                fin     = np.isfinite(avg_ret)
                ax1.plot(rel_grid[fin], avg_ret[fin],
                         color="#888888", linewidth=1.4, alpha=0.85, zorder=2,
                         label=f"Return — mean ± std ({len(ret_arrs)} seeds)")
                ax1.fill_between(rel_grid[fin],
                                 (avg_ret - std_ret)[fin],
                                 (avg_ret + std_ret)[fin],
                                 color="#888888", alpha=0.15, zorder=1)
            ax1.set_ylabel("Episode Return", color="#555555", fontsize=10)
            ax1.tick_params(axis="y", colors="#555555")
            ax1.spines["left"].set_color("#888888")

            # ── Right: mean metric line only (no per-seed lines) ───────────────
            metric_arrs = []
            for seed_id in seeds_with_unlock:
                seed_unlock = per_seed_ach[seed_id][ach]
                pairs: list[tuple[float, float]] = []
                for step, entries in periodic.items():
                    for sid, rec in entries:
                        if sid == seed_id:
                            v = _get(rec, metric_key)
                            if v is not None:
                                rel = float(step - seed_unlock)
                                if -_ZOOM_HALF <= rel <= _ZOOM_HALF:
                                    pairs.append((rel, v))
                if len(pairs) < 2:
                    continue
                pairs.sort()
                r_arr = np.array([p[0] for p in pairs], dtype=float)
                v_arr = np.array([p[1] for p in pairs], dtype=float)
                interp = np.interp(rel_grid, r_arr, v_arr, left=np.nan, right=np.nan)
                metric_arrs.append(interp)

            # Bold cross-seed mean (averaged in relative-step space, no smoothing)
            if metric_arrs:
                m_mat   = np.vstack(metric_arrs)
                mean_m  = np.nanmean(m_mat, axis=0)
                std_m   = np.nanstd(m_mat, axis=0)
                fin_m   = np.isfinite(mean_m)
                ax2.plot(rel_grid[fin_m], mean_m[fin_m],
                         color=metric_color, linewidth=2.2, zorder=4,
                         label=f"{metric_label} — mean (n={len(metric_arrs)})")
                ax2.fill_between(rel_grid[fin_m],
                                 (mean_m - std_m)[fin_m],
                                 (mean_m + std_m)[fin_m],
                                 color=metric_color, alpha=0.15, zorder=3)
            ax2.set_ylabel(metric_label, color=metric_color, fontsize=10)
            ax2.tick_params(axis="y", colors=metric_color)
            ax2.spines["right"].set_color(metric_color)

            # ── Unlock vline at x=0 ────────────────────────────────────────────
            ax1.axvline(0, color="gold", linestyle="--",
                        linewidth=1.8, alpha=0.9, zorder=5)
            unlock_handle = mlines.Line2D(
                [], [], color="gold", linestyle="--", linewidth=1.8,
                label=f"'{ach.replace('_', ' ')}' first unlock  (x=0, {n_seeds_unlock} seeds)",
            )

            # ── Legend ────────────────────────────────────────────────────────
            h1, _ = ax1.get_legend_handles_labels()
            h2, _ = ax2.get_legend_handles_labels()
            ax1.legend(handles=h1 + h2 + [unlock_handle],
                       loc="upper left", fontsize=8)

            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_rel))
            ax1.set_xlabel("Steps relative to first unlock")
            ax1.set_title(
                f"PPO — '{ach.replace('_', ' ')}' First Unlock  (event-aligned)\n"
                f"{metric_label} vs Return  (window ±{_ZOOM_HALF // 1_000}k steps, {n_seeds_unlock} seeds)"
            )

            safe_ach    = ach.replace(".", "_")
            safe_metric = metric_key.replace(".", "_")
            _save_fig(fig, out_dir, f"ppo_zoom_{safe_ach}_{safe_metric}.png", dpi)


# ── Weight-delta alignment plots (Rainbow only) ────────────────────────────────

def plot_weight_delta_alignment(
    periodic: dict[int, list],
    seed_ids: list[int],
    ach_steps: dict[str, float],
    out_dir: str,
    dpi: int,
):
    """Two panels: cosine(gradient variant, Δθ) over training, for success and failure groups.

    Each panel shows three lines (uniform, IS-weighted, reward-weighted) ± std across seeds.
    The key signal is whether G_IS tracks Δθ better than G_uniform — if so, the IS
    reconstruction is directionally validated. Absolute cosines are expected to be low
    (Adam distortion); interpret relative differences, not magnitudes.

    Note: data is only non-None from the second analyzed checkpoint onwards (first
    checkpoint has no prior weights to compare against). None values are excluded
    from averaging automatically.
    """
    _GROUPS = [
        ("success", "Success Group",
         [("cos_uniform_success_delta", "G_uniform",      C_DELTA_UNIFORM_S),
          ("cos_is_success_delta",      "G_IS (primary)", C_DELTA_IS_S),
          ("cos_reward_success_delta",  "G_reward",       C_DELTA_REWARD_S)]),
        ("failure", "Failure Group",
         [("cos_uniform_failure_delta", "G_uniform",      C_DELTA_UNIFORM_F),
          ("cos_is_failure_delta",      "G_IS (primary)", C_DELTA_IS_F),
          ("cos_reward_failure_delta",  "G_reward",       C_DELTA_REWARD_F)]),
    ]

    for group_key, group_title, specs in _GROUPS:
        steps_sorted = sorted(periodic.keys())

        fig, ax = plt.subplots(figsize=(11, 4.5))

        any_data = False
        for metric_key, label, color in specs:
            means, stds, xs = [], [], []
            for step in steps_sorted:
                vals = [_get(rec, metric_key) for _, rec in periodic[step]]
                vals = [v for v in vals if v is not None]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                    xs.append(step)
                    any_data = True

            if xs:
                _make_shaded_line(ax, xs, means, stds, color, label)

        if not any_data:
            plt.close(fig)
            continue

        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

        tier_handles = _add_tier_completion_lines(ax, ach_steps) if ach_steps else []
        metric_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=metric_handles + tier_handles, loc="upper left", fontsize=8)

        _apply_xaxis_millions(ax)
        ax.set_ylabel("Cosine Similarity (gradient vs \u0394\u03b8)")
        ax.set_title(
            f"Rainbow \u2014 Weight \u0394 Alignment ({group_title})\n"
            "Relative G_IS > G_uniform at mid-training = IS correction directionally validated"
        )

        _save_fig(fig, out_dir, f"rainbow_weight_delta_{group_key}.png", dpi)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import matplotlib.ticker

    parser = argparse.ArgumentParser(
        description="Generate analysis graphs from experiment_root data."
    )
    parser.add_argument("experiment_root", help="Path to the experiment root directory")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI (default: 150)")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "rainbow"],
                        help="Which algorithm's logs to load (default: ppo)")
    args = parser.parse_args()

    experiment_root = args.experiment_root
    dpi = args.dpi
    algorithm = args.algorithm
    out_dir = os.path.join(experiment_root, "graphs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {experiment_root}  (algorithm={algorithm})")
    raw = load_all_data(experiment_root, algorithm=algorithm)
    periodic = raw["periodic"]
    milestone_data = raw["milestone"]
    seed_ids = raw["seed_ids"]

    print(f"  Periodic checkpoints: {len(periodic)} steps")
    print(f"  Milestone checkpoints: {len(milestone_data)} achievements")
    print(f"  Seeds: {seed_ids}")

    if not periodic:
        print("ERROR: No periodic checkpoint data found. Check experiment_root path.")
        return

    print("Aggregating data...")
    agg = aggregate_periodic(periodic)
    mil_agg = aggregate_milestone(milestone_data)
    ach_steps = milestone_steps(periodic, milestone_data)

    print(f"  Achievement markers: {len(ach_steps)}")

    # ── Subdirectory layout grouped by metric ─────────────────────────────────
    # graphs/
    #   gradient_signal/   opposition_score, coherence, gradient_magnitude
    #   activation_space/  activation separation, cosine distance, combined
    #   rsa/               rsa_alignment, RDM heatmaps
    #   clustering/        n_clusters, noise_fraction, cluster compositions
    #   training_dynamics/ episode_count, threshold_bounds, n_success_failure
    #   summary/           dashboard
    def d(name):
        p = os.path.join(out_dir, name)
        os.makedirs(p, exist_ok=True)
        return p

    dirs = {
        "gradient":  d("gradient_signal"),
        "activation": d("activation_space"),
        "rsa":        d("rsa"),
        "clustering": d("clustering"),
        "training":   d("training_dynamics"),
        "summary":    d("summary"),
    }

    print(f"\nGenerating graphs -> {out_dir}")
    print("  [A] Periodic line plots...")
    plot_opposition_score(agg, periodic, seed_ids, ach_steps, dirs["gradient"], dpi)
    plot_coherence(agg, periodic, seed_ids, ach_steps, dirs["gradient"], dpi)
    plot_gradient_magnitude(agg, periodic, seed_ids, ach_steps, dirs["gradient"], dpi)
    plot_activation_separation(agg, periodic, seed_ids, ach_steps, dirs["activation"], dpi)
    plot_activation_cosine(agg, periodic, seed_ids, ach_steps, dirs["activation"], dpi)
    plot_activation_combined(agg, periodic, seed_ids, ach_steps, dirs["activation"], dpi)
    plot_rsa_alignment(agg, periodic, seed_ids, ach_steps, dirs["rsa"], dpi)
    plot_n_success_failure(agg, periodic, seed_ids, ach_steps, dirs["training"], dpi)
    plot_cluster_n_clusters(agg, periodic, seed_ids, ach_steps, dirs["clustering"], dpi)
    plot_cluster_noise_fraction(agg, periodic, seed_ids, ach_steps, dirs["clustering"], dpi)
    plot_threshold_bounds(agg, periodic, seed_ids, ach_steps, dirs["training"], dpi)
    plot_summary_dashboard(agg, dirs["summary"], dpi)

    print("  [B] Milestone bar charts...")
    milestone_dirs = {
        "opposition_score":           dirs["gradient"],
        "coherence_success":          dirs["gradient"],
        "coherence_failure":          dirs["gradient"],
        "gradient_magnitude_success": dirs["gradient"],
        "gradient_magnitude_failure": dirs["gradient"],
        "activation_separation":      dirs["activation"],
        "activation_cosine_distance": dirs["activation"],
        "rsa_alignment_fighting":     dirs["rsa"],
        "rsa_alignment_resource":     dirs["rsa"],
        "rsa_alignment_crafting":     dirs["rsa"],
        "rsa_alignment_housing":      dirs["rsa"],
        "episode":                    dirs["training"],
    }
    plot_all_milestone_bars(mil_agg, milestone_dirs, dpi)

    print("  [C] Snapshot heatmaps and cluster charts...")
    plot_rdm_snapshots(periodic, dirs["rsa"], dpi)
    plot_cluster_snapshots(periodic, dirs["clustering"], dpi)

    print("  [D] Per-seed individual graphs...")
    per_seed_root = d("per_seed")
    plot_opposition_per_seed_detail(agg, periodic, seed_ids, ach_steps, per_seed_root, dpi)
    plot_all_per_seed_graphs(periodic, seed_ids, milestone_data, per_seed_root, dpi)

    print("  [E] Return vs metric dual-axis graphs...")
    return_root = d("return_vs_metric")
    plot_return_vs_metrics(periodic, seed_ids, milestone_data, experiment_root, return_root, dpi,
                           algorithm=algorithm)
    plot_return_vs_metrics_averaged(agg, periodic, seed_ids, milestone_data, experiment_root, return_root, dpi,
                                    algorithm=algorithm)

    print("  [F] Survival-event zoomed plots...")
    zoom_root = d("zoomed")
    plot_survival_zoom(agg, periodic, seed_ids, ach_steps, experiment_root, zoom_root, dpi,
                       milestone_data=milestone_data, algorithm=algorithm)

    if algorithm == "rainbow":
        print("  [G] Rainbow Moment of Reward plots...")
        from rainbow.moment_of_reward import plot_moment_of_reward
        mor_root = d("moment_of_reward")
        # Collect all MoR records from periodic data (one per checkpoint per seed)
        mor_records = []
        for step in sorted(periodic.keys()):
            for _seed_id, rec in periodic[step]:
                mor = rec.get("moment_of_reward")
                if mor:
                    mor_records.append((step, mor))
        if mor_records:
            plot_moment_of_reward(mor_records, mor_root, dpi)
        else:
            print("    No moment_of_reward data found in periodic records")

        print("  [H] Rainbow weight-delta alignment plots...")
        delta_root = d("weight_delta")
        plot_weight_delta_alignment(periodic, seed_ids, ach_steps, delta_root, dpi)

    # Count all PNGs recursively
    total = 0
    summary_lines = []
    for sub in sorted(os.listdir(out_dir)):
        subpath = os.path.join(out_dir, sub)
        if not os.path.isdir(subpath):
            continue
        n = sum(
            len([f for f in os.listdir(os.path.join(subpath, s)) if f.endswith(".png")])
            if os.path.isdir(os.path.join(subpath, s)) else
            len([f for f in [s] if f.endswith(".png")])
            for s in os.listdir(subpath)
        ) + len([f for f in os.listdir(subpath) if f.endswith(".png")])
        total += n
        summary_lines.append(f"  {sub}/  ({n} files)")
    print(f"\nDone. ~{total} PNG(s) written to {out_dir}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    import matplotlib.ticker
    main()
