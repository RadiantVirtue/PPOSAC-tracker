"""dissertation_graphs/ablations/ablation_no_gae_ppo.py

Ablation 5: No-GAE PPO Variant — with vs without Generalised Advantage Estimation.

Outputs:
  GRAPHS/Ablation graphs/PPO/abl_no_gae_opposition.pdf
  GRAPHS/Ablation graphs/PPO/abl_no_gae_coherence.pdf

abl_no_gae_opposition.pdf:
  PPO with GAE    (solid  C_RED, 5-seed mean+std)
  No-GAE PPO      (dashed C_RED, alpha=0.65, mean+std)

abl_no_gae_coherence.pdf:
  GAE    success  (solid  C_ORANGE_LIGHT, marker s)
  GAE    failure  (solid  C_COHFAIL,      marker ^)
  No-GAE success  (dashed C_ORANGE_LIGHT, marker s)
  No-GAE failure  (dashed C_COHFAIL,      marker ^)

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ablations/ablation_no_gae_ppo.py \\
        --ppo_root ppo_experiment_root \\
        --no_gae_ppo_root no_gae_ppo_root
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
    C_RED,
    C_ORANGE_LIGHT,
    C_COHFAIL,
    METRIC_MARKERS,
    MARKER_EVERY,
    PPO_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Ablation graphs", "PPO")


def _load_all_seed_results(experiment_root: str, algorithm: str = "ppo") -> dict:
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


def plot(ppo_root: str, no_gae_ppo_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list[str]:
    gae_results = _load_all_seed_results(ppo_root)
    no_gae_results = _load_all_seed_results(no_gae_ppo_root, algorithm="ppo_no_gae")

    if not gae_results and not no_gae_results:
        print("WARNING: no data for no-GAE ablation — skipping")
        return []

    mk_opp, ms_opp = METRIC_MARKERS["opposition_score"]
    mk_suc, ms_suc = METRIC_MARKERS["coherence_success"]
    mk_fai, ms_fai = METRIC_MARKERS["coherence_failure"]

    os.makedirs(_GRAPHS_ROOT, exist_ok=True)
    paths = []

    # --- Opposition plot ---
    fig, ax = plt.subplots(figsize=(12, 4.5))

    if not gae_results:
        print("  WARNING: no GAE PPO data found — opposition plot may be empty")
    if not no_gae_results:
        print(f"  WARNING: no no-GAE data found at {no_gae_ppo_root} — second line will be missing")

    _C_PINK = "#e8609a"

    for results, ls, label_suffix, color in [
        (gae_results,    "-",  "(with GAE)",  C_RED),
        (no_gae_results, "--", "(no GAE)",    _C_PINK),
    ]:
        if not results:
            continue
        step_grid = _build_step_grid(results)
        mean, std, n = _avg_metric_across_seeds(results, "opposition_score", step_grid)
        fin = np.isfinite(mean)
        if fin.any():
            _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                              color=color,
                              label=f"Opposition {label_suffix} ({n} seeds)",
                              linestyle=ls,
                              marker=mk_opp, markersize=ms_opp)

    ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(0, 1_000_000)
    _opp_ach = _ach_steps_from_all_results(gae_results or no_gae_results)
    ach_handles = _add_all_achievement_markers(ax, _opp_ach, clusters=PPO_CLUSTERS)
    ax.set_ylabel("Opposition Score")
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

    _add_seed_note(fig, max(len(gae_results), len(no_gae_results)))
    opp_path = os.path.join(_GRAPHS_ROOT, "abl_no_gae_opposition.pdf")
    fig.savefig(opp_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {opp_path}")
    paths.append(opp_path)

    # --- Coherence plot ---
    fig, ax = plt.subplots(figsize=(12, 4.5))

    _C_PINK = "#e8609a"
    _C_PINK_DARK = "#a03060"

    _colors_by_variant = {
        "(with GAE)": [C_ORANGE_LIGHT, C_COHFAIL],
        "(no GAE)":   [_C_PINK,        _C_PINK_DARK],
    }

    for results, ls, label_suffix in [
        (gae_results,    "-",  "(with GAE)"),
        (no_gae_results, "--", "(no GAE)"),
    ]:
        if not results:
            continue
        step_grid = _build_step_grid(results)
        c_suc, c_fai = _colors_by_variant[label_suffix]

        for field, color, mk, ms, group_label in [
            ("coherence_success", c_suc, mk_suc, ms_suc, "Success"),
            ("coherence_failure", c_fai, mk_fai, ms_fai, "Failure"),
        ]:
            mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
            fin = np.isfinite(mean)
            if fin.any():
                _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                                  color=color,
                                  label=f"{group_label} {label_suffix}",
                                  linestyle=ls,
                                  marker=mk, markersize=ms)

    ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    _coh_ach = _ach_steps_from_all_results(gae_results or no_gae_results)
    ach_handles = _add_all_achievement_markers(ax, _coh_ach, clusters=PPO_CLUSTERS)
    ax.set_ylabel("Gradient Coherence")
    _rq_fmt_millions(ax)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.18), ncol=4)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), ncol=4)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_seed_note(fig, max(len(gae_results), len(no_gae_results)))
    coh_path = os.path.join(_GRAPHS_ROOT, "abl_no_gae_coherence.pdf")
    fig.savefig(coh_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {coh_path}")
    paths.append(coh_path)

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True)
    parser.add_argument("--no_gae_ppo_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    ppo = (os.path.join(_ROOT, args.ppo_root)
           if not os.path.isabs(args.ppo_root) else args.ppo_root)
    no_gae = (os.path.join(_ROOT, args.no_gae_ppo_root)
              if not os.path.isabs(args.no_gae_ppo_root) else args.no_gae_ppo_root)
    plot(ppo, no_gae, dpi=args.dpi)


if __name__ == "__main__":
    main()
