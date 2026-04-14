"""dissertation_graphs/ppo/plot_rq3_rsa_alignment.py

RQ3 dissertation graph: PPO RSA alignment (Spearman rho) averaged across
5 seeds, with a horizontal zero-line and Tier-3 achievement unlock timings
marked as vertical orange annotations.

Key finding visualised:
  - Early training RSA is negative (~-0.10 to -0.29 avg), indicating
    representations organised by difficulty/recency rather than function.
  - Sign flip around step 1.0-1.3M coincides with Tier-3 capability
    emergence (make_stone_pickaxe ~ep 4919, make_stone_sword ~ep 4113).
  - Late training stabilises at +0.10-0.20.

RSA extraction: tries single-field "rsa_alignment" first (old schema),
falls back to nanmean of the four group fields:
  rsa_alignment_fighting / resource / crafting / housing

Usage:
    python dissertation_graphs/ppo/plot_rq3_rsa_alignment.py \\
        --ppo_root ppo_experiment_root
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# ── Path to PPOSAC-tracker root ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _ach_steps_from_all_results,
    _build_step_grid,
    _make_shaded_line,
    _rq_fmt_millions,
    ACHIEVEMENT_TIERS,
    TIER_COLORS,
    C_TEAL,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_checkpoint_results(seed_dir: str, algorithm: str) -> list:
    log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
    if not os.path.isdir(log_dir):
        return []
    results = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            rec = load_analysis_results(fpath)
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
        seed_id = int(m.group(1))
        out[seed_id] = _load_checkpoint_results(
            os.path.join(experiment_root, name), algorithm
        )
    return out


# ── RSA extractor ──────────────────────────────────────────────────────────────

def _extract_rsa_for_seed(checkpoint_results: list) -> tuple[list, list]:
    """Return (steps, rsa_values) for periodic checkpoints.

    Tries single-field "rsa_alignment" first (old schema used by PPO
    analysis pipeline).  Falls back to nanmean of the four group-level
    fields (rsa_alignment_fighting / resource / crafting / housing) if the
    single field is absent or None.
    """
    steps, vals = [], []
    for label, r in checkpoint_results:
        m = re.search(r"step\s*([\d,]+)", label, re.IGNORECASE)
        if not m:
            continue
        step = int(m.group(1).replace(",", ""))

        # Try single field
        v = r.get("rsa_alignment")
        if v is not None:
            try:
                f = float(v)
                if math.isfinite(f):
                    steps.append(step)
                    vals.append(f)
                    continue
            except (TypeError, ValueError):
                pass

        # Fallback: average the four group fields
        group_vals = []
        for k in ("rsa_alignment_fighting", "rsa_alignment_resource",
                  "rsa_alignment_crafting", "rsa_alignment_housing"):
            gv = r.get(k)
            if gv is not None:
                try:
                    gf = float(gv)
                    if math.isfinite(gf):
                        group_vals.append(gf)
                except (TypeError, ValueError):
                    pass
        if group_vals:
            steps.append(step)
            vals.append(float(np.mean(group_vals)))

    return steps, vals


def _avg_rsa_across_seeds(
    all_seed_results: dict,
    step_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate RSA alignment from each seed onto step_grid and average."""
    arrs = []
    for checkpoint_results in all_seed_results.values():
        steps, vals = _extract_rsa_for_seed(checkpoint_results)
        if len(steps) < 2:
            continue
        arrs.append(np.interp(step_grid, steps, vals,
                               left=np.nan, right=np.nan))
    if not arrs:
        nan = np.full(len(step_grid), np.nan)
        return nan, nan, 0
    mat = np.vstack(arrs)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), len(arrs)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(ppo_root: str, out_dir: str, dpi: int = 150):
    results = _load_all_seed_results(ppo_root, "ppo")
    if not results:
        print("ERROR: no PPO seed data found.")
        return None

    step_grid = _build_step_grid(results)
    mean, std, n = _avg_rsa_across_seeds(results, step_grid)

    fin = np.isfinite(mean)
    if not fin.any():
        print("ERROR: no finite RSA data found in PPO JSONs.")
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))

    _make_shaded_line(ax, step_grid, mean, std,
                      color=C_TEAL, label=f"RSA alignment rho  (n={n} seeds)")

    ax.axhline(0.0, color="grey", linestyle="-", lw=1.0, alpha=0.5,
               label="rho = 0  (random alignment)")

    # ── Tier-3 achievement markers ──────────────────────────────────────────────
    ach_steps = _ach_steps_from_all_results(results)
    tier3_steps = {a: s for a, s in ach_steps.items()
                   if ACHIEVEMENT_TIERS.get(a) == 3}

    t3_color = TIER_COLORS[3]   # "#ff7f0e" orange
    tier3_handles = []
    for ach, step in sorted(tier3_steps.items(), key=lambda x: x[1]):
        ax.axvline(step, color=t3_color, linestyle=":", lw=1.1, alpha=0.75)
        short = ach.replace("_", " ")
        tier3_handles.append(
            mlines.Line2D([], [], color=t3_color, linestyle=":", lw=1.1,
                          label=f"T3: {short}")
        )

    ax.set_ylabel("RSA Alignment  (Spearman \u03c1)")
    ax.set_title(
        f"PPO Averaged ({n} seeds) — RQ3: RSA Alignment Over Training\n"
        "Negative early, crosses zero ~1.0-1.3M steps as Tier-3 achievements emerge"
    )
    _rq_fmt_millions(ax)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + tier3_handles, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rq3_rsa_alignment_ppo_averaged.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True,
                        help="Path to PPO experiment root")
    parser.add_argument("--out_dir", default=os.path.join(_HERE, "output"),
                        help="Directory to save the PNG (default: ppo/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = os.path.join(_ROOT, args.ppo_root) if not os.path.isabs(args.ppo_root) else args.ppo_root
    plot(ppo_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
