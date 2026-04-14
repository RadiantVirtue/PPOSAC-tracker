"""dissertation_graphs/shared/plot_milestone_delta_comparison.py

Cross-algorithm milestone d_opp comparison: PPO (top panel) vs Rainbow
(bottom panel), same achievement ordering, same tier colours.

This is the core contrast graph for the dissertation.  PPO shows large
positive spikes at Tier-2 discovery events (d_opp up to +0.65), while
Rainbow stays within +/-0.12 for all achievements -- confirming its
gradient structure is already saturated before any achievement is first
unlocked in live rollouts.

d_opp = milestone_opposition - nearest_periodic_opposition
  "Nearest" is matched by episode count (not step count) per seed so
  that the delta is relative to the local training trend.

Achievements are ordered by PPO d_opp (descending).  Rainbow bars use
the same ordering so the two panels can be read side-by-side.  A shared
achievement is one present in BOTH algorithms' milestone data; an
achievement that only one algorithm logged is still shown with NaN for
the missing algorithm (no bar drawn).

Usage:
    python dissertation_graphs/shared/plot_milestone_delta_comparison.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_experiment_root
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Path to PPOSAC-tracker root ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    load_all_data,
    ACHIEVEMENT_TIERS,
    ACHIEVEMENT_ORDER,
    TIER_COLORS,
)


# ── d_opp computation (shared between both algorithms) ────────────────────────

def _compute_d_opp(experiment_root: str, algorithm: str) -> dict:
    """Compute d_opp = milestone_opp - nearest_periodic_opp per achievement.

    Returns:
        {ach_name: {"mean": float, "std": float, "n": int}}
    """
    data = load_all_data(experiment_root, algorithm)
    periodic = data["periodic"]
    milestone = data["milestone"]

    seed_periodic: dict[int, list] = {}
    for _step, entries in periodic.items():
        for seed_id, rec in entries:
            ep = rec.get("episode")
            opp = rec.get("opposition_score")
            if ep is None or opp is None:
                continue
            try:
                seed_periodic.setdefault(seed_id, []).append(
                    (float(ep), float(opp))
                )
            except (TypeError, ValueError):
                pass
    for sid in seed_periodic:
        seed_periodic[sid].sort()

    result = {}
    for ach, entries in milestone.items():
        d_opps = []
        for seed_id, rec in entries:
            ep_m = rec.get("episode")
            opp_m = rec.get("opposition_score")
            if ep_m is None or opp_m is None:
                continue
            try:
                ep_m = float(ep_m)
                opp_m = float(opp_m)
            except (TypeError, ValueError):
                continue
            pairs = seed_periodic.get(seed_id, [])
            if not pairs:
                continue
            nearest_opp = min(pairs, key=lambda x: abs(x[0] - ep_m))[1]
            d_opps.append(opp_m - nearest_opp)
        if d_opps:
            result[ach] = {
                "mean": float(np.mean(d_opps)),
                "std": float(np.std(d_opps)) if len(d_opps) > 1 else 0.0,
                "n": len(d_opps),
            }
    return result


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(ppo_root: str, rainbow_root: str, out_dir: str, dpi: int = 150):
    ppo_data = _compute_d_opp(ppo_root, "ppo")
    rbw_data = _compute_d_opp(rainbow_root, "rainbow")

    if not ppo_data and not rbw_data:
        print("ERROR: no milestone data found for either algorithm.")
        return None

    # ── Achievement ordering ────────────────────────────────────────────────────
    # Union of all achievements seen, ordered by PPO d_opp descending.
    # Use canonical tier ordering as a tiebreak for missing-PPO achievements.
    tier_rank = {a: i for i, a in enumerate(ACHIEVEMENT_ORDER)}
    all_achs = sorted(
        set(ppo_data) | set(rbw_data),
        key=lambda a: (-(ppo_data.get(a, {}).get("mean", -99)), tier_rank.get(a, 99)),
    )

    ppo_means = np.array([ppo_data.get(a, {}).get("mean", np.nan) for a in all_achs])
    ppo_stds  = np.array([ppo_data.get(a, {}).get("std",  0.0)    for a in all_achs])
    rbw_means = np.array([rbw_data.get(a, {}).get("mean", np.nan) for a in all_achs])
    rbw_stds  = np.array([rbw_data.get(a, {}).get("std",  0.0)    for a in all_achs])

    labels = [a.replace("_", " ") for a in all_achs]
    colors = [TIER_COLORS.get(ACHIEVEMENT_TIERS.get(a, 1), "#999999")
              for a in all_achs]
    y_pos = np.arange(len(all_achs))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig_height = max(5, len(all_achs) * 0.5)
    fig, (ax_ppo, ax_rbw) = plt.subplots(
        1, 2, figsize=(14, fig_height), sharey=True
    )
    fig.subplots_adjust(wspace=0.04)

    def _draw_panel(ax, means, stds, title):
        fin = np.isfinite(means)
        # Draw bars only where data exists
        ax.barh(y_pos[fin], means[fin], xerr=stds[fin],
                color=[colors[i] for i in range(len(all_achs)) if fin[i]],
                ecolor="#444444", capsize=3, alpha=0.85, height=0.7)
        # Grey placeholder for missing bars
        if not fin.all():
            missing = ~fin
            ax.barh(y_pos[missing], np.zeros(missing.sum()),
                    color="#cccccc", height=0.7, alpha=0.5)
        ax.axvline(0.0, color="grey", linestyle="-", lw=0.9, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(
            "d_opp  (milestone opp \u2212 periodic opp)",
            fontsize=9,
        )

    _draw_panel(ax_ppo, ppo_means, ppo_stds, "PPO")
    _draw_panel(ax_rbw, rbw_means, rbw_stds, "Rainbow")

    # Shared y-axis labels on left panel only
    ax_ppo.set_yticks(y_pos)
    ax_ppo.set_yticklabels(labels, fontsize=9)

    # Link x-axis limits so visual scale is comparable
    all_finite = np.concatenate([
        ppo_means[np.isfinite(ppo_means)],
        rbw_means[np.isfinite(rbw_means)],
    ])
    if len(all_finite):
        margin = max(0.1, 0.15 * (all_finite.max() - all_finite.min()))
        x_lo = all_finite.min() - margin
        x_hi = all_finite.max() + margin
        ax_ppo.set_xlim(x_lo, x_hi)
        ax_rbw.set_xlim(x_lo, x_hi)

    # Tier legend
    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], label=f"Tier {t}")
                    for t in sorted(TIER_COLORS)]
    fig.legend(handles=tier_handles, fontsize=8, loc="lower center",
               ncol=len(tier_handles), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "PPO vs Rainbow: Normalised Opposition Spike at Achievement Milestones\n"
        "d_opp = milestone opp \u2212 episode-matched periodic opp  (mean +/- 1 std)",
        fontsize=12,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "milestone_delta_comparison.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True,
                        help="Path to PPO experiment root")
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--out_dir", default=os.path.join(_HERE, "output"),
                        help="Directory to save the PNG (default: shared/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = os.path.join(_ROOT, args.ppo_root) if not os.path.isabs(args.ppo_root) else args.ppo_root
    rbw_root = os.path.join(_ROOT, args.rainbow_root) if not os.path.isabs(args.rainbow_root) else args.rainbow_root
    plot(ppo_root, rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
