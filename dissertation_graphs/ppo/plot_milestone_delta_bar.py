"""dissertation_graphs/ppo/plot_milestone_delta_bar.py

PPO milestone-normalised d_opp bar chart.

For each achievement, computes d_opp = milestone_opposition - nearest_periodic_opposition,
where "nearest" is matched by episode count (not step count) to control for
the underlying training trend.

Key finding visualised:
  Tier-2 achievements (eat_cow +0.65, defeat_zombie +0.56,
  make_wood_pickaxe +0.48, collect_stone +0.44, make_wood_sword +0.39) show
  large positive spikes, indicating discovery events force abrupt realignment
  of the gradient signal.  defeat_skeleton (+0.07) is anomalous (seed 4:
  opposition -0.81 at that milestone).  Tier-1 and Tier-3+ milestones show
  smaller or inconsistent deltas.

Usage:
    python dissertation_graphs/ppo/plot_milestone_delta_bar.py \\
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


# ── d_opp computation ──────────────────────────────────────────────────────────

def _compute_d_opp(experiment_root: str, algorithm: str) -> dict:
    """Compute d_opp = milestone_opp - nearest_periodic_opp per achievement.

    Nearest periodic is matched by episode count within the same seed so
    that the delta reflects the achievement-specific spike above the local
    training trend, not the absolute value.

    Returns:
        {ach_name: {"mean": float, "std": float, "n": int}}
    """
    data = load_all_data(experiment_root, algorithm)
    periodic = data["periodic"]   # {step: [(seed_id, record), ...]}
    milestone = data["milestone"]  # {ach_name: [(seed_id, record), ...]}

    # Build per-seed sorted (episode, opposition_score) from periodic data
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
    for seed_id in seed_periodic:
        seed_periodic[seed_id].sort()

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
            # Find periodic record whose episode count is closest to the milestone
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

def plot(ppo_root: str, out_dir: str, dpi: int = 150):
    d_opp_data = _compute_d_opp(ppo_root, "ppo")
    if not d_opp_data:
        print("ERROR: no milestone d_opp data computed.")
        return None

    # Sort by mean d_opp descending; use canonical tier ordering as tiebreak
    tier_rank = {a: i for i, a in enumerate(ACHIEVEMENT_ORDER)}
    ordered = sorted(
        d_opp_data.items(),
        key=lambda x: (-x[1]["mean"], tier_rank.get(x[0], 99)),
    )

    names = [a.replace("_", " ") for a, _ in ordered]
    means = np.array([v["mean"] for _, v in ordered])
    stds  = np.array([v["std"]  for _, v in ordered])
    colors = [TIER_COLORS.get(ACHIEVEMENT_TIERS.get(a, 1), "#999999")
              for a, _ in ordered]

    fig_height = max(4, len(ordered) * 0.45)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y_pos = np.arange(len(ordered))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors,
                   ecolor="#444444", capsize=3, alpha=0.85, height=0.7)

    ax.axvline(0.0, color="grey", linestyle="-", lw=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("d_opp  (milestone opposition \u2212 nearest periodic opposition)")
    ax.set_title(
        "PPO: Normalised Opposition Spike at Achievement Milestones\n"
        "d_opp = milestone opp \u2212 episode-matched periodic opp  (mean +/- 1 std)"
    )
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tier legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=TIER_COLORS[t], label=f"Tier {t}")
               for t in sorted(TIER_COLORS)]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "milestone_delta_bar_ppo.png")
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
