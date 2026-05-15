"""Plot no-GAE ablation comparison: PPO (GAE, λ=0.95) vs PPO (TD(0)).

Produces a single stacked figure with 5 subplots:
  opposition score / coherence success / coherence failure /
  gradient magnitude success / gradient magnitude failure

Reads analysis JSONs via shared/graphing.py infrastructure.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared.graphing  # noqa: F401 — applies global plt.rcParams on import
from shared.graphing import aggregate_periodic, load_all_data


C_GAE = "#1f77b4"   # blue  — PPO with GAE (λ=0.95)
C_TD0 = "#d62728"   # red   — PPO with no GAE, TD(0)


_ROWS = [
    ("opposition_score",           "Opposition score"),
    ("coherence_success",          "Coherence (success)"),
    ("coherence_failure",          "Coherence (failure)"),
    ("gradient_magnitude_success", "Gradient magnitude (success)"),
    ("gradient_magnitude_failure", "Gradient magnitude (failure)"),
]


def _series(agg_rows: list[dict], metric: str):
    """Extract (steps_k, means, stds) from aggregated rows, steps in thousands."""
    steps, means, stds = [], [], []
    for row in agg_rows:
        m = row.get(f"{metric}_mean")
        s = row.get(f"{metric}_std") or 0.0
        if m is not None:
            steps.append(row["step"] / 1_000)
            means.append(m)
            stds.append(s)
    return np.array(steps), np.array(means), np.array(stds)


def _plot_series(ax, steps, means, stds, color, label=None):
    if len(steps) == 0:
        return
    ax.plot(steps, means, color=color, linewidth=2, label=label)
    ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.15)



@dataclass
class Args:
    no_gae_root: str = "experiment_root_no_gae"
    ppo_root:    str = "ppo_experiment_root"
    output_dir:  str = ""     # defaults to no_gae_root
    dpi:         int = 150


def run(args: Args):
    out_dir = args.output_dir or args.no_gae_root
    os.makedirs(out_dir, exist_ok=True)

    gae_raw = load_all_data(args.ppo_root,    algorithm="ppo")
    td0_raw = load_all_data(args.no_gae_root, algorithm="ppo_no_gae")

    gae_agg = aggregate_periodic(gae_raw["periodic"])
    td0_agg = aggregate_periodic(td0_raw["periodic"])

    n_seeds = max(
        max((r["n_seeds"] for r in gae_agg), default=0),
        max((r["n_seeds"] for r in td0_agg), default=0),
    )

    fig, axes = plt.subplots(len(_ROWS), 1, figsize=(8, 16), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, _ROWS)):
        gae_steps, gae_means, gae_stds = _series(gae_agg, metric)
        td0_steps, td0_means, td0_stds = _series(td0_agg, metric)

        _plot_series(ax, gae_steps, gae_means, gae_stds, C_GAE,
                     label="GAE (λ=0.95)" if i == 0 else None)
        _plot_series(ax, td0_steps, td0_means, td0_stds, C_TD0,
                     label="TD(0)"        if i == 0 else None)

        ax.set_ylabel(ylabel)

        if i == 0:
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Training step (thousands)")

    fig.text(
        0.98, 0.005,
        f"mean ± std, n={n_seeds} seeds",
        ha="right", va="bottom", fontsize=10, color="#555555",
        transform=fig.transFigure,
    )

    out_path = os.path.join(out_dir, "no_gae_comparison.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
