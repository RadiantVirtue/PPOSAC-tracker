"""Outcome-Conditioned Atom Weighting + MORA — comparison experiment.

Runs a 2×2 grid of Rainbow variants:

  | Config              | k  | Outcome weighting | Description                     |
  |---------------------|----|-------------------|---------------------------------|
  | per_baseline        | 0  | Off               | Standard PER (same as MORA k=0) |
  | mora_k{k}           | >0 | Off               | MORA sampling only              |
  | outcome_baseline    | 0  | On                | Outcome weighting, PER sampling |
  | outcome_mora_k{k}   | >0 | On                | Outcome weighting + MORA        |

Outcome-conditioned atom weighting (Part 3):
  Loss becomes L_i = -∑_j w_j(c_i) · m_j · log p_j
  where w_j = C·softmax(±z/τ), neutral w_j=1.0 exactly (gradient magnitude preserved).
  Episode outcome c_i ∈ {success, neutral, failure} uses the same percentile
  partitioning as MORAPriorityTracker (top/bottom percentile_x%).

Usage:
    python rainbow/outcome_mora_train_and_analyze.py
    python rainbow/outcome_mora_train_and_analyze.py --k-values 0 5 10 25 --seeds 1 2 3
    python rainbow/outcome_mora_train_and_analyze.py --k-values 0 5 --seeds 1 --T-max 50000
    python rainbow/outcome_mora_train_and_analyze.py --plot-only

Outputs (under --experiment-root, default outcome_mora_results/):
    outcome_mora_results/
        {config}/seed_{s}/logs/
            rainbowreturnlog.txt
            mora_modifier_log.csv     (MORA runs only)
        outcome_mora_return_comparison.png
        outcome_mora_modifier_curves.png
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.mora_tracker import MORAPriorityTracker, OutcomeTracker
from rainbow.train import build_parser, main_rainbow
# Re-use plotting and loading helpers from mora_train_and_analyze
from rainbow.mora_train_and_analyze import (
    _load_returns,
    _load_modifier,
    _scan_existing_runs,
    _smooth,
    _plot_modifier_curves,
)



@dataclass
class OutcomeMoraArgs:
    """Outcome-conditioned atom weighting + MORA comparison experiment."""
    k_values: list[int] = field(default_factory=lambda: [0, 5, 10, 25])
    """k=0 = standard PER; k>0 = MORA with rolling window k."""
    use_outcome_weighting: list[bool] = field(default_factory=lambda: [False, True])
    """Whether to enable outcome-conditioned atom weighting."""
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    plot_only: bool = False
    """Scan experiment_root for all completed runs and plot without re-running."""
    T_max: int = 1_000_000
    checkpoint_interval: int = 100_000
    experiment_root: str = "outcome_mora_results"
    device: str = "cpu"
    # Rainbow hyperparameters
    hidden_size: int = 512
    atoms: int = 51
    architecture: str = "canonical"
    memory_capacity: int = 500_000
    learning_rate: float = 0.0000625
    batch_size: int = 32
    # Outcome weighting
    outcome_tau: float = 10.0
    """Temperature τ for atom weights.  exp((V_max-V_min)/τ) sets max ratio.
    V_min=-10, V_max=10 → τ=10 gives ~7.4× range (gentle); τ=5 gives ~55× (moderate)."""
    # MORA params
    mora_epsilon: float = 0.01
    mora_percentile_x: int = 25
    mora_window: int = 25
    # Plotting
    smoothing_window: int = 50


def _config_name(k: int, outcome: bool) -> str:
    if k == 0 and not outcome:
        return "per_baseline"
    if k == 0 and outcome:
        return "outcome_baseline"
    if k > 0 and not outcome:
        return f"mora_k{k}"
    return f"outcome_mora_k{k}"



def _build_rainbow_args(args: OutcomeMoraArgs, seed: int, seed_root: str,
                        outcome: bool):
    parser = build_parser()
    ns = parser.parse_args([])
    ns.seed = seed
    ns.T_max = args.T_max
    ns.checkpoint_interval = args.checkpoint_interval
    ns.experiment_root = seed_root
    ns.hidden_size = args.hidden_size
    ns.atoms = args.atoms
    ns.architecture = args.architecture
    ns.memory_capacity = args.memory_capacity
    ns.learning_rate = args.learning_rate
    ns.batch_size = args.batch_size
    ns.disable_cuda = (args.device == "cpu")
    ns.model = None
    if outcome:
        ns.outcome_tau = args.outcome_tau
    # outcome_tau absent → Agent.__init__ uses getattr(args, 'outcome_tau', None) → None
    return ns



def _run_one(args: OutcomeMoraArgs, k: int, seed: int, outcome: bool) -> str:
    config = _config_name(k, outcome)
    seed_root = os.path.join(args.experiment_root, config, f"seed_{seed}")
    os.makedirs(seed_root, exist_ok=True)

    rainbow_args = _build_rainbow_args(args, seed, seed_root, outcome)

    mora_tracker = None
    if k > 0:
        mora_tracker = MORAPriorityTracker(
            k=k,
            epsilon=args.mora_epsilon,
            percentile_x=args.mora_percentile_x,
            window=args.mora_window,
        )

    outcome_tracker = None
    if outcome:
        outcome_tracker = OutcomeTracker(
            percentile_x=args.mora_percentile_x,
            window=args.mora_window,
        )

    tag = f"{config} | seed={seed} | τ={args.outcome_tau if outcome else 'off'}"
    print(f"\n=== {tag} | T_max={args.T_max:,} ===")
    main_rainbow(rainbow_args, on_checkpoint_saved=None,
                 mora_tracker=mora_tracker, outcome_tracker=outcome_tracker)

    return os.path.join(seed_root, "logs", "rainbowreturnlog.txt")



def _plot_return_comparison(
    all_returns: dict[str, list[np.ndarray]],
    out_path: str,
    smoothing_window: int,
):
    """Plot smoothed episode return for each config, ±1 std shading over seeds."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colours = plt.cm.tab10.colors

    # Consistent ordering: per_baseline, outcome_baseline, mora_k*, outcome_mora_k*
    def _sort_key(cfg: str) -> tuple:
        if cfg == "per_baseline":       return (0, 0)
        if cfg == "outcome_baseline":   return (0, 1)
        if cfg.startswith("mora_k"):    return (1, int(cfg[len("mora_k"):]))
        if cfg.startswith("outcome_mora_k"): return (2, int(cfg[len("outcome_mora_k"):]))
        return (3, 0)

    for idx, config in enumerate(sorted(all_returns, key=_sort_key)):
        seed_returns = all_returns[config]
        colour = colours[idx % len(colours)]
        smoothed = [_smooth(r, smoothing_window) for r in seed_returns if len(r) > 0]
        if not smoothed:
            continue
        min_len = min(len(s) for s in smoothed)
        arr = np.stack([s[:min_len] for s in smoothed])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=config, color=colour, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colour)

    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Outcome-Conditioned Atom Weighting + MORA — Episode Return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")



def _scan_outcome_runs(
    experiment_root: str,
) -> dict[str, list[int]]:
    """Scan for all completed config/seed runs under experiment_root.

    Returns {config_name: [seed, ...]} for every seed with a non-empty
    rainbowreturnlog.txt.
    """
    result: dict[str, list[int]] = {}
    if not os.path.isdir(experiment_root):
        return result
    for config in sorted(os.listdir(experiment_root)):
        config_dir = os.path.join(experiment_root, config)
        if not os.path.isdir(config_dir):
            continue
        for seed_entry in sorted(os.listdir(config_dir)):
            if not seed_entry.startswith("seed_"):
                continue
            try:
                seed = int(seed_entry.split("_")[1])
            except (IndexError, ValueError):
                continue
            log = os.path.join(config_dir, seed_entry, "logs", "rainbowreturnlog.txt")
            if os.path.exists(log) and os.path.getsize(log) > 0:
                result.setdefault(config, []).append(seed)
    return result



def run(args: OutcomeMoraArgs):
    os.makedirs(args.experiment_root, exist_ok=True)

    if args.plot_only:
        discovered = _scan_outcome_runs(args.experiment_root)
        if not discovered:
            print(f"No completed runs found under {args.experiment_root}/")
            return
        print(f"Found configs: {sorted(discovered)}")
        for cfg, seeds in sorted(discovered.items()):
            print(f"  {cfg}: seeds {seeds}")

        all_returns: dict[str, list[np.ndarray]] = {}
        modifier_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for config, seeds in discovered.items():
            all_returns[config] = []
            for seed in seeds:
                log = os.path.join(args.experiment_root, config, f"seed_{seed}",
                                   "logs", "rainbowreturnlog.txt")
                all_returns[config].append(_load_returns(log))
            # Modifier curve from first seed only
            first_seed = seeds[0]
            mod_log = os.path.join(args.experiment_root, config, f"seed_{first_seed}",
                                   "logs", "mora_modifier_log.csv")
            if os.path.exists(mod_log):
                modifier_data[config] = _load_modifier(mod_log)
    else:
        all_returns = {}
        modifier_data = {}

        for outcome in args.use_outcome_weighting:
            for k in args.k_values:
                config = _config_name(k, outcome)
                all_returns[config] = []

                for seed in args.seeds:
                    return_log = _run_one(args, k, seed, outcome)
                    all_returns[config].append(_load_returns(return_log))

                    if seed == args.seeds[0] and k > 0:
                        mod_log = os.path.join(
                            args.experiment_root, config, f"seed_{seed}",
                            "logs", "mora_modifier_log.csv",
                        )
                        if os.path.exists(mod_log):
                            modifier_data[config] = _load_modifier(mod_log)

    # Generate plots
    return_plot = os.path.join(args.experiment_root, "outcome_mora_return_comparison.png")
    modifier_plot = os.path.join(args.experiment_root, "outcome_mora_modifier_curves.png")

    _plot_return_comparison(all_returns, return_plot, args.smoothing_window)
    if modifier_data:
        _plot_modifier_curves(modifier_data, modifier_plot)

    print(f"\nResults in: {args.experiment_root}/")


def main():
    run(tyro.cli(OutcomeMoraArgs))


if __name__ == "__main__":
    main()
