"""Hyperparameter ablation experiment runner (Appendix A).

Runs both Rainbow and PPO for 500k steps with 100k-interval analyses and no
achievement checkpointing, varying one hyperparameter at a time from a
pre-defined grid.  Ablation categories:

  Training hyperparameters:
    Rainbow  — learning_rate, batch_size
    PPO      — ent_coef, n_steps

  Analysis hyperparameters (both algorithms):
    percentile_x, n_eval_episodes

All results land under ablation_results/{algo}/{config_name}/seed_{s}/.

Usage:
    # Full run (all configs, 3 seeds each)
    python ablation_experiments.py

    # Dry-run: print config table without training
    python ablation_experiments.py --dry-run

    # Subset of algorithms
    python ablation_experiments.py --algorithms rainbow

    # Single config
    python ablation_experiments.py --only-configs baseline lr_low
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import tyro


# Shared defaults for all ablation runs
_SHARED_DEFAULTS = dict(
    total_timesteps=500_000,   # PPO
    T_max=500_000,             # Rainbow
    checkpoint_freq=100_000,   # PPO
    checkpoint_interval=100_000,  # Rainbow
    checkpoint_achievements=False,
    analyze_every=1,           # analyse every checkpoint → 5 analyses per run
    n_eval_episodes=500,
    split_mode="percentile",
    percentile_x=25,
    auto_push=False,
)

# Rainbow-specific defaults
_RAINBOW_DEFAULTS = dict(
    hidden_size=512,
    atoms=51,
    architecture="canonical",
    memory_capacity=500_000,
    learning_rate=0.0000625,
    batch_size=32,
)

# PPO-specific defaults
_PPO_DEFAULTS = dict(
    num_procs=16,
    n_steps=128,
    ent_coef=0.01,
)

# Grid entry: (name, algorithm, param_overrides)
# algorithm = "rainbow" | "ppo" | "both"
ABLATION_GRID: list[tuple[str, str, dict[str, Any]]] = [
    ("baseline",     "both",    {}),

    ("lr_low",       "rainbow", {"learning_rate": 0.0000625 * 0.5}),
    ("lr_high",      "rainbow", {"learning_rate": 0.0000625 * 2.0}),
    ("batch_small",  "rainbow", {"batch_size": 16}),
    ("batch_large",  "rainbow", {"batch_size": 64}),

    ("ent_low",      "ppo",     {"ent_coef": 0.001}),
    ("ent_high",     "ppo",     {"ent_coef": 0.05}),
    ("nsteps_short", "ppo",     {"n_steps": 64}),
    ("nsteps_long",  "ppo",     {"n_steps": 256}),

    ("pct_10",       "both",    {"percentile_x": 10}),
    ("pct_40",       "both",    {"percentile_x": 40}),
    ("eval_100",     "both",    {"n_eval_episodes": 100}),
    ("eval_500",     "both",    {"n_eval_episodes": 500}),   # explicit anchor
    ("eval_1000",    "both",    {"n_eval_episodes": 1000}),
]



@dataclass
class Args:
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    algorithms: list[str] = field(default_factory=lambda: ["rainbow", "ppo"])
    """Which algorithms to run. Subset of ["rainbow", "ppo"]."""
    only_configs: list[str] = field(default_factory=list)
    """If non-empty, only run configs whose name is in this list."""
    experiment_root: str = "ablation_results"
    device: str = "cpu"
    dry_run: bool = False
    """Print the config table and exit without running any training."""



def _print_grid(grid: list[tuple[str, str, dict]], algorithms: list[str]):
    """Pretty-print the ablation grid that will be executed."""
    algos_set = set(algorithms)
    rows = []
    for name, algo, overrides in grid:
        targets = algos_set & ({"rainbow", "ppo"} if algo == "both" else {algo})
        if not targets:
            continue
        for target in sorted(targets):
            rows.append((name, target, overrides or "—"))
    print(f"\n{'Config':<16} {'Algo':<10} Overrides")
    print("-" * 55)
    for name, algo, overrides in rows:
        print(f"  {name:<14} {algo:<10} {overrides}")
    print(f"\n{len(rows)} configurations total.\n")


def _merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    merged.update(overrides)
    return merged



def _run_rainbow_config(
    config_name: str,
    overrides: dict,
    seeds: list[int],
    experiment_root: str,
    device: str,
):
    """Run Rainbow for one ablation config across all seeds."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rainbow.train_and_analyze import Args as RainbowArgs, run as rainbow_run

    params = _merge(_merge(_SHARED_DEFAULTS, _RAINBOW_DEFAULTS), overrides)

    root = os.path.join(experiment_root, "rainbow", config_name)
    ra = RainbowArgs(
        seeds=seeds,
        T_max=params["T_max"],
        checkpoint_interval=params["checkpoint_interval"],
        experiment_root=root,
        n_eval_episodes=params["n_eval_episodes"],
        split_mode=params["split_mode"],
        percentile_x=params["percentile_x"],
        auto_push=params["auto_push"],
        device=device,
        analyze_every=params["analyze_every"],
        hidden_size=params["hidden_size"],
        atoms=params["atoms"],
        architecture=params["architecture"],
        memory_capacity=params["memory_capacity"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
    )
    print(f"\n{'='*60}")
    print(f"  Rainbow | config={config_name!r} | seeds={seeds}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"{'='*60}")
    rainbow_run(ra)


def _run_ppo_config(
    config_name: str,
    overrides: dict,
    seeds: list[int],
    experiment_root: str,
    device: str,
):
    """Run PPO for one ablation config across all seeds."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ppo.train_and_analyze import Args as PPOArgs, run as ppo_run

    params = _merge(_merge(_SHARED_DEFAULTS, _PPO_DEFAULTS), overrides)

    root = os.path.join(experiment_root, "ppo", config_name)
    pa = PPOArgs(
        seeds=seeds,
        total_timesteps=params["total_timesteps"],
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_achievements=params["checkpoint_achievements"],
        num_procs=params["num_procs"],
        n_steps=params["n_steps"],
        ent_coef=params["ent_coef"],
        experiment_root=root,
        n_eval_episodes=params["n_eval_episodes"],
        split_mode=params["split_mode"],
        percentile_x=params["percentile_x"],
        auto_push=params["auto_push"],
        device=device,
        analyze_every=params["analyze_every"],
    )
    print(f"\n{'='*60}")
    print(f"  PPO     | config={config_name!r} | seeds={seeds}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"{'='*60}")
    ppo_run(pa)



def run(args: Args):
    algos_set = set(args.algorithms)
    filter_set = set(args.only_configs)

    # Expand "both" entries into per-algorithm rows, then filter
    plan: list[tuple[str, str, dict]] = []
    for name, algo, overrides in ABLATION_GRID:
        if filter_set and name not in filter_set:
            continue
        targets = algos_set & ({"rainbow", "ppo"} if algo == "both" else {algo})
        for target in sorted(targets):
            plan.append((name, target, overrides))

    if not plan:
        print("No ablation configs matched the given filters.")
        return

    _print_grid([(n, a, o) for n, a, o in plan], list(algos_set))

    if args.dry_run:
        print("Dry-run mode: exiting without training.")
        return

    total = len(plan) * len(args.seeds)
    done = 0
    for config_name, algo, overrides in plan:
        if algo == "rainbow":
            _run_rainbow_config(config_name, overrides, args.seeds,
                                args.experiment_root, args.device)
        else:
            _run_ppo_config(config_name, overrides, args.seeds,
                            args.experiment_root, args.device)
        done += len(args.seeds)
        print(f"\n[ablation] Progress: {done}/{total} seed-runs complete.")

    print(f"\nAll ablation runs complete. Results in: {args.experiment_root}/")
    print("Run  python ablation_graphing.py  to generate comparison figures.")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
