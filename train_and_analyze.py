"""Train and analyse — PPO on Crafter.

Usage:
    python train_and_analyze.py  [--shared.* --ppo.*]

Sub-script can also be invoked directly:
    python ppo/train_and_analyze.py  [args...]
"""
from dataclasses import dataclass, field

import tyro


@dataclass
class Shared:
    """Arguments shared across runs."""
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    experiment_root: str = "ppo_experiment_root"
    n_eval_episodes: int = 1000
    split_mode: str = "percentile"
    percentile_x: int = 25
    auto_push: bool = False
    device: str = "cuda"
    analyze_every: int = 1


@dataclass
class PPO:
    """PPO-specific training arguments."""
    total_timesteps: int = 3000000
    checkpoint_freq: int = 50_000     # timesteps between checkpoints; 0 = off
    checkpoint_achievements: bool = True
    num_procs: int = 16
    n_steps: int = 128
    ent_coef: float = 0.01


@dataclass
class Args:
    shared: Shared = field(default_factory=Shared)
    ppo: PPO = field(default_factory=PPO)


def main():
    args = tyro.cli(Args)
    s = args.shared

    from ppo.train_and_analyze import run as ppo_run
    from ppo.train_and_analyze import Args as PPOScriptArgs
    ppo_run(PPOScriptArgs(
        seeds=s.seeds,
        experiment_root=s.experiment_root,
        n_eval_episodes=s.n_eval_episodes,
        split_mode=s.split_mode,
        percentile_x=s.percentile_x,
        auto_push=s.auto_push,
        device=s.device,
        analyze_every=s.analyze_every,
        total_timesteps=args.ppo.total_timesteps,
        checkpoint_freq=args.ppo.checkpoint_freq,
        checkpoint_achievements=args.ppo.checkpoint_achievements,
        num_procs=args.ppo.num_procs,
        n_steps=args.ppo.n_steps,
        ent_coef=args.ppo.ent_coef,
    ))


if __name__ == "__main__":
    main()
