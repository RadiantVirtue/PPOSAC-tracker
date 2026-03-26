"""Train and analyse — PPO or Rainbow on Crafter.

Usage:
    python train_and_analyze.py --algorithm ppo   [--shared.* --ppo.*]
    python train_and_analyze.py --algorithm rainbow [--shared.* --rainbow.*]

Sub-scripts can also be invoked directly:
    python ppo/train_and_analyze.py  [args...]
    python rainbow/train_and_analyze.py  [args...]
"""
from dataclasses import dataclass, field

import tyro


@dataclass
class Shared:
    """Arguments shared across runs."""
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    experiment_root: str = "experiment_root"
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
    checkpoint_freq: int = 50_000
    checkpoint_achievements: bool = True
    num_procs: int = 16
    n_steps: int = 128
    ent_coef: float = 0.01


@dataclass
class Rainbow:
    """Rainbow-specific training arguments."""
    T_max: int = 10_000_000
    checkpoint_interval: int = 100_000
    hidden_size: int = 512
    atoms: int = 51
    architecture: str = "canonical"
    memory_capacity: int = 500_000
    learning_rate: float = 0.0000625
    batch_size: int = 32


@dataclass
class Args:
    algorithm: str = "ppo"   # "ppo" or "rainbow"
    shared: Shared = field(default_factory=Shared)
    ppo: PPO = field(default_factory=PPO)
    rainbow: Rainbow = field(default_factory=Rainbow)


def main():
    args = tyro.cli(Args)
    s = args.shared

    if args.algorithm == "ppo":
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

    elif args.algorithm == "rainbow":
        from rainbow.train_and_analyze import run as rainbow_run
        from rainbow.train_and_analyze import Args as RainbowScriptArgs
        rainbow_run(RainbowScriptArgs(
            seeds=s.seeds,
            experiment_root=s.experiment_root,
            n_eval_episodes=s.n_eval_episodes,
            split_mode=s.split_mode,
            percentile_x=s.percentile_x,
            auto_push=s.auto_push,
            device=s.device,
            analyze_every=s.analyze_every,
            T_max=args.rainbow.T_max,
            checkpoint_interval=args.rainbow.checkpoint_interval,
            hidden_size=args.rainbow.hidden_size,
            atoms=args.rainbow.atoms,
            architecture=args.rainbow.architecture,
            memory_capacity=args.rainbow.memory_capacity,
            learning_rate=args.rainbow.learning_rate,
            batch_size=args.rainbow.batch_size,
        ))

    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm!r}. Choose 'ppo' or 'rainbow'.")


if __name__ == "__main__":
    main()
