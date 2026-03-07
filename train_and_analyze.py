"""Train and analyse — top-level dispatcher with grouped arguments.

Usage:
    python train_and_analyze.py --algorithm ppo   [--shared.* --ppo.*]
    python train_and_analyze.py --algorithm rainbow [--shared.* --rainbow.*]

Sub-scripts can also be invoked directly:
    python ppo/train_and_analyze.py     [args...]
    python rainbow/train_and_analyze.py [args...]
"""
from dataclasses import dataclass, field

import tyro


# ── Argument groups ───────────────────────────────────────────────────────────

@dataclass
class Shared:
    """Arguments shared by both algorithms."""
    env_id: str = "MiniGrid-KeyCorridorS3R3-v0"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    experiment_root: str = "Rainbow-Proof-of-Concept-Runs"
    n_eval_episodes: int = 500
    run_rsa: bool = True
    split_mode: str = "percentile"   # "percentile" or "eps"
    percentile_x: int = 25
    auto_push: bool = True
    device: str = "cuda"
    analyze_every: int = 5           # analyse every Nth saved checkpoint


@dataclass
class PPO:
    """PPO-specific training arguments."""
    total_episodes: int = 100_000
    checkpoint_freq: int = 500           # episodes between checkpoints; 0 = auto (total/10)
    checkpoint_achievements: bool = True
    num_procs: int = 16
    frames_per_proc: int = 256
    entropy_coef: float = 0.02


@dataclass
class Rainbow:
    """Rainbow DQN-specific training arguments."""
    total_timesteps: int = 3_000_000
    checkpoint_freq: int = 0             # episodes between checkpoints; 0 = disabled
    checkpoint_step_freq: int = 5_000    # steps between checkpoints; 0 = disabled
    num_envs: int = 16


@dataclass
class Args:
    algorithm: str = "rainbow"           # "ppo" or "rainbow"
    shared: Shared = field(default_factory=Shared)
    ppo: PPO = field(default_factory=PPO)
    rainbow: Rainbow = field(default_factory=Rainbow)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    s = args.shared

    if args.algorithm == "ppo":
        from ppo.train_and_analyze import run as ppo_run
        from ppo.train_and_analyze import Args as PPOScriptArgs
        ppo_run(PPOScriptArgs(
            env_id=s.env_id,
            seeds=s.seeds,
            experiment_root=s.experiment_root,
            n_eval_episodes=s.n_eval_episodes,
            run_rsa=s.run_rsa,
            split_mode=s.split_mode,
            percentile_x=s.percentile_x,
            auto_push=s.auto_push,
            device=s.device,
            analyze_every=s.analyze_every,
            total_episodes=args.ppo.total_episodes,
            checkpoint_freq=args.ppo.checkpoint_freq,
            checkpoint_achievements=args.ppo.checkpoint_achievements,
            num_procs=args.ppo.num_procs,
            frames_per_proc=args.ppo.frames_per_proc,
            entropy_coef=args.ppo.entropy_coef,
        ))

    elif args.algorithm == "rainbow":
        from rainbow.train_and_analyze import run as rainbow_run
        from rainbow.train_and_analyze import Args as RainbowScriptArgs
        rainbow_run(RainbowScriptArgs(
            env_id=s.env_id,
            seeds=s.seeds,
            experiment_root=s.experiment_root,
            n_eval_episodes=s.n_eval_episodes,
            run_rsa=s.run_rsa,
            split_mode=s.split_mode,
            percentile_x=s.percentile_x,
            auto_push=s.auto_push,
            device=s.device,
            analyze_every=s.analyze_every,
            total_timesteps=args.rainbow.total_timesteps,
            checkpoint_freq=args.rainbow.checkpoint_freq,
            checkpoint_step_freq=args.rainbow.checkpoint_step_freq,
            num_envs=args.rainbow.num_envs,
        ))

    else:
        raise ValueError(f"--algorithm must be 'ppo' or 'rainbow', got {args.algorithm!r}")


if __name__ == "__main__":
    main()
