"""PPO training using stable-baselines3 CnnPolicy on Crafter."""
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from wrappers import make_crafter_env
from ppo.checkpoint_gen import MilestoneTracker


@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 3_000_000
    num_procs: int = 16
    n_steps: int = 128           # rollout length per env
    batch_size: int = 256
    n_epochs: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range: float = 0.2
    features_dim: int = 64       # NatureCNN output dim (64-dim encoder)
    checkpoint_freq: int = 50_000   # timesteps between periodic checkpoints (0 = off)
    checkpoint_achievements: bool = True
    experiment_root: str = "experiment_root"


def save_checkpoint_ppo(model, global_step: int, episode_count: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    with open(path + ".meta.json", "w") as f:
        json.dump({"global_step": global_step, "episode_count": episode_count}, f)
    print(f"Checkpoint saved: {path}")


class _TrainingCallback(BaseCallback):
    """SB3 callback that:
    - tracks episode count and global step
    - saves milestone checkpoints on first-time achievements
    - saves periodic timestep-based checkpoints
    - forwards terminal episode info to should_stop
    - logs per-episode returns to a txt file (append-only)
    """

    def __init__(self, args: Args, milestone_tracker: MilestoneTracker,
                 on_checkpoint_saved=None, should_stop=None):
        super().__init__(verbose=0)
        self.args = args
        self.milestone_tracker = milestone_tracker
        self.on_checkpoint_saved = on_checkpoint_saved
        self.should_stop = should_stop

        self.episode_count = 0
        self._stop_requested = False
        self._last_ckpt_step = 0
        self._next_periodic_step = args.checkpoint_freq if args.checkpoint_freq > 0 else None

        self._ep_return_running = None  # init lazily once n_envs is known
        returns_dir = os.path.join(args.experiment_root, "logs")
        os.makedirs(returns_dir, exist_ok=True)
        self._returns_path = os.path.join(returns_dir, "pporeturnlog.txt")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # Accumulate per-env returns
        if self._ep_return_running is None:
            self._ep_return_running = np.zeros(len(rewards))
        self._ep_return_running += np.asarray(rewards)

        completed = []
        for i, done in enumerate(dones):
            if done:
                completed.append(self._ep_return_running[i])
                self._ep_return_running[i] = 0.0
        if completed:
            with open(self._returns_path, "a") as f:
                f.write("\n".join(f"{r:.6f}" for r in completed) + "\n")

        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1

                # Milestone checkpoint on first-time achievements
                if self.args.checkpoint_achievements:
                    self.milestone_tracker.check_and_save(
                        info, self.model,
                        self.num_timesteps, self.episode_count,
                    )

                # should_stop callback
                if self.should_stop is not None:
                    if self.should_stop(info):
                        self._stop_requested = True

        # Periodic checkpoint
        if (
            self._next_periodic_step is not None
            and self.num_timesteps >= self._next_periodic_step
        ):
            path = (
                f"{self.args.experiment_root}/checkpoints/ppo/"
                f"periodic/periodic_step{self.num_timesteps}_ep{self.episode_count}.pt"
            )
            save_checkpoint_ppo(self.model, self.num_timesteps, self.episode_count, path)
            if self.on_checkpoint_saved:
                self.on_checkpoint_saved(path)
            self._next_periodic_step += self.args.checkpoint_freq

        return not self._stop_requested

    def _on_training_end(self):
        pass


def main_ppo(args: Args, on_checkpoint_saved=None, should_stop=None):
    """Train PPO on Crafter with SB3. Returns (episode_count, global_step)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parallel environments — each gets a unique RNG seed derived from args.seed
    if args.num_procs > 1:
        vec_env = SubprocVecEnv(
            [lambda i=i: make_crafter_env(seed=args.seed * 100_000 + i)
             for i in range(args.num_procs)],
            start_method="spawn",
        )
    else:
        vec_env = DummyVecEnv([lambda: make_crafter_env(seed=args.seed)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs={"features_extractor_kwargs": {"features_dim": args.features_dim}},
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        clip_range=args.clip_range,
        seed=args.seed,
        verbose=1,
    )

    def _save_fn(model, global_step, episode_count, path):
        save_checkpoint_ppo(model, global_step, episode_count, path)

    milestone_tracker = MilestoneTracker(
        args.experiment_root, _save_fn,
        on_checkpoint_saved=on_checkpoint_saved,
        algo_subdir="ppo",
    )

    callback = _TrainingCallback(
        args, milestone_tracker,
        on_checkpoint_saved=on_checkpoint_saved,
        should_stop=should_stop,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Final checkpoint
    path = (
        f"{args.experiment_root}/checkpoints/ppo/"
        f"periodic/final_step{model.num_timesteps}_ep{callback.episode_count}.pt"
    )
    save_checkpoint_ppo(model, model.num_timesteps, callback.episode_count, path)
    if on_checkpoint_saved:
        on_checkpoint_saved(path)

    vec_env.close()
    return callback.episode_count, model.num_timesteps


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main_ppo(args)
