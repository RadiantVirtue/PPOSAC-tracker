"""PPO training using rl-starter-files' ACModel + torch_ac.PPOAlgo."""
import _rl_path  # noqa: F401  — adds rl-starter-files to sys.path

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch_ac
import torch_ac.algos.base as _torch_ac_base
from torch_ac.utils.penv import ParallelEnv
from torch.utils.tensorboard import SummaryWriter

from model import ACModel
from utils.format import get_obss_preprocessor

from wrappers import DoorKeyAchievementWrapper, KeyCorridorAchievementWrapper
from ppo.checkpoint_gen import MilestoneTracker


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None

    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    total_episodes: int = 50_000
    num_procs: int = 16
    frames_per_proc: int = 128
    discount: float = 0.99
    lr: float = 0.001
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    epochs: int = 4
    batch_size: int = 256
    checkpoint_freq: int = 1000
    checkpoint_achievements: bool = True
    experiment_root: str = "experiment_root"


class _InfoCapturingParallelEnv(ParallelEnv):
    """ParallelEnv that captures terminal step infos in the main process.

    torch_ac spawns envs[1:] into subprocesses, so any in-process wrapper
    around individual envs cannot be read from the main process after the
    fact.  This subclass intercepts the results of every step() call —
    which ARE returned to the main process via pipes — and stores terminal
    infos here, where pop_all_infos() can retrieve them.
    """

    def __init__(self, envs):
        super().__init__(envs)
        self._terminal_infos = []

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs0, reward0, terminated0, truncated0, info0 = self.envs[0].step(actions[0])
        if terminated0 or truncated0:
            obs0, _ = self.envs[0].reset()
            self._terminal_infos.append(info0)
        others = [local.recv() for local in self.locals]
        for _, _, term_i, trunc_i, info_i in others:
            if term_i or trunc_i:
                self._terminal_infos.append(info_i)
        return zip(*[(obs0, reward0, terminated0, truncated0, info0)] + others)

    def pop_all_infos(self):
        infos = self._terminal_infos[:]
        self._terminal_infos.clear()
        return infos


def _make_env(env_id, seed):
    env = gym.make(env_id)
    if "DoorKey" in env_id:
        env = DoorKeyAchievementWrapper(env)
    elif "KeyCorridor" in env_id:
        env = KeyCorridorAchievementWrapper(env)
    else:
        raise ValueError(f"No achievement wrapper for env_id: {env_id}")
    env.reset(seed=seed)
    return env


def save_checkpoint_ppo(agent, optimizer, global_step, episode_count, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "global_step": global_step,
        "episode_count": episode_count,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")


def main_ppo(args, on_checkpoint_saved=None, should_stop=None):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create parallel envs (each with a different seed offset)
    envs = [_make_env(args.env_id, args.seed + 10000 * i) for i in range(args.num_procs)]

    # Observation space preprocessing (rl-starter-files utility)
    obs_space, preprocess_obss = get_obss_preprocessor(envs[0].observation_space)

    # ACModel: CNN + actor/critic heads
    acmodel = ACModel(obs_space, envs[0].action_space, use_memory=False, use_text=False)
    acmodel.to(device)

    # Patch ParallelEnv so PPOAlgo uses our info-capturing subclass
    _torch_ac_base.ParallelEnv = _InfoCapturingParallelEnv

    # torch_ac PPO algorithm
    algo = torch_ac.PPOAlgo(
        envs, acmodel, device, args.frames_per_proc,
        args.discount, args.lr, args.gae_lambda,
        args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
        recurrence=1, adam_eps=1e-8, clip_eps=args.clip_eps,
        epochs=args.epochs, batch_size=args.batch_size,
        preprocess_obss=preprocess_obss,
    )

    def _save_fn(agent, optimizer, global_step, episode_count, path):
        save_checkpoint_ppo(agent, optimizer, global_step, episode_count, path)

    milestone_tracker = MilestoneTracker(
        args.experiment_root, _save_fn,
        on_checkpoint_saved=on_checkpoint_saved,
        algo_subdir="ppo",
    )

    writer = SummaryWriter(f"runs/{run_name}")

    global_step = 0
    episode_count = 0
    next_checkpoint_ep = args.checkpoint_freq if args.checkpoint_freq > 0 else None

    while episode_count < args.total_episodes:
        exps, logs1 = algo.collect_experiences()
        global_step += logs1["num_frames"]

        logs2 = algo.update_parameters(exps)

        # Collect all episode infos from this batch via the patched ParallelEnv
        all_episode_infos = algo.env.pop_all_infos()

        # Check for milestone achievements
        if args.checkpoint_achievements:
            for info in all_episode_infos:
                milestone_tracker.check_and_save(
                    info, acmodel, algo.optimizer, global_step, episode_count
                )

        # Count completed episodes and handle should_stop
        stop = False
        for info in all_episode_infos:
            episode_count += 1
            if should_stop is not None:
                if should_stop(info):
                    stop = True

        # Periodic checkpoint
        if next_checkpoint_ep is not None and episode_count >= next_checkpoint_ep:
            path = (
                f"{args.experiment_root}/checkpoints/ppo/"
                f"periodic_{episode_count // 1000}k_episodes.pt"
            )
            save_checkpoint_ppo(acmodel, algo.optimizer, global_step, episode_count, path)
            if on_checkpoint_saved:
                on_checkpoint_saved(path)
            next_checkpoint_ep += args.checkpoint_freq

        # Logging
        n_ep = len(logs1.get("return_per_episode", []))
        if n_ep > 0:
            mean_ret = np.mean(logs1["return_per_episode"])
            writer.add_scalar("charts/episodic_return", mean_ret, episode_count)
            writer.add_scalar("losses/policy_loss", logs2["policy_loss"], episode_count)
            writer.add_scalar("losses/value_loss", logs2["value_loss"], episode_count)
            writer.add_scalar("losses/entropy", logs2["entropy"], episode_count)
            print(f"episode={episode_count}, return={mean_ret:.3f}")

        if stop:
            break

    # Final checkpoint
    if args.checkpoint_freq > 0:
        path = (
            f"{args.experiment_root}/checkpoints/ppo/"
            f"final_{episode_count // 1000}k_episodes.pt"
        )
        save_checkpoint_ppo(acmodel, algo.optimizer, global_step, episode_count, path)
        if on_checkpoint_saved:
            on_checkpoint_saved(path)

    writer.close()
    return episode_count, global_step


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main_ppo(args)
