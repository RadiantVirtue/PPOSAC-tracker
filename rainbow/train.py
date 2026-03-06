"""Rainbow DQN training for MiniGrid (DoorKey / KeyCorridor).

Entrypoint used by sweep.py and analysis scripts.  Training is fully
decoupled from analysis — no analysis code runs here.  Checkpoints and
an EpisodeStore snapshot are saved to disk for offline analysis.

Usage:
    from rainbow.train import Args, main_rainbow
    args = Args(env_id="MiniGrid-DoorKey-5x5-v0", total_timesteps=5_000_000)
    episodes, steps = main_rainbow(args, should_stop=callback)
"""
import os
import random
from dataclasses import dataclass, field

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np
import torch

from rainbow.dqn import DQN
from rainbow.replay_buffer import (
    N_Steps_Prioritized_ReplayBuffer,
    Prioritized_ReplayBuffer,
    N_Steps_ReplayBuffer,
    ReplayBuffer,
)
from rainbow.tagged_buffer import EpisodeStore
from wrappers import DoorKeyAchievementWrapper, KeyCorridorAchievementWrapper


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass
class Args:
    # --- Environment ---
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    # obs_key: key into obs dict to extract the image array.
    # Set to "" for envs that return raw arrays (e.g. Crafter).
    obs_key: str = "image"
    seed: int = 1

    # --- Training budget ---
    total_timesteps: int = 10_000_000
    experiment_root: str = "experiment_root"
    # checkpoint_freq: save a checkpoint every N steps; 0 = only at end.
    checkpoint_freq: int = 100_000

    # --- Rainbow hyperparameters ---
    buffer_capacity: int = 200_000
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    n_steps: int = 5
    alpha: float = 0.6      # PER priority exponent
    beta_init: float = 0.4  # PER IS-weight initial value
    tau: float = 0.005      # soft-update coefficient
    use_soft_update: bool = True
    target_update_freq: int = 200
    use_lr_decay: bool = True
    grad_clip: float = 10.0
    use_double: bool = True
    use_dueling: bool = True
    use_noisy: bool = True
    use_per: bool = True
    use_n_steps: bool = True

    # Epsilon-greedy (unused when use_noisy=True; kept for completeness)
    epsilon_init: float = 0.0
    epsilon_min: float = 0.0
    epsilon_decay_steps: int = 1

    # --- Set at runtime from env (not user-facing) ---
    action_dim: int = field(default=0, repr=False)
    obs_shape: tuple = field(default=(), repr=False)
    state_dim: int = field(default=0, repr=False)
    max_train_steps: int = field(default=0, repr=False)


# ── Public entrypoint ─────────────────────────────────────────────────────────

def main_rainbow(args: Args, on_checkpoint_saved=None, should_stop=None):
    """Train a full Rainbow DQN agent on a MiniGrid environment.

    Args:
        args:                 training configuration (Args dataclass).
        on_checkpoint_saved:  optional callback(path) called after each save.
        should_stop:          optional callback(info_dict) → bool.
                              Called at each episode end with
                              {"achievements": dict}.  Return True to stop early.

    Returns:
        (episode_count, global_step)
    """
    # ── Seeding ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Environment ──────────────────────────────────────────────────────────
    env = _make_env(args.env_id)

    obs, info = env.reset(seed=args.seed)
    obs_img = _extract_obs(obs, args.obs_key)
    args.obs_shape = obs_img.shape          # (H, W, C)
    H, W, C = args.obs_shape
    args.state_dim = H * W * C
    args.action_dim = env.action_space.n
    args.max_train_steps = args.total_timesteps

    # ── Replay buffer ─────────────────────────────────────────────────────────
    if args.use_per and args.use_n_steps:
        replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
    elif args.use_per:
        replay_buffer = Prioritized_ReplayBuffer(args)
    elif args.use_n_steps:
        replay_buffer = N_Steps_ReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = DQN(args)

    # ── Episode store (for offline analysis) ─────────────────────────────────
    episode_store = EpisodeStore(max_size=args.buffer_capacity)

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.experiment_root, "checkpoints", "rainbow")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Epsilon (unused with noisy nets, kept for generality) ─────────────────
    epsilon = args.epsilon_init
    epsilon_decay = (
        (args.epsilon_init - args.epsilon_min) / max(args.epsilon_decay_steps, 1)
        if not args.use_noisy else 0.0
    )

    # ── Main training loop ────────────────────────────────────────────────────
    global_step = 0
    episode_count = 0
    last_ckpt_step = 0
    ep_transitions: list = []

    state = obs_img.flatten().astype(np.float32)

    while global_step < args.total_timesteps:
        action = agent.choose_action(state, epsilon=epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # terminal = game over (not a timeout); used for bootstrapping
        terminal = bool(terminated)

        next_obs_img = _extract_obs(next_obs, args.obs_key)
        next_state = next_obs_img.flatten().astype(np.float32)

        replay_buffer.store_transition(state, action, reward, next_state, terminal, done)
        global_step += 1

        ep_transitions.append({
            "state":      state,
            "action":     int(action),
            "reward":     float(reward),
            "next_state": next_state,
            "terminal":   terminal,
        })

        # Learn once buffer is ready
        if replay_buffer.current_size >= args.batch_size:
            agent.learn(replay_buffer, global_step)

        # Epsilon decay (no-op when use_noisy=True)
        if not args.use_noisy:
            epsilon = max(args.epsilon_min, epsilon - epsilon_decay)

        if done:
            episode_count += 1
            eps_score = float(info.get("eps", 0.0))
            achievements = info.get("achievements", {})

            # Tag transitions with episode EPS and store
            episode_store.add_episode(ep_transitions, eps_score)
            ep_transitions = []

            # Periodic progress print
            if episode_count % 200 == 0:
                print(
                    f"  [rainbow] ep={episode_count:6d}  "
                    f"steps={global_step:9d}  eps={eps_score:.3f}"
                )

            # Periodic checkpoint
            if args.checkpoint_freq > 0 and (global_step - last_ckpt_step) >= args.checkpoint_freq:
                ckpt_path = _save_checkpoint(
                    agent, episode_store, global_step, episode_count, args, ckpt_dir
                )
                last_ckpt_step = global_step
                if on_checkpoint_saved is not None:
                    on_checkpoint_saved(ckpt_path)

            # Convergence check
            if should_stop is not None and should_stop({"achievements": achievements}):
                print(f"  [rainbow] Convergence criterion met at episode {episode_count}.")
                break

            obs, info = env.reset()
            obs_img = _extract_obs(obs, args.obs_key)
            state = obs_img.flatten().astype(np.float32)
        else:
            state = next_state

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if global_step > last_ckpt_step:
        ckpt_path = _save_checkpoint(
            agent, episode_store, global_step, episode_count, args, ckpt_dir
        )
        if on_checkpoint_saved is not None:
            on_checkpoint_saved(ckpt_path)

    env.close()
    return episode_count, global_step


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_env(env_id: str):
    env = gym.make(env_id)
    if "DoorKey" in env_id:
        return DoorKeyAchievementWrapper(env)
    if "KeyCorridor" in env_id:
        return KeyCorridorAchievementWrapper(env)
    raise ValueError(
        f"No achievement wrapper registered for env_id '{env_id}'. "
        "Add a wrapper in wrappers.py and register it here."
    )


def _extract_obs(obs, obs_key: str):
    """Extract the image array from an obs dict or return raw obs."""
    if obs_key:
        return obs[obs_key]
    return obs


def _save_checkpoint(agent, episode_store, global_step, episode_count, args, ckpt_dir):
    filename = f"rainbow_ep{episode_count:07d}_step{global_step:09d}.pt"
    ckpt_path = os.path.join(ckpt_dir, filename)

    torch.save({
        "global_step":             global_step,
        "episode_count":           episode_count,
        "net_state_dict":          agent.net.state_dict(),
        "target_net_state_dict":   agent.target_net.state_dict(),
        "optimizer_state_dict":    agent.optimizer.state_dict(),
        "obs_shape":               args.obs_shape,
        "action_dim":              args.action_dim,
    }, ckpt_path)

    eps_path = ckpt_path.replace(".pt", "_episodes.pkl")
    episode_store.save(eps_path)

    print(f"  [rainbow] Checkpoint saved: {ckpt_path}")
    return ckpt_path
