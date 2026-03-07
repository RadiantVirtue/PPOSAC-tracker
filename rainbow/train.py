"""Rainbow DQN training for MiniGrid (DoorKey / KeyCorridor).

Entrypoint used by sweep.py and analysis scripts.  Training is fully
decoupled from analysis — no analysis code runs here.  Checkpoints and
an EpisodeStore snapshot are saved to disk for offline analysis.

Uses num_envs parallel environments (default 16, matching PPO).  Each env
runs its own episode independently; n-step returns are computed per-env to
avoid cross-env contamination, then stored in a shared PER replay buffer.

Usage:
    from rainbow.train import Args, main_rainbow
    args = Args(env_id="MiniGrid-DoorKey-5x5-v0", total_timesteps=5_000_000)
    episodes, steps = main_rainbow(args, should_stop=callback)
"""
import os
import random
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np
import torch

from rainbow.dqn import DQN
from rainbow.replay_buffer import Prioritized_ReplayBuffer, ReplayBuffer
from rainbow.tagged_buffer import EpisodeStore
from wrappers import DoorKeyAchievementWrapper, KeyCorridorAchievementWrapper


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass
class Args:
    # --- Environment ---
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    obs_key: str = "image"   # key into obs dict; "" for raw-array envs
    seed: int = 1
    num_envs: int = 16       # parallel environments (like PPO's num_procs)

    # --- Training budget ---
    total_timesteps: int = 10_000_000
    experiment_root: str = "experiment_root"
    checkpoint_freq: int = 0         # episodes between checkpoints; 0 = disabled
    checkpoint_step_freq: int = 5_000  # steps between checkpoints; 0 = disabled

    # --- Rainbow hyperparameters ---
    buffer_capacity: int = 200_000
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    n_steps: int = 5
    alpha: float = 0.6       # PER priority exponent
    beta_init: float = 0.4   # PER IS-weight initial value
    tau: float = 0.005       # soft-update coefficient
    use_soft_update: bool = True
    target_update_freq: int = 200
    use_lr_decay: bool = True
    grad_clip: float = 10.0
    use_double: bool = True
    use_dueling: bool = True
    use_noisy: bool = True
    use_per: bool = True
    use_n_steps: bool = True  # n-step returns (handled per-env in train loop)

    # Epsilon-greedy (unused when use_noisy=True)
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
    """Train Rainbow DQN with num_envs parallel environments.

    Args:
        args:                training configuration (Args dataclass).
        on_checkpoint_saved: optional callback(path) after each checkpoint.
        should_stop:         optional callback({"achievements": dict}) → bool.
                             Called at each episode end; return True to stop.

    Returns:
        (episode_count, global_step)
    """
    # ── Seeding ───────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Probe env to get obs/action shapes ───────────────────────────────────
    probe_env = _make_env(args.env_id)
    probe_obs, _ = probe_env.reset(seed=args.seed)
    obs_img = _extract_obs(probe_obs, args.obs_key)
    args.obs_shape = obs_img.shape      # (H, W, C)
    H, W, C = args.obs_shape
    args.state_dim = H * W * C
    args.action_dim = probe_env.action_space.n
    args.max_train_steps = args.total_timesteps
    probe_env.close()

    # ── Parallel environments ─────────────────────────────────────────────────
    # Each env gets a unique seed for diversity.
    envs = [_make_env(args.env_id) for _ in range(args.num_envs)]
    env_obs = [env.reset(seed=args.seed + 10000 * i)[0] for i, env in enumerate(envs)]
    env_states = [
        _extract_obs(o, args.obs_key).flatten().astype(np.float32)
        for o in env_obs
    ]

    # Per-env episode bookkeeping
    ep_transitions_per_env = [[] for _ in range(args.num_envs)]

    # Per-env n-step deques (avoids cross-env contamination in n-step returns)
    nstep_gamma = args.gamma  # raw gamma; n-step accumulation uses this
    nstep_deques = [deque(maxlen=args.n_steps) for _ in range(args.num_envs)]

    # ── Replay buffer ─────────────────────────────────────────────────────────
    # N-step returns are computed per-env above; we store the processed
    # transitions directly into a Prioritized_ReplayBuffer.
    # DQN is initialised with use_n_steps=True so agent.gamma = gamma^n,
    # matching the n-step targets we store.
    replay_buffer = Prioritized_ReplayBuffer(args) if args.use_per else ReplayBuffer(args)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = DQN(args)

    # ── Episode store (for offline analysis) ──────────────────────────────────
    episode_store = EpisodeStore(max_size=args.buffer_capacity)

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.experiment_root, "checkpoints", "rainbow")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Epsilon (unused with noisy nets) ─────────────────────────────────────
    epsilon = args.epsilon_init
    epsilon_decay = (
        (args.epsilon_init - args.epsilon_min) / max(args.epsilon_decay_steps, 1)
        if not args.use_noisy else 0.0
    )

    # ── Main training loop ────────────────────────────────────────────────────
    global_step = 0
    episode_count = 0
    last_ckpt_step = 0
    last_ckpt_episode = 0
    stop_requested = False

    while global_step < args.total_timesteps and not stop_requested:
        # Step every environment once per tick
        for i in range(args.num_envs):
            state = env_states[i]
            action = agent.choose_action(state, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = envs[i].step(action)
            done = terminated or truncated
            terminal = bool(terminated)

            next_obs_img = _extract_obs(next_obs, args.obs_key)
            next_state = next_obs_img.flatten().astype(np.float32)

            global_step += 1

            # Track raw transitions for EpisodeStore
            ep_transitions_per_env[i].append({
                "state":      state,
                "action":     int(action),
                "reward":     float(reward),
                "next_state": next_state,
                "terminal":   terminal,
            })

            # N-step accumulation (per-env deque)
            nstep_deques[i].append((state, action, reward, next_state, terminal, done))
            if len(nstep_deques[i]) == args.n_steps:
                s0, a0, n_r, sn, tn = _compute_nstep(nstep_deques[i], nstep_gamma)
                replay_buffer.store_transition(s0, a0, n_r, sn, tn, tn)

            # Epsilon decay (no-op when use_noisy=True)
            if not args.use_noisy:
                epsilon = max(args.epsilon_min, epsilon - epsilon_decay)

            if done:
                nstep_deques[i].clear()  # episode boundary — reset per-env deque
                episode_count += 1
                eps_score = float(info.get("eps", 0.0))
                achievements = info.get("achievements", {})

                episode_store.add_episode(ep_transitions_per_env[i], eps_score)
                ep_transitions_per_env[i] = []

                if episode_count % 10 == 0:
                    print(
                        f"  [rainbow] ep={episode_count:6d}  "
                        f"steps={global_step:9d}  eps={eps_score:.3f}"
                    )

                if should_stop is not None and should_stop({"achievements": achievements}):
                    print(f"  [rainbow] Convergence criterion met at episode {episode_count}.")
                    stop_requested = True
                    break

                # Episode-based checkpoint
                if (
                    args.checkpoint_freq > 0
                    and (episode_count - last_ckpt_episode) >= args.checkpoint_freq
                ):
                    ckpt_path = _save_checkpoint(
                        agent, episode_store, global_step, episode_count, args, ckpt_dir
                    )
                    last_ckpt_episode = episode_count
                    last_ckpt_step = global_step
                    if on_checkpoint_saved is not None:
                        on_checkpoint_saved(ckpt_path)

                reset_obs, _ = envs[i].reset()
                env_states[i] = _extract_obs(reset_obs, args.obs_key).flatten().astype(np.float32)
            else:
                env_states[i] = next_state

        # Learn once per tick (after all envs stepped)
        if replay_buffer.current_size >= args.batch_size:
            agent.learn(replay_buffer, global_step)

        # Step-based periodic checkpoint (checked once per tick)
        if (
            args.checkpoint_step_freq > 0
            and (global_step - last_ckpt_step) >= args.checkpoint_step_freq
        ):
            ckpt_path = _save_checkpoint(
                agent, episode_store, global_step, episode_count, args, ckpt_dir
            )
            last_ckpt_step = global_step
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(ckpt_path)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if global_step > last_ckpt_step:
        ckpt_path = _save_checkpoint(
            agent, episode_store, global_step, episode_count, args, ckpt_dir
        )
        if on_checkpoint_saved is not None:
            on_checkpoint_saved(ckpt_path)

    for env in envs:
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
    if obs_key:
        return obs[obs_key]
    return obs


def _compute_nstep(dq, gamma: float):
    """Compute n-step return from a per-env deque of (s,a,r,s',terminal,done)."""
    s0, a0 = dq[0][0], dq[0][1]
    sn, tn = dq[-1][3], dq[-1][4]   # default: last step's next_state / terminal
    n_r = 0.0
    for j in reversed(range(len(dq))):
        r, s_, ter, d = dq[j][2], dq[j][3], dq[j][4], dq[j][5]
        n_r = r + gamma * (1 - int(d)) * n_r
        if d:  # episode ended mid-window — truncate horizon here
            sn, tn = s_, ter
    return s0, a0, n_r, sn, tn


def _save_checkpoint(agent, episode_store, global_step, episode_count, args, ckpt_dir):
    filename = f"rainbow_ep{episode_count:07d}_step{global_step:09d}.pt"
    ckpt_path = os.path.join(ckpt_dir, filename)

    torch.save({
        "global_step":           global_step,
        "episode_count":         episode_count,
        "net_state_dict":        agent.net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict":  agent.optimizer.state_dict(),
        "obs_shape":             args.obs_shape,
        "action_dim":            args.action_dim,
    }, ckpt_path)

    eps_path = ckpt_path.replace(".pt", "_episodes.pkl")
    episode_store.save(eps_path)

    print(f"  [rainbow] Checkpoint saved: {ckpt_path}")
    return ckpt_path
