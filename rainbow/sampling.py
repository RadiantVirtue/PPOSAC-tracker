"""Rainbow evaluation and episode partitioning (Crafter).

Adapted from sac/sampling.py. Key differences:
  - Loads DQN (online + target) from rich checkpoint instead of DiscreteActor
  - Obs preprocessing: RGB (H,W,3) uint8 → (3,H,W) float32 [0,1] (same as SAC)
  - Action selection via ε-greedy instead of actor.get_distribution()
  - Returns both online_net and target_net for distributional gradient analysis
  - evaluate_frozen_policy uses AsyncVectorEnv for parallel episode collection
"""
import os
from collections import namedtuple

import argparse
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from rainbow.model import DQN
from shared.achievements import CRAFTER_ACHIEVEMENTS
from shared.thresholding import partition_episodes
from wrappers import make_crafter_env

EpisodeData = namedtuple(
    "EpisodeData", ["observations", "actions", "rewards", "dones"]
)


def _obs_to_tensor(obs) -> torch.Tensor:
    """(H, W, 3) uint8 numpy → (3, H, W) float32 [0,1] tensor."""
    arr = np.array(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def _build_args_from_dict(args_dict: dict, n_actions: int, device: str) -> argparse.Namespace:
    """Reconstruct an argparse.Namespace from a checkpoint args_dict."""
    return argparse.Namespace(
        atoms=args_dict.get('atoms', 51),
        hidden_size=args_dict.get('hidden_size', 512),
        architecture=args_dict.get('architecture', 'canonical'),
        history_length=args_dict.get('history_length', 1),
        V_min=args_dict.get('V_min', -10),
        V_max=args_dict.get('V_max', 10),
        multi_step=args_dict.get('multi_step', 3),
        discount=args_dict.get('discount', 0.99),
        noisy_std=args_dict.get('noisy_std', 0.1),
        # IS-weighting params for offline gradient analysis
        priority_exponent=args_dict.get('priority_exponent', 0.5),
        priority_weight=args_dict.get('priority_weight', 0.4),
        T_max=args_dict.get('T_max', int(10e6)),
        learn_start=args_dict.get('learn_start', int(20e3)),
        device=torch.device(device),
        model=None,
    )


def load_rainbow_nets(checkpoint_path: str, device: str = "cpu"):
    """Load frozen online + target DQN networks from a rich checkpoint.

    Returns:
        (online_net, target_net, episode_count, global_step, args_ns)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt.get('args_dict', {})
    n_actions = args_dict.get('n_actions', 17)  # Crafter has 17 actions
    args_ns = _build_args_from_dict(args_dict, n_actions, device)

    online_net = DQN(args_ns, n_actions)
    online_net.load_state_dict(ckpt['online_net_state_dict'])
    online_net.to(device).eval()

    target_net = DQN(args_ns, n_actions)
    if 'target_net_state_dict' in ckpt:
        target_net.load_state_dict(ckpt['target_net_state_dict'])
    else:
        target_net.load_state_dict(ckpt['online_net_state_dict'])  # fallback
    target_net.to(device).eval()
    for p in target_net.parameters():
        p.requires_grad = False

    episode_count = ckpt.get('episode_count', 0)
    global_step = ckpt.get('global_step', 0)
    return online_net, target_net, episode_count, global_step, args_ns


def evaluate_frozen_policy(
    checkpoint_path: str, n_episodes: int = 500, device: str = "cpu",
    seed: int = None, num_envs: int = 16, eps_weight: float = 0.9,
):
    """Run n_episodes with a frozen Rainbow DQN on Crafter using parallel envs.

    Observations stored in EpisodeData are (T, 3, H, W) float32 RGB tensors
    in [0, 1], matching the network's expected input format (history_length=3).

    Args:
        eps_weight: coefficient for materials_progress in EPS score (default 0.9).
            Use values ∈ {0.5, 0.9, 1.2} for sensitivity analysis (Issue #9).

    Returns:
        episodes:                  list of EpisodeData namedtuples
        eps_scores:                list of float EPS scores (one per episode)
        episodes_with_transitions: list of (EpisodeData, {ach: step_idx}) for RSA —
                                   step_idx is the first step within the episode where
                                   that achievement was newly unlocked.
    """
    online_net, _, _, _, args_ns = load_rainbow_nets(checkpoint_path, device)
    online_net.eval()

    support = torch.linspace(
        args_ns.V_min, args_ns.V_max, args_ns.atoms
    ).to(device)

    base_seed = seed if seed is not None else 0
    vec_env = gym.vector.AsyncVectorEnv([
        (lambda i: lambda: make_crafter_env(seed=base_seed + i * 10_000, eps_weight=eps_weight))(i)
        for i in range(num_envs)
    ])

    # Per-env rolling buffers
    env_obs      = [[] for _ in range(num_envs)]
    env_actions  = [[] for _ in range(num_envs)]
    env_rewards  = [[] for _ in range(num_envs)]
    env_dones    = [[] for _ in range(num_envs)]
    env_prev_ach = [{a: False for a in CRAFTER_ACHIEVEMENTS} for _ in range(num_envs)]
    env_trans    = [{} for _ in range(num_envs)]
    env_step_idx = [0] * num_envs

    episodes                  = []
    eps_scores                = []
    episodes_with_transitions = []
    completed                 = 0

    obs, _ = vec_env.reset()
    # Seed numpy random for reproducible ε-greedy noise given the same base_seed
    np.random.seed(base_seed)
    pbar = tqdm(total=n_episodes, desc="Evaluating episodes", unit="ep")

    while completed < n_episodes:
        # Batched inference: (N,H,W,3) uint8 → (N,3,H,W) float32 [0,1]
        obs_t = torch.from_numpy(obs).float().div_(255.0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            actions = (online_net(obs_t) * support).sum(2).argmax(1).cpu().numpy()
        # ε-greedy (ε=0.001)
        for i in range(num_envs):
            if np.random.random() < 0.001:
                actions[i] = np.random.randint(0, online_net.action_space)

        for i in range(num_envs):
            env_obs[i].append(obs[i].copy())
            env_actions[i].append(int(actions[i]))

        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        for i in range(num_envs):
            done = bool(terminateds[i]) or bool(truncateds[i])
            env_rewards[i].append(float(rewards[i]))
            env_dones[i].append(float(done))

            # Track first achievement unlock per step (for RSA stimulus detection)
            if done:
                raw_ach = {a: bool(infos["achievements"][a][i]) for a in CRAFTER_ACHIEVEMENTS}
            else:
                raw_ach_info = infos.get("achievements", {})
                raw_ach = (
                    {a: bool(raw_ach_info[a][i]) for a in CRAFTER_ACHIEVEMENTS}
                    if isinstance(raw_ach_info, dict) else {}
                )
            for ach in CRAFTER_ACHIEVEMENTS:
                if raw_ach.get(ach, False) and not env_prev_ach[i].get(ach, False) and ach not in env_trans[i]:
                    env_trans[i][ach] = env_step_idx[i]
            env_prev_ach[i] = {a: bool(raw_ach.get(a, False)) for a in CRAFTER_ACHIEVEMENTS}
            env_step_idx[i] += 1

            if done and completed < n_episodes:  # guard prevents over-collection beyond n_episodes
                # Stack obs: (T,H,W,3) uint8 → (T,3,H,W) float32 [0,1]
                obs_arr = np.array(env_obs[i], dtype=np.float32) / 255.0
                obs_tensor = torch.from_numpy(obs_arr).permute(0, 3, 1, 2)  # (T,3,H,W)

                ep_data = EpisodeData(
                    observations=obs_tensor,
                    actions=torch.tensor(env_actions[i], dtype=torch.long),
                    rewards=torch.tensor(env_rewards[i], dtype=torch.float32),
                    dones=torch.tensor(env_dones[i], dtype=torch.float32),
                )
                episodes.append(ep_data)
                eps_scores.append(float(infos["eps"][i]))
                episodes_with_transitions.append((ep_data, env_trans[i]))
                completed += 1
                pbar.update(1)

                # Reset this env's buffers
                env_obs[i]      = []
                env_actions[i]  = []
                env_rewards[i]  = []
                env_dones[i]    = []
                env_prev_ach[i] = {a: False for a in CRAFTER_ACHIEVEMENTS}
                env_trans[i]    = {}
                env_step_idx[i] = 0

    pbar.close()
    vec_env.close()
    return episodes, eps_scores, episodes_with_transitions


def partition(episodes, eps_scores, mode: str = "eps", percentile_x: int = 25,
              fixed_thresholds=None):
    """Partition EpisodeData list into success / failure groups."""
    return partition_episodes(episodes, eps_scores, mode=mode, percentile_x=percentile_x,
                              fixed_thresholds=fixed_thresholds)
