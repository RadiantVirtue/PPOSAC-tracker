"""Rainbow evaluation and episode partitioning (Crafter).

Adapted from sac/sampling.py. Key differences:
  - Loads DQN (online + target) from rich checkpoint instead of DiscreteActor
  - Obs preprocessing: RGB (H,W,3) uint8 → (3,H,W) float32 [0,1] (same as SAC)
  - Action selection via ε-greedy instead of actor.get_distribution()
  - Returns both online_net and target_net for distributional gradient analysis
"""
import os
from collections import namedtuple

import argparse
import numpy as np
import torch
from tqdm import tqdm

from rainbow.model import DQN
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
        device=torch.device(device),
        model=None,
    )


def load_rainbow_nets(checkpoint_path: str, device: str = "cpu"):
    """Load frozen online + target DQN networks from a rich checkpoint.

    Returns:
        (online_net, target_net, episode_count, args_ns)
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
    return online_net, target_net, episode_count, args_ns


def evaluate_frozen_policy(
    checkpoint_path: str, n_episodes: int = 500, device: str = "cpu",
    seed: int = None,
):
    """Run n_episodes with a frozen Rainbow DQN on Crafter.

    Observations stored in EpisodeData are (T, 3, H, W) float32 RGB tensors
    in [0, 1], matching the network's expected input format (history_length=3).

    Returns:
        episodes:   list of EpisodeData namedtuples
        eps_scores: list of float EPS scores (one per episode)
    """
    online_net, _, _, args_ns = load_rainbow_nets(checkpoint_path, device)
    online_net.eval()

    support = torch.linspace(
        args_ns.V_min, args_ns.V_max, args_ns.atoms
    ).to(device)

    env = make_crafter_env(seed=seed)
    episodes = []
    eps_scores = []

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes", unit="ep"):
        obs, info = env.reset()
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        done = False
        while not done:
            obs_t = _obs_to_tensor(obs).unsqueeze(0).to(device)  # (1,3,H,W)
            with torch.no_grad():
                # ε-greedy action selection (ε=0.001, eval mode uses weight_mu only)
                if np.random.random() < 0.001:
                    action = np.random.randint(0, online_net.action_space)
                else:
                    action = (online_net(obs_t) * support).sum(2).argmax(1).item()

            ep_obs.append(_obs_to_tensor(obs))   # (3,H,W) float32
            ep_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rewards.append(float(reward))
            ep_dones.append(float(done))

        eps_scores.append(float(info.get("eps", 0.0)))
        episodes.append(
            EpisodeData(
                observations=torch.stack(ep_obs),            # (T, 3, H, W) float32
                actions=torch.tensor(ep_actions, dtype=torch.long),
                rewards=torch.tensor(ep_rewards, dtype=torch.float32),
                dones=torch.tensor(ep_dones, dtype=torch.float32),
            )
        )

    env.close()
    return episodes, eps_scores


def partition(episodes, eps_scores, mode: str = "eps", percentile_x: int = 25):
    """Partition EpisodeData list into success / failure groups."""
    return partition_episodes(episodes, eps_scores, mode=mode, percentile_x=percentile_x)
