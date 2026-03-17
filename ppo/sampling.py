"""PPO evaluation and episode partitioning (SB3 / Crafter)."""
import json
import os
from collections import namedtuple

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from wrappers import make_crafter_env
from shared.thresholding import partition_episodes

EpisodeData = namedtuple(
    "EpisodeData", ["observations", "actions", "rewards", "dones"]
)


def load_ppo_agent(checkpoint_path: str, device: str = "cpu"):
    """Load a frozen SB3 PPO model + metadata.

    Args:
        checkpoint_path: path to the SB3 .zip checkpoint (without .zip extension).
        device:          torch device string.

    Returns:
        (model, episode_count)
    """
    model = PPO.load(checkpoint_path, device=device)
    meta_path = checkpoint_path + ".meta.json"
    episode_count = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        episode_count = meta.get("episode_count", 0)
    return model, episode_count


def evaluate_frozen_policy(
    checkpoint_path: str, n_episodes: int = 1000, device: str = "cpu"
):
    """Run n_episodes with a frozen SB3 PPO policy on Crafter.

    Observations stored in EpisodeData are (H, W, C) uint8 numpy arrays as a
    stacked tensor of shape (T, H, W, C), compatible with ppo/gradients.py and
    ppo/activations.py (which normalise internally via policy.obs_to_tensor).

    Returns:
        episodes:   list of EpisodeData namedtuples
        eps_scores: list of float EPS scores (one per episode)
    """
    model, _ = load_ppo_agent(checkpoint_path, device)
    env = make_crafter_env()

    episodes = []
    eps_scores = []

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes", unit="ep"):
        obs, info = env.reset()
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            ep_obs.append(obs.copy())
            ep_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_rewards.append(float(reward))
            ep_dones.append(float(done))

        eps_scores.append(float(info.get("eps", 0.0)))
        episodes.append(
            EpisodeData(
                observations=torch.tensor(
                    np.array(ep_obs, dtype=np.uint8)
                ),  # (T, H, W, C) uint8
                actions=torch.tensor(ep_actions, dtype=torch.long),
                rewards=torch.tensor(ep_rewards, dtype=torch.float32),
                dones=torch.tensor(ep_dones, dtype=torch.float32),
            )
        )

    env.close()
    return episodes, eps_scores


def partition(episodes, eps_scores, mode: str = "eps", percentile_x: int = 25):
    """Partition episodes into success / failure groups.

    In percentile mode, raw episode returns (sum of shaped rewards) are used
    as the ranking score and the middle episodes are discarded.
    """
    if mode == "percentile":
        scores = [ep.rewards.sum().item() for ep in episodes]
    else:
        scores = eps_scores
    return partition_episodes(episodes, scores, mode=mode, percentile_x=percentile_x)
