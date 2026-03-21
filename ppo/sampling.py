"""PPO evaluation and episode partitioning (SB3 / Crafter)."""
import json
import os
from collections import namedtuple

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from wrappers import make_crafter_env
from shared.achievements import CRAFTER_ACHIEVEMENTS
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
    checkpoint_path: str, n_episodes: int = 1000, device: str = "cuda", seed: int = None,
    num_envs: int = 16,
):
    """Run n_episodes with a frozen SB3 PPO policy on Crafter using parallel envs.

    Observations stored in EpisodeData are (H, W, C) uint8 numpy arrays as a
    stacked tensor of shape (T, H, W, C), compatible with ppo/gradients.py and
    ppo/activations.py (which normalise internally via policy.obs_to_tensor).

    Returns:
        episodes:                list of EpisodeData namedtuples
        eps_scores:              list of float EPS scores (one per episode)
        episodes_with_transitions: list of (EpisodeData, {ach: step_idx}) for RSA —
                                   step_idx is the first step within the episode where
                                   that achievement was newly unlocked. Uses only info
                                   returned by env.step() — no data leak.
    """
    import gymnasium as gym

    model, _ = load_ppo_agent(checkpoint_path, device)

    base_seed = seed if seed is not None else 0
    vec_env = gym.vector.AsyncVectorEnv([
        (lambda i: lambda: make_crafter_env(seed=base_seed + i * 10_000))(i)
        for i in range(num_envs)
    ])

    episodes = []
    eps_scores = []
    episodes_with_transitions = []

    # Per-env rolling buffers
    env_obs      = [[] for _ in range(num_envs)]
    env_actions  = [[] for _ in range(num_envs)]
    env_rewards  = [[] for _ in range(num_envs)]
    env_dones    = [[] for _ in range(num_envs)]
    env_prev_ach = [{a: False for a in CRAFTER_ACHIEVEMENTS} for _ in range(num_envs)]
    env_trans    = [{} for _ in range(num_envs)]
    env_step_idx = [0] * num_envs

    obs, _ = vec_env.reset()
    completed = 0
    pbar = tqdm(total=n_episodes, desc="Evaluating episodes", unit="ep")

    while completed < n_episodes:
        actions, _ = model.predict(obs, deterministic=False)

        for i in range(num_envs):
            env_obs[i].append(obs[i].copy())
            env_actions[i].append(int(actions[i]))

        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        for i in range(num_envs):
            done = bool(terminateds[i]) or bool(truncateds[i])
            env_rewards[i].append(float(rewards[i]))
            env_dones[i].append(float(done))

            # Gymnasium vector envs put terminal-step info under final_info[i];
            # mid-episode info is in infos with dict-of-arrays structure.
            if done:
                step_info = (infos.get("final_info") or [None] * num_envs)[i] or {}
            else:
                raw_ach = infos.get("achievements", {})
                # dict-of-arrays → per-env dict
                step_info = {
                    "achievements": {a: bool(raw_ach[a][i]) for a in CRAFTER_ACHIEVEMENTS}
                    if isinstance(raw_ach, dict) else {},
                    "eps": float(infos["eps"][i]) if "eps" in infos else 0.0,
                }

            cur_ach = step_info.get("achievements", {})
            for ach in CRAFTER_ACHIEVEMENTS:
                if cur_ach.get(ach, False) and not env_prev_ach[i].get(ach, False) and ach not in env_trans[i]:
                    env_trans[i][ach] = env_step_idx[i]
            env_prev_ach[i] = {a: bool(cur_ach.get(a, False)) for a in CRAFTER_ACHIEVEMENTS}
            env_step_idx[i] += 1

            if done and completed < n_episodes:
                ep_data = EpisodeData(
                    observations=torch.tensor(np.array(env_obs[i], dtype=np.uint8)),
                    actions=torch.tensor(env_actions[i], dtype=torch.long),
                    rewards=torch.tensor(env_rewards[i], dtype=torch.float32),
                    dones=torch.tensor(env_dones[i], dtype=torch.float32),
                )
                episodes.append(ep_data)
                eps_scores.append(float(step_info.get("eps", 0.0)))
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
