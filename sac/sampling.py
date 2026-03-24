"""SAC evaluation and episode partitioning (Crafter)."""
import math
import os
from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm

from sac.network import DiscreteActor, DiscreteCritic
from sac.train import _obs_to_tensor
from shared.tagged_buffer import EpisodeStore
from shared.thresholding import partition_episodes
from wrappers import make_crafter_env

EpisodeData = namedtuple(
    "EpisodeData", ["observations", "actions", "rewards", "dones"]
)


def load_sac_agent(checkpoint_path: str, device: str = "cpu"):
    """Load a frozen SAC actor + metadata.

    Args:
        checkpoint_path: path to .pt checkpoint.
        device:          torch device string.

    Returns:
        (actor, episode_count)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = DiscreteActor(n_actions=ckpt.get("action_dim", 17))
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.to(device).eval()
    return actor, ckpt.get("episode_count", 0)


def load_sac_critics(checkpoint_path: str, device: str = "cpu"):
    """Load frozen SAC critics (Q1, Q2) and entropy temperature α.

    Returns:
        (critic1, critic2, alpha)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_actions = ckpt.get("action_dim", 17)
    c1 = DiscreteCritic(n_actions=n_actions)
    c1.load_state_dict(ckpt["critic1_state_dict"])
    c1.to(device).eval()
    c2 = DiscreteCritic(n_actions=n_actions)
    c2.load_state_dict(ckpt["critic2_state_dict"])
    c2.to(device).eval()
    alpha = math.exp(ckpt.get("log_alpha", math.log(0.2)))
    return c1, c2, alpha


def evaluate_frozen_policy(
    checkpoint_path: str, n_episodes: int = 1000, device: str = "cpu"
):
    """Run n_episodes with a frozen SAC actor on Crafter.

    Observations stored in EpisodeData are (T, C, H, W) float32 tensors
    in [0, 1], ready for actor forward passes and activation hooks.

    Returns:
        episodes:   list of EpisodeData namedtuples
        eps_scores: list of float EPS scores (one per episode)
    """
    actor, _ = load_sac_agent(checkpoint_path, device)
    actor.eval()
    env = make_crafter_env()

    episodes = []
    eps_scores = []

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes", unit="ep"):
        obs, info = env.reset()
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        done = False
        while not done:
            obs_t = _obs_to_tensor(obs).unsqueeze(0).to(device)  # (1,C,H,W)
            with torch.no_grad():
                action = actor.get_distribution(obs_t).sample().item()

            ep_obs.append(_obs_to_tensor(obs))   # (C,H,W) float32
            ep_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rewards.append(float(reward))
            ep_dones.append(float(done))

        eps_scores.append(float(info.get("eps", 0.0)))
        episodes.append(
            EpisodeData(
                observations=torch.stack(ep_obs),            # (T, C, H, W) float32
                actions=torch.tensor(ep_actions, dtype=torch.long),
                rewards=torch.tensor(ep_rewards, dtype=torch.float32),
                dones=torch.tensor(ep_dones, dtype=torch.float32),
            )
        )

    env.close()
    return episodes, eps_scores


def partition_episode_store(
    episode_store: EpisodeStore,
    n_samples: int = 5000,
    device: str = "cpu",
    mode: str = "eps",
    percentile_x: int = 25,
):
    """Partition EpisodeStore transitions into success / failure batches.

    Returns:
        success_batch: dict with 'state' (N,C,H,W), 'action' (N,1) tensors
        failure_batch: same structure
        threshold:     float or (lower, upper) tuple
    """
    transitions = episode_store.transitions
    if not transitions:
        empty = {"state": torch.zeros(0, 3, 64, 64), "action": torch.zeros(0, 1, dtype=torch.long)}
        return empty, empty, 0.0

    # Sample up to n_samples transitions
    if len(transitions) > n_samples:
        idx = np.random.choice(len(transitions), n_samples, replace=False)
        transitions = [transitions[i] for i in idx]

    states = torch.stack([torch.tensor(t["state"], dtype=torch.float32) for t in transitions])
    actions = torch.tensor([[t["action"]] for t in transitions], dtype=torch.long)
    eps_scores = [t["eps"] for t in transitions]

    # Build dummy EpisodeData objects for partition_episodes
    from sac.sampling import EpisodeData
    # Use per-transition EPS as score; each "episode" is a single transition
    pseudo_episodes = [
        EpisodeData(
            observations=states[i: i+1],
            actions=actions[i: i+1, 0],
            rewards=torch.tensor([t["reward"]], dtype=torch.float32),
            dones=torch.tensor([t["terminal"]], dtype=torch.float32),
        )
        for i, t in enumerate(transitions)
    ]

    success_eps, failure_eps, threshold = partition_episodes(
        pseudo_episodes, eps_scores, mode=mode, percentile_x=percentile_x
    )

    def _to_batch(eps_list):
        if not eps_list:
            return {"state": torch.zeros(0, 3, 64, 64), "action": torch.zeros(0, 1, dtype=torch.long)}
        obs = torch.cat([ep.observations for ep in eps_list])
        act = torch.cat([ep.actions.unsqueeze(1) for ep in eps_list])
        return {"state": obs.to(device), "action": act.to(device)}

    return _to_batch(success_eps), _to_batch(failure_eps), threshold


def partition(episodes, eps_scores, mode: str = "eps", percentile_x: int = 25):
    """Partition EpisodeData list into success / failure groups."""
    if mode == "percentile":
        scores = [ep.rewards.sum().item() for ep in episodes]
    else:
        scores = eps_scores
    return partition_episodes(episodes, scores, mode=mode, percentile_x=percentile_x)
