"""Partition an EpisodeStore into success / failure transition batches.

Analogous to ppo/sampling.py partition() but operates on the Rainbow
EpisodeStore (transitions tagged with EPS scores) rather than episode lists.
"""
import numpy as np
import torch

from rainbow.tagged_buffer import EpisodeStore


def partition_episode_store(
    episode_store: EpisodeStore,
    n_samples: int = 5000,
    device: str = "cuda",
    mode: str = "eps",
    percentile_x: int = 25,
):
    """Split stored transitions into success / failure batches.

    Args:
        episode_store: loaded EpisodeStore (from <checkpoint>_episodes.pkl).
        n_samples:     max transitions to sample from each group.
        device:        torch device for returned tensors.
        mode:          "eps"        — split at mean EPS score (default).
                       "percentile" — bottom/top percentile_x% of episodes
                                      by raw return; middle is discarded.
        percentile_x:  percentile cutoff when mode="percentile".

    Returns:
        (success_batch, failure_batch, threshold)
        Each batch is a dict with keys:
            state      (N, state_dim) float32 tensor
            action     (N, 1)         int64   tensor
            reward     (N,)           float32 tensor
            next_state (N, state_dim) float32 tensor
            terminal   (N,)           float32 tensor
        threshold is a float (eps mode) or (lower, upper) tuple (percentile mode).
    """
    transitions = episode_store.transitions
    if not transitions:
        empty = _empty_batch(device)
        return empty, empty, 0.0

    rng = np.random.default_rng(42)

    if mode == "percentile":
        # Group transitions by episode_id; score each episode by raw return.
        ep_rewards: dict = {}
        ep_transitions: dict = {}
        for t in transitions:
            eid = t["episode_id"]
            ep_rewards.setdefault(eid, 0.0)
            ep_rewards[eid] += t["reward"]
            ep_transitions.setdefault(eid, []).append(t)

        episode_ids = np.array(sorted(ep_rewards.keys()))
        scores = np.array([ep_rewards[eid] for eid in episode_ids])

        n = len(episode_ids)
        n_each = max(1, int(n * percentile_x / 100))
        sorted_idx = np.argsort(scores, kind="stable")
        failure_eids = set(episode_ids[sorted_idx[:n_each]])
        success_eids = set(episode_ids[sorted_idx[n - n_each:]])

        lower = float(scores[sorted_idx[n_each - 1]])
        upper = float(scores[sorted_idx[n - n_each]])
        threshold = (lower, upper)

        success = [t for t in transitions if t["episode_id"] in success_eids]
        failure = [t for t in transitions if t["episode_id"] in failure_eids]
    else:
        eps_arr = np.array([t["eps"] for t in transitions])
        mu = float(eps_arr.mean())
        threshold = mu
        success = [t for t in transitions if t["eps"] >= mu]
        failure = [t for t in transitions if t["eps"] < mu]

    success = _sample(success, n_samples, rng)
    failure = _sample(failure, n_samples, rng)

    return _to_batch(success, device), _to_batch(failure, device), threshold


# ── helpers ──────────────────────────────────────────────────────────────────

def _sample(transitions, n, rng):
    if len(transitions) <= n:
        return transitions
    idx = rng.choice(len(transitions), n, replace=False)
    return [transitions[i] for i in idx]


def _empty_batch(device):
    return {
        "state":      torch.zeros(0, dtype=torch.float32).to(device),
        "action":     torch.zeros(0, 1, dtype=torch.long).to(device),
        "reward":     torch.zeros(0, dtype=torch.float32).to(device),
        "next_state": torch.zeros(0, dtype=torch.float32).to(device),
        "terminal":   torch.zeros(0, dtype=torch.float32).to(device),
    }


def _to_batch(transitions, device):
    if not transitions:
        return _empty_batch(device)
    states      = np.array([t["state"]      for t in transitions], dtype=np.float32)
    actions     = np.array([t["action"]     for t in transitions], dtype=np.int64)
    rewards     = np.array([t["reward"]     for t in transitions], dtype=np.float32)
    next_states = np.array([t["next_state"] for t in transitions], dtype=np.float32)
    terminals   = np.array([t["terminal"]   for t in transitions], dtype=np.float32)
    return {
        "state":      torch.from_numpy(states).to(device),
        "action":     torch.from_numpy(actions).unsqueeze(-1).to(device),
        "reward":     torch.from_numpy(rewards).to(device),
        "next_state": torch.from_numpy(next_states).to(device),
        "terminal":   torch.from_numpy(terminals).to(device),
    }
