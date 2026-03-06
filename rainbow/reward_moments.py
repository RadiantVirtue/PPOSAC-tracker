"""Reward-moment gradient analysis for Rainbow DQN.

Analogous to sac/reward_moments.py.  Splits the success batch into
sub-groups by the sign of the per-step reward and computes a gradient
vector for each sub-group, revealing which reward structure drives the
network's learning signal.
"""
import numpy as np

from rainbow.gradients import compute_rainbow_gradient


# ── Sub-partitioning ─────────────────────────────────────────────────────────

def sub_partition_success(success_batch):
    """Split a success batch into positive / neutral / negative reward groups.

    Args:
        success_batch: dict with "state", "action", "reward", … tensors
                       (on any device; CPU is fine).

    Returns:
        dict {name: sub-batch dict or None if group is empty}
        Names: "positive", "neutral", "negative".
    """
    rewards = success_batch["reward"].cpu().numpy()
    pos_mask = rewards > 0
    neg_mask = rewards < 0
    neu_mask = ~(pos_mask | neg_mask)

    groups = {"positive": pos_mask, "neutral": neu_mask, "negative": neg_mask}
    result = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            result[name] = None
        else:
            result[name] = {k: v[mask] for k, v in success_batch.items()}
    return result


# ── Gradient computation per reward moment ───────────────────────────────────

def compute_reward_moment_gradients(net, success_batch, device="cuda"):
    """Compute gradient dicts for each reward-sign sub-group of the success batch.

    Args:
        net:           Rainbow network.
        success_batch: dict from partition_episode_store success group.
        device:        torch device string.

    Returns:
        dict {name: gradient_dict or None}
        Matches the structure expected by analyze_checkpoint._analyze_rainbow.
    """
    sub_batches = sub_partition_success(success_batch)
    result = {}
    for name, sub in sub_batches.items():
        if sub is None or len(sub["state"]) == 0:
            result[name] = None
        else:
            result[name] = compute_rainbow_gradient(net, sub, device)
    return result
