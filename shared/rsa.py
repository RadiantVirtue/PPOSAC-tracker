"""Representational Similarity Analysis for Crafter (pixel observations).

Stimulus set (per spec):
  Resources: Wood, Stone, Iron, Coal
  Threats:   Zombie, Skeleton  (Lava has no achievement signal — excluded)
  Tools:     Wood Pickaxe, Stone Pickaxe, Iron Pickaxe (included if achieved)

Stimulus detection: the frame at which an achievement is FIRST unlocked within
an episode is used as the representative observation for that stimulus.
This uses only info returned by env.step() at the moment it happens — no data leak.
Episodes where an achievement was never unlocked contribute no frames for that stimulus.
Only stimuli with at least one collected frame are included in the RDM.
"""
import numpy as np
import torch
from scipy.stats import spearmanr


# Achievement name → human-readable stimulus label
STIMULUS_ACHIEVEMENTS = {
    "collect_wood":       "Wood",
    "collect_stone":      "Stone",
    "collect_iron":       "Iron",
    "collect_coal":       "Coal",
    "defeat_zombie":      "Zombie",
    "defeat_skeleton":    "Skeleton",
    "make_wood_pickaxe":  "Wood Pickaxe",
    "make_stone_pickaxe": "Stone Pickaxe",
    "make_iron_pickaxe":  "Iron Pickaxe",
}

# Functional group for RDM ground-truth alignment
STIMULUS_GROUPS = {
    "Wood":          "Resource",
    "Stone":         "Resource",
    "Iron":          "Resource",
    "Coal":          "Resource",
    "Zombie":        "Threat",
    "Skeleton":      "Threat",
    "Wood Pickaxe":  "Tool",
    "Stone Pickaxe": "Tool",
    "Iron Pickaxe":  "Tool",
}


def _collect_stimulus_frames(episodes_with_transitions):
    """Gather representative observations for each stimulus.

    episodes_with_transitions: list of (EpisodeData, {ach_name: step_idx})
      step_idx is the index into episode.observations where the achievement
      was first unlocked. Comes from env.step() info — no data leak.

    Returns: {stimulus_label: list of (H,W,C) uint8 tensors}
    """
    stimulus_obs = {label: [] for label in STIMULUS_ACHIEVEMENTS.values()}
    for ep_data, transitions in episodes_with_transitions:
        for ach, step_idx in transitions.items():
            if ach in STIMULUS_ACHIEVEMENTS:
                label = STIMULUS_ACHIEVEMENTS[ach]
                stimulus_obs[label].append(ep_data.observations[step_idx])  # (H,W,C) uint8
    return {k: v for k, v in stimulus_obs.items() if v}


def _mean_activation(policy, obs_list, layer_name, device):
    """Extract activations for a list of (H,W,C) uint8 tensors and return their mean."""
    from shared.activation_utils import extract_activations
    obs_np = torch.stack(obs_list).numpy()  # (N, H, W, C) uint8
    obs_tensor, _ = policy.obs_to_tensor(obs_np)
    acts = extract_activations(policy, obs_tensor, layer_name=layer_name, device=device)
    return acts.mean(axis=0)  # (D,)


def _build_rdm(centroids, labels):
    """Cosine dissimilarity RDM: RDM[i,j] = 1 - cos_sim(centroid_i, centroid_j)."""
    n = len(labels)
    rdm = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ci, cj = centroids[li], centroids[lj]
            ni, nj = np.linalg.norm(ci), np.linalg.norm(cj)
            if ni < 1e-8 or nj < 1e-8:
                rdm[i, j] = 1.0
            else:
                rdm[i, j] = 1.0 - float(np.clip(np.dot(ci, cj) / (ni * nj), -1.0, 1.0))
    return rdm


def _alignment_score(rdm, labels):
    """Spearman correlation between the RDM and a ground-truth same/different-group matrix.

    Ground truth: 0 = same functional group, 1 = different group.
    Uses upper triangle only to avoid double-counting.
    """
    n = len(labels)
    gt = np.array([
        [0.0 if STIMULUS_GROUPS.get(labels[i]) == STIMULUS_GROUPS.get(labels[j]) else 1.0
         for j in range(n)]
        for i in range(n)
    ])
    triu = np.triu_indices(n, k=1)
    rdm_vals, gt_vals = rdm[triu], gt[triu]
    if len(rdm_vals) < 2 or np.std(rdm_vals) < 1e-8:
        return None
    corr, _ = spearmanr(rdm_vals, gt_vals)
    return float(corr)


def run_rsa(policy, episodes_with_transitions, layer_name, device="cpu"):
    """Full RSA pipeline for PPO.

    Args:
        policy:                    model.policy (SB3) — used for obs_to_tensor + hook
        episodes_with_transitions: list of (EpisodeData, {ach: step_idx})
        layer_name:                layer to hook for activation extraction
        device:                    torch device string

    Returns dict:
        rdm              — n×n list-of-lists (or None if <2 stimuli found)
        labels           — ordered list of stimulus names included
        alignment_score  — Spearman ρ vs ground-truth group structure (or None)
        n_stimuli        — number of stimuli with data
        n_frames         — {stimulus: count of frames collected}
    """
    stimulus_obs = _collect_stimulus_frames(episodes_with_transitions)
    n = len(stimulus_obs)

    if n < 2:
        return {
            "rdm": None,
            "labels": list(stimulus_obs.keys()),
            "alignment_score": None,
            "n_stimuli": n,
            "n_frames": {k: len(v) for k, v in stimulus_obs.items()},
        }

    centroids = {
        label: _mean_activation(policy, obs_list, layer_name, device)
        for label, obs_list in stimulus_obs.items()
    }
    labels = sorted(centroids.keys())
    rdm = _build_rdm(centroids, labels)
    alignment = _alignment_score(rdm, labels)

    return {
        "rdm": rdm.tolist(),
        "labels": labels,
        "alignment_score": alignment,
        "n_stimuli": len(labels),
        "n_frames": {k: len(v) for k, v in stimulus_obs.items()},
    }
