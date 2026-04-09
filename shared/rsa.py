"""Representational Similarity Analysis for Crafter (pixel observations).

Stimulus set: all 22 Crafter achievements, mapped to human-readable labels.

Stimulus detection: the frame at which an achievement is FIRST unlocked within
an episode is used as the representative observation for that stimulus.
This uses only info returned by env.step() at the moment it happens — no data leak.
Episodes where an achievement was never unlocked contribute no frames for that stimulus.
Only stimuli with at least one collected frame are included in the RDM.

Grouping scheme (4 groups — items may belong to multiple groups):
  Fighting:  Zombie, Skeleton, Wood/Stone/Iron Sword
  Resource:  Wood, Stone, Iron, Coal, Wood/Stone/Iron Pickaxe
  Crafting:  Wood/Stone/Iron Pickaxe, Wood/Stone/Iron Sword, Furnace
  Housing:   Furnace, Table, Place Stone, Wake Up

One RDM is built from all observed stimuli. Four independent Spearman ρ alignment
scores are computed (one per group), so overlapping items contribute to all relevant
scores. Ungrouped stimuli are included in the RDM but always count as cross-group.
"""
import numpy as np
import torch
from scipy.stats import spearmanr


# Achievement name → human-readable stimulus label (all 22 Crafter achievements)
STIMULUS_ACHIEVEMENTS = {
    "collect_wood":        "Wood",
    "collect_stone":       "Stone",
    "collect_iron":        "Iron",
    "collect_coal":        "Coal",
    "collect_diamond":     "Diamond",
    "collect_sapling":     "Sapling",
    "collect_drink":       "Drink",
    "defeat_zombie":       "Zombie",
    "defeat_skeleton":     "Skeleton",
    "eat_plant":           "Eat Plant",
    "eat_cow":             "Eat Cow",
    "wake_up":             "Wake Up",
    "place_table":         "Table",
    "place_stone":         "Place Stone",
    "place_furnace":       "Furnace",
    "place_plant":         "Place Plant",
    "make_wood_pickaxe":   "Wood Pickaxe",
    "make_stone_pickaxe":  "Stone Pickaxe",
    "make_iron_pickaxe":   "Iron Pickaxe",
    "make_wood_sword":     "Wood Sword",
    "make_stone_sword":    "Stone Sword",
    "make_iron_sword":     "Iron Sword",
}

# Functional groups — frozensets of stimulus labels (items may appear in multiple groups)
FIGHTING = frozenset({
    "Zombie", "Skeleton",
    "Wood Sword", "Stone Sword", "Iron Sword",
})
RESOURCE = frozenset({
    "Wood", "Stone", "Iron", "Coal",
    "Wood Pickaxe", "Stone Pickaxe", "Iron Pickaxe",
})
CRAFTING = frozenset({
    "Wood Pickaxe", "Stone Pickaxe", "Iron Pickaxe",
    "Wood Sword", "Stone Sword", "Iron Sword",
    "Furnace",
})
HOUSING = frozenset({
    "Furnace", "Table", "Place Stone", "Wake Up",
})


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


def _alignment_score(rdm, labels, group_set):
    """Spearman ρ between the RDM and a binary GT matrix for one functional group.

    GT[i,j] = 0.0 if both labels[i] and labels[j] are in group_set, else 1.0.
    Items not in group_set (including ungrouped stimuli) always contribute 1.0.
    Uses upper triangle only to avoid double-counting.
    Returns None if insufficient variance in either vector.
    """
    n = len(labels)
    gt = np.array([
        [0.0 if (labels[i] in group_set and labels[j] in group_set) else 1.0
         for j in range(n)]
        for i in range(n)
    ])
    triu = np.triu_indices(n, k=1)
    rdm_vals, gt_vals = rdm[triu], gt[triu]
    if len(rdm_vals) < 2 or np.std(rdm_vals) < 1e-8 or np.std(gt_vals) < 1e-8:
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
        rdm                  — n×n list-of-lists (or None if <2 stimuli found)
        labels               — ordered list of stimulus names included
        alignment_fighting   — Spearman ρ vs Fighting group GT (or None)
        alignment_resource   — Spearman ρ vs Resource group GT (or None)
        alignment_crafting   — Spearman ρ vs Crafting group GT (or None)
        alignment_housing    — Spearman ρ vs Housing group GT (or None)
        n_stimuli            — number of stimuli with data
        n_frames             — {stimulus: count of frames collected}
    """
    stimulus_obs = _collect_stimulus_frames(episodes_with_transitions)
    n = len(stimulus_obs)
    n_frames = {k: len(v) for k, v in stimulus_obs.items()}

    if n < 2:
        return {
            "rdm": None,
            "labels": list(stimulus_obs.keys()),
            "alignment_fighting": None,
            "alignment_resource": None,
            "alignment_crafting": None,
            "alignment_housing":  None,
            "n_stimuli": n,
            "n_frames":  n_frames,
        }

    centroids = {
        label: _mean_activation(policy, obs_list, layer_name, device)
        for label, obs_list in stimulus_obs.items()
    }
    labels = sorted(centroids.keys())
    rdm = _build_rdm(centroids, labels)

    return {
        "rdm":                rdm.tolist(),
        "labels":             labels,
        "alignment_fighting": _alignment_score(rdm, labels, FIGHTING),
        "alignment_resource": _alignment_score(rdm, labels, RESOURCE),
        "alignment_crafting": _alignment_score(rdm, labels, CRAFTING),
        "alignment_housing":  _alignment_score(rdm, labels, HOUSING),
        "n_stimuli":          len(labels),
        "n_frames":           n_frames,
    }
