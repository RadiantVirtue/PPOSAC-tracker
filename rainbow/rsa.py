"""Representational Similarity Analysis for Rainbow DQN (Crafter).

For Rainbow's (T,3,H,W) float32 observation format. For PPO/SAC raw (H,W,C)
uint8 observations, use shared/rsa.py instead.

Mirrors shared/rsa.py but adapted for Rainbow's observation format:
  - EpisodeData.observations are (T, 3, H, W) float32 tensors in [0, 1]
  - obs at step_idx are (3, H, W) float32 - no obs_to_tensor conversion needed
  - Activation extraction uses RAINBOW_HOOK_LAYER ("convs"), flatten_output=True → 1024-dim

Frozen stimulus set (reference_stimuli param):
  Pass a frozenset of stimulus labels derived from the final checkpoint to keep
  the RDM dimensionality constant across checkpoints and eliminate the
  stimulus-set-expansion confound (Issue #6). Stimuli in reference_stimuli but
  not observed at a given checkpoint contribute NaN rows; Spearman ρ is computed
  on the valid (non-NaN) pairs only.

Same 4-group scheme as shared/rsa.py:
  Fighting:  Zombie, Skeleton, Wood/Stone/Iron Sword
  Resource:  Wood, Stone, Iron, Coal, Wood/Stone/Iron Pickaxe
  Crafting:  Wood/Stone/Iron Pickaxe, Wood/Stone/Iron Sword, Furnace
  Housing:   Furnace, Table, Place Stone, Wake Up

One RDM is built from all observed stimuli; four independent alignment scores computed.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from rainbow.activations import RAINBOW_HOOK_LAYER  # noqa: F401 - re-exported for callers
from shared.activation_utils import extract_activations


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

# Functional groups - frozensets of stimulus labels (items may appear in multiple groups)
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
      EpisodeData.observations are (T, 3, H, W) float32 [0, 1] tensors.
      step_idx indexes the first step where the achievement was unlocked.

    Returns: {stimulus_label: list of (3, H, W) float32 tensors}
    """
    stimulus_obs = {label: [] for label in STIMULUS_ACHIEVEMENTS.values()}
    for ep_data, transitions in episodes_with_transitions:
        for ach, step_idx in transitions.items():
            if ach in STIMULUS_ACHIEVEMENTS:
                label = STIMULUS_ACHIEVEMENTS[ach]
                stimulus_obs[label].append(ep_data.observations[step_idx])  # (3, H, W) float32
    return {k: v for k, v in stimulus_obs.items() if v}


def _mean_activation(online_net, obs_list, layer_name, device):
    """Extract activations for a list of (3, H, W) float32 tensors and return their mean.

    obs_list items are already in network input format ([0, 1] float32).
    """
    obs_tensor = torch.stack(obs_list).to(device)  # (N, 3, H, W) float32
    acts = extract_activations(
        online_net, obs_tensor, layer_name=layer_name, device=device,
        flatten_output=True,  # (B, 64, 4, 4) → (B, 1024)
    )
    return acts.mean(axis=0)  # (1024,)


def _build_rdm(centroids, labels):
    """Cosine dissimilarity RDM: RDM[i,j] = 1 - cos_sim(centroid_i, centroid_j).

    centroids is a dict mapping label → centroid array (or None if unobserved).
    Entries where either centroid is None are left as NaN so that Spearman ρ
    computation can exclude them (frozen stimulus set - Issue #6).
    """
    n = len(labels)
    rdm = np.full((n, n), np.nan)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ci, cj = centroids.get(li), centroids.get(lj)
            if ci is None or cj is None:
                continue  # leave as NaN - stimulus not observed at this checkpoint
            ni, nj = np.linalg.norm(ci), np.linalg.norm(cj)
            if ni < 1e-8 or nj < 1e-8:
                rdm[i, j] = 1.0
            else:
                rdm[i, j] = 1.0 - float(np.clip(np.dot(ci, cj) / (ni * nj), -1.0, 1.0))
    # Symmetrize observed pairs; NaN pairs stay NaN.
    valid = ~np.isnan(rdm)
    rdm_sym = np.full_like(rdm, np.nan)
    rdm_sym[valid] = rdm[valid]
    rdm_sym = np.where(valid & valid.T, (rdm + rdm.T) / 2.0, rdm_sym)
    return rdm_sym


def _alignment_score(rdm, labels, group_set):
    """Spearman ρ between the RDM and a binary GT matrix for one functional group.

    GT[i,j] = 0.0 if both labels[i] and labels[j] are in group_set, else 1.0.
    Uses upper triangle only. Returns None if insufficient variance.
    """
    n = len(labels)
    gt = np.array([
        [0.0 if (labels[i] in group_set and labels[j] in group_set) else 1.0
         for j in range(n)]
        for i in range(n)
    ])
    triu = np.triu_indices(n, k=1)
    rdm_vals, gt_vals = rdm[triu], gt[triu]
    # Exclude NaN pairs - arise when reference_stimuli is frozen and a stimulus
    # was not observed at this checkpoint (Issue #6).
    valid = ~np.isnan(rdm_vals)
    rdm_vals, gt_vals = rdm_vals[valid], gt_vals[valid]
    if len(rdm_vals) < 2 or np.std(rdm_vals) < 1e-8 or np.std(gt_vals) < 1e-8:
        return None
    corr, _ = spearmanr(rdm_vals, gt_vals)
    return float(corr)


def run_rsa(
    online_net,
    episodes_with_transitions,
    layer_name=RAINBOW_HOOK_LAYER,
    device="cpu",
    reference_stimuli: frozenset | None = None,
):
    """Full RSA pipeline for a Rainbow checkpoint.

    Args:
        online_net:                DQN online network (eval mode) - used for activation hook
        episodes_with_transitions: list of (EpisodeData, {ach: step_idx})
        layer_name:                layer to hook (default: RAINBOW_HOOK_LAYER = "convs")
        device:                    torch device string
        reference_stimuli:         optional frozenset of stimulus labels (Issue #6).
            When provided, the RDM is always built over this fixed set so that
            dimensionality is constant across checkpoints. Stimuli in
            reference_stimuli but not observed at this checkpoint contribute NaN
            rows; Spearman ρ is computed on valid (non-NaN) pairs only.
            Derive from the final checkpoint's observed stimulus set.

    Returns dict:
        rdm                  - n×n list-of-lists (or None if <2 stimuli with data)
        labels               - ordered list of stimulus names in the RDM
        alignment_fighting   - Spearman ρ vs Fighting group GT (or None)
        alignment_resource   - Spearman ρ vs Resource group GT (or None)
        alignment_crafting   - Spearman ρ vs Crafting group GT (or None)
        alignment_housing    - Spearman ρ vs Housing group GT (or None)
        n_stimuli            - number of stimuli with observed frames (not NaN)
        n_frames             - {stimulus: count of frames collected}
    """
    online_net = online_net.to(device)
    stimulus_obs = _collect_stimulus_frames(episodes_with_transitions)
    n_frames = {k: len(v) for k, v in stimulus_obs.items()}

    # Determine label set for the RDM
    if reference_stimuli is not None:
        labels = sorted(reference_stimuli)
    else:
        if len(stimulus_obs) < 2:
            return {
                "rdm": None,
                "labels": list(stimulus_obs.keys()),
                "alignment_fighting": None,
                "alignment_resource": None,
                "alignment_crafting": None,
                "alignment_housing":  None,
                "n_stimuli": len(stimulus_obs),
                "n_frames":  n_frames,
            }
        labels = sorted(stimulus_obs.keys())

    # Build centroids; None for stimuli not observed at this checkpoint
    centroids = {}
    for label in labels:
        if label in stimulus_obs:
            centroids[label] = _mean_activation(online_net, stimulus_obs[label], layer_name, device)
        else:
            centroids[label] = None  # will produce NaN row in RDM

    n_observed = sum(1 for v in centroids.values() if v is not None)
    if n_observed < 2:
        return {
            "rdm": None,
            "labels": labels,
            "alignment_fighting": None,
            "alignment_resource": None,
            "alignment_crafting": None,
            "alignment_housing":  None,
            "n_stimuli": n_observed,
            "n_frames":  n_frames,
        }

    rdm = _build_rdm(centroids, labels)

    return {
        "rdm":                rdm.tolist(),
        "labels":             labels,
        "alignment_fighting": _alignment_score(rdm, labels, FIGHTING),
        "alignment_resource": _alignment_score(rdm, labels, RESOURCE),
        "alignment_crafting": _alignment_score(rdm, labels, CRAFTING),
        "alignment_housing":  _alignment_score(rdm, labels, HOUSING),
        "n_stimuli":          n_observed,
        "n_frames":           n_frames,
    }
