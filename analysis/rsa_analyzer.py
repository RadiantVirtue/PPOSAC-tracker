"""RSA analysis using achievement stimulus frames collected during evaluation.

Consolidates shared/rsa.py and rainbow/rsa.py — the only difference between
them was the obs_to_tensor call, now handled by entity.preprocess_obs.

Builds a cosine-dissimilarity RDM over mean activation centroids per stimulus,
then computes Spearman ρ alignment to each group in entity.achievement_groups.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

import core.activation_utils as activation_utils
from core.entity import Entity


def analyze(
    entity: Entity,
    model: Any,
    achievement_frames: dict[str, list[np.ndarray]],
    reference_stimuli: frozenset | None,
    device: str,
) -> dict:
    """Run the full RSA pipeline.

    Args:
        entity:             entity instance (provides preprocess_obs, get_policy, hook_layer,
                            achievement_groups)
        model:              loaded checkpoint model
        achievement_frames: display_label -> list of (H,W,C) uint8 frames collected during eval
        reference_stimuli:  frozen label set from final checkpoint to keep RDM dimensions
                            constant across checkpoints; unobserved stimuli → NaN rows
        device:             torch device string

    Returns:
        {
            "rdm":        list[list[float]] | None,
            "labels":     list[str],
            "n_stimuli":  int,
            "alignments": {group_name: float | None},
        }
    """
    if reference_stimuli is not None:
        labels = sorted(reference_stimuli)
    else:
        observed = [k for k, v in achievement_frames.items() if len(v) > 0]
        if len(observed) < 2:
            return {
                "rdm":        None,
                "labels":     observed,
                "n_stimuli":  len(observed),
                "alignments": {g: None for g in entity.achievement_groups},
            }
        labels = sorted(observed)

    policy = entity.get_policy(model)

    centroids = {}
    for label in labels:
        frames = achievement_frames.get(label, [])
        if not frames:
            centroids[label] = None
        else:
            obs_np     = np.stack(frames)                               # (N, H, W, C) uint8
            obs_tensor = entity.preprocess_obs(model, obs_np, device)
            acts       = activation_utils.extract_activations(
                             policy, obs_tensor, entity.hook_layer, device)
            centroids[label] = acts.mean(axis=0)

    n_observed = sum(1 for v in centroids.values() if v is not None)
    if n_observed < 2:
        return {
            "rdm":        None,
            "labels":     labels,
            "n_stimuli":  n_observed,
            "alignments": {g: None for g in entity.achievement_groups},
        }

    rdm = _build_rdm(centroids, labels)

    alignments = {
        group_name: _alignment_score(rdm, labels, group_set)
        for group_name, group_set in entity.achievement_groups.items()
    }

    return {
        "rdm":        rdm.tolist(),
        "labels":     labels,
        "n_stimuli":  n_observed,
        "alignments": alignments,
    }


def _build_rdm(centroids: dict, labels: list[str]) -> np.ndarray:
    """Cosine dissimilarity RDM: RDM[i,j] = 1 - cos_sim(centroid_i, centroid_j).

    Entries where either centroid is None are left as NaN (stimulus not observed
    at this checkpoint — preserved across checkpoints via frozen reference set).
    """
    n = len(labels)
    rdm = np.full((n, n), np.nan)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ci, cj = centroids.get(li), centroids.get(lj)
            if ci is None or cj is None:
                continue
            ni, nj = np.linalg.norm(ci), np.linalg.norm(cj)
            if ni < 1e-8 or nj < 1e-8:
                rdm[i, j] = 1.0
            else:
                rdm[i, j] = 1.0 - float(np.clip(np.dot(ci, cj) / (ni * nj), -1.0, 1.0))
    return rdm


def _alignment_score(rdm: np.ndarray, labels: list[str],
                     group_set: frozenset) -> float | None:
    """Spearman ρ between the RDM and a binary GT matrix for one functional group.

    GT[i,j] = 0.0 if both labels are in group_set, else 1.0.
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
    valid = ~np.isnan(rdm_vals)
    rdm_vals, gt_vals = rdm_vals[valid], gt_vals[valid]
    if len(rdm_vals) < 2 or np.std(rdm_vals) < 1e-8 or np.std(gt_vals) < 1e-8:
        return None
    corr, _ = spearmanr(rdm_vals, gt_vals)
    return float(corr)
