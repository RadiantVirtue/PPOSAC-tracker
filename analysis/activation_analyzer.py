"""Activation analysis: UMAP + HDBSCAN + centroids for success and failure groups."""
from __future__ import annotations

from typing import Any

import numpy as np

import core.activation_utils as activation_utils
from core.data import ActivationResult, EpisodeData
from core.entity import Entity


def analyze(entity: Entity, model: Any,
            success_episodes: list[EpisodeData],
            failure_episodes: list[EpisodeData],
            device: str) -> ActivationResult:
    """Extract activations, reduce to 2-D, cluster, and compute centroids."""
    success_obs = np.concatenate([ep.observations for ep in success_episodes])
    failure_obs = np.concatenate([ep.observations for ep in failure_episodes])
    all_obs     = np.concatenate([success_obs, failure_obs])
    group_labels = np.array([1] * len(success_obs) + [0] * len(failure_obs))

    policy     = entity.get_policy(model)
    obs_tensor = entity.preprocess_obs(model, all_obs, device)

    activations    = activation_utils.extract_activations(
                         policy, obs_tensor, entity.hook_layer, device)
    projected      = activation_utils.reduce_dimensions(activations, method="umap")
    cluster_labels = activation_utils.cluster_activations(projected)
    centroids      = activation_utils.compute_centroids(activations, group_labels)
    cluster_stats  = activation_utils.compute_cluster_stats(
                         projected, cluster_labels, group_labels)

    return ActivationResult(
        activations    = activations,
        projected      = projected,
        cluster_labels = cluster_labels,
        centroids      = centroids,
        cluster_stats  = cluster_stats,
    )
