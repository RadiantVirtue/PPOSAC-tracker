import numpy as np
import torch

from shared.activation_utils import (
    cluster_activations,
    compute_centroids,
    extract_activations,
    reduce_dimensions,
)


def run_activation_analysis(
    agent, success_episodes, failure_episodes, device="cuda"
):
    # full activation analysis pipeline for a PPO checkpoint (hidden layer = actor.0, Linear→64)
    success_obs = torch.cat([ep.observations for ep in success_episodes])
    failure_obs = torch.cat([ep.observations for ep in failure_episodes])
    all_obs = torch.cat([success_obs, failure_obs])
    labels = np.array([1] * len(success_obs) + [0] * len(failure_obs))

    activations = extract_activations(
        agent, all_obs, layer_name="actor.0", device=device
    )

    projected = reduce_dimensions(activations, method="umap")
    cluster_stats = cluster_activations(projected, labels)
    centroids = compute_centroids(activations, labels)

    return {
        "activations": activations,
        "projected": projected,
        "cluster_stats": cluster_stats,
        "centroids": centroids,
        "labels": labels,
    }
