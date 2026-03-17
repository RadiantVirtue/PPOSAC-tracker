"""SAC activation analysis (Crafter).

Hook target: "encoder.linear" on DiscreteActor
— the CrafterCNN's 64-dim output Sequential (Linear → ReLU).
Same dimension as PPO's "features_extractor.linear" for direct comparability.
"""
import numpy as np
import torch

from shared.activation_utils import (
    cluster_activations,
    compute_centroids,
    extract_activations,
    reduce_dimensions,
)

# Layer in DiscreteActor whose output is the 64-dim embedding used for analysis.
SAC_HOOK_LAYER = "encoder.linear"


def run_activation_analysis(actor, success_episodes, failure_episodes, device: str = "cpu"):
    """Full activation analysis pipeline for a SAC checkpoint.

    Hooks SAC_HOOK_LAYER on the DiscreteActor to extract 64-dim embeddings.
    Observations must be (T, C, H, W) float32 tensors in [0, 1].
    """
    actor = actor.to(device)

    success_obs = torch.cat([ep.observations for ep in success_episodes])  # (N, C, H, W)
    failure_obs = torch.cat([ep.observations for ep in failure_episodes])
    all_obs = torch.cat([success_obs, failure_obs]).to(device)

    labels = np.array(
        [1] * len(success_obs) + [0] * len(failure_obs)
    )

    activations = extract_activations(
        actor, all_obs, layer_name=SAC_HOOK_LAYER, device=device
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
