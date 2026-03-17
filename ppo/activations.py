"""PPO activation analysis (SB3 / Crafter).

Hook target: "features_extractor.linear" on model.policy
— the SB3 NatureCNN's 64-dim output Sequential (Linear → ReLU).
"""
import numpy as np
import torch

from shared.activation_utils import (
    cluster_activations,
    compute_centroids,
    extract_activations,
    reduce_dimensions,
)

# Layer in model.policy whose output is the 64-dim embedding used for analysis.
# With features_dim=64, features_extractor.linear is Sequential(Linear, ReLU) → 64-dim.
PPO_HOOK_LAYER = "features_extractor.linear"


def run_activation_analysis(model, success_episodes, failure_episodes, device: str = "cpu"):
    """Full activation analysis pipeline for a PPO checkpoint.

    Hooks PPO_HOOK_LAYER on model.policy to extract 64-dim embeddings.
    """
    policy = model.policy.to(device)

    # Stack all observations; policy.obs_to_tensor normalises uint8 → float32
    success_obs = torch.cat([ep.observations for ep in success_episodes])  # (N, H, W, C)
    failure_obs = torch.cat([ep.observations for ep in failure_episodes])
    all_obs_np = torch.cat([success_obs, failure_obs]).numpy()  # uint8

    labels = np.array(
        [1] * len(success_obs) + [0] * len(failure_obs)
    )

    # Convert to tensor via policy normalisation
    obs_tensor, _ = policy.obs_to_tensor(all_obs_np)

    activations = extract_activations(
        policy, obs_tensor, layer_name=PPO_HOOK_LAYER, device=device
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
