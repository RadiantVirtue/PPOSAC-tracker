"""Rainbow activation analysis (Crafter).

Hook target: "convs" on DQN - the shared CNN block output.
Output is 4D (B, 64, 4, 4) for 64×64 canonical input; flattened to (B, 1024)
via flatten_output=True in extract_activations().

Analogous to sac/activations.py ("encoder.linear") and ppo/activations.py
("features_extractor.linear"), but uses the pre-FC shared representation
since Rainbow has no single flat bottleneck module.
"""
import numpy as np
import torch

from shared.activation_utils import (
    cluster_activations,
    compute_centroids,
    extract_activations,
    reduce_dimensions,
)

# Layer in DQN whose output is used for activation analysis.
# convs output: (B, 64, 4, 4) for 64x64 canonical → flattened to (B, 1024)
RAINBOW_HOOK_LAYER = "convs"


def run_activation_analysis(model, success_episodes, failure_episodes, device: str = "cpu"):
    """Full activation analysis pipeline for a Rainbow checkpoint.

    Hooks RAINBOW_HOOK_LAYER on the DQN to extract 1024-dim conv embeddings.
    Observations must be (T, 3, H, W) float32 tensors in [0, 1].
    """
    model = model.to(device)

    success_obs = torch.cat([ep.observations for ep in success_episodes])  # (N, 3, H, W)
    failure_obs = torch.cat([ep.observations for ep in failure_episodes])
    all_obs = torch.cat([success_obs, failure_obs]).to(device)

    labels = np.array(
        [1] * len(success_obs) + [0] * len(failure_obs)
    )

    activations = extract_activations(
        model, all_obs, layer_name=RAINBOW_HOOK_LAYER, device=device,
        flatten_output=True,  # (B, 64, 4, 4) → (B, 1024)
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
