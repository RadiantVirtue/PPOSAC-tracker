"""Activation extraction, dimensionality reduction, clustering, and centroid utilities."""
import numpy as np
import torch
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN


def extract_activations(model, observations, layer_name, device="cuda"):
    """Extract activations from a named layer using a forward hook.

    Args:
        model:        hookable module (e.g. policy for SB3, network for Rainbow).
                      Must have named_modules()[layer_name] and be callable.
        observations: torch.Tensor already on the correct device
        layer_name:   dotted module path relative to model
        device:       torch device string
    """
    activations = []

    def hook_fn(module, input, output):
        act = output.detach().cpu()
        if act.dim() > 2:
            act = act.flatten(1)
        activations.append(act)

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        batch_size = 256
        n_batches = (len(observations) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(observations), batch_size),
                      total=n_batches, desc="activations", unit="batch", leave=False):
            batch = observations[i: i + batch_size].to(device)
            model(batch)

    handle.remove()
    return torch.cat(activations).numpy()


def reduce_dimensions(activations, method="umap", n_components=2):
    """Project activations to 2-D via UMAP or t-SNE."""
    if method == "umap":
        reducer = UMAP(n_components=n_components, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
    return reducer.fit_transform(activations)


def cluster_activations(projected):
    """Run HDBSCAN on 2-D projected activations. Returns cluster IDs (-1 = noise)."""
    clusterer = HDBSCAN(min_cluster_size=15)
    return clusterer.fit_predict(projected)


def compute_cluster_stats(projected, cluster_ids, labels):
    """Compute cluster statistics given HDBSCAN IDs and success/failure labels (1/0)."""
    stats = {
        "n_clusters": len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0),
        "noise_fraction": float((cluster_ids == -1).mean()),
        "clusters": [],
    }
    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        stats["clusters"].append({
            "success_count": int((labels[mask] == 1).sum()),
            "failure_count": int((labels[mask] == 0).sum()),
            "size": int(mask.sum()),
        })
    return stats


def compute_centroids(activations, labels):
    """Mean activation vector for success (label=1) and failure (label=0) groups."""
    return {
        "success": activations[labels == 1].mean(axis=0),
        "failure": activations[labels == 0].mean(axis=0),
    }
