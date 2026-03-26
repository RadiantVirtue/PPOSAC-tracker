import numpy as np
import torch
from umap import UMAP
from hdbscan import HDBSCAN


# extract activations from a named layer using a forward hook
def extract_activations(model, observations, layer_name, device="cuda", flatten_output=False):
    """Extract activations from a named layer.

    Args:
        flatten_output: if True, flatten each activation tensor via .flatten(1)
                        before collecting. Required when hooking a conv layer that
                        returns 4D output (e.g. Rainbow's 'convs' block).
    """
    activations = []

    def hook_fn(module, input, output):
        act = output.detach().cpu()
        if flatten_output:
            act = act.flatten(1)
        activations.append(act)

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(observations), batch_size):
            batch = observations[i : i + batch_size].to(device)
            model(batch)

    handle.remove()
    return torch.cat(activations).numpy()


# project activations to 2-D via UMAP or t-SNE
def reduce_dimensions(activations, method="umap", n_components=2):
    if method == "umap":
        reducer = UMAP(n_components=n_components, random_state=42)
    else:
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, random_state=42)
    return reducer.fit_transform(activations)


# run HDBSCAN on projected activations, return cluster statistics
def cluster_activations(projected, labels):
    clusterer = HDBSCAN(min_cluster_size=15)
    cluster_ids = clusterer.fit_predict(projected)

    stats = {
        "n_clusters": len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0),
        "noise_fraction": float((cluster_ids == -1).mean()),
        "clusters": [],
    }

    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        stats["clusters"].append(
            {
                "success_count": int((labels[mask] == 1).sum()),
                "failure_count": int((labels[mask] == 0).sum()),
                "size": int(mask.sum()),
            }
        )

    return stats


# mean activation vector for Success (1) and Failure (0) groups
def compute_centroids(activations, labels):
    return {
        "success": activations[labels == 1].mean(axis=0),
        "failure": activations[labels == 0].mean(axis=0),
    }
