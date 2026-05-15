import numpy as np
import torch
from scipy.stats import spearmanr

from shared.gradient_utils import cosine_similarity_flat


# cosine similarity between success and failure mean gradients
# -1.0 = perfect distinction, 0.0 = orthogonal, +1.0 = confused
def opposition_score(grad_success, grad_failure):
    return cosine_similarity_flat(grad_success, grad_failure)


# average pairwise cosine similarity within a list of gradient dicts
# each element is an L2-normalised gradient dict from a single episode
def coherence(minibatch_gradients):
    n = len(minibatch_gradients)
    if n < 2:
        return None
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(
                cosine_similarity_flat(minibatch_gradients[i], minibatch_gradients[j])
            )
    return float(np.mean(sims))


# L2 norm of the concatenated mean gradient vector
def gradient_magnitude(mean_gradient):
    if mean_gradient is None:
        return None
    flat = torch.cat([mean_gradient[n].flatten() for n in sorted(mean_gradient)])
    return flat.norm().item()


# euclidean distance between Success / Failure activation centroids
def activation_separation(centroid_success, centroid_failure):
    return float(np.linalg.norm(centroid_success - centroid_failure))


# cosine distance (1 - cosine similarity) between Success / Failure activation centroids
def centroid_cosine_distance(centroid_success, centroid_failure):
    s = centroid_success / (np.linalg.norm(centroid_success) + 1e-8)
    f = centroid_failure / (np.linalg.norm(centroid_failure) + 1e-8)
    return float(1.0 - np.dot(s, f))


# spearman correlation between upper triangles of model and GT RDMs
def rsa_alignment(model_rdm, ground_truth_rdm):
    idx = np.triu_indices(model_rdm.shape[0], k=1)
    r, p = spearmanr(model_rdm[idx], ground_truth_rdm[idx])
    return {"correlation": float(r), "p_value": float(p)}
