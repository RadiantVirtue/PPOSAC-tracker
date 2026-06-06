"""Gradient accumulation and cosine similarity utilities."""
import torch
import torch.nn.functional as F


class OnlineGradientAggregator:
    """Accumulates gradients online without storing individual vectors (avoids OOM)."""

    def __init__(self, named_params):
        self.sum_grads = {}
        for name, p in named_params:
            self.sum_grads[name] = torch.zeros_like(p.data)
        self.count = 0

    def accumulate(self, named_params):
        """Add current .grad from each parameter to running sum."""
        for name, p in named_params:
            if p.grad is not None:
                self.sum_grads[name] += p.grad.detach().clone()
        self.count += 1

    def mean_gradient(self):
        """Return mean gradient per layer."""
        return {name: g / self.count for name, g in self.sum_grads.items()}

    def l2_normalized(self):
        """Return L2-normalised mean gradient per layer."""
        mean = self.mean_gradient()
        result = {}
        for name, g in mean.items():
            norm = g.norm()
            result[name] = g / (norm + 1e-8)
        return result


def cosine_similarity_layerwise(grads_a, grads_b):
    """Compute per-layer cosine similarity between two gradient dicts."""
    result = {}
    for name in grads_a:
        a_flat = grads_a[name].flatten()
        b_flat = grads_b[name].flatten()
        result[name] = F.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()
    return result


def cosine_similarity_flat(grads_a, grads_b):
    """Flatten all layers, compute a single cosine similarity."""
    vec_a = torch.cat([grads_a[n].flatten() for n in sorted(grads_a)])
    vec_b = torch.cat([grads_b[n].flatten() for n in sorted(grads_b)])
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
