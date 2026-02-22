import torch
import torch.nn.functional as F


# accumulates gradients online without storing individual vectors (avoids OOM)
class OnlineGradientAggregator:

    def __init__(self, named_params):
        self.sum_grads = {}
        for name, p in named_params:
            self.sum_grads[name] = torch.zeros_like(p.data)
        self.count = 0

    # add current .grad from each parameter to running sum
    def accumulate(self, named_params):
        for name, p in named_params:
            if p.grad is not None:
                self.sum_grads[name] += p.grad.detach().clone()
        self.count += 1

    # return mean gradient per layer
    def mean_gradient(self):
        return {name: g / self.count for name, g in self.sum_grads.items()}

    # return L2-normalized mean gradient per layer
    def l2_normalized(self):
        mean = self.mean_gradient()
        result = {}
        for name, g in mean.items():
            norm = g.norm()
            result[name] = g / (norm + 1e-8)
        return result


# compute per-layer cosine similarity between two gradient dicts
def cosine_similarity_layerwise(grads_a, grads_b):
    result = {}
    for name in grads_a:
        a_flat = grads_a[name].flatten()
        b_flat = grads_b[name].flatten()
        result[name] = F.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()
    return result


# flatten all layers, compute a single cosine similarity
def cosine_similarity_flat(grads_a, grads_b):
    vec_a = torch.cat([grads_a[n].flatten() for n in sorted(grads_a)])
    vec_b = torch.cat([grads_b[n].flatten() for n in sorted(grads_b)])
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
