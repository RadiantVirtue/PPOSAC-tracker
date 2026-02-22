import numpy as np
import torch

from shared.thresholding import compute_threshold


# partition a tagged replay buffer into success / failure pools
# returns (success_batch, failure_batch, mu)
def partition_buffer(buffer, n_samples=5000, device="cuda"):
    data = buffer.get_all_valid()
    mu = compute_threshold(data["episode_eps"])

    success_mask = data["episode_eps"] >= mu
    failure_mask = data["episode_eps"] < mu

    def sample_batch(mask, n):
        indices = np.where(mask)[0]
        if len(indices) < n:
            chosen = indices
        else:
            chosen = np.random.choice(indices, size=n, replace=False)
        return {
            "observations": torch.tensor(data["observations"][chosen]).to(
                device
            ),
            "actions": torch.tensor(data["actions"][chosen]).to(device),
            "rewards": torch.tensor(data["rewards"][chosen]).to(device),
        }

    success_batch = sample_batch(success_mask, n_samples)
    failure_batch = sample_batch(failure_mask, n_samples)

    return success_batch, failure_batch, mu
