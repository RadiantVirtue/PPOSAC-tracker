"""Gradient analysis for Rainbow DQN.

Analogous to ppo/gradients.py but operates on (state, action) transition
batches rather than full episodes.

Loss used: Q(s, a_taken).mean()
  — directly differentiates the Q-values for actions the agent took,
    giving a signal analogous to the PPO policy gradient.
"""
import torch

from shared.gradient_utils import OnlineGradientAggregator

_MINI_BATCH = 256   # process in chunks to avoid OOM


def compute_rainbow_gradient(net, batch, device="cuda"):
    """Compute L2-normalised mean gradient of Q(s, a_taken) over a batch.

    Args:
        net:    online Rainbow network (Dueling_CNN_Net or CNN_Net).
        batch:  dict with "state" (N, state_dim) and "action" (N, 1) tensors
                (may be on any device; moved to `device` internally).
        device: torch device string.

    Returns:
        dict {layer_name: L2-normalised mean gradient tensor}
    """
    net.to(device)
    net.train()

    states  = batch["state"].to(device)
    actions = batch["action"].to(device)   # (N, 1)

    aggregator = OnlineGradientAggregator(list(net.named_parameters()))

    for start in range(0, len(states), _MINI_BATCH):
        s_mb = states[start : start + _MINI_BATCH]
        a_mb = actions[start : start + _MINI_BATCH]

        net.zero_grad()
        q = net(s_mb)                               # (mb, n_actions)
        q_taken = q.gather(-1, a_mb).squeeze(-1)    # (mb,)
        loss = q_taken.mean()
        loss.backward()

        aggregator.accumulate(list(net.named_parameters()))

    net.zero_grad()
    net.eval()
    return aggregator.l2_normalized()
