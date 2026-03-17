"""PPO gradient analysis (SB3 / Crafter).

Computes ∇θ log π(a|s) via SB3's policy.evaluate_actions(), which mirrors
the CLIP objective and is directly comparable to SAC's log_softmax gradient.
"""
import torch
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages for a single episode."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1].item()
        next_non_terminal = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_value * next_non_terminal - values[t].item()
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

    return advantages


def compute_group_gradient_with_coherence(
    model, episodes, batch_size: int = 10, device: str = "cpu", desc: str = "Grads"
):
    """Compute ∇θ log π(a|s) over a group of episodes.

    Uses SB3's policy.evaluate_actions(obs, actions) → (values, log_probs, entropy).
    GAE advantages weight each log-prob; loss = -(advantages * log_probs).mean().

    Returns:
        (l2_normalised_mean, raw_mean, list_of_batch_gradient_dicts)
    """
    policy = model.policy.to(device)
    policy.set_training_mode(True)

    overall_agg = OnlineGradientAggregator(list(policy.named_parameters()))
    batch_grads = []

    batches = range(0, len(episodes), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch"):
        batch = episodes[i: i + batch_size]
        batch_agg = OnlineGradientAggregator(list(policy.named_parameters()))

        for episode in batch:
            policy.zero_grad()

            # SB3 expects uint8 (H,W,C) numpy obs; obs_to_tensor handles normalisation
            obs_np = episode.observations.numpy()   # (T, H, W, C) uint8
            obs_tensor, _ = policy.obs_to_tensor(obs_np)  # (T, C, H, W) float32 on device

            actions_tensor = episode.actions.to(device)

            values, log_probs, _ = policy.evaluate_actions(obs_tensor, actions_tensor)

            advantages = compute_gae(
                episode.rewards, values.detach().cpu(), episode.dones
            ).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pg_loss = -(advantages * log_probs).mean()
            pg_loss.backward()

            batch_agg.accumulate(list(policy.named_parameters()))
            overall_agg.accumulate(list(policy.named_parameters()))
            policy.zero_grad()

        batch_grads.append(batch_agg.l2_normalized())

    policy.set_training_mode(False)
    return overall_agg.l2_normalized(), overall_agg.mean_gradient(), batch_grads
