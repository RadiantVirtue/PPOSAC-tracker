"""SAC gradient analysis (Crafter).

Computes ∇θ log π(a|s) through the DiscreteActor, directly comparable
to PPO's gradient computation in ppo/gradients.py.

SAC's policy is a learned Categorical distribution:
    π(a|s) = softmax(actor(obs))
    log π(a|s) = log_softmax(actor(obs))[a]

This is NOT a Boltzmann approximation — log π comes from parameters
trained explicitly to maximise the entropy-regularised objective,
so the gradient is architecturally honest and directly comparable to PPO.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator


def compute_group_gradient_with_coherence(
    actor, episodes, batch_size: int = 10, device: str = "cpu", desc: str = "Grads"
):
    """Compute ∇θ log π(a|s) over a group of episodes.

    Args:
        actor:      DiscreteActor (from sac/network.py).
        episodes:   list of EpisodeData with .observations (T,C,H,W) float32,
                    .actions (T,) long.
        batch_size: episodes per coherence mini-batch.
        device:     torch device string.
        desc:       tqdm label.

    Returns:
        (l2_normalised_mean, raw_mean, list_of_batch_gradient_dicts)
    """
    actor = actor.to(device)
    actor.train()

    overall_agg = OnlineGradientAggregator(list(actor.named_parameters()))
    batch_grads = []

    batches = range(0, len(episodes), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch"):
        batch = episodes[i: i + batch_size]
        batch_agg = OnlineGradientAggregator(list(actor.named_parameters()))

        for episode in batch:
            actor.zero_grad()

            obs = episode.observations.to(device)        # (T, C, H, W) float32
            actions = episode.actions.to(device)          # (T,) long

            logits = actor(obs)                           # (T, n_actions)
            log_pi = F.log_softmax(logits, dim=-1)        # (T, n_actions)
            log_pi_taken = log_pi.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)                                  # (T,)

            loss = log_pi_taken.mean()
            loss.backward()

            batch_agg.accumulate(list(actor.named_parameters()))
            overall_agg.accumulate(list(actor.named_parameters()))
            actor.zero_grad()

        batch_grads.append(batch_agg.l2_normalized())

    actor.eval()
    return overall_agg.l2_normalized(), overall_agg.mean_gradient(), batch_grads
