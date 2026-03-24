"""SAC gradient analysis (Crafter).

When critics are supplied, computes the full SAC actor gradient:

    ∇θ E_{s~D} [ Σ_a π(a|s) (α log π(a|s) − min(Q1,Q2)(s,a)) ]

This matches the actor loss used during training exactly (see sac/train.py
`_learn`).  α and Q are treated as constants w.r.t. θ — only the actor
parameters receive gradients.

When critics are omitted (critic1=None), falls back to ∇θ log π(a|s)
for the taken action only, which is directly comparable to PPO.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator


def compute_group_gradient_with_coherence(
    actor, episodes, batch_size: int = 10, device: str = "cpu", desc: str = "Grads",
    critic1=None, critic2=None, alpha: float = 1.0,
):
    """Compute SAC actor gradients over a group of episodes.

    Args:
        actor:      DiscreteActor (from sac/network.py).
        episodes:   list of EpisodeData with .observations (T,C,H,W) float32,
                    .actions (T,) long.
        batch_size: episodes per coherence mini-batch.
        device:     torch device string.
        desc:       tqdm label.
        critic1:    DiscreteCritic Q1 (frozen). If provided together with
                    critic2, uses the full SAC actor objective.
        critic2:    DiscreteCritic Q2 (frozen).
        alpha:      entropy temperature (scalar float).

    Returns:
        (l2_normalised_mean, raw_mean, list_of_batch_gradient_dicts)
    """
    use_q = critic1 is not None and critic2 is not None

    actor = actor.to(device)
    actor.train()
    if use_q:
        critic1 = critic1.to(device).eval()
        critic2 = critic2.to(device).eval()

    overall_agg = OnlineGradientAggregator(list(actor.named_parameters()))
    batch_grads = []

    batches = range(0, len(episodes), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch"):
        batch = episodes[i: i + batch_size]
        batch_agg = OnlineGradientAggregator(list(actor.named_parameters()))

        for episode in batch:
            actor.zero_grad()

            obs = episode.observations.to(device)        # (T, C, H, W) float32

            logits = actor(obs)                          # (T, n_actions)

            if use_q:
                # Full SAC actor loss: Σ_a π(a|s) [α log π(a|s) − min_Q(s,a)]
                log_pi = F.log_softmax(logits, dim=-1)   # (T, n_actions)
                pi     = log_pi.exp()                    # (T, n_actions)
                with torch.no_grad():
                    min_q = torch.min(
                        critic1(obs), critic2(obs)       # each (T, n_actions)
                    )
                loss = (pi * (alpha * log_pi - min_q)).sum(dim=-1).mean()
            else:
                # Fallback: ∇θ log π for taken action (PPO-comparable)
                actions = episode.actions.to(device)     # (T,) long
                log_pi = F.log_softmax(logits, dim=-1)
                log_pi_taken = log_pi.gather(
                    1, actions.unsqueeze(1)
                ).squeeze(1)
                loss = log_pi_taken.mean()

            loss.backward()

            batch_agg.accumulate(list(actor.named_parameters()))
            overall_agg.accumulate(list(actor.named_parameters()))
            actor.zero_grad()

        batch_grads.append(batch_agg.l2_normalized())

    actor.eval()
    return overall_agg.l2_normalized(), overall_agg.mean_gradient(), batch_grads
