"""Moment of Reward Analysis for SAC (Crafter).

Sub-partitions the Success Pool by reward sign and computes per-subgroup
gradient metrics. Answers the question: does SAC's learning signal
over-prioritize immediate reward transitions (r > 0) relative to neutral
preparatory steps (r = 0)?

Reference: Gradient Analysis v3.md, Section 3 Step 4.
"""
from sac.gradients import compute_group_gradient_with_coherence
from sac.sampling import EpisodeData
from shared.gradient_utils import cosine_similarity_flat
from shared.metrics import coherence, gradient_magnitude


MIN_MOR_EPISODES = 5  # minimum filtered episodes to compute gradients for a subgroup


def _filter_by_reward(episodes, condition_fn):
    """Keep only timesteps from each episode where condition_fn(reward) is True.

    Args:
        episodes:     list of EpisodeData
        condition_fn: callable (rewards tensor) -> bool tensor (T,)

    Returns:
        (filtered_episodes, total_transitions)
        filtered_episodes: list of EpisodeData (episodes with zero matching
                           timesteps are dropped)
        total_transitions: int total timesteps kept across all episodes
    """
    filtered = []
    n_transitions = 0
    for ep in episodes:
        mask = condition_fn(ep.rewards)   # (T,) bool
        if mask.sum() == 0:
            continue
        filtered.append(EpisodeData(
            observations=ep.observations[mask],
            actions=ep.actions[mask],
            rewards=ep.rewards[mask],
            dones=ep.dones[mask],
        ))
        n_transitions += int(mask.sum())
    return filtered, n_transitions


def run_moment_of_reward_analysis(
    actor, success_eps, raw_failure_grad=None, device="cpu",
    critic1=None, critic2=None, alpha: float = 1.0,
):
    """Sub-partition the Success Pool by reward sign and compute gradient metrics.

    Args:
        actor:            DiscreteActor (frozen weights, from sac/network.py)
        success_eps:      list of EpisodeData — the Success partition
        raw_failure_grad: optional dict of raw mean gradients for the Failure
                          group (from compute_group_gradient_with_coherence).
                          If provided, cross-group opposition scores are computed.
        device:           torch device string
        critic1:          DiscreteCritic Q1 (frozen). If provided with critic2,
                          uses the full SAC actor objective for gradient computation.
        critic2:          DiscreteCritic Q2 (frozen).
        alpha:            entropy temperature (scalar float).

    Returns:
        dict with 15 keys (all float | int | None, JSON-serializable):
            n_positive, n_neutral, n_negative
            gradient_magnitude_{positive,neutral,negative}
            coherence_{positive,neutral,negative}
            opp_pos_vs_neutral, opp_pos_vs_negative, opp_neutral_vs_negative
            opp_pos_vs_failure, opp_neutral_vs_failure, opp_negative_vs_failure
    """
    pos_eps, n_pos = _filter_by_reward(success_eps, lambda r: r > 0)
    neu_eps, n_neu = _filter_by_reward(success_eps, lambda r: r == 0)
    neg_eps, n_neg = _filter_by_reward(success_eps, lambda r: r < 0)

    print(
        f"  MOR split — positive: {n_pos} transitions ({len(pos_eps)} eps), "
        f"neutral: {n_neu} ({len(neu_eps)} eps), "
        f"negative: {n_neg} ({len(neg_eps)} eps)"
    )

    def _compute(group_eps, label):
        if len(group_eps) < MIN_MOR_EPISODES:
            print(f"  MOR [{label}]: only {len(group_eps)} episodes — skipping gradient computation")
            return None, None, None
        bs = max(2, len(group_eps) // 5)
        _norm, raw_mean, batch_grads = compute_group_gradient_with_coherence(
            actor, group_eps, batch_size=bs, device=device, desc=f"MOR [{label}]",
            critic1=critic1, critic2=critic2, alpha=alpha,
        )
        return raw_mean, gradient_magnitude(raw_mean), coherence(batch_grads)

    raw_pos, mag_pos, coh_pos = _compute(pos_eps, "positive")
    raw_neu, mag_neu, coh_neu = _compute(neu_eps, "neutral")
    raw_neg, mag_neg, coh_neg = _compute(neg_eps, "negative")

    def _opp(a, b):
        if a is None or b is None:
            return None
        return cosine_similarity_flat(a, b)

    return {
        "n_positive": n_pos,
        "n_neutral":  n_neu,
        "n_negative": n_neg,
        # per-subgroup metrics
        "gradient_magnitude_positive": mag_pos,
        "gradient_magnitude_neutral":  mag_neu,
        "gradient_magnitude_negative": mag_neg,
        "coherence_positive": coh_pos,
        "coherence_neutral":  coh_neu,
        "coherence_negative": coh_neg,
        # cross-subgroup opposition scores (within success)
        "opp_pos_vs_neutral":      _opp(raw_pos, raw_neu),
        "opp_pos_vs_negative":     _opp(raw_pos, raw_neg),
        "opp_neutral_vs_negative": _opp(raw_neu, raw_neg),
        # vs failure group (if provided)
        "opp_pos_vs_failure":      _opp(raw_pos, raw_failure_grad),
        "opp_neutral_vs_failure":  _opp(raw_neu, raw_failure_grad),
        "opp_negative_vs_failure": _opp(raw_neg, raw_failure_grad),
    }
