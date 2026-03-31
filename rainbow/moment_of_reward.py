"""Moment of Reward Analysis for Rainbow DQN (Crafter).

Sub-partitions the Success Pool by reward sign and computes per-subgroup
gradient metrics. Answers: does Rainbow's learning signal over-prioritise
immediate reward transitions (r > 0) relative to neutral preparatory steps
(r = 0)?

Adapted from sac/moment_of_reward.py. Key difference: no critic / alpha
parameters — Rainbow gradients use the distributional Bellman loss.

Reference: Gradient Analysis v3.md, Section 3 Step 4.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from rainbow.gradients import compute_group_gradient_with_coherence
from rainbow.sampling import EpisodeData
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
    """
    filtered = []
    n_transitions = 0
    for ep in episodes:
        mask = condition_fn(ep.rewards)
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
    online_net, success_eps, raw_failure_grad=None, device="cpu",
    target_net=None, args_ns=None,
):
    """Sub-partition the Success Pool by reward sign and compute gradient metrics.

    Args:
        online_net:       DQN online network (frozen weights).
        success_eps:      list of EpisodeData — the Success partition.
        raw_failure_grad: optional dict of raw mean gradients for the Failure
                          group. If provided, cross-group opposition scores computed.
        device:           torch device string.
        target_net:       DQN target network (frozen). Used for distributional loss.
        args_ns:          argparse.Namespace with distribution parameters.

    Returns:
        dict with 15 keys (all float | int | None, JSON-serializable).
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
        result = compute_group_gradient_with_coherence(
            online_net, group_eps, batch_size=bs, device=device,
            desc=f"MOR [{label}]",
            target_net=target_net, args_ns=args_ns,
        )
        raw_mean = result["uniform"]["raw"]
        batch_grads = result["uniform"]["batch_grads"]
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


# ── Plotting ──────────────────────────────────────────────────────────────────

# Colour scheme for MoR subgroups
_C_POS = "#2ca02c"   # positive reward — green
_C_NEU = "#1f77b4"   # neutral reward  — blue
_C_NEG = "#d62728"   # negative reward — red


def plot_moment_of_reward(mor_records: list, out_dir: str, dpi: int = 150):
    """Plot Moment of Reward metrics over training steps.

    Args:
        mor_records: list of (step, mor_dict) pairs — one per analysis checkpoint.
                     mor_dict is the "moment_of_reward" sub-dict from the analysis JSON.
        out_dir:     directory to write PNG files.
        dpi:         output resolution.
    """
    if not mor_records:
        return

    os.makedirs(out_dir, exist_ok=True)

    records = [r for _, r in mor_records if r is not None]
    steps_valid = [s for s, r in mor_records if r is not None]

    if not records:
        return

    def _get(rec, key):
        v = rec.get(key)
        try:
            f = float(v)
            import math
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def _vals(key):
        return [_get(r, key) for r in records]

    def _plot_metric(keys, labels, colors, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(10, 4))
        for key, label, color in zip(keys, labels, colors):
            ys = _vals(key)
            xs = np.array(steps_valid, dtype=float)
            ys_arr = np.array([v if v is not None else np.nan for v in ys])
            ax.plot(xs, ys_arr, color=color, label=label, linewidth=2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax.set_xlabel("Global Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # Gradient magnitude per subgroup
    _plot_metric(
        ["gradient_magnitude_positive", "gradient_magnitude_neutral", "gradient_magnitude_negative"],
        ["r>0 (positive)", "r=0 (neutral)", "r<0 (negative)"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Gradient Magnitude by Reward Sign (Rainbow)",
        "Gradient Magnitude",
        "mor_gradient_magnitude.png",
    )

    # Coherence per subgroup
    _plot_metric(
        ["coherence_positive", "coherence_neutral", "coherence_negative"],
        ["r>0 (positive)", "r=0 (neutral)", "r<0 (negative)"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Gradient Coherence by Reward Sign (Rainbow)",
        "Coherence",
        "mor_coherence.png",
    )

    # Cross-subgroup opposition scores
    _plot_metric(
        ["opp_pos_vs_neutral", "opp_pos_vs_negative", "opp_neutral_vs_negative"],
        ["pos vs neutral", "pos vs negative", "neutral vs negative"],
        ["#9467bd", "#8c564b", "#e377c2"],
        "MoR: Cross-Subgroup Opposition Scores (Rainbow)",
        "Cosine Similarity",
        "mor_opposition_within.png",
    )

    # Opposition vs failure group
    _plot_metric(
        ["opp_pos_vs_failure", "opp_neutral_vs_failure", "opp_negative_vs_failure"],
        ["pos vs failure", "neutral vs failure", "negative vs failure"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Subgroup vs Failure Opposition Scores (Rainbow)",
        "Cosine Similarity",
        "mor_opposition_vs_failure.png",
    )

    # Transition counts
    _plot_metric(
        ["n_positive", "n_neutral", "n_negative"],
        ["r>0 transitions", "r=0 transitions", "r<0 transitions"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Transition Counts by Reward Sign (Rainbow)",
        "Transition Count",
        "mor_transition_counts.png",
    )
