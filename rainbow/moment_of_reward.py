"""Moment of Reward Analysis for Rainbow DQN (Crafter).

Sub-partitions a group of episodes by reward sign and computes per-subgroup
gradient metrics using indicator-weighted Bellman loss on intact full episodes.

Answers: does Rainbow's learning signal over-prioritise immediate reward
transitions (r > 0) relative to neutral preparatory steps (r = 0)?

Key design choice - indicator-weighted gradients (not pseudo-episodes):
    For each episode, the full per-transition loss (T,) is computed once via
    _forward_per_loss. For each sign group s, we backpropagate the weighted
    scalar (indicator_s / n_s * per_loss).sum() - three backward passes per
    episode, retain_graph for all but the last active group.

    This avoids the pseudo-episode artefact of the previous _filter_by_reward
    approach, which created artificial EpisodeData sequences (e.g. an episode
    of only r>0 transitions) that broke n-step return context.

NoisyLinear σ handling:
    In eval() mode, sigma_weight / sigma_bias have no path through the
    computation graph → .grad is None after backward →
    OnlineGradientAggregator.accumulate() skips them (checks p.grad is not
    None) → sigma entries stay at zero in mean_gradient(). Per-episode dicts
    are captured via a per-episode OnlineGradientAggregator so sigma=0 entries
    are included, consistent with compute_group_gradient_with_coherence.

Reference: Gradient Analysis v3.md, Section 3 Step 4 (corrected in v5).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator, cosine_similarity_flat
from shared.metrics import coherence, gradient_magnitude


MIN_MOR_EPISODES = 5  # minimum episodes to compute gradients for a subgroup


def _compute_sign_indicator_gradients(online_net, episodes, device, target_net, args_ns,
                                       max_episodes=None, track_coherence=False):
    """Compute per-reward-sign mean gradients via indicator-weighted Bellman loss.

    For each intact episode: calls _forward_per_loss (no torch.no_grad() wrapper -
    the online_net forward builds the computation graph), then for each non-empty
    sign group s backpropagates (indicator_s / n_s * per_loss).sum().

    Three backward passes per episode with retain_graph=True on all but the last
    active group. Per-episode gradient dicts captured via a per-episode
    OnlineGradientAggregator so sigma=0 entries are included consistently.

    Args:
        online_net:       DQN online network (should be in eval() mode).
        episodes:         list of EpisodeData - intact full episodes (success or failure).
        device:           torch device string.
        target_net:       DQN target network (frozen).
        args_ns:          argparse.Namespace with atoms, V_min, V_max, discount, multi_step.
        max_episodes:     if set, randomly subsample this many episodes before processing.
                          The gradient is still per-transition normalized - subsampling
                          reduces variance slightly but does not bias the mean estimate.
        track_coherence:  if True, capture per-episode gradient dicts for coherence
                          computation (expensive: O(n) allocations + O(n²) cosines later).
                          Set False (default) when only gradient magnitudes are needed.

    Returns:
        dict with keys "pos", "neu", "neg", each containing:
            "raw":         mean gradient dict (param_name -> tensor), or None if empty
            "batch_grads": list of per-episode mean gradient dicts (for coherence),
                           or [] if track_coherence=False
    """
    from rainbow.gradients import _forward_per_loss

    atoms   = args_ns.atoms
    Vmin    = args_ns.V_min
    Vmax    = args_ns.V_max
    delta_z = (Vmax - Vmin) / (atoms - 1)
    support = torch.linspace(Vmin, Vmax, atoms).to(device)
    gamma   = args_ns.discount
    n       = args_ns.multi_step

    named_params = list(online_net.named_parameters())

    # Stratified subsample: keep all episodes containing non-neutral rewards first,
    # then fill remaining slots from neutral-only episodes.
    # Rationale: pos/neg reward episodes are rare in Crafter; a flat random draw
    # of 50 from 1000 episodes would leave G_pos estimated from ~1-2 episodes.
    if max_episodes is not None and len(episodes) > max_episodes:
        rng = np.random.default_rng(42)
        non_neutral = [ep for ep in episodes
                       if (ep.rewards > 0).any() or (ep.rewards < 0).any()]
        neutral_only = [ep for ep in episodes
                        if not (ep.rewards > 0).any() and not (ep.rewards < 0).any()]
        if len(non_neutral) >= max_episodes:
            idx = rng.choice(len(non_neutral), size=max_episodes, replace=False)
            episodes = [non_neutral[i] for i in sorted(idx)]
        else:
            n_fill = max_episodes - len(non_neutral)
            if n_fill > 0 and neutral_only:
                idx = rng.choice(len(neutral_only), size=min(n_fill, len(neutral_only)),
                                 replace=False)
                fill = [neutral_only[i] for i in sorted(idx)]
            else:
                fill = []
            episodes = non_neutral + fill

    # Per-transition normalization: pre-compute total transition counts per sign
    # across all episodes so each transition contributes equally (not each episode).
    # Episode-weighted normalization (old approach) under-represented sign groups
    # concentrated in a few episodes with many transitions of that sign.
    n_pos_total = int(sum((ep.rewards > 0).sum().item() for ep in episodes))
    n_neu_total = int(sum((ep.rewards == 0).sum().item() for ep in episodes))
    n_neg_total = int(sum((ep.rewards < 0).sum().item() for ep in episodes))
    n_total_by_key = {"pos": n_pos_total, "neu": n_neu_total, "neg": n_neg_total}

    overall_agg = {
        "pos": OnlineGradientAggregator(named_params),
        "neu": OnlineGradientAggregator(named_params),
        "neg": OnlineGradientAggregator(named_params),
    }
    ep_grads = {"pos": [], "neu": [], "neg": []}

    for ep in tqdm(episodes, desc="MOR [indicator]", leave=False):
        if len(ep.rewards) < 2:
            continue  # skip degenerate episodes (same guard as compute_group_gradient)

        # Full per-transition Bellman loss - gradient-tracked (no no_grad wrapper)
        per_loss = _forward_per_loss(
            ep, online_net, target_net,
            support, Vmin, Vmax, delta_z, atoms, gamma, n, device
        )  # (T,)

        r = ep.rewards.to(device) if hasattr(ep.rewards, "to") else torch.tensor(
            ep.rewards, dtype=torch.float32, device=device
        )
        masks = {
            "pos": (r > 0).float(),
            "neu": (r == 0).float(),
            "neg": (r < 0).float(),
        }

        # Only process groups with transitions in this episode AND globally non-zero.
        # If masks[k].sum() > 0, then n_total_by_key[k] > 0 is guaranteed.
        active = [
            (k, masks[k])
            for k in ("pos", "neu", "neg")
            if masks[k].sum().item() > 0
        ]

        for i, (key, mask) in enumerate(active):
            is_last = (i == len(active) - 1)
            online_net.zero_grad()
            # Per-transition normalization: divide by TOTAL transitions of this sign
            # across ALL episodes, so each transition contributes equally to G_s.
            (mask / n_total_by_key[key] * per_loss).sum().backward(retain_graph=not is_last)
            # Optionally capture per-episode gradient for coherence computation.
            if track_coherence:
                ep_agg = OnlineGradientAggregator(named_params)
                ep_agg.accumulate(named_params)
                ep_grads[key].append(ep_agg.mean_gradient())
            overall_agg[key].accumulate(named_params)

        online_net.zero_grad()  # clean up after final sign group's backward

    # Sanity check: OnlineGradientAggregator.accumulate() must be a pure +=
    # (no implicit division by count). If this ever changes, the per-transition
    # normalization encoded in the backward call above becomes silently wrong.
    for key, agg in overall_agg.items():
        if agg.count >= 2:
            for name in list(agg.sum_grads)[:1]:  # one param is sufficient
                expected = agg.sum_grads[name] / agg.count
                actual   = agg.mean_gradient()[name]
                assert torch.allclose(expected, actual, atol=1e-6), (
                    f"OnlineGradientAggregator.accumulate() semantics changed for "
                    f"'{name}': sum_grads is no longer a pure running sum. "
                    "Per-transition normalization in _compute_sign_indicator_gradients "
                    "is broken - revisit this function."
                )
            break  # one sign group is enough

    # Return the raw accumulated sum, NOT mean_gradient().
    # Each episode's contribution was already scaled by /n_total_by_key[key], so
    # summing across episodes gives exactly the per-transition mean gradient:
    #   G_s = (1/N_s) * sum_{all t with r_t in s} loss_t
    return {
        key: {
            "raw": {name: overall_agg[key].sum_grads[name].clone()
                    for name in overall_agg[key].sum_grads}
                   if overall_agg[key].count > 0 else None,
            "batch_grads": ep_grads[key],
        }
        for key, agg in overall_agg.items()
    }


def run_moment_of_reward_analysis(
    online_net, episodes, raw_failure_grad=None, device="cpu",
    target_net=None, args_ns=None,
    cross_group_label="failure",
    max_episodes=None, track_coherence=False,
):
    """Sub-partition episodes by reward sign and compute gradient metrics.

    Uses indicator-weighted gradients on intact full episodes - no pseudo-episodes.

    Args:
        online_net:          DQN online network (frozen weights).
        episodes:            list of EpisodeData - the episode group to analyse
                             (typically the success or failure partition).
        raw_failure_grad:    optional mean gradient dict for the cross-group
                             comparison (pass the failure-group gradient when
                             analysing success episodes; pass the success-group
                             gradient when analysing failure episodes).
        device:              torch device string.
        target_net:          DQN target network (frozen).
        args_ns:             argparse.Namespace with distribution parameters.
        cross_group_label:   key suffix for cross-group opposition scores.
                             Use "failure" (default) when analysing success episodes
                             so keys are opp_*_vs_failure. Use "success" when
                             analysing failure episodes so keys are opp_*_vs_success.

    Returns:
        dict with 15 keys (all float | int | None, JSON-serializable).
    """
    if target_net is None or args_ns is None:
        print("  MOR: target_net or args_ns missing - skipping")
        return None

    sign_grads = _compute_sign_indicator_gradients(
        online_net, episodes, device, target_net, args_ns,
        max_episodes=max_episodes, track_coherence=track_coherence,
    )
    raw_pos = sign_grads["pos"]["raw"]
    raw_neu = sign_grads["neu"]["raw"]
    raw_neg = sign_grads["neg"]["raw"]

    # Count transitions by reward sign directly - no filtering artefact
    n_pos = int(sum((ep.rewards > 0).sum().item() for ep in episodes))
    n_neu = int(sum((ep.rewards == 0).sum().item() for ep in episodes))
    n_neg = int(sum((ep.rewards < 0).sum().item() for ep in episodes))

    print(
        f"  MOR split - positive: {n_pos} transitions, "
        f"neutral: {n_neu}, negative: {n_neg}"
    )

    def _opp(a, b):
        if a is None or b is None:
            return None
        return cosine_similarity_flat(a, b)

    return {
        "n_positive": n_pos,
        "n_neutral":  n_neu,
        "n_negative": n_neg,
        # per-subgroup metrics
        "gradient_magnitude_positive": gradient_magnitude(raw_pos),
        "gradient_magnitude_neutral":  gradient_magnitude(raw_neu),
        "gradient_magnitude_negative": gradient_magnitude(raw_neg),
        "coherence_positive": coherence(sign_grads["pos"]["batch_grads"]),
        "coherence_neutral":  coherence(sign_grads["neu"]["batch_grads"]),
        "coherence_negative": coherence(sign_grads["neg"]["batch_grads"]),
        # cross-subgroup opposition scores (within-group)
        "opp_pos_vs_neutral":      _opp(raw_pos, raw_neu),
        "opp_pos_vs_negative":     _opp(raw_pos, raw_neg),
        "opp_neutral_vs_negative": _opp(raw_neu, raw_neg),
        # vs cross-group gradient (opp_*_vs_failure or opp_*_vs_success)
        f"opp_pos_vs_{cross_group_label}":      _opp(raw_pos, raw_failure_grad),
        f"opp_neutral_vs_{cross_group_label}":  _opp(raw_neu, raw_failure_grad),
        f"opp_negative_vs_{cross_group_label}": _opp(raw_neg, raw_failure_grad),
    }



# Colour scheme for MoR subgroups
_C_POS = "#2ca02c"   # positive reward - green
_C_NEU = "#1f77b4"   # neutral reward  - blue
_C_NEG = "#d62728"   # negative reward - red


def plot_moment_of_reward(mor_records: list, out_dir: str, dpi: int = 150):
    """Plot Moment of Reward metrics over training steps.

    Args:
        mor_records: list of (step, mor_dict) pairs - one per analysis checkpoint.
                     mor_dict is the "moment_of_reward" sub-dict from the analysis JSON.
        out_dir:     directory to write PNG files.
        dpi:         output resolution.

    Note: these plots use pre-fix pseudo-episode data from the existing JSONs.
    The corrected indicator-weighted results are available only at the final step
    via corrected_analysis_results/.
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
        "MoR: Gradient Magnitude by Reward Sign (Rainbow)*",
        "Gradient Magnitude",
        "mor_gradient_magnitude.png",
    )

    # Coherence per subgroup
    _plot_metric(
        ["coherence_positive", "coherence_neutral", "coherence_negative"],
        ["r>0 (positive)", "r=0 (neutral)", "r<0 (negative)"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Gradient Coherence by Reward Sign (Rainbow)*",
        "Coherence",
        "mor_coherence.png",
    )

    # Cross-subgroup opposition scores (within-group)
    _plot_metric(
        ["opp_pos_vs_neutral", "opp_pos_vs_negative", "opp_neutral_vs_negative"],
        ["pos vs neutral", "pos vs negative", "neutral vs negative"],
        ["#9467bd", "#8c564b", "#e377c2"],
        "MoR: Cross-Subgroup Opposition Scores (Rainbow)*",
        "Cosine Similarity",
        "mor_opposition_within.png",
    )

    # Opposition vs failure group
    _plot_metric(
        ["opp_pos_vs_failure", "opp_neutral_vs_failure", "opp_negative_vs_failure"],
        ["pos vs failure", "neutral vs failure", "negative vs failure"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Subgroup vs Failure Opposition Scores (Rainbow)*",
        "Cosine Similarity",
        "mor_opposition_vs_failure.png",
    )

    # Transition counts
    _plot_metric(
        ["n_positive", "n_neutral", "n_negative"],
        ["r>0 transitions", "r=0 transitions", "r<0 transitions"],
        [_C_POS, _C_NEU, _C_NEG],
        "MoR: Transition Counts by Reward Sign (Rainbow)*",
        "Transition Count",
        "mor_transition_counts.png",
    )
    # * = computed with pre-fix pseudo-episode data from existing JSONs
