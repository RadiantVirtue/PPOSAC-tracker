"""Unified analysis pipeline for PPO and Rainbow checkpoints.

Computes gradient and activation metrics:
  - opposition_score  (∇θ success vs failure)
  - coherence         (within-group gradient alignment)
  - activation_separation / activation_cosine_distance  (centroid distances)
  - cluster_stats     (UMAP + HDBSCAN clustering on activations)

gradient_magnitude is reported per-algorithm only (absolute scale differs).

RSA has been removed — pixel observations make object-index stimulus sets
inapplicable to Crafter.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch


MIN_FAILURE_FOR_OPPOSITION = 10


def analyze_checkpoint(
    algorithm: str,
    checkpoint_path: str,
    experiment_root: str,
    n_episodes: int = 500,
    device: str = "cpu",
    reason: str = "",
    split_mode: str = "percentile",
    percentile_x: int = 25,
    seed: int = None,
    prev_checkpoint_path: str = None,
    fixed_thresholds: tuple | None = None,
    reference_stimuli: frozenset | None = None,
    eps_weight: float = 0.9,
):
    from shared.storage import save_analysis_results

    args = argparse.Namespace(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        experiment_root=experiment_root,
        n_episodes=n_episodes,
        device=device,
        reason=reason,
        split_mode="fixed" if fixed_thresholds is not None else split_mode,
        percentile_x=percentile_x,
        seed=seed,
        prev_checkpoint_path=prev_checkpoint_path,
        fixed_thresholds=fixed_thresholds,
        reference_stimuli=reference_stimuli,
        eps_weight=eps_weight,
    )

    if algorithm == "ppo":
        result = _analyze_ppo(args)
    elif algorithm == "rainbow":
        result = _analyze_rainbow(args)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose 'ppo' or 'rainbow'.")

    if result is not None:
        result["reason"] = reason
        basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        out_dir = os.path.join(experiment_root, "analysis_logs", algorithm)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{basename}.json")
        save_analysis_results(result, out_path)
        print(f"Analysis saved: {out_path}")
    else:
        print("Analysis returned no results (one partition group was empty).")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a single checkpoint (gradients + activations)"
    )
    parser.add_argument("--algorithm", required=True, choices=["ppo", "rainbow"])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reason", default="")
    parser.add_argument("--split_mode", default="percentile", choices=["eps", "percentile"])
    parser.add_argument("--percentile_x", type=int, default=25)
    args = parser.parse_args()

    analyze_checkpoint(
        args.algorithm, args.checkpoint_path, args.experiment_root,
        args.n_episodes, args.device, args.reason,
        args.split_mode, args.percentile_x,
    )



def _analyze_ppo(args):
    from ppo.sampling import evaluate_frozen_policy, load_ppo_agent, partition
    from ppo.gradients import compute_group_gradient_with_coherence
    from ppo.activations import run_activation_analysis, PPO_HOOK_LAYER
    from shared.metrics import (
        opposition_score, coherence, gradient_magnitude,
        activation_separation, centroid_cosine_distance,
    )
    from shared.rsa import run_rsa

    # Step 1: Sample & partition
    episodes, eps_scores, episodes_with_transitions = evaluate_frozen_policy(
        args.checkpoint_path, n_episodes=args.n_episodes, device=args.device,
        seed=getattr(args, "seed", None),
        eps_weight=getattr(args, "eps_weight", 0.9),
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores, mode=args.split_mode, percentile_x=args.percentile_x,
        fixed_thresholds=getattr(args, "fixed_thresholds", None),
    )
    _log_threshold(threshold, success_eps, failure_eps)

    if not success_eps or not failure_eps:
        print("  Skipping: one group is empty")
        return None

    model, episode_count = load_ppo_agent(args.checkpoint_path, device=args.device)

    # Step 2: Gradients (∇θ log π)
    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)
    _norm_s, raw_success, s_mb = compute_group_gradient_with_coherence(
        model, success_eps, batch_size=s_batch, device=args.device, desc="Grads [success]"
    )
    _norm_f, raw_failure, f_mb = compute_group_gradient_with_coherence(
        model, failure_eps, batch_size=f_batch, device=args.device, desc="Grads [failure]"
    )

    # Step 3: Activations (hook features_extractor.linear, 64-dim)
    act_results = run_activation_analysis(model, success_eps, failure_eps, device=args.device)

    # Step 4: RSA — uses all evaluation episodes (not filtered by success/failure)
    policy = model.policy.to(args.device)
    rsa_results = run_rsa(
        policy, episodes_with_transitions, layer_name=PPO_HOOK_LAYER, device=args.device,
        reference_stimuli=getattr(args, "reference_stimuli", None),
    )
    print(
        f"  RSA: {rsa_results['n_stimuli']} stimuli, "
        f"fighting={rsa_results['alignment_fighting']}, "
        f"resource={rsa_results['alignment_resource']}, "
        f"crafting={rsa_results['alignment_crafting']}, "
        f"housing={rsa_results['alignment_housing']}"
    )

    return _build_result(
        episode_count, args, threshold,
        success_eps, failure_eps,
        raw_success, raw_failure, s_mb, f_mb,
        act_results, rsa_results,
    )



def _compute_joint_is_weights(online_net, target_net, episodes, args_ns, global_step, device):
    """Forward-only pass over a combined episode pool for joint IS normalisation.

    Computes IS weights with a single shared priority denominator across all
    provided episodes, so that gradient magnitudes from different sub-groups
    (e.g. success vs failure) are on a comparable scale.

    Args:
        online_net:   DQN online network (eval mode, no grad).
        target_net:   DQN target network (eval mode, no grad).
        episodes:     combined list of EpisodeData (e.g. success_eps + failure_eps).
        args_ns:      argparse.Namespace with distribution + IS parameters.
        global_step:  checkpoint training step (for beta annealing).
        device:       torch device string.

    Returns:
        list of (T,) float32 tensors — per-episode IS weights, one per episode,
        in the same order as `episodes`. Episodes with T <= 1 get None.
    """
    import torch
    from rainbow.gradients import _forward_per_loss, _compute_is_weights

    atoms    = args_ns.atoms
    Vmin     = args_ns.V_min
    Vmax     = args_ns.V_max
    delta_z  = (Vmax - Vmin) / (atoms - 1)
    support  = torch.linspace(Vmin, Vmax, atoms).to(device)
    n        = args_ns.multi_step
    gamma    = args_ns.discount

    alpha       = getattr(args_ns, 'priority_exponent', 0.5)
    beta_start  = getattr(args_ns, 'priority_weight',   0.4)
    T_max       = getattr(args_ns, 'T_max',             int(10e6))
    learn_start = getattr(args_ns, 'learn_start',       int(20e3))
    anneal_frac = max(0.0, min(1.0, (global_step - learn_start) / max(T_max - learn_start, 1)))
    beta        = beta_start + (1.0 - beta_start) * anneal_frac

    losses_per_ep = []
    with torch.no_grad():
        for episode in episodes:
            if len(episode.rewards) > 1:
                per_loss = _forward_per_loss(
                    episode, online_net, target_net,
                    support, Vmin, Vmax, delta_z, atoms, gamma, n, device
                )
                losses_per_ep.append(per_loss.detach())
            else:
                losses_per_ep.append(None)

    valid = [l for l in losses_per_ep if l is not None]
    if not valid:
        return [None] * len(episodes)

    all_losses    = torch.cat(valid)
    all_is_w      = _compute_is_weights(all_losses, alpha, beta)

    result = []
    ptr = 0
    for l in losses_per_ep:
        if l is not None:
            T = len(l)
            result.append(all_is_w[ptr:ptr + T])
            ptr += T
        else:
            result.append(None)
    return result


def _analyze_rainbow(args):
    from rainbow.sampling import evaluate_frozen_policy, load_rainbow_nets, partition
    from rainbow.gradients import compute_group_gradient_with_coherence
    from rainbow.activations import run_activation_analysis
    from rainbow.moment_of_reward import run_moment_of_reward_analysis
    from shared.metrics import (
        opposition_score, coherence, gradient_magnitude,
        activation_separation, centroid_cosine_distance,
    )

    # Step 1: Sample & partition
    episodes, eps_scores, episodes_with_transitions = evaluate_frozen_policy(
        args.checkpoint_path, n_episodes=args.n_episodes, device=args.device,
        seed=getattr(args, "seed", None),
        eps_weight=getattr(args, "eps_weight", 0.9),
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores, mode=args.split_mode, percentile_x=args.percentile_x,
        fixed_thresholds=getattr(args, "fixed_thresholds", None),
    )
    _log_threshold(threshold, success_eps, failure_eps)

    if not success_eps or not failure_eps:
        print("  Skipping: one group is empty")
        return None

    online_net, target_net, episode_count, global_step, args_ns = load_rainbow_nets(
        args.checkpoint_path, device=args.device
    )

    # Step 2: Gradients — full offline distributional Bellman loss (three variants)
    # Joint IS weight computation: normalise over success+failure pool together so that
    # gradient_magnitude_*_is values share a common denominator and are cross-comparable.
    joint_is_weights = _compute_joint_is_weights(
        online_net, target_net, success_eps + failure_eps,
        args_ns, global_step, args.device,
    )
    is_w_success = joint_is_weights[:len(success_eps)]
    is_w_failure = joint_is_weights[len(success_eps):]

    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)
    grad_s = compute_group_gradient_with_coherence(
        online_net, success_eps, batch_size=s_batch, device=args.device,
        desc="Grads [success]", target_net=target_net, args_ns=args_ns,
        global_step=global_step, precomputed_is_weights=is_w_success,
    )
    grad_f = compute_group_gradient_with_coherence(
        online_net, failure_eps, batch_size=f_batch, device=args.device,
        desc="Grads [failure]", target_net=target_net, args_ns=args_ns,
        global_step=global_step, precomputed_is_weights=is_w_failure,
    )

    # Unpack uniform variant for backward-compatible downstream calls
    raw_success = grad_s["uniform"]["raw"]
    s_mb        = grad_s["uniform"]["batch_grads"]
    raw_failure = grad_f["uniform"]["raw"]
    f_mb        = grad_f["uniform"]["batch_grads"]

    # Step 3: Activations (hook convs, flatten to 1024-dim)
    act_results = run_activation_analysis(online_net, success_eps, failure_eps, device=args.device)

    # Step 4: RSA — uses all evaluation episodes (not filtered by success/failure)
    from rainbow.rsa import run_rsa as rainbow_run_rsa, RAINBOW_HOOK_LAYER
    rsa_results = rainbow_run_rsa(
        online_net, episodes_with_transitions,
        layer_name=RAINBOW_HOOK_LAYER, device=args.device,
        reference_stimuli=getattr(args, "reference_stimuli", None),
    )
    print(
        f"  RSA: {rsa_results['n_stimuli']} stimuli, "
        f"fighting={rsa_results['alignment_fighting']}, "
        f"resource={rsa_results['alignment_resource']}, "
        f"crafting={rsa_results['alignment_crafting']}, "
        f"housing={rsa_results['alignment_housing']}"
    )

    # Step 5: Moment of Reward Analysis (uses uniform raw_failure for comparison).
    # max_episodes=50: gradient magnitudes converge quickly; subsampling 50/250 episodes
    # gives ~10k neutral transitions — more than enough for a stable estimate, 5x faster.
    # track_coherence=False: per-episode coherence within sign groups is secondary and
    # expensive (O(n) allocations + O(n²) cosines); skip for inline longitudinal runs.
    # Exception: full 1000-episode MoR at the final checkpoint for cross-phase
    # comparability with Phase C (council condition 4).
    is_final_ckpt = global_step >= getattr(args_ns, "T_max", int(10e6))
    mor_max_eps   = None if is_final_ckpt else getattr(args, "mor_max_episodes", 50)

    mor_results = run_moment_of_reward_analysis(
        online_net, success_eps, raw_failure_grad=raw_failure, device=args.device,
        target_net=target_net, args_ns=args_ns,
        max_episodes=mor_max_eps,
        track_coherence=getattr(args, "mor_track_coherence", False),
    )

    # Step 5b: Moment of Reward Analysis — failure group (secondary metric).
    mor_failure_results = run_moment_of_reward_analysis(
        online_net, failure_eps, raw_failure_grad=raw_success, device=args.device,
        target_net=target_net, args_ns=args_ns,
        cross_group_label="success",
        max_episodes=mor_max_eps,
        track_coherence=getattr(args, "mor_track_coherence", False),
    )

    # Step 6: Weight-delta empirical validation (requires prev checkpoint)
    delta_metrics = None
    if getattr(args, "prev_checkpoint_path", None):
        from rainbow.weight_delta import compute_weight_delta_metrics
        prev_net, _, _, _, _ = load_rainbow_nets(args.prev_checkpoint_path, device=args.device)
        delta_metrics = compute_weight_delta_metrics(
            prev_net.state_dict(), online_net.state_dict(), grad_s, grad_f
        )

    return _build_result(
        episode_count, args, threshold,
        success_eps, failure_eps,
        raw_success, raw_failure, s_mb, f_mb,
        act_results, rsa_results=rsa_results, mor_results=mor_results,
        mor_failure_results=mor_failure_results,
        grad_meta_success=grad_s, grad_meta_failure=grad_f,
        delta_metrics=delta_metrics,
    )



def _log_threshold(threshold, success_eps, failure_eps):
    if isinstance(threshold, tuple):
        print(
            f"  Threshold lower={threshold[0]:.3f}, upper={threshold[1]:.3f}, "
            f"success={len(success_eps)}, failure={len(failure_eps)}"
        )
    else:
        print(
            f"  Threshold mu={threshold:.3f}, "
            f"success={len(success_eps)}, failure={len(failure_eps)}"
        )


def _build_result(
    episode_count, args, threshold,
    success_eps, failure_eps,
    raw_success, raw_failure, s_mb, f_mb,
    act_results, rsa_results=None, mor_results=None,
    mor_failure_results=None,
    grad_meta_success=None, grad_meta_failure=None,
    delta_metrics=None,
):
    from shared.metrics import (
        opposition_score, coherence, gradient_magnitude,
        activation_separation, centroid_cosine_distance,
    )

    def _safe_raw(meta, variant):
        if meta is None:
            return None
        return meta.get(variant, {}).get("raw")

    def _safe_mb(meta, variant):
        if meta is None:
            return []
        return meta.get(variant, {}).get("batch_grads", [])

    raw_s_is = _safe_raw(grad_meta_success, "is_weighted")
    raw_f_is = _safe_raw(grad_meta_failure, "is_weighted")
    mb_s_is  = _safe_mb(grad_meta_success, "is_weighted")
    mb_f_is  = _safe_mb(grad_meta_failure, "is_weighted")
    mb_s_rw  = _safe_mb(grad_meta_success, "reward_weighted")
    mb_f_rw  = _safe_mb(grad_meta_failure, "reward_weighted")

    result = {
        "episode": episode_count,
        "algorithm": args.algorithm,
        "split_mode": args.split_mode,
        "percentile_x": args.percentile_x,
        "threshold_mu":    threshold if isinstance(threshold, float) else None,
        "threshold_lower": threshold[0] if isinstance(threshold, tuple) else None,
        "threshold_upper": threshold[1] if isinstance(threshold, tuple) else None,
        "n_success": len(success_eps),
        "n_failure": len(failure_eps),
        "opposition_score": (
            opposition_score(raw_success, raw_failure)
            if len(failure_eps) >= MIN_FAILURE_FOR_OPPOSITION else None
        ),
        "coherence_success": coherence(s_mb),
        "coherence_failure": coherence(f_mb),
        "activation_separation": activation_separation(
            act_results["centroids"]["success"],
            act_results["centroids"]["failure"],
        ),
        "activation_cosine_distance": centroid_cosine_distance(
            act_results["centroids"]["success"],
            act_results["centroids"]["failure"],
        ),
        "cluster_stats": act_results["cluster_stats"],
        "gradient_magnitude_success": gradient_magnitude(raw_success),
        "gradient_magnitude_failure": gradient_magnitude(raw_failure),
        "rsa_alignment":          None,  # superseded by per-group scores below
        "rsa_alignment_fighting": rsa_results.get("alignment_fighting") if rsa_results else None,
        "rsa_alignment_resource": rsa_results.get("alignment_resource") if rsa_results else None,
        "rsa_alignment_crafting": rsa_results.get("alignment_crafting") if rsa_results else None,
        "rsa_alignment_housing":  rsa_results.get("alignment_housing")  if rsa_results else None,
        "rsa_n_stimuli": rsa_results["n_stimuli"] if rsa_results else 0,
        "rsa_labels":    rsa_results["labels"]    if rsa_results else [],
        "rsa_rdm":       rsa_results["rdm"]       if rsa_results else None,
        "rsa_n_frames":  rsa_results["n_frames"]  if rsa_results else {},
        "moment_of_reward":         mor_results,
        "moment_of_reward_failure": mor_failure_results,
        "opposition_score_is": (
            opposition_score(raw_s_is, raw_f_is)
            if (raw_s_is is not None and raw_f_is is not None
                and len(failure_eps) >= MIN_FAILURE_FOR_OPPOSITION)
            else None
        ),
        "coherence_success_is":     coherence(mb_s_is) if mb_s_is else None,
        "coherence_failure_is":     coherence(mb_f_is) if mb_f_is else None,
        "gradient_magnitude_success_is": gradient_magnitude(raw_s_is) if raw_s_is is not None else None,
        "gradient_magnitude_failure_is": gradient_magnitude(raw_f_is) if raw_f_is is not None else None,
        "coherence_success_reward": coherence(mb_s_rw) if mb_s_rw else None,
        "coherence_failure_reward": coherence(mb_f_rw) if mb_f_rw else None,
        "cos_uniform_is_success":  grad_meta_success["cos_uniform_is"] if grad_meta_success else None,
        "cos_uniform_is_failure":  grad_meta_failure["cos_uniform_is"] if grad_meta_failure else None,
        "cos_is_reward_success":   grad_meta_success["cos_is_reward"]  if grad_meta_success else None,
        "cos_is_reward_failure":   grad_meta_failure["cos_is_reward"]  if grad_meta_failure else None,
        "beta_used":              grad_meta_success["beta_used"]      if grad_meta_success else None,
        "n_transitions_success":  grad_meta_success["n_transitions"]  if grad_meta_success else None,
        "n_transitions_failure":  grad_meta_failure["n_transitions"]  if grad_meta_failure else None,
        # Use .get() per key: handles both delta_metrics=None (no prev checkpoint) and
        # delta_metrics={"cos_is_success": None, ...} (prev provided but variant unavailable).
        **{k: (delta_metrics or {}).get(v) for k, v in (
            ("cos_uniform_success_delta", "cos_uniform_success"),
            ("cos_is_success_delta",      "cos_is_success"),
            ("cos_reward_success_delta",  "cos_reward_success"),
            ("cos_uniform_failure_delta", "cos_uniform_failure"),
            ("cos_is_failure_delta",      "cos_is_failure"),
            ("cos_reward_failure_delta",  "cos_reward_failure"),
        )},
    }
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
