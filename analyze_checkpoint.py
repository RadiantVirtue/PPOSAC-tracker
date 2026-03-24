"""Unified analysis pipeline for PPO and SAC checkpoints.

Computes gradient and activation metrics that are directly comparable
between algorithms:
  - opposition_score  (∇θ log π success vs failure)
  - coherence         (within-group gradient alignment)
  - activation_separation / activation_cosine_distance  (64-dim centroids)
  - cluster_stats     (UMAP + clustering on 64-dim activations)

gradient_magnitude is reported per-algorithm but excluded from cross-algorithm
comparison (absolute scale is not comparable between PPO and SAC).

RSA has been removed — pixel observations make object-index stimulus sets
inapplicable to Crafter.
"""
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
    split_mode: str = "eps",
    percentile_x: int = 25,
    seed: int = None,
):
    from shared.storage import save_analysis_results

    args = argparse.Namespace(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        experiment_root=experiment_root,
        n_episodes=n_episodes,
        device=device,
        reason=reason,
        split_mode=split_mode,
        percentile_x=percentile_x,
        seed=seed,
    )

    if algorithm == "ppo":
        result = _analyze_ppo(args)
    elif algorithm == "sac":
        result = _analyze_sac(args)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose 'ppo' or 'sac'.")

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
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac"])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reason", default="")
    parser.add_argument("--split_mode", default="eps", choices=["eps", "percentile"])
    parser.add_argument("--percentile_x", type=int, default=25)
    args = parser.parse_args()

    analyze_checkpoint(
        args.algorithm, args.checkpoint_path, args.experiment_root,
        args.n_episodes, args.device, args.reason,
        args.split_mode, args.percentile_x,
    )


# ── PPO analysis ──────────────────────────────────────────────────────────────

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
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores, mode=args.split_mode, percentile_x=args.percentile_x,
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
    rsa_results = run_rsa(policy, episodes_with_transitions, layer_name=PPO_HOOK_LAYER, device=args.device)
    print(f"  RSA: {rsa_results['n_stimuli']} stimuli, alignment={rsa_results['alignment_score']}")

    return _build_result(
        episode_count, args, threshold,
        success_eps, failure_eps,
        raw_success, raw_failure, s_mb, f_mb,
        act_results, rsa_results,
    )


# ── SAC analysis ──────────────────────────────────────────────────────────────

def _analyze_sac(args):
    from sac.sampling import evaluate_frozen_policy, load_sac_agent, load_sac_critics, partition
    from sac.gradients import compute_group_gradient_with_coherence
    from sac.activations import run_activation_analysis
    from shared.metrics import (
        opposition_score, coherence, gradient_magnitude,
        activation_separation, centroid_cosine_distance,
    )

    # Step 1: Sample & partition
    episodes, eps_scores = evaluate_frozen_policy(
        args.checkpoint_path, n_episodes=args.n_episodes, device=args.device,
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores, mode=args.split_mode, percentile_x=args.percentile_x,
    )
    _log_threshold(threshold, success_eps, failure_eps)

    if not success_eps or not failure_eps:
        print("  Skipping: one group is empty")
        return None

    actor, episode_count = load_sac_agent(args.checkpoint_path, device=args.device)
    critic1, critic2, alpha = load_sac_critics(args.checkpoint_path, device=args.device)
    print(f"  SAC alpha={alpha:.4f}")

    # Step 2: Gradients (∇θ Σ_a π(a|s)[α log π(a|s) − min_Q(s,a)])
    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)
    _norm_s, raw_success, s_mb = compute_group_gradient_with_coherence(
        actor, success_eps, batch_size=s_batch, device=args.device, desc="Grads [success]",
        critic1=critic1, critic2=critic2, alpha=alpha,
    )
    _norm_f, raw_failure, f_mb = compute_group_gradient_with_coherence(
        actor, failure_eps, batch_size=f_batch, device=args.device, desc="Grads [failure]",
        critic1=critic1, critic2=critic2, alpha=alpha,
    )

    # Step 3: Activations (hook encoder.linear, 64-dim)
    act_results = run_activation_analysis(actor, success_eps, failure_eps, device=args.device)

    # Step 4: Moment of Reward Analysis
    from sac.moment_of_reward import run_moment_of_reward_analysis
    mor_results = run_moment_of_reward_analysis(
        actor, success_eps, raw_failure_grad=raw_failure, device=args.device,
        critic1=critic1, critic2=critic2, alpha=alpha,
    )

    return _build_result(
        episode_count, args, threshold,
        success_eps, failure_eps,
        raw_success, raw_failure, s_mb, f_mb,
        act_results, rsa_results=None, mor_results=mor_results,
    )


# ── Shared helpers ────────────────────────────────────────────────────────────

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
):
    from shared.metrics import (
        opposition_score, coherence, gradient_magnitude,
        activation_separation, centroid_cosine_distance,
    )

    return {
        "episode": episode_count,
        "algorithm": args.algorithm,
        "split_mode": args.split_mode,
        "percentile_x": args.percentile_x,
        "threshold_mu":    threshold if isinstance(threshold, float) else None,
        "threshold_lower": threshold[0] if isinstance(threshold, tuple) else None,
        "threshold_upper": threshold[1] if isinstance(threshold, tuple) else None,
        "n_success": len(success_eps),
        "n_failure": len(failure_eps),
        # ── Cross-algorithm comparable metrics ────────────────────────────────
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
        # ── Per-algorithm only (not cross-algorithm comparable) ───────────────
        "gradient_magnitude_success": gradient_magnitude(raw_success),
        "gradient_magnitude_failure": gradient_magnitude(raw_failure),
        # ── RSA ───────────────────────────────────────────────────────────────
        "rsa_alignment": rsa_results["alignment_score"] if rsa_results else None,
        "rsa_n_stimuli": rsa_results["n_stimuli"] if rsa_results else 0,
        "rsa_labels": rsa_results["labels"] if rsa_results else [],
        "rsa_rdm": rsa_results["rdm"] if rsa_results else None,
        "rsa_n_frames": rsa_results["n_frames"] if rsa_results else {},
        # ── Moment of Reward (SAC only) ───────────────────────────────────────
        "moment_of_reward": mor_results,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
