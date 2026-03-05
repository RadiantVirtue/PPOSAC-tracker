import argparse
import os
import sys
import traceback

import gymnasium as gym
import numpy as np
import torch


MIN_FAILURE_FOR_OPPOSITION = 10


def analyze_checkpoint(algorithm, checkpoint_path, experiment_root,
                       env_id="MiniGrid-DoorKey-8x8-v0", n_episodes=500,
                       device="cuda", reason="", run_rsa=True,
                       split_mode="eps", percentile_x=25):
    from shared.storage import save_analysis_results

    args = argparse.Namespace(
        algorithm=algorithm, checkpoint_path=checkpoint_path,
        experiment_root=experiment_root, env_id=env_id,
        n_episodes=n_episodes, device=device, reason=reason,
        run_rsa=run_rsa, split_mode=split_mode, percentile_x=percentile_x,
    )

    if algorithm == "ppo":
        result = _analyze_ppo(args)
    else:
        result = _analyze_sac(args)

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
        description="Analyse a single checkpoint (gradients + activations + RSA)"
    )
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac"])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--env_id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reason", default="")
    parser.add_argument("--no_rsa", action="store_true")
    parser.add_argument("--split_mode", default="eps", choices=["eps", "percentile"])
    parser.add_argument("--percentile_x", type=int, default=25)
    args = parser.parse_args()

    analyze_checkpoint(
        args.algorithm, args.checkpoint_path, args.experiment_root,
        args.env_id, args.n_episodes, args.device, args.reason,
        run_rsa=not args.no_rsa,
        split_mode=args.split_mode, percentile_x=args.percentile_x,
    )


# full PPO analysis: sampling, gradients, activations, RSA
def _analyze_ppo(args):
    from ppo.sampling import evaluate_frozen_policy, load_ppo_agent, partition
    from ppo.gradients import compute_group_gradient_with_coherence
    from ppo.activations import run_activation_analysis
    from shared.metrics import (
        opposition_score,
        coherence,
        gradient_magnitude,
        activation_separation,
        centroid_cosine_distance,
    )
    from analysis.run_rsa import run_rsa_for_checkpoint
    from shared.rsa import get_stimulus_config

    stimulus_set, gt_groups = get_stimulus_config(args.env_id)

    # Step 1: Sampling & Partitioning
    episodes, eps_scores = evaluate_frozen_policy(
        args.checkpoint_path, args.env_id,
        n_episodes=args.n_episodes, device=args.device,
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores,
        mode=args.split_mode, percentile_x=args.percentile_x,
    )
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

    if len(success_eps) == 0 or len(failure_eps) == 0:
        print("  Skipping: one group is empty")
        return None

    # Load agent for gradient / activation computation (no ImgObsWrapper — ACModel uses raw obs)
    env = gym.make(args.env_id)
    agent, episode = load_ppo_agent(args.checkpoint_path, env, args.device)
    env.close()

    # Step 2: Gradient Computation
    # batch_size scales with group size (analogous to checkpoint_freq = total_episodes // 10)
    # floor at 5 to avoid degenerate single-episode batches on small groups
    # _norm: per-layer L2-normalised mean (discarded — not used downstream)
    # _raw:  true mean gradient (used for opposition score and magnitude)
    # _mb:   per-batch normalised gradient dicts (used for coherence)
    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)
    _norm_s, raw_success, s_mb = compute_group_gradient_with_coherence(
        agent, success_eps, batch_size=s_batch, device=args.device, desc="Grads [success]"
    )
    _norm_f, raw_failure, f_mb = compute_group_gradient_with_coherence(
        agent, failure_eps, batch_size=f_batch, device=args.device, desc="Grads [failure]"
    )

    # Step 3: Activation Analysis
    act_results = run_activation_analysis(
        agent, success_eps, failure_eps, device=args.device
    )

    # Step 4: RSA
    if args.run_rsa:
        all_obs = torch.cat([ep.observations for ep in episodes]).numpy()
        rsa_result = run_rsa_for_checkpoint(
            agent, all_obs, "actor.0", args.device,
            stimulus_set=stimulus_set, gt_groups=gt_groups,
        )
    else:
        rsa_result = None

    return {
        "episode": episode,
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
        "gradient_magnitude_success": gradient_magnitude(raw_success),
        "gradient_magnitude_failure": gradient_magnitude(raw_failure),
        "activation_separation": activation_separation(
            act_results["centroids"]["success"],
            act_results["centroids"]["failure"],
        ),
        "activation_cosine_distance": centroid_cosine_distance(
            act_results["centroids"]["success"],
            act_results["centroids"]["failure"],
        ),
        "cluster_stats": act_results["cluster_stats"],
        "rsa": rsa_result,
    }


# full SAC analysis: buffer partitioning, gradients, activations, RSA
def _analyze_sac(args):
    from analysis.run_sac_analysis import load_sac_networks
    from sac.tagged_buffer import TaggedReplayBuffer
    from sac.gradients import compute_reinterpreted_gradient
    from sac.reward_moments import compute_reward_moment_gradients
    from sac.sampling import partition_buffer
    from shared.activation_utils import (
        extract_activations,
        reduce_dimensions,
        cluster_activations,
        compute_centroids,
    )
    from shared.metrics import (
        opposition_score,
        gradient_magnitude,
        activation_separation,
        centroid_cosine_distance,
    )
    from analysis.run_rsa import run_rsa_for_checkpoint
    from shared.rsa import get_stimulus_config

    stimulus_set, gt_groups = get_stimulus_config(args.env_id)

    buffer_path = args.checkpoint_path.replace(".pt", "_buffer.pkl")
    if not os.path.exists(buffer_path):
        print(f"  Buffer not found: {buffer_path}")
        return None

    # Step 1: Partition buffer
    buffer = TaggedReplayBuffer.load(buffer_path, args.device)
    success_batch, failure_batch, mu = partition_buffer(
        buffer, n_samples=5000, device=args.device
    )
    print(f"  Threshold mu={mu:.3f}")

    # Load networks
    env = gym.make(args.env_id)
    n_actions = env.action_space.n
    env.close()
    actor, qf1, qf2, alpha, episode = load_sac_networks(
        args.checkpoint_path, n_actions, args.device
    )

    # Step 2: Gradient computation
    grad_success = compute_reinterpreted_gradient(
        actor, qf1, qf2, success_batch, alpha, args.device
    )
    grad_failure = compute_reinterpreted_gradient(
        actor, qf1, qf2, failure_batch, alpha, args.device
    )

    # Step 3: Reward moment analysis
    reward_grads = compute_reward_moment_gradients(
        actor, qf1, qf2, success_batch, alpha, args.device
    )

    # Step 4: Activation analysis
    all_obs = torch.cat(
        [success_batch["observations"], failure_batch["observations"]]
    )
    labels = np.array(
        [1] * len(success_batch["observations"])
        + [0] * len(failure_batch["observations"])
    )
    activations = extract_activations(
        actor, all_obs, layer_name="actor.0", device=args.device
    )
    projected = reduce_dimensions(activations)
    cluster_stats = cluster_activations(projected, labels)
    centroids = compute_centroids(activations, labels)

    # Step 5: RSA
    data = buffer.get_all_valid()
    rsa_result = run_rsa_for_checkpoint(
        actor, data["observations"], "actor.0", args.device,
        stimulus_set=stimulus_set, gt_groups=gt_groups,
    )

    return {
        "episode": episode,
        "threshold_mu": mu,
        "opposition_score": opposition_score(grad_success, grad_failure),
        "gradient_magnitude_success": gradient_magnitude(grad_success),
        "gradient_magnitude_failure": gradient_magnitude(grad_failure),
        "activation_separation": activation_separation(
            centroids["success"], centroids["failure"]
        ),
        "activation_cosine_distance": centroid_cosine_distance(
            centroids["success"], centroids["failure"]
        ),
        "cluster_stats": cluster_stats,
        "reward_moments": {
            name: {
                "gradient_magnitude": (
                    gradient_magnitude(g) if g else None
                ),
                "cosine_vs_full_success": (
                    opposition_score(g, grad_success) if g else None
                ),
            }
            for name, g in reward_grads.items()
        },
        "rsa": rsa_result,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
