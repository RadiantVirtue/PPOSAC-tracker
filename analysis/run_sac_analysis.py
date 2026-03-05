import os

import gymnasium as gym
import numpy as np
import torch

from sac.gradients import compute_reinterpreted_gradient
from sac.reward_moments import compute_reward_moment_gradients
from sac.sampling import partition_buffer
from sac.tagged_buffer import TaggedReplayBuffer
from shared.activation_utils import (
    cluster_activations,
    compute_centroids,
    extract_activations,
    reduce_dimensions,
)
from shared.metrics import (
    activation_separation,
    centroid_cosine_distance,
    gradient_magnitude,
    opposition_score,
)
from shared.storage import get_checkpoint_paths, save_analysis_results


# load frozen SAC actor and Q-networks from a checkpoint
def load_sac_networks(checkpoint_path, n_actions, device="cuda"):
    from shared.networks import SACCNNActor, SACCNNQNetwork

    actor = SACCNNActor(n_actions).to(device)
    qf1 = SACCNNQNetwork(n_actions).to(device)
    qf2 = SACCNNQNetwork(n_actions).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    qf1.load_state_dict(ckpt["qf1_state_dict"])
    qf2.load_state_dict(ckpt["qf2_state_dict"])

    alpha = ckpt.get("log_alpha", torch.tensor(0.0)).exp().item()

    actor.eval()
    qf1.eval()
    qf2.eval()

    return actor, qf1, qf2, alpha, ckpt.get("episode_count", ckpt.get("global_step"))


def analyze_checkpoint(
    checkpoint_path, buffer_path, env_id="MiniGrid-DoorKey-8x8-v0",
    device="cuda",
):
    # run the full SAC analysis pipeline on a single checkpoint
    print(f"Analyzing SAC checkpoint: {checkpoint_path}")

    buffer = TaggedReplayBuffer.load(buffer_path, device)

    # Step 1: Partition buffer into success / failure
    success_batch, failure_batch, mu = partition_buffer(
        buffer, n_samples=5000, device=device
    )
    print(f"  Threshold mu={mu:.3f}")

    # Derive n_actions from env (no ImgObsWrapper needed — just action space)
    env = gym.make(env_id)
    n_actions = env.action_space.n
    env.close()

    # Load networks
    actor, qf1, qf2, alpha, episode = load_sac_networks(
        checkpoint_path, n_actions, device
    )

    # Step 2: Gradient computation
    grad_success = compute_reinterpreted_gradient(
        actor, qf1, qf2, success_batch, alpha, device
    )
    grad_failure = compute_reinterpreted_gradient(
        actor, qf1, qf2, failure_batch, alpha, device
    )

    # Step 3: Reward moment analysis
    reward_grads = compute_reward_moment_gradients(
        actor, qf1, qf2, success_batch, alpha, device
    )

    # Step 4: Activation analysis — hook "actor.0" (Linear 64→64, pre-Tanh hidden layer)
    # This is the same semantic layer as PPO's analysis, enabling direct comparison.
    all_obs = torch.cat(
        [success_batch["observations"], failure_batch["observations"]]
    )
    labels = np.array(
        [1] * len(success_batch["observations"])
        + [0] * len(failure_batch["observations"])
    )
    activations = extract_activations(
        actor, all_obs, layer_name="actor.0", device=device
    )
    projected = reduce_dimensions(activations)
    cluster_stats = cluster_activations(projected, labels)
    centroids = compute_centroids(activations, labels)

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
    }


# run analysis across all SAC checkpoints
def run_sac_analysis(experiment_root, env_id="MiniGrid-DoorKey-8x8-v0",
                     device="cuda"):
    checkpoints = get_checkpoint_paths(experiment_root, "sac")
    all_results = []

    for ckpt in checkpoints:
        buffer_path = ckpt.replace(".pt", "_buffer.pkl")
        if not os.path.exists(buffer_path):
            print(f"  Buffer not found for {ckpt}, skipping")
            continue
        result = analyze_checkpoint(ckpt, buffer_path, env_id, device)
        if result:
            all_results.append(result)

    output_path = os.path.join(
        experiment_root, "analysis_logs", "sac_gradients.json"
    )
    save_analysis_results(all_results, output_path)
    print(f"SAC analysis saved to {output_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--env_id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_sac_analysis(args.experiment_root, args.env_id, args.device)
