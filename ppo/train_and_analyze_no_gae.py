"""PPO ablation: TD(0) advantages instead of GAE (gae_lambda=0).

Trains PPO with gae_lambda=0.0 in SB3 and uses TD(0) (lam=0) in the offline
gradient analysis. 1M steps, 5 seeds. Measures only opposition score and coherence —
no RSA, no activation analysis.
"""
import os
import sys
from dataclasses import dataclass, field

import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.gradients import compute_gae
from ppo.sampling import evaluate_frozen_policy, load_ppo_agent, partition
from ppo.train import Args as PPOArgs, main_ppo
from shared.gradient_utils import OnlineGradientAggregator
from shared.metrics import coherence, gradient_magnitude, opposition_score
from shared.storage import save_analysis_results

MIN_FAILURE_FOR_OPPOSITION = 10


@dataclass
class Args:
    seeds: list[int]          = field(default_factory=lambda: [1, 2, 3, 4, 5])
    total_timesteps: int      = 1_000_000
    checkpoint_freq: int      = 50_000
    checkpoint_achievements: bool = True
    num_procs: int            = 16
    n_steps: int              = 128
    ent_coef: float           = 0.01
    n_eval_episodes: int      = 500
    split_mode: str           = "percentile"
    percentile_x: int         = 25
    experiment_root: str      = "experiment_root_no_gae"
    device: str               = "cpu"
    analyze_every: int        = 1


def compute_group_gradient_td0(
    model, episodes, batch_size: int = 10, device: str = "cpu", desc: str = "Grads"
):
    """∇θ log π(a|s) using TD(0) advantages (lam=0).

    Identical to ppo/gradients.py::compute_group_gradient_with_coherence except
    compute_gae is called with lam=0.0 — one-step bootstrap, no multi-step credit.
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

            obs_np = episode.observations.numpy()           # (T, H, W, C) uint8
            obs_tensor, _ = policy.obs_to_tensor(obs_np)   # (T, C, H, W) float32
            actions_tensor = episode.actions.to(device)

            values, log_probs, _ = policy.evaluate_actions(obs_tensor, actions_tensor)

            # TD(0): lam=0 collapses GAE to a single-step delta
            advantages = compute_gae(
                episode.rewards, values.detach().cpu(), episode.dones, lam=0.0
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


def _analyze_checkpoint(
    checkpoint_path: str,
    seed_root: str,
    n_episodes: int,
    device: str,
    seed: int,
    split_mode: str,
    percentile_x: int,
):
    episodes, eps_scores, _ = evaluate_frozen_policy(
        checkpoint_path, n_episodes=n_episodes, device=device, seed=seed
    )
    success_eps, failure_eps, threshold = partition(
        episodes, eps_scores, mode=split_mode, percentile_x=percentile_x
    )

    if isinstance(threshold, tuple):
        print(f"  Threshold lower={threshold[0]:.3f}, upper={threshold[1]:.3f}, "
              f"success={len(success_eps)}, failure={len(failure_eps)}")
    else:
        print(f"  Threshold mu={threshold:.3f}, "
              f"success={len(success_eps)}, failure={len(failure_eps)}")

    if not success_eps or not failure_eps:
        print("  Skipping: one group is empty")
        return

    model, episode_count = load_ppo_agent(checkpoint_path, device=device)

    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)
    _, raw_success, s_mb = compute_group_gradient_td0(
        model, success_eps, batch_size=s_batch, device=device, desc="Grads [success]"
    )
    _, raw_failure, f_mb = compute_group_gradient_td0(
        model, failure_eps, batch_size=f_batch, device=device, desc="Grads [failure]"
    )

    basename = os.path.splitext(os.path.basename(checkpoint_path))[0]

    result = {
        "episode":                    episode_count,
        "algorithm":                  "ppo_no_gae",
        "reason":                     basename,
        "split_mode":                 split_mode,
        "percentile_x":               percentile_x,
        "n_success":                  len(success_eps),
        "n_failure":                  len(failure_eps),
        "opposition_score": (
            opposition_score(raw_success, raw_failure)
            if len(failure_eps) >= MIN_FAILURE_FOR_OPPOSITION else None
        ),
        "coherence_success":          coherence(s_mb),
        "coherence_failure":          coherence(f_mb),
        "gradient_magnitude_success": gradient_magnitude(raw_success),
        "gradient_magnitude_failure": gradient_magnitude(raw_failure),
    }
    out_dir = os.path.join(seed_root, "analysis_logs", "ppo_no_gae")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}.json")
    save_analysis_results(result, out_path)
    print(f"  Saved: {out_path}")


def _run_seed(args: Args, seed: int) -> str:
    seed_root = os.path.join(args.experiment_root, f"seed_{seed}")

    ppo_args = PPOArgs(
        seed=seed,
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_achievements=args.checkpoint_achievements,
        num_procs=args.num_procs,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        experiment_root=seed_root,
        gae_lambda=0.0,     # TD(0): the ablation variable
    )

    saved_paths: list[str] = []

    def on_checkpoint(path: str):
        saved_paths.append(path)
        print(f"  [checkpoint] {os.path.basename(path)}")

    print(f"=== Training PPO (no GAE, TD(0)) seed={seed} ===")
    episode_count, _ = main_ppo(ppo_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes\n")

    seen: set = set()
    unique_paths: list[str] = []
    for p in saved_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    paths_to_analyze = unique_paths[::args.analyze_every]
    if unique_paths and unique_paths[-1] not in paths_to_analyze:
        paths_to_analyze.append(unique_paths[-1])

    print(f"=== Analysing {len(paths_to_analyze)}/{len(unique_paths)} checkpoint(s) "
          f"for seed {seed} ===\n")
    for path in paths_to_analyze:
        print(f"--- {os.path.basename(path)} ---")
        _analyze_checkpoint(
            path, seed_root,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            seed=seed,
            split_mode=args.split_mode,
            percentile_x=args.percentile_x,
        )

    return seed_root


def run(args: Args):
    summary = []
    for seed in args.seeds:
        seed_root = _run_seed(args, seed)
        log_dir = os.path.join(seed_root, "analysis_logs", "ppo_no_gae")
        n_jsons = len([f for f in os.listdir(log_dir) if f.endswith(".json")]) \
            if os.path.isdir(log_dir) else 0
        summary.append((seed, n_jsons))

    print("\n=== Summary ===")
    for seed, n in summary:
        print(f"  seed {seed}: {n} checkpoints analysed")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
