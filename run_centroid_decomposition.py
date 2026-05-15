"""Centroid decomposition ablation.

Reanalysis of existing PPO and Rainbow checkpoints — no new training.

At each checkpoint computes:
  d+  = ||μ+ - μ₀||   (Euclidean drift of success centroid from baseline)
  d-  = ||μ- - μ₀||   (Euclidean drift of failure centroid from baseline)
  θ+  = 1 - cos(μ+, μ₀)  (cosine distance, success)
  θ-  = 1 - cos(μ-, μ₀)  (cosine distance, failure)

μ₀ is the pooled (success+failure) activation centroid at the earliest checkpoint,
before meaningful learning has occurred (~step 50k).

Produces per-seed JSONs and a two-subplot figure (PPO left, Rainbow right).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.activation_utils import compute_centroids, extract_activations
from shared.storage import save_analysis_results



def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _step_from_path(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _find_checkpoint_paths(seed_root: str, algorithm: str) -> list[str]:
    """Find all .pt checkpoint files for a seed, sorted by step number."""
    base = os.path.join(seed_root, "checkpoints", algorithm)
    paths = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".pt") and not f.endswith(".meta.json"):
                paths.append(os.path.join(root, f))
    paths.sort(key=_step_from_path)
    return paths



def _ppo_activations(
    checkpoint_path: str,
    success_eps,
    failure_eps,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (activations [N, 64], labels [N], episode_count)."""
    from ppo.sampling import load_ppo_agent

    model, episode_count = load_ppo_agent(checkpoint_path, device=device)
    policy = model.policy.to(device)

    success_obs = torch.cat([ep.observations for ep in success_eps])   # (N, H, W, C) uint8
    failure_obs = torch.cat([ep.observations for ep in failure_eps])
    all_obs_np  = torch.cat([success_obs, failure_obs]).numpy()
    obs_tensor, _ = policy.obs_to_tensor(all_obs_np)                   # (N, C, H, W) float32

    activations = extract_activations(
        policy, obs_tensor,
        layer_name="features_extractor.linear",
        device=device,
        flatten_output=False,
    )
    labels = np.array([1] * len(success_obs) + [0] * len(failure_obs))
    return activations, labels, episode_count



def _rainbow_activations(
    checkpoint_path: str,
    success_eps,
    failure_eps,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (activations [N, 1024], labels [N], episode_count)."""
    from rainbow.sampling import load_rainbow_nets

    online_net, _, episode_count, _, _ = load_rainbow_nets(checkpoint_path, device=device)
    online_net = online_net.to(device).eval()

    success_obs = torch.cat([ep.observations for ep in success_eps])   # (N, 3, H, W) float32
    failure_obs = torch.cat([ep.observations for ep in failure_eps])
    all_obs = torch.cat([success_obs, failure_obs])

    activations = extract_activations(
        online_net, all_obs,
        layer_name="convs",
        device=device,
        flatten_output=True,    # (B, 64, 4, 4) → (B, 1024)
    )
    labels = np.array([1] * len(success_obs) + [0] * len(failure_obs))
    return activations, labels, episode_count



def _evaluate_ppo(checkpoint_path: str, n_episodes: int, device: str, seed: int, eps_weight: float):
    from ppo.sampling import evaluate_frozen_policy, partition
    episodes, eps_scores, _ = evaluate_frozen_policy(
        checkpoint_path, n_episodes=n_episodes, device=device, seed=seed
    )
    success_eps, failure_eps, _ = partition(episodes, eps_scores)
    return success_eps, failure_eps


def _evaluate_rainbow(checkpoint_path: str, n_episodes: int, device: str, seed: int, eps_weight: float):
    from rainbow.sampling import evaluate_frozen_policy, partition
    episodes, eps_scores, _ = evaluate_frozen_policy(
        checkpoint_path, n_episodes=n_episodes, device=device, seed=seed,
        eps_weight=eps_weight,
    )
    success_eps, failure_eps, _ = partition(episodes, eps_scores)
    return success_eps, failure_eps



def _run_seed(
    algorithm: str,
    seed: int,
    seed_root: str,
    n_eval_episodes: int,
    device: str,
    eps_weight: float,
    max_step: int,
    output_dir: str,
):
    ckpt_paths = _find_checkpoint_paths(seed_root, algorithm)
    if not ckpt_paths:
        print(f"  [seed {seed}] No checkpoints found in {seed_root}")
        return []

    if max_step > 0:
        ckpt_paths = [p for p in ckpt_paths if _step_from_path(p) <= max_step]

    evaluate_fn    = _evaluate_ppo    if algorithm == "ppo" else _evaluate_rainbow
    activations_fn = _ppo_activations if algorithm == "ppo" else _rainbow_activations

    out_dir   = os.path.join(output_dir, algorithm)
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f"seed_{seed}.json")
    mu0_path  = os.path.join(out_dir, f"seed_{seed}_mu0.npy")

    results: list[dict] = []
    done_steps: set[int] = set()
    if os.path.exists(out_path):
        try:
            import json as _json
            with open(out_path) as fh:
                results = _json.load(fh)
            done_steps = {r["step"] for r in results}
            print(f"  [seed {seed}] Resuming — {len(done_steps)} step(s) already done: "
                  f"{sorted(done_steps)}")
        except Exception as exc:
            print(f"  [seed {seed}] Could not load existing results ({exc}), starting fresh")
            results, done_steps = [], set()

    mu_0: np.ndarray | None = None
    if os.path.exists(mu0_path):
        mu_0 = np.load(mu0_path)
        print(f"  [seed {seed}] Loaded μ₀ from disk (dim={mu_0.shape[0]})")

    # If every checkpoint is already accounted for, nothing left to do.
    if done_steps and len(done_steps) >= len(ckpt_paths) and all(
        _step_from_path(p) in done_steps for p in ckpt_paths
    ):
        print(f"  [seed {seed}] All {len(done_steps)} checkpoint(s) already done — skipping")
        return results

    # μ₀ MUST come from the first checkpoint; if it wasn't saved, restart from scratch
    # so the reference point is consistent.
    if mu_0 is None and done_steps:
        print(f"  [seed {seed}] μ₀ not on disk but partial results exist — "
              f"restarting seed to re-establish μ₀")
        results, done_steps = [], set()

    for ckpt_path in ckpt_paths:
        step = _step_from_path(ckpt_path)

        if step in done_steps:
            print(f"  [seed {seed}, step {step:,}] already done — skipping")
            continue

        print(f"  [seed {seed}, step {step:,}] {os.path.basename(ckpt_path)}")

        success_eps, failure_eps = evaluate_fn(
            ckpt_path, n_eval_episodes, device, seed, eps_weight
        )
        if not success_eps or not failure_eps:
            print("    Skipping: one partition group is empty")
            continue

        activations, labels, episode_count = activations_fn(
            ckpt_path, success_eps, failure_eps, device
        )

        if mu_0 is None:
            mu_0 = activations.mean(axis=0)
            np.save(mu0_path, mu_0)
            print(f"    μ₀ set and saved (dim={mu_0.shape[0]})")

        centroids = compute_centroids(activations, labels)
        mu_plus  = centroids["success"]
        mu_minus = centroids["failure"]

        d_plus      = float(np.linalg.norm(mu_plus  - mu_0))
        d_minus     = float(np.linalg.norm(mu_minus - mu_0))
        theta_plus  = float(1.0 - _cosine_sim(mu_plus,  mu_0))
        theta_minus = float(1.0 - _cosine_sim(mu_minus, mu_0))

        results.append({
            "step":      step,
            "episode":   episode_count,
            "n_success": int((labels == 1).sum()),
            "n_failure": int((labels == 0).sum()),
            "d_plus":    d_plus,
            "d_minus":   d_minus,
            "theta_plus":   theta_plus,
            "theta_minus":  theta_minus,
        })
        print(f"    d+={d_plus:.4f}  d-={d_minus:.4f}  "
              f"θ+={theta_plus:.4f}  θ-={theta_minus:.4f}")

        # Write after every checkpoint so a restart loses at most one step
        results_sorted = sorted(results, key=lambda r: r["step"])
        save_analysis_results(results_sorted, out_path)

    print(f"  Complete: {out_path}")
    return results



def _plot(all_results: dict[str, dict[int, list]], output_dir: str):
    """Two-subplot figure: PPO left, Rainbow right. Lines for d+ and d- with std bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    algorithms = ["ppo", "rainbow"]
    titles = ["PPO (standard GAE)", "Rainbow"]

    for ax, algo, title in zip(axes, algorithms, titles):
        seed_data = all_results.get(algo, {})
        if not seed_data:
            ax.set_title(f"{title} (no data)")
            continue

        # Collect per-step values across seeds
        step_vals: dict[int, dict[str, list]] = {}
        for seed_results in seed_data.values():
            for entry in seed_results:
                s = entry["step"]
                if s not in step_vals:
                    step_vals[s] = {"d_plus": [], "d_minus": []}
                step_vals[s]["d_plus"].append(entry["d_plus"])
                step_vals[s]["d_minus"].append(entry["d_minus"])

        steps = sorted(step_vals)
        d_plus_mean  = np.array([np.mean(step_vals[s]["d_plus"])  for s in steps])
        d_plus_std   = np.array([np.std(step_vals[s]["d_plus"])   for s in steps])
        d_minus_mean = np.array([np.mean(step_vals[s]["d_minus"]) for s in steps])
        d_minus_std  = np.array([np.std(step_vals[s]["d_minus"])  for s in steps])
        steps_k = np.array(steps) / 1_000

        ax.plot(steps_k, d_plus_mean,  label="d⁺ (success)", color="steelblue")
        ax.fill_between(steps_k, d_plus_mean - d_plus_std, d_plus_mean + d_plus_std,
                        alpha=0.2, color="steelblue")
        ax.plot(steps_k, d_minus_mean, label="d⁻ (failure)", color="tomato")
        ax.fill_between(steps_k, d_minus_mean - d_minus_std, d_minus_mean + d_minus_std,
                        alpha=0.2, color="tomato")

        ax.set_title(title)
        ax.set_xlabel("Training step (thousands)")
        ax.set_ylabel("Euclidean drift from μ₀")
        ax.legend()

    fig.suptitle("Centroid Decomposition — drift of success/failure centroids from baseline μ₀")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "centroid_decomposition.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)



@dataclass
class Args:
    ppo_experiment_root:     str       = "ppo_experiment_root"
    rainbow_experiment_root: str       = "rainbow_experiment_root"
    seeds: list[int]                   = field(default_factory=lambda: [1, 2, 3, 4, 5])
    n_eval_episodes: int               = 500
    device: str                        = "cpu"
    output_dir: str                    = "ablation_results/centroid_decomposition"
    max_step: int                      = 0     # 0 = no cap
    eps_weight: float                  = 0.9
    skip_ppo: bool                     = False
    skip_rainbow: bool                 = False


def run(args: Args):
    algo_configs = []
    if not args.skip_ppo:
        algo_configs.append(("ppo",     args.ppo_experiment_root))
    if not args.skip_rainbow:
        algo_configs.append(("rainbow", args.rainbow_experiment_root))

    all_results: dict[str, dict[int, list]] = {}

    for algorithm, exp_root in algo_configs:
        print(f"\n{'='*60}")
        print(f"Algorithm: {algorithm.upper()}  root: {exp_root}")
        print(f"{'='*60}")
        all_results[algorithm] = {}

        for seed in args.seeds:
            seed_root = os.path.join(exp_root, f"seed_{seed}")
            if not os.path.isdir(seed_root):
                print(f"  [seed {seed}] Directory not found: {seed_root} — skipping")
                continue

            print(f"\n-- seed {seed} --")
            results = _run_seed(
                algorithm=algorithm,
                seed=seed,
                seed_root=seed_root,
                n_eval_episodes=args.n_eval_episodes,
                device=args.device,
                eps_weight=args.eps_weight,
                max_step=args.max_step,
                output_dir=args.output_dir,
            )
            if results:
                all_results[algorithm][seed] = results

    _plot(all_results, args.output_dir)


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
