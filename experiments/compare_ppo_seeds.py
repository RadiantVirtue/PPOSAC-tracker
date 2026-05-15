"""PPO null-model comparison — Issue #5 (appendix).

Tests whether the success-group gradient is a seed-specific artefact or a
shared representational pattern across independently trained PPO agents.

Method:
  For all C(5,2) = 10 seed pairs, compute:
      similarity(i, j) = cos(G_success^seed_i, G_success^seed_j)
  where G_success^seed_k is the success-group mean gradient of seed k at its
  final checkpoint.

  If cross-seed cosine similarity >> within-seed success/failure opposition score,
  the analysis identifies algorithm-level behaviour, not seed-specific variation.

  Bootstrap CI is computed over the 10 pairs; N is reported explicitly so
  readers can calibrate interval width against sample size.

Paper reporting:
  Report mean ± std of pairwise cosines (N=10 pairs, 95% bootstrap CI) alongside
  within-seed success/failure opposition scores from the analysis JSONs.

Usage (from PPOSAC-tracker/):
    python experiments/compare_ppo_seeds.py

    # Custom roots:
    python experiments/compare_ppo_seeds.py \\
        --experiment_root ppo_experiment_root \\
        --analysis_root ppo_experiment_root \\
        --seeds 1 2 3 4 5 \\
        --n_episodes 300 \\
        --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Add PPOSAC-tracker root to path (this script lives in experiments/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations



def _find_final_ppo_checkpoint(seed_dir: str) -> str | None:
    """Return path to the highest-step periodic PPO checkpoint (without .zip extension)."""
    pattern = os.path.join(seed_dir, "checkpoints", "ppo", "periodic", "periodic_step*.pt")
    candidates = sorted(glob.glob(pattern))
    # Filter to those with a corresponding .meta.json (i.e. not .pt.meta.json itself)
    ckpts = [p for p in candidates if not p.endswith(".meta.json")]
    return ckpts[-1] if ckpts else None



def _cosine_flat(g_a: dict, g_b: dict) -> float:
    """Cosine similarity between two gradient dicts (flattened, common keys only)."""
    common = sorted(set(g_a) & set(g_b))
    if not common:
        raise ValueError("No common parameter keys between gradient dicts.")
    a = torch.cat([g_a[k].float().flatten() for k in common])
    b = torch.cat([g_b[k].float().flatten() for k in common])
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()



def _bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), \
           float(np.percentile(boot_means, 100 * (1 - alpha)))



def _load_within_seed_opposition(analysis_root: str, seeds: list[int]) -> dict[int, float | None]:
    """Load opposition_score from final checkpoint JSON for each seed."""
    result = {}
    for seed in seeds:
        pattern = os.path.join(
            analysis_root, f"seed_{seed}", "analysis_logs", "ppo",
            "periodic_step*.json"
        )
        paths = sorted(glob.glob(pattern))
        if not paths:
            result[seed] = None
            continue
        # Use the last (highest-step) periodic JSON
        try:
            with open(paths[-1]) as f:
                d = json.load(f)
            opp = d.get("opposition_score")
            result[seed] = float(opp) if opp is not None else None
        except Exception:
            result[seed] = None
    return result



def _compute_success_gradient(
    ckpt_path: str,
    n_episodes: int,
    percentile_x: int,
    device: str,
    seed: int,
) -> dict | None:
    """Load PPO checkpoint, collect episodes, return success-group mean gradient dict.

    Returns None if the success group is empty.
    """
    from ppo.sampling import load_ppo_agent, evaluate_frozen_policy, partition
    from ppo.gradients import compute_group_gradient_with_coherence

    model, _ = load_ppo_agent(ckpt_path, device=device)

    episodes, eps_scores, _ = evaluate_frozen_policy(
        ckpt_path, n_episodes=n_episodes, device=device, seed=seed
    )

    success_eps, failure_eps, _ = partition(
        episodes, eps_scores, mode="eps", percentile_x=percentile_x
    )

    if not success_eps:
        print(f"  [seed {seed}] No success episodes — skipping.")
        return None

    batch_size = max(5, len(success_eps) // 10)
    _, raw_mean, _ = compute_group_gradient_with_coherence(
        model, success_eps, batch_size=batch_size, device=device,
        desc=f"G_success seed {seed}"
    )

    return {name: p.cpu().float() for name, p in raw_mean.items()}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default="ppo_experiment_root",
                        help="Root of the PPO experiment (contains seed_N/ dirs)")
    parser.add_argument("--analysis_root", default="ppo_experiment_root",
                        help="Root where PPO analysis JSONs live (default: same as "
                             "--experiment_root; seed_N/analysis_logs/ppo/)")
    parser.add_argument("--seeds",       nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n_episodes",  type=int, default=300)
    parser.add_argument("--percentile_x", type=int, default=25)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--output_json", default="ppo_cross_seed_similarity.json")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_root    = (args.experiment_root if os.path.isabs(args.experiment_root)
                   else os.path.join(script_dir, args.experiment_root))
    analysis_root = (args.analysis_root if os.path.isabs(args.analysis_root)
                     else os.path.join(script_dir, args.analysis_root))


    print(f"\n{'='*60}")
    print("PPO Cross-Seed Gradient Similarity (Issue #5 null-model)")
    print(f"{'='*60}\n")

    seed_grads: dict[int, dict | None] = {}
    for seed in args.seeds:
        seed_dir = os.path.join(exp_root, f"seed_{seed}")
        ckpt     = _find_final_ppo_checkpoint(seed_dir)
        if ckpt is None:
            print(f"[seed {seed}] No final checkpoint found under {seed_dir} — skipping.")
            seed_grads[seed] = None
            continue
        print(f"[seed {seed}] Loading checkpoint: {os.path.basename(ckpt)}")
        try:
            seed_grads[seed] = _compute_success_gradient(
                ckpt_path=ckpt,
                n_episodes=args.n_episodes,
                percentile_x=args.percentile_x,
                device=args.device,
                seed=seed,
            )
        except Exception as e:
            import traceback
            print(f"[seed {seed}] ERROR: {e}")
            traceback.print_exc()
            seed_grads[seed] = None


    valid_seeds = [s for s in args.seeds if seed_grads.get(s) is not None]
    if len(valid_seeds) < 2:
        print("[ERROR] Need at least 2 valid seeds to compute pairwise cosines.")
        sys.exit(1)

    pairs   = list(combinations(valid_seeds, 2))
    cosines = []
    pair_results = []
    print(f"\nComputing pairwise cosines for {len(pairs)} pairs ...")
    for (i, j) in pairs:
        c = _cosine_flat(seed_grads[i], seed_grads[j])
        cosines.append(c)
        pair_results.append({"seed_i": i, "seed_j": j, "cosine": c})
        print(f"  cos(G_success^{i}, G_success^{j}) = {c:.4f}")

    arr = np.array(cosines)
    rng = np.random.default_rng(42)
    ci_lo, ci_hi = _bootstrap_ci(cosines, rng=rng)


    within_opp = _load_within_seed_opposition(analysis_root, args.seeds)


    print(f"\n{'='*60}")
    print("  CROSS-SEED SIMILARITY — SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Cross-seed cos(G_success^i, G_success^j):  N={len(pairs)} pairs")
    print(f"    mean = {arr.mean():.4f}")
    print(f"    std  = {arr.std():.4f}")
    print(f"    min  = {arr.min():.4f}")
    print(f"    max  = {arr.max():.4f}")
    print(f"    95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"\n  Within-seed success/failure opposition scores (final checkpoint):")
    for seed in args.seeds:
        opp = within_opp.get(seed)
        opp_s = f"{opp:.4f}" if opp is not None else "N/A"
        print(f"    seed {seed}: {opp_s}")

    print(f"\n  Interpretation:")
    print(f"    If cross-seed cosine mean >> within-seed opposition mean,")
    print(f"    the success-group gradient identifies algorithm-level behaviour,")
    print(f"    not seed-specific variation — the null model is implicit.")
    print(f"    If cross-seed cosine ≈ within-seed opposition,")
    print(f"    within-seed variation may be comparable to cross-seed similarity.")

    print(f"\n{'='*60}\n")


    out_path = (args.output_json if os.path.isabs(args.output_json)
                else os.path.join(script_dir, args.output_json))
    result = {
        "n_pairs": len(pairs),
        "seeds_used": valid_seeds,
        "pairs": pair_results,
        "cross_seed_cosine_mean": float(arr.mean()),
        "cross_seed_cosine_std":  float(arr.std()),
        "cross_seed_cosine_ci_lo": ci_lo,
        "cross_seed_cosine_ci_hi": ci_hi,
        "within_seed_opposition": {str(s): v for s, v in within_opp.items()},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
