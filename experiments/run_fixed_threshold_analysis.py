"""Fixed-threshold partition analysis — Experiment 2.

Re-runs the RQ1 longitudinal Rainbow analysis using thresholds fixed from the
final trained checkpoint, rather than percentile thresholds re-computed at each
checkpoint.

Interpretation:
  failure-coherence-collapse persists  → finding is about the learning signal
  failure-coherence-collapse disappears → finding is about partition boundaries

Workflow:
  1. Determine reference thresholds (lower_return, upper_return) from the
     final checkpoint's episode return distribution (or supplied explicitly).
  2. Re-run analyze_checkpoint() on every periodic checkpoint in
     rainbow_experiment_root/seed_N/checkpoints/rainbow/ using these fixed
     thresholds.
  3. Save results to fixed_threshold_results/seed_N/analysis_logs/rainbow/.
  4. Print a comparison table: coherence_failure over time (original vs fixed).

Usage (from PPOSAC-tracker/):
    # Auto-derive thresholds from final checkpoint, re-run all seeds:
    python run_fixed_threshold_analysis.py \\
        --experiment_root rainbow_experiment_root

    # Explicit thresholds (raw return values):
    python run_fixed_threshold_analysis.py \\
        --lower_threshold 3.0 --upper_threshold 9.0

    # Single seed, explicit thresholds:
    python run_fixed_threshold_analysis.py \\
        --seeds 1 --lower_threshold 3.0 --upper_threshold 9.0

    # Plot-only (after re-analysis is complete):
    python run_fixed_threshold_analysis.py --plot_only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np



def _derive_thresholds_from_checkpoint(
    ckpt_path: str,
    n_episodes: int,
    percentile_x: int,
    device: str,
    seed: int,
) -> tuple[float, float]:
    """Evaluate the final checkpoint and return (lower, upper) return percentiles.

    lower = percentile_x-th percentile of episode returns
    upper = (100 - percentile_x)-th percentile of episode returns
    """
    from rainbow.sampling import evaluate_frozen_policy
    print(f"  Deriving thresholds from {os.path.basename(ckpt_path)} ...")
    _, eps_scores, _ = evaluate_frozen_policy(
        ckpt_path, n_episodes=n_episodes, device=device, seed=seed
    )
    scores_arr = np.array(eps_scores)
    lower = float(np.percentile(scores_arr, percentile_x))
    upper = float(np.percentile(scores_arr, 100 - percentile_x))
    print(f"  Fixed thresholds (EPS): lower={lower:.4f}  upper={upper:.4f}")
    return lower, upper


def _derive_thresholds_pooled(
    exp_root: str,
    seeds: list[int],
    n_episodes: int,
    percentile_x: int,
    device: str,
) -> tuple[float, float]:
    """Derive fixed thresholds by pooling episode scores from all seeds' final checkpoints.

    Evaluates each seed's final checkpoint, pools all episode EPS scores, then
    takes the percentile_x-th and (100 - percentile_x)-th percentiles of the
    pooled distribution. This eliminates the seed-anchor degree of freedom
    present in the single-seed derivation.
    """
    from rainbow.sampling import evaluate_frozen_policy

    all_eps_scores = []
    for seed in seeds:
        seed_dir  = os.path.join(exp_root, f"seed_{seed}")
        final_ckpt = _find_final_checkpoint(seed_dir)
        if final_ckpt is None:
            print(f"  [pool] seed_{seed}: no checkpoint found — skipping")
            continue
        print(f"  [pool] seed_{seed}: evaluating {os.path.basename(final_ckpt)} ...")
        _, eps_scores, _ = evaluate_frozen_policy(
            final_ckpt, n_episodes=n_episodes, device=device, seed=seed
        )
        all_eps_scores.extend(eps_scores)

    if not all_eps_scores:
        raise RuntimeError("No episode scores collected — check experiment_root and seeds.")

    pooled = np.array(all_eps_scores)
    lower  = float(np.percentile(pooled, percentile_x))
    upper  = float(np.percentile(pooled, 100 - percentile_x))
    print(f"  Pooled thresholds ({len(all_eps_scores)} episodes across {len(seeds)} seeds): "
          f"lower={lower:.4f}  upper={upper:.4f}")
    return lower, upper


def _find_final_checkpoint(seed_dir: str) -> str | None:
    """Return checkpoint_live.pt if it exists, else the highest-step checkpoint."""
    live = os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_live.pt")
    if os.path.exists(live):
        return live
    candidates = sorted(
        glob.glob(os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_step*.pt"))
    )
    return candidates[-1] if candidates else None


def _find_periodic_checkpoints(seed_dir: str) -> list[str]:
    """Return all checkpoint_step*.pt files in ascending step order."""
    pattern = os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_step*.pt")
    paths = sorted(glob.glob(pattern))
    return paths


def _find_archive_checkpoints(seed_dir: str) -> list[str]:
    """Return all checkpoint_archive_step*.pt files in ascending step order.

    These are the permanent archive checkpoints saved every ARCHIVE_INTERVAL steps
    by run_rainbow_full_pipeline.py and are never deleted by the training pointer rotation.
    """
    pattern = os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_archive_step*.pt")
    paths = sorted(glob.glob(pattern))
    return paths



def _run_seed(
    seed: int,
    exp_root: str,
    out_root: str,
    fixed_thresholds: tuple[float, float],
    n_episodes: int,
    percentile_x: int,
    device: str,
) -> list[dict]:
    """Re-run analysis on all periodic checkpoints for one seed.

    Returns list of result dicts (one per checkpoint).
    """
    from analyze_checkpoint import analyze_checkpoint

    seed_dir = os.path.join(exp_root,  f"seed_{seed}")
    out_dir  = os.path.join(out_root,  f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    ckpts = _find_periodic_checkpoints(seed_dir)
    if not ckpts:
        print(f"  [seed {seed}] No periodic checkpoints found under {seed_dir}")
        return []

    print(f"  [seed {seed}] {len(ckpts)} periodic checkpoints, "
          f"thresholds=({fixed_thresholds[0]:.3f}, {fixed_thresholds[1]:.3f})")

    results = []
    for ckpt in ckpts:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        out_json = os.path.join(out_dir, "analysis_logs", "rainbow", f"{name}.json")

        if os.path.exists(out_json):
            print(f"    [skip] {name} already exists")
            with open(out_json) as f:
                results.append(json.load(f))
            continue

        print(f"    Analysing {name} ...")
        try:
            analyze_checkpoint(
                algorithm="rainbow",
                checkpoint_path=ckpt,
                experiment_root=out_dir,
                n_episodes=n_episodes,
                device=device,
                reason="fixed_threshold_rq1",
                split_mode="fixed",
                percentile_x=percentile_x,
                seed=seed,
                fixed_thresholds=fixed_thresholds,
            )
            if os.path.exists(out_json):
                with open(out_json) as f:
                    results.append(json.load(f))
        except Exception as e:
            import traceback
            print(f"    [ERROR] {name}: {e}")
            traceback.print_exc()

    return results



def _print_comparison(
    seeds: list[int],
    exp_root: str,
    out_root: str,
):
    """Print coherence_failure over training steps: original vs fixed-threshold."""
    print("\n" + "=" * 90)
    print("  FIXED-THRESHOLD vs ORIGINAL — coherence_failure over training steps")
    print("  Finding persists → learning signal. Disappears → partition boundary.")
    print("=" * 90)

    for seed in seeds:
        orig_dir  = os.path.join(exp_root, f"seed_{seed}", "analysis_logs", "rainbow")
        fixed_dir = os.path.join(out_root,  f"seed_{seed}", "analysis_logs", "rainbow")

        orig_jsons  = sorted(glob.glob(os.path.join(orig_dir,  "checkpoint_step*.json")))
        fixed_jsons = sorted(glob.glob(os.path.join(fixed_dir, "checkpoint_step*.json")))

        if not orig_jsons or not fixed_jsons:
            print(f"\n  [seed {seed}] Missing data — skipping comparison")
            continue

        def _load_metric(paths, key):
            out = {}
            for p in paths:
                try:
                    with open(p) as f:
                        d = json.load(f)
                    step = d.get("episode") or _step_from_name(os.path.basename(p))
                    v = d.get(key)
                    if v is not None:
                        out[step] = float(v)
                except Exception:
                    pass
            return out

        orig_coh  = _load_metric(orig_jsons,  "coherence_failure")
        fixed_coh = _load_metric(fixed_jsons, "coherence_failure")

        all_steps = sorted(set(orig_coh) | set(fixed_coh))
        print(f"\n  Seed {seed}:")
        print(f"  {'Step':>10}  {'orig coh_fail':>14}  {'fixed coh_fail':>14}  {'Δ':>8}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*8}")
        for step in all_steps:
            o = orig_coh.get(step)
            fx = fixed_coh.get(step)
            delta = (fx - o) if (o is not None and fx is not None) else None
            o_s  = f"{o:.4f}"  if o  is not None else "   N/A"
            fx_s = f"{fx:.4f}" if fx is not None else "   N/A"
            d_s  = f"{delta:+.4f}" if delta is not None else "   N/A"
            print(f"  {step:>10}  {o_s:>14}  {fx_s:>14}  {d_s:>8}")

    print("\n" + "=" * 90 + "\n")


def _step_from_name(name: str) -> int:
    """Extract step from 'checkpoint_step3000000.json'."""
    import re
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else 0



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default="rainbow_experiment_root")
    parser.add_argument("--output_root",     default="fixed_threshold_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n_episodes",   type=int, default=500)
    parser.add_argument("--percentile_x", type=int, default=25,
                        help="Percentile used both for threshold derivation and to define "
                             "the success/failure bands (default: 25)")
    parser.add_argument("--lower_threshold", type=float, default=None,
                        help="Fixed lower return cutoff (episodes ≤ this → failure). "
                             "If omitted, derived from the final checkpoint of seed_1.")
    parser.add_argument("--upper_threshold", type=float, default=None,
                        help="Fixed upper return cutoff (episodes ≥ this → success).")
    parser.add_argument("--threshold_seed", type=int, default=1,
                        help="Which seed's final checkpoint to use for threshold derivation "
                             "(only used when --lower_threshold / --upper_threshold are omitted "
                             "and --pool_all_seeds is not set).")
    parser.add_argument("--pool_all_seeds", action="store_true",
                        help="Derive fixed thresholds by pooling episode EPS scores from all "
                             "seeds' final checkpoints, then taking percentile_x / "
                             "(100 - percentile_x) of the pooled distribution. Eliminates the "
                             "seed-anchor degree of freedom present in single-seed derivation. "
                             "Overrides --threshold_seed.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip re-analysis; just print the comparison table.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    exp_root = _abs(args.experiment_root)
    out_root = _abs(args.output_root)
    os.makedirs(out_root, exist_ok=True)


    if args.lower_threshold is not None and args.upper_threshold is not None:
        fixed_thresholds = (args.lower_threshold, args.upper_threshold)
        print(f"Using explicit thresholds: lower={fixed_thresholds[0]}  upper={fixed_thresholds[1]}")
    elif args.pool_all_seeds:
        print("Deriving thresholds by pooling all seeds' final checkpoints ...")
        fixed_thresholds = _derive_thresholds_pooled(
            exp_root=exp_root,
            seeds=args.seeds,
            n_episodes=args.n_episodes,
            percentile_x=args.percentile_x,
            device=args.device,
        )
    else:
        ref_seed_dir = os.path.join(exp_root, f"seed_{args.threshold_seed}")
        final_ckpt   = _find_final_checkpoint(ref_seed_dir)
        if final_ckpt is None:
            print(f"[ERROR] No checkpoint found under {ref_seed_dir}. "
                  "Provide --lower_threshold and --upper_threshold explicitly, "
                  "or use --pool_all_seeds.")
            sys.exit(1)
        fixed_thresholds = _derive_thresholds_from_checkpoint(
            final_ckpt,
            n_episodes=args.n_episodes,
            percentile_x=args.percentile_x,
            device=args.device,
            seed=args.threshold_seed,
        )

    # Save thresholds so the plot_only path can reload them
    thresh_path = os.path.join(out_root, "fixed_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump({"lower": fixed_thresholds[0], "upper": fixed_thresholds[1]}, f)

    if not args.plot_only:
        for seed in args.seeds:
            seed_dir = os.path.join(exp_root, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                print(f"[skip] seed_{seed} directory not found: {seed_dir}")
                continue
            print(f"\n{'='*60}")
            print(f"=== Seed {seed}")
            print(f"{'='*60}")
            _run_seed(
                seed=seed,
                exp_root=exp_root,
                out_root=out_root,
                fixed_thresholds=fixed_thresholds,
                n_episodes=args.n_episodes,
                percentile_x=args.percentile_x,
                device=args.device,
            )

    _print_comparison(args.seeds, exp_root, out_root)
    print(f"Results in: {out_root}")


if __name__ == "__main__":
    main()
