"""Dissertation data pipeline — all analysis phases in dependency order.

Phases
------
  B  Corrected Rainbow analysis on checkpoint_live.pt (all seeds)       [critical]
  C  Scalar DQN ablation — compares MORA ratio, gates credit-assignment [critical]
  D  Fixed-threshold partition analysis — eliminates boundary confound  [important]
  E  Counterfactual gradient validation — fresh training required       [important, SLOW]
  F  Robustness CI statistics report                                     [important]
  G  PPO cross-seed gradient similarity                                  [appendix]
  H  Frozen-RSA re-analysis with locked stimulus set                    [appendix]
  I  EPS weight sensitivity analysis (eps_weight in {0.5, 0.9, 1.2})   [appendix]

Dependencies
------------
  Phase B must complete before C, F, H (they read its output JSONs).
  D, E, G, I are fully independent.

Phase E is excluded by default because it requires training a fresh Rainbow model
with gradient capture enabled (~same wall-clock as the original training run).
Pass --include_e to add it.

Output directories (all under PPOSAC-tracker/)
----------------------------------------------
  corrected_analysis_results/   Phase B
  scalar_ablation_results/      Phase C
  fixed_threshold_results/      Phase D
  grad_validation_results/      Phase E
  ppo_cross_seed_similarity.json Phase G
  frozen_rsa_results/           Phase H
  eps_sensitivity_results/      Phase I
  Report printed to stdout      Phase F

Usage (from PPOSAC-tracker/)
-----------------------------
  python run_pipeline.py                               # B C D F G H I
  python run_pipeline.py --device cuda
  python run_pipeline.py --include_e                  # also run Phase E
  python run_pipeline.py --skip D G H I               # critical path only
  python run_pipeline.py --only B F                   # just two phases
  python run_pipeline.py --only B --seeds 1 2         # subset of seeds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))



def _header(phase: str, title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  PHASE {phase}: {title}")
    print(f"{bar}")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def _warn_missing_b(output_root: str, seeds: list[int]):
    missing = [
        s for s in seeds
        if not os.path.exists(os.path.join(
            output_root, f"seed_{s}",
            "analysis_logs", "rainbow", "checkpoint_live.json",
        ))
    ]
    if missing:
        print(f"  [warn] Phase B output missing for seeds {missing}.")
        print(f"         Run Phase B first for complete results.")



def phase_b(rainbow_root: str, output_root: str, seeds: list[int],
            n_episodes: int, device: str):
    """Run corrected analysis on checkpoint_live.pt for every seed."""
    from analyze_checkpoint import analyze_checkpoint

    completed, skipped = [], []
    for seed in seeds:
        ckpt = os.path.join(
            rainbow_root, f"seed_{seed}",
            "checkpoints", "rainbow", "checkpoint_live.pt",
        )
        out_dir  = os.path.join(output_root, f"seed_{seed}")
        out_json = os.path.join(out_dir, "analysis_logs", "rainbow", "checkpoint_live.json")

        if not os.path.exists(ckpt):
            print(f"  [seed {seed}] checkpoint not found — skipped ({ckpt})")
            skipped.append(seed)
            continue

        if os.path.exists(out_json):
            print(f"  [seed {seed}] already done — {out_json}")
            completed.append(seed)
            continue

        print(f"\n  --- seed {seed} ---")
        try:
            analyze_checkpoint(
                algorithm="rainbow",
                checkpoint_path=ckpt,
                experiment_root=out_dir,
                n_episodes=n_episodes,
                device=device,
                reason="corrected_indicator_weighted_mor",
                seed=seed,
            )
            completed.append(seed)
        except Exception:
            traceback.print_exc()
            skipped.append(seed)

    print(f"\n  Completed: {completed}   Skipped: {skipped}")
    print(f"  Output:    {output_root}/seed_N/analysis_logs/rainbow/checkpoint_live.json")



def phase_c(output_root: str, seeds: list[int], n_episodes: int,
            device: str, T_max: int):
    """Train ScalarDQN (seed 1) and compare its MORA ratio to Rainbow."""
    from run_scalar_ablation import (
        _build_rainbow_args, _load_rainbow_mora_ratio, _print_verdict,
    )
    from rainbow.scalar_dqn import train_scalar_dqn, compute_scalar_mora_ratio

    seed        = seeds[0] if seeds else 1
    scalar_root = _abs("scalar_ablation_results")
    seed_root   = os.path.join(scalar_root, f"seed_{seed}")
    os.makedirs(seed_root, exist_ok=True)

    ckpt_path = os.path.join(seed_root, "checkpoints", "scalar_dqn", "checkpoint_live.pt")
    if os.path.exists(ckpt_path):
        print(f"  [info] Scalar checkpoint exists — skipping training")
        print(f"         {ckpt_path}")
    else:
        print(f"  Training ScalarDQN seed={seed}, T_max={T_max:,} ...")
        rb_args   = _build_rainbow_args(
            experiment_root=seed_root,
            T_max=T_max,
            hidden_size=512,
            architecture="canonical",
            memory_capacity=500_000,
            device=device,
        )
        ckpt_path = train_scalar_dqn(rb_args, seed=seed, out_root=seed_root)

    print(f"\n  Computing MORA ratio ...")
    scalar_result = compute_scalar_mora_ratio(
        ckpt_path, n_episodes=n_episodes, device=device, seed=seed,
    )

    rainbow_json = os.path.join(
        output_root, f"seed_{seed}",
        "analysis_logs", "rainbow", "checkpoint_live.json",
    )
    rainbow_ref = _load_rainbow_mora_ratio(rainbow_json)
    if rainbow_ref is None:
        print(f"  [warn] Rainbow reference not found at {rainbow_json}")
        print(f"         Phase B must run first for a side-by-side comparison.")

    out_path = os.path.join(scalar_root, "scalar_ablation_result.json")
    with open(out_path, "w") as f:
        json.dump(
            {"scalar_dqn": scalar_result, "rainbow_reference": rainbow_ref},
            f, indent=2,
            default=lambda x: None if (isinstance(x, float) and x != x) else str(x),
        )
    print(f"  Saved: {out_path}")
    _print_verdict(scalar_result, rainbow_ref)



def phase_d(rainbow_root: str, seeds: list[int], n_episodes: int,
            device: str, percentile_x: int = 25):
    """Re-run longitudinal analysis using EPS thresholds fixed from all seeds' final checkpoints."""
    from run_fixed_threshold_analysis import (
        _derive_thresholds_pooled, _run_seed, _print_comparison,
    )

    out_root = _abs("fixed_threshold_results")
    os.makedirs(out_root, exist_ok=True)

    print(f"  Deriving pooled EPS thresholds from checkpoint_live.pt across all seeds ...")
    try:
        fixed_thresholds = _derive_thresholds_pooled(
            exp_root=rainbow_root,
            seeds=seeds,
            n_episodes=n_episodes,
            percentile_x=percentile_x,
            device=device,
        )
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return

    thresh_path = os.path.join(out_root, "fixed_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump({"lower": fixed_thresholds[0], "upper": fixed_thresholds[1]}, f)

    for seed in seeds:
        seed_dir = os.path.join(rainbow_root, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"  [skip] seed_{seed} directory not found")
            continue
        print(f"\n  --- seed {seed} ---")
        _run_seed(
            seed=seed,
            exp_root=rainbow_root,
            out_root=out_root,
            fixed_thresholds=fixed_thresholds,
            n_episodes=n_episodes,
            percentile_x=percentile_x,
            device=device,
        )

    _print_comparison(seeds, rainbow_root, out_root)
    print(f"\n  Results in: {out_root}")



def phase_e(seeds: list[int], n_episodes: int, device: str, T_max: int):
    """Train one Rainbow model with gradient capture, then validate offline gradients."""
    from run_gradient_validation import _validate_seed, _print_summary
    from rainbow.train import build_parser, main_rainbow

    exp_root = _abs("grad_validation_root")
    out_root = _abs("grad_validation_results")

    # One seed is sufficient for validation
    seed      = seeds[0] if seeds else 1
    seed_root = os.path.join(exp_root, f"seed_{seed}")
    live_ckpt = os.path.join(seed_root, "checkpoints", "rainbow", "checkpoint_live.pt")

    if os.path.exists(live_ckpt):
        print(f"  [seed {seed}] Training checkpoint exists — skipping training")
    else:
        print(f"  Training Rainbow with gradient capture: seed={seed}, T_max={T_max:,}")
        parser_rb = build_parser()
        rb_args   = parser_rb.parse_args([])
        rb_args.seed                = seed
        rb_args.T_max               = T_max
        rb_args.checkpoint_interval = 500_000   # 6 checkpoints over 3M steps
        rb_args.experiment_root     = seed_root
        rb_args.disable_cuda        = (device == "cpu")
        rb_args.model               = None
        main_rainbow(rb_args, on_checkpoint_saved=None, capture_training_grads=True)

    rows = _validate_seed(
        seed=seed,
        exp_root=exp_root,
        out_root=out_root,
        n_episodes=n_episodes,
        percentile_x=25,
        device=device,
    )
    _print_summary({seed: rows})
    print(f"\n  Results in: {out_root}")



def phase_f(output_root: str, seeds: list[int]):
    """Print robustness statistics with 95% bootstrap CIs from Phase B output."""
    from report_robustness_stats import report
    report(output_root, seeds)



def phase_g(ppo_root: str, seeds: list[int], n_episodes: int, device: str):
    """Compute cos(G_success^i, G_success^j) for all C(5,2)=10 PPO seed pairs."""
    from compare_ppo_seeds import (
        _find_final_ppo_checkpoint, _compute_success_gradient,
        _cosine_flat, _bootstrap_ci, _load_within_seed_opposition,
    )
    from itertools import combinations
    import numpy as np

    seed_grads: dict[int, dict | None] = {}
    for seed in seeds:
        seed_dir = os.path.join(ppo_root, f"seed_{seed}")
        ckpt     = _find_final_ppo_checkpoint(seed_dir)
        if ckpt is None:
            print(f"  [seed {seed}] No checkpoint found — skipping")
            seed_grads[seed] = None
            continue
        print(f"  [seed {seed}] {os.path.basename(ckpt)}")
        try:
            seed_grads[seed] = _compute_success_gradient(
                ckpt_path=ckpt, n_episodes=n_episodes,
                percentile_x=25, device=device, seed=seed,
            )
        except Exception:
            traceback.print_exc()
            seed_grads[seed] = None

    valid = [s for s in seeds if seed_grads.get(s) is not None]
    if len(valid) < 2:
        print("  [ERROR] Need ≥2 valid seeds for pairwise cosines — aborting Phase G")
        return

    pairs   = list(combinations(valid, 2))
    cosines = []
    for (i, j) in pairs:
        c = _cosine_flat(seed_grads[i], seed_grads[j])
        cosines.append(c)
        print(f"  cos(G_success^{i}, G_success^{j}) = {c:.4f}")

    arr            = np.array(cosines)
    rng            = np.random.default_rng(42)
    ci_lo, ci_hi   = _bootstrap_ci(cosines, rng=rng)
    within_opp     = _load_within_seed_opposition(ppo_root, seeds)

    print(f"\n  Cross-seed cos(G_success^i, G_success^j):  N={len(pairs)} pairs")
    print(f"    mean = {arr.mean():.4f}   std = {arr.std():.4f}")
    print(f"    95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\n  Within-seed opposition scores (final checkpoint):")
    for seed in seeds:
        opp = within_opp.get(seed)
        print(f"    seed {seed}: {opp:.4f}" if opp is not None else f"    seed {seed}: N/A")

    out_path = _abs("ppo_cross_seed_similarity.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_pairs":                  len(pairs),
            "seeds_used":               valid,
            "cross_seed_cosine_mean":   float(arr.mean()),
            "cross_seed_cosine_std":    float(arr.std()),
            "cross_seed_cosine_ci_lo":  ci_lo,
            "cross_seed_cosine_ci_hi":  ci_hi,
            "within_seed_opposition":   {str(s): v for s, v in within_opp.items()},
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")



def phase_h(rainbow_root: str, output_root: str, seeds: list[int],
            n_episodes: int, device: str):
    """Re-run analysis with RSA stimulus set fixed to union of Phase B observed labels."""
    from analyze_checkpoint import analyze_checkpoint

    # Build frozen stimulus set from analysis JSONs (union across all seeds + all checkpoints).
    # Globs checkpoint_*.json to pick up both checkpoint_live.json (Phase B) and
    # checkpoint_step*.json (inline runs from run_rainbow_full_pipeline.py).
    all_labels: set[str] = set()
    for seed in seeds:
        json_dir = os.path.join(
            output_root, f"seed_{seed}", "analysis_logs", "rainbow",
        )
        seed_jsons = glob.glob(os.path.join(json_dir, "checkpoint_*.json"))
        if not seed_jsons:
            print(f"  [seed {seed}] no analysis JSONs found in {json_dir} — skipping for label collection")
            continue
        for json_path in seed_jsons:
            try:
                with open(json_path) as f:
                    d = json.load(f)
                all_labels.update(d.get("rsa_labels", []))
            except Exception:
                pass

    if not all_labels:
        print("  [ERROR] No RSA labels found in Phase B output — run Phase B first.")
        return

    reference_stimuli = frozenset(all_labels)
    print(f"  Frozen stimulus set ({len(reference_stimuli)} stimuli):")
    print(f"    {sorted(reference_stimuli)}")

    frozen_root         = _abs("frozen_rsa_results")
    completed, skipped  = [], []
    for seed in seeds:
        ckpt = os.path.join(
            rainbow_root, f"seed_{seed}",
            "checkpoints", "rainbow", "checkpoint_live.pt",
        )
        out_dir  = os.path.join(frozen_root, f"seed_{seed}")
        out_json = os.path.join(out_dir, "analysis_logs", "rainbow", "checkpoint_live.json")

        if not os.path.exists(ckpt):
            print(f"  [seed {seed}] checkpoint not found — skipped")
            skipped.append(seed)
            continue

        if os.path.exists(out_json):
            print(f"  [seed {seed}] already done")
            completed.append(seed)
            continue

        print(f"\n  --- seed {seed} ---")
        try:
            analyze_checkpoint(
                algorithm="rainbow",
                checkpoint_path=ckpt,
                experiment_root=out_dir,
                n_episodes=n_episodes,
                device=device,
                reason="frozen_rsa",
                seed=seed,
                reference_stimuli=reference_stimuli,
            )
            completed.append(seed)
        except Exception:
            traceback.print_exc()
            skipped.append(seed)

    print(f"\n  Completed: {completed}   Skipped: {skipped}")
    print(f"  Results in: {frozen_root}")



def phase_i(rainbow_root: str, seeds: list[int], n_episodes: int, device: str):
    """Re-run analysis with eps_weight in {0.5, 1.2} (0.9 is Phase B baseline)."""
    from analyze_checkpoint import analyze_checkpoint

    eps_root = _abs("eps_sensitivity_results")
    for eps_weight in (0.5, 1.2):
        w_tag = str(eps_weight).replace(".", "p")   # "0p5" or "1p2"
        print(f"\n  --- eps_weight = {eps_weight} ---")
        completed, skipped = [], []
        for seed in seeds:
            ckpt = os.path.join(
                rainbow_root, f"seed_{seed}",
                "checkpoints", "rainbow", "checkpoint_live.pt",
            )
            out_dir  = os.path.join(eps_root, f"w{w_tag}", f"seed_{seed}")
            out_json = os.path.join(out_dir, "analysis_logs", "rainbow", "checkpoint_live.json")

            if not os.path.exists(ckpt):
                print(f"  [seed {seed}] checkpoint not found — skipped")
                skipped.append(seed)
                continue

            if os.path.exists(out_json):
                print(f"  [seed {seed}] already done")
                completed.append(seed)
                continue

            print(f"  seed {seed} ...")
            try:
                analyze_checkpoint(
                    algorithm="rainbow",
                    checkpoint_path=ckpt,
                    experiment_root=out_dir,
                    n_episodes=n_episodes,
                    device=device,
                    reason=f"eps_sensitivity_w{w_tag}",
                    seed=seed,
                    eps_weight=eps_weight,
                )
                completed.append(seed)
            except Exception:
                traceback.print_exc()
                skipped.append(seed)

        print(f"  eps_weight={eps_weight}: done={completed}  skipped={skipped}")

    print(f"\n  Results in: {eps_root}")
    print(f"  Baseline (eps_weight=0.9) is Phase B output in corrected_analysis_results/")



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rainbow_root", default="rainbow_experiment_root",
        help="Root of Rainbow experiment containing seed_N/checkpoints/ (default: rainbow_experiment_root)",
    )
    parser.add_argument(
        "--ppo_root", default="ppo_experiment_root",
        help="Root of PPO experiment containing seed_N/checkpoints/ (default: ppo_experiment_root)",
    )
    parser.add_argument(
        "--output_root", default="corrected_analysis_results",
        help="Phase B output (also used as input by C, F, H) (default: corrected_analysis_results)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Seeds to process (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000,
        help="Episodes per analysis checkpoint (default: 1000)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--scalar_T_max", type=int, default=3_000_000,
        help="Training steps for Scalar DQN ablation (default: 3M)",
    )
    parser.add_argument(
        "--grad_T_max", type=int, default=3_000_000,
        help="Training steps for gradient validation run (default: 3M)",
    )
    parser.add_argument(
        "--skip", nargs="*", metavar="PHASE", default=[],
        help="Phases to skip, e.g. --skip D G H I",
    )
    parser.add_argument(
        "--only", nargs="*", metavar="PHASE", default=None,
        help="Run only these phases, e.g. --only B F",
    )
    parser.add_argument(
        "--include_e", action="store_true",
        help="Include Phase E (counterfactual gradient validation). "
             "Requires training a fresh Rainbow model — very slow.",
    )
    args = parser.parse_args()

    rainbow_root = _abs(args.rainbow_root)
    ppo_root     = _abs(args.ppo_root)
    output_root  = _abs(args.output_root)

    # Determine which phases to run
    all_phases      = ["B", "C", "D", "E", "F", "G", "H", "I"]
    user_skip       = {p.upper() for p in args.skip}
    default_skip    = {"E"} if not args.include_e else set()
    skip            = user_skip | default_skip

    if args.only is not None:
        run_phases = {p.upper() for p in args.only}
    else:
        run_phases = set(all_phases) - skip

    def _run(phase: str) -> bool:
        return phase in run_phases

    print(f"\nPipeline configuration")
    print(f"  Rainbow root : {rainbow_root}")
    print(f"  PPO root     : {ppo_root}")
    print(f"  Output root  : {output_root}")
    print(f"  Seeds        : {args.seeds}")
    print(f"  n_episodes   : {args.n_episodes}")
    print(f"  Device       : {args.device}")
    print(f"  Phases to run: {sorted(run_phases)}")
    if "E" in skip and "E" not in user_skip:
        print(f"  (Phase E excluded by default — pass --include_e to enable)")


    if _run("B"):
        _header("B", "Corrected Rainbow Analysis  [checkpoint_live.pt × all seeds]")
        phase_b(rainbow_root, output_root, args.seeds, args.n_episodes, args.device)

    if _run("C"):
        _header("C", "Scalar DQN Ablation  [MORA ratio gate test]")
        _warn_missing_b(output_root, args.seeds[:1])
        phase_c(output_root, args.seeds, args.n_episodes, args.device, args.scalar_T_max)

    if _run("D"):
        _header("D", "Fixed-Threshold Partition Analysis")
        phase_d(rainbow_root, args.seeds, args.n_episodes, args.device)

    if _run("E"):
        _header("E", "Counterfactual Gradient Validation  [VERY SLOW — fresh training]")
        phase_e(args.seeds, args.n_episodes, args.device, args.grad_T_max)

    if _run("F"):
        _header("F", "Robustness CI Report")
        _warn_missing_b(output_root, args.seeds)
        phase_f(output_root, args.seeds)

    if _run("G"):
        _header("G", "PPO Cross-Seed Gradient Similarity  [appendix, N=10 pairs]")
        phase_g(ppo_root, args.seeds, args.n_episodes, args.device)

    if _run("H"):
        _header("H", "Frozen RSA Re-Analysis  [appendix]")
        _warn_missing_b(output_root, args.seeds)
        phase_h(rainbow_root, output_root, args.seeds, args.n_episodes, args.device)

    if _run("I"):
        _header("I", "EPS Weight Sensitivity  [appendix, eps_weight ∈ {0.5, 0.9, 1.2}]")
        phase_i(rainbow_root, args.seeds, args.n_episodes, args.device)

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Pipeline done. Phases run: {sorted(run_phases)}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
