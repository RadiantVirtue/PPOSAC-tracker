"""Scalar-loss DQN ablation — Experiment 1 gate test.

Trains a non-distributional Rainbow (ScalarDQN: same CNN/dueling/PER/n-step/
target-net, Huber loss instead of C51) and computes the MORA ratio:

    mora_ratio = gradient_magnitude_positive / gradient_magnitude_neutral

Interpretation:
  mora_ratio ≈ 33×  → credit-assignment drives the finding (defensible claim)
  mora_ratio << 33× → C51 distributional loss is the mechanism (reframe required)

Optionally loads an existing Rainbow corrected-analysis JSON to compare ratios
side-by-side. Outputs a JSON result and prints a verdict.

Usage (from PPOSAC-tracker/):
    # Train from scratch + analyse:
    python run_scalar_ablation.py --experiment_root rainbow_experiment_root

    # Skip training (checkpoint already exists):
    python run_scalar_ablation.py --skip_training \\
        --scalar_ckpt path/to/checkpoint_live.pt

    # Compare with existing Rainbow corrected-analysis JSON:
    python run_scalar_ablation.py \\
        --rainbow_json corrected_analysis_results/seed_1/analysis_logs/rainbow/checkpoint_live.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def _build_rainbow_args(
    experiment_root: str,
    T_max: int,
    hidden_size: int,
    architecture: str,
    memory_capacity: int,
    device: str,
) -> argparse.Namespace:
    """Build argparse.Namespace for ScalarDQN training using Rainbow defaults."""
    from rainbow.train import build_parser
    parser = build_parser()
    ns = parser.parse_args([])
    ns.T_max             = T_max
    ns.experiment_root   = experiment_root
    ns.hidden_size       = hidden_size
    ns.architecture      = architecture
    ns.memory_capacity   = memory_capacity
    ns.disable_cuda      = (device == "cpu")
    ns.model             = None
    ns.history_length    = 3
    # Distributional params (unused by ScalarDQN but ReplayMemory needs the namespace)
    ns.atoms             = 51
    ns.V_min             = -10.0
    ns.V_max             = 10.0
    return ns


def _load_rainbow_mora_ratio(json_path: str) -> dict | None:
    """Extract gradient_magnitude_positive/neutral and ratio from a corrected analysis JSON."""
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    mor = d.get("moment_of_reward")
    if not isinstance(mor, dict):
        return None
    gm_pos = mor.get("gradient_magnitude_positive")
    gm_neu = mor.get("gradient_magnitude_neutral")
    ratio  = (gm_pos / gm_neu) if (gm_pos and gm_neu and gm_neu > 0) else None
    return {
        "gradient_magnitude_positive": gm_pos,
        "gradient_magnitude_neutral":  gm_neu,
        "mora_ratio":                  ratio,
        "source_json":                 json_path,
    }


def _print_verdict(scalar_result: dict, rainbow_ref: dict | None):
    print("\n" + "=" * 70)
    print("  SCALAR DQN ABLATION — MORA RATIO VERDICT")
    print("=" * 70)

    r_scalar = scalar_result.get("mora_ratio")
    print(f"\n  Scalar DQN (non-distributional):")
    print(f"    grad_mag_pos  = {scalar_result.get('gradient_magnitude_positive')}")
    print(f"    grad_mag_neu  = {scalar_result.get('gradient_magnitude_neutral')}")
    print(f"    MORA ratio    = {r_scalar}")
    print(f"    n_pos / n_neu = {scalar_result.get('n_pos')} / {scalar_result.get('n_neu')}")

    if rainbow_ref:
        r_rainbow = rainbow_ref.get("mora_ratio")
        print(f"\n  Rainbow C51 (reference, from corrected analysis):")
        print(f"    grad_mag_pos  = {rainbow_ref.get('gradient_magnitude_positive')}")
        print(f"    grad_mag_neu  = {rainbow_ref.get('gradient_magnitude_neutral')}")
        print(f"    MORA ratio    = {r_rainbow}")
        print(f"    source        = {rainbow_ref.get('source_json')}")

        if r_scalar is not None and r_rainbow is not None:
            collapse_pct = (1.0 - r_scalar / r_rainbow) * 100
            print(f"\n  Ratio collapse: {collapse_pct:+.1f}%  (0% = no change, 100% = total collapse)")

            if r_scalar >= 0.5 * r_rainbow:
                verdict = "SURVIVES — ratio ≥ 50% of Rainbow. Credit-assignment interpretation defensible."
            elif r_scalar >= 0.2 * r_rainbow:
                verdict = "PARTIAL COLLAPSE — ratio 20-50% of Rainbow. Distributional loss amplifies but does not create the effect. Caveat required."
            else:
                verdict = "COLLAPSES — ratio < 20% of Rainbow. C51 is driving the finding. REFRAME: distributional-loss artefact."
            print(f"\n  VERDICT: {verdict}")
    else:
        if r_scalar is not None:
            if r_scalar >= 20.0:
                verdict = "RATIO HIGH (≥20×) — credit-assignment interpretation appears defensible even without Rainbow reference."
            elif r_scalar >= 5.0:
                verdict = "RATIO MODERATE (5-20×) — borderline; compare against Rainbow directly."
            else:
                verdict = "RATIO LOW (<5×) — distributional loss likely drives Rainbow finding; reframe."
            print(f"\n  VERDICT (no Rainbow reference): {verdict}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default="rainbow_experiment_root",
                        help="Root of existing Rainbow experiment (for training reference)")
    parser.add_argument("--scalar_root", default="scalar_ablation_results",
                        help="Output directory for scalar DQN training and results")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for training and episode collection")
    parser.add_argument("--T_max", type=int, default=3_000_000,
                        help="Training steps for scalar DQN (default: 3M, same as RQ1 horizon)")
    parser.add_argument("--n_episodes", type=int, default=500,
                        help="Episodes for MORA ratio analysis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--architecture", default="canonical")
    parser.add_argument("--memory_capacity", type=int, default=500_000)
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training; use --scalar_ckpt to point to an existing checkpoint")
    parser.add_argument("--scalar_ckpt", default=None,
                        help="Existing ScalarDQN checkpoint (used when --skip_training)")
    parser.add_argument("--rainbow_json", default=None,
                        help="Path to a corrected Rainbow analysis JSON for comparison "
                             "(moment_of_reward sub-dict). If omitted, only scalar ratio is reported.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    scalar_root = _abs(args.scalar_root)
    seed_root   = os.path.join(scalar_root, f"seed_{args.seed}")
    os.makedirs(seed_root, exist_ok=True)


    if args.skip_training:
        if not args.scalar_ckpt:
            parser.error("--skip_training requires --scalar_ckpt")
        ckpt_path = _abs(args.scalar_ckpt)
        if not os.path.exists(ckpt_path):
            parser.error(f"Checkpoint not found: {ckpt_path}")
        print(f"[skip_training] Using existing checkpoint: {ckpt_path}")
    else:
        ckpt_dir  = os.path.join(seed_root, "checkpoints", "scalar_dqn")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_live.pt")

        if os.path.exists(ckpt_path):
            print(f"[info] Checkpoint already exists: {ckpt_path}")
            print("  Use --skip_training to skip or delete the file to retrain.")
        else:
            from rainbow.scalar_dqn import train_scalar_dqn
            rainbow_args = _build_rainbow_args(
                seed_root, args.T_max, args.hidden_size,
                args.architecture, args.memory_capacity, args.device,
            )
            ckpt_path = train_scalar_dqn(rainbow_args, seed=args.seed, out_root=seed_root)


    from rainbow.scalar_dqn import compute_scalar_mora_ratio
    print(f"\nComputing MORA ratio on {ckpt_path} ...")
    scalar_result = compute_scalar_mora_ratio(
        ckpt_path,
        n_episodes=args.n_episodes,
        device=args.device,
        seed=args.seed,
    )


    rainbow_ref = None
    if args.rainbow_json:
        rainbow_ref = _load_rainbow_mora_ratio(_abs(args.rainbow_json))
        if rainbow_ref is None:
            print(f"[warn] Could not load Rainbow MoR ratio from {args.rainbow_json}")

    # Auto-discover if not provided: look for corrected_analysis_results/seed_N/.../checkpoint_live.json
    if rainbow_ref is None:
        candidate = os.path.join(
            script_dir, "corrected_analysis_results",
            f"seed_{args.seed}", "analysis_logs", "rainbow", "checkpoint_live.json",
        )
        rainbow_ref = _load_rainbow_mora_ratio(candidate)
        if rainbow_ref:
            print(f"  [info] Auto-discovered Rainbow reference: {candidate}")


    out = {
        "scalar_dqn": scalar_result,
        "rainbow_reference": rainbow_ref,
    }
    out_path = os.path.join(seed_root, "scalar_ablation_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if x != x else str(x))
    print(f"\nResults saved: {out_path}")


    _print_verdict(scalar_result, rainbow_ref)


if __name__ == "__main__":
    main()
