"""Re-run Rainbow analysis on checkpoint_live.pt with corrected MoR.

Uses the indicator-weighted gradient fix in rainbow/moment_of_reward.py and
also computes the failure-group MoR sub-partition (moment_of_reward_failure).

Results saved to --output_root/seed_{N}/analysis_logs/rainbow/checkpoint_live.json.

Usage (from PPOSAC-tracker/):
    python run_corrected_analysis.py \\
        --experiment_root rainbow_experiment_root \\
        --output_root corrected_analysis_results

    # Single seed:
    python run_corrected_analysis.py \\
        --experiment_root rainbow_experiment_root \\
        --output_root corrected_analysis_results \\
        --seeds 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_checkpoint import analyze_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_root", required=True,
        help="Root directory of the Rainbow experiment (e.g. rainbow_experiment_root)"
    )
    parser.add_argument(
        "--output_root", required=True,
        help="Output directory for corrected results (e.g. corrected_analysis_results)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Seed indices to process (default: 1 2 3 4 5)"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=500,
        help="Episodes to sample for analysis (default: 500)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu)"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location if not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = (os.path.join(script_dir, args.experiment_root)
                if not os.path.isabs(args.experiment_root) else args.experiment_root)
    out_root = (os.path.join(script_dir, args.output_root)
                if not os.path.isabs(args.output_root) else args.output_root)

    skipped, completed = [], []

    for seed in args.seeds:
        ckpt = os.path.join(exp_root, f"seed_{seed}",
                            "checkpoints", "rainbow", "checkpoint_live.pt")
        out  = os.path.join(out_root, f"seed_{seed}")

        if not os.path.exists(ckpt):
            print(f"[skip] checkpoint not found: {ckpt}")
            skipped.append(seed)
            continue

        print(f"\n{'='*60}")
        print(f"=== Seed {seed} — {ckpt}")
        print(f"{'='*60}")

        try:
            analyze_checkpoint(
                algorithm="rainbow",
                checkpoint_path=ckpt,
                experiment_root=out,
                n_episodes=args.n_episodes,
                device=args.device,
                reason="corrected_indicator_weighted_mor",
                seed=seed,
            )
            completed.append(seed)
        except Exception as e:
            print(f"[ERROR] seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            skipped.append(seed)

    print(f"\n{'='*60}")
    print(f"Done. Completed: {completed}  Skipped: {skipped}")
    if completed:
        print(f"Results in: {out_root}/seed_{{N}}/analysis_logs/rainbow/checkpoint_live.json")


if __name__ == "__main__":
    main()
