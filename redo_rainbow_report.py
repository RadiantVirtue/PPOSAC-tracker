"""Regenerate Rainbow markdown reports from existing analysis JSONs.

Reads all *.json files under <experiment_root>/analysis_logs/rainbow/,
rebuilds the (label, result) list, and re-writes the report markdown.
Use this after updating shared/reporting.py to pick up new sections
(e.g. Gradient Variant Analysis) without re-running expensive analysis.

Usage:
    # Single seed
    python redo_rainbow_report.py --experiment_root rainbow_experiment_root/seed_1

    # All seeds at once
    python redo_rainbow_report.py --experiment_root rainbow_experiment_root --all_seeds
    python redo_rainbow_report.py --experiment_root rainbow_experiment_root --all_seeds --seeds 1 2 3
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.reporting import generate_report, label_from_path
from shared.storage import load_analysis_results


def _redo_seed(seed_root: str, seed: int):
    log_dir = os.path.join(seed_root, "analysis_logs", "rainbow")
    if not os.path.isdir(log_dir):
        print(f"  [SKIP] No analysis_logs/rainbow/ found at {seed_root}")
        return

    json_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    if not json_files:
        print(f"  [SKIP] No JSON files in {log_dir}")
        return

    checkpoint_results = []
    for json_path in json_files:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        label = label_from_path(basename)
        try:
            r = load_analysis_results(json_path)
            checkpoint_results.append((label, r))
        except Exception as e:
            print(f"  [WARN] Could not load {json_path}: {e}")

    if not checkpoint_results:
        print(f"  [SKIP] No results loaded for seed {seed}")
        return

    # Sort by episode count (same ordering as train_and_analyze.py)
    checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

    total_episodes = max(r.get("episode", 0) for _, r in checkpoint_results)

    report_md = generate_report(
        checkpoint_results, "Crafter", seed, total_episodes, seed_root
    )

    report_path = os.path.join(seed_root, f"report_crafter_rainbow_{seed}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"  Written: {report_path}  ({len(checkpoint_results)} checkpoints)")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Rainbow reports from existing analysis JSONs"
    )
    parser.add_argument(
        "--experiment_root", required=True,
        help=(
            "Path to a single seed root (e.g. rainbow_experiment_root/seed_1) "
            "OR the parent directory when --all_seeds is set "
            "(e.g. rainbow_experiment_root)"
        ),
    )
    parser.add_argument(
        "--all_seeds", action="store_true",
        help="Iterate over seed_<N> subdirectories inside experiment_root",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Which seeds to process when --all_seeds is set (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.all_seeds:
        root = args.experiment_root
        if args.seeds:
            seed_dirs = [(s, os.path.join(root, f"seed_{s}")) for s in args.seeds]
        else:
            seed_dirs = []
            for entry in sorted(os.listdir(root)):
                m = __import__("re").match(r"seed_(\d+)$", entry)
                if m:
                    seed_dirs.append((int(m.group(1)), os.path.join(root, entry)))
        if not seed_dirs:
            print(f"No seed_<N> directories found under {root}")
            sys.exit(1)
        print(f"Processing {len(seed_dirs)} seed(s) under {root} ...")
        for seed, seed_root in seed_dirs:
            print(f"\n--- seed {seed} ---")
            _redo_seed(seed_root, seed)
    else:
        # Single seed: derive seed number from directory name
        import re
        m = re.search(r"seed_(\d+)", args.experiment_root)
        seed = int(m.group(1)) if m else 0
        print(f"Processing seed {seed} at {args.experiment_root} ...")
        _redo_seed(args.experiment_root, seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
