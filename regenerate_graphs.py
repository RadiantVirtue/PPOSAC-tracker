"""Regenerate all graphs from saved analysis JSON files.

Usage:
    python regenerate_graphs.py --algorithm rainbow --experiment_root rainbow_experiment_root
    python regenerate_graphs.py --algorithm ppo     --experiment_root ppo_experiment_root
    python regenerate_graphs.py --algorithm rainbow --experiment_root rainbow_experiment_root --dpi 200

What this does:
  1. Runs the aggregate periodic/milestone graphing (shared/graphing.py main)
     → writes to <experiment_root>/graphs/
  2. Loads per-seed JSON analysis logs and reconstructs checkpoint_results
  3. Calls generate_rq_graphs per seed
     → writes to <seed_root>/graphs/rq/
  4. Calls generate_rq_graphs_averaged across all seeds
     → writes to <experiment_root>/graphs/rq/
  5. (PPO only) Calls generate_ppo_rq_graphs per seed and averaged
"""
import argparse
import os
import re
import sys

# Ensure PPOSAC-tracker root is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from shared.graphing import (
    generate_ppo_rq_graphs,
    generate_ppo_rq_graphs_averaged,
    generate_rq_graphs,
    generate_rq_graphs_averaged,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results


def _load_checkpoint_results(seed_dir: str, algorithm: str) -> list:
    """Load all analysis JSONs for one seed and return checkpoint_results.

    Returns [(label, record), ...] sorted by episode count (then step).
    Periodic and milestone files are both included, matching the format
    produced during training.
    """
    log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
    if not os.path.isdir(log_dir):
        print(f"  WARNING: no analysis_logs found at {log_dir}")
        return []

    results = []
    for fname in os.listdir(log_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            rec = load_analysis_results(fpath)
        except Exception as e:
            print(f"  WARNING: could not load {fname}: {e}")
            continue
        stem = os.path.splitext(fname)[0]
        label = label_from_path(stem)
        results.append((label, rec))

    # Sort: periodic records have global_step; milestone records have episode.
    # Primary sort: episode count (both have it); secondary: step (periodic only).
    def sort_key(item):
        _, rec = item
        ep = rec.get("episode", 0) or 0
        step = rec.get("global_step", 0) or 0
        return (ep, step)

    results.sort(key=sort_key)
    return results


def _discover_seeds(experiment_root: str, algorithm: str) -> list[int]:
    """Return sorted list of seed integers found under experiment_root."""
    seeds = []
    for name in os.listdir(experiment_root):
        m = re.match(r"seed_(\d+)$", name)
        if not m:
            continue
        seed_dir = os.path.join(experiment_root, name)
        log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
        if os.path.isdir(log_dir):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def _run_aggregate_graphs(experiment_root: str, algorithm: str, dpi: int):
    """Run shared/graphing.py main() for periodic/milestone aggregate graphs."""
    import subprocess
    script = os.path.join(_HERE, "shared", "graphing.py")
    cmd = [sys.executable, script, experiment_root, "--algorithm", algorithm, "--dpi", str(dpi)]
    print(f"\n[1] Aggregate graphs: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        print(f"  WARNING: graphing.py exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all graphs from saved JSON analysis logs.")
    parser.add_argument("--algorithm", default="rainbow", choices=["rainbow", "ppo"],
                        help="Which algorithm's data to graph (default: rainbow)")
    parser.add_argument("--experiment_root", default=None,
                        help="Path to experiment root (default: <algorithm>_experiment_root)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI (default: 150)")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="Skip the aggregate periodic/milestone graphs (step 1)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Limit to specific seeds (default: all found)")
    args = parser.parse_args()

    experiment_root = args.experiment_root or f"{args.algorithm}_experiment_root"
    if not os.path.isabs(experiment_root):
        experiment_root = os.path.join(_HERE, experiment_root)

    if not os.path.isdir(experiment_root):
        print(f"ERROR: experiment_root not found: {experiment_root}")
        sys.exit(1)

    print(f"Algorithm : {args.algorithm}")
    print(f"Root      : {experiment_root}")
    print(f"DPI       : {args.dpi}")

    if not args.skip_aggregate:
        _run_aggregate_graphs(experiment_root, args.algorithm, args.dpi)

    all_seeds = _discover_seeds(experiment_root, args.algorithm)
    if not all_seeds:
        print(f"\nERROR: no seed_* directories with analysis_logs/{args.algorithm}/ found.")
        sys.exit(1)

    seeds = args.seeds if args.seeds else all_seeds
    seeds = [s for s in seeds if s in all_seeds]
    if not seeds:
        print(f"ERROR: none of the requested seeds {args.seeds} exist in {experiment_root}")
        sys.exit(1)

    print(f"\n[2] Found seeds: {all_seeds}  (processing: {seeds})")

    all_seed_results: dict[int, list] = {}

    for seed in seeds:
        seed_dir = os.path.join(experiment_root, f"seed_{seed}")
        print(f"\n  Seed {seed}: loading JSONs from {seed_dir}")
        checkpoint_results = _load_checkpoint_results(seed_dir, args.algorithm)
        if not checkpoint_results:
            print(f"  Seed {seed}: no data — skipping RQ graphs")
            continue
        all_seed_results[seed] = checkpoint_results
        print(f"  Seed {seed}: {len(checkpoint_results)} records loaded")

        print(f"  Seed {seed}: generating per-seed RQ graphs ...")
        if args.algorithm == "rainbow":
            rq_graphs = generate_rq_graphs(checkpoint_results, seed_dir, seed, dpi=args.dpi)
        else:
            rq_graphs = generate_ppo_rq_graphs(checkpoint_results, seed_dir, seed, dpi=args.dpi)
        print(f"  Seed {seed}: {len(rq_graphs)} RQ graphs written")

    if len(all_seed_results) >= 1:
        print(f"\n[3] Generating averaged RQ graphs across seeds {list(all_seed_results)} ...")
        if args.algorithm == "rainbow":
            avg_graphs = generate_rq_graphs_averaged(all_seed_results, experiment_root, dpi=args.dpi)
        else:
            avg_graphs = generate_ppo_rq_graphs_averaged(all_seed_results, experiment_root, dpi=args.dpi)
        print(f"    {len(avg_graphs)} averaged RQ graphs written")
    else:
        print("\nNo seed data loaded — skipping averaged RQ graphs")

    print("\nDone.")


if __name__ == "__main__":
    main()
