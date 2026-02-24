"""Analyze checkpoints in an experiment folder within an episode range.

Expects the structure:
    experiment_root/
        seed_1/checkpoints/ppo/*.pt
        seed_2/checkpoints/ppo/*.pt
        ...

If the folder contains no seed_* subdirs, the root itself is treated as a
single seed folder.
"""
import os
import re
from dataclasses import dataclass

import tyro

from analyze_checkpoint import analyze_checkpoint
from shared.storage import load_analysis_results
from train_and_analyze import generate_report, generate_averaged_report, label_from_path, _push_reports


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # Folder containing seed_N subdirs (or checkpoints directly)
    experiment_root: str = "Proof-of-Concept-Runs"

    # Only analyze checkpoints with episode number >= from_ep (0 = no lower limit)
    from_ep: int = 0

    # Only analyze checkpoints with episode number <= to_ep (-1 = no upper limit)
    to_ep: int = -1

    # Algorithm
    algo: str = "ppo"

    # Environment
    env_id: str = "MiniGrid-KeyCorridorS3R3-v0"

    # Number of evaluation episodes per checkpoint
    n_eval_episodes: int = 500

    # Device
    device: str = "cuda"

    # How to split episodes into success/failure groups
    split_mode: str = "percentile"   # "percentile" or "eps"
    percentile_x: int = 25

    # Whether to run RSA analysis
    run_rsa: bool = False

    # Skip checkpoints that already have an analysis JSON
    skip_existing: bool = True

    # Write per-seed and averaged markdown reports after analysis
    report: bool = True

    # Push generated reports and JSONs to remote git repo when done
    auto_push: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_seed_dirs(root: str) -> list:
    """Return [(seed_number_or_None, seed_dir), ...] sorted by seed number."""
    results = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        m = re.match(r"seed_(\d+)$", name)
        if m:
            results.append((int(m.group(1)), path))
    if results:
        return results
    # No seed_* subdirs — treat root itself as a single seed folder
    return [(None, root)]


def _ep(p):
    """Extract episode number from a checkpoint filename."""
    name = os.path.basename(p)
    m = re.search(r"ep(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)k", name)
    if m:
        return int(m.group(1)) * 1000
    return 0


def _find_checkpoints(seed_dir: str, algo: str) -> list:
    """Return checkpoint paths sorted by episode number."""
    ckpt_dir = os.path.join(seed_dir, "checkpoints", algo)
    if not os.path.isdir(ckpt_dir):
        return []
    paths = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pt")
    ]
    return sorted(paths, key=_ep)


def _in_range(path, from_ep, to_ep):
    """Return True if the checkpoint's episode falls within [from_ep, to_ep]."""
    ep = _ep(path)
    if ep < from_ep:
        return False
    if to_ep >= 0 and ep > to_ep:
        return False
    return True


def _already_analyzed(checkpoint_path: str, seed_dir: str, algo: str) -> bool:
    basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    json_path = os.path.join(seed_dir, "analysis_logs", algo, f"{basename}.json")
    return os.path.exists(json_path)


def _load_result(checkpoint_path: str, seed_dir: str, algo: str):
    basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    json_path = os.path.join(seed_dir, "analysis_logs", algo, f"{basename}.json")
    if os.path.exists(json_path):
        return load_analysis_results(json_path)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)

    seed_dirs = _find_seed_dirs(args.experiment_root)
    print(f"Found {len(seed_dirs)} seed dir(s) in '{args.experiment_root}'")

    all_seed_results: dict = {}
    report_paths: list = []

    for seed, seed_dir in seed_dirs:
        checkpoints = _find_checkpoints(seed_dir, args.algo)
        if not checkpoints:
            print(f"  [seed {seed}] No checkpoints found in {seed_dir}/checkpoints/{args.algo}/")
            continue

        # Filter to the episode range, always include the final checkpoint
        selected = [p for p in checkpoints if _in_range(p, args.from_ep, args.to_ep)]
        if checkpoints[-1] not in selected:
            selected.append(checkpoints[-1])

        range_desc = f"ep {args.from_ep}–{'∞' if args.to_ep < 0 else args.to_ep}"
        print(f"\n=== Seed {seed} — {len(selected)}/{len(checkpoints)} checkpoint(s) in range [{range_desc}] ===")

        for path in selected:
            label = label_from_path(path)
            if args.skip_existing and _already_analyzed(path, seed_dir, args.algo):
                print(f"  [skip] {label} (already analyzed)")
                continue
            print(f"  Analyzing: {label}")
            analyze_checkpoint(
                args.algo, path, seed_dir,
                env_id=args.env_id,
                n_episodes=args.n_eval_episodes,
                device=args.device,
                reason=label,
                run_rsa=args.run_rsa,
                split_mode=args.split_mode,
                percentile_x=args.percentile_x,
            )
            # Track the JSON so it gets pushed alongside reports
            basename = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(seed_dir, "analysis_logs", args.algo, f"{basename}.json")
            if os.path.exists(json_path):
                report_paths.append(json_path)

        # Collect results for reports
        checkpoint_results = []
        for path in selected:
            r = _load_result(path, seed_dir, args.algo)
            if r is not None:
                checkpoint_results.append((label_from_path(path), r))
        checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

        if checkpoint_results:
            all_seed_results[seed] = checkpoint_results

        if args.report and checkpoint_results:
            seed_label = f"seed_{seed}" if seed is not None else "run"
            safe_env = args.env_id.replace("/", "_").replace("\\", "_")
            md = generate_report(
                checkpoint_results, args.env_id, seed,
                checkpoint_results[-1][1].get("episode", 0),
                seed_dir,
            )
            report_path = os.path.join(seed_dir, f"report_{safe_env}_{seed_label}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"  Report saved: {report_path}")
            report_paths.append(report_path)
        elif args.report:
            print(f"  [warn] No results for seed {seed} — report skipped")

    if args.report and len(all_seed_results) > 1:
        safe_env = args.env_id.replace("/", "_").replace("\\", "_")
        seeds = [s for s in all_seed_results if s is not None]
        avg_md = generate_averaged_report(all_seed_results, args.env_id, seeds)
        avg_path = os.path.join(args.experiment_root, f"report_averaged_{safe_env}.md")
        os.makedirs(args.experiment_root, exist_ok=True)
        with open(avg_path, "w", encoding="utf-8") as f:
            f.write(avg_md)
        print(f"\nAveraged report saved: {avg_path}")
        report_paths.append(avg_path)

    if args.auto_push and report_paths:
        _push_reports(report_paths, args.env_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
