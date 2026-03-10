"""Analyze checkpoints in an experiment folder within an episode/step range.

Expects the structure:
    experiment_root/
        seed_1/checkpoints/ppo/*.pt
        seed_2/checkpoints/ppo/*.pt
        ...

If the folder contains no seed_* subdirs, the root itself is treated as a
single seed folder.

Usage:
    python analyze_range.py --algorithm rainbow [--shared.* --rainbow.*]
    python analyze_range.py --algorithm ppo     [--shared.* --ppo.*]
"""
import os
import re
from dataclasses import dataclass, field

import tyro

from analyze_checkpoint import analyze_checkpoint
from shared.reporting import generate_report, generate_averaged_report, label_from_path, push_reports
from shared.storage import load_analysis_results


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Shared:
    """Arguments shared by both algorithms."""

    # Folder containing seed_N subdirs (or checkpoints directly)
    experiment_root: str = "Rainbow-Proof-of-Concept-Runs"

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
    auto_push: bool = False


@dataclass
class PPO:
    """PPO-specific analysis arguments."""
    # Only analyze checkpoints with episode number in [from_ep, to_ep]
    # -1 means no limit on that end.
    from_ep: int = 0
    to_ep: int = -1


@dataclass
class Rainbow:
    """Rainbow-specific analysis arguments."""
    # Only analyze checkpoints with step number in [from_step, to_step]
    # -1 means no limit on that end.
    from_step: int = 0
    to_step: int = -1


@dataclass
class Args:
    algorithm: str = "rainbow"   # "ppo" or "rainbow"
    shared: Shared = field(default_factory=Shared)
    ppo: PPO = field(default_factory=PPO)
    rainbow: Rainbow = field(default_factory=Rainbow)


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
    """Extract episode number from a PPO checkpoint filename (ep<N>)."""
    name = os.path.basename(p)
    m = re.search(r"ep(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)k", name)
    if m:
        return int(m.group(1)) * 1000
    return 0


def _step(p):
    """Extract step number from a Rainbow checkpoint filename (step<N> or <N>k)."""
    name = os.path.basename(p)
    m = re.search(r"step(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)k", name)
    if m:
        return int(m.group(1)) * 1000
    return 0


def _key(algo: str):
    """Return the numeric-key function for the given algorithm."""
    return _ep if algo == "ppo" else _step


def _find_checkpoints(seed_dir: str, algo: str) -> list:
    """Return checkpoint paths sorted by episode (PPO) or step (Rainbow)."""
    ckpt_dir = os.path.join(seed_dir, "checkpoints", algo)
    if not os.path.isdir(ckpt_dir):
        return []
    paths = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pt")
    ]
    return sorted(paths, key=_key(algo))


def _in_range(path, from_val, to_val, key_fn):
    """Return True if the checkpoint's number falls within [from_val, to_val]."""
    val = key_fn(path)
    if val < from_val:
        return False
    if to_val >= 0 and val > to_val:
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
    s = args.shared
    algo = args.algorithm

    # Resolve algorithm-specific range
    if algo == "ppo":
        from_val = args.ppo.from_ep
        to_val   = args.ppo.to_ep
        unit     = "ep"
    else:
        from_val = args.rainbow.from_step
        to_val   = args.rainbow.to_step
        unit     = "step"
    key_fn = _key(algo)

    seed_dirs = _find_seed_dirs(s.experiment_root)
    print(f"Found {len(seed_dirs)} seed dir(s) in '{s.experiment_root}'")

    all_seed_results: dict = {}
    report_paths: list = []

    for seed, seed_dir in seed_dirs:
        checkpoints = _find_checkpoints(seed_dir, algo)
        if not checkpoints:
            print(f"  [seed {seed}] No checkpoints found in {seed_dir}/checkpoints/{algo}/")
            continue

        # Filter to the range; always include the final checkpoint
        selected = [p for p in checkpoints if _in_range(p, from_val, to_val, key_fn)]
        if checkpoints[-1] not in selected:
            selected.append(checkpoints[-1])

        to_desc = "∞" if to_val < 0 else to_val
        range_desc = f"{unit} {from_val}–{to_desc}"
        print(f"\n=== Seed {seed} — {len(selected)}/{len(checkpoints)} checkpoint(s) in range [{range_desc}] ===")

        for path in selected:
            label = label_from_path(path)
            if s.skip_existing and _already_analyzed(path, seed_dir, algo):
                print(f"  [skip] {label} (already analyzed)")
                continue
            print(f"  Analyzing: {label}")
            analyze_checkpoint(
                algo, path, seed_dir,
                env_id=s.env_id,
                n_episodes=s.n_eval_episodes,
                device=s.device,
                reason=label,
                run_rsa=s.run_rsa,
                split_mode=s.split_mode,
                percentile_x=s.percentile_x,
            )
            # Track the JSON so it gets pushed alongside reports
            basename = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(seed_dir, "analysis_logs", algo, f"{basename}.json")
            if os.path.exists(json_path):
                report_paths.append(json_path)

        # Collect results for reports
        checkpoint_results = []
        for path in selected:
            r = _load_result(path, seed_dir, algo)
            if r is not None:
                checkpoint_results.append((label_from_path(path), r))
        checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

        if checkpoint_results:
            all_seed_results[seed] = checkpoint_results

        if s.report and checkpoint_results:
            safe_env = s.env_id.replace("/", "_").replace("\\", "_")
            from_k = from_val // 1000
            to_k = to_val // 1000 if to_val >= 0 else "end"
            seed_label = f"{seed}-{from_k}k-{to_k}k" if seed is not None else f"run-{from_k}k-{to_k}k"
            md = generate_report(
                checkpoint_results, s.env_id, seed,
                checkpoint_results[-1][1].get("episode", 0),
                seed_dir,
            )
            report_path = os.path.join(seed_dir, f"report_{safe_env}_{seed_label}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"  Report saved: {report_path}")
            report_paths.append(report_path)
        elif s.report:
            print(f"  [warn] No results for seed {seed} — report skipped")

    if s.report and len(all_seed_results) > 1:
        safe_env = s.env_id.replace("/", "_").replace("\\", "_")
        seeds = [s_ for s_ in all_seed_results if s_ is not None]
        avg_md = generate_averaged_report(all_seed_results, s.env_id, seeds)
        avg_path = os.path.join(s.experiment_root, f"report_averaged_{safe_env}.md")
        os.makedirs(s.experiment_root, exist_ok=True)
        with open(avg_path, "w", encoding="utf-8") as f:
            f.write(avg_md)
        print(f"\nAveraged report saved: {avg_path}")
        report_paths.append(avg_path)

    if s.auto_push and report_paths:
        push_reports(report_paths, s.env_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
