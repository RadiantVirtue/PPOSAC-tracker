"""Train Rainbow DQN, analyse every saved checkpoint, write a markdown report."""
import os
import sys
from dataclasses import dataclass, field

import tyro

# Allow running directly as a script from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_checkpoint import analyze_checkpoint
from rainbow.train import Args as RainbowArgs, main_rainbow
from shared.reporting import (
    generate_averaged_report,
    generate_report,
    label_from_path,
    push_reports,
)
from shared.storage import load_analysis_results


@dataclass
class Args:
    env_id: str = "MiniGrid-KeyCorridorS3R3-v0"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    total_timesteps: int = 5_000_000
    checkpoint_freq: int = 0             # episodes between checkpoints; 0 = disabled
    checkpoint_step_freq: int = 5_000    # steps between checkpoints; 0 = disabled
    num_envs: int = 16
    experiment_root: str = "Proof-of-Concept-Runs"
    n_eval_episodes: int = 500
    run_rsa: bool = True
    split_mode: str = "percentile"       # "percentile" or "eps"
    percentile_x: int = 25
    auto_push: bool = True
    device: str = "cuda"
    analyze_every: int = 5               # analyse every Nth saved checkpoint


# ── per-seed training + analysis ─────────────────────────────────────────────

def _run_seed(args: Args, seed: int) -> tuple:
    """Train Rainbow + analyse all checkpoints for one seed.
    Returns (checkpoint_results, report_path_or_None)."""
    seed_root = os.path.join(args.experiment_root, f"seed_{seed}")

    rainbow_args = RainbowArgs(
        env_id=args.env_id,
        seed=seed,
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_step_freq=args.checkpoint_step_freq,
        experiment_root=seed_root,
        num_envs=args.num_envs,
    )

    saved_paths: list[str] = []

    def on_checkpoint(path: str):
        saved_paths.append(path)
        print(f"  [checkpoint] {os.path.basename(path)}")

    print(f"=== Training Rainbow on {args.env_id} (seed={seed}) ===")
    episode_count, _ = main_rainbow(rainbow_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes\n")

    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in saved_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    paths_to_analyze = unique_paths[::args.analyze_every]
    if unique_paths and unique_paths[-1] not in paths_to_analyze:
        paths_to_analyze.append(unique_paths[-1])

    print(f"=== Analysing {len(paths_to_analyze)}/{len(unique_paths)} checkpoint(s) for seed {seed} ===\n")
    for path in paths_to_analyze:
        label = label_from_path(path)
        print(f"--- {label} ---")
        analyze_checkpoint(
            "rainbow", path, seed_root,
            env_id=args.env_id,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            reason=label,
            run_rsa=args.run_rsa,
            split_mode=args.split_mode,
            percentile_x=args.percentile_x,
        )

    checkpoint_results = []
    for path in paths_to_analyze:
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(seed_root, "analysis_logs", "rainbow", f"{basename}.json")
        if os.path.exists(json_path):
            r = load_analysis_results(json_path)
            checkpoint_results.append((label_from_path(path), r))
        else:
            print(f"  [warn] No analysis JSON for {basename} — skipped in report")

    checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

    if checkpoint_results:
        safe_env = args.env_id.replace("/", "_").replace("\\", "_")
        report_md = generate_report(
            checkpoint_results, args.env_id, seed, episode_count, seed_root
        )
        report_path = os.path.join(seed_root, f"report_{safe_env}_{seed}.md")
        os.makedirs(seed_root, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Per-seed report saved: {report_path}")
        return checkpoint_results, report_path

    return checkpoint_results, None


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: Args):
    """Run the full Rainbow train-and-analyse loop for all seeds."""
    all_seed_results: dict[int, list] = {}
    report_paths: list[str] = []
    for seed in args.seeds:
        results, rpath = _run_seed(args, seed)
        if results:
            all_seed_results[seed] = results
        else:
            print(f"[warn] Seed {seed} produced no results — skipped in averaged report")
        if rpath:
            report_paths.append(rpath)

    if not all_seed_results:
        print("No results across any seed.")
        return

    safe_env = args.env_id.replace("/", "_").replace("\\", "_")
    avg_md = generate_averaged_report(all_seed_results, args.env_id, args.seeds)
    avg_path = os.path.join(args.experiment_root, f"report_averaged_{safe_env}.md")
    os.makedirs(args.experiment_root, exist_ok=True)
    with open(avg_path, "w", encoding="utf-8") as f:
        f.write(avg_md)
    print(f"\nAveraged report saved: {avg_path}")
    report_paths.append(avg_path)

    if args.auto_push and report_paths:
        push_reports(report_paths, args.env_id)


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
