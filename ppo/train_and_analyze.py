"""Train PPO on Crafter, analyse every saved checkpoint, write a markdown report."""
import os
import sys
from dataclasses import dataclass, field

import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_checkpoint import analyze_checkpoint
from ppo.train import Args as PPOArgs, main_ppo
from shared.graphing import (
    generate_achievement_zoom_graphs,
    generate_ppo_rq_graphs,
    generate_ppo_rq_graphs_averaged,
)
from shared.reporting import (
    generate_averaged_report,
    generate_report,
    label_from_path,
    push_reports,
)
from shared.storage import load_analysis_results


@dataclass
class Args:
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    total_timesteps: int = 10_000_000
    checkpoint_freq: int = 50_000        # timesteps between periodic checkpoints
    checkpoint_achievements: bool = True
    num_procs: int = 16
    n_steps: int = 128
    ent_coef: float = 0.01
    experiment_root: str = "experiment_root"
    n_eval_episodes: int = 500
    split_mode: str = "percentile"
    percentile_x: int = 25
    auto_push: bool = False
    device: str = "cpu"
    analyze_every: int = 5


def _run_seed(args: Args, seed: int) -> tuple:
    seed_root = os.path.join(args.experiment_root, f"seed_{seed}")

    ppo_args = PPOArgs(
        seed=seed,
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_achievements=args.checkpoint_achievements,
        num_procs=args.num_procs,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        experiment_root=seed_root,
    )

    saved_paths: list[str] = []

    def on_checkpoint(path: str):
        saved_paths.append(path)
        print(f"  [checkpoint] {os.path.basename(path)}")

    print(f"=== Training PPO on Crafter (seed={seed}) ===")
    episode_count, _ = main_ppo(ppo_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes\n")

    seen: set = set()
    unique_paths = []
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
            "ppo", path, seed_root,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            reason=label,
            split_mode=args.split_mode,
            percentile_x=args.percentile_x,
            seed=seed,
        )

    checkpoint_results = []
    for path in paths_to_analyze:
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(seed_root, "analysis_logs", "ppo", f"{basename}.json")
        if os.path.exists(json_path):
            r = load_analysis_results(json_path)
            checkpoint_results.append((label_from_path(path), r))

    checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

    if checkpoint_results:
        rq_graphs = generate_ppo_rq_graphs(checkpoint_results, seed_root, seed)
        report_md = generate_report(
            checkpoint_results, "Crafter", seed, episode_count, seed_root,
            rq_graphs=rq_graphs,
        )
        report_path = os.path.join(seed_root, f"report_crafter_ppo_{seed}.md")
        os.makedirs(seed_root, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Per-seed report saved: {report_path}")
        return checkpoint_results, report_path

    return checkpoint_results, None


def run(args: Args):
    all_seed_results: dict = {}
    report_paths: list = []

    for seed in args.seeds:
        results, rpath = _run_seed(args, seed)
        if results:
            all_seed_results[seed] = results
        if rpath:
            report_paths.append(rpath)

    if not all_seed_results:
        print("No results across any seed.")
        return

    avg_md = generate_averaged_report(all_seed_results, "Crafter", args.seeds)
    avg_path = os.path.join(args.experiment_root, "report_averaged_crafter_ppo.md")
    os.makedirs(args.experiment_root, exist_ok=True)
    with open(avg_path, "w", encoding="utf-8") as f:
        f.write(avg_md)
    print(f"\nAveraged report saved: {avg_path}")
    report_paths.append(avg_path)

    if len(all_seed_results) == len(args.seeds):
        print(f"\nAll {len(args.seeds)} seeds complete — generating averaged & zoomed graphs ...")
        generate_ppo_rq_graphs_averaged(all_seed_results, args.experiment_root)
        generate_achievement_zoom_graphs(
            args.experiment_root, list(all_seed_results.keys()), algorithm="ppo"
        )

    if args.auto_push and report_paths:
        push_reports(report_paths, "Crafter")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
