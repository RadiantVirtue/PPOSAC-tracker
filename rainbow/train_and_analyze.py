"""Train Rainbow DQN on Crafter, analyse every N checkpoints, write a markdown report.

Checkpoint lifecycle (save → analyse → delete):
  - Training saves checkpoint_step{N}.pt every checkpoint_interval steps.
  - After saving, analysis runs immediately on that checkpoint.
  - Checkpoint N is deleted when checkpoint N+1 finishes analysis (not when N
    finishes), so that N's weights are available for the weight-delta comparison
    during N+1's analysis. At most one extra .pt file is on disk at any time.
  - The final held checkpoint is deleted after training completes.
  - checkpoint_live.pt is always kept for resume (overwritten in-place).

Adapted from ppo/train_and_analyze.py.
"""
import os
import sys
from dataclasses import dataclass, field

import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_checkpoint import analyze_checkpoint
from rainbow.train import build_parser, main_rainbow
from shared.graphing import (
    generate_achievement_zoom_graphs,
    generate_rq_graphs,
    generate_rq_graphs_averaged,
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
    T_max: int = 10_000_000
    checkpoint_interval: int = 100_000     # steps between analysis checkpoints
    experiment_root: str = "rainbow_results"
    n_eval_episodes: int = 500
    split_mode: str = "percentile"
    percentile_x: int = 25
    auto_push: bool = False
    device: str = "cpu"
    analyze_every: int = 5              # analyse 1 in every N checkpoints saved
    # Rainbow hyperparameters
    hidden_size: int = 512
    atoms: int = 51
    architecture: str = "canonical"
    memory_capacity: int = 500_000
    learning_rate: float = 0.0000625
    batch_size: int = 32


def _build_rainbow_args(args: Args, seed: int, seed_root: str):
    """Build argparse.Namespace for main_rainbow from our dataclass."""
    parser = build_parser()
    ns = parser.parse_args([])  # defaults
    ns.seed = seed
    ns.T_max = args.T_max
    ns.checkpoint_interval = args.checkpoint_interval
    ns.experiment_root = seed_root
    ns.hidden_size = args.hidden_size
    ns.atoms = args.atoms
    ns.architecture = args.architecture
    ns.memory_capacity = args.memory_capacity
    ns.learning_rate = args.learning_rate
    ns.batch_size = args.batch_size
    ns.disable_cuda = (args.device == "cpu")
    ns.model = None
    return ns


def _run_seed(args: Args, seed: int) -> tuple:
    seed_root = os.path.join(args.experiment_root, f"seed_{seed}")

    rainbow_args = _build_rainbow_args(args, seed, seed_root)

    saved_paths: list[str] = []
    analyzed_count = 0
    checkpoint_results = []

    def on_checkpoint(path: str, prev_path=None, is_milestone=False):
        nonlocal analyzed_count

        if is_milestone:
            # Milestone: analyze immediately using prev_path for weight-delta (may be None).
            # Deletion is handled by train.py immediately after this call returns.
            label = label_from_path(path)
            print(f"  [milestone] {label}")
            analyze_checkpoint(
                "rainbow", path, seed_root,
                n_episodes=args.n_eval_episodes,
                device=args.device,
                reason=label,
                split_mode=args.split_mode,
                percentile_x=args.percentile_x,
                seed=seed,
                prev_checkpoint_path=prev_path,
            )
            basename = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(seed_root, "analysis_logs", "rainbow", f"{basename}.json")
            if os.path.exists(json_path):
                r = load_analysis_results(json_path)
                checkpoint_results.append((label, r))
            return

        # Periodic checkpoint
        saved_paths.append(path)
        analyzed_count += 1

        # Analyse every N-th checkpoint only (skip intermediate ones).
        # Deletion of skipped checkpoints is handled by train.py pointer rotation.
        if analyzed_count % args.analyze_every != 0:
            return

        label = label_from_path(path)
        print(f"  [analyse] {label}")
        analyze_checkpoint(
            "rainbow", path, seed_root,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            reason=label,
            split_mode=args.split_mode,
            percentile_x=args.percentile_x,
            seed=seed,
            prev_checkpoint_path=prev_path,
        )

        # Load result for report
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(seed_root, "analysis_logs", "rainbow", f"{basename}.json")
        if os.path.exists(json_path):
            r = load_analysis_results(json_path)
            checkpoint_results.append((label, r))
        # Deletion of this checkpoint is handled by train.py pointer rotation
        # when the next periodic checkpoint fires.

    print(f"=== Training Rainbow on Crafter (seed={seed}) ===")
    episode_count, _ = main_rainbow(rainbow_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes\n")

    # Final held periodic checkpoints (A and B) are deleted by train.py after the
    # training loop exits - no cleanup needed here.

    checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

    if checkpoint_results:
        rq_graphs = generate_rq_graphs(checkpoint_results, seed_root, seed)
        report_md = generate_report(
            checkpoint_results, "Crafter", seed, episode_count, seed_root,
            rq_graphs=rq_graphs,
        )
        report_path = os.path.join(seed_root, f"report_crafter_rainbow_{seed}.md")
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
    avg_path = os.path.join(args.experiment_root, "report_averaged_crafter_rainbow.md")
    os.makedirs(args.experiment_root, exist_ok=True)
    with open(avg_path, "w", encoding="utf-8") as f:
        f.write(avg_md)
    print(f"\nAveraged report saved: {avg_path}")
    report_paths.append(avg_path)

    if len(all_seed_results) == len(args.seeds):
        print(f"\nAll {len(args.seeds)} seeds complete - generating averaged & zoomed graphs ...")
        generate_rq_graphs_averaged(all_seed_results, args.experiment_root)
        generate_achievement_zoom_graphs(
            args.experiment_root, list(all_seed_results.keys()), algorithm="rainbow"
        )

    if args.auto_push and report_paths:
        push_reports(report_paths, "Crafter")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
