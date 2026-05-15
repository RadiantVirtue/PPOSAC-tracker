import os
import re
from dataclasses import dataclass, field

import tyro

from analyze_checkpoint import analyze_checkpoint
from shared.reporting import generate_averaged_report, generate_report, label_from_path, push_reports
from shared.storage import load_analysis_results


@dataclass
class Shared:
    experiment_root: str = "ppo_experiment_root"
    n_eval_episodes: int = 1000
    device: str = "cpu"
    split_mode: str = "percentile"
    percentile_x: int = 25
    skip_existing: bool = True
    report: bool = True
    auto_push: bool = False
    interrupted: bool = False  # resume a crashed run: report covers ALL analyzed checkpoints, not just the selected range


@dataclass
class PPO:
    from_step: int = 0
    to_step: int = 3_000_000


@dataclass
class Rainbow:
    from_step: int = 0
    to_step: int = -1


@dataclass
class Args:
    algorithm: str = "ppo"   # "ppo" or "rainbow"
    shared: Shared = field(default_factory=Shared)
    ppo: PPO = field(default_factory=PPO)
    rainbow: Rainbow = field(default_factory=Rainbow)



def _find_seed_dirs(root: str) -> list:
    results = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        m = re.match(r"seed_(\d+)$", name)
        if m:
            results.append((int(m.group(1)), path))
    return results or [(None, root)]


def _ep(p):
    name = os.path.basename(p)
    m = re.search(r"ep(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)k", name)
    return int(m.group(1)) * 1000 if m else 0


def _step(p):
    name = os.path.basename(p)
    m = re.search(r"step(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)k", name)
    return int(m.group(1)) * 1000 if m else 0


def _key(algo: str):
    return _ep if algo == "ppo" else _step


def _find_checkpoints(seed_dir: str, algo: str) -> list:
    ckpt_dir = os.path.join(seed_dir, "checkpoints", algo)
    if not os.path.isdir(ckpt_dir):
        return []
    paths = []
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt"):
                paths.append(os.path.join(root, f))
    return sorted(paths, key=_key(algo))


def _in_range(path, from_val, to_val, key_fn):
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
    return load_analysis_results(json_path) if os.path.exists(json_path) else None



def main():
    args = tyro.cli(Args)
    s = args.shared
    algo = args.algorithm

    if algo == "ppo":
        from_val, to_val = args.ppo.from_step, args.ppo.to_step
    else:
        from_val, to_val = args.rainbow.from_step, args.rainbow.to_step
    key_fn = _step if algo == "rainbow" else _ep

    seed_dirs = _find_seed_dirs(s.experiment_root)
    print(f"Found {len(seed_dirs)} seed dir(s) in '{s.experiment_root}'")

    all_seed_results: dict = {}
    report_paths: list = []

    for seed, seed_dir in seed_dirs:
        checkpoints = _find_checkpoints(seed_dir, algo)
        if not checkpoints:
            print(f"  [seed {seed}] No checkpoints in {seed_dir}/checkpoints/{algo}/")
            continue

        selected = [p for p in checkpoints if _in_range(p, from_val, to_val, key_fn)]
        if checkpoints[-1] not in selected:
            selected.append(checkpoints[-1])

        to_desc = "∞" if to_val < 0 else to_val
        print(f"\n=== Seed {seed} — {len(selected)}/{len(checkpoints)} checkpoints [step {from_val}–{to_desc}] ===")

        prev_analyzed_path = None
        for path in selected:
            label = label_from_path(path)
            if s.skip_existing and _already_analyzed(path, seed_dir, algo):
                # Update prev even on skip: ensures the next un-skipped checkpoint
                # computes delta against the immediately prior path in the sorted list
                # (whether or not that prior path was itself analyzed this run).
                # The .pt file still exists on disk (analyze_range never deletes checkpoints).
                prev_analyzed_path = path
                print(f"  [skip] {label}")
                continue
            print(f"  Analyzing: {label}")
            analyze_checkpoint(
                algo, path, seed_dir,
                n_episodes=s.n_eval_episodes,
                device=s.device,
                reason=label,
                split_mode=s.split_mode,
                percentile_x=s.percentile_x,
                seed=seed,
                prev_checkpoint_path=prev_analyzed_path,
            )
            prev_analyzed_path = path

        # When resuming an interrupted run, report over all analyzed checkpoints
        # (not just the selected range) so the seed report is complete.
        report_paths_source = checkpoints if s.interrupted else selected
        checkpoint_results = []
        for path in report_paths_source:
            r = _load_result(path, seed_dir, algo)
            if r is not None:
                checkpoint_results.append((label_from_path(path), r))
        checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

        if checkpoint_results:
            all_seed_results[seed] = checkpoint_results

        if s.report and checkpoint_results:
            if s.interrupted:
                seed_label = str(seed) if seed is not None else "run"
            else:
                from_k = from_val // 1000
                to_k = to_val // 1000 if to_val >= 0 else "end"
                seed_label = f"{seed}-{from_k}k-{to_k}k" if seed is not None else f"run-{from_k}k-{to_k}k"
            md = generate_report(
                checkpoint_results, "Crafter", seed,
                checkpoint_results[-1][1].get("episode", 0),
                seed_dir,
            )
            report_path = os.path.join(seed_dir, f"report_crafter_{algo}_{seed_label}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"  Report saved: {report_path}")
            report_paths.append(report_path)

    if s.report and len(all_seed_results) > 1:
        seeds = [s_ for s_ in all_seed_results if s_ is not None]
        avg_md = generate_averaged_report(all_seed_results, "Crafter", seeds)
        avg_path = os.path.join(s.experiment_root, f"report_averaged_crafter_{algo}.md")
        os.makedirs(s.experiment_root, exist_ok=True)
        with open(avg_path, "w", encoding="utf-8") as f:
            f.write(avg_md)
        print(f"\nAveraged report saved: {avg_path}")
        report_paths.append(avg_path)

    if s.auto_push and report_paths:
        push_reports(report_paths, "Crafter")

    print("\nDone.")


if __name__ == "__main__":
    main()
