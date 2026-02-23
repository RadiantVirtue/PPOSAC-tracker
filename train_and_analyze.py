"""Train PPO, analyse every saved checkpoint, write a markdown report."""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import tyro

from analyze_checkpoint import analyze_checkpoint
from ppo.train import Args as PPOArgs, main_ppo
from shared.storage import load_analysis_results


@dataclass
class Args:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    seed: int = 1
    total_episodes: int = 100_000
    checkpoint_freq: Optional[int] = None
    """Episodes between periodic checkpoints. Defaults to total_episodes // 10."""
    checkpoint_achievements: bool = False
    num_procs: int = 16
    frames_per_proc: int = 128
    entropy_coef: float = 0.01
    experiment_root: str = "train_analysis_results"
    n_eval_episodes: int = 100
    device: str = "cuda"


# ── label helpers ────────────────────────────────────────────────────────────

def label_from_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"periodic_(\d+k)_episodes", name)
    if m:
        return f"{m.group(1)} episodes"
    m = re.match(r"final_(\d+k)_episodes", name)
    if m:
        return f"{m.group(1)} episodes (final)"
    m = re.match(r"milestone_first_(.+)_ep(\d+)", name)
    if m:
        return f"{m.group(1)} @ ep{m.group(2)}"
    return name


# ── formatting helpers ────────────────────────────────────────────────────────

def _f(val, decimals=4):
    """Format a float/None for a table cell."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fi(val):
    """Format an int/None for a table cell."""
    if val is None:
        return "—"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return str(val)


# ── report generation ─────────────────────────────────────────────────────────

def generate_report(checkpoint_results, env_id, seed, total_episodes, experiment_root):
    """
    checkpoint_results: list of (label: str, result: dict)
    Returns a markdown string.
    """
    lines = []

    # ── header ────────────────────────────────────────────────────────────────
    lines += [
        "# Training & Analysis Report",
        "",
        f"**Environment:** `{env_id}`  ",
        f"**Seed:** {seed}  ",
        f"**Total episodes:** {total_episodes:,}  ",
        f"**Experiment root:** `{experiment_root}`  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ── summary table ─────────────────────────────────────────────────────────
    lines += [
        "## Summary",
        "",
        "| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) "
        "| Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in checkpoint_results:
        rsa_align = None
        rsa = r.get("rsa")
        if isinstance(rsa, dict):
            rsa_align = rsa.get("alignment")
        lines.append(
            f"| {label} "
            f"| {_fi(r.get('episode'))} "
            f"| {_f(r.get('opposition_score'))} "
            f"| {_f(r.get('coherence_success'))} "
            f"| {_f(r.get('coherence_failure'))} "
            f"| {_f(r.get('gradient_magnitude_success'))} "
            f"| {_f(r.get('gradient_magnitude_failure'))} "
            f"| {_f(r.get('activation_separation'))} "
            f"| {_f(rsa_align)} |"
        )
    lines += [""]

    # ── per-checkpoint sections ───────────────────────────────────────────────
    for label, r in checkpoint_results:
        lines += ["---", "", f"## {label}", ""]
        lines += [
            f"**Episodes:** {_fi(r.get('episode'))}  ",
            f"**Success:** {_fi(r.get('n_success'))}  ",
            f"**Failure:** {_fi(r.get('n_failure'))}  ",
            f"**Threshold μ:** {_f(r.get('threshold_mu'), 3)}",
            "",
        ]

        # gradient metrics
        lines += [
            "### Gradient Metrics", "",
            "| Metric | Value |",
            "|---|---|",
            f"| Opposition Score | {_f(r.get('opposition_score'))} |",
            f"| Coherence (Success) | {_f(r.get('coherence_success'))} |",
            f"| Coherence (Failure) | {_f(r.get('coherence_failure'))} |",
            f"| Gradient Magnitude (Success) | {_f(r.get('gradient_magnitude_success'))} |",
            f"| Gradient Magnitude (Failure) | {_f(r.get('gradient_magnitude_failure'))} |",
            "",
        ]

        # activation metrics
        cs = r.get("cluster_stats") or {}
        lines += [
            "### Activation Metrics", "",
            "| Metric | Value |",
            "|---|---|",
            f"| Activation Separation | {_f(r.get('activation_separation'))} |",
            f"| Clusters | {_fi(cs.get('n_clusters'))} |",
            f"| Noise Fraction | {_f(cs.get('noise_fraction'))} |",
            "",
        ]

        # cluster breakdown
        clusters = cs.get("clusters") or []
        if clusters:
            lines += [
                "### Cluster Breakdown", "",
                "| Cluster | Success | Failure | Size |",
                "|---|---:|---:|---:|",
            ]
            for i, c in enumerate(clusters):
                lines.append(
                    f"| {i} "
                    f"| {_fi(c.get('success_count'))} "
                    f"| {_fi(c.get('failure_count'))} "
                    f"| {_fi(c.get('size'))} |"
                )
            lines += [""]

        # RSA
        rsa = r.get("rsa")
        if isinstance(rsa, dict):
            lines += [
                "### RSA", "",
                "| Metric | Value |",
                "|---|---|",
                f"| Alignment (Spearman ρ) | {_f(rsa.get('alignment'))} |",
                "",
            ]
            rdm = rsa.get("rdm")
            elem_names = rsa.get("element_names") or []
            if rdm and elem_names:
                lines += ["**Representational Dissimilarity Matrix:**", ""]
                lines.append("| |" + "".join(f" {n} |" for n in elem_names))
                lines.append("|---|" + "---|" * len(elem_names))
                for i, row in enumerate(rdm):
                    lines.append(
                        f"| {elem_names[i]} |"
                        + "".join(f" {_f(v, 3)} |" for v in row)
                    )
                lines += [""]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)

    freq = args.checkpoint_freq if args.checkpoint_freq is not None else (args.total_episodes // 10)

    ppo_args = PPOArgs(
        env_id=args.env_id,
        seed=args.seed,
        total_episodes=args.total_episodes,
        checkpoint_freq=freq,
        checkpoint_achievements=args.checkpoint_achievements,
        experiment_root=args.experiment_root,
        num_procs=args.num_procs,
        frames_per_proc=args.frames_per_proc,
        entropy_coef=args.entropy_coef,
        cuda=(args.device == "cuda"),
    )

    saved_paths: list[str] = []

    def on_checkpoint(path: str):
        saved_paths.append(path)
        print(f"  [checkpoint] {os.path.basename(path)}")

    print(f"=== Training PPO on {args.env_id} (seed={args.seed}) ===")
    episode_count, global_step = main_ppo(ppo_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes, {global_step:,} frames\n")

    # deduplicate while preserving order (final ckpt can repeat the last periodic path)
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in saved_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    print(f"=== Analysing {len(unique_paths)} checkpoint(s) ===\n")
    for path in unique_paths:
        label = label_from_path(path)
        print(f"--- {label} ---")
        analyze_checkpoint(
            "ppo", path, args.experiment_root,
            env_id=args.env_id,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            reason=label,
        )

    # load JSONs and build report
    checkpoint_results = []
    for path in unique_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(
            args.experiment_root, "analysis_logs", "ppo", f"{basename}.json"
        )
        if os.path.exists(json_path):
            r = load_analysis_results(json_path)
            checkpoint_results.append((label_from_path(path), r))
        else:
            print(f"  [warn] No analysis JSON for {basename} — skipped in report")

    if not checkpoint_results:
        print("No analysis results to report.")
        return

    checkpoint_results.sort(key=lambda x: x[1].get("episode", 0))

    report_md = generate_report(
        checkpoint_results, args.env_id, args.seed, episode_count, args.experiment_root
    )

    safe_env = args.env_id.replace("/", "_").replace("\\", "_")
    report_path = os.path.join(args.experiment_root, f"report_{safe_env}_{args.seed}.md")
    os.makedirs(args.experiment_root, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
