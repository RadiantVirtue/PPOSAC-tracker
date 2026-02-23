"""Train PPO, analyse every saved checkpoint, write a markdown report."""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

import tyro

from analyze_checkpoint import analyze_checkpoint
from ppo.train import Args as PPOArgs, main_ppo
from shared.storage import load_analysis_results


@dataclass
class Args:
    env_id: str = "MiniGrid-KeyCorridorS3R3-v0"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    total_episodes: int = 30000
    checkpoint_freq: Optional[int] = None
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
    m = re.match(r"periodic_(\d+)_(\d+k)_ep(\d+)", name)
    if m:
        return f"Checkpoint {int(m.group(1))} — {m.group(2)} ({m.group(3)} episodes)"
    m = re.match(r"final_(\d+k)_episodes", name)
    if m:
        return f"Final — {m.group(1)} episodes"
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


# ── per-seed training + analysis ─────────────────────────────────────────────

def _run_seed(args, seed: int, freq: int) -> list:
    """Train PPO + analyse all checkpoints for one seed.
    Returns checkpoint_results sorted by episode count."""
    seed_root = os.path.join(args.experiment_root, f"seed_{seed}")

    ppo_args = PPOArgs(
        env_id=args.env_id,
        seed=seed,
        total_episodes=args.total_episodes,
        checkpoint_freq=freq,
        checkpoint_achievements=args.checkpoint_achievements,
        experiment_root=seed_root,
        num_procs=args.num_procs,
        frames_per_proc=args.frames_per_proc,
        entropy_coef=args.entropy_coef,
        cuda=(args.device == "cuda"),
    )

    saved_paths: list[str] = []

    def on_checkpoint(path: str):
        saved_paths.append(path)
        print(f"  [checkpoint] {os.path.basename(path)}")

    print(f"=== Training PPO on {args.env_id} (seed={seed}) ===")
    episode_count, _ = main_ppo(ppo_args, on_checkpoint_saved=on_checkpoint)
    print(f"Training complete: {episode_count:,} episodes\n")

    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in saved_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    print(f"=== Analysing {len(unique_paths)} checkpoint(s) for seed {seed} ===\n")
    for path in unique_paths:
        label = label_from_path(path)
        print(f"--- {label} ---")
        analyze_checkpoint(
            "ppo", path, seed_root,
            env_id=args.env_id,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            reason=label,
        )

    checkpoint_results = []
    for path in unique_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(seed_root, "analysis_logs", "ppo", f"{basename}.json")
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

    return checkpoint_results


# ── cross-seed averaged report ────────────────────────────────────────────────

# scalar metrics reported in the averaged report
_AVG_METRICS = [
    ("opposition_score",            "Opp. Score"),
    ("coherence_success",           "Coh. (S)"),
    ("coherence_failure",           "Coh. (F)"),
    ("gradient_magnitude_success",  "Grad Mag (S)"),
    ("gradient_magnitude_failure",  "Grad Mag (F)"),
    ("activation_separation",       "Act. Sep."),
]


def _stage_key(label: str):
    """Sortable alignment key from a checkpoint label."""
    m = re.search(r"Checkpoint (\d+)", label)
    if m:
        return (0, int(m.group(1)))
    if "Final" in label:
        return (1, 0)
    return (2, label)


def _fms(vals: list) -> str:
    """Format a list of floats as 'mean (±std)', handling None entries."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return "—"
    if len(clean) == 1:
        return f"{clean[0]:.4f}"
    return f"{np.mean(clean):.4f} (±{np.std(clean):.4f})"


def generate_averaged_report(
    all_seed_results: dict, env_id: str, seeds: list
) -> str:
    """Build a cross-seed averaged markdown report.

    all_seed_results: {seed: [(label, result_dict), ...]}
    """
    # collect all checkpoint stages present across any seed
    all_stages: dict = {}
    for results in all_seed_results.values():
        for label, _ in results:
            k = _stage_key(label)
            if k not in all_stages:
                all_stages[k] = label
    sorted_stages = sorted(all_stages.keys())

    lines = [
        "# Averaged Training & Analysis Report",
        "",
        f"**Environment:** `{env_id}`  ",
        f"**Seeds:** {', '.join(str(s) for s in seeds)}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary (mean ± std across seeds)",
        "",
    ]

    metric_headers = " | ".join(short for _, short in _AVG_METRICS)
    lines.append(f"| Checkpoint | Episodes | {metric_headers} |")
    lines.append("|---|---:|" + "---:|" * len(_AVG_METRICS))

    for k in sorted_stages:
        label = all_stages[k]
        ep_vals, metric_vals = [], {key: [] for key, _ in _AVG_METRICS}
        for results in all_seed_results.values():
            for lbl, r in results:
                if _stage_key(lbl) == k:
                    ep_vals.append(r.get("episode"))
                    for key, _ in _AVG_METRICS:
                        metric_vals[key].append(r.get(key))
        cells = " | ".join(_fms(metric_vals[key]) for key, _ in _AVG_METRICS)
        lines.append(f"| {label} | {_fms(ep_vals)} | {cells} |")

    lines += [""]

    # per-seed breakdown
    lines += ["---", "", "## Per-Seed Breakdown", ""]
    header = "| Seed | Episodes | " + " | ".join(s for _, s in _AVG_METRICS) + " |"
    sep = "|---|---:|" + "---:|" * len(_AVG_METRICS)

    for k in sorted_stages:
        label = all_stages[k]
        lines += [f"### {label}", "", header, sep]
        for seed in seeds:
            results = all_seed_results.get(seed, [])
            row = next((r for lbl, r in results if _stage_key(lbl) == k), None)
            if row is None:
                lines.append(f"| {seed} | — |" + " — |" * len(_AVG_METRICS))
            else:
                cells = " | ".join(_f(row.get(key)) for key, _ in _AVG_METRICS)
                lines.append(f"| {seed} | {_fi(row.get('episode'))} | {cells} |")
        lines += [""]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)

    freq = args.checkpoint_freq if args.checkpoint_freq is not None else (args.total_episodes // 10)

    all_seed_results: dict[int, list] = {}
    for seed in args.seeds:
        results = _run_seed(args, seed, freq)
        if results:
            all_seed_results[seed] = results
        else:
            print(f"[warn] Seed {seed} produced no results — skipped in averaged report")

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


if __name__ == "__main__":
    main()
