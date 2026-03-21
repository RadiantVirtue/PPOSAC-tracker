"""Shared report generation and git-push utilities for train_and_analyze scripts."""
import os
import re
import subprocess
from datetime import datetime

import numpy as np


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
    m = re.match(r"sac_step(\d+)_ep(\d+)", name)
    if m:
        return f"Checkpoint step {int(m.group(1)):,} — {int(m.group(2)):,} episodes"
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


def _fms(vals: list) -> str:
    """Format a list of floats as 'mean (±std)', handling None entries."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return "—"
    if len(clean) == 1:
        return f"{clean[0]:.4f}"
    return f"{np.mean(clean):.4f} (±{np.std(clean):.4f})"


# ── averaged report helpers ───────────────────────────────────────────────────

_AVG_METRICS = [
    ("opposition_score",            "Opp. Score"),
    ("coherence_success",           "Coh. (S)"),
    ("coherence_failure",           "Coh. (F)"),
    ("gradient_magnitude_success",  "Grad Mag (S)"),
    ("gradient_magnitude_failure",  "Grad Mag (F)"),
    ("activation_separation",       "Act. Sep."),
    ("activation_cosine_distance",  "Act. Cos. Dist."),
]


def _stage_key(label: str):
    """Sortable alignment key from a checkpoint label."""
    m = re.search(r"Checkpoint (\d+)", label)
    if m:
        return (0, int(m.group(1)))
    if "Final" in label:
        return (1, 0)
    return (2, label)


def _classify_checkpoint(label: str) -> str:
    """Return 'achievement' or 'periodic' based on label."""
    if re.search(r"@ ep\d+", label):
        return "achievement"
    return "periodic"


def _section_title(label: str, r: dict) -> str:
    """Build per-checkpoint section title: name_epN_lowerX_upperY."""
    ep = r.get("episode", 0)
    lower = r.get("threshold_lower")
    upper = r.get("threshold_upper")
    lower_s = f"{lower:.3f}" if lower is not None else "?"
    upper_s = f"{upper:.3f}" if upper is not None else "?"

    m = re.match(r"^(.+?) @ ep\d+", label)
    if m:
        name = m.group(1).replace(" ", "_")
        return f"{name}_ep{ep}_lower{lower_s}_upper{upper_s}"

    m = re.search(r"(\d+)k", label)
    if m:
        step = int(m.group(1)) * 1000
        return f"step_{step}_ep{ep}_lower{lower_s}_upper{upper_s}"

    return f"ep{ep}_lower{lower_s}_upper{upper_s}"


_SUMMARY_HEADER = (
    "| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) "
    "| Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |"
)
_SUMMARY_SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _summary_row(label: str, r: dict) -> str:
    return (
        f"| {label} "
        f"| {_fi(r.get('episode'))} "
        f"| {_f(r.get('opposition_score'))} "
        f"| {_f(r.get('coherence_success'))} "
        f"| {_f(r.get('coherence_failure'))} "
        f"| {_f(r.get('gradient_magnitude_success'))} "
        f"| {_f(r.get('gradient_magnitude_failure'))} "
        f"| {_f(r.get('activation_separation'))} "
        f"| {_f(r.get('activation_cosine_distance'))} "
        f"| {_f(r.get('rsa_alignment'))} |"
    )


# ── report generation ─────────────────────────────────────────────────────────

def generate_report(checkpoint_results, env_id, seed, total_episodes, experiment_root):
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
    lines += ["## Summary", "", _SUMMARY_HEADER, _SUMMARY_SEP]
    for label, r in checkpoint_results:
        lines.append(_summary_row(label, r))
    lines += [""]

    # ── per-checkpoint sections ───────────────────────────────────────────────
    for label, r in checkpoint_results:
        title = _section_title(label, r)
        lines += ["---", "", f"## {title}", ""]

        # merged metrics table
        cs = r.get("cluster_stats") or {}
        rsa_labels = r.get("rsa_labels") or []
        rsa_stimuli = ", ".join(rsa_labels) if rsa_labels else "—"
        lines += [
            "### Metrics", "",
            "| Metric | Value |",
            "|---|---|",
            f"| Opposition Score | {_f(r.get('opposition_score'))} |",
            f"| Coherence (Success) | {_f(r.get('coherence_success'))} |",
            f"| Coherence (Failure) | {_f(r.get('coherence_failure'))} |",
            f"| Gradient Magnitude (Success) | {_f(r.get('gradient_magnitude_success'))} |",
            f"| Gradient Magnitude (Failure) | {_f(r.get('gradient_magnitude_failure'))} |",
            f"| Activation Separation | {_f(r.get('activation_separation'))} |",
            f"| Cosine Distance | {_f(r.get('activation_cosine_distance'))} |",
            f"| Clusters | {_fi(cs.get('n_clusters'))} |",
            f"| Noise Fraction | {_f(cs.get('noise_fraction'))} |",
            f"| RSA Alignment (ρ) | {_f(r.get('rsa_alignment'))} |",
            f"| RSA Stimuli ({_fi(r.get('rsa_n_stimuli'))}) | {rsa_stimuli} |",
            "",
        ]

    # ── aggregate tables ──────────────────────────────────────────────────────
    ach_results = [(l, r) for l, r in checkpoint_results if _classify_checkpoint(l) == "achievement"]
    per_results = [(l, r) for l, r in checkpoint_results if _classify_checkpoint(l) == "periodic"]

    if ach_results:
        lines += ["---", "", "## Achievement Checkpoints", "", _SUMMARY_HEADER, _SUMMARY_SEP]
        for label, r in ach_results:
            lines.append(_summary_row(label, r))
        lines += [""]

    if per_results:
        lines += ["---", "", "## Periodic Checkpoints", "", _SUMMARY_HEADER, _SUMMARY_SEP]
        for label, r in per_results:
            lines.append(_summary_row(label, r))
        lines += [""]

    return "\n".join(lines)


def generate_averaged_report(
    all_seed_results: dict, env_id: str, seeds: list
) -> str:
    """Build a cross-seed averaged markdown report.

    all_seed_results: {seed: [(label, result_dict), ...]}
    """
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


# ── git push ──────────────────────────────────────────────────────────────────

def push_reports(report_paths: list, env_id: str):
    """Stage the given report files, commit, and push to remote."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(cmd):
        return subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    add = _run(["git", "add", "--"] + report_paths)
    if add.returncode != 0:
        print(f"[push] git add failed:\n{add.stderr}")
        return

    status = _run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        print("[push] Nothing new to commit — remote already up to date.")
        return

    commit = _run(["git", "commit", "-m", f"auto: analysis reports [{env_id}]"])
    if commit.returncode != 0:
        print(f"[push] git commit failed:\n{commit.stderr}")
        return

    push = _run(["git", "push"])
    if push.returncode != 0:
        print(f"[push] git push failed:\n{push.stderr}")
        return

    print(f"[push] Reports pushed to remote ({len(report_paths)} file(s)).")
