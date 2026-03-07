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
    m = re.match(r"rainbow_ep(\d+)_step(\d+)", name)
    if m:
        return f"Checkpoint {int(m.group(1)):,} episodes — step {int(m.group(2)):,}"
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
    lines += [
        "## Summary",
        "",
        "| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) "
        "| Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"| {_f(r.get('activation_cosine_distance'))} "
            f"| {_f(rsa_align)} |"
        )
    lines += [""]

    # ── per-checkpoint sections ───────────────────────────────────────────────
    for label, r in checkpoint_results:
        lines += ["---", "", f"## {label}", ""]
        if r.get("threshold_mu") is not None:
            threshold_line = f"**Threshold μ:** {_f(r.get('threshold_mu'), 3)}"
        else:
            px = r.get("percentile_x", 25)
            threshold_line = (
                f"**Threshold:** bottom {px}% (≤ {_f(r.get('threshold_lower'), 3)}) | "
                f"top {px}% (≥ {_f(r.get('threshold_upper'), 3)})"
            )
        lines += [
            f"**Episodes:** {_fi(r.get('episode'))}  ",
            f"**Success:** {_fi(r.get('n_success'))}  ",
            f"**Failure:** {_fi(r.get('n_failure'))}  ",
            threshold_line,
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

        # reward moments
        rm = r.get("reward_moments") or {}
        if rm:
            lines += [
                "### Reward Moments", "",
                "| Moment | Grad Magnitude | Cosine vs. Success |",
                "|---|---:|---:|",
            ]
            for moment in ("positive", "neutral", "negative"):
                m_data = rm.get(moment) or {}
                lines.append(
                    f"| {moment.capitalize()} "
                    f"| {_f(m_data.get('gradient_magnitude'))} "
                    f"| {_f(m_data.get('cosine_vs_full_success'))} |"
                )
            lines += [""]

        # activation metrics
        cs = r.get("cluster_stats") or {}
        lines += [
            "### Activation Metrics", "",
            "| Metric | Value |",
            "|---|---|",
            f"| Activation Separation | {_f(r.get('activation_separation'))} |",
            f"| Cosine Distance | {_f(r.get('activation_cosine_distance'))} |",
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

    lines += ["---", "", "## Reward Moments (mean ± std across seeds)", ""]
    for moment in ("positive", "neutral", "negative"):
        lines += [
            f"### {moment.capitalize()} reward sub-group", "",
            "| Checkpoint | Grad Magnitude | Cosine vs. Success |",
            "|---|---:|---:|",
        ]
        for k in sorted_stages:
            label = all_stages[k]
            grad_vals, cos_vals = [], []
            for results in all_seed_results.values():
                for lbl, r in results:
                    if _stage_key(lbl) == k:
                        rm = r.get("reward_moments") or {}
                        m_data = rm.get(moment) or {}
                        grad_vals.append(m_data.get("gradient_magnitude"))
                        cos_vals.append(m_data.get("cosine_vs_full_success"))
            lines.append(f"| {label} | {_fms(grad_vals)} | {_fms(cos_vals)} |")
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
