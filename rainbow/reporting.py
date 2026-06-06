"""Shared report generation and git-push utilities for train_and_analyze scripts."""
import os
import re
import subprocess
from datetime import datetime

import numpy as np


def label_from_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"periodic_(\d+)_(\d+k)_ep(\d+)", name)
    if m:
        return f"Checkpoint {int(m.group(1))} — {m.group(2)} ({m.group(3)} episodes)"
    m = re.match(r"periodic_step(\d+)_ep(\d+)", name)
    if m:
        return f"Step {int(m.group(1)):,}"
    m = re.match(r"final_(\d+k)_episodes", name)
    if m:
        return f"Final — {m.group(1)} episodes"
    m = re.match(r"final_step(\d+)_ep(\d+)", name)
    if m:
        return "Final"
    m = re.match(r"milestone_first_(.+)_ep(\d+)", name)
    if m:
        return f"{m.group(1)} @ ep{m.group(2)}"
    # Rainbow periodic checkpoint format: checkpoint_step2000000
    m = re.match(r"checkpoint_step(\d+)", name)
    if m:
        return f"Step {int(m.group(1)):,}"
    m = re.match(r"sac_step(\d+)_ep(\d+)", name)
    if m:
        return f"Checkpoint step {int(m.group(1)):,} — {int(m.group(2)):,} episodes"
    return name


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
    # SAC: "Checkpoint step 50,000 — 123 episodes"
    m = re.search(r"Checkpoint step ([\d,]+)", label)
    if m:
        return (0, int(m.group(1).replace(",", "")))
    # PPO periodic (clean label): "Step 1,000,000"
    m = re.match(r"^Step ([\d,]+)$", label)
    if m:
        return (0, int(m.group(1).replace(",", "")))
    # PPO periodic (raw filename fallback): "periodic_step1000000_ep5271"
    m = re.match(r"periodic_step(\d+)_ep\d+", label)
    if m:
        return (0, int(m.group(1)))
    # PPO periodic (old format): "Checkpoint N — 50k (123 episodes)"
    m = re.search(r"Checkpoint (\d+)", label)
    if m:
        return (0, int(m.group(1)))
    # Final checkpoints (raw filename fallback): "final_step3000320_ep14207"
    if "Final" in label or re.match(r"final_step\d+_ep\d+", label):
        return (1, 0)
    # Milestone: "collect_wood @ ep123"  — strip episode, match on name only
    m = re.match(r"^(.+?) @ ep\d+", label)
    if m:
        return (2, m.group(1))
    return (3, label)


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


def generate_report(checkpoint_results, env_id, seed, total_episodes, experiment_root,
                    rq_graphs=None):
    """Build a per-seed markdown report.

    Args:
        checkpoint_results: list of (label, result_dict) pairs.
        env_id:             environment name string.
        seed:               seed integer.
        total_episodes:     total episode count at end of training.
        experiment_root:    path to the seed experiment directory.
        rq_graphs:          optional dict {key: relative_path} returned by
                            shared.graphing.generate_rq_graphs().  When provided,
                            a "Longitudinal Analysis" section is inserted between
                            the summary table and the per-checkpoint sections.
    """
    lines = []

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

    lines += ["## Summary", "", _SUMMARY_HEADER, _SUMMARY_SEP]
    for label, r in checkpoint_results:
        lines.append(_summary_row(label, r))
    lines += [""]

    if rq_graphs:
        _RQ_LABELS = {
            # PPO-specific
            "rq1_opposition": (
                "RQ1 — G_uniform Opposition Score Over Training",
                "PPO has no G_IS analog (on-policy, no PER). The low magnitude (typically 0.1–0.6) "
                "and high variance across checkpoints contrasts sharply with Rainbow's stable 0.7–0.98 "
                "range, reflecting the noisier gradient structure of on-policy learning."
            ),
            "rq3_activation_rsa": (
                "RQ3 — Activation Separation and RSA Alignment Co-trajectory",
                "Top: activation separation (Euclidean centroid distance) grows from ~0.8 to 3–5 over "
                "training. Bottom: RSA alignment ρ starts negative (–0.4 to –0.6 early) and transitions "
                "to positive (0.15–0.35) by mid/late training, indicating emerging semantic structure."
            ),
            "rq3_coherence": (
                "RQ3 — Gradient Coherence and Magnitude Over Training",
                "Top: gradient coherence for success and failure groups — PPO coherence is consistently "
                "low (0.05–0.4) throughout, contrasting with Rainbow's 0.83–0.97. "
                "Bottom: gradient magnitude success vs failure — similar magnitudes with slight "
                "failure-group advantage in early training."
            ),
            "rq3_coherence_vs_rsa": (
                "RQ3 — Gradient Coherence vs RSA Alignment (Scatter)",
                "Each point is one periodic checkpoint, coloured by training stage. "
                "Tests whether higher coherence predicts better semantic structure. "
                "A weak positive trend in late training is expected; the scatter pattern "
                "reveals whether the relationship holds longitudinally for PPO."
            ),
            # Rainbow-specific
            "rq1_gradient_variants":  (
                "RQ1 — Directional Stability: cos(G_uniform, G_IS) and Opposition Score",
                "Top panel: cosine similarity between G_uniform and G_IS for success/failure groups "
                "(expected ~0.97–1.0 throughout). Bottom panel: opposition score under both weightings — "
                "G_IS tracks G_uniform closely, confirming IS re-weighting does not substantially "
                "redirect gradient direction."
            ),
            "rq2_cos_is_reward": (
                "RQ2 — PER Directional Influence: cos(G_IS, G_reward)",
                "Alignment between the IS-weighted gradient and the reward-proximal gradient proxy. "
                "High values indicate PER tends to up-weight reward-proximal transitions; "
                "variance across training reflects inconsistency of this alignment."
            ),
            "rq3_coherence_vs_rsa": (
                "RQ3 — Coherence vs Representational Structure (Scatter)",
                "Each point is one periodic checkpoint. Colour encodes training stage (early=dark, "
                "late=bright). A positive slope would support the RQ3 prediction that high gradient "
                "coherence predicts better semantic structure. Weak/absent correlation is itself informative."
            ),
            "rq4_mora_budget": (
                "RQ4 — MORA: Weighted Gradient Budget by Reward Sign",
                "Proportional gradient contribution = gradient_magnitude × n_transitions, normalised "
                "to sum to 1. Resolves the scale problem: despite ~5–10× higher per-transition "
                "magnitude, positive transitions do not overwhelmingly dominate because neutral "
                "transitions vastly outnumber them."
            ),
            "rq4_mora_magnitude_log": (
                "RQ4 — MORA: Per-Transition Gradient Magnitude (Log Scale)",
                "Log y-axis makes the 5–10× gap between positive and neutral per-transition magnitudes "
                "readable without flattening the neutral baseline. Negative transitions sit in between."
            ),
            "rq4_mora_opposition": (
                "RQ4 — MORA: Cross-Group Opposition Scores",
                "Three pairwise comparisons: Positive vs Neutral (directional conflict — persistently "
                "negative means reward moments and exploratory steps push the network in opposite "
                "directions); Positive vs Failure; Neutral vs Failure."
            ),
        }
        lines += ["---", "", "## Longitudinal Analysis", ""]
        for key, rel_path in rq_graphs.items():
            title, caption = _RQ_LABELS.get(key, (key, ""))
            lines += [
                f"### {title}", "",
                f"![{title}]({rel_path})", "",
                f"*{caption}*", "",
            ]

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

        if r.get("opposition_score_is") is not None:
            beta = r.get("beta_used")
            beta_label = f"β={beta:.3f}" if beta is not None else "β=?"
            has_delta = r.get("cos_uniform_success_delta") is not None
            lines += [
                "### Gradient Variant Analysis (RQ1 / RQ2)", "",
                f"| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |",
                "|---|---:|---:|---:|---:|---:|",
                (
                    f"| G_uniform |"
                    f" {_f(r.get('opposition_score'))} |"
                    f" {_f(r.get('coherence_success'))} |"
                    f" {_f(r.get('coherence_failure'))} |"
                    f" {_f(r.get('gradient_magnitude_success'))} |"
                    f" {_f(r.get('gradient_magnitude_failure'))} |"
                ),
                (
                    f"| G_IS ({beta_label}) |"
                    f" {_f(r.get('opposition_score_is'))} |"
                    f" {_f(r.get('coherence_success_is'))} |"
                    f" {_f(r.get('coherence_failure_is'))} |"
                    f" {_f(r.get('gradient_magnitude_success_is'))} |"
                    f" {_f(r.get('gradient_magnitude_failure_is'))} |"
                ),
                "",
                "| Directional Alignment | Success | Failure |",
                "|---|---:|---:|",
                f"| cos(G_uniform, G_IS) | {_f(r.get('cos_uniform_is_success'))} | {_f(r.get('cos_uniform_is_failure'))} |",
                f"| cos(G_IS, G_reward)  | {_f(r.get('cos_is_reward_success'))} | {_f(r.get('cos_is_reward_failure'))} |",
                "",
            ]
            if has_delta:
                lines += [
                    "| Weight-Delta Alignment | Success | Failure |",
                    "|---|---:|---:|",
                    f"| cos(G_uniform, Δθ) | {_f(r.get('cos_uniform_success_delta'))} | {_f(r.get('cos_uniform_failure_delta'))} |",
                    f"| cos(G_IS, Δθ)      | {_f(r.get('cos_is_success_delta'))} | {_f(r.get('cos_is_failure_delta'))} |",
                    f"| cos(G_reward, Δθ)  | {_f(r.get('cos_reward_success_delta'))} | {_f(r.get('cos_reward_failure_delta'))} |",
                    "",
                ]

        mor = r.get("moment_of_reward")
        if mor:
            lines += [
                "### Moment of Reward (Rainbow)", "",
                "| Subgroup | N transitions | Grad Mag | Coherence |",
                "|---|---:|---:|---:|",
                f"| Positive (r > 0) | {_fi(mor.get('n_positive'))} | {_f(mor.get('gradient_magnitude_positive'))} | {_f(mor.get('coherence_positive'))} |",
                f"| Neutral  (r = 0) | {_fi(mor.get('n_neutral'))} | {_f(mor.get('gradient_magnitude_neutral'))} | {_f(mor.get('coherence_neutral'))} |",
                f"| Negative (r < 0) | {_fi(mor.get('n_negative'))} | {_f(mor.get('gradient_magnitude_negative'))} | {_f(mor.get('coherence_negative'))} |",
                "",
                "| Comparison | Opp. Score |",
                "|---|---:|",
                f"| Pos vs Neutral   | {_f(mor.get('opp_pos_vs_neutral'))} |",
                f"| Pos vs Negative  | {_f(mor.get('opp_pos_vs_negative'))} |",
                f"| Neutral vs Neg.  | {_f(mor.get('opp_neutral_vs_negative'))} |",
                f"| Pos vs Failure   | {_f(mor.get('opp_pos_vs_failure'))} |",
                f"| Neutral vs Fail. | {_f(mor.get('opp_neutral_vs_failure'))} |",
                f"| Neg. vs Failure  | {_f(mor.get('opp_negative_vs_failure'))} |",
                "",
            ]

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
