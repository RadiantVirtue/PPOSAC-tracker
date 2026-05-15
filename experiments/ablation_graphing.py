"""Ablation comparison figures (Appendix A).

Loads analysis JSON logs produced by ablation_experiments.py and generates:

  1. Training-param sensitivity bars:
     For each metric, a grouped bar chart showing median value at the final
     checkpoint across configs, grouped by algorithm.  Covers:
       Rainbow: baseline, lr_low, lr_high, batch_small, batch_large
       PPO:     baseline, ent_low, ent_high, nsteps_short, nsteps_long

  2. Analysis-param stability bars:
     Same layout but for:
       percentile_x ablations:    pct_10, baseline(pct_25), pct_40
       n_eval_episodes ablations: eval_100, eval_500, baseline, eval_1000
     eval_500 should match baseline (it uses the same n_eval_episodes as the
     baseline default), providing a visible anchor for the stability claim.

Figures are saved to ablation_results/figures/.

Usage:
    python ablation_graphing.py
    python ablation_graphing.py --experiment-root ablation_results --algorithms rainbow ppo
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro



# (label for axis, config_name as stored in ablation_results/)
RAINBOW_TRAINING_CONFIGS = [
    ("baseline",    "baseline"),
    ("lr ×0.5",     "lr_low"),
    ("lr ×2",       "lr_high"),
    ("batch=16",    "batch_small"),
    ("batch=64",    "batch_large"),
]

PPO_TRAINING_CONFIGS = [
    ("baseline",    "baseline"),
    ("ent=0.001",   "ent_low"),
    ("ent=0.05",    "ent_high"),
    ("n_steps=64",  "nsteps_short"),
    ("n_steps=256", "nsteps_long"),
]

PERCENTILE_CONFIGS = [
    ("pct=10",  "pct_10"),
    ("pct=25",  "baseline"),
    ("pct=40",  "pct_40"),
]

EVAL_CONFIGS = [
    ("eval=100",  "eval_100"),
    ("eval=500",  "eval_500"),
    ("eval=1000", "eval_1000"),
]

# Metrics to plot (json_key, display_name)
METRICS = [
    ("opposition_score",  "Opposition Score"),
    ("coherence_success", "Coherence (success)"),
    ("activation_separation", "Activation Separation"),
]



@dataclass
class Args:
    experiment_root: str = "ablation_results"
    algorithms: list[str] = field(default_factory=lambda: ["rainbow", "ppo"])
    output_dir: str = ""
    """Defaults to {experiment_root}/figures/ if empty."""



def _find_json_files(root: str) -> list[str]:
    """Recursively find all .json files under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                paths.append(os.path.join(dirpath, fn))
    return paths


def _extract_final_metric(
    experiment_root: str,
    algo: str,
    config_name: str,
    metric: str,
) -> list[float]:
    """Return per-seed final-checkpoint metric values for a given config.

    Looks under {experiment_root}/{algo}/{config_name}/seed_*/analysis_logs/{algo}/*.json,
    picks the JSON with the highest step number for each seed, and extracts
    the metric value.  Returns a list of floats (one per seed; NaN if missing).
    """
    config_root = os.path.join(experiment_root, algo, config_name)
    if not os.path.isdir(config_root):
        return []

    seed_dirs = sorted(
        d for d in os.listdir(config_root)
        if d.startswith("seed_") and os.path.isdir(os.path.join(config_root, d))
    )
    if not seed_dirs:
        return []

    values = []
    for seed_dir in seed_dirs:
        log_dir = os.path.join(config_root, seed_dir, "analysis_logs", algo)
        if not os.path.isdir(log_dir):
            values.append(float("nan"))
            continue

        # Pick the checkpoint with the highest step number
        jsons = [f for f in os.listdir(log_dir) if f.endswith(".json")]
        if not jsons:
            values.append(float("nan"))
            continue

        def _step(fn: str) -> int:
            import re
            m = re.search(r"step(\d+)", fn)
            return int(m.group(1)) if m else 0

        latest = max(jsons, key=_step)
        path = os.path.join(log_dir, latest)
        try:
            with open(path) as f:
                data = json.load(f)
            val = data.get(metric)
            values.append(float(val) if val is not None else float("nan"))
        except Exception:
            values.append(float("nan"))

    return values



def _bar_group_plot(
    experiment_root: str,
    algo: str,
    configs: list[tuple[str, str]],   # (label, config_name)
    metrics: list[tuple[str, str]],   # (json_key, display_name)
    title: str,
    out_path: str,
):
    """Grouped bar chart: one subplot per metric, one bar per config.

    Each bar shows the median across seeds; error bar shows IQR.
    """
    n_metrics = len(metrics)
    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
    axes = axes[0]

    colours = plt.cm.tab10.colors
    x = np.arange(n_configs)
    bar_width = 0.65

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        medians, q25s, q75s = [], [], []
        for _, config_name in configs:
            vals = _extract_final_metric(experiment_root, algo, config_name, metric_key)
            vals_clean = [v for v in vals if not np.isnan(v)]
            if vals_clean:
                medians.append(float(np.median(vals_clean)))
                q25s.append(float(np.percentile(vals_clean, 25)))
                q75s.append(float(np.percentile(vals_clean, 75)))
            else:
                medians.append(float("nan"))
                q25s.append(float("nan"))
                q75s.append(float("nan"))

        medians_arr = np.array(medians, dtype=float)
        q25_arr = np.array(q25s, dtype=float)
        q75_arr = np.array(q75s, dtype=float)

        bar_colours = [colours[i % len(colours)] for i in range(n_configs)]
        bars = ax.bar(x, medians_arr, width=bar_width, color=bar_colours, alpha=0.8)

        # Error bars: IQR
        valid = ~np.isnan(medians_arr)
        if valid.any():
            ax.errorbar(
                x[valid], medians_arr[valid],
                yerr=[
                    (medians_arr[valid] - q25_arr[valid]),
                    (q75_arr[valid] - medians_arr[valid]),
                ],
                fmt="none", color="black", capsize=4, linewidth=1.2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _ in configs], rotation=30, ha="right", fontsize=9)
        ax.set_title(metric_name, fontsize=10)
        ax.set_ylabel("Value (median ± IQR)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title}  [{algo.upper()}]", fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def run(args: Args):
    out_dir = args.output_dir or os.path.join(args.experiment_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    for algo in args.algorithms:
        if algo not in ("rainbow", "ppo"):
            print(f"Unknown algorithm {algo!r}, skipping.")
            continue

        training_configs = RAINBOW_TRAINING_CONFIGS if algo == "rainbow" else PPO_TRAINING_CONFIGS

        _bar_group_plot(
            args.experiment_root, algo,
            training_configs, METRICS,
            title="Training Hyperparameter Sensitivity",
            out_path=os.path.join(out_dir, f"{algo}_training_sensitivity.png"),
        )

        _bar_group_plot(
            args.experiment_root, algo,
            PERCENTILE_CONFIGS, METRICS,
            title="Analysis Stability — percentile_x",
            out_path=os.path.join(out_dir, f"{algo}_percentile_stability.png"),
        )

        _bar_group_plot(
            args.experiment_root, algo,
            EVAL_CONFIGS, METRICS,
            title="Analysis Stability — n_eval_episodes",
            out_path=os.path.join(out_dir, f"{algo}_eval_stability.png"),
        )

    print(f"\nAll ablation figures saved to: {out_dir}/")


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
