"""dissertation_graphs/ppo/plot_all_metrics.py

Per-metric timeseries for PPO.  One PNG per metric saved to ppo/output/metrics/.

Each graph shows mean +/- 1 std across all seeds, with achievement milestone
markers (clustered where achievements are close together on the x-axis).

Metrics produced:
  Scalar:  opposition_score, coherence_success/failure, gradient_magnitude_success/failure,
           activation_separation, activation_cosine_distance,
           rsa_alignment_{fighting,resource,crafting,housing}, opposition_score_is
  MORA:    opp(r>0,r=0), opp(r>0,failure), opp(r=0,failure),
           grad_mag_{positive,neutral,negative}

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/ppo/plot_all_metrics.py --ppo_root ppo_experiment_root
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _build_step_grid,
    _avg_metric_across_seeds,
    _avg_mora_series,
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _rq_fmt_millions,
    _make_shaded_line,
    C_RED, C_ORANGE_LIGHT, C_ORANGE_DARK,
    C_YELLOW_LIGHT, C_YELLOW_DARK,
    C_INDIGO, C_VIOLET, C_TEAL,
    _C_POS_MOR, _C_NEU_MOR, _C_NEG_MOR,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_ALGORITHM = "ppo"


# ── Local data loader ─────────────────────────────────────────────────────────

def _load_checkpoint_results(seed_dir: str, algorithm: str) -> list:
    log_dir = os.path.join(seed_dir, "analysis_logs", algorithm)
    if not os.path.isdir(log_dir):
        return []
    results = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            rec = load_analysis_results(fpath)
        except Exception:
            continue
        label = label_from_path(os.path.splitext(fname)[0])
        results.append((label, rec))
    results.sort(key=lambda x: (x[1].get("episode", 0) or 0,
                                 x[1].get("global_step", 0) or 0))
    return results


def _load_all_seed_results(experiment_root: str, algorithm: str) -> dict:
    out = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if not m:
            continue
        seed_id = int(m.group(1))
        out[seed_id] = _load_checkpoint_results(
            os.path.join(experiment_root, name), algorithm
        )
    return out


# ── Metric definitions ────────────────────────────────────────────────────────

# (field_key, colour, y_label, output_filename_stem)
_SCALAR_METRICS = [
    ("opposition_score",           C_RED,          "Opposition Score",             "opposition_score"),
    ("coherence_success",          C_ORANGE_LIGHT, "Coherence (success group)",    "coherence_success"),
    ("coherence_failure",          C_ORANGE_DARK,  "Coherence (failure group)",    "coherence_failure"),
    ("gradient_magnitude_success", C_YELLOW_LIGHT, "Gradient Magnitude (success)", "grad_mag_success"),
    ("gradient_magnitude_failure", C_YELLOW_DARK,  "Gradient Magnitude (failure)", "grad_mag_failure"),
    ("activation_separation",      C_INDIGO,       "Activation Separation",        "activation_separation"),
    ("activation_cosine_distance", C_VIOLET,       "Activation Cosine Distance",   "activation_cosine_dist"),
    ("rsa_alignment_fighting",     C_TEAL,         "RSA Alignment: Fighting",      "rsa_fighting"),
    ("rsa_alignment_resource",     C_TEAL,         "RSA Alignment: Resource",      "rsa_resource"),
    ("rsa_alignment_crafting",     C_TEAL,         "RSA Alignment: Crafting",      "rsa_crafting"),
    ("rsa_alignment_housing",      C_TEAL,         "RSA Alignment: Housing",       "rsa_housing"),
    ("opposition_score_is",        C_RED,          "Opposition Score (IS)",        "opposition_score_is"),
]

# MORA sub-keys (accessed via record["moment_of_reward"][mora_key])
_MORA_METRICS = [
    ("opp_pos_vs_neutral",          _C_POS_MOR, "MORA: opp(r>0, r=0)",     "mora_opp_pos_neutral"),
    ("opp_pos_vs_failure",          _C_POS_MOR, "MORA: opp(r>0, failure)", "mora_opp_pos_failure"),
    ("opp_neutral_vs_failure",      _C_NEU_MOR, "MORA: opp(r=0, failure)", "mora_opp_neutral_failure"),
    ("gradient_magnitude_positive", _C_POS_MOR, "Grad Magnitude (r>0)",    "mora_grad_mag_positive"),
    ("gradient_magnitude_neutral",  _C_NEU_MOR, "Grad Magnitude (r=0)",    "mora_grad_mag_neutral"),
    ("gradient_magnitude_negative", _C_NEG_MOR, "Grad Magnitude (r<0)",    "mora_grad_mag_negative"),
]


# ── Individual graph helpers ──────────────────────────────────────────────────

def _save_fig(fig, out_dir: str, fname: str, dpi: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _plot_scalar(results, step_grid, ach_steps, field, color,
                 ylabel, stem, out_dir, dpi) -> str | None:
    mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
    fin = np.isfinite(mean)
    if not fin.any():
        print(f"  skip {field!r} — no finite data")
        return None

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin], color,
                      label=f"{ylabel}  (n={n})")
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")

    handles, _ = _add_all_achievement_markers(ax, ach_steps)
    _rq_fmt_millions(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(f"PPO — {ylabel}  (mean +/- 1 std, {n} seeds)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=3)

    return _save_fig(fig, out_dir, f"{stem}_ppo.png", dpi)


def _plot_mora(results, step_grid, ach_steps, mora_key, color,
               ylabel, stem, out_dir, dpi) -> str | None:
    mean, std, n = _avg_mora_series(results, mora_key, step_grid)
    fin = np.isfinite(mean)
    if not fin.any():
        print(f"  skip mora:{mora_key!r} — no finite data")
        return None

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin], color,
                      label=f"{ylabel}  (n={n})")
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")

    handles, _ = _add_all_achievement_markers(ax, ach_steps)
    _rq_fmt_millions(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(f"PPO — {ylabel}  (mean +/- 1 std, {n} seeds)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=3)

    return _save_fig(fig, out_dir, f"{stem}_ppo.png", dpi)


# ── Public entry point ────────────────────────────────────────────────────────

def plot(ppo_root: str, out_dir: str, dpi: int = 150) -> list:
    results = _load_all_seed_results(ppo_root, _ALGORITHM)
    if not results:
        print("ERROR: no PPO seed data found.")
        return []

    step_grid = _build_step_grid(results)
    ach_steps = _ach_steps_from_all_results(results)
    paths = []

    print(f"  Generating {len(_SCALAR_METRICS)} scalar metric graphs...")
    for field, color, ylabel, stem in _SCALAR_METRICS:
        p = _plot_scalar(results, step_grid, ach_steps,
                         field, color, ylabel, stem, out_dir, dpi)
        if p:
            paths.append(p)

    print(f"  Generating {len(_MORA_METRICS)} MORA metric graphs...")
    for mora_key, color, ylabel, stem in _MORA_METRICS:
        p = _plot_mora(results, step_grid, ach_steps,
                       mora_key, color, ylabel, stem, out_dir, dpi)
        if p:
            paths.append(p)

    print(f"  Total: {len(paths)} graphs saved to {out_dir}")
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True,
                        help="Path to PPO experiment root")
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output", "metrics"),
                        help="Output directory (default: ppo/output/metrics/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ppo_root = (os.path.join(_ROOT, args.ppo_root)
                if not os.path.isabs(args.ppo_root) else args.ppo_root)
    plot(ppo_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
