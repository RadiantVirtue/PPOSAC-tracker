"""dissertation_graphs/rainbow/plot_rq2_weight_delta.py

RQ2 dissertation graph: cos(G_variant, delta_theta) weight-delta validation
for Rainbow, averaged across 5 seeds.

All three variants (G_uniform, G_IS, G_reward) remain near zero throughout
training, confirming that offline single-checkpoint gradients do not reliably
predict the direction of actual weight updates.  The negative result is itself
significant: it limits the causal claims that can be drawn from the gradient
proxy metrics.

Colours (consistent with shared/graphing.py):
  G_uniform -- _C_UNIFORM (#2c7bb6)
  G_IS      -- _C_IS      (#7b2d8b)
  G_reward  -- _C_REWARD  (#c0507a)

Usage:
    python dissertation_graphs/rainbow/plot_rq2_weight_delta.py \\
        --rainbow_root rainbow_experiment_root
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

# ── Path to PPOSAC-tracker root ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from shared.graphing import (
    _avg_metric_across_seeds,
    _build_step_grid,
    _rq_fmt_millions,
    _C_UNIFORM,
    _C_IS,
    _C_REWARD,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results


# ── Data loading ───────────────────────────────────────────────────────────────

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


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(rainbow_root: str, out_dir: str, dpi: int = 150):
    results = _load_all_seed_results(rainbow_root, "rainbow")
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return None

    step_grid = _build_step_grid(results)

    # (field_key, display_label, color, linestyle)
    series = [
        ("cos_uniform_success_delta", "cos(G_uniform, \u0394\u03b8)", _C_UNIFORM, "-"),
        ("cos_is_success_delta",      "cos(G_IS, \u0394\u03b8)",      _C_IS,      "--"),
        ("cos_reward_success_delta",  "cos(G_reward, \u0394\u03b8)",  _C_REWARD,  "-."),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_seeds = None

    for key, label, color, ls in series:
        mean, std, n = _avg_metric_across_seeds(results, key, step_grid)
        fin = np.isfinite(mean)
        if not fin.any():
            print(f"  WARNING: no data for {key} — skipping")
            continue
        if n_seeds is None:
            n_seeds = n
        ax.plot(step_grid[fin], mean[fin], color=color, lw=2.0,
                linestyle=ls, label=f"{label}  (n={n} seeds)")
        ax.fill_between(step_grid[fin],
                        (mean - std)[fin], (mean + std)[fin],
                        color=color, alpha=0.15)

    ax.axhline(0.0, color="grey", linestyle="-", lw=1.0, alpha=0.5)
    ax.set_ylabel("cos(G_variant, \u0394\u03b8)")
    n_label = n_seeds if n_seeds is not None else "?"
    ax.set_title(
        f"Rainbow Averaged ({n_label} seeds) — RQ2: Weight-Delta Validation\n"
        "All variants near zero — IS weighting does not improve weight-update prediction"
    )
    _rq_fmt_millions(ax)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rq2_weight_delta_rainbow_averaged.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--out_dir", default=os.path.join(_HERE, "output"),
                        help="Directory to save the PNG (default: rainbow/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    rbw_root = os.path.join(_ROOT, args.rainbow_root) if not os.path.isabs(args.rainbow_root) else args.rainbow_root
    plot(rbw_root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
