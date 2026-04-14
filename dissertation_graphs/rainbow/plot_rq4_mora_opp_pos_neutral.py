"""dissertation_graphs/rainbow/plot_rq4_mora_opp_pos_neutral.py

RQ4 dissertation graph: opp(r>0, r=0) trajectory for Rainbow, averaged
across 5 seeds with a confidence band.

Key finding visualised:
  - Step ~100k: +0.014  (near-aligned, no opposition)
  - Step ~300k: +0.189
  - Step ~450k: -0.428  (opposition begins developing)
  - Step ~1500k: -0.745
  - Step ~1700k: -0.830  (near-maximal opposition)

The developing antagonism between achievement-moment gradients (r>0)
and preparatory-step gradients (r=0) is consistent across all 5 seeds.
It is NOT a fixed structural property -- it emerges progressively through
mid-to-late training.

Colour: _C_POS_MOR green (#2ca02c) for the positive-reward sub-group,
consistent with the MORA colour coding in shared/graphing.py.

Usage:
    python dissertation_graphs/rainbow/plot_rq4_mora_opp_pos_neutral.py \\
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
    _avg_mora_series,
    _build_step_grid,
    _rq_fmt_millions,
    _C_POS_MOR,
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
    mean, std, n = _avg_mora_series(results, "opp_pos_vs_neutral", step_grid)

    fin = np.isfinite(mean)
    if not fin.any():
        print("ERROR: no 'opp_pos_vs_neutral' data found in Rainbow JSONs.")
        print("  Check that moment_of_reward is present in the periodic checkpoints.")
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(step_grid[fin], mean[fin], color=_C_POS_MOR, lw=2.0,
            label=f"opp(r>0, r=0)  (n={n} seeds)")
    ax.fill_between(step_grid[fin],
                    (mean - std)[fin], (mean + std)[fin],
                    color=_C_POS_MOR, alpha=0.2, label="+/- 1 std")

    ax.axhline(0.0, color="grey", linestyle="-", lw=1.0, alpha=0.5,
               label="0  (no opposition)")

    ax.set_ylabel("Opposition Score  (cosine similarity)")
    ax.set_title(
        f"Rainbow Averaged ({n} seeds) — RQ4: Achievement vs Preparatory Gradient Opposition\n"
        "opp(r>0, r=0): near-zero at step 100k, deteriorates to -0.83 by step 1.7M"
    )
    _rq_fmt_millions(ax)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rq4_mora_opp_pos_neutral_rainbow_averaged.png")
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
