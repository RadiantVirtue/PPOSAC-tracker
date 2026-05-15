"""dissertation_graphs/rainbow/plot_mora_opp_pos_neutral.py

opp(r>0, r=0) over the stable plateau (1M–3M steps).

Plots mean ± 1 std across seeds for the cosine similarity between the
reward-bearing gradient and the preparatory gradient, both within the
success group. This is the within-success directional comparison set up
in §4.3.2 but not given a dedicated dissertation figure.

Output: dissertation_graphs/rainbow/output/mora_opp_pos_neutral_rainbow.pdf

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_mora_opp_pos_neutral.py \\
        --rainbow_root rainbow_v2
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
    _avg_mora_series,
    _build_step_grid,
    _make_shaded_line,
    _rq_fmt_millions,
    _add_seed_note,
    _C_POS_MOR,
    _C_NEU_MOR,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_START_STEP = 0


def _load_checkpoint_results(seed_dir: str) -> list:
    log_dir = os.path.join(seed_dir, "analysis_logs", "rainbow")
    if not os.path.isdir(log_dir):
        return []
    results = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            rec = load_analysis_results(os.path.join(log_dir, fname))
        except Exception:
            continue
        label = label_from_path(os.path.splitext(fname)[0])
        results.append((label, rec))
    results.sort(key=lambda x: (x[1].get("episode", 0) or 0,
                                 x[1].get("global_step", 0) or 0))
    return results


def _load_all_seed_results(experiment_root: str) -> dict:
    out = {}
    for name in sorted(os.listdir(experiment_root)):
        m = re.match(r"seed_(\d+)$", name)
        if not m:
            continue
        out[int(m.group(1))] = _load_checkpoint_results(
            os.path.join(experiment_root, name)
        )
    return out


def plot(rainbow_root: str, out_dir: str, dpi: int = 150) -> str:
    results = _load_all_seed_results(rainbow_root)
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return ""

    full_grid = _build_step_grid(results)
    end = float(full_grid[-1]) if full_grid.size else 3_000_000.0
    step_grid = np.linspace(_START_STEP, end, 200)

    mean, std, n = _avg_mora_series(results, "opp_pos_vs_neutral", step_grid)
    fin = np.isfinite(mean)
    if not fin.any():
        print("ERROR: no finite opp_pos_vs_neutral data found.")
        return ""

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin],
                      color=_C_POS_MOR, label=r"opp($r{>}0$, $r{=}0$)")

    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(r"MORA: opp($r{>}0$, $r{=}0$) over training")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _rq_fmt_millions(ax)
    ax.legend(fontsize=13, loc="upper right")
    _add_seed_note(fig, n)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mora_opp_pos_neutral_rainbow.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root (e.g. rainbow_v2)")
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "output"),
                        help="Output directory (default: dissertation_graphs/rainbow/output/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    root = (os.path.join(_ROOT, args.rainbow_root)
            if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(root, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
