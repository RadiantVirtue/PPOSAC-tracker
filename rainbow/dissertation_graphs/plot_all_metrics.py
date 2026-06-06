"""dissertation_graphs/rainbow/plot_all_metrics.py

Per-metric timeseries for Rainbow.

Produces exactly 8 PDFs (the only ones referenced in the dissertation):
  GRAPHS/Dissertation graphs/7.2.2/coherence_success_rainbow.pdf
  GRAPHS/Dissertation graphs/7.2.2/coherence_failure_rainbow.pdf
  GRAPHS/Dissertation graphs/7.2.3/activation_separation_rainbow.pdf
  GRAPHS/Dissertation graphs/7.2.3/activation_cosine_dist_rainbow.pdf
  GRAPHS/Dissertation graphs/7.3/cos_uniform_is_success_rainbow.pdf
  GRAPHS/Dissertation graphs/7.3/cos_uniform_is_failure_rainbow.pdf
  GRAPHS/Dissertation graphs/7.5.2/rsa_fighting_resource_rainbow.pdf
  GRAPHS/Dissertation graphs/7.5.2/rsa_crafting_housing_rainbow.pdf

Usage (from PPOSAC-tracker/):
    python dissertation_graphs/rainbow/plot_all_metrics.py --rainbow_root rainbow_experiment_root
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
    _ach_steps_from_all_results,
    _add_all_achievement_markers,
    _rq_fmt_millions,
    _make_shaded_line,
    _add_seed_note,
    C_ORANGE_LIGHT,
    C_COHFAIL,
    C_INDIGO,
    C_VIOLET,
    C_RSA_FIGHTING,
    C_RSA_RESOURCE,
    C_RSA_CRAFTING,
    C_RSA_HOUSING,
    _C_UNIFORM,
    METRIC_MARKERS,
    MARKER_EVERY,
    RAINBOW_CLUSTERS,
)
from shared.reporting import label_from_path
from shared.storage import load_analysis_results

_ALGORITHM = "rainbow"
_GRAPHS_ROOT = os.path.join(_ROOT, "GRAPHS", "Dissertation graphs")


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
        out[int(m.group(1))] = _load_checkpoint_results(
            os.path.join(experiment_root, name), algorithm
        )
    return out


def _save_fig(fig, out_dir: str, fname: str, dpi: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _plot_single(results, step_grid, ach_steps, field, color, marker_key,
                 ylabel, fname, subdir, dpi, high_cosine=False) -> str | None:
    mean, std, n = _avg_metric_across_seeds(results, field, step_grid)
    fin = np.isfinite(mean)
    if not fin.any():
        print(f"  skip {field!r} - no finite data")
        return None

    marker, ms = METRIC_MARKERS.get(marker_key, (None, 6)) if marker_key else (None, 6)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _make_shaded_line(ax, step_grid[fin], mean[fin], std[fin], color,
                      label=ylabel, marker=marker, markersize=ms)
    if high_cosine:
        ax.set_ylim(None, 1.002)
    else:
        ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")

    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=RAINBOW_CLUSTERS)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.14), ncol=3)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=4)
    _rq_fmt_millions(ax)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_seed_note(fig, n)
    return _save_fig(fig, os.path.join(_GRAPHS_ROOT, subdir), fname, dpi)


def _plot_rsa_pair(results, step_grid, ach_steps,
                   field1, color1, mkey1, label1,
                   field2, color2, mkey2, label2,
                   fname, subdir, dpi) -> str | None:
    m1, s1, n1 = _avg_metric_across_seeds(results, field1, step_grid)
    m2, s2, n2 = _avg_metric_across_seeds(results, field2, step_grid)
    if not np.isfinite(m1).any() and not np.isfinite(m2).any():
        print(f"  skip RSA pair {field1}/{field2} - no finite data")
        return None

    mk1, ms1 = METRIC_MARKERS.get(mkey1, (None, 6))
    mk2, ms2 = METRIC_MARKERS.get(mkey2, (None, 6))
    n = max(n1, n2)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fin1 = np.isfinite(m1)
    if fin1.any():
        _make_shaded_line(ax, step_grid[fin1], m1[fin1], s1[fin1], color1,
                          label=label1, marker=mk1, markersize=ms1)
    fin2 = np.isfinite(m2)
    if fin2.any():
        _make_shaded_line(ax, step_grid[fin2], m2[fin2], s2[fin2], color2,
                          label=label2, marker=mk2, markersize=ms2)

    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.5, linestyle="--")
    ach_handles = _add_all_achievement_markers(ax, ach_steps, clusters=RAINBOW_CLUSTERS)
    _rq_fmt_millions(ax)
    ax.set_ylabel("RSA Alignment (Spearman ρ)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    data_h, _ = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles=data_h, fontsize=14, loc="upper center",
                     bbox_to_anchor=(0.5, -0.18), ncol=2)
    if ach_handles:
        ax.add_artist(leg1)
        ax.legend(handles=ach_handles, fontsize=14, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), ncol=4)
    _add_seed_note(fig, n)
    return _save_fig(fig, os.path.join(_GRAPHS_ROOT, subdir), fname, dpi)


def plot(rainbow_root: str, _out_dir_ignored: str = "", dpi: int = 150) -> list:
    results = _load_all_seed_results(rainbow_root, _ALGORITHM)
    if not results:
        print("ERROR: no Rainbow seed data found.")
        return []

    step_grid = _build_step_grid(results)
    ach_steps = _ach_steps_from_all_results(results)
    paths = []

    # Section 7.2.2 - Gradient Coherence
    for field, color, mkey, ylabel, fname in [
        ("coherence_success", C_ORANGE_LIGHT, "coherence_success",
         "Coherence (success group)", "coherence_success_rainbow.pdf"),
        ("coherence_failure", C_COHFAIL, "coherence_failure",
         "Coherence (failure group)", "coherence_failure_rainbow.pdf"),
    ]:
        p = _plot_single(results, step_grid, ach_steps,
                         field, color, mkey, ylabel, fname, "7.2.2", dpi)
        if p:
            paths.append(p)

    # Section 7.2.3 - Activation Space Structure
    for field, color, mkey, ylabel, fname in [
        ("activation_separation",      C_INDIGO, None,
         "Activation Separation",      "activation_separation_rainbow.pdf"),
        ("activation_cosine_distance", C_VIOLET, None,
         "Activation Cosine Distance", "activation_cosine_dist_rainbow.pdf"),
    ]:
        p = _plot_single(results, step_grid, ach_steps,
                         field, color, mkey, ylabel, fname, "7.2.3", dpi)
        if p:
            paths.append(p)

    # Section 7.3 - IS Correction
    for field, color, mkey, ylabel, fname, hc in [
        ("cos_uniform_is_success", _C_UNIFORM, "g_uniform",
         "cos(G_uniform, G_IS) - success", "cos_uniform_is_success_rainbow.pdf", True),
        ("cos_uniform_is_failure", _C_UNIFORM, "g_uniform",
         "cos(G_uniform, G_IS) - failure", "cos_uniform_is_failure_rainbow.pdf", True),
    ]:
        p = _plot_single(results, step_grid, ach_steps,
                         field, color, mkey, ylabel, fname, "7.3", dpi,
                         high_cosine=hc)
        if p:
            paths.append(p)

    # Section 7.5.2 - RSA Alignment (paired plots)
    p = _plot_rsa_pair(
        results, step_grid, ach_steps,
        "rsa_alignment_fighting", C_RSA_FIGHTING, "rsa_fighting", "Fighting",
        "rsa_alignment_resource", C_RSA_RESOURCE, "rsa_resource", "Resource",
        "rsa_fighting_resource_rainbow.pdf", "7.5.2", dpi,
    )
    if p:
        paths.append(p)

    p = _plot_rsa_pair(
        results, step_grid, ach_steps,
        "rsa_alignment_crafting", C_RSA_CRAFTING, "rsa_crafting", "Crafting",
        "rsa_alignment_housing",  C_RSA_HOUSING,  "rsa_housing",  "Housing",
        "rsa_crafting_housing_rainbow.pdf", "7.5.2", dpi,
    )
    if p:
        paths.append(p)

    print(f"  Total: {len(paths)} graphs saved.")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rainbow_root", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    rbw_root = (os.path.join(_ROOT, args.rainbow_root)
                if not os.path.isabs(args.rainbow_root) else args.rainbow_root)
    plot(rbw_root, dpi=args.dpi)


if __name__ == "__main__":
    main()
