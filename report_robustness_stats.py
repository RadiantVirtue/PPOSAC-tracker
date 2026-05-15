"""Robustness statistics — Issue #10.

Reads corrected Rainbow analysis JSONs and reports fraction-of-checkpoints
metrics with 95% bootstrap CIs. No pre-committed verdict thresholds — raw
numbers are reported so readers and referees can draw their own interpretive lines.

Metrics reported:
  coherence_failure       — fraction of (seed, checkpoint) pairs where value < 0
  coherence_failure_is    — same for IS-weighted variant
  mora_ratio              — gradient_magnitude_positive / gradient_magnitude_neutral
                            from moment_of_reward sub-dict; distribution reported
  cos_uniform_success_delta — weight-delta cosine validation proxy; mean ± std

Usage (from PPOSAC-tracker/):
    # After run_corrected_analysis.py has populated corrected_analysis_results/:
    python report_robustness_stats.py

    # Custom root (e.g. still using rainbow_experiment_root):
    python report_robustness_stats.py --experiment_root rainbow_experiment_root

    # Report only specific seeds:
    python report_robustness_stats.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np



def _bootstrap_ci(
    values: list[float],
    statistic=np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for statistic over values."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_stats = [statistic(rng.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2
    return float(np.percentile(boot_stats, 100 * alpha)), \
           float(np.percentile(boot_stats, 100 * (1 - alpha)))



def _load_jsons(exp_root: str, seeds: list[int]) -> list[dict]:
    """Load all checkpoint_*.json files from exp_root/seed_N/analysis_logs/rainbow/.

    Matches both checkpoint_live.json and checkpoint_step*.json so it works
    whether Phase B produced per-step checkpoints or only checkpoint_live.json.
    """
    records = []
    for seed in seeds:
        pattern = os.path.join(
            exp_root, f"seed_{seed}", "analysis_logs", "rainbow",
            "checkpoint_*.json"
        )
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  [seed {seed}] No checkpoint JSONs found at {pattern}")
            continue
        for p in paths:
            try:
                with open(p) as f:
                    d = json.load(f)
                d["_seed"] = seed
                d["_path"] = p
                records.append(d)
            except Exception as e:
                print(f"  [warn] Could not load {p}: {e}")
    return records



def _safe_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        import math
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _mora_ratio(d: dict) -> float | None:
    mor = d.get("moment_of_reward")
    if not isinstance(mor, dict):
        return None
    gm_pos = _safe_float(mor, "gradient_magnitude_positive")
    gm_neu = _safe_float(mor, "gradient_magnitude_neutral")
    if gm_pos is None or gm_neu is None or gm_neu == 0:
        return None
    return gm_pos / gm_neu



def _fraction_below(values: list[float], threshold: float) -> float:
    if not values:
        return float("nan")
    return sum(v < threshold for v in values) / len(values)


def _report_metric(
    label: str,
    values: list[float | None],
    threshold: float | None = None,
    threshold_label: str = "",
    rng: np.random.Generator | None = None,
):
    """Print mean ± std, and optionally a fraction-below-threshold with CI."""
    valid = [v for v in values if v is not None]
    n = len(valid)
    if n == 0:
        print(f"  {label}: N/A (no valid values)")
        return
    arr = np.array(valid)
    print(f"  {label}:")
    print(f"    N         = {n}")
    print(f"    mean      = {arr.mean():.4f}")
    print(f"    std       = {arr.std():.4f}")
    print(f"    min       = {arr.min():.4f}")
    print(f"    max       = {arr.max():.4f}")
    if threshold is not None:
        frac = _fraction_below(valid, threshold)
        ci_lo, ci_hi = _bootstrap_ci(
            [float(v < threshold) for v in valid],
            statistic=np.mean, rng=rng
        )
        print(f"    fraction {threshold_label}: {frac:.3f}  "
              f"[95% CI: {ci_lo:.3f}–{ci_hi:.3f}]  (N={n})")


def report(exp_root: str, seeds: list[int]):
    records = _load_jsons(exp_root, seeds)
    if not records:
        print("[ERROR] No records loaded. Check --experiment_root and --seeds.")
        sys.exit(1)

    print(f"\nLoaded {len(records)} checkpoint JSONs "
          f"({len(seeds)} seeds × ~{len(records)//max(len(seeds),1)} checkpoints)\n")

    rng = np.random.default_rng(42)

    coh_f       = [_safe_float(d, "coherence_failure")    for d in records]
    coh_f_is    = [_safe_float(d, "coherence_failure_is") for d in records]
    opp         = [_safe_float(d, "opposition_score")     for d in records]
    opp_is      = [_safe_float(d, "opposition_score_is")  for d in records]
    mora        = [_mora_ratio(d)                          for d in records]
    wd_unif_s   = [_safe_float(d, "cos_uniform_success_delta") for d in records]
    wd_is_s     = [_safe_float(d, "cos_is_success_delta")      for d in records]
    wd_rew_s    = [_safe_float(d, "cos_reward_success_delta")   for d in records]

    print("=" * 70)
    print("  ROBUSTNESS STATISTICS — Rainbow DQN Analysis")
    print("  (No threshold-based verdicts — report numbers directly)")
    print("=" * 70)

    print("\n-- Coherence (failure group) ----------------------------------------")
    _report_metric("coherence_failure (uniform)", coh_f,
                   threshold=0.0, threshold_label="< 0", rng=rng)
    _report_metric("coherence_failure (IS-weighted)", coh_f_is,
                   threshold=0.0, threshold_label="< 0", rng=rng)

    print("\n-- Opposition score --------------------------------------------------")
    _report_metric("opposition_score (uniform)", opp, rng=rng)
    _report_metric("opposition_score (IS-weighted)", opp_is, rng=rng)

    print("\n-- MORA ratio (gradient_magnitude_positive / gradient_magnitude_neutral) --")
    _report_metric("mora_ratio", mora, rng=rng)

    print("\n-- Weight-delta cosine validation (proxy for Experiment 3) ----------")
    _report_metric("cos_uniform_success_delta", wd_unif_s, rng=rng)
    _report_metric("cos_is_success_delta",      wd_is_s,   rng=rng)
    _report_metric("cos_reward_success_delta",  wd_rew_s,  rng=rng)

    # Per-seed breakdown for coherence_failure
    print("\n-- Per-seed coherence_failure summary --------------------------------")
    print(f"  {'Seed':>5}  {'N':>5}  {'mean':>8}  {'std':>8}  {'frac<0':>8}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    for seed in seeds:
        seed_vals = [_safe_float(d, "coherence_failure")
                     for d in records if d.get("_seed") == seed
                     and _safe_float(d, "coherence_failure") is not None]
        if not seed_vals:
            print(f"  {seed:>5}  {'N/A':>5}")
            continue
        arr = np.array(seed_vals)
        frac = _fraction_below(seed_vals, 0.0)
        print(f"  {seed:>5}  {len(seed_vals):>5}  {arr.mean():>8.4f}  "
              f"{arr.std():>8.4f}  {frac:>8.3f}")

    print("\n" + "=" * 70 + "\n")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default="corrected_analysis_results",
                        help="Root directory containing seed_N/analysis_logs/rainbow/ "
                             "(default: corrected_analysis_results)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = (args.experiment_root if os.path.isabs(args.experiment_root)
                else os.path.join(script_dir, args.experiment_root))

    if not os.path.isdir(exp_root):
        print(f"[ERROR] experiment_root not found: {exp_root}")
        print("  Run run_corrected_analysis.py first, or pass --experiment_root.")
        sys.exit(1)

    report(exp_root, args.seeds)


if __name__ == "__main__":
    main()
