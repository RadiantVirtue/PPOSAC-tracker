"""compare_old_vs_corrected.py

Loads old (pre-fix) and corrected (post-fix) analysis JSONs for each seed
and prints a side-by-side comparison table for all gradient-derived metrics.

Old source  : rainbow_experiment_root/seed_{N}/analysis_logs/rainbow/checkpoint_step{STEP}.json
New source  : corrected_analysis_results/seed_{N}/analysis_logs/rainbow/checkpoint_live.json

Output:
  1. Per-metric comparison table (per-seed values + cross-seed mean ± std, Δ%, sign-flip flag)
  2. Sign-change summary
  3. Dissertation claim verdict table

Usage:
    python compare_old_vs_corrected.py
    python compare_old_vs_corrected.py \\
        --old_root rainbow_experiment_root \\
        --new_root corrected_analysis_results \\
        --old_step 3000000 \\
        --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


ROOT_METRICS: list[tuple[str, str]] = [
    # (json_key, display_label)
    ("opposition_score",               "opposition_score [uniform]"),
    ("opposition_score_is",            "opposition_score [IS]"),
    ("coherence_success",              "coherence_success [uniform]"),
    ("coherence_failure",              "coherence_failure [uniform]"),
    ("coherence_success_is",           "coherence_success [IS]"),
    ("coherence_failure_is",           "coherence_failure [IS]"),
    ("coherence_success_reward",       "coherence_success [reward]"),
    ("coherence_failure_reward",       "coherence_failure [reward]"),
    ("gradient_magnitude_success",     "grad_mag_success [uniform]"),
    ("gradient_magnitude_failure",     "grad_mag_failure [uniform]"),
    ("gradient_magnitude_success_is",  "grad_mag_success [IS]"),
    ("gradient_magnitude_failure_is",  "grad_mag_failure [IS]"),
    ("cos_uniform_is_success",         "cos(G_unif, G_IS) success"),
    ("cos_uniform_is_failure",         "cos(G_unif, G_IS) failure"),
    ("cos_is_reward_success",          "cos(G_IS, G_rew) success"),
    ("cos_is_reward_failure",          "cos(G_IS, G_rew) failure"),
]

# MoR sub-dict keys (present in moment_of_reward and moment_of_reward_failure)
_MOR_BASE_KEYS: list[str] = [
    "gradient_magnitude_positive",
    "gradient_magnitude_neutral",
    "gradient_magnitude_negative",
    "coherence_positive",
    "coherence_neutral",
    "coherence_negative",
    "opp_pos_vs_neutral",
    "opp_pos_vs_negative",
    "opp_neutral_vs_negative",
]

# These keys appear only in moment_of_reward (success group, vs failure)
_MOR_SUCCESS_ONLY: list[str] = [
    "opp_pos_vs_failure",
    "opp_neutral_vs_failure",
    "opp_negative_vs_failure",
]

# These keys appear only in moment_of_reward_failure (failure group, vs success)
_MOR_FAILURE_ONLY: list[str] = [
    "opp_pos_vs_success",
    "opp_neutral_vs_success",
    "opp_negative_vs_success",
]


def _mor_metrics() -> list[tuple[str, str, str]]:
    """Return (flat_key, subdict_name, sub_key) triples for all MoR metrics."""
    rows = []
    for k in _MOR_BASE_KEYS + _MOR_SUCCESS_ONLY:
        rows.append((f"MoR[success].{k}", "moment_of_reward", k))
    for k in _MOR_BASE_KEYS + _MOR_FAILURE_ONLY:
        rows.append((f"MoR[failure].{k}", "moment_of_reward_failure", k))
    return rows



def _load_old(old_root: str, seed: int, step: int) -> dict | None:
    path = os.path.join(
        old_root, f"seed_{seed}", "analysis_logs", "rainbow",
        f"checkpoint_step{step}.json"
    )
    if not os.path.exists(path):
        # Try to find highest available step JSON as fallback
        import glob
        candidates = sorted(
            glob.glob(os.path.join(
                old_root, f"seed_{seed}", "analysis_logs", "rainbow",
                "checkpoint_step*.json"
            ))
        )
        if not candidates:
            return None
        path = candidates[-1]
        print(f"  [warn] step {step} not found for seed {seed}, using {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


def _load_new(new_root: str, seed: int) -> dict | None:
    path = os.path.join(
        new_root, f"seed_{seed}", "analysis_logs", "rainbow",
        "checkpoint_live.json"
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _get(d: dict | None, key: str, subdict: str | None = None) -> float | None:
    if d is None:
        return None
    if subdict:
        sub = d.get(subdict)
        if not isinstance(sub, dict):
            return None
        v = sub.get(key)
    else:
        v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None



def _fmt(v: float | None, w: int = 8) -> str:
    if v is None:
        return f"{'N/A':>{w}}"
    return f"{v:>{w}.4f}"


def _pct_change(old: float | None, new: float | None) -> str:
    if old is None or new is None:
        return "   N/A"
    if abs(old) < 1e-9:
        return "  ±INF"
    pct = (new - old) / abs(old) * 100
    return f"{pct:+6.1f}%"


def _sign_flip(old_vals: list, new_vals: list) -> bool:
    """True if the mean sign changed and is not near zero."""
    old_finite = [v for v in old_vals if v is not None]
    new_finite = [v for v in new_vals if v is not None]
    if not old_finite or not new_finite:
        return False
    m_old = float(np.mean(old_finite))
    m_new = float(np.mean(new_finite))
    # Both must be non-trivially away from zero
    if abs(m_old) < 0.02 and abs(m_new) < 0.02:
        return False
    return (m_old > 0) != (m_new > 0)



def run_comparison(old_root: str, new_root: str, seeds: list[int], old_step: int):

    print(f"\n{'='*100}")
    print(f"  Old (buggy)   : {old_root}/seed_N/analysis_logs/rainbow/checkpoint_step{old_step}.json")
    print(f"  New (corrected): {new_root}/seed_N/analysis_logs/rainbow/checkpoint_live.json")
    print(f"  Seeds: {seeds}")
    print(f"{'='*100}\n")

    # Load all data
    old_data = {s: _load_old(old_root, s, old_step) for s in seeds}
    new_data = {s: _load_new(new_root, s) for s in seeds}

    missing_new = [s for s in seeds if new_data[s] is None]
    if missing_new:
        print(f"[WARN] Corrected JSONs not found for seeds: {missing_new}")
        print(f"       Run:  python run_corrected_analysis.py "
              f"--experiment_root {old_root} --output_root {new_root} "
              f"--seeds {' '.join(str(s) for s in missing_new)} --n_episodes 500\n")
        seeds = [s for s in seeds if new_data[s] is not None]
        if not seeds:
            print("[ERROR] No corrected data available. Aborting comparison.")
            sys.exit(1)

    # Build all rows
    all_rows: list[tuple[str, list, list, bool]] = []  # (label, old_vals, new_vals, is_sign_flip)

    def _add_rows(metric_list, subdict=None):
        for entry in metric_list:
            if subdict is not None:
                flat_key, sd, sk = entry
                label = flat_key
                old_vals = [_get(old_data[s], sk, sd) for s in seeds]
                new_vals = [_get(new_data[s], sk, sd) for s in seeds]
            else:
                json_key, label = entry
                old_vals = [_get(old_data[s], json_key) for s in seeds]
                new_vals = [_get(new_data[s], json_key) for s in seeds]

            flip = _sign_flip(old_vals, new_vals)
            all_rows.append((label, old_vals, new_vals, flip))

    _add_rows(ROOT_METRICS)
    _add_rows(_mor_metrics(), subdict=True)


    LABEL_W = 42
    COL_W   = 8

    # Header
    header_seeds = "".join(
        f"  {'s'+str(s)+'_old':>{COL_W}}  {'s'+str(s)+'_new':>{COL_W}}"
        for s in seeds
    )
    header_stats = f"  {'mean_old':>{COL_W}}  {'mean_new':>{COL_W}}  {'Δ%':>7}  {'flip?':>6}"
    print(f"{'Metric':<{LABEL_W}}{header_seeds}{header_stats}")
    print("-" * (LABEL_W + len(seeds) * (2 * COL_W + 4) + len(header_stats)))

    sign_flips = []

    for label, old_vals, new_vals, flip in all_rows:
        seed_cols = "".join(
            f"  {_fmt(old_vals[i], COL_W)}  {_fmt(new_vals[i], COL_W)}"
            for i in range(len(seeds))
        )
        old_finite = [v for v in old_vals if v is not None]
        new_finite = [v for v in new_vals if v is not None]
        m_old = float(np.mean(old_finite)) if old_finite else None
        m_new = float(np.mean(new_finite)) if new_finite else None
        delta = _pct_change(m_old, m_new)
        flip_str = "*** YES" if flip else "no"
        stats_col = f"  {_fmt(m_old, COL_W)}  {_fmt(m_new, COL_W)}  {delta}  {flip_str:>7}"
        print(f"{label:<{LABEL_W}}{seed_cols}{stats_col}")

        if flip:
            sign_flips.append((label, m_old, m_new))

    print(f"\n{'='*80}")
    print("  SIGN-CHANGE SUMMARY")
    print(f"{'='*80}")
    if not sign_flips:
        print("  No sign changes detected.")
    else:
        for label, m_old, m_new in sign_flips:
            print(f"  *** {label}")
            print(f"      mean_old={m_old:.4f}  →  mean_new={m_new:.4f}")

    print(f"\n{'='*80}")
    print("  DISSERTATION CLAIM VERDICTS")
    print(f"{'='*80}")

    CLAIMS: list[tuple[str, str, str | None]] = [
        # (claim_text, metric_key_for_lookup, subdict_or_None)
        ("pos-reward grad magnitude > neutral (MoR)",
         "gradient_magnitude_positive", "moment_of_reward"),
        ("opp(pos, neutral) < 0 — opposition exists (MoR)",
         "opp_pos_vs_neutral", "moment_of_reward"),
        ("opp(pos, negative) direction (MoR)",
         "opp_pos_vs_negative", "moment_of_reward"),
        ("success vs failure gradient opposition < 0",
         "opposition_score", None),
        ("IS-weighted opposition preserves direction",
         "opposition_score_is", None),
        ("success coherence > failure coherence",
         None, None),  # computed from two metrics
        ("grad magnitude success > failure",
         None, None),
    ]

    def _verdict(m_old, m_new, all_old, all_new):
        if m_old is None or m_new is None:
            return "MISSING DATA"
        if _sign_flip(all_old, all_new):
            # Count how many seeds agree on new sign
            new_sign = m_new > 0
            n_agree = sum(1 for v in all_new if v is not None and (v > 0) == new_sign)
            n_total = sum(1 for v in all_new if v is not None)
            if n_agree >= max(1, n_total - 1):
                return f"INVALIDATED — sign flipped {n_agree}/{n_total} seeds"
            else:
                return "INVESTIGATE — sign flip inconsistent across seeds"
        pct = abs((m_new - m_old) / max(abs(m_old), 1e-9)) * 100
        if pct < 20:
            return "SURVIVES (< 20% magnitude change)"
        else:
            return f"UPDATE VALUE ({pct:.0f}% change, same direction)"

    # Simple claims with a direct metric
    for claim, metric_key, subdict in CLAIMS:
        if metric_key is None:
            continue  # handled separately below
        all_old = [_get(old_data[s], metric_key, subdict) for s in seeds]
        all_new = [_get(new_data[s], metric_key, subdict) for s in seeds]
        old_f = [v for v in all_old if v is not None]
        new_f = [v for v in all_new if v is not None]
        m_old = float(np.mean(old_f)) if old_f else None
        m_new = float(np.mean(new_f)) if new_f else None
        v = _verdict(m_old, m_new, all_old, all_new)
        print(f"  {claim}")
        print(f"    old mean={_fmt(m_old).strip()}  new mean={_fmt(m_new).strip()}  → {v}")
        print()

    # Compound: success coherence > failure coherence
    coh_s_old = [_get(old_data[s], "coherence_success") for s in seeds]
    coh_f_old = [_get(old_data[s], "coherence_failure") for s in seeds]
    coh_s_new = [_get(new_data[s], "coherence_success") for s in seeds]
    coh_f_new = [_get(new_data[s], "coherence_failure") for s in seeds]
    diff_old = [s - f for s, f in zip(coh_s_old, coh_f_old)
                if s is not None and f is not None]
    diff_new = [s - f for s, f in zip(coh_s_new, coh_f_new)
                if s is not None and f is not None]
    m_diff_old = float(np.mean(diff_old)) if diff_old else None
    m_diff_new = float(np.mean(diff_new)) if diff_new else None
    v = _verdict(m_diff_old, m_diff_new, diff_old, diff_new)
    print(f"  success coherence > failure coherence")
    print(f"    old mean(coh_s - coh_f)={_fmt(m_diff_old).strip()}  "
          f"new mean={_fmt(m_diff_new).strip()}  → {v}")
    print()

    # Compound: grad_mag_success > grad_mag_failure
    gms_old = [_get(old_data[s], "gradient_magnitude_success") for s in seeds]
    gmf_old = [_get(old_data[s], "gradient_magnitude_failure") for s in seeds]
    gms_new = [_get(new_data[s], "gradient_magnitude_success") for s in seeds]
    gmf_new = [_get(new_data[s], "gradient_magnitude_failure") for s in seeds]
    diff_old2 = [s - f for s, f in zip(gms_old, gmf_old) if s is not None and f is not None]
    diff_new2 = [s - f for s, f in zip(gms_new, gmf_new) if s is not None and f is not None]
    m2_old = float(np.mean(diff_old2)) if diff_old2 else None
    m2_new = float(np.mean(diff_new2)) if diff_new2 else None
    v2 = _verdict(m2_old, m2_new, diff_old2, diff_new2)
    print(f"  grad magnitude success > failure")
    print(f"    old mean(gm_s - gm_f)={_fmt(m2_old).strip()}  "
          f"new mean={_fmt(m2_new).strip()}  → {v2}")
    print()

    print(f"{'='*80}")
    print("  Legend:")
    print("    SURVIVES       — same sign, < 20% magnitude change")
    print("    UPDATE VALUE   — same sign, > 20% magnitude change; update dissertation number")
    print("    INVESTIGATE    — sign flip inconsistent across seeds; report cautiously")
    print("    INVALIDATED    — sign flip consistent across ≥N-1 seeds; revise claim")
    print(f"{'='*80}\n")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_root", default="rainbow_experiment_root")
    parser.add_argument("--new_root", default="corrected_analysis_results")
    parser.add_argument("--old_step", type=int, default=3_000_000,
                        help="Training step of the old checkpoint JSON to compare (default: 3000000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    # Resolve relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    old_root = (os.path.join(script_dir, args.old_root)
                if not os.path.isabs(args.old_root) else args.old_root)
    new_root = (os.path.join(script_dir, args.new_root)
                if not os.path.isabs(args.new_root) else args.new_root)

    run_comparison(old_root, new_root, args.seeds, args.old_step)


if __name__ == "__main__":
    main()
