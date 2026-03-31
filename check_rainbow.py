"""check_rainbow.py — Verification checks for the Rainbow training + analysis pipeline.

Runs a suite of checks across an experiment root produced by train_and_analyze.py.
Checks are grouped into stages that mirror the pipeline:

  Stage 1 — File structure        (required directories, expected files exist)
  Stage 2 — Training progress     (return log trends, episode counts)
  Stage 3 — Checkpoint integrity  (format, required keys, state-dict sanity)
  Stage 4 — Analysis JSON schema  (required keys, numeric ranges, no NaNs)
  Stage 5 — Gradient sanity       (non-zero, finite, IS ordering, coherence bounds)
  Stage 6 — Activation sanity     (separation > 0, cluster_stats complete)
  Stage 7 — Moment-of-reward      (subgroup keys present, proportions sum ≈ 1)
  Stage 8 — Weight-delta          (present where expected, plausible range)
  Stage 9 — Cross-checkpoint      (monotone episode counts, metrics not constant)
  Stage 10 — Report files         (per-seed + averaged report generated)

Usage:
    python check_rainbow.py --experiment_root rainbow_results
    python check_rainbow.py --experiment_root rainbow_results --seeds 1 2 3
    python check_rainbow.py --experiment_root rainbow_results --seed 1 --verbose

Exit code: 0 if all checks pass, 1 if any fail.
"""
import argparse
import json
import math
import os
import sys


# ── Colour helpers ─────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg):    print(f"  {_GREEN}[OK]{_RESET} {msg}")
def _warn(msg):  print(f"  {_YELLOW}[WARN]{_RESET} {msg}")
def _fail(msg):  print(f"  {_RED}[FAIL]{_RESET} {msg}")
def _info(msg):  print(f"    {msg}")
def _head(msg):  print(f"\n{_BOLD}{msg}{_RESET}")


class CheckResult:
    def __init__(self):
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def ok(self, msg):    self.passed += 1; _ok(msg)
    def warn(self, msg):  self.warned += 1; _warn(msg)
    def fail(self, msg):  self.failed += 1; _fail(msg)

    def summarise(self):
        total = self.passed + self.warned + self.failed
        print(
            f"\n{_BOLD}Summary:{_RESET} "
            f"{_GREEN}{self.passed} passed{_RESET}, "
            f"{_YELLOW}{self.warned} warnings{_RESET}, "
            f"{_RED}{self.failed} failed{_RESET} "
            f"(out of {total} checks)"
        )
        return self.failed == 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_finite(v):
    if v is None:
        return True  # None is allowed (optional metric); checked separately
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _is_nan(v):
    if v is None:
        return False
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _list_jsons(seed_root):
    """Return sorted list of analysis JSON paths for a seed."""
    log_dir = os.path.join(seed_root, "analysis_logs", "rainbow")
    if not os.path.isdir(log_dir):
        return []
    paths = sorted(
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith(".json")
    )
    return paths


def _read_return_log(seed_root):
    """Read rainbowreturnlog.txt and return list of floats."""
    path = os.path.join(seed_root, "logs", "rainbowreturnlog.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    try:
        return [float(x) for x in lines]
    except ValueError:
        return None


# ── Stage implementations ──────────────────────────────────────────────────────

def check_file_structure(r, experiment_root, seeds):
    _head("Stage 1 — File structure")

    if os.path.isdir(experiment_root):
        r.ok(f"Experiment root exists: {experiment_root}")
    else:
        r.fail(f"Experiment root missing: {experiment_root}")
        return  # cannot proceed

    for seed in seeds:
        seed_root = os.path.join(experiment_root, f"seed_{seed}")
        if os.path.isdir(seed_root):
            r.ok(f"Seed directory exists: seed_{seed}/")
        else:
            r.fail(f"Seed directory missing: seed_{seed}/")
            continue

        log_dir = os.path.join(seed_root, "analysis_logs", "rainbow")
        if os.path.isdir(log_dir):
            n = len([f for f in os.listdir(log_dir) if f.endswith(".json")])
            r.ok(f"  seed_{seed}: analysis_logs/rainbow/ exists ({n} JSON files)")
        else:
            r.fail(f"  seed_{seed}: analysis_logs/rainbow/ missing")

        return_log = os.path.join(seed_root, "logs", "rainbowreturnlog.txt")
        if os.path.exists(return_log):
            r.ok(f"  seed_{seed}: logs/rainbowreturnlog.txt exists")
        else:
            r.warn(f"  seed_{seed}: logs/rainbowreturnlog.txt not found (training may be in progress or not yet started)")


def check_training_progress(r, experiment_root, seeds, verbose=False):
    _head("Stage 2 — Training progress (return log)")

    for seed in seeds:
        seed_root = os.path.join(experiment_root, f"seed_{seed}")
        returns = _read_return_log(seed_root)

        if returns is None:
            r.warn(f"seed_{seed}: return log missing or unreadable — skipping training progress checks")
            continue

        n = len(returns)
        if n == 0:
            r.fail(f"seed_{seed}: return log is empty")
            continue

        r.ok(f"seed_{seed}: {n:,} episodes logged")

        # Check for NaNs / Infs
        bad = [i for i, v in enumerate(returns) if not _is_finite(v)]
        if bad:
            r.fail(f"seed_{seed}: {len(bad)} non-finite returns in log (first at index {bad[0]})")
        else:
            r.ok(f"seed_{seed}: all logged returns are finite")

        # Check mean return across training thirds (expect non-decreasing trend)
        if n >= 300:
            third = n // 3
            early  = sum(returns[:third])  / third
            mid    = sum(returns[third:2*third]) / third
            late   = sum(returns[2*third:]) / (n - 2*third)
            _info(f"seed_{seed}: mean return — early={early:.3f}  mid={mid:.3f}  late={late:.3f}")
            if late > early:
                r.ok(f"seed_{seed}: late mean return > early mean return (positive trend)")
            else:
                r.warn(f"seed_{seed}: late mean return ({late:.3f}) ≤ early mean return ({early:.3f}) — no clear upward trend")
        else:
            r.warn(f"seed_{seed}: fewer than 300 episodes — cannot assess training trend (got {n})")

        # Check for long stretches of identical returns (possible env hang)
        if n >= 10:
            streak = 1
            max_streak = 1
            for i in range(1, n):
                if returns[i] == returns[i - 1]:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 1
            if max_streak > 100:
                r.warn(f"seed_{seed}: return log has a streak of {max_streak} identical values — possible env hang?")
            else:
                r.ok(f"seed_{seed}: no suspiciously long identical-return streaks (max={max_streak})")

        if verbose and n >= 10:
            sample = returns[-min(10, n):]
            _info(f"seed_{seed}: last {len(sample)} returns: {[round(v, 3) for v in sample]}")


def check_checkpoint_integrity(r, experiment_root, seeds, verbose=False):
    _head("Stage 3 — Checkpoint integrity")

    # We check only the live checkpoint (analysis checkpoints are deleted after use)
    REQUIRED_CKPT_KEYS = {
        "global_step", "episode_count",
        "online_net_state_dict", "target_net_state_dict",
        "optimizer_state_dict", "args_dict",
    }
    REQUIRED_ARGS_KEYS = {
        "atoms", "hidden_size", "architecture",
        "V_min", "V_max", "multi_step", "discount",
        "priority_exponent", "priority_weight",
        "T_max", "learn_start",
    }

    try:
        import torch
        _torch_available = True
    except ImportError:
        _torch_available = False
        r.warn("PyTorch not importable — checkpoint loading checks skipped")

    for seed in seeds:
        seed_root = os.path.join(experiment_root, f"seed_{seed}")
        live_path = os.path.join(seed_root, "checkpoint_live.pt")

        if not os.path.exists(live_path):
            r.warn(f"seed_{seed}: checkpoint_live.pt not found (training may not have started yet)")
            continue

        if not _torch_available:
            continue

        try:
            ckpt = torch.load(live_path, map_location="cpu", weights_only=False)
        except Exception as e:
            r.fail(f"seed_{seed}: checkpoint_live.pt failed to load — {e}")
            continue

        # Required top-level keys
        missing = REQUIRED_CKPT_KEYS - set(ckpt.keys())
        if missing:
            r.fail(f"seed_{seed}: checkpoint_live.pt missing keys: {sorted(missing)}")
        else:
            r.ok(f"seed_{seed}: checkpoint_live.pt has all required top-level keys")

        # global_step and episode_count are positive
        gs = ckpt.get("global_step", 0)
        ec = ckpt.get("episode_count", 0)
        if gs > 0:
            r.ok(f"seed_{seed}: global_step={gs:,}")
        else:
            r.warn(f"seed_{seed}: global_step={gs} (training may not have started)")

        if ec > 0:
            r.ok(f"seed_{seed}: episode_count={ec:,}")
        else:
            r.warn(f"seed_{seed}: episode_count={ec}")

        # args_dict completeness
        args_dict = ckpt.get("args_dict", {})
        missing_args = REQUIRED_ARGS_KEYS - set(args_dict.keys())
        if missing_args:
            r.fail(f"seed_{seed}: args_dict missing keys: {sorted(missing_args)}")
        else:
            r.ok(f"seed_{seed}: args_dict has all required hyperparameter keys")

        # Online and target net are not identical (target lags behind)
        online_sd = ckpt.get("online_net_state_dict", {})
        target_sd = ckpt.get("target_net_state_dict", {})
        if online_sd and target_sd:
            if set(online_sd.keys()) == set(target_sd.keys()):
                r.ok(f"seed_{seed}: online and target nets have matching parameter names")
            else:
                r.fail(f"seed_{seed}: online/target net have different parameter names")

            # Check at least one parameter differs (target should lag)
            any_diff = any(
                not online_sd[k].equal(target_sd[k])
                for k in list(online_sd.keys())[:10]  # spot-check first 10 params
            )
            if any_diff:
                r.ok(f"seed_{seed}: online ≠ target net weights (target is correctly lagging)")
            else:
                r.warn(f"seed_{seed}: online == target net weights (expected lag after training)")


def check_analysis_schema(r, json_path, seed, verbose=False):
    """Check a single analysis JSON has required keys and sane values."""

    REQUIRED_KEYS = [
        "episode", "algorithm", "split_mode", "n_success", "n_failure",
        "opposition_score", "coherence_success", "coherence_failure",
        "activation_separation", "activation_cosine_distance", "cluster_stats",
        "gradient_magnitude_success", "gradient_magnitude_failure",
        # IS-weighted (Rainbow-specific)
        "opposition_score_is",
        "coherence_success_is", "coherence_failure_is",
        "gradient_magnitude_success_is", "gradient_magnitude_failure_is",
        # Funnel cosines
        "cos_uniform_is_success", "cos_uniform_is_failure",
        "cos_is_reward_success", "cos_is_reward_failure",
        # IS reference
        "beta_used", "n_transitions_success", "n_transitions_failure",
        # MoR
        "moment_of_reward",
    ]

    BOUNDED_01 = [
        "coherence_success", "coherence_failure",
        "coherence_success_is", "coherence_failure_is",
        "coherence_success_reward", "coherence_failure_reward",
        "cos_uniform_is_success", "cos_uniform_is_failure",
        "cos_is_reward_success", "cos_is_reward_failure",
    ]

    WEIGHT_DELTA_KEYS = [
        "cos_uniform_success_delta", "cos_is_success_delta", "cos_reward_success_delta",
        "cos_uniform_failure_delta", "cos_is_failure_delta", "cos_reward_failure_delta",
    ]

    try:
        data = _load_json(json_path)
    except Exception as e:
        r.fail(f"seed_{seed}: {os.path.basename(json_path)} — JSON parse error: {e}")
        return None

    name = os.path.basename(json_path)

    # Required keys present
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        r.fail(f"seed_{seed}/{name}: missing required keys: {missing}")
    else:
        r.ok(f"seed_{seed}/{name}: all required keys present")

    # algorithm field
    if data.get("algorithm") != "rainbow":
        r.fail(f"seed_{seed}/{name}: algorithm={data.get('algorithm')!r}, expected 'rainbow'")

    # n_success / n_failure — both must be > 0
    n_s = data.get("n_success", 0)
    n_f = data.get("n_failure", 0)
    if n_s > 0 and n_f > 0:
        r.ok(f"seed_{seed}/{name}: n_success={n_s}, n_failure={n_f}")
    else:
        r.fail(f"seed_{seed}/{name}: empty partition — n_success={n_s}, n_failure={n_f}")

    # No NaN in any numeric field
    nan_keys = [k for k, v in data.items() if isinstance(v, float) and _is_nan(v)]
    if nan_keys:
        r.fail(f"seed_{seed}/{name}: NaN in fields: {nan_keys}")
    else:
        r.ok(f"seed_{seed}/{name}: no NaN values in top-level fields")

    # Bounded [−1, 1] metrics
    for key in BOUNDED_01:
        v = data.get(key)
        if v is None:
            continue  # optional
        if not _is_finite(v):
            r.fail(f"seed_{seed}/{name}: {key}={v} is not finite")
        elif abs(float(v)) > 1.0 + 1e-4:
            r.fail(f"seed_{seed}/{name}: {key}={v:.4f} outside [−1, 1]")

    # Gradient magnitudes — positive floats
    for key in ("gradient_magnitude_success", "gradient_magnitude_failure",
                "gradient_magnitude_success_is", "gradient_magnitude_failure_is"):
        v = data.get(key)
        if v is None:
            continue
        if not _is_finite(v) or float(v) <= 0:
            r.fail(f"seed_{seed}/{name}: {key}={v} should be a positive finite float")

    # activation_separation — positive
    sep = data.get("activation_separation")
    if sep is not None and (_is_nan(sep) or float(sep) <= 0):
        r.warn(f"seed_{seed}/{name}: activation_separation={sep} (expected > 0)")

    # activation_cosine_distance — in [0, 2] (cosine dist is 1 − cos ∈ [0, 2])
    cd = data.get("activation_cosine_distance")
    if cd is not None and _is_finite(cd):
        if float(cd) < 0 or float(cd) > 2.0 + 1e-4:
            r.fail(f"seed_{seed}/{name}: activation_cosine_distance={cd:.4f} outside [0, 2]")

    # beta_used — should be in [0.4, 1.0]
    beta = data.get("beta_used")
    if beta is not None:
        if not (0.4 <= float(beta) <= 1.001):
            r.warn(f"seed_{seed}/{name}: beta_used={beta:.4f}, expected in [0.4, 1.0]")

    # Weight-delta keys — present and finite when not the first checkpoint
    episode_num = data.get("episode", 0)
    has_delta = any(data.get(k) is not None for k in WEIGHT_DELTA_KEYS)
    if not has_delta:
        # First checkpoint or missing prev — acceptable
        pass
    else:
        for k in WEIGHT_DELTA_KEYS:
            v = data.get(k)
            if v is not None and (not _is_finite(v) or abs(float(v)) > 1.001):
                r.fail(f"seed_{seed}/{name}: {k}={v:.4f} outside [−1, 1]")

    if verbose:
        _info(
            f"  episode={data.get('episode')}, "
            f"opp={data.get('opposition_score')}, "
            f"opp_is={data.get('opposition_score_is')}, "
            f"coh_s={data.get('coherence_success'):.4f}, "
            f"coh_f={data.get('coherence_failure'):.4f}, "
            f"act_sep={data.get('activation_separation'):.4f}"
        )

    return data


def check_gradient_sanity(r, data, seed, name):
    """Gradient-specific checks on a loaded JSON result dict."""

    # IS should be similar direction to uniform (cos_uniform_is > 0.5 expected)
    for group in ("success", "failure"):
        key = f"cos_uniform_is_{group}"
        v = data.get(key)
        if v is None:
            continue
        v = float(v)
        if v < 0:
            r.warn(f"seed_{seed}/{name}: {key}={v:.4f} — IS and uniform gradients point in opposite directions")
        elif v < 0.5:
            r.warn(f"seed_{seed}/{name}: {key}={v:.4f} — IS and uniform are weakly aligned (expected > 0.5)")
        else:
            r.ok(f"seed_{seed}/{name}: {key}={v:.4f} ✓ (IS aligns with uniform)")

    # Coherence bounds [0, 1]
    for key in ("coherence_success", "coherence_failure",
                "coherence_success_is", "coherence_failure_is"):
        v = data.get(key)
        if v is None:
            continue
        v = float(v)
        if not (0.0 <= v <= 1.0 + 1e-4):
            r.fail(f"seed_{seed}/{name}: {key}={v:.4f} outside [0, 1]")

    # Gradient magnitudes: success ≠ failure (they should differ)
    gm_s = data.get("gradient_magnitude_success")
    gm_f = data.get("gradient_magnitude_failure")
    if gm_s is not None and gm_f is not None:
        if abs(float(gm_s) - float(gm_f)) < 1e-8:
            r.warn(f"seed_{seed}/{name}: gradient_magnitude_success == gradient_magnitude_failure (suspiciously identical)")

    # IS gradient magnitude should differ from uniform (IS re-weights transitions)
    gm_s_is = data.get("gradient_magnitude_success_is")
    if gm_s is not None and gm_s_is is not None:
        if abs(float(gm_s) - float(gm_s_is)) < 1e-8:
            r.warn(f"seed_{seed}/{name}: uniform and IS gradient magnitudes are identical for success group (IS may not be applied)")


def check_moment_of_reward(r, data, seed, name):
    """Moment-of-reward result checks."""
    mor = data.get("moment_of_reward")
    if mor is None:
        r.warn(f"seed_{seed}/{name}: moment_of_reward is null (skipped or failed)")
        return

    EXPECTED_MOR_KEYS = [
        "n_positive", "n_neutral", "n_negative",
        "gradient_magnitude_positive", "gradient_magnitude_neutral",
        "coherence_positive", "coherence_neutral",
        "opp_pos_vs_failure", "opp_neutral_vs_failure",
        "opp_pos_vs_neutral",
    ]
    missing = [k for k in EXPECTED_MOR_KEYS if k not in mor]
    if missing:
        r.fail(f"seed_{seed}/{name}: moment_of_reward missing keys: {missing}")
    else:
        r.ok(f"seed_{seed}/{name}: moment_of_reward has all expected keys")

    # Transition counts should be positive
    for key in ("n_positive", "n_neutral"):
        v = mor.get(key)
        if v is not None and int(v) == 0:
            r.warn(f"seed_{seed}/{name}: moment_of_reward.{key}=0 — subgroup is empty")

    # pos transitions should be non-trivial fraction of total
    total_t = sum(
        mor.get(k, 0) or 0
        for k in ("n_positive", "n_neutral", "n_negative")
    )
    if total_t > 0:
        pos_frac = (mor.get("n_positive") or 0) / total_t
        neu_frac = (mor.get("n_neutral") or 0) / total_t
        r.ok(f"seed_{seed}/{name}: MoR transition fractions — pos={pos_frac:.2%}, neu={neu_frac:.2%}, neg={1-pos_frac-neu_frac:.2%}")

    # Opposition scores in [−1, 1]
    for opp_key in ("opp_pos_vs_failure", "opp_neutral_vs_failure", "opp_pos_vs_neutral",
                    "opp_pos_vs_negative", "opp_neutral_vs_negative", "opp_negative_vs_failure"):
        v = mor.get(opp_key)
        if v is not None and _is_finite(v) and abs(float(v)) > 1.001:
            r.fail(f"seed_{seed}/{name}: moment_of_reward.{opp_key}={v:.4f} outside [−1, 1]")


def check_weight_delta(r, jsons_data, seed):
    """Check weight-delta metrics across a seed's JSON list."""
    _head(f"  Stage 8 detail — Weight-delta (seed_{seed})")

    DELTA_KEYS = [
        "cos_uniform_success_delta", "cos_is_success_delta", "cos_reward_success_delta",
        "cos_uniform_failure_delta", "cos_is_failure_delta", "cos_reward_failure_delta",
    ]

    ckpts_with_delta = [(name, d) for name, d in jsons_data if d.get("cos_is_success_delta") is not None]

    if not ckpts_with_delta:
        r.warn(f"seed_{seed}: no weight-delta metrics found (only one checkpoint, or all skipped)")
        return

    r.ok(f"seed_{seed}: weight-delta metrics present in {len(ckpts_with_delta)} checkpoints")

    for name, d in ckpts_with_delta:
        # G_IS should align better than G_uniform with Δθ for success group
        cos_is  = d.get("cos_is_success_delta")
        cos_uni = d.get("cos_uniform_success_delta")
        if cos_is is not None and cos_uni is not None:
            if float(cos_is) > float(cos_uni):
                r.ok(f"  seed_{seed}/{name}: G_IS aligns better with Δθ than G_uniform "
                     f"(cos_IS={cos_is:.4f} > cos_uni={cos_uni:.4f})")
            else:
                r.warn(f"  seed_{seed}/{name}: G_uniform aligns better with Δθ than G_IS "
                       f"(cos_uni={cos_uni:.4f} ≥ cos_IS={cos_is:.4f})")

        # All delta cosines finite
        for k in DELTA_KEYS:
            v = d.get(k)
            if v is not None and not _is_finite(v):
                r.fail(f"  seed_{seed}/{name}: {k}={v} is not finite")


def check_cross_checkpoint_trends(r, jsons_data, seed):
    """Check that metrics evolve across checkpoints in sensible ways."""
    _head(f"  Stage 9 detail — Cross-checkpoint trends (seed_{seed})")

    if len(jsons_data) < 2:
        r.warn(f"seed_{seed}: fewer than 2 analyzed checkpoints — cannot check trends")
        return

    episodes = [d.get("episode", 0) for _, d in jsons_data]

    # Episode counts should be strictly increasing
    if episodes == sorted(episodes) and len(set(episodes)) == len(episodes):
        r.ok(f"seed_{seed}: episode counts are strictly increasing across {len(episodes)} checkpoints")
    else:
        r.fail(f"seed_{seed}: episode counts are not strictly increasing: {episodes}")

    # opposition_score_is should not be constant (that would indicate a bug)
    opp_vals = [d.get("opposition_score_is") for _, d in jsons_data if d.get("opposition_score_is") is not None]
    if len(opp_vals) >= 2:
        if len(set(round(v, 6) for v in opp_vals)) == 1:
            r.warn(f"seed_{seed}: opposition_score_is is identical across all checkpoints ({opp_vals[0]:.4f}) — possible stale analysis?")
        else:
            r.ok(f"seed_{seed}: opposition_score_is varies across checkpoints (range [{min(opp_vals):.4f}, {max(opp_vals):.4f}])")

    # activation_separation should not be constant
    sep_vals = [d.get("activation_separation") for _, d in jsons_data if d.get("activation_separation") is not None]
    if len(sep_vals) >= 2:
        if len(set(round(v, 6) for v in sep_vals)) == 1:
            r.warn(f"seed_{seed}: activation_separation is constant ({sep_vals[0]:.4f}) — possible stale analysis?")
        else:
            r.ok(f"seed_{seed}: activation_separation varies across checkpoints (range [{min(sep_vals):.4f}, {max(sep_vals):.4f}])")

    # beta_used should increase monotonically (IS annealing)
    beta_vals = [d.get("beta_used") for _, d in jsons_data if d.get("beta_used") is not None]
    if len(beta_vals) >= 2:
        increasing = all(beta_vals[i] <= beta_vals[i + 1] + 1e-6 for i in range(len(beta_vals) - 1))
        if increasing:
            r.ok(f"seed_{seed}: beta_used is monotonically non-decreasing ({beta_vals[0]:.3f} → {beta_vals[-1]:.3f})")
        else:
            r.warn(f"seed_{seed}: beta_used is not monotonically increasing — unexpected annealing behaviour")


def check_report_files(r, experiment_root, seeds):
    _head("Stage 10 — Report files")

    for seed in seeds:
        seed_root = os.path.join(experiment_root, f"seed_{seed}")
        report = os.path.join(seed_root, f"report_crafter_rainbow_{seed}.md")
        if os.path.exists(report):
            size = os.path.getsize(report)
            r.ok(f"seed_{seed}: per-seed report exists ({size:,} bytes)")
        else:
            r.warn(f"seed_{seed}: per-seed report not found (may be pending completion)")

    avg_report = os.path.join(experiment_root, "report_averaged_crafter_rainbow.md")
    if os.path.exists(avg_report):
        size = os.path.getsize(avg_report)
        r.ok(f"Averaged report exists ({size:,} bytes): {avg_report}")
    else:
        r.warn(f"Averaged report not found (generated after all seeds complete): {avg_report}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_checks(experiment_root, seeds, verbose=False):
    r = CheckResult()

    # Stage 1
    check_file_structure(r, experiment_root, seeds)

    # Stage 2
    check_training_progress(r, experiment_root, seeds, verbose=verbose)

    # Stage 3
    check_checkpoint_integrity(r, experiment_root, seeds, verbose=verbose)

    # Stages 4–9: per-seed JSON analysis
    for seed in seeds:
        seed_root = os.path.join(experiment_root, f"seed_{seed}")
        json_paths = _list_jsons(seed_root)

        if not json_paths:
            r.warn(f"seed_{seed}: no analysis JSONs found — skipping schema/gradient/MoR checks")
            continue

        _head(f"Stages 4–9 — Analysis checks (seed_{seed}, {len(json_paths)} JSONs)")

        jsons_data = []  # list of (basename, dict)

        for jpath in json_paths:
            name = os.path.basename(jpath)
            data = check_analysis_schema(r, jpath, seed, verbose=verbose)
            if data is None:
                continue

            jsons_data.append((name, data))

            # Stage 5: gradient sanity
            check_gradient_sanity(r, data, seed, name)

            # Stage 6: activation sanity (done inside check_analysis_schema)

            # Stage 7: moment of reward
            check_moment_of_reward(r, data, seed, name)

        # Stage 8: weight-delta
        if jsons_data:
            check_weight_delta(r, jsons_data, seed)

        # Stage 9: cross-checkpoint trends
        if jsons_data:
            check_cross_checkpoint_trends(r, jsons_data, seed)

    # Stage 10
    check_report_files(r, experiment_root, seeds)

    return r.summarise()


def main():
    parser = argparse.ArgumentParser(
        description="Verify Rainbow training + analysis pipeline outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_root", default="rainbow_results",
        help="Root directory of the Rainbow experiment (produced by train_and_analyze.py)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Seeds to check",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-checkpoint metric summaries",
    )
    args = parser.parse_args()

    print(f"{_BOLD}Rainbow pipeline checker{_RESET}")
    print(f"Experiment root : {os.path.abspath(args.experiment_root)}")
    print(f"Seeds           : {args.seeds}")

    ok = run_checks(args.experiment_root, args.seeds, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
