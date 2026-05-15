"""One-off MORA subsampling variance check.

Runs MoR 3x at two checkpoints with independent random seeds and reports
the coefficient of variation (CV = std/mean) of the MORA ratio
(gradient_magnitude_positive / gradient_magnitude_neutral).

Council acceptance criterion: CV < 0.15 (15%) at both checkpoints.

Usage:
    python scripts/mora_variance_check.py \\
        --ckpt1 rainbow_v2/seed_1/checkpoints/rainbow/checkpoint_step1500000.pt \\
        --ckpt2 rainbow_v2/seed_1/checkpoints/rainbow/checkpoint_step3000000.pt \\
        --n_episodes 1000 --device cuda
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mora_ratio_one_run(ckpt_path, n_episodes, device, rng_seed):
    """Run MoR on success_eps with a specific subsampling RNG seed; return MORA ratio."""
    import rainbow.moment_of_reward as _mor_mod

    # Monkey-patch np.random.default_rng inside the MoR module to vary the subsample.
    _orig_rng = _mor_mod.np.random.default_rng

    def _seeded(seed=None):
        return _orig_rng(rng_seed)

    _mor_mod.np.random.default_rng = _seeded
    try:
        from rainbow.sampling import evaluate_frozen_policy, load_rainbow_nets, partition
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        from rainbow.gradients import compute_group_gradient_with_coherence

        episodes, eps_scores, _ = evaluate_frozen_policy(
            ckpt_path, n_episodes=n_episodes, device=device, seed=1,
        )
        success_eps, failure_eps, _ = partition(episodes, eps_scores)
        if not success_eps or not failure_eps:
            print("  [warn] empty partition — skipping run")
            return None

        online_net, target_net, _, _, args_ns = load_rainbow_nets(ckpt_path, device=device)

        grad_f = compute_group_gradient_with_coherence(
            online_net, failure_eps,
            batch_size=max(5, len(failure_eps) // 10),
            device=device, desc="Grads [failure]",
            target_net=target_net, args_ns=args_ns, global_step=0,
        )
        raw_failure = grad_f["uniform"]["raw"]

        mor = run_moment_of_reward_analysis(
            online_net, success_eps, raw_failure_grad=raw_failure,
            device=device, target_net=target_net, args_ns=args_ns,
            max_episodes=50, track_coherence=False,
        )
        if mor is None:
            return None

        gm_pos = mor.get("gradient_magnitude_positive")
        gm_neu = mor.get("gradient_magnitude_neutral")
        if gm_pos is None or gm_neu is None or gm_neu == 0:
            return None
        return gm_pos / gm_neu

    finally:
        _mor_mod.np.random.default_rng = _orig_rng


def _check_one_ckpt(ckpt_path, n_episodes, device, label):
    print(f"\n--- {label}: {os.path.basename(ckpt_path)} ---")
    ratios = []
    for run_idx, seed in enumerate([11, 22, 33]):
        print(f"  run {run_idx + 1}/3 (rng_seed={seed}) ...", flush=True)
        r = _mora_ratio_one_run(ckpt_path, n_episodes, device, seed)
        if r is not None:
            ratios.append(r)
            print(f"    MORA ratio = {r:.4f}")
        else:
            print("    MORA ratio = None (skipped)")

    if len(ratios) < 2:
        print("  Insufficient valid runs — cannot compute variance")
        return

    arr  = np.array(ratios)
    mean = arr.mean()
    std  = arr.std(ddof=1)
    cv   = std / mean if mean != 0 else float("inf")
    print(f"\n  Ratios : {[f'{r:.4f}' for r in ratios]}")
    print(f"  Mean   : {mean:.4f}")
    print(f"  Std    : {std:.4f}")
    print(f"  CV     : {cv:.4f}  ({'PASS' if cv < 0.15 else 'FAIL'} -- threshold 0.15)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt1", required=True,
                        help="Early checkpoint path (e.g. checkpoint_step1500000.pt)")
    parser.add_argument("--ckpt2", required=True,
                        help="Late checkpoint path (e.g. checkpoint_step3000000.pt)")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _check_one_ckpt(args.ckpt1, args.n_episodes, args.device, "Early checkpoint")
    _check_one_ckpt(args.ckpt2, args.n_episodes, args.device, "Late checkpoint")


if __name__ == "__main__":
    main()
