"""Counterfactual gradient validation — Experiment 3.

Trains a fresh Rainbow run with gradient capture enabled.  At each periodic
checkpoint, the mean training gradient G_training (accumulated over all learn()
steps since the previous checkpoint) is stored in the checkpoint file.

After training, this script computes the offline counterfactual gradient
G_counterfactual on fresh evaluation episodes for each checkpoint, then reports:

    cos(G_training, G_counterfactual_uniform)
    cos(G_training, G_counterfactual_IS)

Interpretation:
  cosine consistently > 0.5  → counterfactual framework validated; the analysis
                               measures the same gradient direction as training.
  cosine consistently < 0.2  → analysis measures a different object from training;
                               the entire counterfactual framing needs caveating.
  cosine intermediate / mixed → partial overlap; report the distribution, not
                               just a single checkpoint value.

Usage (from PPOSAC-tracker/):
    # Fresh training + validation for seed 1:
    python run_gradient_validation.py --seed 1 --T_max 3000000

    # Validation only (training already done, checkpoints have training_gradient):
    python run_gradient_validation.py --seed 1 --skip_training \\
        --experiment_root grad_validation_root

    # Use existing Rainbow experiment (must have been run with capture_training_grads):
    python run_gradient_validation.py --seed 1 --skip_training \\
        --experiment_root rainbow_experiment_root

Notes:
  - Training gradient is ONLY stored in checkpoint_step*.pt files, NOT in
    checkpoint_live.pt (which is overwritten without gradient accumulation).
  - --checkpoint_interval controls how often checkpoints (and gradients) are saved.
    Default 500_000 gives 6 checkpoints over a 3M-step validation run.
  - The offline gradient uses the SAME episode sampling as analyze_checkpoint.py,
    so the comparison is: training-buffer gradient vs fresh-episode gradient.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch



def _load_training_gradient(ckpt_path: str) -> dict | None:
    """Load stored training_gradient from a checkpoint; return None if absent."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tg = ckpt.get("training_gradient")
    if tg is None:
        return None
    # Tensors may already be on CPU; ensure float32
    return {k: v.float() for k, v in tg.items()}


def _cosine_flat(g_a: dict, g_b: dict) -> float | None:
    """Compute cosine similarity between two gradient dicts (flattened)."""
    if g_a is None or g_b is None:
        return None
    from shared.gradient_utils import cosine_similarity_flat
    # Only compare keys present in both (in case sigma params differ)
    common = sorted(set(g_a) & set(g_b))
    if not common:
        return None
    a = torch.cat([g_a[k].flatten() for k in common])
    b = torch.cat([g_b[k].flatten() for k in common])
    import torch.nn.functional as F
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()



def _compute_counterfactual_gradients(
    ckpt_path: str,
    n_episodes: int,
    device: str,
    seed: int,
    percentile_x: int,
) -> dict | None:
    """Compute offline G_uniform and G_IS for the given checkpoint.

    Returns dict with keys "uniform" and "is_weighted" (each a raw gradient dict),
    or None if one group is empty.
    """
    from rainbow.sampling import evaluate_frozen_policy, load_rainbow_nets, partition
    from rainbow.gradients import compute_group_gradient_with_coherence
    from analyze_checkpoint import _compute_joint_is_weights

    episodes, eps_scores, _ = evaluate_frozen_policy(
        ckpt_path, n_episodes=n_episodes, device=device, seed=seed
    )
    success_eps, failure_eps, _ = partition(
        episodes, eps_scores, mode="percentile", percentile_x=percentile_x
    )
    if not success_eps or not failure_eps:
        return None

    online_net, target_net, _, global_step, args_ns = load_rainbow_nets(ckpt_path, device)

    joint_is = _compute_joint_is_weights(
        online_net, target_net, success_eps + failure_eps, args_ns, global_step, device
    )
    is_w_s = joint_is[:len(success_eps)]
    is_w_f = joint_is[len(success_eps):]

    s_batch = max(5, len(success_eps) // 10)
    f_batch = max(5, len(failure_eps) // 10)

    grad_s = compute_group_gradient_with_coherence(
        online_net, success_eps, batch_size=s_batch, device=device,
        desc="Grads [success]", target_net=target_net, args_ns=args_ns,
        global_step=global_step, precomputed_is_weights=is_w_s,
    )
    grad_f = compute_group_gradient_with_coherence(
        online_net, failure_eps, batch_size=f_batch, device=device,
        desc="Grads [failure]", target_net=target_net, args_ns=args_ns,
        global_step=global_step, precomputed_is_weights=is_w_f,
    )

    # Return both success and failure groups for comparison
    return {
        "success": {
            "uniform":     grad_s["uniform"]["raw"],
            "is_weighted": grad_s["is_weighted"]["raw"],
        },
        "failure": {
            "uniform":     grad_f["uniform"]["raw"],
            "is_weighted": grad_f["is_weighted"]["raw"],
        },
        "global_step": global_step,
    }



def _validate_seed(
    seed: int,
    exp_root: str,
    out_root: str,
    n_episodes: int,
    percentile_x: int,
    device: str,
) -> list[dict]:
    seed_dir = os.path.join(exp_root, f"seed_{seed}")
    ckpts    = sorted(
        glob.glob(os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_step*.pt"))
    )

    if not ckpts:
        print(f"  [seed {seed}] No periodic checkpoints found — skipping")
        return []

    results = []
    for ckpt in ckpts:
        name         = os.path.splitext(os.path.basename(ckpt))[0]
        training_grad = _load_training_gradient(ckpt)

        if training_grad is None:
            print(f"  {name}: no training_gradient stored — was capture_training_grads enabled?")
            results.append({"checkpoint": name, "error": "no_training_gradient"})
            continue

        print(f"  {name}: computing offline gradient ...")
        try:
            cf = _compute_counterfactual_gradients(
                ckpt, n_episodes=n_episodes, device=device,
                seed=seed, percentile_x=percentile_x,
            )
        except Exception as e:
            import traceback
            print(f"    [ERROR] {e}")
            traceback.print_exc()
            results.append({"checkpoint": name, "error": str(e)})
            continue

        if cf is None:
            results.append({"checkpoint": name, "error": "empty_group"})
            continue

        step = cf["global_step"]
        row = {
            "checkpoint":  name,
            "global_step": step,
            # cos(G_training, G_counterfactual) for success and failure groups
            "cos_train_success_uniform": _cosine_flat(
                training_grad, cf["success"]["uniform"]
            ),
            "cos_train_success_IS": _cosine_flat(
                training_grad, cf["success"]["is_weighted"]
            ),
            "cos_train_failure_uniform": _cosine_flat(
                training_grad, cf["failure"]["uniform"]
            ),
            "cos_train_failure_IS": _cosine_flat(
                training_grad, cf["failure"]["is_weighted"]
            ),
        }
        results.append(row)
        print(
            f"    cos(G_train, G_success_unif) = {row['cos_train_success_uniform']:.4f}  "
            f"cos(G_train, G_success_IS) = {row['cos_train_success_IS']:.4f}"
        )

    # Save per-seed results
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"grad_validation_seed_{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None)
    print(f"  Saved: {out_path}")
    return results



def _print_summary(all_results: dict[int, list[dict]]):
    print("\n" + "=" * 80)
    print("  COUNTERFACTUAL GRADIENT VALIDATION — SUMMARY")
    print("=" * 80)

    all_cos_s_u, all_cos_s_i = [], []
    all_cos_f_u, all_cos_f_i = [], []

    for seed, rows in all_results.items():
        print(f"\n  Seed {seed}:")
        print(f"  {'Checkpoint':>25}  {'cos(G_t,G_s_U)':>14}  {'cos(G_t,G_s_IS)':>15}  "
              f"{'cos(G_t,G_f_U)':>14}  {'cos(G_t,G_f_IS)':>15}")
        print(f"  {'-'*25}  {'-'*14}  {'-'*15}  {'-'*14}  {'-'*15}")
        for row in rows:
            if "error" in row:
                print(f"  {row['checkpoint']:>25}  [ERROR: {row['error']}]")
                continue
            def _f(v):
                return f"{v:.4f}" if v is not None else "  N/A"
            cu  = row.get("cos_train_success_uniform")
            ci  = row.get("cos_train_success_IS")
            cfu = row.get("cos_train_failure_uniform")
            cfi = row.get("cos_train_failure_IS")
            print(f"  {row['checkpoint']:>25}  {_f(cu):>14}  {_f(ci):>15}  {_f(cfu):>14}  {_f(cfi):>15}")
            if cu  is not None: all_cos_s_u.append(cu)
            if ci  is not None: all_cos_s_i.append(ci)
            if cfu is not None: all_cos_f_u.append(cfu)
            if cfi is not None: all_cos_f_i.append(cfi)

    print(f"\n  Cross-seed means:")
    def _stat(vals, label):
        if not vals:
            print(f"    {label}: N/A")
        else:
            print(f"    {label}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    _stat(all_cos_s_u,  "cos(G_train, G_success_uniform)")
    _stat(all_cos_s_i,  "cos(G_train, G_success_IS)    ")
    _stat(all_cos_f_u,  "cos(G_train, G_failure_uniform)")
    _stat(all_cos_f_i,  "cos(G_train, G_failure_IS)    ")

    # Verdict
    if all_cos_s_u:
        mean_cos = np.mean(all_cos_s_u)
        if mean_cos > 0.5:
            verdict = ("VALIDATED — mean cosine > 0.5.  The counterfactual gradient "
                       "points in the same direction as the training gradient.  "
                       "The gradient analysis measures what it claims to measure.")
        elif mean_cos > 0.2:
            verdict = ("PARTIAL OVERLAP — mean cosine 0.2–0.5.  Counterfactual and "
                       "training gradients share orientation but differ in magnitude/composition.  "
                       "Report the cosine as a calibration caveat.")
        else:
            verdict = ("LOW ALIGNMENT — mean cosine < 0.2.  The counterfactual gradient "
                       "does not reliably approximate the training gradient.  "
                       "CRITICAL CAVEAT REQUIRED: analysis may not reflect the "
                       "actual gradient that drives learning.")
        print(f"\n  VERDICT: {verdict}")

    print("=" * 80 + "\n")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default="grad_validation_root",
                        help="Root of the Rainbow experiment with grad capture enabled")
    parser.add_argument("--output_root", default="grad_validation_results",
                        help="Where to save per-checkpoint cosine JSON files")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="Seeds to validate (default: 1 — one seed is sufficient)")
    parser.add_argument("--T_max", type=int, default=3_000_000,
                        help="Training steps (only used when --skip_training is NOT set)")
    parser.add_argument("--checkpoint_interval", type=int, default=500_000,
                        help="Steps between checkpoints (default: 500k → 6 checkpoints over 3M)")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--percentile_x", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    # Rainbow hyperparameters (used only when training)
    parser.add_argument("--hidden_size",      type=int,   default=512)
    parser.add_argument("--architecture",     default="canonical")
    parser.add_argument("--memory_capacity",  type=int,   default=500_000)
    parser.add_argument("--learning_rate",    type=float, default=0.0000625)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training; load checkpoints from --experiment_root directly")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    exp_root = _abs(args.experiment_root)
    out_root = _abs(args.output_root)


    if not args.skip_training:
        from rainbow.train import build_parser, main_rainbow

        for seed in args.seeds:
            seed_root = os.path.join(exp_root, f"seed_{seed}")
            live_ckpt = os.path.join(seed_root, "checkpoints", "rainbow", "checkpoint_live.pt")

            if os.path.exists(live_ckpt):
                print(f"[seed {seed}] Training checkpoint exists — skipping training.")
                print(f"  (Delete {live_ckpt} to retrain)")
            else:
                print(f"\n{'='*60}")
                print(f"=== Training seed {seed} with gradient capture ===")
                print(f"{'='*60}")

                parser_rb = build_parser()
                rb_args = parser_rb.parse_args([])
                rb_args.seed                = seed
                rb_args.T_max               = args.T_max
                rb_args.checkpoint_interval = args.checkpoint_interval
                rb_args.experiment_root     = seed_root
                rb_args.hidden_size         = args.hidden_size
                rb_args.architecture        = args.architecture
                rb_args.memory_capacity     = args.memory_capacity
                rb_args.learning_rate       = args.learning_rate
                rb_args.batch_size          = args.batch_size
                rb_args.disable_cuda        = (args.device == "cpu")
                rb_args.model               = None

                main_rainbow(
                    rb_args,
                    on_checkpoint_saved=None,
                    capture_training_grads=True,   # KEY: enables gradient capture
                )


    all_results = {}
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"=== Validating seed {seed}")
        print(f"{'='*60}")
        rows = _validate_seed(
            seed=seed,
            exp_root=exp_root,
            out_root=out_root,
            n_episodes=args.n_episodes,
            percentile_x=args.percentile_x,
            device=args.device,
        )
        all_results[seed] = rows


    _print_summary(all_results)
    print(f"Validation results in: {out_root}")


if __name__ == "__main__":
    main()
