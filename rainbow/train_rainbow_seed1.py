"""Resume Rainbow training for seed 1 from checkpoint_live.pt to 3M steps.

Seed 1's checkpoint_live.pt is at ~1.5M steps. This script resumes from
that checkpoint and trains to 3M steps, overwriting checkpoint_live.pt
when done. All hyperparameters match the original run.

Usage (from PPOSAC-tracker/):
    python train_rainbow_seed1.py
    python train_rainbow_seed1.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--experiment_root", default="rainbow_experiment_root/seed_1",
        help="Seed-1 experiment dir (default: rainbow_experiment_root/seed_1)",
    )
    parser.add_argument(
        "--T_max", type=int, default=3_000_000,
        help="Total training steps (default: 3M)",
    )
    args = parser.parse_args()

    exp_root  = os.path.join(_ROOT, args.experiment_root)
    ckpt_path = os.path.join(exp_root, "checkpoints", "rainbow", "checkpoint_live.pt")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    current_step = ckpt.get("global_step", 0)
    print(f"Resuming seed 1 from step {current_step:,} → {args.T_max:,}")

    if current_step >= args.T_max:
        print(f"[info] Already at {current_step:,} steps — nothing to do.")
        return

    from rainbow.train import build_parser, main_rainbow

    rb_parser = build_parser()
    rb_args   = rb_parser.parse_args([])

    # Match original run hyperparameters
    rb_args.seed                = 1
    rb_args.T_max               = args.T_max
    rb_args.experiment_root     = exp_root
    rb_args.model               = ckpt_path          # resume from here
    rb_args.disable_cuda        = (args.device == "cpu")
    rb_args.checkpoint_interval = 50_000             # save every 50k steps

    # All other hyperparams stay at build_parser() defaults, which match the
    # original training run (canonical architecture, 512 hidden, 51 atoms, etc.)

    print(f"Experiment root : {exp_root}")
    print(f"Resuming from   : {ckpt_path}")
    print(f"Device          : {args.device}")
    print(f"Steps remaining : {args.T_max - current_step:,}")
    print()

    main_rainbow(rb_args)

    final_ckpt = os.path.join(exp_root, "checkpoints", "rainbow", "checkpoint_live.pt")
    final      = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    print(f"\nDone. checkpoint_live.pt is now at step {final.get('global_step', '?'):,}")


if __name__ == "__main__":
    main()
