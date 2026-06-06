"""End-to-end Rainbow training + corrected analysis pipeline.

Trains Rainbow DQN from scratch (or resumes) for all seeds, running the corrected
analysis inline at every checkpoint_interval steps. All checkpoint_step{N}.pt files
are kept on disk (keep_checkpoints=True) so the post-training longitudinal phases
(D, H, I) can re-run analysis across the full training trajectory.

After all seeds are trained, runs post-training robustness phases:
  C - Scalar DQN ablation (MORA ratio gate test)
  D - Fixed-threshold longitudinal analysis (all 50k-step checkpoints)
  F - Robustness CI report (all inline analysis JSONs)
  H - Frozen-RSA longitudinal analysis (all 50k-step checkpoints)
  I - EPS sensitivity longitudinal analysis (all 50k-step checkpoints)

Checkpoint lifecycle
--------------------
  checkpoint_live.pt              - overwritten in-place each interval; used for resume
  checkpoint_step{N}.pt           - saved at every checkpoint_interval; kept permanently
  milestones/milestone_first_*.pt - permanent copy of each first-achievement checkpoint

Output layout (all under --experiment_root, default rainbow_v2/)
----------------------------------------------------------------
  seed_N/
    checkpoints/rainbow/checkpoint_live.pt
    checkpoints/rainbow/checkpoint_step50000.pt
    checkpoints/rainbow/checkpoint_step100000.pt
    ...  (60 files at 3M / 50k)
    analysis_logs/rainbow/checkpoint_step50000.json
    analysis_logs/rainbow/checkpoint_step100000.json
    ...
    logs/rainbowreturnlog.txt
  scalar_ablation/
    seed_1/checkpoints/scalar_dqn/checkpoint_live.pt
    scalar_ablation_result.json
  fixed_threshold/
    fixed_thresholds.json
    seed_N/analysis_logs/rainbow/checkpoint_step{N}.json
  frozen_rsa/
    seed_N/analysis_logs/rainbow/checkpoint_step{N}.json
  eps_sensitivity/
    w{tag}/seed_N/analysis_logs/rainbow/checkpoint_step{N}.json

Disk estimate: ~60 checkpoints x ~250 MB x 5 seeds ≈ 75 GB for .pt files alone.

Usage (from PPOSAC-tracker/)
-----------------------------
  python run_rainbow_full_pipeline.py --device cuda
  python run_rainbow_full_pipeline.py --device cuda --seeds 1 2
  python run_rainbow_full_pipeline.py --skip_training --device cuda
  python run_rainbow_full_pipeline.py --skip C D

Smoke test (~2 min on GPU)
  python run_rainbow_full_pipeline.py \\
      --T_max 50000 --checkpoint_interval 20000 --n_episodes 10 --seeds 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

PIPELINE_VERSION = "rainbow_full_v1"



def _notify(title: str, message: str):
    """Send a Windows toast notification. Silent no-op if PowerShell unavailable."""
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
        " ContentType=WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument,"
        " ContentType=WindowsRuntime] | Out-Null;"
        "$t = [Windows.UI.Notifications.ToastTemplateType]::ToastText02;"
        "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($t);"
        f"$x.SelectSingleNode('//text[@id=1]').InnerText = '{title}';"
        f"$x.SelectSingleNode('//text[@id=2]').InnerText = '{message}';"
        "$n = [Windows.UI.Notifications.ToastNotification]::new($x);"
        "[Windows.UI.Notifications.ToastNotificationManager]"
        "::CreateToastNotifier('Rainbow Pipeline').Show($n)"
    )
    try:
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _header(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _inject_version(json_path: str):
    """Add pipeline_version tag to an analysis JSON in-place."""
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path) as f:
            d = json.load(f)
        d["pipeline_version"] = PIPELINE_VERSION
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2, default=str)
    except Exception as e:
        print(f"  [warn] Could not inject version tag into {json_path}: {e}")


def _achievement_from_path(path: str) -> str | None:
    """Extract achievement name from milestone_first_{achievement}_ep{N}.pt/.json."""
    m = re.match(r"milestone_first_(.+)_ep\d+", os.path.splitext(os.path.basename(path))[0])
    return m.group(1) if m else None


def _episode_from_path(path: str) -> int:
    """Extract episode number from milestone_first_{achievement}_ep{N}.pt/.json."""
    m = re.search(r"_ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _load_seen_achievements(seed_root: str) -> set[str]:
    """Reconstruct seen_achievements from milestone files on disk.

    Scans both milestones/*.pt and analysis_logs/rainbow/milestone_*.json so
    achievements are recovered even if the .pt was not yet copied (e.g. early
    JSONs from before milestone saving was added).
    """
    seen: set[str] = set()
    milestones_dir = os.path.join(seed_root, "checkpoints", "rainbow", "milestones")
    for path in glob.glob(os.path.join(milestones_dir, "milestone_first_*.pt")):
        ach = _achievement_from_path(path)
        if ach:
            seen.add(ach)
    json_dir = os.path.join(seed_root, "analysis_logs", "rainbow")
    for path in glob.glob(os.path.join(json_dir, "milestone_first_*.json")):
        ach = _achievement_from_path(path)
        if ach:
            seen.add(ach)
    return seen


def _dedup_milestones(seed_root: str):
    """Remove duplicate milestone files, keeping the earliest episode per achievement.

    Operates on both milestones/*.pt and analysis_logs/rainbow/milestone_*.json.
    Duplicates arise when training resumes without seen_achievements, causing
    already-seen achievements to re-trigger.
    """
    def _dedup_dir(directory: str, ext: str):
        by_achievement: dict[str, list[str]] = {}
        for path in glob.glob(os.path.join(directory, f"milestone_first_*{ext}")):
            ach = _achievement_from_path(path)
            if ach:
                by_achievement.setdefault(ach, []).append(path)
        removed = 0
        for ach, paths in by_achievement.items():
            if len(paths) <= 1:
                continue
            paths.sort(key=_episode_from_path)
            for dup in paths[1:]:   # keep lowest episode, delete the rest
                try:
                    os.remove(dup)
                    print(f"    [dedup] removed {os.path.basename(dup)}")
                    removed += 1
                except OSError as e:
                    print(f"    [dedup] could not remove {dup}: {e}")
        return removed

    milestones_dir = os.path.join(seed_root, "checkpoints", "rainbow", "milestones")
    json_dir       = os.path.join(seed_root, "analysis_logs", "rainbow")
    n  = _dedup_dir(milestones_dir, ".pt")
    n += _dedup_dir(json_dir, ".json")
    if n:
        print(f"  [dedup] removed {n} duplicate milestone file(s)")


def _find_step_checkpoints(seed_dir: str, stride: int = 1) -> list[str]:
    """Return checkpoint_step*.pt files in ascending step order.

    stride=1  → every checkpoint (default, used for inline analysis)
    stride=N  → every Nth checkpoint (e.g. stride=10 keeps every 500k with 50k interval)
    """
    pattern = os.path.join(seed_dir, "checkpoints", "rainbow", "checkpoint_step*.pt")
    paths = sorted(glob.glob(pattern))
    if stride > 1:
        paths = paths[stride - 1 :: stride]
    return paths



def _build_rainbow_args(seed: int, seed_root: str, args: argparse.Namespace,
                        live_ckpt: str) -> argparse.Namespace:
    from rainbow.train import build_parser
    parser = build_parser()
    ns = parser.parse_args([])
    ns.seed                = seed
    ns.T_max               = args.T_max
    ns.checkpoint_interval = args.checkpoint_interval
    ns.experiment_root     = seed_root
    ns.disable_cuda        = (args.device == "cpu")
    ns.model               = live_ckpt if os.path.exists(live_ckpt) else None
    ns.keep_checkpoints    = True   # retain all checkpoint_step*.pt for longitudinal phases
    return ns



def _run_seed(seed: int, args: argparse.Namespace, exp_root: str):
    from analyze_checkpoint import analyze_checkpoint
    from rainbow.train import main_rainbow
    import torch

    seed_root = os.path.join(exp_root, f"seed_{seed}")
    live_ckpt = os.path.join(seed_root, "checkpoints", "rainbow", "checkpoint_live.pt")

    # Skip only if training is complete AND at least one analysis JSON exists
    if os.path.exists(live_ckpt):
        ckpt_data = torch.load(live_ckpt, map_location="cpu", weights_only=False)
        if ckpt_data.get("global_step", 0) >= args.T_max:
            n_jsons = len(glob.glob(os.path.join(
                seed_root, "analysis_logs", "rainbow", "checkpoint_step*.json"
            )))
            if n_jsons > 0:
                print(f"  [seed {seed}] already complete ({n_jsons} JSONs) - skip training")
                return
            else:
                print(f"  [seed {seed}] training complete but no analysis JSONs - "
                      "skipping training; post phases will still run")
                return

    # Clean up duplicate milestone files from previous crash-resume cycles
    # BEFORE loading seen_achievements so the set reflects the canonical (earliest) files.
    _dedup_milestones(seed_root)

    # Reconstruct seen_achievements from surviving milestone files so already-triggered
    # achievements are not re-triggered on resume.
    seen = _load_seen_achievements(seed_root)
    if seen:
        print(f"  [resume] Restoring {len(seen)} seen achievement(s): {sorted(seen)}")

    # Back-fill analysis for any milestone .pt that has no matching .json.
    # This handles the case where training was interrupted after the .pt was
    # copied but before (or during) analyze_checkpoint().
    milestones_dir = os.path.join(seed_root, "checkpoints", "rainbow", "milestones")
    json_dir       = os.path.join(seed_root, "analysis_logs", "rainbow")
    for pt_path in sorted(glob.glob(os.path.join(milestones_dir, "milestone_first_*.pt"))):
        basename  = os.path.splitext(os.path.basename(pt_path))[0]
        json_path = os.path.join(json_dir, f"{basename}.json")
        if not os.path.exists(json_path):
            print(f"  [back-fill] analysing {basename} (pt exists, json missing)")
            try:
                analyze_checkpoint(
                    algorithm="rainbow",
                    checkpoint_path=pt_path,
                    experiment_root=seed_root,
                    n_episodes=args.n_episodes,
                    device=args.device,
                    reason="milestone",
                    seed=seed,
                )
                _inject_version(json_path)
            except Exception:
                traceback.print_exc()

    rb_args = _build_rainbow_args(seed, seed_root, args, live_ckpt)

    def on_checkpoint(path: str, prev_path: str | None = None,
                      is_milestone: bool = False):
        reason   = "milestone" if is_milestone else "corrected_inline"
        basename = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(
            seed_root, "analysis_logs", "rainbow", f"{basename}.json"
        )

        if os.path.exists(json_path):
            print(f"    [skip] {basename} already analysed")
        else:
            print(f"    [analyse] {basename} (reason={reason})")
            try:
                analyze_checkpoint(
                    algorithm="rainbow",
                    checkpoint_path=path,
                    experiment_root=seed_root,
                    n_episodes=args.n_episodes,
                    device=args.device,
                    reason=reason,
                    seed=seed,
                    prev_checkpoint_path=prev_path,
                )
                _inject_version(json_path)
            except Exception:
                traceback.print_exc()

        # Permanently copy milestone .pt files before train.py deletes them.
        # The deletion happens after this callback returns, so the file is still
        # on disk here. Periodic checkpoints are already kept via keep_checkpoints=True.
        if is_milestone:
            import shutil
            milestones_dir = os.path.join(seed_root, "checkpoints", "rainbow", "milestones")
            os.makedirs(milestones_dir, exist_ok=True)
            dest = os.path.join(milestones_dir, os.path.basename(path))
            if not os.path.exists(dest):
                shutil.copy2(path, dest)
                print(f"    [milestone saved] {os.path.basename(dest)}")

    print(f"\n  Training Rainbow seed={seed}, T_max={args.T_max:,}, "
          f"checkpoint_interval={args.checkpoint_interval:,}")
    if rb_args.model:
        print(f"  Resuming from: {rb_args.model}")

    main_rainbow(rb_args, on_checkpoint_saved=on_checkpoint, seen_achievements=seen)
    print(f"  Seed {seed} training complete.")



def _run_phase_c(args: argparse.Namespace, exp_root: str):
    from run_scalar_ablation import (
        _build_rainbow_args as _build_scalar_args,
        _load_rainbow_mora_ratio, _print_verdict,
    )
    from rainbow.scalar_dqn import train_scalar_dqn, compute_scalar_mora_ratio

    seed         = args.seeds[0] if args.seeds else 1
    scalar_root  = os.path.join(exp_root, "scalar_ablation")
    seed_root_sc = os.path.join(scalar_root, f"seed_{seed}")
    os.makedirs(seed_root_sc, exist_ok=True)

    live_ckpt = os.path.join(seed_root_sc, "checkpoints", "scalar_dqn",
                             "checkpoint_live.pt")
    if os.path.exists(live_ckpt):
        print(f"  [info] Scalar checkpoint exists - skipping training\n         {live_ckpt}")
        ckpt_path = live_ckpt
    else:
        print(f"  Training ScalarDQN seed={seed}, T_max={args.T_max:,} ...")
        sc_args = _build_scalar_args(
            experiment_root=seed_root_sc,
            T_max=args.T_max,
            hidden_size=512,
            architecture="canonical",
            memory_capacity=500_000,
            device=args.device,
        )
        sc_args.seed = seed
        ckpt_path = train_scalar_dqn(sc_args, seed=seed, out_root=seed_root_sc)

    print("\n  Computing Scalar DQN MORA ratio ...")
    scalar_result = compute_scalar_mora_ratio(
        ckpt_path, n_episodes=args.n_episodes, device=args.device, seed=seed,
    )

    # Use the final-step inline Rainbow JSON as reference
    ref_json = os.path.join(
        exp_root, f"seed_{seed}", "analysis_logs", "rainbow",
        f"checkpoint_step{args.T_max}.json",
    )
    if not os.path.exists(ref_json):
        candidates = sorted(glob.glob(os.path.join(
            exp_root, f"seed_{seed}", "analysis_logs", "rainbow",
            "checkpoint_step*.json",
        )))
        ref_json = candidates[-1] if candidates else None
        if ref_json:
            print(f"  [warn] Final step JSON missing - using {os.path.basename(ref_json)}")

    rainbow_ref = _load_rainbow_mora_ratio(ref_json) if ref_json else None
    if rainbow_ref is None:
        print("  [warn] No Rainbow reference JSON - comparison unavailable")

    out_path = os.path.join(scalar_root, "scalar_ablation_result.json")
    with open(out_path, "w") as f:
        json.dump(
            {"scalar_dqn": scalar_result, "rainbow_reference": rainbow_ref,
             "pipeline_version": PIPELINE_VERSION},
            f, indent=2,
            default=lambda x: None if (isinstance(x, float) and x != x) else str(x),
        )
    print(f"  Saved: {out_path}")
    _print_verdict(scalar_result, rainbow_ref)



def _run_phase_d(args: argparse.Namespace, exp_root: str, stride: int = 1):
    from experiments.run_fixed_threshold_analysis import _derive_thresholds_pooled
    from analyze_checkpoint import analyze_checkpoint

    out_root = os.path.join(exp_root, "fixed_threshold")
    os.makedirs(out_root, exist_ok=True)

    print("  Deriving pooled EPS thresholds from checkpoint_live.pt across all seeds ...")
    try:
        fixed_thresholds = _derive_thresholds_pooled(
            exp_root=exp_root,
            seeds=args.seeds,
            n_episodes=args.n_episodes,
            percentile_x=25,
            device=args.device,
        )
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return

    thresh_path = os.path.join(out_root, "fixed_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump({"lower": fixed_thresholds[0], "upper": fixed_thresholds[1]}, f)
    print(f"  Thresholds: lower={fixed_thresholds[0]:.4f}  upper={fixed_thresholds[1]:.4f}")

    phase_d_seed = args.seeds[0]
    print(f"  Phase D runs on seed {phase_d_seed} only (robustness check - one seed is sufficient)")
    for seed in [phase_d_seed]:
        seed_dir  = os.path.join(exp_root, f"seed_{seed}")
        ckpts     = _find_step_checkpoints(seed_dir, stride=stride)
        if not ckpts:
            print(f"  [seed {seed}] no checkpoint_step*.pt files found - skipping")
            continue

        out_dir = os.path.join(out_root, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [seed {seed}] {len(ckpts)} checkpoints (stride={stride})")

        for ckpt in ckpts:
            name     = os.path.splitext(os.path.basename(ckpt))[0]
            out_json = os.path.join(out_dir, "analysis_logs", "rainbow", f"{name}.json")
            if os.path.exists(out_json):
                print(f"    [skip] {name}")
                continue
            print(f"    Analysing {name} ...")
            try:
                analyze_checkpoint(
                    algorithm="rainbow",
                    checkpoint_path=ckpt,
                    experiment_root=out_dir,
                    n_episodes=args.n_episodes,
                    device=args.device,
                    reason="fixed_threshold_longitudinal",
                    split_mode="fixed",
                    percentile_x=25,
                    seed=seed,
                    fixed_thresholds=fixed_thresholds,
                )
                _inject_version(out_json)
            except Exception:
                traceback.print_exc()



def _run_phase_f(args: argparse.Namespace, exp_root: str):
    from report_robustness_stats import report
    report(exp_root=exp_root, seeds=args.seeds)



def _run_phase_h(args: argparse.Namespace, exp_root: str, stride: int = 1):
    from analyze_checkpoint import analyze_checkpoint

    # Build frozen stimulus set from all inline analysis JSONs
    all_labels: set[str] = set()
    for seed in args.seeds:
        json_dir = os.path.join(exp_root, f"seed_{seed}", "analysis_logs", "rainbow")
        for jp in glob.glob(os.path.join(json_dir, "checkpoint_*.json")):
            try:
                with open(jp) as f:
                    d = json.load(f)
                all_labels.update(d.get("rsa_labels", []))
            except Exception:
                pass

    if not all_labels:
        print("  [ERROR] No RSA labels in inline analysis JSONs - run training first.")
        return

    reference_stimuli = frozenset(all_labels)
    print(f"  Frozen stimulus set ({len(reference_stimuli)} stimuli): {sorted(reference_stimuli)}")

    out_root = os.path.join(exp_root, "frozen_rsa")

    for seed in args.seeds:
        seed_dir = os.path.join(exp_root, f"seed_{seed}")
        ckpts    = _find_step_checkpoints(seed_dir, stride=stride)
        if not ckpts:
            print(f"  [seed {seed}] no checkpoint_step*.pt files - skipping")
            continue

        out_dir = os.path.join(out_root, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [seed {seed}] {len(ckpts)} checkpoints (stride={stride})")

        for ckpt in ckpts:
            name     = os.path.splitext(os.path.basename(ckpt))[0]
            out_json = os.path.join(out_dir, "analysis_logs", "rainbow", f"{name}.json")
            if os.path.exists(out_json):
                print(f"    [skip] {name}")
                continue
            print(f"    Analysing {name} ...")
            try:
                analyze_checkpoint(
                    algorithm="rainbow",
                    checkpoint_path=ckpt,
                    experiment_root=out_dir,
                    n_episodes=args.n_episodes,
                    device=args.device,
                    reason="frozen_rsa",
                    seed=seed,
                    reference_stimuli=reference_stimuli,
                )
                _inject_version(out_json)
            except Exception:
                traceback.print_exc()

    print(f"\n  Results in: {out_root}")



def _run_phase_i(args: argparse.Namespace, exp_root: str, stride: int = 1):
    from analyze_checkpoint import analyze_checkpoint

    out_root = os.path.join(exp_root, "eps_sensitivity")

    for eps_weight in (0.5, 1.2):
        w_tag = str(eps_weight).replace(".", "p")
        print(f"\n  --- eps_weight = {eps_weight} ---")

        for seed in args.seeds:
            seed_dir = os.path.join(exp_root, f"seed_{seed}")
            ckpts    = _find_step_checkpoints(seed_dir, stride=stride)
            if not ckpts:
                print(f"  [seed {seed}] no checkpoint_step*.pt files - skipping")
                continue

            out_dir = os.path.join(out_root, f"w{w_tag}", f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"  [seed {seed}] {len(ckpts)} checkpoints (stride={stride})")

            for ckpt in ckpts:
                name     = os.path.splitext(os.path.basename(ckpt))[0]
                out_json = os.path.join(out_dir, "analysis_logs", "rainbow", f"{name}.json")
                if os.path.exists(out_json):
                    print(f"    [skip] {name}")
                    continue
                print(f"    Analysing {name} ...")
                try:
                    analyze_checkpoint(
                        algorithm="rainbow",
                        checkpoint_path=ckpt,
                        experiment_root=out_dir,
                        n_episodes=args.n_episodes,
                        device=args.device,
                        reason=f"eps_sensitivity_w{w_tag}",
                        seed=seed,
                        eps_weight=eps_weight,
                    )
                    _inject_version(out_json)
                except Exception:
                    traceback.print_exc()

    print(f"\n  Results in: {out_root}")



def _run_mora_variance_check(args: argparse.Namespace, exp_root: str):
    """Run the MORA subsampling variance check using seed 1's checkpoints.

    Picks the earliest available checkpoint as ckpt1 and the latest as ckpt2
    (ideally the final step). Skips silently if fewer than 2 checkpoints exist.
    Results are printed to stdout; pass/fail written to mora_variance_check.txt.
    """
    from scripts.mora_variance_check import _check_one_ckpt

    seed = args.seeds[0] if args.seeds else 1
    ckpts = _find_step_checkpoints(os.path.join(exp_root, f"seed_{seed}"))
    if len(ckpts) < 2:
        print(f"  [skip] Need at least 2 checkpoints for variance check "
              f"(found {len(ckpts)} for seed {seed})")
        return

    ckpt1 = ckpts[0]
    ckpt2 = ckpts[-1]
    print(f"  Early : {os.path.basename(ckpt1)}")
    print(f"  Late  : {os.path.basename(ckpt2)}")

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _check_one_ckpt(ckpt1, args.n_episodes, args.device, "Early checkpoint")
        _check_one_ckpt(ckpt2, args.n_episodes, args.device, "Late checkpoint")
    output = buf.getvalue()
    print(output)

    out_path = os.path.join(exp_root, "mora_variance_check.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"  Saved: {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment_root", default="rainbow_v2",
        help="Root directory for all output (default: rainbow_v2)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--T_max", type=int, default=3_000_000)
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50_000,
        help="Steps between checkpoints (default: 50k - 60 per seed over 3M steps)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000,
        help="Evaluation episodes per analysis call (default: 1000)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip", nargs="*", metavar="PHASE", default=[],
        help="Post-training phases to skip: C D F H I",
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training; jump straight to post-training phases",
    )
    parser.add_argument(
        "--post_phase_stride", type=int, default=10,
        help="Use every Nth checkpoint for post-training phases D/H/I "
             "(default: 10 → every 500k with 50k interval). "
             "Use 1 for all checkpoints.",
    )
    parser.add_argument(
        "--single_seed", type=int, default=None,
        help="Internal: run only this seed then exit. Used by the subprocess launcher "
             "to isolate each seed's CUDA context.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Max restart attempts per seed on crash (default: 3). "
             "Each retry resumes from checkpoint_live.pt + replay_buffer_live.npz.",
    )
    args = parser.parse_args()

    exp_root = (args.experiment_root if os.path.isabs(args.experiment_root)
                else os.path.join(_ROOT, args.experiment_root))
    skip_set = {p.upper() for p in args.skip}

    os.makedirs(exp_root, exist_ok=True)

    # Run just this seed in-process and exit. This gives each seed an isolated
    # CUDA context so an OOM in one seed cannot poison subsequent seeds.
    if args.single_seed is not None:
        _header(f"SEED {args.single_seed}  - Train + Inline Analysis")
        _run_seed(args.single_seed, args, exp_root)
        return

    n_ckpts = args.T_max // args.checkpoint_interval
    print(f"\nRainbow Full Pipeline  [{PIPELINE_VERSION}]")
    print(f"  experiment_root    : {exp_root}")
    print(f"  seeds              : {args.seeds}")
    print(f"  T_max              : {args.T_max:,}")
    print(f"  checkpoint_interval: {args.checkpoint_interval:,}  ({n_ckpts} checkpoints/seed)")
    print(f"  n_episodes         : {args.n_episodes}")
    print(f"  device             : {args.device}")
    post_phases = ["C", "D", "F", "H", "I"]
    will_run    = [p for p in post_phases if p not in skip_set]
    print(f"  post phases        : {will_run}  (skipped: {sorted(skip_set)})")
    if args.skip_training:
        print("  [skip_training] jumping straight to post phases")

    # Each seed runs as a fresh Python process so CUDA OOM in one seed does not
    # corrupt the CUDA context for subsequent seeds.

    if not args.skip_training:
        # Forward all CLI args that _run_seed cares about, adding --single_seed N.
        base_cmd = [
            sys.executable, __file__,
            "--experiment_root", exp_root,
            "--T_max", str(args.T_max),
            "--checkpoint_interval", str(args.checkpoint_interval),
            "--n_episodes", str(args.n_episodes),
            "--device", args.device,
            "--skip_training",   # subprocess only trains one seed; skip post phases
            "--skip", "C", "D", "F", "H", "I",
        ]
        for seed in args.seeds:
            _header(f"SEED {seed}  - Train + Inline Analysis")
            cmd = base_cmd + ["--single_seed", str(seed)]
            for attempt in range(1, args.max_retries + 1):
                print(f"  Launching subprocess: seed={seed} "
                      f"(attempt {attempt}/{args.max_retries})")
                ret = subprocess.run(cmd, cwd=_ROOT)
                if ret.returncode == 0:
                    break
                msg = (f"Seed {seed} crashed (exit {ret.returncode}), "
                       f"attempt {attempt}/{args.max_retries}")
                print(f"\n  [ERROR] {msg}")
                _notify("Rainbow Pipeline Crash", msg)
                if attempt < args.max_retries:
                    print("  Waiting 60s for GPU memory to clear before retry ...")
                    time.sleep(60)
            else:
                final_msg = (f"Seed {seed} failed all {args.max_retries} attempts - skipping")
                print(f"\n  [FATAL] {final_msg}")
                _notify("Rainbow Pipeline - Seed Abandoned", final_msg)


    if "C" not in skip_set:
        _header("PHASE C - Scalar DQN Ablation")
        try:
            _run_phase_c(args, exp_root)
        except Exception:
            traceback.print_exc()

    if "D" not in skip_set:
        _header("PHASE D - Fixed-Threshold Longitudinal Analysis")
        try:
            _run_phase_d(args, exp_root, stride=args.post_phase_stride)
        except Exception:
            traceback.print_exc()

    if "F" not in skip_set:
        _header("PHASE F - Robustness CI Report")
        try:
            _run_phase_f(args, exp_root)
        except Exception:
            traceback.print_exc()

    if "H" not in skip_set:
        _header("PHASE H - Frozen RSA Longitudinal Analysis")
        try:
            _run_phase_h(args, exp_root, stride=args.post_phase_stride)
        except Exception:
            traceback.print_exc()

    if "I" not in skip_set:
        _header("PHASE I - EPS Weight Sensitivity")
        try:
            _run_phase_i(args, exp_root, stride=args.post_phase_stride)
        except Exception:
            traceback.print_exc()

    _header("MORA Variance Check")
    try:
        _run_mora_variance_check(args, exp_root)
    except Exception:
        traceback.print_exc()

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Pipeline complete.")
    print(f"  All output under: {exp_root}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
