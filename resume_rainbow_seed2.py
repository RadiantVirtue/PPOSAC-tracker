"""Resume Rainbow DQN training for seed 2 from the live checkpoint.

Training crashed at step 1,950,000. This script resumes to T_max=3,000,000
using the same hyperparameters as the original run, with full inline analysis.

Usage (from PPOSAC-tracker/):
    python resume_rainbow_seed2.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_checkpoint import analyze_checkpoint
from rainbow.train import build_parser, main_rainbow
from shared.reporting import generate_report, label_from_path
from shared.storage import load_analysis_results

# ── Config matching the original run ──────────────────────────────────────────
SEED              = 2
T_MAX             = 3_000_000
CHECKPOINT_INTERVAL = 50_000   # every 50k steps (matches analysis log spacing)
ANALYZE_EVERY     = 1          # analyse every periodic checkpoint
N_EVAL_EPISODES   = 1000
SPLIT_MODE        = "percentile"
PERCENTILE_X      = 25
DEVICE            = "cuda"

EXPERIMENT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rainbow_experiment_root"
)
SEED_ROOT = os.path.join(EXPERIMENT_ROOT, f"seed_{SEED}")
LIVE_CKPT = os.path.join(SEED_ROOT, "checkpoints", "rainbow", "checkpoint_live.pt")

# Survived the crash — use as prev_path for the first new checkpoint's Δθ.
CRASH_PREV_CKPT = os.path.join(
    SEED_ROOT, "checkpoints", "rainbow", "checkpoint_step1950000.pt"
)
# Also stale from crash — no longer needed for weight-delta, delete at startup.
STALE_CKPT = os.path.join(
    SEED_ROOT, "checkpoints", "rainbow", "checkpoint_step1900000.pt"
)


def _ckpt_step_key(fname: str) -> int:
    """Numeric sort key for analysis log filenames (e.g. checkpoint_step200000)."""
    m = re.search(r"(\d+)", fname)
    return int(m.group(1)) if m else 0


def _build_args():
    parser = build_parser()
    ns = parser.parse_args([])
    # Core training
    ns.seed = SEED
    ns.T_max = T_MAX
    ns.checkpoint_interval = CHECKPOINT_INTERVAL
    ns.experiment_root = SEED_ROOT
    # Hyperparameters (from args_dict in live checkpoint)
    ns.hidden_size = 512
    ns.atoms = 51
    ns.architecture = "canonical"
    ns.history_length = 3
    ns.V_min = -10
    ns.V_max = 10
    ns.multi_step = 3
    ns.discount = 0.99
    ns.noisy_std = 0.1
    ns.priority_exponent = 0.5
    ns.priority_weight = 0.4
    ns.memory_capacity = 500_000
    ns.learning_rate = 0.0000625
    ns.batch_size = 32
    ns.disable_cuda = (DEVICE == "cpu")
    # Resume from live checkpoint
    ns.model = LIVE_CKPT
    return ns


def _load_seen_achievements(logs_dir: str) -> set:
    """Derive which achievements were already milestoned from existing JSON filenames.

    Scans for files matching milestone_first_<achievement>_ep<N>.json and returns
    the set of achievement names found. Passed to main_rainbow so that training
    doesn't re-fire milestones that already have analysis JSONs from before the crash.
    """
    seen = set()
    if not os.path.isdir(logs_dir):
        return seen
    for fname in os.listdir(logs_dir):
        m = re.match(r"milestone_first_(.+)_ep\d+\.json", fname)
        if m:
            seen.add(m.group(1))
    return seen


def main():
    # Sanity-check
    if not os.path.exists(LIVE_CKPT):
        sys.exit(f"ERROR: live checkpoint not found at {LIVE_CKPT}")

    import torch
    meta = torch.load(LIVE_CKPT, map_location="cpu", weights_only=False)
    resume_step = meta["global_step"]
    resume_eps  = meta["episode_count"]
    print(f"Live checkpoint: step={resume_step:,}  episodes={resume_eps:,}")
    print(f"Resuming to T_max={T_MAX:,}  ({T_MAX - resume_step:,} steps remaining)")

    # Clean up the stale checkpoint_step1900000.pt (no longer needed).
    if os.path.exists(STALE_CKPT):
        os.remove(STALE_CKPT)
        print(f"Deleted stale checkpoint: {os.path.basename(STALE_CKPT)}")

    args = _build_args()

    logs_dir = os.path.join(SEED_ROOT, "analysis_logs", "rainbow")
    seen_achievements = _load_seen_achievements(logs_dir)
    if seen_achievements:
        print(f"Pre-seeding seen_achievements ({len(seen_achievements)}): {sorted(seen_achievements)}")

    analyzed_count = [0]
    first_call = [True]   # flag to prime weight-delta on first periodic checkpoint

    def on_checkpoint(path, prev_path=None, is_milestone=False):
        if is_milestone:
            label = label_from_path(path)
            print(f"  [milestone] {label}")
            analyze_checkpoint(
                "rainbow", path, SEED_ROOT,
                n_episodes=N_EVAL_EPISODES,
                device=DEVICE,
                reason=label,
                split_mode=SPLIT_MODE,
                percentile_x=PERCENTILE_X,
                seed=SEED,
                prev_checkpoint_path=prev_path,
            )
            return

        # Periodic checkpoint
        analyzed_count[0] += 1

        # The training loop's pointer state starts fresh after resume, so
        # prev_path=None for the very first new checkpoint.  Override it with
        # the crash-survivor checkpoint_step1950000.pt so that the first Δθ
        # comparison is valid.  Delete it afterwards — train.py won't manage it.
        if first_call[0]:
            first_call[0] = False
            if prev_path is None and os.path.exists(CRASH_PREV_CKPT):
                prev_path = CRASH_PREV_CKPT
                print(f"  [delta] primed prev_path → {os.path.basename(CRASH_PREV_CKPT)}")

        if analyzed_count[0] % ANALYZE_EVERY != 0:
            return

        label = label_from_path(path)
        print(f"  [analyse] {label}")
        analyze_checkpoint(
            "rainbow", path, SEED_ROOT,
            n_episodes=N_EVAL_EPISODES,
            device=DEVICE,
            reason=label,
            split_mode=SPLIT_MODE,
            percentile_x=PERCENTILE_X,
            seed=SEED,
            prev_checkpoint_path=prev_path,
        )

        # Bug 1: delete the crash-survivor once the first analysis has finished
        # using it for weight-delta — train.py's pointer rotation won't clean it up.
        if prev_path == CRASH_PREV_CKPT and os.path.exists(CRASH_PREV_CKPT):
            os.remove(CRASH_PREV_CKPT)
            print(f"  [ckpt] Deleted {os.path.basename(CRASH_PREV_CKPT)}")

    print(f"\n=== Resuming Rainbow seed={SEED} ===")
    episode_count, _ = main_rainbow(args, on_checkpoint_saved=on_checkpoint,
                                    seen_achievements=seen_achievements)
    print(f"\nTraining complete: {episode_count:,} total episodes")

    # Build full report from ALL analysis JSONs (pre-crash + newly produced).
    # Bug 10: sort filenames numerically (not lexicographically) so that e.g.
    # checkpoint_step200000 appears before checkpoint_step1000000.
    logs_dir = os.path.join(SEED_ROOT, "analysis_logs", "rainbow")
    all_results = []
    if os.path.isdir(logs_dir):
        # For milestone files, keep only the earliest episode per achievement
        # (the crashed run re-fired all milestones, creating duplicate JSONs).
        seen_milestone_ach: dict = {}  # achievement_name → (fname, ep_number)
        milestone_fnames: set = set()
        for fname in os.listdir(logs_dir):
            m = re.match(r"(milestone_first_(.+))_ep(\d+)\.json", fname)
            if m:
                _, ach, ep = m.group(1, 2, 3)
                ep = int(ep)
                if ach not in seen_milestone_ach or ep < seen_milestone_ach[ach][1]:
                    seen_milestone_ach[ach] = (fname, ep)
        milestone_fnames = {v[0] for v in seen_milestone_ach.values()}

        for fname in sorted(os.listdir(logs_dir), key=_ckpt_step_key):
            if not fname.endswith(".json"):
                continue
            # Skip duplicate (later) milestone files
            is_milestone = fname.startswith("milestone_")
            if is_milestone and fname not in milestone_fnames:
                continue
            r = load_analysis_results(os.path.join(logs_dir, fname))
            label = label_from_path(os.path.splitext(fname)[0])
            all_results.append((label, r))

    if all_results:
        all_results.sort(key=lambda x: x[1].get("episode", 0))
        report_md = generate_report(
            all_results, "Crafter", SEED, episode_count, SEED_ROOT
        )
        report_path = os.path.join(SEED_ROOT, f"report_crafter_rainbow_{SEED}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Full report saved ({len(all_results)} checkpoints): {report_path}")
    else:
        print("No analysis results found.")


if __name__ == "__main__":
    main()
