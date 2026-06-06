"""MORA-Informed Priority - comparison experiment.

Runs standard Rainbow (k=0, PER baseline) and MORA-Rainbow for each k in
k_values over T_max steps, then plots smoothed return curves and modifier
curves for comparison.

Usage:
    python rainbow/mora_train_and_analyze.py
    python rainbow/mora_train_and_analyze.py --k-values 0 5 10 25 --seeds 1 2 3 --T-max 1000000
    python rainbow/mora_train_and_analyze.py --k-values 0 5 --seeds 1 --T-max 50000  # smoke test

Outputs (under --experiment-root, default mora_results/):
    mora_results/
        k{k}/seed_{s}/logs/
            rainbowreturnlog.txt          - episode returns (one per line)
            mora_modifier_log.csv         - episode, m, frac_neg_batch  (MORA runs only)
        mora_return_comparison.png        - smoothed return curves (all k, seeds)
        mora_modifier_curves.png          - m over episodes (one seed per k, sanity check)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.mora_tracker import MORAPriorityTracker
from rainbow.train import build_parser, main_rainbow



@dataclass
class MoraArgs:
    """MORA comparison experiment arguments."""
    k_values: list[int] | None = None
    """k=0 = standard PER baseline; k>0 = MORA with rolling window k.
    In --plot-only mode defaults to discovering all k values in experiment_root.
    In training mode defaults to [0, 5, 10, 25]."""
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    plot_only: bool = False
    """Scan experiment_root for all completed k*/seed_* runs and plot without re-running."""
    T_max: int = 1_000_000
    checkpoint_interval: int = 100_000
    experiment_root: str = "mora_results"
    device: str = "cpu"
    # Rainbow hyperparameters (use build_parser() defaults unless overridden)
    hidden_size: int = 512
    atoms: int = 51
    architecture: str = "canonical"
    memory_capacity: int = 500_000
    learning_rate: float = 0.0000625
    batch_size: int = 32
    # MORA-specific
    mora_epsilon: float = 0.01
    mora_percentile_x: int = 25
    """Top/bottom percentile_x % used for success/failure classification."""
    mora_window: int = 25
    """Rolling window of recent returns for computing percentile thresholds."""
    # Adaptive seed convergence
    convergence_std_pct: float | None = None
    """If set (e.g. 2.0), keep adding seeds for each k until the coefficient of
    variation of per-seed final-window mean return drops below this percentage,
    or max_seeds is reached.  None = run exactly args.seeds (fixed behaviour)."""
    max_seeds: int = 10
    """Hard cap on seeds per k when convergence_std_pct is set."""
    # Plotting
    smoothing_window: int = 50
    """Rolling-mean window for smoothing return curves."""



def _build_rainbow_args(args: MoraArgs, seed: int, seed_root: str):
    parser = build_parser()
    ns = parser.parse_args([])
    ns.seed = seed
    ns.T_max = args.T_max
    ns.checkpoint_interval = args.checkpoint_interval
    ns.experiment_root = seed_root
    ns.hidden_size = args.hidden_size
    ns.atoms = args.atoms
    ns.architecture = args.architecture
    ns.memory_capacity = args.memory_capacity
    ns.learning_rate = args.learning_rate
    ns.batch_size = args.batch_size
    ns.disable_cuda = (args.device == "cpu")
    ns.model = None
    return ns



def _run_one(args: MoraArgs, k: int, seed: int) -> str:
    """Train one Rainbow run (MORA if k>0, standard PER if k=0).

    Returns path to rainbowreturnlog.txt.
    """
    label = f"k{k}"
    seed_root = os.path.join(args.experiment_root, label, f"seed_{seed}")
    os.makedirs(seed_root, exist_ok=True)

    rainbow_args = _build_rainbow_args(args, seed, seed_root)

    tracker = None
    if k > 0:
        tracker = MORAPriorityTracker(
            k=k,
            epsilon=args.mora_epsilon,
            percentile_x=args.mora_percentile_x,
            window=args.mora_window,
        )

    algo_tag = f"MORA k={k}" if k > 0 else "Standard PER"
    print(f"\n=== {algo_tag} | seed={seed} | T_max={args.T_max:,} ===")
    main_rainbow(rainbow_args, on_checkpoint_saved=None, mora_tracker=tracker)

    return os.path.join(seed_root, "logs", "rainbowreturnlog.txt")



def _load_returns(log_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (returns, steps_or_None).

    Supports both old format (one float per line) and new format (step,return).
    Returns steps=None for old-format logs.
    """
    if not os.path.exists(log_path):
        return np.array([]), None
    with open(log_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return np.array([]), None
    returns, steps = [], []
    has_steps = None
    for line in lines:
        parts = line.split(",")
        if len(parts) == 2:
            steps.append(int(parts[0]))
            returns.append(float(parts[1]))
            has_steps = True
        else:
            returns.append(float(parts[0]))
            has_steps = False
    if has_steps:
        return np.array(returns), np.array(steps)
    return np.array(returns), None


def _load_modifier(log_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (episodes, m_values) from mora_modifier_log.csv."""
    if not os.path.exists(log_path):
        return np.array([]), np.array([])
    episodes, ms = [], []
    with open(log_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    episodes.append(int(parts[0]))
                    ms.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(episodes), np.array(ms)


def _scan_existing_runs(experiment_root: str) -> dict[int, list[int]]:
    """Scan experiment_root for completed k*/seed_* directories.

    Returns {k: [seed, ...]} for every seed that has a non-empty
    rainbowreturnlog.txt.
    """
    result: dict[int, list[int]] = {}
    if not os.path.isdir(experiment_root):
        return result
    for entry in sorted(os.listdir(experiment_root)):
        if not entry.startswith("k"):
            continue
        try:
            k = int(entry[1:])
        except ValueError:
            continue
        k_dir = os.path.join(experiment_root, entry)
        if not os.path.isdir(k_dir):
            continue
        for seed_entry in sorted(os.listdir(k_dir)):
            if not seed_entry.startswith("seed_"):
                continue
            try:
                seed = int(seed_entry.split("_")[1])
            except (IndexError, ValueError):
                continue
            log = os.path.join(k_dir, seed_entry, "logs", "rainbowreturnlog.txt")
            if os.path.exists(log) and os.path.getsize(log) > 0:
                result.setdefault(k, []).append(seed)
    return result


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")



def _plot_return_comparison(
    all_returns: dict[int, list[np.ndarray]],
    out_path: str,
    smoothing_window: int,
    all_steps: dict[int, list[np.ndarray | None]] | None = None,
    T_max: int = 1_000_000,
):
    """Plot smoothed episode return for each k, with ±1 std shading over seeds.

    If all_steps contains step arrays (new-format logs), the x-axis shows
    training steps.  For old-format logs without step info, steps are
    approximated by linear interpolation over T_max.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.tab10.colors

    for idx, (k, seed_returns) in enumerate(sorted(all_returns.items())):
        label = f"k={k}" if k > 0 else "Standard PER (k=0)"
        colour = colours[idx % len(colours)]

        seed_steps_list = (all_steps or {}).get(k, [None] * len(seed_returns))

        # Build per-seed (steps, smoothed_returns) pairs
        smoothed_pairs = []
        for r, s in zip(seed_returns, seed_steps_list):
            if len(r) == 0:
                continue
            sr = _smooth(r, smoothing_window)
            if s is not None:
                # Align step axis to smoothed length (convolve drops window-1 from start)
                offset = len(r) - len(sr)
                sx = s[offset:offset + len(sr)].astype(float)
            else:
                # Approximate: linearly map episode index → step
                n = len(r)
                episode_steps = np.linspace(0, T_max, n + 1)[1:]  # step at end of each ep
                sx = episode_steps[offset:offset + len(sr)] if (offset := len(r) - len(sr)) >= 0 else episode_steps[:len(sr)]
            smoothed_pairs.append((sx, sr))

        if not smoothed_pairs:
            continue

        # Interpolate all seeds onto a common step grid
        step_min = max(p[0][0] for p in smoothed_pairs)
        step_max = min(p[0][-1] for p in smoothed_pairs)
        grid = np.linspace(step_min, step_max, 500)

        interp_arr = np.stack([
            np.interp(grid, sx, sr) for sx, sr in smoothed_pairs
        ])

        mean = interp_arr.mean(axis=0)
        std = interp_arr.std(axis=0)

        ax.plot(grid, mean, label=label, color=colour, linewidth=1.5)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=colour)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title("MORA-Informed Priority vs Standard PER (Episode Return)z")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_modifier_curves(
    modifier_data: dict[int, tuple[np.ndarray, np.ndarray]],
    out_path: str,
):
    """Plot m over episodes for each k (one seed shown per k)."""
    k_values = [k for k in sorted(modifier_data) if k > 0]
    if not k_values:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colours = plt.cm.tab10.colors

    for idx, k in enumerate(k_values):
        eps, ms = modifier_data[k]
        if len(eps) == 0:
            continue
        ax.plot(eps, ms, label=f"k={k}", color=colours[idx % len(colours)], linewidth=1.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Modifier m")
    ax.set_title("MORA Modifier m over Training (seed 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")



def run(args: MoraArgs):
    os.makedirs(args.experiment_root, exist_ok=True)

    if args.plot_only:
        # Discover all completed runs under experiment_root
        discovered = _scan_existing_runs(args.experiment_root)
        if not discovered:
            print(f"No completed runs found under {args.experiment_root}/")
            return

        if args.k_values is not None:
            # Filter to only the explicitly requested k values
            discovered = {k: v for k, v in discovered.items() if k in args.k_values}
            if not discovered:
                print(f"No completed runs found for k_values={args.k_values}")
                return

        k_list = sorted(discovered.keys())
        print(f"Plotting k values: {k_list}")
        for k, seeds in sorted(discovered.items()):
            print(f"  k={k}: seeds {seeds}")

        all_returns: dict[int, list[np.ndarray]] = {k: [] for k in k_list}
        all_steps: dict[int, list[np.ndarray | None]] = {k: [] for k in k_list}
        modifier_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for k in k_list:
            for seed in discovered[k]:
                return_log = os.path.join(
                    args.experiment_root, f"k{k}", f"seed_{seed}",
                    "logs", "rainbowreturnlog.txt",
                )
                rets, steps = _load_returns(return_log)
                all_returns[k].append(rets)
                all_steps[k].append(steps)

            # Modifier curve: first available seed only
            first_seed = discovered[k][0]
            if k > 0:
                mod_log = os.path.join(
                    args.experiment_root, f"k{k}", f"seed_{first_seed}",
                    "logs", "mora_modifier_log.csv",
                )
                modifier_data[k] = _load_modifier(mod_log)
    else:
        k_values = args.k_values if args.k_values is not None else [0, 5, 10, 25]
        # all_returns[k] = [array_seed1, array_seed2, ...]
        all_returns = {k: [] for k in k_values}
        all_steps: dict[int, list[np.ndarray | None]] = {k: [] for k in k_values}
        # modifier_data[k] = (episodes, m_values) from seed 1 only
        modifier_data = {}

        for k in k_values:
            # Determine seed sequence: fixed list or adaptive (1, 2, 3, …).
            if args.convergence_std_pct is not None:
                seed_iter = range(1, args.max_seeds + 1)
            else:
                seed_iter = args.seeds

            summaries: list[float] = []
            first_seed = True
            for seed in seed_iter:
                return_log = _run_one(args, k, seed)
                rets, steps = _load_returns(return_log)
                all_returns[k].append(rets)
                all_steps[k].append(steps)

                # Load modifier curve from first seed only (for sanity plot)
                if first_seed and k > 0:
                    mod_log = os.path.join(
                        args.experiment_root, f"k{k}", f"seed_{seed}",
                        "logs", "mora_modifier_log.csv",
                    )
                    modifier_data[k] = _load_modifier(mod_log)
                first_seed = False

                # Adaptive convergence check
                if args.convergence_std_pct is not None:
                    tail = rets[-args.smoothing_window:] if len(rets) >= args.smoothing_window else rets
                    summaries.append(float(tail.mean()) if len(tail) > 0 else 0.0)
                    if len(summaries) >= 2:
                        mean_s = abs(float(np.mean(summaries)))
                        cv = (float(np.std(summaries)) / mean_s * 100) if mean_s > 1e-8 else float("inf")
                        print(f"  k={k} seed={seed}: cv={cv:.2f}% (target <{args.convergence_std_pct}%)")
                        if cv < args.convergence_std_pct:
                            print(f"  Converged at {len(summaries)} seeds.")
                            break

    # Generate plots
    return_plot = os.path.join(args.experiment_root, "mora_return_comparison.png")
    modifier_plot = os.path.join(args.experiment_root, "mora_modifier_curves.png")

    _plot_return_comparison(
        all_returns, return_plot, args.smoothing_window,
        all_steps=all_steps, T_max=args.T_max,
    )
    _plot_modifier_curves(modifier_data, modifier_plot)

    print(f"\nResults in: {args.experiment_root}/")


def main():
    run(tyro.cli(MoraArgs))


if __name__ == "__main__":
    main()
