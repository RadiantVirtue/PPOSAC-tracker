"""Sweep orchestration: train PPO and SAC across DoorKey and KeyCorridor."""
import time

from ppo.train import Args as PPOArgs, main_ppo
from sac.train import Args as SACArgs, main_sac

ENVIRONMENT_FAMILIES = {
    "DoorKey": [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ],
    "KeyCorridor": [
        "MiniGrid-KeyCorridorS3R3-v0",
        "MiniGrid-KeyCorridorS4R3-v0",
        "MiniGrid-KeyCorridorS5R3-v0",
    ],
}

FAMILIES_TO_RUN = list(ENVIRONMENT_FAMILIES.keys())
ALGORITHMS = ["ppo", "sac"]

SOLVE_THRESHOLD = 0.75    # fraction of episodes that must be solved
EVAL_WINDOW = 100         # rolling window size for solve rate
MAX_EPISODES = 100_000    # PPO episode budget
MAX_TIMESTEPS = 10_000_000  # SAC timestep budget
SEED = 1
CHECKPOINT_FREQ = 1000
EXPERIMENT_ROOT = "sweep_results"


class SolveRateTracker:
    """Tracks rolling solve rate and signals when training threshold is met."""

    def __init__(self, threshold=SOLVE_THRESHOLD, window=EVAL_WINDOW):
        self.threshold = threshold
        self.window = window
        self._results = []

    def update(self, ep_solved):
        """Call once per episode. Returns True when solve rate criterion is met."""
        self._results.append(bool(ep_solved))
        if len(self._results) >= self.window:
            rate = sum(self._results[-self.window:]) / self.window
            return rate >= self.threshold
        return False

    @property
    def solve_rate(self):
        if not self._results:
            return 0.0
        return sum(self._results[-self.window:]) / min(len(self._results), self.window)


def run_sweep():
    results = []

    for family in FAMILIES_TO_RUN:
        env_ids = ENVIRONMENT_FAMILIES[family]

        for env_id in env_ids:
            for algorithm in ALGORITHMS:
                print(f"\n{'='*60}")
                print(f"  {algorithm.upper()} on {env_id}")
                print(f"{'='*60}")

                tracker = SolveRateTracker()
                checkpoints_saved = []

                def on_checkpoint_saved(path):
                    checkpoints_saved.append(path)

                t_start = time.time()

                if algorithm == "ppo":
                    args = PPOArgs(
                        env_id=env_id,
                        total_episodes=MAX_EPISODES,
                        seed=SEED,
                        checkpoint_freq=CHECKPOINT_FREQ,
                        experiment_root=EXPERIMENT_ROOT,
                        num_procs=16,
                    )
                    episodes, frames = main_ppo(
                        args,
                        on_checkpoint_saved=on_checkpoint_saved,
                        should_stop=tracker.update,
                    )
                    wall_time = time.time() - t_start
                    results.append({
                        "family": family,
                        "env_id": env_id,
                        "algorithm": "ppo",
                        "episodes": episodes,
                        "frames": frames,
                        "wall_time_s": round(wall_time, 1),
                        "solve_rate": round(tracker.solve_rate, 3),
                        "solved": tracker.solve_rate >= SOLVE_THRESHOLD,
                        "checkpoints": len(checkpoints_saved),
                    })

                elif algorithm == "sac":
                    args = SACArgs(
                        env_id=env_id,
                        total_timesteps=MAX_TIMESTEPS,
                        seed=SEED,
                        checkpoint_freq=CHECKPOINT_FREQ,
                        experiment_root=EXPERIMENT_ROOT,
                    )
                    episodes, steps = main_sac(
                        args,
                        on_checkpoint_saved=on_checkpoint_saved,
                        should_stop=tracker.update,
                    )
                    wall_time = time.time() - t_start
                    results.append({
                        "family": family,
                        "env_id": env_id,
                        "algorithm": "sac",
                        "episodes": episodes,
                        "frames": steps,
                        "wall_time_s": round(wall_time, 1),
                        "solve_rate": round(tracker.solve_rate, 3),
                        "solved": tracker.solve_rate >= SOLVE_THRESHOLD,
                        "checkpoints": len(checkpoints_saved),
                    })

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'ENV':<40} {'ALG':<6} {'EPS':>8} {'FRAMES':>10} {'RATE':>6} {'SOLVED'}")
    print(f"{'-'*80}")
    for r in results:
        print(
            f"{r['env_id']:<40} {r['algorithm']:<6} "
            f"{r['episodes']:>8} {r['frames']:>10} "
            f"{r['solve_rate']:>6.3f} {'YES' if r['solved'] else 'no'}"
        )

    return results


if __name__ == "__main__":
    run_sweep()
