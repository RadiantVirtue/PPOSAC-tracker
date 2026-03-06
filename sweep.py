"""Sweep orchestration: train PPO and Rainbow across DoorKey and KeyCorridor."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)


import time

from ppo.train import Args as PPOArgs, main_ppo
from rainbow.train import Args as RainbowArgs, main_rainbow
from shared.achievements import get_achievements_for_env

ENVIRONMENT_FAMILIES = {
    "DoorKey": [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ],
    "KeyCorridor": [
        "MiniGrid-KeyCorridorS3R3-v0"
    ],
}

FAMILIES_TO_RUN = ["DoorKey"]
ALGORITHMS = ["ppo", "rainbow"]

SOLVE_THRESHOLD = 0.75    # fraction of episodes that must be solved
EVAL_WINDOW = 100         # episodes per evaluation window
REQUIRED_CONSECUTIVE = 5  # consecutive passing windows before stopping
MAX_EPISODES = 100_000    # PPO episode budget
MAX_TIMESTEPS = 10_000_000  # Rainbow timestep budget
SEEDS = [1, 2, 3]
CONSISTENCY_WINDOW = 5_000  # max spread in convergence episodes across seeds
EXPERIMENT_ROOT = "sweep_results"


class SolveRateTracker:
    #Tracks rolling solve rate

    def __init__(self, env_id, threshold=SOLVE_THRESHOLD, window=EVAL_WINDOW,
                 required_consecutive=REQUIRED_CONSECUTIVE):
        self.threshold = threshold
        self.window = window
        self.required_consecutive = required_consecutive
        self._results = []
        self._consecutive = 0
        self._final_achievement = get_achievements_for_env(env_id)[-1]

    def update(self, info):
        """Call once per episode. Returns True when convergence criterion is met."""
        solved = info.get("achievements", {}).get(self._final_achievement, False)
        self._results.append(bool(solved))
        # Evaluate at the end of each complete window
        if len(self._results) % self.window == 0:
            rate = sum(self._results[-self.window:]) / self.window
            if rate >= self.threshold:
                self._consecutive += 1
            else:
                self._consecutive = 0
        return self._consecutive >= self.required_consecutive

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

                seed_results = []

                for seed in SEEDS:
                    print(f"\n  -- seed={seed} --")
                    tracker = SolveRateTracker(env_id)
                    t_start = time.time()

                    if algorithm == "ppo":
                        args = PPOArgs(
                            env_id=env_id,
                            total_episodes=MAX_EPISODES,
                            seed=seed,
                            checkpoint_freq=0,
                            checkpoint_achievements=False,
                            experiment_root=EXPERIMENT_ROOT,
                            num_procs=16,
                            frames_per_proc=512,
                            entropy_coef=0.05,
                        )
                        episodes, frames = main_ppo(args, should_stop=tracker.update)

                    elif algorithm == "rainbow":
                        args = RainbowArgs(
                            env_id=env_id,
                            total_timesteps=MAX_TIMESTEPS,
                            seed=seed,
                            checkpoint_freq=0,
                            experiment_root=EXPERIMENT_ROOT,
                        )
                        episodes, frames = main_rainbow(args, should_stop=tracker.update)

                    seed_results.append({
                        "seed": seed,
                        "episodes": episodes,
                        "frames": frames,
                        "wall_time_s": round(time.time() - t_start, 1),
                        "solve_rate": round(tracker.solve_rate, 3),
                        "converged": tracker._consecutive >= REQUIRED_CONSECUTIVE,
                        "consecutive_windows": tracker._consecutive,
                    })

                # Consistency check across seeds
                converged = [r for r in seed_results if r["converged"]]
                all_converged = len(converged) == len(SEEDS)
                if all_converged:
                    eps_spread = (
                        max(r["episodes"] for r in converged)
                        - min(r["episodes"] for r in converged)
                    )
                    consistent = eps_spread <= CONSISTENCY_WINDOW
                else:
                    eps_spread = None
                    consistent = False

                results.append({
                    "family": family,
                    "env_id": env_id,
                    "algorithm": algorithm,
                    "seed_results": seed_results,
                    "n_converged": len(converged),
                    "eps_spread": eps_spread,
                    "consistent": consistent,
                })

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'ENV':<40} {'ALG':<8} {'CONV':>6}  {'SPREAD':>7}  {'CONSISTENT'}")
    print(f"{'-'*80}")
    for r in results:
        spread_str = f"{r['eps_spread']:>7}" if r["eps_spread"] is not None else "      -"
        print(
            f"{r['env_id']:<40} {r['algorithm']:<8} "
            f"{r['n_converged']:>2}/{len(SEEDS)}  {spread_str}  "
            f"{'YES' if r['consistent'] else 'no'}"
        )
        for s in r["seed_results"]:
            converged_tag = "" if s["converged"] else "  [did not converge]"
            print(
                f"    seed={s['seed']}: ep={s['episodes']:>7}  "
                f"rate={s['solve_rate']:.3f}  consec={s['consecutive_windows']}"
                f"{converged_tag}"
            )

    return results


if __name__ == "__main__":
    run_sweep()
