"""Milestone checkpoint tracker for PPO (SB3).

Saves a checkpoint the first time each achievement is unlocked.
Works with stable-baselines3 PPO: model is passed directly instead of (agent, optimizer).
"""


class MilestoneTracker:

    def __init__(self, experiment_root, save_fn, on_checkpoint_saved=None, algo_subdir="ppo"):
        self.seen_achievements = set()
        self.experiment_root = experiment_root
        self._save_fn = save_fn
        self._on_checkpoint_saved = on_checkpoint_saved
        self._algo_subdir = algo_subdir

    def check_and_save(self, info, model, global_step: int, episode_count: int):
        """If info contains a new achievement, save a milestone checkpoint.

        Args:
            info:          episode terminal info dict (must contain "achievements").
            model:         SB3 PPO model (or SAC actor for sac/) to checkpoint.
            global_step:   current training step count.
            episode_count: current episode count.
        """
        if "achievements" not in info:
            return
        current = {k for k, v in info["achievements"].items() if v}
        new = current - self.seen_achievements
        if new:
            self.seen_achievements.update(new)
            for ach in sorted(new):
                path = (
                    f"{self.experiment_root}/checkpoints/{self._algo_subdir}/"
                    f"milestone_first_{ach}_ep{episode_count}.pt"
                )
                self._save_fn(model, global_step, episode_count, path)
                print(f"Milestone checkpoint: {ach} at episode {episode_count}")
                if self._on_checkpoint_saved:
                    self._on_checkpoint_saved(path)
