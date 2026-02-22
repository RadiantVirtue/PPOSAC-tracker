# tracks first-time achievements and saves milestone checkpoints
class MilestoneTracker:

    def __init__(self, experiment_root, save_fn, on_checkpoint_saved=None, algo_subdir="ppo"):
        self.seen_achievements = set()
        self.experiment_root = experiment_root
        self._save_fn = save_fn
        self._on_checkpoint_saved = on_checkpoint_saved
        self._algo_subdir = algo_subdir

    # if info contains a new achievement, save a milestone checkpoint
    def check_and_save(self, info, agent, optimizer, global_step, episode_count):
        if "achievements" not in info:
            return
        current = {k for k, v in info["achievements"].items() if v}
        new = current - self.seen_achievements
        if new:
            self.seen_achievements.update(new)
            for ach in new:
                path = (
                    f"{self.experiment_root}/checkpoints/{self._algo_subdir}/"
                    f"milestone_first_{ach}_ep{episode_count}.pt"
                )
                self._save_fn(agent, optimizer, global_step, episode_count, path)
                print(f"Milestone checkpoint: {ach} at episode {episode_count}")
                if self._on_checkpoint_saved:
                    self._on_checkpoint_saved(path)
