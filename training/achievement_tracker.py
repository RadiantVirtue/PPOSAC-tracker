"""Per-episode achievement frame logging for RSA stimulus collection.

Tracks the first observation frame within each episode where each achievement
is unlocked. Used to build EvaluationBatch.achievement_frames.

NOT used for EPS scoring - that is entity.compute_eps().
"""
from __future__ import annotations

import numpy as np


class AchievementTracker:
    """Stateful per-episode tracker. Call reset() at the start of each episode."""

    def __init__(self, achievement_names: list[str], label_map: dict[str, str]):
        """Initialise with achievement definitions from the entity.

        Args:
            achievement_names: ordered list of achievement IDs (from entity)
            label_map:         ach_id -> display label (from entity)
        """
        self._names     = achievement_names
        self._label_map = label_map
        self.reset()

    def reset(self) -> None:
        """Clear per-episode state. Call at episode start."""
        self._prev_ach     = {a: False for a in self._names}
        self._first_frames = {}    # ach_id -> (H, W, C) uint8 obs at first unlock
        self._step_idx     = 0

    def step(self, obs: np.ndarray, info: dict) -> None:
        """Record the first obs frame for each newly-unlocked achievement.

        Args:
            obs:  (H, W, C) uint8 current observation
            info: step info dict containing "achievements" bool dict
        """
        cur_ach = info.get("achievements", {})
        for ach in self._names:
            if (cur_ach.get(ach, False)
                    and not self._prev_ach[ach]
                    and ach not in self._first_frames):
                self._first_frames[ach] = obs.copy()
        self._prev_ach  = {a: bool(cur_ach.get(a, False)) for a in self._names}
        self._step_idx += 1

    def get_labelled_frames(self) -> dict[str, np.ndarray]:
        """Return {display_label: (H,W,C) uint8} for achievements unlocked this episode.

        Called by eval_runner on episode completion. Results are accumulated into
        EvaluationBatch.achievement_frames across all episodes.
        """
        return {
            self._label_map[ach]: frame
            for ach, frame in self._first_frames.items()
            if ach in self._label_map
        }
