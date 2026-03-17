"""Episode-tagged transition store for offline SAC analysis.

EpisodeStore accumulates transitions from the training loop, tagging each with
the EPS score of the episode it came from.  A bounded rolling window (max_size
transitions) is kept so memory stays manageable over long runs.

Saved alongside each checkpoint as <checkpoint>_episodes.pkl and loaded by
analyze_checkpoint.py for partitioning into success / failure batches.
"""
import os
import pickle
from collections import deque


class EpisodeStore:
    """Rolling store of transitions annotated with episode EPS scores."""

    def __init__(self, max_size: int = 200_000):
        self._transitions: deque = deque(maxlen=max_size)
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(self, transitions: list, eps: float) -> None:
        """Add all transitions from one episode.

        Args:
            transitions: list of dicts with keys
                         {state, action, reward, next_state, terminal}.
                         Values are plain Python scalars / numpy arrays.
            eps:         EPS score of this episode (used for partitioning).
        """
        episode_id = self._episode_count
        self._episode_count += 1
        for t in transitions:
            self._transitions.append({
                "state":      t["state"],
                "action":     t["action"],
                "reward":     t["reward"],
                "next_state": t["next_state"],
                "terminal":   t["terminal"],
                "eps":        eps,
                "episode_id": episode_id,
            })

    @property
    def transitions(self) -> list:
        """Return all stored transitions as a plain list."""
        return list(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "EpisodeStore":
        with open(path, "rb") as f:
            return pickle.load(f)
