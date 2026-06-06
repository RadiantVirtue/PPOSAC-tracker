"""Rolling store of episode-tagged transitions for offline analysis."""
import os
import pickle
from collections import deque


class EpisodeStore:
    """Rolling store of transitions annotated with episode EPS scores."""

    def __init__(self, max_size: int = 200_000):
        self._transitions: deque = deque(maxlen=max_size)
        self._episode_count: int = 0

    def add_episode(self, transitions: list, eps: float) -> None:
        """Add all transitions from one episode, tagged with its EPS score."""
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
        return list(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "EpisodeStore":
        with open(path, "rb") as f:
            return pickle.load(f)
