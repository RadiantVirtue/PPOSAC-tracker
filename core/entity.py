"""Entity protocol — the interface every entity must satisfy."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import torch

from core.data import EpisodeData, GradientResult


@runtime_checkable
class Entity(Protocol):
    """One algorithm + one environment combination."""

    entity_id: str
    obs_shape: tuple            # (H, W, C)
    n_actions: int

    achievement_names:     list[str]
    achievement_label_map: dict[str, str]
    achievement_groups:    dict[str, frozenset]

    hook_layer: str             # dotted module name for activation hook, relative to get_policy(model)

    def make_env(self, seed: int) -> gym.Env: ...
    def load_checkpoint(self, path: str, device: str) -> Any: ...
    def select_action(self, model: Any, obs: np.ndarray,
                      deterministic: bool = True) -> int: ...
    def compute_eps(self, achievements: dict[str, bool],
                    inventory: dict) -> float: ...
    def preprocess_obs(self, model: Any, obs_np: np.ndarray,
                       device: str) -> torch.Tensor: ...
    def get_policy(self, model: Any) -> Any: ...
    def compute_gradients(self, model: Any, episodes: list[EpisodeData],
                          device: str, batch_size: int) -> GradientResult: ...
