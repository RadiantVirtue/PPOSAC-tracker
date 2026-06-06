"""Gradient analysis. Delegates to entity.compute_gradients()."""
from __future__ import annotations

from typing import Any

from core.data import EpisodeData, GradientResult
from core.entity import Entity


def analyze(entity: Entity, model: Any, episodes: list[EpisodeData],
            device: str, batch_size: int = 10) -> GradientResult:
    """Run gradient computation for one episode group via the entity."""
    return entity.compute_gradients(model, episodes, device, batch_size)
