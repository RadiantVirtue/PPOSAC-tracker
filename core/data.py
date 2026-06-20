"""Canonical data structures shared across the entire pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeData:
    """One episode's trajectory in canonical format."""
    observations: np.ndarray   # (T, H, W, C) uint8 - always; never float32
    actions:      np.ndarray   # (T,) int64
    rewards:      np.ndarray   # (T,) float32
    dones:        np.ndarray   # (T,) bool


@dataclass
class EvaluationBatch:
    """Full output of one evaluation run. Saved to a temp file, deleted after analysis."""
    entity_id:          str
    checkpoint_path:    str             # absolute path on disk
    checkpoint_step:    int
    episodes:           list[EpisodeData]
    eps_scores:         np.ndarray      # (N,) float32
    achievement_frames: dict[str, list[np.ndarray]]
    # display_label -> list of (H,W,C) uint8 frames; one frame per episode-unlock event
    # Produced by AchievementTracker. Separate from EPS scoring.


@dataclass
class GradientResult:
    """Gradient analysis output for one episode group."""
    raw_mean:    dict[str, np.ndarray]          # layer_name -> mean gradient tensor
    per_episode: list[dict[str, np.ndarray]]    # per-batch L2-normalised gradient dicts
    variants:    dict[str, "GradientResult"] | None   # None for PPO; named dict for Rainbow
    metadata:    dict


# ---------------------------------------------------------------------------
# Analysis sub-dataclasses — one per analyzer, grouped into AnalysisResult
# ---------------------------------------------------------------------------

@dataclass
class GradientMetrics:
    """Scalar metrics derived from gradient analysis."""
    opposition_score:  float | None
    coherence_success: float | None
    coherence_failure: float | None
    magnitude_success: float | None
    magnitude_failure: float | None
    variants:          dict[str, GradientResult] | None  # None for PPO


@dataclass
class ActivationMetrics:
    """Scalar metrics derived from activation analysis."""
    separation:      float | None
    cosine_distance: float | None
    cluster_stats:   dict


@dataclass
class RSAMetrics:
    """Scalar metrics and artifacts from RSA analysis."""
    alignment: dict[str, float | None]   # {group_name: spearman_rho}
    rdm:       list[list[float]] | None
    labels:    list[str]


@dataclass
class AnalysisResult:
    """Final output of one checkpoint analysis. Logged to MLflow."""
    entity_id:       str
    checkpoint_step: int
    n_success:       int
    n_failure:       int
    threshold_eps:   object          # (float, float) lower/upper percentile boundaries
    gradients:       GradientMetrics
    activations:     ActivationMetrics
    rsa:             RSAMetrics
    achievement_observations: dict  # display_label -> frame count at this checkpoint
    metadata:        dict

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict. Handles numpy scalar types and nested dataclasses."""
        def _convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {kk: _convert(vv) for kk, vv in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return {k: _convert(v) for k, v in self.__dict__.items()}
