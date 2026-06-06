"""Canonical data structures shared across the entire pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeData:
    """One episode's trajectory in canonical format."""
    observations: np.ndarray   # (T, H, W, C) uint8 — always; never float32
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
    variants:    dict[str, "GradientResult"]    # empty dict {} for PPO
    metadata:    dict


@dataclass
class ActivationResult:
    """Activation analysis output for success + failure groups combined."""
    activations:    np.ndarray   # (N_total, D) float32
    projected:      np.ndarray   # (N_total, 2) float32 — UMAP 2-D
    cluster_labels: np.ndarray   # (N_total,) int — HDBSCAN (-1 = noise)
    centroids:      dict         # {"success": (D,), "failure": (D,)}
    cluster_stats:  dict


@dataclass
class AnalysisResult:
    """Final output of one checkpoint analysis. Logged to MLflow."""
    entity_id:                     str
    checkpoint_step:               int
    split_mode:                    str
    n_success:                     int
    n_failure:                     int
    threshold_eps:                 object          # float or (float, float)
    opposition_score:              float | None
    coherence_success:             float | None
    coherence_failure:             float | None
    gradient_magnitude_success:    float | None
    gradient_magnitude_failure:    float | None
    activation_separation:         float | None
    activation_cosine_distance:    float | None
    cluster_stats:                 dict
    rsa_alignment:                 dict            # {group_name: float | None}
    rsa_rdm:                       object | None   # list[list[float]] or None
    rsa_labels:                    list[str]
    gradient_variants:             dict            # Rainbow IS-weighted extras; {} for PPO
    metadata:                      dict

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict. Handles numpy scalar types."""
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, dict):
                d[k] = {kk: _convert(vv) for kk, vv in v.items()}
            else:
                d[k] = _convert(v)
        return d
