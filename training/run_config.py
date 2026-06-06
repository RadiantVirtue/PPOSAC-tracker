"""Unified run configuration. Single source of truth for all tunable parameters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """All parameters for one training + analysis run."""
    entity_id:   str
    n_steps:     int
    eval_every:  int

    n_episodes:  int   = 500
    num_envs:    int   = 16
    device:      str   = "cpu"
    seed:        int   = 0

    split_mode:        str         = "percentile"
    percentile_x:      int         = 25
    fixed_thresholds:  tuple | None = None
    eps_weight:        float        = 0.9

    temp_dir:            str = "temp/"
    mlflow_tracking_uri: str = "mlruns/"
