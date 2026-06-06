"""MLflow reader: loads run metric history into a RunData dataclass.

Usage:
    from graphing.loader import load_run
    run_data = load_run("ppo_crafter")               # latest run
    run_data = load_run("ppo_crafter", run_id="abc") # specific run
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient


# Metric keys produced by AnalysisResult (see storage/mlflow_logger.py)
STANDARD_METRIC_KEYS = [
    "opposition_score",
    "coherence_success",
    "coherence_failure",
    "gradient_magnitude_success",
    "gradient_magnitude_failure",
    "activation_separation",
    "activation_cosine_distance",
    "n_success",
    "n_failure",
]

RSA_METRIC_PREFIX = "rsa_alignment_"
ACHIEVEMENT_OBS_PREFIX = "achievement_obs_"


@dataclass
class RunData:
    """All metric history for one MLflow run, ready for plotting.

    Fields:
        entity_id:               MLflow experiment name (= entity ID)
        run_id:                  Full MLflow run ID
        params:                  Run-level params logged at init (n_episodes, seed, ...)
        steps:                   Sorted list of checkpoint steps that have logged metrics
        metrics:                 {metric_name: list[float]} aligned to steps (NaN if absent)
        rsa_groups:              Group names found, e.g. ["crafting", "fighting", ...]
        achievement_first_steps: {display_label: first checkpoint_step where obs > 0}
    """
    entity_id:               str
    run_id:                  str
    params:                  dict
    steps:                   list[int]
    metrics:                 dict[str, list[float]]
    rsa_groups:              list[str]
    achievement_first_steps: dict[str, int] = field(default_factory=dict)


def load_run(
    experiment_name: str,
    run_id: str | None = None,
    tracking_uri: str = "mlruns/",
) -> RunData:
    """Load all metric history for an MLflow run.

    Args:
        experiment_name: MLflow experiment name (= entity_id, e.g. "ppo_crafter")
        run_id:          Specific run ID. If None, uses the most recent finished run.
        tracking_uri:    Path to the MLflow tracking store.

    Returns:
        RunData with all metrics aligned to a common step list.

    Raises:
        ValueError: if the experiment or run is not found.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found in {tracking_uri}")

    if run_id is None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    params = dict(run.data.params)

    # Discover all metric keys for this run
    all_metric_keys = list(run.data.metrics.keys())
    rsa_groups = [
        k[len(RSA_METRIC_PREFIX):]
        for k in all_metric_keys if k.startswith(RSA_METRIC_PREFIX)
    ]
    ach_obs_keys = [k for k in all_metric_keys if k.startswith(ACHIEVEMENT_OBS_PREFIX)]

    keys_to_load = (
        STANDARD_METRIC_KEYS
        + [f"{RSA_METRIC_PREFIX}{g}" for g in rsa_groups]
        + ach_obs_keys
    )

    # Load full history for each metric; build a union of all steps
    raw: dict[str, dict[int, float]] = {}
    all_steps: set[int] = set()
    for key in keys_to_load:
        history = client.get_metric_history(run_id, key)
        step_map = {m.step: m.value for m in history}
        raw[key] = step_map
        all_steps.update(step_map.keys())

    steps = sorted(all_steps)

    # Align every metric to the common step list (NaN where absent)
    metrics: dict[str, list[float]] = {}
    for key in keys_to_load:
        step_map = raw.get(key, {})
        metrics[key] = [step_map.get(s, float("nan")) for s in steps]

    # Derive first-unlock step for each achievement label
    achievement_first_steps: dict[str, int] = {}
    for obs_key in ach_obs_keys:
        label = _label_from_obs_key(obs_key)
        step_map = raw.get(obs_key, {})
        for s in steps:
            if step_map.get(s, 0.0) >= 1.0:
                achievement_first_steps[label] = s
                break

    return RunData(
        entity_id               = experiment_name,
        run_id                  = run_id,
        params                  = params,
        steps                   = steps,
        metrics                 = metrics,
        rsa_groups              = rsa_groups,
        achievement_first_steps = achievement_first_steps,
    )


def load_rdm_artifact(run_id: str, step: int | None = None,
                      tracking_uri: str = "mlruns/") -> dict | None:
    """Load an RDM artifact from MLflow.

    Args:
        run_id: MLflow run ID
        step:   Checkpoint step. If None, loads the artifact from the last checkpoint.

    Returns:
        Dict with keys "rdm" (list[list[float]]), "labels" (list[str]), "step" (int).
        None if no RDM artifacts exist.
    """
    import json, os
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    artifacts = client.list_artifacts(run_id, path="rdm")
    if not artifacts:
        return None

    if step is None:
        # Pick the artifact with the highest step number
        def _step_of(a):
            name = os.path.splitext(os.path.basename(a.path))[0]
            try:
                return int(name.replace("rdm_step", ""))
            except ValueError:
                return -1
        artifacts = sorted(artifacts, key=_step_of)
        target = artifacts[-1]
    else:
        target = next((a for a in artifacts if f"step{step}" in a.path), None)
        if target is None:
            return None

    local_path = client.download_artifacts(run_id, target.path)
    with open(local_path) as f:
        return json.load(f)


def _label_from_obs_key(obs_key: str) -> str:
    """Convert 'achievement_obs_eat_plant' → 'Eat Plant'."""
    suffix = obs_key[len(ACHIEVEMENT_OBS_PREFIX):]
    return suffix.replace("_", " ").title()
