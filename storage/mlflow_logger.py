"""MLflow logging for AnalysisResult. One MLflow experiment per entity_id."""
from __future__ import annotations

import json
import os

import mlflow

from core.data import AnalysisResult
from training.run_config import RunConfig


def init_run(entity_id: str, config: RunConfig) -> str:
    """Create or resume an MLflow experiment for entity_id. Returns the run_id."""
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(entity_id)

    run = mlflow.start_run(
        run_name=f"seed_{config.seed}_{_timestamp()}",
        tags={"entity_id": entity_id},
    )
    mlflow.log_params({
        "entity_id":   entity_id,
        "split_mode":  config.split_mode,
        "n_episodes":  config.n_episodes,
        "seed":        config.seed,
        "num_envs":    config.num_envs,
        "device":      config.device,
        "percentile_x": config.percentile_x,
    })
    return run.info.run_id


def log(result: AnalysisResult, run_id: str, step: int) -> None:
    """Log metrics and artifacts for one checkpoint to the active MLflow run.

    Relies on the run started by init_run() still being active in this process.
    """
    metrics = {
        k: v for k, v in {
            "opposition_score":           result.opposition_score,
            "coherence_success":          result.coherence_success,
            "coherence_failure":          result.coherence_failure,
            "gradient_magnitude_success": result.gradient_magnitude_success,
            "gradient_magnitude_failure": result.gradient_magnitude_failure,
            "activation_separation":      result.activation_separation,
            "activation_cosine_distance": result.activation_cosine_distance,
            "n_success":                  float(result.n_success),
            "n_failure":                  float(result.n_failure),
        }.items()
        if v is not None
    }

    for group_name, alignment in result.rsa_alignment.items():
        if alignment is not None:
            metrics[f"rsa_alignment_{group_name}"] = alignment

    mlflow.log_metrics(metrics, step=step)

    if result.rsa_rdm is not None:
        rdm_data = {"rdm": result.rsa_rdm, "labels": result.rsa_labels, "step": step}
        artifact_path = f"rdm_step{step}.json"
        tmp_path = os.path.join(os.environ.get("TEMP", "/tmp"), artifact_path)
        with open(tmp_path, "w") as f:
            json.dump(rdm_data, f, indent=2)
        mlflow.log_artifact(tmp_path, artifact_path="rdm")
        os.remove(tmp_path)


def finalise_run(run_id: str) -> None:
    """End the MLflow run."""
    mlflow.end_run()


def _timestamp() -> str:
    import time
    return str(int(time.time()))
