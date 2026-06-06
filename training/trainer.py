"""Training orchestrator. Calls entity.train() and triggers eval + analysis at each checkpoint."""
from __future__ import annotations

import analysis.pipeline as analysis_pipeline
import storage.mlflow_logger as mlflow_logger
import training.eval_runner as eval_runner
from core.entity import Entity
from training.run_config import RunConfig


def run(entity: Entity, config: RunConfig) -> None:
    """Run training, evaluation, and analysis for the given entity and config."""
    run_id = mlflow_logger.init_run(entity.entity_id, config)

    entity.train(
        n_steps       = config.n_steps,
        on_checkpoint = lambda step, ckpt_path: _on_checkpoint(
            entity, config, run_id, step, ckpt_path),
    )

    mlflow_logger.finalise_run(run_id)


def _on_checkpoint(entity: Entity, config: RunConfig, run_id: str,
                   step: int, ckpt_path: str) -> None:
    """Run eval + analysis for one checkpoint. Temp file is deleted inside pipeline.run."""
    temp_path = eval_runner.run(entity, config, ckpt_path, step)
    analysis_pipeline.run(
        entity,
        temp_path,
        config,
        reference_stimuli=None,
        mlflow_run_id=run_id,
    )
