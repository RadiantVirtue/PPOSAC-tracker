"""Analysis pipeline. Loads temp batch, partitions, analyses, logs to MLflow, deletes temp."""
from __future__ import annotations

from typing import Any

import analysis.activation_analyzer as activation_analyzer
import analysis.gradient_analyzer as gradient_analyzer
import analysis.rsa_analyzer as rsa_analyzer
import core.metrics as metrics
import storage.mlflow_logger as mlflow_logger
import storage.temp_store as temp_store
from core.data import AnalysisResult
from core.entity import Entity
from core.thresholding import partition_episodes
from training.run_config import RunConfig

MIN_GROUP_SIZE = 10  # minimum episodes per partition to run analysis


def run(
    entity: Entity,
    temp_batch_path: str,
    config: RunConfig,
    reference_stimuli: frozenset | None,
    mlflow_run_id: str,
) -> AnalysisResult | None:
    """Run gradient, activation, and RSA analysis for one checkpoint.

    Returns None if either partition group is smaller than MIN_GROUP_SIZE.
    Deletes temp_batch_path on successful completion.
    """
    batch = temp_store.load_batch(temp_batch_path)
    model = entity.load_checkpoint(batch.checkpoint_path, config.device)

    success_eps, failure_eps, threshold = partition_episodes(
        batch.episodes,
        batch.eps_scores,
        mode              = config.split_mode,
        percentile_x      = config.percentile_x,
        fixed_thresholds  = config.fixed_thresholds,
    )

    if len(success_eps) < MIN_GROUP_SIZE or len(failure_eps) < MIN_GROUP_SIZE:
        temp_store.delete_batch(temp_batch_path)
        return None

    grad_success = gradient_analyzer.analyze(entity, model, success_eps, config.device)
    grad_failure  = gradient_analyzer.analyze(entity, model, failure_eps,  config.device)
    act_result   = activation_analyzer.analyze(
                       entity, model, success_eps, failure_eps, config.device)
    rsa_result   = rsa_analyzer.analyze(
                       entity, model, batch.achievement_frames,
                       reference_stimuli, config.device)

    result = AnalysisResult(
        entity_id         = entity.entity_id,
        checkpoint_step   = batch.checkpoint_step,
        split_mode        = config.split_mode,
        n_success         = len(success_eps),
        n_failure         = len(failure_eps),
        threshold_eps     = threshold,
        opposition_score  = metrics.opposition_score(
                                grad_success.raw_mean, grad_failure.raw_mean),
        coherence_success = metrics.coherence(grad_success.per_episode),
        coherence_failure = metrics.coherence(grad_failure.per_episode),
        gradient_magnitude_success = metrics.gradient_magnitude(grad_success.raw_mean),
        gradient_magnitude_failure = metrics.gradient_magnitude(grad_failure.raw_mean),
        activation_separation      = metrics.activation_separation(
                                         act_result.centroids["success"],
                                         act_result.centroids["failure"]),
        activation_cosine_distance = metrics.centroid_cosine_distance(
                                         act_result.centroids["success"],
                                         act_result.centroids["failure"]),
        cluster_stats     = act_result.cluster_stats,
        rsa_alignment     = rsa_result["alignments"],
        rsa_rdm           = rsa_result["rdm"],
        rsa_labels        = rsa_result["labels"],
        gradient_variants = grad_success.variants,
        metadata          = {
            "n_transitions_success": grad_success.metadata.get("n_transitions"),
            "n_transitions_failure": grad_failure.metadata.get("n_transitions"),
            "n_stimuli":             rsa_result["n_stimuli"],
        },
    )

    mlflow_logger.log(result, mlflow_run_id, step=batch.checkpoint_step)
    temp_store.delete_batch(temp_batch_path)
    return result
