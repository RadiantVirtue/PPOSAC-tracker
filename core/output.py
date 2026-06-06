"""Standardized pipeline progress output.

All stdout from the pipeline goes through this module.
Never import or call from entity code.
"""
from __future__ import annotations

import time
from datetime import datetime


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str) -> None:
    print(f"[{_ts()}] [{tag:<8}] {msg}", flush=True)


def training_start(entity_id: str, config) -> None:
    _log("TRAIN", (
        f"{entity_id} - "
        f"{config.n_steps:,} steps, "
        f"eval every {config.eval_every:,}, "
        f"seed {config.seed}"
    ))


def eval_start(step: int, n_steps: int, n_episodes: int, num_envs: int) -> None:
    _log("EVAL", (
        f"step {step:,}/{n_steps:,} - "
        f"collecting {n_episodes} episodes ({num_envs} envs)..."
    ))


def eval_done(n_episodes: int, mean_eps: float, duration_s: float) -> None:
    _log("EVAL", f"done in {duration_s:.0f}s - mean EPS {mean_eps:.2f}")


def analysis_start(n_success: int, n_failure: int, threshold: tuple) -> None:
    lo, hi = threshold
    _log("ANALYSIS", (
        f"partitioned: {n_success} success / {n_failure} failure "
        f"(EPS threshold {lo:.2f} / {hi:.2f})"
    ))


def analysis_stage(name: str, detail: str = "") -> None:
    msg = name if not detail else f"{name} - {detail}"
    _log("ANALYSIS", msg)


def analysis_done(result) -> None:
    parts = []
    if result.opposition_score is not None:
        parts.append(f"opp={result.opposition_score:+.2f}")
    if result.coherence_success is not None:
        parts.append(f"coh_s={result.coherence_success:.2f}")
    if result.coherence_failure is not None:
        parts.append(f"coh_f={result.coherence_failure:.2f}")
    if result.activation_separation is not None:
        parts.append(f"sep={result.activation_separation:.2f}")
    rsa_items = {k: v for k, v in result.rsa_alignment.items() if v is not None}
    for k, v in list(rsa_items.items())[:2]:
        parts.append(f"rsa_{k}={v:.2f}")
    _log("RESULT", "  ".join(parts) if parts else "(no metrics)")


def mlflow_logged(step: int, run_id: str) -> None:
    _log("MLflow", f"step {step:,} -> run {run_id[:8]}")


def skipped_checkpoint(step: int, n_success: int, n_failure: int) -> None:
    _log("ANALYSIS", (
        f"step {step:,} skipped - "
        f"groups too small ({n_success} success, {n_failure} failure)"
    ))
