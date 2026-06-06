"""Temp storage for EvaluationBatch. Files are deleted after analysis."""
from __future__ import annotations

import os
import time

import numpy as np

from core.data import EpisodeData, EvaluationBatch


def save_batch(batch: EvaluationBatch, temp_dir: str) -> str:
    """Compress and save batch to temp_dir. Returns absolute path.

    File format ({entity_id}_step{step}_{ts}.npz):
        entity_id, checkpoint_path, checkpoint_step  - str/int scalars
        n_episodes                                   - int scalar
        eps_scores          float32 (N,)
        episode_lengths     int32   (N,)
        observations        uint8   (N_total, H, W, C)
        actions             int64   (N_total,)
        rewards             float32 (N_total,)
        dones               bool    (N_total,)
        ach_labels          U256    (K,)
        ach_frame_counts    int32   (K,)
        ach_frames          uint8   (sum(frame_counts), H, W, C)
    """
    os.makedirs(temp_dir, exist_ok=True)
    ts = int(time.time())
    filename = f"{batch.entity_id}_step{batch.checkpoint_step}_{ts}.npz"
    path = os.path.abspath(os.path.join(temp_dir, filename))

    episode_lengths = np.array([len(ep.rewards) for ep in batch.episodes], dtype=np.int32)
    all_obs     = np.concatenate([ep.observations for ep in batch.episodes], axis=0)
    all_actions = np.concatenate([ep.actions      for ep in batch.episodes], axis=0)
    all_rewards = np.concatenate([ep.rewards      for ep in batch.episodes], axis=0)
    all_dones   = np.concatenate([ep.dones        for ep in batch.episodes], axis=0)

    ach_labels = np.array(list(batch.achievement_frames.keys()), dtype="U256")
    ach_frame_counts = np.array(
        [len(frames) for frames in batch.achievement_frames.values()], dtype=np.int32
    )
    if len(batch.achievement_frames) > 0:
        ach_frames = np.concatenate(
            [np.stack(frames) for frames in batch.achievement_frames.values()], axis=0
        )
    else:
        ach_frames = np.zeros((0, *batch.episodes[0].observations.shape[1:]), dtype=np.uint8)

    np.savez_compressed(
        path,
        entity_id         = np.array(batch.entity_id),
        checkpoint_path   = np.array(batch.checkpoint_path),
        checkpoint_step   = np.array(batch.checkpoint_step, dtype=np.int64),
        n_episodes        = np.array(len(batch.episodes),   dtype=np.int64),
        eps_scores        = batch.eps_scores.astype(np.float32),
        episode_lengths   = episode_lengths,
        observations      = all_obs.astype(np.uint8),
        actions           = all_actions.astype(np.int64),
        rewards           = all_rewards.astype(np.float32),
        dones             = all_dones.astype(bool),
        ach_labels        = ach_labels,
        ach_frame_counts  = ach_frame_counts,
        ach_frames        = ach_frames.astype(np.uint8),
    )
    return path


def load_batch(path: str) -> EvaluationBatch:
    """Load and reconstruct batch. Splits concatenated arrays using episode_lengths."""
    data = np.load(path, allow_pickle=False)

    entity_id       = str(data["entity_id"])
    checkpoint_path = str(data["checkpoint_path"])
    checkpoint_step = int(data["checkpoint_step"])
    eps_scores      = data["eps_scores"]
    ep_lengths      = data["episode_lengths"]
    all_obs         = data["observations"]
    all_actions     = data["actions"]
    all_rewards     = data["rewards"]
    all_dones       = data["dones"]

    episodes: list[EpisodeData] = []
    offset = 0
    for length in ep_lengths:
        length = int(length)
        episodes.append(EpisodeData(
            observations = all_obs    [offset: offset + length],
            actions      = all_actions[offset: offset + length],
            rewards      = all_rewards[offset: offset + length],
            dones        = all_dones  [offset: offset + length],
        ))
        offset += length

    ach_labels       = [str(s) for s in data["ach_labels"]]
    ach_frame_counts = data["ach_frame_counts"].tolist()
    ach_frames_all   = data["ach_frames"]

    achievement_frames: dict[str, list[np.ndarray]] = {}
    frame_offset = 0
    for label, count in zip(ach_labels, ach_frame_counts):
        count = int(count)
        frames = [ach_frames_all[frame_offset + j] for j in range(count)]
        achievement_frames[label] = frames
        frame_offset += count

    return EvaluationBatch(
        entity_id          = entity_id,
        checkpoint_path    = checkpoint_path,
        checkpoint_step    = checkpoint_step,
        episodes           = episodes,
        eps_scores         = eps_scores,
        achievement_frames = achievement_frames,
    )


def delete_batch(path: str) -> None:
    """Delete the temp file. Called by analysis/pipeline.py on successful completion."""
    if os.path.exists(path):
        os.remove(path)


def estimate_size_mb(batch: EvaluationBatch) -> float:
    """Estimate uncompressed size in MB before saving."""
    n_obs = sum(len(ep.observations) for ep in batch.episodes)
    h, w, c = batch.episodes[0].observations.shape[1:]
    obs_bytes    = n_obs * h * w * c               # uint8
    actions_bytes = n_obs * 8                       # int64
    rewards_bytes = n_obs * 4                       # float32
    dones_bytes   = n_obs * 1                       # bool
    n_frames      = sum(len(f) for f in batch.achievement_frames.values())
    frames_bytes  = n_frames * h * w * c           # uint8
    total = obs_bytes + actions_bytes + rewards_bytes + dones_bytes + frames_bytes
    return total / (1024 ** 2)
