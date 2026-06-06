"""Episode collection runner. Saves EvaluationBatch to a compressed temp file."""
from __future__ import annotations

import time

import numpy as np
import gymnasium as gym
from tqdm import tqdm

import core.output as output
from core.data import EpisodeData, EvaluationBatch
from core.entity import Entity
from storage import temp_store
from training.achievement_tracker import AchievementTracker
from training.run_config import RunConfig


def run(entity: Entity, config: RunConfig,
        checkpoint_path: str, checkpoint_step: int) -> str:
    """Collect n_episodes and save to temp file. Returns the temp file path."""
    output.eval_start(checkpoint_step, config.n_steps, config.n_episodes, config.num_envs)
    t0 = time.monotonic()

    model   = entity.load_checkpoint(checkpoint_path, config.device)
    vec_env = gym.vector.AsyncVectorEnv(
        [(lambda i: lambda: entity.make_env(config.seed + i))(i)
         for i in range(config.num_envs)]
    )

    env_obs     = [[] for _ in range(config.num_envs)]
    env_actions = [[] for _ in range(config.num_envs)]
    env_rewards = [[] for _ in range(config.num_envs)]
    env_dones   = [[] for _ in range(config.num_envs)]

    trackers = [
        AchievementTracker(entity.achievement_names, entity.achievement_label_map)
        for _ in range(config.num_envs)
    ]

    episodes:            list[EpisodeData]          = []
    eps_scores:          list[float]                = []
    achievement_frames:  dict[str, list[np.ndarray]] = {}

    obs, _ = vec_env.reset()

    pbar = tqdm(total=config.n_episodes, desc="episodes", unit="ep", leave=False)
    while len(episodes) < config.n_episodes:
        # Buffer pre-step obs - these are what the agent saw when selecting each action.
        # Post-step obs from vec_env may be the reset obs when done=True.
        pre_obs = obs.copy()
        actions = [entity.select_action(model, pre_obs[i]) for i in range(config.num_envs)]
        obs, rewards, terminated, truncated, infos = vec_env.step(np.array(actions))

        for i in range(config.num_envs):
            done = bool(terminated[i]) or bool(truncated[i])

            step_info = _extract_env_info(infos, i, entity.achievement_names, done)
            trackers[i].step(pre_obs[i], step_info)

            env_obs[i].append(pre_obs[i].copy())
            env_actions[i].append(int(actions[i]))
            env_rewards[i].append(float(rewards[i]))
            env_dones[i].append(done)

            if done and len(episodes) < config.n_episodes:
                eps = entity.compute_eps(
                    step_info.get("achievements", {}),
                    step_info.get("inventory", {}),
                )

                ep = EpisodeData(
                    observations = np.array(env_obs[i],     dtype=np.uint8),
                    actions      = np.array(env_actions[i], dtype=np.int64),
                    rewards      = np.array(env_rewards[i], dtype=np.float32),
                    dones        = np.array(env_dones[i],   dtype=bool),
                )
                episodes.append(ep)
                eps_scores.append(eps)
                pbar.update(1)

                for label, frame in trackers[i].get_labelled_frames().items():
                    achievement_frames.setdefault(label, []).append(frame)

                trackers[i].reset()
                env_obs[i]     = []
                env_actions[i] = []
                env_rewards[i] = []
                env_dones[i]   = []

    pbar.close()
    vec_env.close()

    mean_eps = float(np.mean(eps_scores)) if eps_scores else 0.0
    output.eval_done(len(episodes), mean_eps, time.monotonic() - t0)

    batch = EvaluationBatch(
        entity_id          = entity.entity_id,
        checkpoint_path    = checkpoint_path,
        checkpoint_step    = checkpoint_step,
        episodes           = episodes,
        eps_scores         = np.array(eps_scores, dtype=np.float32),
        achievement_frames = achievement_frames,
    )
    return temp_store.save_batch(batch, config.temp_dir)


def _extract_env_info(infos: dict, env_idx: int,
                      achievement_names: list[str], done: bool) -> dict:
    """Extract per-env info from a gymnasium vectorised env info dict.

    Handles both final_info (on done) and mid-episode dict-of-arrays format.
    """
    # On terminal step, gymnasium's AsyncVectorEnv may store the last real info
    # in infos["final_info"][env_idx] - try that first when done.
    if done:
        final = infos.get("final_info")
        if final is not None and env_idx < len(final) and final[env_idx] is not None:
            return final[env_idx]

    # Mid-episode: infos values are arrays indexed by env
    raw_ach = infos.get("achievements", {})
    if isinstance(raw_ach, dict):
        achievements = {a: bool(raw_ach[a][env_idx]) for a in achievement_names
                        if a in raw_ach}
    else:
        achievements = {}

    raw_inv = infos.get("inventory", {})
    if isinstance(raw_inv, dict):
        inventory = {k: int(v[env_idx]) for k, v in raw_inv.items()}
    else:
        inventory = {}

    return {"achievements": achievements, "inventory": inventory}
