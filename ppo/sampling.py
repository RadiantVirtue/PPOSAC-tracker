import _rl_path  # noqa: F401  — adds rl-starter-files to sys.path
from collections import namedtuple

import gymnasium as gym
import minigrid  # noqa: F401  — registers MiniGrid envs with gymnasium
import numpy as np
import torch
from tqdm import tqdm

from wrappers import DoorKeyAchievementWrapper, KeyCorridorAchievementWrapper
from shared.thresholding import partition_episodes

EpisodeData = namedtuple(
    "EpisodeData", ["observations", "actions", "rewards", "dones"]
)


# Load a frozen PPO agent (ACModel) from a checkpoint. Returns (ACModelWrapper, episode_count).
def load_ppo_agent(checkpoint_path, env, device="cuda"):
    from model import ACModel
    from utils.format import get_obss_preprocessor
    from shared.networks import ACModelWrapper

    obs_space, _ = get_obss_preprocessor(env.observation_space)
    acmodel = ACModel(obs_space, env.action_space)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    acmodel.load_state_dict(ckpt["agent_state_dict"])
    acmodel.to(device)
    acmodel.eval()

    wrapper = ACModelWrapper(acmodel).to(device)
    wrapper.eval()
    return wrapper, ckpt.get("episode_count", ckpt.get("global_step"))


def evaluate_frozen_policy(
    checkpoint_path, env_id, n_episodes=1000, device="cuda"
):
    """Run n_episodes with a frozen PPO policy. Returns (episodes, eps_scores).

    Observations stored in EpisodeData are (H, W, C) image arrays as float32 tensors,
    compatible with ACModelWrapper.forward() and extract_activations().
    """
    env = gym.make(env_id)
    if "DoorKey" in env_id:
        env = DoorKeyAchievementWrapper(env)
    elif "KeyCorridor" in env_id:
        env = KeyCorridorAchievementWrapper(env)
    else:
        raise ValueError(f"No achievement wrapper for env_id: {env_id}")
    # No ImgObsWrapper — ACModel accepts raw MiniGrid dict observations

    agent, _step = load_ppo_agent(checkpoint_path, env, device)

    episodes = []
    eps_scores = []

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes", unit="ep"):
        obs, info = env.reset()
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        done = False
        while not done:
            # obs is a dict {"image": (H,W,C) array, "mission": str}
            obs_img = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.get_action(obs_img, deterministic=False)
            action_int = action.cpu().item()

            ep_obs.append(obs["image"])   # store just the image array
            ep_actions.append(action_int)

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            ep_rewards.append(reward)
            ep_dones.append(done)

        eps_scores.append(info["eps"])
        episodes.append(
            EpisodeData(
                observations=torch.tensor(
                    np.array(ep_obs), dtype=torch.float32
                ),
                actions=torch.tensor(ep_actions, dtype=torch.long),
                rewards=torch.tensor(ep_rewards, dtype=torch.float32),
                dones=torch.tensor(ep_dones, dtype=torch.float32),
            )
        )

    env.close()
    return episodes, eps_scores


# Partition episodes into success / failure.
# In percentile mode, raw episode returns (sum of shaped rewards) are used as
# the ranking score rather than EPS, and the middle episodes are discarded.
def partition(episodes, eps_scores, mode="eps", percentile_x=25):
    if mode == "percentile":
        scores = [ep.rewards.sum().item() for ep in episodes]
    else:
        scores = eps_scores
    return partition_episodes(episodes, scores, mode=mode, percentile_x=percentile_x)
