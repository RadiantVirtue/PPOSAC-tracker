import os
import pickle

import numpy as np


# replay buffer with per-transition episode tagging for analysis
class TaggedReplayBuffer:

    def __init__(self, buffer_size, obs_shape, n_actions, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros(
            (buffer_size, *obs_shape), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (buffer_size, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Tagging fields
        self.episode_ids = np.zeros(buffer_size, dtype=np.int64)
        self.episode_returns = np.full(buffer_size, np.nan, dtype=np.float32)
        self.episode_eps = np.full(buffer_size, np.nan, dtype=np.float32)

    # store a single transition (episode_return / eps filled later)
    def add(self, obs, next_obs, action, reward, done, episode_id):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.episode_ids[self.pos] = episode_id
        self.episode_returns[self.pos] = np.nan
        self.episode_eps[self.pos] = np.nan

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    # fill episode-level tags for all transitions of the given episode_id
    def backfill_episode(self, episode_id, episode_return, episode_eps):
        size = self.buffer_size if self.full else self.pos
        mask = self.episode_ids[:size] == episode_id
        self.episode_returns[:size][mask] = episode_return
        self.episode_eps[:size][mask] = episode_eps

    # sampling

    # uniform random sample for SAC training updates
    def sample(self, batch_size):
        import torch

        size = self.buffer_size if self.full else self.pos
        idxs = np.random.randint(0, size, size=batch_size)
        return {
            "observations": torch.tensor(self.observations[idxs]).to(
                self.device
            ),
            "next_observations": torch.tensor(
                self.next_observations[idxs]
            ).to(self.device),
            "actions": torch.tensor(self.actions[idxs]).to(self.device),
            "rewards": torch.tensor(self.rewards[idxs]).to(self.device),
            "dones": torch.tensor(self.dones[idxs]).to(self.device),
        }

    # analysis helpers

    # return all transitions that have been backfilled with EPS
    def get_all_valid(self):
        size = self.buffer_size if self.full else self.pos
        valid = ~np.isnan(self.episode_eps[:size])
        return {
            "observations": self.observations[:size][valid],
            "next_observations": self.next_observations[:size][valid],
            "actions": self.actions[:size][valid],
            "rewards": self.rewards[:size][valid],
            "dones": self.dones[:size][valid],
            "episode_eps": self.episode_eps[:size][valid],
            "episode_returns": self.episode_returns[:size][valid],
        }

    # serialisation

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "episode_ids": self.episode_ids,
            "episode_returns": self.episode_returns,
            "episode_eps": self.episode_eps,
            "pos": self.pos,
            "full": self.full,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, device="cuda"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obs_shape = data["observations"].shape[1:]
        buf = cls(data["observations"].shape[0], obs_shape, 1, device)
        for key in data:
            setattr(buf, key, data[key])
        return buf


# stores complete episodes as ordered sequences for offline analysis
class EpisodeStore:

    def __init__(self):
        self.episodes = []

    def add_episode(
        self,
        obs_list,
        action_list,
        reward_list,
        done_list,
        episode_return,
        episode_eps,
        achievements,
    ):
        self.episodes.append(
            {
                "observations": np.array(obs_list),
                "actions": np.array(action_list),
                "rewards": np.array(reward_list),
                "dones": np.array(done_list),
                "return": episode_return,
                "eps": episode_eps,
                "achievements": achievements,
            }
        )

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.episodes, f)

    @classmethod
    def load(cls, path):
        store = cls()
        with open(path, "rb") as f:
            store.episodes = pickle.load(f)
        return store
