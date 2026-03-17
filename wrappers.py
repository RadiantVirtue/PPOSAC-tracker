"""Crafter environment wrappers.

CrafterGymnasiumWrapper  — adapts crafter.Env (old gym 4-tuple API) to gymnasium 5-tuple.
CrafterAchievementWrapper — detects newly-unlocked achievements each step,
                            applies shaped rewards, injects info["achievements"]
                            (bool dict) and info["eps"] (float).
make_crafter_env()       — factory used everywhere in place of _make_env(env_id).
"""

import warnings

import crafter
import gymnasium as gym
import numpy as np

from shared.achievements import (
    CRAFTER_ACHIEVEMENTS,
    CRAFTER_ACHIEVEMENT_REWARDS,
    compute_eps,
    count_achievements,
)

# Suppress crafter's old-gym deprecation warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")


class CrafterGymnasiumWrapper(gym.Env):
    """Thin adapter: crafter.Env (old gym API) → gymnasium 5-tuple API."""

    metadata = {"render_modes": []}

    def __init__(self, **crafter_kwargs):
        super().__init__()
        self._env = crafter.Env(**crafter_kwargs)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            # crafter.Env doesn't support seeding via reset; ignore gracefully
            pass
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, float(reward), bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class CrafterAchievementWrapper(gym.Wrapper):
    """Wraps CrafterGymnasiumWrapper to inject info["achievements"] and info["eps"].

    Crafter already returns cumulative per-episode achievement counts in info.
    This wrapper:
      - Converts counts → bool dict (achieved at least once this episode)
      - Diffs against previous step to detect newly-unlocked achievements
      - Adds shaped reward bonuses for each new unlock
      - Injects info["eps"] = count_achievements(ach) + 0.9 * 0.0
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_ach: dict = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_ach = {a: False for a in CRAFTER_ACHIEVEMENTS}
        info["achievements"] = dict(self._prev_ach)
        info["eps"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Crafter returns integer counts; convert to bool
        raw = info.get("achievements", {})
        cur_ach = {a: bool(raw.get(a, 0)) for a in CRAFTER_ACHIEVEMENTS}

        # Shaped reward: one-time bonus per newly unlocked achievement
        shaped = sum(
            CRAFTER_ACHIEVEMENT_REWARDS[a]
            for a in CRAFTER_ACHIEVEMENTS
            if cur_ach[a] and not self._prev_ach.get(a, False)
        )

        self._prev_ach = cur_ach
        info["achievements"] = cur_ach
        info["eps"] = float(compute_eps(count_achievements(cur_ach), 0.0))

        return obs, reward + shaped, terminated, truncated, info


def make_crafter_env(**crafter_kwargs) -> gym.Env:
    """Factory for a fully-wrapped Crafter gymnasium environment."""
    return CrafterAchievementWrapper(CrafterGymnasiumWrapper(**crafter_kwargs))
