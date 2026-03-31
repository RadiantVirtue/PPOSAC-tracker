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
    compute_eps,
    compute_materials_fraction,
    count_achievements,
)

# Suppress crafter's old-gym deprecation warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
# Suppress gymnasium upgrade deprecation warnings
warnings.filterwarnings("ignore", message=".*is not within the observation space.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Box bound precision.*", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")


class CrafterGymnasiumWrapper(gym.Env):
    """Thin adapter: crafter.Env (old gym API) → gymnasium 5-tuple API.

    seed controls the RNG used to draw a fresh world seed for every episode,
    giving reproducible but varied world layouts across episodes.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None, **crafter_kwargs):
        super().__init__()
        self._env = crafter.Env(**crafter_kwargs)
        self._rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

    def reset(self, *, seed=None, **kwargs):
        # Draw a fresh world seed from this env's RNG each episode
        episode_seed = int(self._rng.integers(0, 2**31))
        self._env._seed = episode_seed
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
      - Injects info["eps"] = count_achievements(ach)
    Reward is passed through unchanged (no shaping).
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["achievements"] = {a: False for a in CRAFTER_ACHIEVEMENTS}
        info["eps"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Crafter returns integer counts; convert to bool
        raw = info.get("achievements", {})
        cur_ach = {a: bool(raw.get(a, 0)) for a in CRAFTER_ACHIEVEMENTS}
        info["achievements"] = cur_ach
        inventory = info.get("inventory", {})
        materials_frac = compute_materials_fraction(cur_ach, inventory)
        info["eps"] = float(compute_eps(count_achievements(cur_ach), materials_frac))

        return obs, reward, terminated, truncated, info


def make_crafter_env(seed=None, **crafter_kwargs) -> gym.Env:
    """Factory for a fully-wrapped Crafter gymnasium environment."""
    return CrafterAchievementWrapper(CrafterGymnasiumWrapper(seed=seed, **crafter_kwargs))
