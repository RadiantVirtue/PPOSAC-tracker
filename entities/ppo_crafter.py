"""PPO + Crafter entity.

Implements the Entity protocol for SB3 PPO trained on the Crafter environment.
Absorbs: wrappers.py, ppo/sampling.py (entity methods), ppo/gradients.py.
"""
from __future__ import annotations

import warnings

import crafter
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from core.data import EpisodeData, GradientResult
from core.gradient_utils import OnlineGradientAggregator
from entities.definitions.crafter import (
    ACHIEVEMENT_NAMES,
    ACHIEVEMENT_LABEL_MAP,
    ACHIEVEMENT_GROUPS,
    ACHIEVEMENT_MATERIALS,
)

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*is not within the observation space.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*Box bound precision.*", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")

ENTITY_ID  = "ppo_crafter"
OBS_SHAPE  = (64, 64, 3)
N_ACTIONS  = 17
HOOK_LAYER = "features_extractor.linear"   # SB3 NatureCNN 64-dim output, relative to policy
EPS_WEIGHT = 0.9
GAE_GAMMA  = 0.99
GAE_LAMBDA = 0.95


class PPOCrafter:
    """PPO (Stable-Baselines3) trained on Crafter."""

    entity_id             = ENTITY_ID
    obs_shape             = OBS_SHAPE
    n_actions             = N_ACTIONS
    hook_layer            = HOOK_LAYER
    achievement_names     = ACHIEVEMENT_NAMES
    achievement_label_map = ACHIEVEMENT_LABEL_MAP
    achievement_groups    = ACHIEVEMENT_GROUPS

    def make_env(self, seed: int) -> gym.Env:
        """Return CrafterAchievementWrapper(CrafterGymnasiumWrapper(seed))."""
        return CrafterAchievementWrapper(CrafterGymnasiumWrapper(seed=seed))

    def load_checkpoint(self, path: str, device: str) -> PPO:
        """Load a frozen SB3 PPO model from checkpoint."""
        return PPO.load(path, device=device)

    def select_action(self, model: PPO, obs: np.ndarray,
                      deterministic: bool = True) -> int:
        """Return model.predict(obs[None], deterministic=deterministic)[0].item()."""
        action, _ = model.predict(obs[None], deterministic=deterministic)
        return int(action[0])

    def compute_eps(self, achievements: dict[str, bool], inventory: dict) -> float:
        """EPS = count_achievements + EPS_WEIGHT * materials_fraction.

        Used for episode partitioning only. Not related to achievement frame logging.
        """
        n_achieved     = sum(1 for v in achievements.values() if v)
        materials_frac = _materials_fraction(achievements, inventory)
        return float(n_achieved + EPS_WEIGHT * materials_frac)

    def preprocess_obs(self, model: PPO, obs_np: np.ndarray,
                       device: str) -> torch.Tensor:
        """Return policy.obs_to_tensor(obs_np) → (N, C, H, W) float32 on device."""
        policy = model.policy.to(device)
        obs_tensor, _ = policy.obs_to_tensor(obs_np)
        return obs_tensor

    def get_policy(self, model: PPO) -> object:
        """Return model.policy - the hookable ActorCriticPolicy."""
        return model.policy

    def compute_gradients(self, model: PPO, episodes: list[EpisodeData],
                          device: str, batch_size: int = 10) -> GradientResult:
        """GAE-weighted policy gradient via SB3 policy.evaluate_actions().

        Loss = -(advantages * log_probs).mean(). Backward per episode.
        Returns GradientResult with variants={} (no IS weighting for PPO).
        """
        policy = model.policy.to(device)
        policy.set_training_mode(True)

        overall_agg = OnlineGradientAggregator(list(policy.named_parameters()))
        per_episode_grads = []

        for i in range(0, len(episodes), batch_size):
            batch = episodes[i: i + batch_size]
            batch_agg = OnlineGradientAggregator(list(policy.named_parameters()))

            for episode in batch:
                policy.zero_grad()

                obs_tensor, _ = policy.obs_to_tensor(episode.observations)
                actions_tensor = torch.tensor(episode.actions, dtype=torch.long).to(device)

                values, log_probs, _ = policy.evaluate_actions(obs_tensor, actions_tensor)

                rewards = torch.tensor(episode.rewards, dtype=torch.float32)
                dones   = torch.tensor(episode.dones,   dtype=torch.float32)
                advantages = _compute_gae(rewards, values.detach().cpu(), dones).to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                pg_loss = -(advantages * log_probs).mean()
                pg_loss.backward()

                batch_agg.accumulate(list(policy.named_parameters()))
                overall_agg.accumulate(list(policy.named_parameters()))
                policy.zero_grad()

            per_episode_grads.append(batch_agg.l2_normalized())

        policy.set_training_mode(False)

        return GradientResult(
            raw_mean    = overall_agg.mean_gradient(),
            per_episode = per_episode_grads,
            variants    = {},
            metadata    = {"n_transitions": sum(len(ep.rewards) for ep in episodes)},
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                 dones: torch.Tensor,
                 gamma: float = GAE_GAMMA, lam: float = GAE_LAMBDA) -> torch.Tensor:
    """Compute GAE advantages for a single episode."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value        = 0.0 if t == T - 1 else values[t + 1].item()
        next_non_terminal = 1.0 - dones[t].item()
        delta             = rewards[t].item() + gamma * next_value * next_non_terminal - values[t].item()
        advantages[t]     = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    return advantages


def _materials_fraction(achievements: dict[str, bool], inventory: dict) -> float:
    """Fraction of materials held toward the nearest unachieved craftable achievement."""
    best = 0.0
    for ach, required in ACHIEVEMENT_MATERIALS.items():
        if achievements.get(ach, False):
            continue
        total = sum(required.values())
        if total == 0:
            continue
        held = sum(min(int(inventory.get(mat, 0)), qty) for mat, qty in required.items())
        best = max(best, held / total)
    return best


# ---------------------------------------------------------------------------
# Environment wrappers  (absorbed from wrappers.py)
# ---------------------------------------------------------------------------

class CrafterGymnasiumWrapper(gym.Env):
    """Adapts crafter.Env (old gym 4-tuple API) to gymnasium 5-tuple.

    seed controls the RNG used to draw a fresh world seed every episode,
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
    """Injects info["achievements"] (bool dict) and info["inventory"] each step.

    Crafter already returns cumulative per-episode achievement counts in info.
    This wrapper converts counts → bool dict. Reward is passed through unchanged.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["achievements"] = {a: False for a in ACHIEVEMENT_NAMES}
        info["inventory"]    = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = info.get("achievements", {})
        info["achievements"] = {a: bool(raw.get(a, 0)) for a in ACHIEVEMENT_NAMES}
        return obs, reward, terminated, truncated, info
