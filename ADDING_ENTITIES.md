# Adding a new entity

An entity = one algorithm + one environment. Each combination gets its own file in `entities/`.

The pipeline (`training/`, `analysis/`, `storage/`) is entirely HARD-CODED and never changes when you add a new entity. You only add SOFT-CODED files.

---

## When to add what

| Situation | Action |
|-----------|--------|
| New algorithm on an existing environment | Create `entities/{algorithm}_{environment}.py`, import from existing `entities/definitions/{environment}.py` |
| New environment (any algorithm) | First create `entities/definitions/{environment}.py`, then create the entity file |
| New algorithm on a new environment | Create both |

---

## Step 1 - Create the environment definitions file

**Skip if `entities/definitions/{environment}.py` already exists.**

**File:** `entities/definitions/{environment}.py`

```python
"""{environment} achievement definitions.

Imported by all entities that use this environment. Contains no logic -
only the data that describes what achievements exist and how to categorise them.
"""

# Complete ordered list of achievement IDs as returned by env info["achievements"]
ACHIEVEMENT_NAMES: list[str] = [
    # "achievement_id_1",
    # "achievement_id_2",
    # ...
]

# Maps achievement ID -> display label shown in RSA plots and MLflow artifacts
ACHIEVEMENT_LABEL_MAP: dict[str, str] = {
    # "achievement_id": "Display Label",
}

# Functional groups for RSA alignment. Keys become MLflow metric names:
# rsa_alignment_{group_name}. Labels are values from ACHIEVEMENT_LABEL_MAP.
ACHIEVEMENT_GROUPS: dict[str, frozenset] = {
    # "group_name": frozenset({"Label A", "Label B"}),
}

# Materials required to craft each achievement - used by compute_eps() only.
# Omit if the environment has no material-based progress tracking.
ACHIEVEMENT_MATERIALS: dict[str, dict[str, int]] = {
    # "achievement_id": {"material_name": quantity_required},
}
```

---

## Step 2 - Create the entity file

**File:** `entities/{algorithm}_{environment}.py`

```python
"""{Algorithm} + {Environment} entity.

Implements the Entity protocol from core/entity.py. All algorithm-specific
and environment-specific logic lives here.
"""
from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym

from core.data import EpisodeData, GradientResult
from core.gradient_utils import OnlineGradientAggregator
from entities.definitions.{environment} import (
    ACHIEVEMENT_NAMES, ACHIEVEMENT_LABEL_MAP, ACHIEVEMENT_GROUPS,
    ACHIEVEMENT_MATERIALS,    # remove if not used
)

ENTITY_ID  = "{algorithm}_{environment}"  # unique string; used as MLflow experiment name
OBS_SHAPE  = (H, W, C)                   # observation shape (height, width, channels)
N_ACTIONS  = N                           # number of discrete actions
HOOK_LAYER = "..."                       # dotted module path for activation hook,
                                         # relative to the object returned by get_policy(model)
                                         # e.g. "features_extractor.linear" for SB3 CNN policy
EPS_WEIGHT = 0.9                         # coefficient for materials_fraction in EPS


class {Algorithm}{Environment}:
    """Entity for {Algorithm} trained on {Environment}."""

    entity_id             = ENTITY_ID
    obs_shape             = OBS_SHAPE
    n_actions             = N_ACTIONS
    hook_layer            = HOOK_LAYER
    achievement_names     = ACHIEVEMENT_NAMES
    achievement_label_map = ACHIEVEMENT_LABEL_MAP
    achievement_groups    = ACHIEVEMENT_GROUPS

    def make_env(self, seed: int) -> gym.Env:
        """Return a fully-wrapped gymnasium environment.

        Requirements for the returned env:
            observation_space: Box(low=0, high=255, shape=(H,W,C), dtype=np.uint8)
            action_space:      Discrete(N_ACTIONS)
            info (each step):
                "achievements": dict[str, bool] - achievement status
                "inventory":    dict[str, int]  - material counts (if EPS uses materials)
        """
        # return YourWrapper(YourBaseEnv(seed=seed))
        raise NotImplementedError

    def load_checkpoint(self, path: str, device: str) -> object:
        """Load and return the model from a checkpoint file.

        The returned object is passed to select_action, preprocess_obs, get_policy,
        compute_gradients, and used as the target for activation hooks.

        Returns: model object (type depends on algorithm framework)
        """
        raise NotImplementedError

    def select_action(self, model: object, obs: np.ndarray,
                      deterministic: bool = True) -> int:
        """Select an action for one environment step.

        Args:
            model: model object from load_checkpoint()
            obs:   (H, W, C) uint8 numpy array - single observation
        Returns:
            int - action index in range [0, n_actions)
        """
        raise NotImplementedError

    def compute_eps(self, info: dict) -> float:
        """Compute Episode Performance Score for one completed episode.

        Used for success/failure partitioning only. NOT the same as
        achievement frame logging (handled separately by AchievementTracker).

        Args:
            info: raw env info dict from the final step of the episode.
                  Extract whatever your environment provides, e.g.:
                    achievements = info.get("achievements", {})
                    inventory    = info.get("inventory", {})
        Returns:
            float - EPS score; higher = better episode
        """
        achievements   = info.get("achievements", {})
        inventory      = info.get("inventory", {})
        n_achieved     = sum(1 for v in achievements.values() if v)
        materials_frac = _materials_fraction(achievements, inventory)
        return n_achieved + EPS_WEIGHT * materials_frac

    def preprocess_obs(self, model: object, obs_np: np.ndarray,
                       device: str) -> torch.Tensor:
        """Convert observations from canonical uint8 format to model-ready tensor.

        Args:
            model:  model object from load_checkpoint()
            obs_np: (N, H, W, C) uint8 numpy array - may be a batch
            device: torch device string
        Returns:
            torch.Tensor on device in the format required by the model
            (e.g. (N, C, H, W) float32 normalised to [0, 1])
        """
        raise NotImplementedError

    def get_policy(self, model: object) -> object:
        """Return the hookable sub-module for activation extraction.

        The returned object is used as:
          - the first argument to activation_utils.extract_activations()
          - its named_modules() is searched for hook_layer
          - it is called directly in a forward pass to trigger the hook

        For SB3 algorithms: return model.policy
        For custom networks: return model
        """
        raise NotImplementedError

    def compute_gradients(self, model: object, episodes: list[EpisodeData],
                          device: str, batch_size: int = 10) -> GradientResult:
        """Compute gradient analysis for a list of episodes.

        Args:
            model:      model object from load_checkpoint()
            episodes:   list of EpisodeData; observations are (T, H, W, C) uint8 numpy
            device:     torch device string
            batch_size: episodes per gradient accumulation batch
        Returns:
            GradientResult with:
                raw_mean:    dict[layer_name -> mean gradient tensor]
                per_episode: list of L2-normalised gradient dicts (one per batch)
                variants:    None for single-variant algorithms (e.g. PPO);
                             named dict for multi-variant (e.g. Rainbow IS/uniform)
                metadata:    {"n_transitions": int, ...}
        """
        policy      = self.get_policy(model)
        overall_agg = OnlineGradientAggregator(list(policy.named_parameters()))
        per_episode_grads = []

        for i in range(0, len(episodes), batch_size):
            batch     = episodes[i: i + batch_size]
            batch_agg = OnlineGradientAggregator(list(policy.named_parameters()))

            for episode in batch:
                policy.zero_grad()
                obs_tensor = self.preprocess_obs(model, episode.observations, device)
                # --- algorithm-specific loss ---
                # Example (PPO):
                # actions_tensor = torch.tensor(episode.actions, dtype=torch.long).to(device)
                # values, log_probs, _ = policy.evaluate_actions(obs_tensor, actions_tensor)
                # advantages = _compute_gae(episode.rewards, values.detach().cpu(), episode.dones)
                # loss = -(advantages * log_probs).mean()
                # loss.backward()
                raise NotImplementedError

                batch_agg.accumulate(list(policy.named_parameters()))
                overall_agg.accumulate(list(policy.named_parameters()))
                policy.zero_grad()

            per_episode_grads.append(batch_agg.l2_normalized())

        return GradientResult(
            raw_mean    = overall_agg.mean_gradient(),
            per_episode = per_episode_grads,
            variants    = None,
            metadata    = {"n_transitions": sum(len(ep.rewards) for ep in episodes)},
        )


def _materials_fraction(achievements: dict[str, bool], inventory: dict) -> float:
    """Fraction of materials held toward the nearest unachieved craftable achievement."""
    best = 0.0
    for ach, required in ACHIEVEMENT_MATERIALS.items():
        if achievements.get(ach, False):
            continue
        total = sum(required.values())
        if total == 0:
            continue
        held  = sum(min(int(inventory.get(mat, 0)), qty) for mat, qty in required.items())
        best  = max(best, held / total)
    return best
```

---

## Step 3 - Verify the entity satisfies the protocol

```python
python -c "
from entities.{algorithm}_{environment} import {Algorithm}{Environment}
from core.entity import Entity
assert isinstance({Algorithm}{Environment}(), Entity), 'Entity protocol not satisfied'
print('OK')
"
```

If this fails, the entity is missing one or more methods or attributes defined in `core/entity.py`.

---

## Step 4 - Run the pipeline

```python
from training.run_config import RunConfig
from training.trainer import run as train
from entities.{algorithm}_{environment} import {Algorithm}{Environment}

config = RunConfig(
    entity_id  = "{algorithm}_{environment}",
    n_steps    = 1_000_000,
    eval_every = 50_000,
    n_episodes = 500,
)
train({Algorithm}{Environment}(), config)
```

Or run via CLI (if a `__main__` entry point is added to `training/trainer.py`):

```bash
python -m training.trainer --entity {algorithm}_{environment} --n_steps 1_000_000
```

---

## Entity protocol reference

Defined in `core/entity.py`. All fields and methods must be satisfied structurally (no inheritance required - `typing.Protocol`).

| Attribute / method | Type | Description |
|--------------------|------|-------------|
| `entity_id` | `str` | Unique ID used as MLflow experiment name |
| `obs_shape` | `tuple` | `(H, W, C)` - canonical observation shape |
| `n_actions` | `int` | Number of discrete actions |
| `achievement_names` | `list[str]` | Ordered achievement IDs from env info |
| `achievement_label_map` | `dict[str, str]` | ID -> display label for RSA plots |
| `achievement_groups` | `dict[str, frozenset]` | Group name -> set of labels; keys become MLflow metric names |
| `hook_layer` | `str` | Dotted module path relative to `get_policy(model)` |
| `make_env(seed)` | `gym.Env` | Returns a fully-wrapped gymnasium environment |
| `load_checkpoint(path, device)` | `Any` | Loads and returns the model |
| `select_action(model, obs, deterministic)` | `int` | Returns action index for single obs |
| `compute_eps(info)` | `float` | EPS score for one episode; extract env-specific fields from `info` dict internally |
| `preprocess_obs(model, obs_np, device)` | `torch.Tensor` | `(N, H, W, C)` uint8 → model-ready tensor |
| `get_policy(model)` | `Any` | Returns the hookable sub-module (e.g. `model.policy`) |
| `compute_gradients(model, episodes, device, batch_size)` | `GradientResult` | Gradient analysis for one episode group |
| `train(n_steps, checkpoint_every, on_checkpoint)` | `None` | Train the agent for `n_steps` steps; call `on_checkpoint(step, ckpt_path)` every `checkpoint_every` steps |

---

## Notes

- The pipeline never imports from `entities/` directly - it only calls Entity protocol methods.
- `compute_eps` and `AchievementTracker` are completely separate. EPS counts achievements for partitioning; the tracker logs first-unlock frames for RSA stimuli.
- `hook_layer` must be a valid key in `dict(get_policy(model).named_modules())`.
- `preprocess_obs` must return a tensor that `get_policy(model)` can accept as input.
