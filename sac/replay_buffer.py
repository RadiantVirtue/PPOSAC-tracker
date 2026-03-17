"""Simple uniform replay buffer for Discrete SAC.

Stores transitions as stacked tensors. Returns TensorDict batches
compatible with TorchRL's DiscreteSACLoss in_keys:
  'observation', 'action', ('next','reward'), ('next','observation'),
  ('next','terminated'), ('next','done')
"""
import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Circular buffer storing (obs, action, reward, next_obs, terminated, done)."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self._ptr = 0
        self.size = 0

        # Pre-allocate storage (lazily filled on first add)
        self._obs = None           # (capacity, C, H, W) float32
        self._next_obs = None
        self._actions = None       # (capacity,) long
        self._rewards = None       # (capacity,) float32
        self._terminated = None    # (capacity,) bool
        self._done = None          # (capacity,) bool

    def add(
        self,
        obs: torch.Tensor,       # (C, H, W) float32
        action: torch.Tensor,    # scalar long
        reward: torch.Tensor,    # scalar float32
        next_obs: torch.Tensor,  # (C, H, W) float32
        terminated: torch.Tensor,
        done: torch.Tensor,
    ):
        if self._obs is None:
            C, H, W = obs.shape
            self._obs = torch.zeros(self.capacity, C, H, W, dtype=torch.float32)
            self._next_obs = torch.zeros_like(self._obs)
            self._actions = torch.zeros(self.capacity, dtype=torch.long)
            self._rewards = torch.zeros(self.capacity, dtype=torch.float32)
            self._terminated = torch.zeros(self.capacity, dtype=torch.bool)
            self._done = torch.zeros(self.capacity, dtype=torch.bool)

        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._terminated[i] = terminated
        self._done[i] = done

        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TensorDict:
        idx = torch.randint(0, self.size, (batch_size,))
        return TensorDict(
            {
                "observation": self._obs[idx],
                "action": self._actions[idx],
                "next": TensorDict(
                    {
                        "observation": self._next_obs[idx],
                        "reward": self._rewards[idx].unsqueeze(-1),
                        "terminated": self._terminated[idx].unsqueeze(-1),
                        "done": self._done[idx].unsqueeze(-1),
                    },
                    batch_size=[batch_size],
                ),
            },
            batch_size=[batch_size],
        )
