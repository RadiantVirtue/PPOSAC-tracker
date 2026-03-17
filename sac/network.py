"""Neural network architectures for Discrete SAC on Crafter.

CrafterCNN      — shared CNN encoder: Conv3 → Linear(flat→64) → ReLU → 64-dim.
                  Architecture mirrors SB3's NatureCNN (features_dim=64) for
                  direct activation-layer comparability with PPO.
DiscreteActor   — CrafterCNN + Linear(64→n_actions); outputs action logits.
                  Hook target: "encoder.linear" (64-dim embedding).
DiscreteCritic  — CrafterCNN + Linear(64→n_actions); outputs per-action Q-values.
                  Two instances (Q1, Q2) are used in training.

Input convention: float32 tensors of shape (B, 3, 64, 64), values in [0, 1].
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrafterCNN(nn.Module):
    """NatureCNN-style encoder for 64×64×3 Crafter observations.

    Architecture (identical to SB3 NatureCNN with features_dim=64):
      Conv2d(3,  32, 8, stride=4) → ReLU
      Conv2d(32, 64, 4, stride=2) → ReLU
      Conv2d(64, 64, 3, stride=1) → ReLU
      Flatten → Linear(flat→64) → ReLU

    The named submodule "linear" is a nn.Sequential([Linear, ReLU]) so that
    extract_activations(actor, obs, layer_name="encoder.linear") returns
    the 64-dim post-ReLU embedding — the analysis hook target.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flat dim with a dry-run (64×64 input)
        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, 3, 64, 64)).shape[1]

        # Named as "linear" so the hook path is "encoder.linear"
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) float32 in [0,1] → (B, 64) embedding."""
        return self.linear(self.cnn(x))


class DiscreteActor(nn.Module):
    """Policy network for Discrete SAC.

    Returns logits (unnormalised log-probabilities) over n_actions.
    Use .get_distribution(obs) to obtain the Categorical distribution.

    Analysis hook target: "encoder.linear" (64-dim embedding, same as
    PPO's "features_extractor.linear").
    """

    def __init__(self, n_actions: int = 17):
        super().__init__()
        self.encoder = CrafterCNN()
        self.head = nn.Linear(64, n_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller init for the policy head
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, 64, 64) → logits (B, n_actions)."""
        return self.head(self.encoder(obs))

    def get_distribution(self, obs: torch.Tensor):
        """Returns Categorical distribution over actions."""
        return torch.distributions.Categorical(logits=self.forward(obs))

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(a|s) for given observations and actions."""
        return self.get_distribution(obs).log_prob(actions)


class DiscreteCritic(nn.Module):
    """Q-value network for Discrete SAC.

    Returns Q(s, a) for all actions simultaneously → (B, n_actions).
    Instantiate two copies (Q1, Q2) for the double-Q trick.
    """

    def __init__(self, n_actions: int = 17):
        super().__init__()
        self.encoder = CrafterCNN()
        self.head = nn.Linear(64, n_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, 64, 64) → Q-values (B, n_actions)."""
        return self.head(self.encoder(obs))


# ── TorchRL wrappers ──────────────────────────────────────────────────────────

def make_torchrl_actor(n_actions: int = 17):
    """Wrap DiscreteActor as a TorchRL ProbabilisticActor.

    in_keys:  ['observation']
    out_keys: ['action', 'log_prob']
    Internally: observation → logits → Categorical sample + log_prob.
    """
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor

    actor_net = DiscreteActor(n_actions)
    actor_mod = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )
    return ProbabilisticActor(
        actor_mod,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    ), actor_net   # return raw net for analysis hooks


def make_torchrl_critic(n_actions: int = 17):
    """Wrap DiscreteCritic as a TorchRL TensorDictModule.

    in_keys:  ['observation']
    out_keys: ['action_value']
    """
    from tensordict.nn import TensorDictModule

    critic_net = DiscreteCritic(n_actions)
    return TensorDictModule(
        critic_net, in_keys=["observation"], out_keys=["action_value"]
    ), critic_net
