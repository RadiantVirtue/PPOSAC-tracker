import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# ACModelWrapper — wraps ACModel to expose actor/critic/image_conv
# as direct named submodules (so named_modules()["actor.0"] works for activation hooks)
# and a forward(images) method that accepts (B, H, W, C) image tensors.
class ACModelWrapper(nn.Module):
    def __init__(self, acmodel):
        super().__init__()
        # Register as direct submodules so named_modules() yields "actor.0", "actor.2", etc.
        self.image_conv = acmodel.image_conv
        self.actor = acmodel.actor
        self.critic = acmodel.critic
        self._memory_size = acmodel.memory_size

    @property
    def memory_size(self):
        return self._memory_size

    def _embed(self, images):
        """CNN forward pass. images: (B, H, W, C) float32 tensor → embedding."""
        x = images.transpose(1, 3).transpose(2, 3)   # (B, C, H, W)
        x = self.image_conv(x)
        return x.reshape(x.shape[0], -1)

    def forward(self, images):
        """Accept (B, H, W, C) image tensor → actor logits (for extract_activations)."""
        return self.actor(self._embed(images))

    def get_action(self, obs_t, deterministic=False):
        """obs_t: (1, H, W, C) float32 tensor → sampled or greedy action."""
        logits = self.forward(obs_t)
        dist = Categorical(logits=logits)
        return dist.probs.argmax(dim=-1) if deterministic else dist.sample()


# SAC with CNN — matches PPO's ACModel encoder for direct cross-algorithm comparison.
# All three networks use the same Conv2d architecture (separate weights per network).
# 64-unit Tanh heads match PPO's actor/critic heads, making activation geometry
# directly comparable. layer_name="actor.0" hooks the same semantic layer as PPO.

def _cnn_init(m):
    """Weight-normalised init for Linear layers; default PyTorch init for Conv2d."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class SACCNNEncoder(nn.Module):
    """Shared CNN architecture (instantiated independently per SAC network).

    Identical to ACModel.image_conv. For a 7×7 MiniGrid observation the output
    embedding is 64-dimensional (1×1×64 after the three conv+pool stages).
    """

    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

    def forward(self, x):
        """x: (B, H, W, C) float32 → (B, 64) embedding."""
        x = x.transpose(1, 3).transpose(2, 3)   # (B, C, H, W)
        x = self.image_conv(x)
        return x.reshape(x.shape[0], -1)


class SACCNNActor(nn.Module):
    """SAC actor with CNN encoder and 64-unit Tanh head.

    The actor head is registered as self.actor so that activation hooks use
    layer_name="actor.0" — the same convention as PPO's ACModelWrapper.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.encoder = SACCNNEncoder()
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.apply(_cnn_init)

    def forward(self, x):
        """x: (B, H, W, C) float32 → (B, n_actions) logits."""
        return self.actor(self.encoder(x))

    def get_action(self, x, deterministic=False):
        """Returns (action, log_prob_all_actions, action_probs)."""
        logits = self(x)
        if deterministic:
            return torch.argmax(logits, dim=1)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class SACCNNQNetwork(nn.Module):
    """SAC Q-network with CNN encoder and 64-unit Tanh head.

    Outputs Q(s, a) for ALL actions simultaneously: forward → (B, n_actions).
    Use .gather(1, actions) to extract the Q-value for the taken action.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.encoder = SACCNNEncoder()
        self.q_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.apply(_cnn_init)

    def forward(self, x):
        """x: (B, H, W, C) float32 → (B, n_actions) Q-values."""
        return self.q_head(self.encoder(x))
