import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# ACModelWrapper — wraps rl-starter-files ACModel to expose actor/critic/image_conv
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


# SAC — 256-unit hidden layers with Kaiming-normal init, matching sac/train.py

def _sac_layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SACQNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        import numpy as np
        obs_dim = int(np.array(obs_shape).prod())
        self.fc1 = _sac_layer_init(nn.Linear(obs_dim, 256))
        self.fc2 = _sac_layer_init(nn.Linear(256, 256))
        self.fc_q = _sac_layer_init(nn.Linear(256, n_actions))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_q(x)


class SACActor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        import numpy as np
        obs_dim = int(np.array(obs_shape).prod())
        self.fc1 = _sac_layer_init(nn.Linear(obs_dim, 256))
        self.fc2 = _sac_layer_init(nn.Linear(256, 256))
        self.fc_logits = _sac_layer_init(nn.Linear(256, n_actions))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_logits(x)

    def get_action(self, x, deterministic=False):
        logits = self(x)
        if deterministic:
            return torch.argmax(logits, dim=1)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
