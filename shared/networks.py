import torch.nn as nn
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
