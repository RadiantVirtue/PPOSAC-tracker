import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# Parameter initialisation from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module):
    """CNN actor-critic model for MiniGrid (no memory, no text).

    Compatible with torch_ac.PPOAlgo (non-recurrent: forward returns (dist, value)).
    Compatible with ACModelWrapper in shared/networks.py via .image_conv, .actor,
    .critic, and .memory_size.
    """

    recurrent = False  # required by torch_ac.PPOAlgo base class assertion

    def __init__(self, obs_space, action_space):
        super().__init__()

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(init_params)

    # Kept for ACModelWrapper compatibility (reads acmodel.memory_size on init).
    @property
    def memory_size(self):
        return 2 * self.image_embedding_size

    def forward(self, obs):
        """obs: torch_ac.DictList with .image of shape (B, H, W, C).
        Returns (dist, value) — non-recurrent interface for torch_ac.PPOAlgo.
        """
        x = obs.image.transpose(1, 3).transpose(2, 3)  # (B, H, W, C) → (B, C, H, W)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
