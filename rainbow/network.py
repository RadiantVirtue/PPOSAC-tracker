import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Original Rainbow networks (flat state vector input) ─────────────────────

class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.V = NoisyLinear(args.hidden_dim, 1)
            self.A = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        Q = self.fc3(s)
        return Q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


# ── CNN encoder (obs_shape-aware) ────────────────────────────────────────────

class _CNNEncoder(nn.Module):
    """CNN encoder for image observations.

    Selects architecture based on obs_shape:
      (7, 7, C)  → MiniGrid CNN: three Conv2d layers → 64-dim flat
      otherwise  → Nature CNN: three strided Conv2d + Linear projection → 64-dim flat

    Input to forward: (B, state_dim) flat float32 tensor.
    The flat tensor is reshaped to (B, H, W, C) then permuted to (B, C, H, W).
    """

    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape  # (H, W, C)
        H, W, C = obs_shape

        if H == 7 and W == 7:
            # MiniGrid CNN — identical to PPO/SAC encoder; outputs 64-dim directly
            self.cnn = nn.Sequential(
                nn.Conv2d(C, 16, (2, 2)), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)), nn.ReLU(),
            )
            self.proj = nn.Identity()
            self._flat_dim = 64
        else:
            # Nature CNN for larger images (e.g. 64×64 Crafter)
            self.cnn = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            )
            # Compute flattened dim by dry-running a dummy tensor
            with torch.no_grad():
                dummy = torch.zeros(1, C, H, W)
                flat = self.cnn(dummy).flatten(1).shape[1]
            self.proj = nn.Linear(flat, 64)
            self._flat_dim = 64

    def forward(self, s):
        """s: (B, state_dim) flat → (B, 64) embedding."""
        B = s.shape[0]
        H, W, C = self.obs_shape
        x = s.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.cnn(x).flatten(1)
        return self.proj(x)


# ── CNN-based Rainbow networks (image obs input, flat-stored) ────────────────

class Dueling_CNN_Net(nn.Module):
    """Dueling Rainbow network for image observations.

    Takes (B, state_dim) flat tensors (images stored flattened in replay buffer).
    Uses _CNNEncoder → 64-dim → fc1 (hook target) → V/A heads.
    layer_name='fc1' hooks the 64-dim pre-head representation for analysis.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder = _CNNEncoder(args.obs_shape)
        self.fc1 = nn.Linear(64, 64)
        if args.use_noisy:
            self.V = NoisyLinear(64, 1)
            self.A = NoisyLinear(64, args.action_dim)
        else:
            self.V = nn.Linear(64, 1)
            self.A = nn.Linear(64, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(self.encoder(s)))
        V = self.V(s)
        A = self.A(s)
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))
        return Q


class CNN_Net(nn.Module):
    """Standard Rainbow network for image observations.

    Takes (B, state_dim) flat tensors.
    Uses _CNNEncoder → 64-dim → fc1 (hook target) → fc2 output head.
    layer_name='fc1' hooks the 64-dim pre-head representation for analysis.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder = _CNNEncoder(args.obs_shape)
        self.fc1 = nn.Linear(64, 64)
        if args.use_noisy:
            self.fc2 = NoisyLinear(64, args.action_dim)
        else:
            self.fc2 = nn.Linear(64, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(self.encoder(s)))
        return self.fc2(s)
