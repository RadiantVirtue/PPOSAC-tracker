"""Scalar-loss DQN ablation - non-distributional Rainbow.

Architecture identical to Rainbow: same CNN backbone, dueling NoisyLinear heads,
PER, n-step returns, target network.  The only removal is C51: output is scalar
Q-values with per-transition Huber (SmoothL1) loss.

Purpose: gate the headline claim.  If the MORA ratio
  gradient_magnitude_positive / gradient_magnitude_neutral  ≈ 33×
survives here, credit-assignment is the driver.  If it collapses, the finding
is a distributional-loss artefact and must be reframed.

Key public API:
  train_scalar_dqn(args, seed, out_root) → checkpoint_path (str)
  compute_scalar_mora_ratio(ckpt_path, n_episodes, device) → dict
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange

from rainbow.gradients import _compute_nstep_returns
from shared.gradient_utils import OnlineGradientAggregator
from shared.metrics import gradient_magnitude



class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        ei = self._scale_noise(self.in_features)
        eo = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eo.ger(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)



class ScalarDQN(nn.Module):
    """Dueling scalar Q-network (no distributional head).

    Outputs (batch, action_space) Q-values via dueling combination.
    Architecture otherwise identical to rainbow/model.py DQN.
    """

    def __init__(
        self,
        history_length: int,
        action_space: int,
        hidden_size: int = 512,
        architecture: str = "canonical",
        noisy_std: float = 0.1,
    ):
        super().__init__()
        self.action_space = action_space

        if architecture == "canonical":
            self.convs = nn.Sequential(
                nn.Conv2d(history_length, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            )
            self.conv_output_size = 1024   # 64×64 input
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(history_length, 32, 5, stride=5), nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=5), nn.ReLU(),
            )
            self.conv_output_size = 256

        self.fc_h_v = NoisyLinear(self.conv_output_size, hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(hidden_size, 1,            std_init=noisy_std)
        self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=noisy_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, action_space) Q-values."""
        h = self.convs(x).reshape(x.size(0), -1)
        v = self.fc_z_v(F.relu(self.fc_h_v(h)))          # (B, 1)
        a = self.fc_z_a(F.relu(self.fc_h_a(h)))          # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)        # dueling

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()



class ScalarAgent:
    """Wraps ScalarDQN for training: Huber loss, PER, n-step, target network."""

    def __init__(self, args, action_space: int):
        self.action_space = action_space
        self.n            = args.multi_step
        self.discount     = args.discount
        self.batch_size   = args.batch_size
        self.norm_clip    = getattr(args, "norm_clip", 10.0)
        self.device       = args.device

        net_kw = dict(
            history_length=args.history_length,
            action_space=action_space,
            hidden_size=args.hidden_size,
            architecture=args.architecture,
            noisy_std=args.noisy_std,
        )
        self.online_net = ScalarDQN(**net_kw).to(self.device)
        self.target_net = ScalarDQN(**net_kw).to(self.device)
        self._sync_target()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(),
            lr=args.learning_rate,
            eps=getattr(args, "adam_eps", 1.5e-4),
        )

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def reset_noise(self):
        self.online_net.reset_noise()

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            return self.online_net(state.unsqueeze(0)).argmax(1).item()

    def learn(self, mem) -> None:
        idxs, states, actions, returns, next_states, nonterminals, weights, _ = mem.sample(
            self.batch_size
        )

        q_a = self.online_net(states)[range(self.batch_size), actions]   # (B,)

        with torch.no_grad():
            best_a   = self.online_net(next_states).argmax(1)            # double-Q
            q_next   = self.target_net(next_states)[range(self.batch_size), best_a]
            targets  = returns + nonterminals.squeeze(1) * (self.discount ** self.n) * q_next

        loss = F.smooth_l1_loss(q_a, targets, reduction="none")          # (B,)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def update_target_net(self):
        self._sync_target()

    def save(self, ckpt_dir: str, name: str, global_step: int = 0,
             episode_count: int = 0, args=None) -> str:
        os.makedirs(ckpt_dir, exist_ok=True)
        args_dict: dict = {}
        if args is not None:
            args_dict = {
                "history_length":    args.history_length,
                "hidden_size":       args.hidden_size,
                "architecture":      args.architecture,
                "noisy_std":         args.noisy_std,
                "multi_step":        args.multi_step,
                "discount":          args.discount,
                "priority_exponent": args.priority_exponent,
                "priority_weight":   args.priority_weight,
                "T_max":             args.T_max,
                "learn_start":       getattr(args, "learn_start", int(20e3)),
                "n_actions":         self.action_space,
                "algorithm":         "scalar_dqn",
            }
        out = os.path.join(ckpt_dir, name)
        torch.save({
            "global_step":           global_step,
            "episode_count":         episode_count,
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict":  self.optimiser.state_dict(),
            "args_dict":             args_dict,
        }, out)
        return out



def load_scalar_nets(checkpoint_path: str, device: str = "cpu"):
    """Load frozen online + target ScalarDQN from a checkpoint.

    Returns (online_net, target_net, episode_count, global_step, args_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    d    = ckpt.get("args_dict", {})
    n_actions      = d.get("n_actions", 17)
    history_length = d.get("history_length", 3)
    hidden_size    = d.get("hidden_size", 512)
    architecture   = d.get("architecture", "canonical")
    noisy_std      = d.get("noisy_std", 0.1)

    def _make():
        net = ScalarDQN(history_length, n_actions, hidden_size, architecture, noisy_std)
        return net.to(device).eval()

    online_net = _make()
    online_net.load_state_dict(ckpt["online_net_state_dict"])

    target_net = _make()
    if "target_net_state_dict" in ckpt:
        target_net.load_state_dict(ckpt["target_net_state_dict"])
    else:
        target_net.load_state_dict(ckpt["online_net_state_dict"])
    for p in target_net.parameters():
        p.requires_grad = False

    return (
        online_net,
        target_net,
        ckpt.get("episode_count", 0),
        ckpt.get("global_step", 0),
        d,
    )



def _scalar_per_loss(
    episode,
    online_net: ScalarDQN,
    target_net: ScalarDQN,
    gamma: float,
    n: int,
    device: str,
) -> torch.Tensor:
    """Per-transition Huber Bellman loss for one full episode.

    Call without torch.no_grad() so that online_net.parameters() retain
    computation graph for backward passes.

    Returns:
        per_loss: (T,) float32 - Huber loss per transition (>= 0).
    """
    obs     = episode.observations.to(device)   # (T, 3, H, W)
    actions = episode.actions.to(device)        # (T,) long
    T       = len(obs)

    R, nonterminal = _compute_nstep_returns(episode.rewards, episode.dones, gamma, n)
    R           = R.to(device)
    nonterminal = nonterminal.to(device)

    next_obs = torch.zeros_like(obs)
    if n < T:
        next_obs[:-n] = obs[n:]   # next_obs[t] = obs[t+n]; last n rows get zero (no bootstrap)

    # Q(s, a_t) - grad-tracked
    q_a = online_net(obs)[range(T), actions]   # (T,)

    with torch.no_grad():
        # Double-Q: action selection via online, evaluation via target
        best_a   = online_net(next_obs).argmax(1)
        q_next   = target_net(next_obs)[range(T), best_a]
        targets  = R + nonterminal * (gamma ** n) * q_next

    return F.smooth_l1_loss(q_a, targets.detach(), reduction="none")   # (T,)



def compute_scalar_mora_ratio(
    checkpoint_path: str,
    n_episodes: int = 500,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Compute MORA ratio for a trained ScalarDQN checkpoint.

    Runs n_episodes with the frozen scalar policy, then computes per-reward-sign
    indicator-weighted gradients (identical method to moment_of_reward.py for
    the distributional Rainbow) and returns the ratio:

        mora_ratio = gradient_magnitude_positive / gradient_magnitude_neutral

    If this ratio ≈ 33× (same as Rainbow) the finding is about credit assignment.
    If it collapses, C51 is driving it.

    Returns dict with:
        n_pos, n_neu, n_neg, gradient_magnitude_positive/neutral/negative,
        mora_ratio, n_episodes_collected, checkpoint_path.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from rainbow.sampling import evaluate_frozen_policy as _eval_rainbow

    online_net, target_net, episode_count, global_step, args_dict = load_scalar_nets(
        checkpoint_path, device
    )
    gamma = args_dict.get("discount", 0.99)
    n     = args_dict.get("multi_step", 3)

    # Collect episodes using the scalar policy's Q-values for action selection
    episodes = _collect_episodes_scalar(online_net, n_episodes, device, seed)

    print(f"  Scalar MoR: {len(episodes)} episodes collected")

    named_params = list(online_net.named_parameters())

    # Per-transition normalization: pre-compute total transition counts per sign
    # across all episodes so each transition contributes equally (not each episode).
    n_pos_total = int(sum((ep.rewards > 0).sum().item() for ep in episodes))
    n_neu_total = int(sum((ep.rewards == 0).sum().item() for ep in episodes))
    n_neg_total = int(sum((ep.rewards < 0).sum().item() for ep in episodes))
    n_total_by_key = {"pos": n_pos_total, "neu": n_neu_total, "neg": n_neg_total}

    overall_agg = {
        k: OnlineGradientAggregator(named_params)
        for k in ("pos", "neu", "neg")
    }

    online_net.eval()   # NoisyLinear uses weight_mu only (deterministic)

    for ep in tqdm(episodes, desc="Scalar MoR [indicator]", leave=False):
        if len(ep.rewards) < 2:
            continue

        per_loss = _scalar_per_loss(ep, online_net, target_net, gamma, n, device)

        r = ep.rewards.to(device) if hasattr(ep.rewards, "to") else torch.tensor(
            ep.rewards, dtype=torch.float32, device=device
        )
        masks = {
            "pos": (r > 0).float(),
            "neu": (r == 0).float(),
            "neg": (r < 0).float(),
        }

        # If masks[k].sum() > 0, then n_total_by_key[k] > 0 is guaranteed.
        active = [
            (k, masks[k])
            for k in ("pos", "neu", "neg")
            if masks[k].sum().item() > 0
        ]

        for i, (key, mask) in enumerate(active):
            is_last = (i == len(active) - 1)
            online_net.zero_grad()
            # Per-transition normalization: divide by TOTAL transitions of this sign
            # across ALL episodes, so each transition contributes equally to G_s.
            (mask / n_total_by_key[key] * per_loss).sum().backward(retain_graph=not is_last)
            overall_agg[key].accumulate(named_params)

        online_net.zero_grad()

    # Return raw accumulated sum (not mean_gradient()). The /n_total_by_key[key]
    # normalization in each backward call already encodes the per-transition mean.
    raw = {
        k: ({name: overall_agg[k].sum_grads[name].clone() for name in overall_agg[k].sum_grads}
            if overall_agg[k].count > 0 else None)
        for k in ("pos", "neu", "neg")
    }

    gm_pos = gradient_magnitude(raw["pos"])
    gm_neu = gradient_magnitude(raw["neu"])
    gm_neg = gradient_magnitude(raw["neg"])

    mora_ratio = None
    if gm_pos is not None and gm_neu is not None and gm_neu > 0:
        mora_ratio = gm_pos / gm_neu

    return {
        "checkpoint_path": checkpoint_path,
        "global_step":     global_step,
        "episode_count":   episode_count,
        "n_episodes_collected": len(episodes),
        "n_pos": n_pos_total,
        "n_neu": n_neu_total,
        "n_neg": n_neg_total,
        "gradient_magnitude_positive": gm_pos,
        "gradient_magnitude_neutral":  gm_neu,
        "gradient_magnitude_negative": gm_neg,
        "mora_ratio": mora_ratio,
    }


def _collect_episodes_scalar(
    online_net: ScalarDQN,
    n_episodes: int,
    device: str,
    seed: int = 0,
    num_envs: int = 16,
):
    """Collect episodes using a frozen ScalarDQN policy."""
    import gymnasium as gym
    from rainbow.sampling import EpisodeData
    from wrappers import make_crafter_env

    online_net.eval()
    base_seed = seed

    vec_env = gym.vector.AsyncVectorEnv([
        (lambda i: lambda: make_crafter_env(seed=base_seed + i * 10_000))(i)
        for i in range(num_envs)
    ])

    env_obs     = [[] for _ in range(num_envs)]
    env_actions = [[] for _ in range(num_envs)]
    env_rewards = [[] for _ in range(num_envs)]
    env_dones   = [[] for _ in range(num_envs)]

    episodes  = []
    completed = 0

    obs, _ = vec_env.reset()
    np.random.seed(base_seed)
    pbar = tqdm(total=n_episodes, desc="Collecting episodes (scalar)", unit="ep")

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).float().div_(255.0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            actions = online_net(obs_t).argmax(1).cpu().numpy()
        for i in range(num_envs):
            if np.random.random() < 0.001:
                actions[i] = np.random.randint(0, online_net.action_space)

        for i in range(num_envs):
            env_obs[i].append(obs[i].copy())
            env_actions[i].append(int(actions[i]))

        obs, rewards, terminateds, truncateds, _ = vec_env.step(actions)

        for i in range(num_envs):
            done = bool(terminateds[i]) or bool(truncateds[i])
            env_rewards[i].append(float(rewards[i]))
            env_dones[i].append(float(done))

            if done and completed < n_episodes:
                obs_arr = np.array(env_obs[i], dtype=np.float32) / 255.0
                obs_tensor = torch.from_numpy(obs_arr).permute(0, 3, 1, 2)
                ep = EpisodeData(
                    observations=obs_tensor,
                    actions=torch.tensor(env_actions[i], dtype=torch.long),
                    rewards=torch.tensor(env_rewards[i], dtype=torch.float32),
                    dones=torch.tensor(env_dones[i],   dtype=torch.float32),
                )
                episodes.append(ep)
                completed += 1
                pbar.update(1)
                env_obs[i]     = []
                env_actions[i] = []
                env_rewards[i] = []
                env_dones[i]   = []

    pbar.close()
    vec_env.close()
    return episodes



def train_scalar_dqn(args, seed: int, out_root: str) -> str:
    """Train a scalar DQN (non-distributional Rainbow ablation) on Crafter.

    Args:
        args:     argparse.Namespace with Rainbow hyperparameters (atoms/V_min/V_max
                  are unused - scalar DQN ignores distributional parameters).
        seed:     random seed.
        out_root: experiment root directory for checkpoints and logs.

    Returns:
        Path to final checkpoint (checkpoint_live.pt).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from rainbow.memory import ReplayMemory
    from rainbow.train import CrafterEnvWrapper, _log_return

    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1, 10000))

    if torch.cuda.is_available() and not getattr(args, "disable_cuda", True):
        device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
    else:
        device = torch.device("cpu")

    args.device         = device
    args.seed           = seed
    args.experiment_root = out_root
    # Scalar DQN does not use distributional params but ReplayMemory needs
    # history_length - ensure it is set to 3 (RGB channels).
    if not hasattr(args, "history_length"):
        args.history_length = 3

    ckpt_dir = os.path.join(out_root, "checkpoints", "scalar_dqn")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir  = os.path.join(out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    env          = CrafterEnvWrapper(args, seed=seed)
    action_space = env.action_space()
    agent        = ScalarAgent(args, action_space)
    agent.train()

    mem = ReplayMemory(args, args.memory_capacity)
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    done           = True
    episode_count  = 0
    ep_return      = 0.0

    pbar = trange(1, args.T_max + 1, desc=f"ScalarDQN seed={seed}")
    for T in pbar:
        if done:
            state    = env.reset()
            ep_return = 0.0

        if T % args.replay_frequency == 0:
            agent.reset_noise()

        action          = agent.act(state)
        next_state, reward, done = env.step(action)
        ep_return      += reward

        clipped = max(min(reward, args.reward_clip), -args.reward_clip) if args.reward_clip > 0 else reward
        mem.append(state, action, clipped, done)

        if done:
            episode_count += 1
            _log_return(out_root, ep_return, step=T)

        buf_size = mem.capacity if mem.transitions.full else mem.transitions.index

        if T >= args.learn_start and buf_size >= max(args.learn_start, args.batch_size * 10):
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                agent.learn(mem)

            if T % args.target_update == 0:
                agent.update_target_net()

        state = next_state

    env.close()

    # Save final checkpoint
    live_path = os.path.join(ckpt_dir, "checkpoint_live.pt")
    agent.save(ckpt_dir, "checkpoint_live.pt",
               global_step=args.T_max, episode_count=episode_count, args=args)
    print(f"  Scalar DQN training complete. Checkpoint: {live_path}")
    return live_path
