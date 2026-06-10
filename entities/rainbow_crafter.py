"""Rainbow DQN + Crafter entity.

Implements the Entity protocol for Rainbow DQN trained on the Crafter environment.

This file is structured in two halves:

  Half 1 - Custom Rainbow DQN implementation
  ──────────────────────────────────────────
  The algorithm implementation, unchanged from its standalone form.
  This code is algorithm-specific and would look the same in any project
  using Rainbow DQN on 64×64 RGB input:

    - NoisyLinear, DQN (network)
    - SegmentTree, ReplayMemory (prioritised experience replay)
    - TrainingGradientCapture, _RainbowAgent (training internals)
    - MORAPriorityTracker, OutcomeTracker (MORA-PER training; always active)
    - _CrafterEnvWrapper (env adapter for the training loop)
    - Gradient analysis utilities (_forward_per_loss, IS weights, etc.)
    - _compute_sign_indicator_gradients (MORA post-hoc analysis)

  Half 2 - Pipeline glue
  ──────────────────────
  Adapter code connecting the Rainbow implementation to the entity protocol
  and the data formats expected by training/trainer.py, training/eval_runner.py,
  and analysis/pipeline.py:

    - CrafterGymnasiumWrapper, CrafterAchievementWrapper, make_crafter_env
    - RainbowModel dataclass (model container for all protocol methods)
    - _RbEp dataclass (EpisodeData HWC uint8 → CHW float32 conversion)
    - RainbowCrafter class (all protocol methods + train())
    - Private helpers (_build_args, _build_train_args, _to_rb_ep, etc.)

compute_gradients() returns five gradient variants:
  raw_mean               IS-weighted mean (primary result)
  variants["uniform"]    Uniform-weighted mean
  variants["adam"]       Adam effective update: exp_avg / sqrt(exp_avg_sq + ε)
                         Addresses the Adam decoupling finding (Appendix B).
  variants["training"]   Mean gradient from actual training batches between checkpoints
                         (TrainingGradientCapture). Ground truth against which eval-episode
                         proxy variants can be compared.
  variants["positive"]   Indicator-weighted gradient from r>0 transitions (MORA)
  variants["neutral"]    Indicator-weighted gradient from r=0 transitions (MORA)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import crafter
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange

from core.data import EpisodeData, GradientResult
from core.gradient_utils import OnlineGradientAggregator
from entities.definitions.crafter import (
    ACHIEVEMENT_MATERIALS,
    ACHIEVEMENT_NAMES,
    ACHIEVEMENT_LABEL_MAP,
    ACHIEVEMENT_GROUPS,
)


ENTITY_ID  = "rainbow_crafter"
OBS_SHAPE  = (64, 64, 3)
N_ACTIONS  = 17
HOOK_LAYER = "fc_h_v"   # value-stream hidden NoisyLinear (512-dim, pre-ReLU)
EPS_WEIGHT = 0.9
ADAM_EPS   = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Half 1 - Custom Rainbow DQN implementation
# ─────────────────────────────────────────────────────────────────────────────

# ── A: Network ───────────────────────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer.

    In train() mode: y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
    In eval() mode:  y = μ_w x + μ_b  (ε = 0, deterministic)

    σ parameters carry no gradient during eval() - excluded from gradient
    analysis to avoid zero-gradient artefacts.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        p = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        q = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(q.outer(p))
        self.bias_epsilon.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class DQN(nn.Module):
    """Dueling distributional DQN with NoisyLinear exploration.

    Architecture (canonical, Crafter 64×64 input):
      Conv block: (3, 64, 64) → 32 → 64 → 64 → flatten 1024
      Value stream:     NoisyLinear(1024→512) → ReLU → NoisyLinear(512→atoms)
      Advantage stream: NoisyLinear(1024→512) → ReLU → NoisyLinear(512→n_actions*atoms)
      Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))  then softmax/log_softmax over atoms
    """

    def __init__(self, args, action_space: int):
        super().__init__()
        self.atoms        = args.atoms
        self.action_space = action_space

        in_channels = args.history_length  # 3 for RGB

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        if getattr(args, "architecture", "canonical") == "data-efficient":
            conv_out = 64 * 7 * 7
        else:
            conv_out = 1024  # canonical: 64×64 → 64×4×4

        hidden = args.hidden_size
        std    = args.noisy_std

        self.fc_h_v = NoisyLinear(conv_out, hidden, std_init=std)
        self.fc_h_a = NoisyLinear(conv_out, hidden, std_init=std)
        self.fc_z_v = NoisyLinear(hidden, self.atoms,                std_init=std)
        self.fc_z_a = NoisyLinear(hidden, action_space * self.atoms, std_init=std)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        h = self.convs(x).flatten(1)
        v = F.relu(self.fc_h_v(h))
        a = F.relu(self.fc_h_a(h))
        v = self.fc_z_v(v).view(-1, 1,                self.atoms)
        a = self.fc_z_a(a).view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        if log:
            return F.log_softmax(q, dim=2)
        return F.softmax(q, dim=2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ── B: Replay memory ─────────────────────────────────────────────────────────

_Transition_dtype = np.dtype([
    ('timestep', np.int32),
    ('state', np.uint8, (3, 64, 64)),
    ('action', np.int32),
    ('reward', np.float32),
    ('nonterminal', np.bool_),
])
_blank_trans = (0, np.zeros((3, 64, 64), dtype=np.uint8), 0, 0.0, False)


class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size  = size
        self.full  = False
        self.tree_start = 2 ** (size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([_blank_trans] * size, dtype=_Transition_dtype)
        self.max  = 1

    def _update_nodes(self, indices):
        children = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique  = np.unique(parents)
        self._update_nodes(unique)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices, values):
        self.sum_tree[indices] = values
        self._propagate(indices)
        self.max = max(float(np.max(values)), self.max)

    def _update_index(self, index, value):
        self.sum_tree[index] = value
        self._propagate_index(index)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data
        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full  = self.full or self.index == 0
        self.max   = max(value, self.max)

    def _retrieve(self, indices, values):
        children = (indices * 2 + np.expand_dims([1, 2], axis=1))
        if children[0, 0] >= self.sum_tree.shape[0]:
            return indices
        elif children[0, 0] >= self.tree_start:
            children = np.minimum(children, self.sum_tree.shape[0] - 1)
        left_vals  = self.sum_tree[children[0]]
        choices    = np.greater(values, left_vals).astype(np.int32)
        succ_idx   = children[choices, np.arange(indices.size)]
        succ_vals  = values - choices * left_vals
        return self._retrieve(succ_idx, succ_vals)

    def find(self, values):
        indices    = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return self.sum_tree[indices], data_index, indices

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory:
    def __init__(self, args, capacity):
        self.device           = args.device
        self.capacity         = capacity
        self.history          = 1          # temporal depth fixed to 1 (RGB channels ≠ time stack)
        self.discount         = args.discount
        self.n                = args.multi_step
        self.priority_weight  = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.t                = 0
        self.n_step_scaling   = torch.tensor(
            [self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device
        )
        self.transitions      = SegmentTree(capacity)
        self._outcome_label   = np.zeros(capacity, dtype=np.int8)

    def current_index(self) -> int:
        return (self.transitions.index - 1) % self.capacity

    def set_outcome_label(self, buf_positions: np.ndarray, label: int) -> None:
        self._outcome_label[buf_positions % self.capacity] = label

    def append(self, state, action, reward, terminal):
        state = state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        self.transitions.append((self.t, state.numpy(), action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1

    def _get_transitions(self, idxs):
        tidxs       = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(tidxs)
        firsts      = transitions['timestep'] == 0
        blank_mask  = np.zeros_like(firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], firsts[:, t + 1])
        for t in range(self.history, self.history + self.n):
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], firsts[:, t])
        transitions[blank_mask] = _blank_trans
        return transitions

    def _get_samples_from_segments(self, batch_size, p_total):
        seg_len = p_total / batch_size
        seg_starts = np.arange(batch_size) * seg_len
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, seg_len, [batch_size]) + seg_starts
            probs, idxs, tree_idxs = self.transitions.find(samples)
            if (np.all((self.transitions.index - idxs) % self.capacity > self.n) and
                    np.all((idxs - self.transitions.index) % self.capacity >= self.history) and
                    np.all(probs != 0)):
                valid = True
        transitions  = self._get_transitions(idxs)
        all_states   = transitions['state']
        states      = torch.tensor(all_states[:, 0], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n], device=self.device, dtype=torch.float32).div_(255)
        actions     = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        rewards     = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R           = torch.matmul(rewards, self.n_step_scaling)
        nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

    def sample(self, batch_size):
        p_total  = self.transitions.total()
        probs, _, tree_idxs, states, actions, returns, next_states, nonterminals = \
            self._get_samples_from_segments(batch_size, p_total)
        probs   = probs / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights  = (capacity * probs) ** -self.priority_weight
        weights  = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[
            np.arange(self.current_idx - self.history + 1, self.current_idx + 1)
        ]
        firsts     = transitions['timestep'] == 0
        blank_mask = np.zeros_like(firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], firsts[t + 1])
        transitions[blank_mask] = _blank_trans
        state = torch.tensor(transitions['state'][-1], dtype=torch.float32, device=self.device).div_(255)
        self.current_idx += 1
        return state

    def save_buffer(self, path: str) -> None:
        np.savez_compressed(
            path,
            data=self.transitions.data,
            sum_tree=self.transitions.sum_tree,
            outcome_label=self._outcome_label,
            index=np.array(self.transitions.index),
            full=np.array(self.transitions.full),
            tree_max=np.array(self.transitions.max),
            priority_weight=np.array(self.priority_weight),
            t=np.array(self.t),
        )

    def load_buffer(self, path: str) -> None:
        d = np.load(path, allow_pickle=False)
        self.transitions.data[:]     = d['data']
        self.transitions.sum_tree[:] = d['sum_tree']
        self._outcome_label[:]       = d['outcome_label']
        self.transitions.index       = int(d['index'])
        self.transitions.full        = bool(d['full'])
        self.transitions.max         = float(d['tree_max'])
        self.priority_weight         = float(d['priority_weight'])
        self.t                       = int(d['t'])


# ── C: Training internals ────────────────────────────────────────────────────

class TrainingGradientCapture:
    """Accumulates per-step training gradients between checkpoints."""

    def __init__(self, named_params):
        self._named_params = named_params
        self._agg = OnlineGradientAggregator(named_params)

    def accumulate(self, named_params):
        self._agg.accumulate(named_params)

    def consume(self) -> dict | None:
        if self._agg.count == 0:
            return None
        mean_g = {name: g.clone().cpu() for name, g in self._agg.mean_gradient().items()}
        for k in self._agg.sum_grads:
            self._agg.sum_grads[k].zero_()
        self._agg.count = 0
        return mean_g


class _RainbowAgent:
    """Rainbow DQN agent: learn(), act(), save(), update_target_net()."""

    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms        = args.atoms
        self.Vmin         = args.V_min
        self.Vmax         = args.V_max
        self.support      = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
        self.delta_z      = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size   = args.batch_size
        self.n            = args.multi_step
        self.discount     = args.discount
        self.norm_clip    = args.norm_clip
        self.device       = args.device

        self._grad_capture: TrainingGradientCapture | None = None

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu', weights_only=False)
                if 'online_net_state_dict' in state_dict:
                    state_dict = state_dict['online_net_state_dict']
                if 'conv1.weight' in state_dict:
                    for old, new in (
                        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'),
                        ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'),
                        ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias'),
                    ):
                        state_dict[new] = state_dict[old]
                        del state_dict[old]
                self.online_net.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )

    def enable_grad_capture(self):
        self._grad_capture = TrainingGradientCapture(list(self.online_net.named_parameters()))

    def consume_grad_capture(self) -> dict | None:
        return self._grad_capture.consume() if self._grad_capture is not None else None

    def reset_noise(self):
        self.online_net.reset_noise()

    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem, mora_tracker=None):
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            mem.sample(self.batch_size)

        log_ps   = self.online_net(states, log=True)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            pns     = self.online_net(next_states)
            dns     = self.support.expand_as(pns) * pns
            argmax  = dns.sum(2).argmax(1)
            self.target_net.reset_noise()
            pns     = self.target_net(next_states)
            pns_a   = pns[range(self.batch_size), argmax]

            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            b  = (Tz - self.Vmin) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            m      = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        loss = -torch.sum(m * log_ps_a, 1)

        self.online_net.zero_grad()

        # MORA: r=0-masked backward before main pass (retain_graph=True)
        grad_flat_r0 = None
        if mora_tracker is not None:
            r0_mask = (returns <= 0).float()
            if r0_mask.any():
                (weights * loss * r0_mask).mean().backward(retain_graph=True)
                grad_flat_r0 = torch.cat([
                    p.grad.detach().flatten()
                    for n, p in self.online_net.named_parameters()
                    if p.grad is not None and 'sigma' not in n
                ])
                self.online_net.zero_grad()

        (weights * loss).mean().backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

        if self._grad_capture is not None:
            self._grad_capture.accumulate([
                (n, p) for n, p in self.online_net.named_parameters()
                if p.grad is not None and 'sigma' not in n
            ])

        if mora_tracker is not None:
            grad_flat = torch.cat([
                p.grad.detach().flatten()
                for n, p in self.online_net.named_parameters()
                if p.grad is not None and 'sigma' not in n
            ])
            mora_tracker.accumulate(grad_flat, grad_flat_r0=grad_flat_r0)

        self.optimiser.step()
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, name, global_step=0, episode_count=0, args=None, training_gradient=None):
        os.makedirs(path, exist_ok=True)
        args_dict = {}
        if args is not None:
            args_dict = {
                'atoms': args.atoms, 'hidden_size': args.hidden_size,
                'architecture': args.architecture, 'history_length': args.history_length,
                'V_min': args.V_min, 'V_max': args.V_max,
                'multi_step': args.multi_step, 'discount': args.discount,
                'noisy_std': args.noisy_std, 'n_actions': self.action_space,
                'priority_exponent': args.priority_exponent,
                'priority_weight':   args.priority_weight,
                'T_max':             args.T_max,
                'learn_start':       args.learn_start,
            }
        torch.save({
            'global_step':             global_step,
            'episode_count':           episode_count,
            'online_net_state_dict':   self.online_net.state_dict(),
            'target_net_state_dict':   self.target_net.state_dict(),
            'optimizer_state_dict':    self.optimiser.state_dict(),
            'args_dict':               args_dict,
            'training_gradient':       training_gradient,
        }, os.path.join(path, name))

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()


# ── D: MORA training ─────────────────────────────────────────────────────────

class MORAPriorityTracker:
    """Rolling gradient-coherence modifier for PER priority adjustment.

    Computes p_i = |δ_i|^α · m_{r=0} where
    m_{r=0} = clamp(1 − cos(G_{r=0 in success}, G_failure), 0, 1) + ε

    Always active during training (hardcoded per pipeline spec).
    """

    def __init__(self, k: int, epsilon: float = 0.01, percentile_x: int = 25, window: int = 25):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k            = k
        self.epsilon      = epsilon
        self.percentile_x = percentile_x
        self.window       = window

        self._preparatory_grads: deque = deque(maxlen=k)
        self._failure_grads:     deque = deque(maxlen=k)

        self._episode_accum:    Optional[torch.Tensor] = None
        self._episode_n:        int = 0
        self._episode_r0_accum: Optional[torch.Tensor] = None
        self._episode_r0_n:     int = 0

        self._recent_returns: deque = deque(maxlen=window)
        self.m: float = epsilon
        self._last_frac_neg: float = float("nan")

    def accumulate(self, grad_flat: torch.Tensor, grad_flat_r0: Optional[torch.Tensor] = None) -> None:
        g = grad_flat.cpu()
        self._episode_accum = g.clone() if self._episode_accum is None else self._episode_accum.add_(g)
        self._episode_n += 1
        if grad_flat_r0 is not None:
            g0 = grad_flat_r0.cpu()
            self._episode_r0_accum = g0.clone() if self._episode_r0_accum is None else self._episode_r0_accum.add_(g0)
            self._episode_r0_n += 1

    def episode_end(self, ep_return: float) -> None:
        if self._episode_n == 0 or self._episode_accum is None:
            self._episode_accum = None;  self._episode_n = 0
            self._episode_r0_accum = None;  self._episode_r0_n = 0
            return

        mean_grad_all = (self._episode_accum / self._episode_n).detach()
        self._recent_returns.append(ep_return)
        recent = list(self._recent_returns)

        if len(recent) >= 2:
            low  = float(np.percentile(recent, self.percentile_x))
            high = float(np.percentile(recent, 100 - self.percentile_x))
            if ep_return >= high:
                if self._episode_r0_n > 0 and self._episode_r0_accum is not None:
                    self._preparatory_grads.append(
                        (self._episode_r0_accum / self._episode_r0_n).detach()
                    )
            elif ep_return <= low:
                self._failure_grads.append(mean_grad_all)

        if self._preparatory_grads and self._failure_grads:
            G_prep = torch.stack(list(self._preparatory_grads)).mean(dim=0)
            G_f    = torch.stack(list(self._failure_grads)).mean(dim=0)
            c = F.cosine_similarity(G_prep.flatten().unsqueeze(0), G_f.flatten().unsqueeze(0)).item()
            self.m = float(np.clip(1.0 - c, 0.0, 1.0)) + self.epsilon

        self._episode_accum = None;  self._episode_n = 0
        self._episode_r0_accum = None;  self._episode_r0_n = 0

    def set_last_frac_neg(self, frac: float) -> None:
        self._last_frac_neg = frac

    def log_m(self, log_path: str, episode: int) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["episode", "m", "frac_neg_batch"])
            writer.writerow([episode, f"{self.m:.6f}", f"{self._last_frac_neg:.4f}"])


class OutcomeTracker:
    """Episode outcome classifier (success +1, failure -1, neutral 0) via percentile partitioning."""

    def __init__(self, percentile_x: int = 25, window: int = 25):
        self.percentile_x = percentile_x
        self.window       = window
        self._recent_returns: deque = deque(maxlen=window)

    def classify(self, ep_return: float) -> int:
        self._recent_returns.append(ep_return)
        recent = list(self._recent_returns)
        if len(recent) < 2:
            return 0
        low  = float(np.percentile(recent, self.percentile_x))
        high = float(np.percentile(recent, 100 - self.percentile_x))
        if ep_return >= high:
            return 1
        elif ep_return <= low:
            return -1
        return 0


# ── E: Training env adapter ──────────────────────────────────────────────────

class _CrafterEnvWrapper:
    """Adapts a Crafter gymnasium env to Rainbow's (state, reward, done) interface."""

    def __init__(self, args, seed=None):
        self.device = args.device
        self._env   = make_crafter_env(seed=seed)
        self._info  = {}

    def action_space(self):
        return self._env.action_space.n

    def reset(self):
        obs, info = self._env.reset()
        self._info = info
        return _obs_to_state(obs, self.device)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._info = info
        return _obs_to_state(obs, self.device), reward, terminated or truncated

    def train(self): pass
    def eval(self):  pass

    def close(self):
        self._env.close()

    @property
    def info(self):
        return self._info


def _obs_to_state(obs, device):
    """HWC uint8 numpy → CHW float32 tensor on device."""
    arr = np.array(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


# ── F: Gradient analysis utilities ───────────────────────────────────────────

def _compute_nstep_returns(rewards, dones, gamma, n):
    T = len(rewards)
    R          = torch.zeros(T, dtype=torch.float32)
    nonterminal = torch.zeros(T, dtype=torch.float32)
    for t in range(T):
        disc, ret, terminal = 1.0, 0.0, False
        for k in range(n):
            if t + k >= T:
                terminal = True; break
            ret  += disc * rewards[t + k].item()
            disc *= gamma
            if dones[t + k].item() > 0.5:
                terminal = True; break
        R[t]          = ret
        nonterminal[t] = 0.0 if terminal else 1.0
    return R, nonterminal


def _project_distribution(next_obs, online_net, target_net,
                           R, nonterminal, support, Vmin, Vmax, delta_z, atoms, gamma, n, device):
    T = len(R)
    R           = R.to(device)
    nonterminal = nonterminal.unsqueeze(1).to(device)
    support_dev = support.to(device)
    with torch.no_grad():
        pns_online = online_net(next_obs)
        dns        = support_dev.expand_as(pns_online) * pns_online
        argmax_a   = dns.sum(2).argmax(1)
        pns_target = target_net(next_obs)
        pns_a      = pns_target[range(T), argmax_a]
        Tz = R.unsqueeze(1) + nonterminal * (gamma ** n) * support_dev.unsqueeze(0)
        Tz = Tz.clamp(min=Vmin, max=Vmax)
        b  = (Tz - Vmin) / delta_z
        l  = b.floor().to(torch.int64)
        u  = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (atoms - 1)) * (l == u)] += 1
        m      = pns_a.new_zeros(T, atoms)
        offset = torch.linspace(0, (T - 1) * atoms, T).unsqueeze(1).expand(T, atoms).to(l)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))
    return m


def _forward_per_loss(episode, online_net, target_net,
                      support, Vmin, Vmax, delta_z, atoms, gamma, n, device):
    """Per-transition distributional Bellman loss for one episode (_RbEp format)."""
    obs     = episode.observations.to(device)
    actions = episode.actions.to(device)
    T       = len(obs)
    next_obs = torch.zeros_like(obs)
    if n < T:
        next_obs[:-n] = obs[n:]
    R, nonterminal = _compute_nstep_returns(episode.rewards, episode.dones, gamma, n)
    m = _project_distribution(
        next_obs, online_net, target_net,
        R, nonterminal, support, Vmin, Vmax, delta_z, atoms, gamma, n, device,
    )
    log_ps   = online_net(obs, log=True)
    log_ps_a = log_ps[range(T), actions]
    return -torch.sum(m * log_ps_a, dim=1)


def _compute_is_weights(all_per_losses, alpha, beta):
    priorities = all_per_losses.clamp(min=1e-8) ** alpha
    probs      = priorities / priorities.sum()
    N          = len(all_per_losses)
    w          = (1.0 / (N * probs)) ** beta
    return w / w.max()


# ── G: MORA post-hoc analysis ────────────────────────────────────────────────

def _compute_sign_indicator_gradients(
    online_net, rb_eps, device, target_net, args_ns,
    max_episodes=None, track_coherence=False,
):
    """Indicator-weighted per-reward-sign gradients on intact full episodes.

    For each episode: computes full per-transition Bellman loss once, then
    backpropagates (indicator_s / n_s * per_loss).sum() for each sign group.

    Args:
        rb_eps:          list[_RbEp] - episodes in CHW float32 format
        track_coherence: if True, captures per-episode gradient dicts

    Returns:
        dict with keys "pos", "neu", "neg", each: {"raw": dict|None, "batch_grads": list}
    """
    atoms   = args_ns.atoms
    Vmin    = args_ns.V_min
    Vmax    = args_ns.V_max
    delta_z = (Vmax - Vmin) / (atoms - 1)
    support = torch.linspace(Vmin, Vmax, atoms).to(device)
    gamma   = args_ns.discount
    n       = args_ns.multi_step

    named_params = list(online_net.named_parameters())

    # Stratified subsample: keep episodes with non-neutral rewards first
    if max_episodes is not None and len(rb_eps) > max_episodes:
        rng = np.random.default_rng(42)
        non_neutral  = [ep for ep in rb_eps if (ep.rewards > 0).any() or (ep.rewards < 0).any()]
        neutral_only = [ep for ep in rb_eps if not (ep.rewards > 0).any() and not (ep.rewards < 0).any()]
        if len(non_neutral) >= max_episodes:
            idx = rng.choice(len(non_neutral), size=max_episodes, replace=False)
            rb_eps = [non_neutral[i] for i in sorted(idx)]
        else:
            n_fill = max_episodes - len(non_neutral)
            fill   = []
            if n_fill > 0 and neutral_only:
                idx  = rng.choice(len(neutral_only), size=min(n_fill, len(neutral_only)), replace=False)
                fill = [neutral_only[i] for i in sorted(idx)]
            rb_eps = non_neutral + fill

    n_pos_total = int(sum((ep.rewards > 0).sum().item() for ep in rb_eps))
    n_neu_total = int(sum((ep.rewards == 0).sum().item() for ep in rb_eps))
    n_neg_total = int(sum((ep.rewards < 0).sum().item() for ep in rb_eps))
    n_total     = {"pos": n_pos_total, "neu": n_neu_total, "neg": n_neg_total}

    aggs     = {k: OnlineGradientAggregator(named_params) for k in ("pos", "neu", "neg")}
    ep_grads = {"pos": [], "neu": [], "neg": []}

    for ep in tqdm(rb_eps, desc="MOR sign grads", leave=False):
        if len(ep.rewards) < 2:
            continue

        per_loss = _forward_per_loss(
            ep, online_net, target_net,
            support, Vmin, Vmax, delta_z, atoms, gamma, n, device,
        )

        r = ep.rewards.to(device)
        masks  = {"pos": (r > 0).float(), "neu": (r == 0).float(), "neg": (r < 0).float()}
        active = [(k, masks[k]) for k in ("pos", "neu", "neg")
                  if masks[k].sum().item() > 0 and n_total[k] > 0]

        for i, (key, mask) in enumerate(active):
            is_last = (i == len(active) - 1)
            online_net.zero_grad()
            (mask / n_total[key] * per_loss).sum().backward(retain_graph=not is_last)
            if track_coherence:
                ep_agg = OnlineGradientAggregator(named_params)
                ep_agg.accumulate(named_params)
                ep_grads[key].append(ep_agg.mean_gradient())
            aggs[key].accumulate(named_params)

        online_net.zero_grad()

    return {
        key: {
            "raw": {name: aggs[key].sum_grads[name].clone() for name in aggs[key].sum_grads}
                   if aggs[key].count > 0 else None,
            "batch_grads": ep_grads[key],
        }
        for key in ("pos", "neu", "neg")
    }


# ─────────────────────────────────────────────────────────────────────────────
# Half 2 - Pipeline glue
# ─────────────────────────────────────────────────────────────────────────────

# ── H: Env wrappers ──────────────────────────────────────────────────────────

class CrafterGymnasiumWrapper(gym.Env):
    """Adapts crafter.Env (old gym 4-tuple API) to gymnasium 5-tuple."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None, **crafter_kwargs):
        super().__init__()
        self._env = crafter.Env(**crafter_kwargs)
        self._rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space      = gym.spaces.Discrete(self._env.action_space.n)

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
    """Injects info["achievements"] bool dict and info["inventory"] each step."""

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


def make_crafter_env(seed=None, eps_weight: float = 0.9) -> gym.Env:
    """Return a Crafter env with achievement/inventory info injected each step."""
    return CrafterAchievementWrapper(CrafterGymnasiumWrapper(seed=seed))


# ── I: Data containers ───────────────────────────────────────────────────────

@dataclass
class RainbowModel:
    """Loaded Rainbow DQN checkpoint. Passed as the 'model' arg to all entity methods."""
    online_net:         nn.Module
    target_net:         nn.Module
    args:               argparse.Namespace
    support:            torch.Tensor       # (atoms,) precomputed value support on CPU
    adam_state:         dict               # optimizer_state_dict['state'] keyed by param index
    global_step:        int = 0
    training_gradient:  dict | None = None  # mean gradient from TrainingGradientCapture between checkpoints


@dataclass
class _RbEp:
    """Core EpisodeData re-expressed in Rainbow's CHW float32 format."""
    observations: torch.Tensor   # (T, C, H, W) float32 [0, 1]
    actions:      torch.Tensor   # (T,) long
    rewards:      torch.Tensor   # (T,) float32
    dones:        torch.Tensor   # (T,) float32


# ── J: RainbowCrafter entity class ───────────────────────────────────────────

class RainbowCrafter:
    """Rainbow DQN trained on Crafter.

    Implements the Entity protocol. train() is always active with MORA-PER
    and outcome-conditioned atom weighting (both hardcoded, no flags needed).
    """

    entity_id             = ENTITY_ID
    obs_shape             = OBS_SHAPE
    n_actions             = N_ACTIONS
    hook_layer            = HOOK_LAYER
    achievement_names     = ACHIEVEMENT_NAMES
    achievement_label_map = ACHIEVEMENT_LABEL_MAP
    achievement_groups    = ACHIEVEMENT_GROUPS

    # ── Protocol methods ─────────────────────────────────────────────────────

    def make_env(self, seed: int) -> gym.Env:
        return make_crafter_env(seed=seed)

    def load_checkpoint(self, path: str, device: str) -> RainbowModel:
        ckpt      = torch.load(path, map_location=device, weights_only=False)
        args_dict = ckpt.get("args_dict", {})
        n_actions = args_dict.get("n_actions", N_ACTIONS)
        args_ns   = _build_args(args_dict, n_actions, device)

        online_net = DQN(args_ns, n_actions)
        online_net.load_state_dict(ckpt["online_net_state_dict"])
        online_net.to(device).eval()

        target_net = DQN(args_ns, n_actions)
        sd = ckpt.get("target_net_state_dict", ckpt["online_net_state_dict"])
        target_net.load_state_dict(sd)
        target_net.to(device).eval()
        for p in target_net.parameters():
            p.requires_grad = False

        support    = torch.linspace(args_ns.V_min, args_ns.V_max, args_ns.atoms)
        opt_state  = ckpt.get("optimizer_state_dict", {})
        adam_state = opt_state.get("state", {})

        return RainbowModel(
            online_net        = online_net,
            target_net        = target_net,
            args              = args_ns,
            support           = support,
            adam_state        = adam_state,
            global_step       = int(ckpt.get("global_step", 0)),
            training_gradient = ckpt.get("training_gradient"),
        )

    def select_action(self, model: RainbowModel, obs: np.ndarray,
                      deterministic: bool = True) -> int:
        if deterministic:
            model.online_net.eval()
        else:
            model.online_net.train()
        dev  = next(model.online_net.parameters()).device
        obs_t = torch.from_numpy(np.array(obs, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            q = (model.online_net(obs_t) * model.support.to(dev)).sum(2)
        return int(q.argmax(1).item())

    def compute_eps(self, achievements: dict[str, bool], inventory: dict) -> float:
        n_achieved     = sum(1 for v in achievements.values() if v)
        materials_frac = _materials_fraction(achievements, inventory)
        return float(n_achieved + EPS_WEIGHT * materials_frac)

    def preprocess_obs(self, model: RainbowModel, obs_np: np.ndarray,
                       device: str) -> torch.Tensor:
        arr = torch.from_numpy(np.array(obs_np, dtype=np.float32) / 255.0)
        if arr.dim() == 3:
            arr = arr.unsqueeze(0)
        return arr.permute(0, 3, 1, 2).to(device)

    def get_policy(self, model: RainbowModel) -> nn.Module:
        return model.online_net

    def compute_gradients(self, model: RainbowModel, episodes: list[EpisodeData],
                          device: str, batch_size: int = 10) -> GradientResult:
        """IS-weighted gradient and named variants: uniform, adam, positive, neutral."""
        online_net = model.online_net.to(device).eval()
        target_net = model.target_net.to(device).eval()
        args       = model.args

        atoms   = args.atoms
        Vmin    = args.V_min
        Vmax    = args.V_max
        delta_z = (Vmax - Vmin) / (atoms - 1)
        support = model.support.to(device)
        n       = args.multi_step
        gamma   = args.discount

        alpha       = getattr(args, "priority_exponent", 0.5)
        beta_start  = getattr(args, "priority_weight",   0.4)
        T_max       = getattr(args, "T_max",             int(10e6))
        learn_start = getattr(args, "learn_start",       int(20e3))
        frac        = max(0.0, min(1.0,
            (model.global_step - learn_start) / max(T_max - learn_start, 1)
        ))
        beta_used = beta_start + (1.0 - beta_start) * frac

        rb_eps = [_to_rb_ep(ep) for ep in episodes]
        named_params = list(online_net.named_parameters())

        # Pass 1: per-transition losses for IS weights
        losses_detached = []
        with torch.no_grad():
            for ep in rb_eps:
                if len(ep.rewards) > 1:
                    per_loss = _forward_per_loss(
                        ep, online_net, target_net,
                        support, Vmin, Vmax, delta_z, atoms, gamma, n, device,
                    )
                    losses_detached.append(per_loss.detach())
                else:
                    losses_detached.append(None)

        valid = [l for l in losses_detached if l is not None]
        if valid:
            all_is_w = _compute_is_weights(torch.cat(valid), alpha, beta_used)
            is_w_per_ep: list[torch.Tensor | None] = []
            ptr = 0
            for l in losses_detached:
                if l is not None:
                    T = len(l)
                    is_w_per_ep.append(all_is_w[ptr: ptr + T])
                    ptr += T
                else:
                    is_w_per_ep.append(None)
        else:
            is_w_per_ep = [None] * len(rb_eps)

        # Pass 2: IS-weighted and uniform backward passes
        overall_is_agg      = OnlineGradientAggregator(named_params)
        overall_uniform_agg = OnlineGradientAggregator(named_params)
        per_episode_grads   = []

        for i in tqdm(range(0, len(rb_eps), batch_size), desc="rainbow grads", unit="batch", leave=False):
            batch_rb  = rb_eps[i: i + batch_size]
            batch_isw = is_w_per_ep[i: i + batch_size]

            batch_uniform_agg = OnlineGradientAggregator(named_params)
            for ep, is_w in zip(batch_rb, batch_isw):
                if len(ep.rewards) <= 1 or is_w is None:
                    continue

                # Uniform backward
                online_net.zero_grad()
                per_loss = _forward_per_loss(
                    ep, online_net, target_net,
                    support, Vmin, Vmax, delta_z, atoms, gamma, n, device,
                )
                per_loss.mean().backward(retain_graph=True)
                batch_uniform_agg.accumulate(named_params)
                overall_uniform_agg.accumulate(named_params)

                # IS-weighted backward
                online_net.zero_grad()
                (is_w.to(device) * per_loss).mean().backward()
                overall_is_agg.accumulate(named_params)
                online_net.zero_grad()

            per_episode_grads.append(batch_uniform_agg.l2_normalized())

        raw_mean = (overall_is_agg.mean_gradient() if overall_is_agg.count > 0
                    else overall_uniform_agg.mean_gradient())

        # Variants dict
        variants: dict[str, GradientResult] = {}

        if overall_uniform_agg.count > 0:
            variants["uniform"] = GradientResult(
                raw_mean    = overall_uniform_agg.mean_gradient(),
                per_episode = per_episode_grads,
                variants    = {},
                metadata    = {"beta_used": beta_used},
            )

        adam_grad = _extract_adam_gradient(model.adam_state, named_params)
        if adam_grad:
            variants["adam"] = GradientResult(
                raw_mean    = adam_grad,
                per_episode = [],
                variants    = {},
                metadata    = {"source": "optimizer_state"},
            )

        if model.training_gradient is not None:
            variants["training"] = GradientResult(
                raw_mean    = model.training_gradient,
                per_episode = [],
                variants    = {},
                metadata    = {"source": "live_training_batches"},
            )

        sign_grads = _compute_sign_indicator_gradients(
            online_net, rb_eps, device, target_net, args,
        )
        for key, label in [("pos", "positive"), ("neu", "neutral")]:
            if sign_grads[key]["raw"] is not None:
                variants[label] = GradientResult(
                    raw_mean    = sign_grads[key]["raw"],
                    per_episode = [],
                    variants    = {},
                    metadata    = {},
                )

        return GradientResult(
            raw_mean    = raw_mean,
            per_episode = per_episode_grads,
            variants    = variants,
            metadata    = {
                "n_atoms":       atoms,
                "v_min":         Vmin,
                "v_max":         Vmax,
                "beta_used":     beta_used,
                "n_transitions": sum(len(ep.rewards) for ep in episodes),
            },
        )

    def train(self, n_steps: int, on_checkpoint: callable, **kwargs) -> None:
        """Train Rainbow DQN on Crafter. MORA-PER and outcome weighting are always active.

        kwargs (all optional, defaults match original training configuration):
          seed (int, 123), device (str, auto), experiment_root (str, "experiment_root"),
          checkpoint_interval (int, 100_000), mora_k (int, 10),
          percentile_x (int, 25), resume_from (str|None), keep_checkpoints (bool, False)
        """
        seed                = kwargs.get("seed",                123)
        experiment_root     = kwargs.get("experiment_root",     "experiment_root")
        checkpoint_interval = kwargs.get("checkpoint_interval", 100_000)
        mora_k              = kwargs.get("mora_k",              10)
        percentile_x        = kwargs.get("percentile_x",        25)
        resume_from         = kwargs.get("resume_from",         None)
        keep_checkpoints    = kwargs.get("keep_checkpoints",    False)

        device_str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        args = _build_train_args(
            n_steps, seed=seed, experiment_root=experiment_root,
            checkpoint_interval=checkpoint_interval, device=device_str,
            resume_from=resume_from, **{
                k: v for k, v in kwargs.items()
                if k not in ("seed", "experiment_root", "checkpoint_interval",
                             "mora_k", "percentile_x", "resume_from",
                             "keep_checkpoints", "device")
            }
        )

        np.random.seed(args.seed)
        torch.manual_seed(np.random.randint(1, 10000))
        if torch.cuda.is_available() and args.device.type == "cuda":
            torch.cuda.manual_seed(np.random.randint(1, 10000))

        env = _CrafterEnvWrapper(args, seed=args.seed)
        dqn = _RainbowAgent(args, env)
        dqn.enable_grad_capture()

        if args.model and os.path.isfile(args.model):
            ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
            dqn.target_net.load_state_dict(ckpt['target_net_state_dict'])
            dqn.optimiser.load_state_dict(ckpt['optimizer_state_dict'])
            resume_step = ckpt.get('global_step', 0)
            resume_eps  = ckpt.get('episode_count', 0)
        else:
            resume_step = 0
            resume_eps  = 0

        mem = ReplayMemory(args, args.memory_capacity)
        buf_path = _buffer_live_path(experiment_root)
        if resume_step > 0 and os.path.exists(buf_path):
            mem.load_buffer(buf_path)

        priority_weight_increase = (1 - args.priority_weight) / max(args.T_max - args.learn_start, 1)
        if resume_step > 0:
            anneal_frac          = max(0.0, min(1.0,
                (resume_step - args.learn_start) / max(args.T_max - args.learn_start, 1)))
            mem.priority_weight  = args.priority_weight + (1 - args.priority_weight) * anneal_frac

        mora_tracker    = MORAPriorityTracker(k=mora_k, percentile_x=percentile_x)
        outcome_tracker = OutcomeTracker(percentile_x=percentile_x)

        dqn.train()
        done          = True
        episode_count = resume_eps
        ep_return     = 0.0
        seen_achievements: set = set()

        periodic_A = None
        periodic_B = None

        log_dir = os.path.join(experiment_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        episode_bufpos: list[int] = []
        return_log = open(os.path.join(log_dir, "rainbowreturnlog.txt"), "a")

        _POSTFIX = 500
        pbar = trange(1 + resume_step, args.T_max + 1)
        for T in pbar:
            if done:
                state     = env.reset()
                ep_return = 0.0

            if T % args.replay_frequency == 0:
                dqn.reset_noise()

            action           = dqn.act(state)
            next_state, reward, done = env.step(action)
            ep_return       += reward

            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            mem.append(state, action, reward, done)
            episode_bufpos.append(mem.current_index())

            if T % _POSTFIX == 0:
                buf_size = mem.capacity if mem.transitions.full else mem.transitions.index
                pbar.set_postfix(buf=f"{buf_size:,}", ep=episode_count)

            if done:
                episode_count += 1
                return_log.write(f"{T},{ep_return}\n")
                return_log.flush()

                mora_tracker.episode_end(ep_return)
                mora_tracker.log_m(os.path.join(log_dir, "mora_modifier_log.csv"), episode_count)

                label = outcome_tracker.classify(ep_return)
                if label != 0 and episode_bufpos:
                    mem.set_outcome_label(np.array(episode_bufpos, dtype=np.int64), label)
                episode_bufpos.clear()

                # Milestone checkpoints (save for research; do NOT trigger analysis pipeline)
                cur_ach = {k: bool(v) for k, v in env.info.get("achievements", {}).items()}
                new_ach = {a for a, v in cur_ach.items() if v} - seen_achievements
                if new_ach:
                    seen_achievements.update(new_ach)
                    for ach in sorted(new_ach):
                        _save_milestone(dqn, experiment_root, args, T, episode_count, ach)

            buf_size = mem.capacity if mem.transitions.full else mem.transitions.index
            if T >= args.learn_start and buf_size >= max(args.learn_start, args.batch_size * 10):
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem, mora_tracker=mora_tracker)

                if T % args.target_update == 0:
                    dqn.update_target_net()

                if args.checkpoint_interval > 0 and T % args.checkpoint_interval == 0:
                    _save_live(dqn, experiment_root, args, T, episode_count, mem=mem)
                    ckpt_path = _save_analysis(dqn, experiment_root, args, T, episode_count)

                    old_A      = periodic_A
                    periodic_A = periodic_B
                    periodic_B = ckpt_path

                    if not keep_checkpoints and old_A and old_A not in (periodic_A, periodic_B):
                        _delete_ckpt(old_A)

                    on_checkpoint(T, ckpt_path)

            state = next_state

        return_log.close()
        env.close()

        if not keep_checkpoints:
            for held in (periodic_A, periodic_B):
                if held and os.path.exists(held):
                    _delete_ckpt(held)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_args(args_dict: dict, n_actions: int, device: str) -> argparse.Namespace:
    """Build analysis-time argparse.Namespace from a checkpoint args_dict."""
    return argparse.Namespace(
        atoms             = args_dict.get("atoms",            51),
        hidden_size       = args_dict.get("hidden_size",      512),
        architecture      = args_dict.get("architecture",     "canonical"),
        history_length    = args_dict.get("history_length",   3),
        V_min             = args_dict.get("V_min",           -10),
        V_max             = args_dict.get("V_max",            10),
        multi_step        = args_dict.get("multi_step",       3),
        discount          = args_dict.get("discount",         0.99),
        noisy_std         = args_dict.get("noisy_std",        0.1),
        priority_exponent = args_dict.get("priority_exponent", 0.5),
        priority_weight   = args_dict.get("priority_weight",   0.4),
        T_max             = args_dict.get("T_max",             int(10e6)),
        learn_start       = args_dict.get("learn_start",       int(20e3)),
        device            = torch.device(device),
        model             = None,
    )


def _build_train_args(n_steps: int, **kwargs) -> argparse.Namespace:
    """Build training-time argparse.Namespace with defaults from original train.py."""
    device_str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return argparse.Namespace(
        seed                = kwargs.get("seed",                123),
        T_max               = n_steps,
        history_length      = kwargs.get("history_length",      3),
        architecture        = kwargs.get("architecture",         "canonical"),
        hidden_size         = kwargs.get("hidden_size",          512),
        noisy_std           = kwargs.get("noisy_std",            0.1),
        atoms               = kwargs.get("atoms",                51),
        V_min               = kwargs.get("V_min",               -10.0),
        V_max               = kwargs.get("V_max",                10.0),
        model               = kwargs.get("resume_from",          None),
        memory_capacity     = kwargs.get("memory_capacity",      int(500e3)),
        replay_frequency    = kwargs.get("replay_frequency",     4),
        priority_exponent   = kwargs.get("priority_exponent",    0.5),
        priority_weight     = kwargs.get("priority_weight",      0.4),
        multi_step          = kwargs.get("multi_step",           3),
        discount            = kwargs.get("discount",             0.99),
        target_update       = kwargs.get("target_update",        int(8e3)),
        reward_clip         = kwargs.get("reward_clip",          1),
        learning_rate       = kwargs.get("learning_rate",        0.0000625),
        adam_eps            = kwargs.get("adam_eps",             1.5e-4),
        batch_size          = kwargs.get("batch_size",           32),
        norm_clip           = kwargs.get("norm_clip",            10),
        learn_start         = kwargs.get("learn_start",          int(20e3)),
        evaluation_size     = kwargs.get("evaluation_size",      500),
        checkpoint_interval = kwargs.get("checkpoint_interval",  100_000),
        experiment_root     = kwargs.get("experiment_root",      "experiment_root"),
        device              = torch.device(device_str),
    )


def _to_rb_ep(ep: EpisodeData) -> _RbEp:
    """Convert core.data.EpisodeData (HWC uint8) → _RbEp (CHW float32 [0,1])."""
    obs = torch.from_numpy(ep.observations.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return _RbEp(
        observations = obs,
        actions      = torch.from_numpy(ep.actions.astype(np.int64)),
        rewards      = torch.from_numpy(ep.rewards.astype(np.float32)),
        dones        = torch.from_numpy(ep.dones.astype(np.float32)),
    )


def _extract_adam_gradient(
    adam_state: dict, named_params: list
) -> dict[str, torch.Tensor] | None:
    """Build Adam effective update direction from optimizer state.

    Returns {param_name: tensor}; sigma parameters are zeroed out.
    Returns None if adam_state is empty (pre-dates gradient capture).
    """
    if not adam_state:
        return None
    result = {}
    for idx, (name, param) in enumerate(named_params):
        state = adam_state.get(idx)
        if state is None:
            continue
        if "sigma" in name:
            result[name] = torch.zeros_like(param.data)
            continue
        exp_avg    = state["exp_avg"].cpu().float()
        exp_avg_sq = state["exp_avg_sq"].cpu().float()
        step_raw   = state.get("step", 1)
        step       = int(step_raw.item()) if isinstance(step_raw, torch.Tensor) else int(step_raw)
        step       = max(step, 1)
        bc1        = 1.0 - 0.9   ** step
        bc2        = 1.0 - 0.999 ** step
        m_hat      = exp_avg    / bc1
        v_hat      = exp_avg_sq / bc2
        result[name] = (m_hat / (v_hat.sqrt() + ADAM_EPS)).to(param.data.dtype)
    return result if result else None


def _materials_fraction(achievements: dict[str, bool], inventory: dict) -> float:
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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_dir(experiment_root: str) -> str:
    d = os.path.join(experiment_root, "checkpoints", "rainbow")
    os.makedirs(d, exist_ok=True)
    return d


def _buffer_live_path(experiment_root: str) -> str:
    return os.path.join(_ckpt_dir(experiment_root), "replay_buffer_live.npz")


def _save_live(dqn, experiment_root, args, global_step, episode_count, mem=None):
    dqn.save(_ckpt_dir(experiment_root), "checkpoint_live.pt",
             global_step=global_step, episode_count=episode_count, args=args)
    if mem is not None:
        mem.save_buffer(_buffer_live_path(experiment_root))


def _save_analysis(dqn, experiment_root, args, global_step, episode_count) -> str:
    name = f"checkpoint_step{global_step}.pt"
    training_gradient = dqn.consume_grad_capture()
    dqn.save(_ckpt_dir(experiment_root), name,
             global_step=global_step, episode_count=episode_count, args=args,
             training_gradient=training_gradient)
    return os.path.join(_ckpt_dir(experiment_root), name)


def _save_milestone(dqn, experiment_root, args, global_step, episode_count, achievement) -> str:
    name = f"milestone_first_{achievement}_ep{episode_count}.pt"
    dqn.save(_ckpt_dir(experiment_root), name,
             global_step=global_step, episode_count=episode_count, args=args)
    return os.path.join(_ckpt_dir(experiment_root), name)


def _delete_ckpt(path: str):
    try:
        os.remove(path)
    except OSError:
        pass
