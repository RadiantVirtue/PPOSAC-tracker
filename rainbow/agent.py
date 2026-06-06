# Adapted from Rainbow/agent.py.
# save() stores a rich checkpoint dict including both online and target net weights
# so the offline distributional Bellman loss can be reconstructed during analysis.
from __future__ import annotations
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rainbow.model import DQN
from shared.gradient_utils import OnlineGradientAggregator


class TrainingGradientCapture:
    """Accumulates per-step training gradients between checkpoints.

    Call accumulate() after each backward pass (before optimizer.step()).
    Call consume() at checkpoint time to get the mean gradient and reset.

    Captures the gradient of the IS-weighted Bellman loss - i.e. the direction
    the optimizer actually pulls parameters - for later comparison against the
    offline counterfactual gradient (G_counterfactual).
    """

    def __init__(self, named_params):
        self._named_params = named_params
        self._agg = OnlineGradientAggregator(named_params)

    def accumulate(self, named_params):
        """Accumulate current .grad from all parameters (call after backward)."""
        self._agg.accumulate(named_params)

    def consume(self) -> dict | None:
        """Return mean gradient dict and reset the accumulator.

        Returns None if no steps have been accumulated since last consume().
        Gradient dict format: {param_name: CPU float32 tensor}.
        """
        if self._agg.count == 0:
            return None
        mean_g = {
            name: g.clone().cpu()
            for name, g in self._agg.mean_gradient().items()
        }
        # Reset
        for k in self._agg.sum_grads:
            self._agg.sum_grads[k].zero_()
        self._agg.count = 0
        return mean_g


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    # Optional gradient capture for Experiment 3 (counterfactual validation).
    # Set externally via agent.enable_grad_capture() before training.
    self._grad_capture: TrainingGradientCapture | None = None

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu', weights_only=False)
        # Rich checkpoint format (saved by Agent.save()) - extract online net only.
        # train.py's resume block will load target net + optimiser separately.
        if 'online_net_state_dict' in state_dict:
          state_dict = state_dict['online_net_state_dict']
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Outcome-conditioned atom weighting (Part 3).
    # outcome_tau=None disables weighting (standard C51 loss); set via args.outcome_tau.
    # Weights: w_j = C · softmax(±z/τ), so neutral weights = 1.0 exactly and gradient
    # magnitude is preserved.  Sensible τ for V_min=-10, V_max=10: {5, 10, 20}.
    outcome_tau = getattr(args, 'outcome_tau', None)
    self.outcome_tau = outcome_tau
    if outcome_tau is not None:
      C = float(self.atoms)
      self._w_success = C * torch.softmax( self.support / outcome_tau, dim=0)  # (atoms,)
      self._w_failure = C * torch.softmax(-self.support / outcome_tau, dim=0)  # (atoms,)
      self._w_neutral = torch.ones(self.atoms, dtype=torch.float32, device=args.device)  # (atoms,)

  def enable_grad_capture(self):
    """Enable training gradient capture for counterfactual validation (Experiment 3).

    Must be called after the online_net is constructed.  Gradients are
    accumulated by learn() after each backward pass and consumed at
    checkpoint time via consume_grad_capture().
    """
    self._grad_capture = TrainingGradientCapture(
        list(self.online_net.named_parameters())
    )

  def consume_grad_capture(self) -> dict | None:
    """Return accumulated mean training gradient and reset (call at checkpoint time)."""
    if self._grad_capture is None:
        return None
    return self._grad_capture.consume()

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def _outcome_atom_weights(self, outcome_labels: np.ndarray) -> torch.Tensor:
    """Build (batch_size, atoms) outcome weight tensor from int8 label array.

    labels: numpy int8, 1=success, 0=neutral, -1=failure.
    Weights are C·softmax(±z/τ) so neutral rows = 1.0 exactly.
    Result is on self.device.
    """
    labels = torch.tensor(outcome_labels, dtype=torch.int8, device=self.device)  # (B,)
    W = self._w_neutral.unsqueeze(0).expand(self.batch_size, -1).clone()  # (B, atoms)
    success_mask = labels == 1
    failure_mask = labels == -1
    if success_mask.any():
      W[success_mask] = self._w_success
    if failure_mask.any():
      W[failure_mask] = self._w_failure
    return W

  def learn(self, mem, mora_tracker=None):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights, outcome_labels = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    # Cross-entropy loss - with optional outcome-conditioned atom weighting.
    # Standard: L_i = -∑_j m_j · log p_j
    # Outcome-conditioned: L_i = -∑_j w_j(c_i) · m_j · log p_j
    # where w_j = C·softmax(±z/τ), neutral w_j=1 exactly, preserving gradient magnitude.
    if self.outcome_tau is not None:
      atom_weights = self._outcome_atom_weights(outcome_labels)  # (B, atoms)
      loss = -torch.sum(atom_weights * m * log_ps_a, 1)
    else:
      loss = -torch.sum(m * log_ps_a, 1)
    self.online_net.zero_grad()

    # MORA: compute r=0-only gradient via a separate masked backward before
    # the main backward.  retain_graph=True keeps the graph alive for the
    # main pass below.  Skipped when mora_tracker is None (no overhead for
    # standard PER / k=0 runs).
    grad_flat_r0 = None
    if mora_tracker is not None:
      r0_mask = (returns <= 0).float()
      if r0_mask.any():
        (weights * loss * r0_mask).mean().backward(retain_graph=True)
        grad_flat_r0 = torch.cat([
          p.grad.detach().flatten()
          for name, p in self.online_net.named_parameters()
          if p.grad is not None and 'sigma' not in name
        ])
        self.online_net.zero_grad()

    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm

    # Experiment 3: capture training gradient (after clip, before Adam step).
    # Excludes sigma params (no gradient in eval mode; consistent with MoR analysis).
    if self._grad_capture is not None:
        self._grad_capture.accumulate(
            [(n, p) for n, p in self.online_net.named_parameters()
             if p.grad is not None and 'sigma' not in n]
        )

    # MORA: save gradient (before Adam applies momentum) as a flat μ-only vector
    if mora_tracker is not None:
      grad_flat = torch.cat([
        p.grad.detach().flatten()
        for name, p in self.online_net.named_parameters()
        if p.grad is not None and 'sigma' not in name
      ])
      mora_tracker.accumulate(grad_flat, grad_flat_r0=grad_flat_r0)

    self.optimiser.step()

    mem.update_priorities(  # Update priorities of sampled transitions
      idxs,
      loss.detach().cpu().numpy(),
      mora_modifier=mora_tracker.m if mora_tracker is not None else None,
      rewards=returns.detach().cpu().numpy(),
      mora_tracker=mora_tracker,
    )

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def save(self, path, name, global_step=0, episode_count=0, args=None,
           training_gradient=None):
    """Save online + target nets plus optimizer state for resumption and offline analysis."""
    os.makedirs(path, exist_ok=True)
    args_dict = {}
    if args is not None:
      args_dict = {
        'atoms': args.atoms,
        'hidden_size': args.hidden_size,
        'architecture': args.architecture,
        'history_length': args.history_length,
        'V_min': args.V_min,
        'V_max': args.V_max,
        'multi_step': args.multi_step,
        'discount': args.discount,
        'noisy_std': args.noisy_std,
        'n_actions': self.action_space,
        # IS-weighting params (needed for offline gradient analysis)
        'priority_exponent': args.priority_exponent,
        'priority_weight':   args.priority_weight,
        'T_max':             args.T_max,
        'learn_start':       args.learn_start,
      }
    torch.save({
      'global_step': global_step,
      'episode_count': episode_count,
      'online_net_state_dict': self.online_net.state_dict(),
      'target_net_state_dict': self.target_net.state_dict(),
      'optimizer_state_dict': self.optimiser.state_dict(),
      'args_dict': args_dict,
      'training_gradient': training_gradient,  # None unless grad capture enabled
    }, os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
