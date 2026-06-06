"""Rainbow gradient analysis: G_uniform, G_IS (PER-corrected), G_reward variants.

Two-pass design: Pass 1 collects per-transition losses for IS weight normalization;
Pass 2 runs three backward() calls per episode to accumulate each gradient variant.
"""
import torch
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator, cosine_similarity_flat


def _compute_nstep_returns(rewards, dones, gamma, n):
    """Compute n-step discounted returns for each timestep in an episode.

    Args:
        rewards: (T,) float32 tensor
        dones:   (T,) float32 tensor  (1.0 at terminal steps)
        gamma:   discount factor
        n:       number of steps

    Returns:
        R:          (T,) float32 - n-step return for each step
        nonterminal:(T,) float32 - 1 if the nth-next state is non-terminal, else 0
    """
    T = len(rewards)
    R = torch.zeros(T, dtype=torch.float32)
    nonterminal = torch.zeros(T, dtype=torch.float32)

    for t in range(T):
        disc = 1.0
        ret = 0.0
        terminal_before_n = False
        for k in range(n):
            if t + k >= T:
                terminal_before_n = True
                break
            ret += disc * rewards[t + k].item()
            disc *= gamma
            if dones[t + k].item() > 0.5:
                terminal_before_n = True
                break
        R[t] = ret
        nonterminal[t] = 0.0 if terminal_before_n else 1.0

    return R, nonterminal


def _project_distribution(next_obs, online_net, target_net, R, nonterminal,
                           support, Vmin, Vmax, delta_z, atoms, gamma, n, device):
    """Compute projected target distribution m for a batch of transitions.

    Adapted from Rainbow/agent.py::learn().

    Args:
        next_obs:    (T, C, H, W) float32 - next observations
        online_net:  DQN (for double-Q action selection)
        target_net:  DQN (for target distribution)
        R:           (T,) float32 - n-step returns
        nonterminal: (T,) float32 - 1 if non-terminal
        support:     (atoms,) float32 - value support z
        Vmin, Vmax, delta_z, atoms: distribution parameters
        gamma:       discount factor
        n:           n-step
        device:      torch device

    Returns:
        m: (T, atoms) float32 - target distribution
    """
    T = len(R)
    R = R.to(device)
    nonterminal = nonterminal.unsqueeze(1).to(device)  # (T, 1)
    support_dev = support.to(device)

    with torch.no_grad():
        # Double-Q: action selection via online, evaluation via target
        pns_online = online_net(next_obs)                         # (T, action_space, atoms)
        dns = support_dev.expand_as(pns_online) * pns_online     # expected values
        argmax_a = dns.sum(2).argmax(1)                           # (T,) best actions

        pns_target = target_net(next_obs)                         # (T, action_space, atoms)
        pns_a = pns_target[range(T), argmax_a]                    # (T, atoms)

        # Tz = R^n + γ^n * z  (clipped to [Vmin, Vmax])
        Tz = R.unsqueeze(1) + nonterminal * (gamma ** n) * support_dev.unsqueeze(0)
        Tz = Tz.clamp(min=Vmin, max=Vmax)

        # L2 projection onto fixed support
        b = (Tz - Vmin) / delta_z
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)

        # Fix disappearing mass when l = b = u
        l[(u > 0) * (l == u)] -= 1
        u[(l < (atoms - 1)) * (l == u)] += 1

        m = pns_a.new_zeros(T, atoms)
        offset = (
            torch.linspace(0, (T - 1) * atoms, T)
            .unsqueeze(1).expand(T, atoms).to(l)
        )
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

    return m  # (T, atoms)


def _forward_per_loss(episode, online_net, target_net,
                      support, Vmin, Vmax, delta_z, atoms, gamma, n, device):
    """Compute per-transition distributional Bellman losses for one episode.

    Gradient tracking on online_net is controlled by the calling context:
      - wrap in torch.no_grad() for Pass 1 (IS pre-computation, no backward needed)
      - call bare for Pass 2 (backward passes)

    Args:
        episode:    EpisodeData namedtuple with .observations, .actions, .rewards, .dones
        online_net: DQN (frozen for analysis)
        target_net: DQN (frozen, no grad)
        support:    (atoms,) value support on device
        Vmin, Vmax, delta_z, atoms, gamma, n: distribution / training params
        device:     torch device

    Returns:
        per_loss: (T,) float32 - per-transition cross-entropy losses (>= 0)
    """
    obs     = episode.observations.to(device)   # (T, 3, H, W) float32
    actions = episode.actions.to(device)        # (T,) long
    T       = len(obs)

    next_obs = torch.zeros_like(obs)
    if n < T:
        next_obs[:-n] = obs[n:]  # next_obs[t] = obs[t+n]; last n rows stay zero (no bootstrap)

    R, nonterminal = _compute_nstep_returns(episode.rewards, episode.dones, gamma, n)

    m = _project_distribution(
        next_obs, online_net, target_net,
        R, nonterminal,
        support, Vmin, Vmax, delta_z, atoms, gamma, n, device
    )  # (T, atoms), always no_grad internally

    log_ps   = online_net(obs, log=True)        # (T, action_space, atoms)
    log_ps_a = log_ps[range(T), actions]        # (T, atoms)
    return -torch.sum(m * log_ps_a, dim=1)      # (T,) per-transition loss


def _compute_is_weights(all_per_losses, alpha, beta):
    """Derive IS weights from per-transition losses across an entire group.

    Replicates PER's IS correction: priority_i = L_i^α, P(i) = priority_i / Σ priority_k,
    w_i = (1 / (N · P(i)))^β, normalized by max(w).

    Args:
        all_per_losses: (N_total,) float32 tensor - detached per-transition losses
        alpha:          float - priority exponent (default 0.5)
        beta:           float - IS exponent (annealed 0.4 → 1.0)

    Returns:
        w: (N_total,) float32 - IS weights in (0, 1], max = 1.0
    """
    priorities = all_per_losses.clamp(min=1e-8) ** alpha
    probs = priorities / priorities.sum()         # P(i) = p_i^α / Σ p_k^α
    N = len(all_per_losses)
    w = (1.0 / (N * probs)) ** beta
    return w / w.max()                            # normalize by max (matches training)


def _compute_reward_weights(rewards, device, epsilon=0.01):
    """Reward-magnitude proxy weights for one episode.

    High-|reward| transitions (achievements, death) get higher weight.
    When all rewards are zero, degenerates to uniform (each weight = 1/T).

    Args:
        rewards: (T,) float32 CPU tensor
        device:  torch device
        epsilon: floor weight (prevents zero weight on zero-reward transitions)

    Returns:
        w: (T,) float32 on device, sums to 1
    """
    w = rewards.abs().to(device) + epsilon
    return w / w.sum()


def compute_group_gradient_with_coherence(
    online_net, episodes, batch_size: int = 10, device: str = "cpu",
    desc: str = "Grads",
    target_net=None, args_ns=None, global_step: int = 0,
    precomputed_is_weights=None,
):
    """Compute G_uniform, G_IS, and G_reward over a group of episodes.

    precomputed_is_weights: optional pre-normalised IS weights (list of per-episode tensors)
    for joint success+failure pool normalisation. When None, Pass 1 computes them within-group.
    Returns dict with "uniform", "is_weighted", "reward_weighted", cross-variant cosines,
    "beta_used", and "n_transitions".
    """
    online_net = online_net.to(device)
    online_net.eval()   # eval mode: NoisyLinear uses weight_mu only (deterministic)

    use_target = target_net is not None and args_ns is not None
    beta_used = None
    N_total = sum(len(ep.rewards) for ep in episodes)

    if use_target:
        target_net = target_net.to(device).eval()
        atoms    = args_ns.atoms
        Vmin     = args_ns.V_min
        Vmax     = args_ns.V_max
        delta_z  = (Vmax - Vmin) / (atoms - 1)
        support  = torch.linspace(Vmin, Vmax, atoms).to(device)
        n        = args_ns.multi_step
        gamma    = args_ns.discount

        # Annealed beta at checkpoint step
        alpha       = getattr(args_ns, 'priority_exponent', 0.5)
        beta_start  = getattr(args_ns, 'priority_weight',   0.4)
        T_max       = getattr(args_ns, 'T_max',             int(10e6))
        learn_start = getattr(args_ns, 'learn_start',       int(20e3))
        anneal_frac = max(0.0, min(1.0, (global_step - learn_start) / max(T_max - learn_start, 1)))
        beta_used   = beta_start + (1.0 - beta_start) * anneal_frac

        if precomputed_is_weights is not None:
            # pre-computed over joint pool - skip Pass 1
            is_weights_per_ep = precomputed_is_weights
        else:
            # Pass 1: collect per-transition losses for within-group IS normalisation
            episode_losses_detached = []  # list of (T,) detached tensors
            with torch.no_grad():
                for episode in episodes:
                    if len(episode.rewards) > 1:
                        per_loss = _forward_per_loss(
                            episode, online_net, target_net,
                            support, Vmin, Vmax, delta_z, atoms, gamma, n, device
                        )
                        episode_losses_detached.append(per_loss.detach())
                    else:
                        episode_losses_detached.append(None)

            valid_losses = [l for l in episode_losses_detached if l is not None]
            if valid_losses:
                all_losses = torch.cat(valid_losses)                    # (N_total_valid,)
                all_is_weights = _compute_is_weights(all_losses, alpha, beta_used)
                # Split back to per-episode - rebuild index
                is_weights_per_ep = []
                ptr = 0
                for l in episode_losses_detached:
                    if l is not None:
                        T = len(l)
                        is_weights_per_ep.append(all_is_weights[ptr:ptr + T])
                        ptr += T
                    else:
                        is_weights_per_ep.append(None)
            else:
                is_weights_per_ep = [None] * len(episodes)

    named_params = list(online_net.named_parameters())
    overall_uniform_agg = OnlineGradientAggregator(named_params)
    overall_is_agg      = OnlineGradientAggregator(named_params) if use_target else None
    overall_rw_agg      = OnlineGradientAggregator(named_params) if use_target else None

    batch_grads_uniform = []
    batch_grads_is      = []
    batch_grads_rw      = []

    # Pass 2: backward passes
    batches = range(0, len(episodes), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch"):
        batch_eps = episodes[i: i + batch_size]
        batch_is_w = is_weights_per_ep[i: i + batch_size] if use_target else [None] * len(batch_eps)

        batch_uniform_agg = OnlineGradientAggregator(named_params)
        batch_is_agg      = OnlineGradientAggregator(named_params) if use_target else None
        batch_rw_agg      = OnlineGradientAggregator(named_params) if use_target else None

        for episode, is_w_ep in zip(batch_eps, batch_is_w):
            T = len(episode.rewards)

            if use_target and T > 1 and is_w_ep is not None:
                # Backward 1 - uniform
                online_net.zero_grad()
                per_loss = _forward_per_loss(
                    episode, online_net, target_net,
                    support, Vmin, Vmax, delta_z, atoms, gamma, n, device
                )  # (T,)
                per_loss.mean().backward(retain_graph=True)
                batch_uniform_agg.accumulate(named_params)
                overall_uniform_agg.accumulate(named_params)

                # Backward 2 - IS-weighted
                online_net.zero_grad()
                (is_w_ep.to(device) * per_loss).mean().backward(retain_graph=True)
                batch_is_agg.accumulate(named_params)
                overall_is_agg.accumulate(named_params)

                # Backward 3 - reward-weighted (no retain_graph - free the graph)
                online_net.zero_grad()
                rw_w = _compute_reward_weights(episode.rewards, device)
                (rw_w * per_loss).mean().backward()
                batch_rw_agg.accumulate(named_params)
                overall_rw_agg.accumulate(named_params)

                online_net.zero_grad()

            else:
                # Fallback: log-prob of taken actions (proxy, no target net)
                online_net.zero_grad()
                obs     = episode.observations.to(device)
                actions = episode.actions.to(device)
                log_ps   = online_net(obs, log=True)
                log_ps_a = log_ps[range(T), actions]  # (T, atoms)
                loss     = -log_ps_a.sum(dim=-1).mean()
                loss.backward()
                batch_uniform_agg.accumulate(named_params)
                overall_uniform_agg.accumulate(named_params)
                online_net.zero_grad()

        batch_grads_uniform.append(batch_uniform_agg.l2_normalized())
        if use_target and batch_is_agg.count > 0:
            batch_grads_is.append(batch_is_agg.l2_normalized())
            batch_grads_rw.append(batch_rw_agg.l2_normalized())

    raw_uniform = overall_uniform_agg.mean_gradient()
    raw_is      = overall_is_agg.mean_gradient()  if (use_target and overall_is_agg.count > 0)  else None
    raw_rw      = overall_rw_agg.mean_gradient()  if (use_target and overall_rw_agg.count > 0)  else None

    cos_uniform_is = cosine_similarity_flat(raw_uniform, raw_is) if raw_is is not None else None
    cos_is_reward  = cosine_similarity_flat(raw_is, raw_rw)      if (raw_is is not None and raw_rw is not None) else None

    return {
        "uniform": {
            "raw":        raw_uniform,
            "batch_grads": batch_grads_uniform,
        },
        "is_weighted": {
            "raw":        raw_is,
            "batch_grads": batch_grads_is,
        },
        "reward_weighted": {
            "raw":        raw_rw,
            "batch_grads": batch_grads_rw,
        },
        "cos_uniform_is":  cos_uniform_is,
        "cos_is_reward":   cos_is_reward,
        "beta_used":       beta_used,
        "n_transitions":   N_total,
    }
