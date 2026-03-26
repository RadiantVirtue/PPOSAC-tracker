"""Rainbow gradient analysis (Crafter).

Computes the full offline distributional Bellman loss gradient:

    ∇θ E[ -Σ_i m_i log p(s_t, a_t; θ)_i ]

where m is the projected target distribution (Tz → fixed support) built from
the stored episode trajectory and the frozen target network.  This matches the
training gradient of Agent.learn() exactly (no PER weights applied — uniform).

Key differences from sac/gradients.py:
  - Uses DQN online_net + target_net instead of DiscreteActor + critics
  - Tz projection logic adapted from Rainbow/agent.py::learn()
  - next_obs derived inline from episode trajectory: next_obs[t] = obs[t+1]
  - n-step returns computed from episode rewards tensor
  - No alpha / entropy temperature

Reference: Gradient Analysis v3.md, Section 3 Step 2.
"""
import torch
from tqdm import tqdm

from shared.gradient_utils import OnlineGradientAggregator


def _compute_nstep_returns(rewards, dones, gamma, n):
    """Compute n-step discounted returns for each timestep in an episode.

    Args:
        rewards: (T,) float32 tensor
        dones:   (T,) float32 tensor  (1.0 at terminal steps)
        gamma:   discount factor
        n:       number of steps

    Returns:
        R:          (T,) float32 — n-step return for each step
        nonterminal:(T,) float32 — 1 if the nth-next state is non-terminal, else 0
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
        next_obs:    (T, C, H, W) float32 — next observations
        online_net:  DQN (for double-Q action selection)
        target_net:  DQN (for target distribution)
        R:           (T,) float32 — n-step returns
        nonterminal: (T,) float32 — 1 if non-terminal
        support:     (atoms,) float32 — value support z
        Vmin, Vmax, delta_z, atoms: distribution parameters
        gamma:       discount factor
        n:           n-step
        device:      torch device

    Returns:
        m: (T, atoms) float32 — target distribution
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


def compute_group_gradient_with_coherence(
    online_net, episodes, batch_size: int = 10, device: str = "cpu",
    desc: str = "Grads",
    target_net=None, args_ns=None,
):
    """Compute Rainbow distributional Bellman gradients over a group of episodes.

    Args:
        online_net:  DQN online network (frozen weights for analysis).
        episodes:    list of EpisodeData with .observations (T,3,H,W) float32,
                     .actions (T,) long, .rewards (T,) float32, .dones (T,) float32.
        batch_size:  episodes per coherence mini-batch.
        device:      torch device string.
        desc:        tqdm label.
        target_net:  DQN target network (frozen). Required for full distributional loss.
                     Falls back to log-prob proxy if None.
        args_ns:     argparse.Namespace with atoms, V_min, V_max, multi_step, discount.

    Returns:
        (l2_normalised_mean, raw_mean, list_of_batch_gradient_dicts)
    """
    online_net = online_net.to(device)
    online_net.eval()   # eval mode: NoisyLinear uses weight_mu only (deterministic, no noise sampling)

    use_target = target_net is not None and args_ns is not None
    if use_target:
        target_net = target_net.to(device).eval()
        atoms    = args_ns.atoms
        Vmin     = args_ns.V_min
        Vmax     = args_ns.V_max
        delta_z  = (Vmax - Vmin) / (atoms - 1)
        support  = torch.linspace(Vmin, Vmax, atoms).to(device)
        n        = args_ns.multi_step
        gamma    = args_ns.discount

    overall_agg = OnlineGradientAggregator(list(online_net.named_parameters()))
    batch_grads = []

    batches = range(0, len(episodes), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch"):
        batch = episodes[i: i + batch_size]
        batch_agg = OnlineGradientAggregator(list(online_net.named_parameters()))

        for episode in batch:
            online_net.zero_grad()

            obs     = episode.observations.to(device)   # (T, 3, H, W) float32
            actions = episode.actions.to(device)        # (T,) long
            T       = len(obs)

            if use_target and T > 1:
                # ── Full offline distributional Bellman loss ──────────────────
                # next_obs[t] = obs[t+1]; terminal step uses zero frame
                next_obs = torch.zeros_like(obs)
                next_obs[:-1] = obs[1:]  # (T, 3, H, W); last row stays zero (terminal)

                R, nonterminal = _compute_nstep_returns(
                    episode.rewards, episode.dones, gamma, n
                )

                m = _project_distribution(
                    next_obs, online_net, target_net,
                    R, nonterminal,
                    support, Vmin, Vmax, delta_z, atoms, gamma, n, device
                )  # (T, atoms)

                log_ps   = online_net(obs, log=True)                # (T, action_space, atoms)
                log_ps_a = log_ps[range(T), actions]               # (T, atoms)
                loss     = -torch.sum(m * log_ps_a, dim=1).mean()  # scalar
            else:
                # ── Fallback: log-prob of taken actions (proxy) ──────────────
                log_ps   = online_net(obs, log=True)
                log_ps_a = log_ps[range(T), actions]  # (T, atoms)
                loss     = -log_ps_a.sum(dim=-1).mean()

            loss.backward()

            batch_agg.accumulate(list(online_net.named_parameters()))
            overall_agg.accumulate(list(online_net.named_parameters()))
            online_net.zero_grad()

        batch_grads.append(batch_agg.l2_normalized())

    return overall_agg.l2_normalized(), overall_agg.mean_gradient(), batch_grads
