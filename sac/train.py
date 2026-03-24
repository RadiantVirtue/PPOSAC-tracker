"""Discrete SAC training on Crafter.

Architecture:
  - Custom training loop with hand-rolled Discrete SAC update equations.
  - num_envs parallel Crafter environments (mirrors PPO's num_procs).
  - Observations stored as (C, H, W) float32 [0,1] tensors.
  - EpisodeStore saved alongside each checkpoint for offline analysis.
  - Double Q-network with soft target updates.

Usage:
    from sac.train import Args, main_sac
    args = Args(seed=1, total_timesteps=10_000_000)
    episodes, steps = main_sac(args)
"""
import copy
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from sac.network import DiscreteActor, DiscreteCritic
from sac.replay_buffer import ReplayBuffer
from shared.tagged_buffer import EpisodeStore
from wrappers import make_crafter_env


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 10_000_000
    num_envs: int = 16

    # Replay buffer
    buffer_capacity: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 1_000

    # Optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005          # soft target update coefficient
    grad_clip: float = 10.0

    # Entropy
    target_entropy: float = -2.833   # ≈ -log(1/17) * 0.98; auto-tune target
    fixed_alpha: bool = False
    alpha_init: float = 0.2

    # Checkpointing
    checkpoint_freq: int = 0         # episodes between checkpoints; 0 = off
    checkpoint_step_freq: int = 5_000  # steps between checkpoints; 0 = off
    checkpoint_achievements: bool = True
    max_checkpoints: int = 5         # keep N most recent checkpoints; 0 = keep all
    experiment_root: str = "experiment_root"

    # Set at runtime
    action_dim: int = field(default=17, repr=False)


# ── Public entrypoint ─────────────────────────────────────────────────────────

def main_sac(args: Args, on_checkpoint_saved=None, should_stop=None):
    """Train Discrete SAC on Crafter.

    Args:
        args:                training configuration.
        on_checkpoint_saved: optional callback(path) after each checkpoint.
        should_stop:         optional callback(info_dict) → bool; return True
                             to stop training.

    Returns:
        (episode_count, global_step)
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    args.action_dim = 17  # Crafter always has 17 actions

    # ── Networks ──────────────────────────────────────────────────────────────
    actor  = DiscreteActor(args.action_dim).to(device)
    q1_net = DiscreteCritic(args.action_dim).to(device)
    q2_net = DiscreteCritic(args.action_dim).to(device)
    q1_target = copy.deepcopy(q1_net)
    q2_target = copy.deepcopy(q2_net)
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    # Learnable log(alpha) for automatic entropy tuning
    log_alpha = torch.tensor(
        np.log(args.alpha_init), dtype=torch.float32, requires_grad=not args.fixed_alpha
    )

    # ── Optimisers ────────────────────────────────────────────────────────────
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=args.lr_actor)
    critic_opt = torch.optim.Adam(
        list(q1_net.parameters()) + list(q2_net.parameters()), lr=args.lr_critic
    )
    alpha_opt = (
        torch.optim.Adam([log_alpha], lr=args.lr_alpha)
        if not args.fixed_alpha else None
    )

    # ── Replay buffer & episode store ─────────────────────────────────────────
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    episode_store = EpisodeStore(max_size=args.buffer_capacity)

    # ── Environments ──────────────────────────────────────────────────────────
    envs = [make_crafter_env() for _ in range(args.num_envs)]
    env_obs = [_reset_obs(env, args.seed + i * 10_000) for i, env in enumerate(envs)]
    ep_transitions = [[] for _ in range(args.num_envs)]

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.experiment_root, "checkpoints", "sac")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Return logging ────────────────────────────────────────────────────────
    logs_dir = os.path.join(args.experiment_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    returns_path = os.path.join(logs_dir, "sacreturnlog.txt")
    ep_return_running = np.zeros(args.num_envs)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step     = 0
    episode_count   = 0
    last_ckpt_step  = 0
    last_ckpt_ep    = 0
    stop_requested  = False

    while global_step < args.total_timesteps and not stop_requested:
        for i, env in enumerate(envs):
            obs = env_obs[i]          # (H, W, C) uint8 numpy

            # Choose action
            with torch.no_grad():
                obs_t = _obs_to_tensor(obs).unsqueeze(0).to(device)  # (1,3,64,64)
                logits = actor(obs_t)
                action = torch.distributions.Categorical(logits=logits).sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            global_step += 1
            ep_return_running[i] += reward

            # Store raw transition for EpisodeStore
            ep_transitions[i].append({
                "state":      _obs_to_tensor(obs).numpy(),       # (3,64,64) float32
                "action":     int(action),
                "reward":     float(reward),
                "next_state": _obs_to_tensor(next_obs).numpy(),  # (3,64,64) float32
                "terminal":   bool(terminated),
            })

            # Store in replay buffer
            replay_buffer.add(
                obs=_obs_to_tensor(obs),
                action=torch.tensor(action, dtype=torch.long),
                reward=torch.tensor(reward, dtype=torch.float32),
                next_obs=_obs_to_tensor(next_obs),
                terminated=torch.tensor(terminated, dtype=torch.bool),
                done=torch.tensor(done, dtype=torch.bool),
            )

            if done:
                episode_count += 1
                eps_score    = float(info.get("eps", 0.0))
                achievements = info.get("achievements", {})

                with open(returns_path, "a") as f:
                    f.write(f"{ep_return_running[i]:.6f}\n")
                ep_return_running[i] = 0.0

                episode_store.add_episode(ep_transitions[i], eps_score)
                ep_transitions[i] = []

                if episode_count % 10 == 0:
                    alpha_val = log_alpha.exp().item()
                    print(
                        f"  [sac] ep={episode_count:6d}  "
                        f"steps={global_step:9d}  eps={eps_score:.3f}  "
                        f"alpha={alpha_val:.4f}"
                    )

                if should_stop is not None and should_stop(
                    {"achievements": achievements, "eps": eps_score}
                ):
                    print(f"  [sac] Convergence criterion met at episode {episode_count}.")
                    stop_requested = True
                    break

                # Episode-based checkpoint
                if (
                    args.checkpoint_freq > 0
                    and (episode_count - last_ckpt_ep) >= args.checkpoint_freq
                ):
                    ckpt_path = _save_checkpoint(
                        actor, q1_net, q2_net, log_alpha,
                        actor_opt, critic_opt,
                        episode_store, global_step, episode_count, args, ckpt_dir,
                    )
                    _prune_checkpoints(ckpt_dir, args.max_checkpoints)
                    last_ckpt_ep   = episode_count
                    last_ckpt_step = global_step
                    if on_checkpoint_saved:
                        on_checkpoint_saved(ckpt_path)

                env_obs[i] = _reset_obs(env, None)
            else:
                env_obs[i] = next_obs

        # ── Learning step ─────────────────────────────────────────────────────
        if replay_buffer.size >= args.batch_size and global_step >= args.warmup_steps:
            batch = replay_buffer.sample(args.batch_size)
            _learn(
                actor, q1_net, q2_net, q1_target, q2_target,
                log_alpha, actor_opt, critic_opt, alpha_opt,
                batch, args, device,
            )
            _soft_update(q1_target, q1_net, args.tau)
            _soft_update(q2_target, q2_net, args.tau)

        # Step-based checkpoint
        if (
            args.checkpoint_step_freq > 0
            and (global_step - last_ckpt_step) >= args.checkpoint_step_freq
        ):
            ckpt_path = _save_checkpoint(
                actor, q1_net, q2_net, log_alpha,
                actor_opt, critic_opt,
                episode_store, global_step, episode_count, args, ckpt_dir,
            )
            _prune_checkpoints(ckpt_dir, args.max_checkpoints)
            last_ckpt_step = global_step
            if on_checkpoint_saved:
                on_checkpoint_saved(ckpt_path)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if global_step > last_ckpt_step:
        ckpt_path = _save_checkpoint(
            actor, q1_net, q2_net, log_alpha,
            actor_opt, critic_opt,
            episode_store, global_step, episode_count, args, ckpt_dir,
        )
        _prune_checkpoints(ckpt_dir, args.max_checkpoints)
        if on_checkpoint_saved:
            on_checkpoint_saved(ckpt_path)

    for env in envs:
        env.close()
    return episode_count, global_step


# ── SAC update ────────────────────────────────────────────────────────────────

def _learn(actor, q1, q2, q1_tgt, q2_tgt, log_alpha,
           actor_opt, critic_opt, alpha_opt, batch, args, device):
    """One Discrete SAC update: critic → actor → alpha."""
    td = batch  # TensorDict from ReplayBuffer

    obs      = td["observation"].to(device)                    # (B,3,64,64)
    actions  = td["action"].to(device)                         # (B,)
    rewards  = td["next"]["reward"].squeeze(-1).to(device)     # (B,)
    next_obs = td["next"]["observation"].to(device)            # (B,3,64,64)
    terminated = td["next"]["terminated"].squeeze(-1).float().to(device)  # (B,)

    alpha = log_alpha.exp().detach()

    # ── Critic loss ────────────────────────────────────────────────────────
    with torch.no_grad():
        next_logits    = actor(next_obs)                        # (B,A)
        next_probs     = F.softmax(next_logits, dim=-1)         # (B,A)
        next_log_probs = F.log_softmax(next_logits, dim=-1)     # (B,A)

        next_q1 = q1_tgt(next_obs)                             # (B,A)
        next_q2 = q2_tgt(next_obs)                             # (B,A)
        next_q  = torch.min(next_q1, next_q2)                  # (B,A)

        # Discrete SAC V(s') = Σ_a π(a|s') [Q(s',a) − α log π(a|s')]
        next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)  # (B,)
        target_q = rewards + args.gamma * (1.0 - terminated) * next_v         # (B,)

    q1_pred = q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
    q2_pred = q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
    critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(q1.parameters()) + list(q2.parameters()), args.grad_clip
    )
    critic_opt.step()

    # ── Actor loss ─────────────────────────────────────────────────────────
    logits    = actor(obs)
    probs     = F.softmax(logits, dim=-1)       # (B,A)
    log_probs = F.log_softmax(logits, dim=-1)   # (B,A)

    with torch.no_grad():
        q_val = torch.min(q1(obs), q2(obs))     # (B,A)

    # Σ_a π(a|s) [α log π(a|s) − Q(s,a)]
    actor_loss = (probs * (alpha * log_probs - q_val)).sum(dim=-1).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
    actor_opt.step()

    # ── Alpha (entropy) loss ───────────────────────────────────────────────
    if alpha_opt is not None:
        with torch.no_grad():
            logits_    = actor(obs)
            probs_     = F.softmax(logits_, dim=-1)
            log_probs_ = F.log_softmax(logits_, dim=-1)
            entropy    = -(probs_ * log_probs_).sum(dim=-1)   # (B,)

        alpha_loss = -(log_alpha * (entropy - args.target_entropy).detach()).mean()

        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_to_tensor(obs) -> torch.Tensor:
    """(H, W, C) uint8 numpy → (C, H, W) float32 [0,1] tensor."""
    arr = np.array(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _reset_obs(env, seed):
    obs, _ = env.reset(seed=seed)
    return obs


def _prune_checkpoints(ckpt_dir: str, max_checkpoints: int) -> None:
    """Delete oldest checkpoints (and their _episodes.pkl siblings) so that at
    most max_checkpoints .pt files remain in ckpt_dir."""
    if max_checkpoints <= 0:
        return
    pts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_step")[-1].replace(".pt", "")),
    )
    for old in pts[:-max_checkpoints]:
        old_path = os.path.join(ckpt_dir, old)
        os.remove(old_path)
        pkl_path = old_path.replace(".pt", "_episodes.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        print(f"  [sac] Pruned old checkpoint: {old_path}")


def _save_checkpoint(actor, q1_net, q2_net, log_alpha,
                     actor_opt, critic_opt,
                     episode_store, global_step, episode_count, args, ckpt_dir):
    filename = f"sac_ep{episode_count:07d}_step{global_step:09d}.pt"
    ckpt_path = os.path.join(ckpt_dir, filename)

    torch.save({
        "global_step":                  global_step,
        "episode_count":                episode_count,
        "actor_state_dict":             actor.state_dict(),
        "critic1_state_dict":           q1_net.state_dict(),
        "critic2_state_dict":           q2_net.state_dict(),
        "actor_optimizer_state_dict":   actor_opt.state_dict(),
        "critic_optimizer_state_dict":  critic_opt.state_dict(),
        "log_alpha":                    log_alpha.item(),
        "obs_shape":                    (64, 64, 3),
        "action_dim":                   args.action_dim,
    }, ckpt_path)

    eps_path = ckpt_path.replace(".pt", "_episodes.pkl")
    episode_store.save(eps_path)

    print(f"  [sac] Checkpoint saved: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main_sac(args)
