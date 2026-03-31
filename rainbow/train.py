# -*- coding: utf-8 -*-
"""Rainbow DQN training on Crafter.

Adapted from Rainbow/main.py. Key differences from the original:
  - Uses make_crafter_env() (Crafter gymnasium wrapper) instead of Atari env.py
  - Observations: RGB (64,64,3) uint8 → (3,64,64) float32 [0,1]  (no temporal stacking)
  - history_length=3 (3 RGB input channels; memory uses temporal depth=1 — see memory.py)
  - conv_output_size=1024 (64x64 canonical, see rainbow/model.py)
  - Rich checkpoint format (online + target net, args_dict) for offline analysis
  - Two checkpoint slots:
      checkpoint_live.pt  — always the latest weights; overwritten in-place; used for resume
      checkpoint_step{N}.pt — written every checkpoint_interval steps, passed to
                              on_checkpoint_saved callback, then deleted after analysis
  - episode_count tracking for comparison with PPO
  - Returns log is written to <experiment_root>/logs/rainbowreturnlog.txt
"""
from __future__ import division

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.agent import Agent
from rainbow.memory import ReplayMemory
from wrappers import make_crafter_env


# ── Obs preprocessing ─────────────────────────────────────────────────────────

def _obs_to_state(obs, device):
    """Convert Crafter obs (H,W,3) uint8 → (3,H,W) float32 [0,1] on device."""
    import numpy as np
    arr = np.array(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)  # (3, H, W)


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(description='Rainbow on Crafter')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(10e6), metavar='STEPS', help='Number of training steps')
    parser.add_argument('--history-length', type=int, default=3, metavar='T', help='Model input channels (3 = RGB; memory temporal depth is always 1)')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'])
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ')
    parser.add_argument('--atoms', type=int, default=51, metavar='C')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V')
    parser.add_argument('--V-max', type=float, default=10, metavar='V')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Checkpoint path to resume from')
    parser.add_argument('--memory-capacity', type=int, default=int(500e3), metavar='CAPACITY')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM')
    parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS')
    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N')
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N')
    parser.add_argument('--enable-cudnn', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=100000, help='Steps between analysis checkpoints (0 = off)')
    parser.add_argument('--experiment-root', type=str, default='experiment_root')
    return parser


# ── Env wrapper that matches Rainbow's (state, reward, done) interface ─────────

class CrafterEnvWrapper:
    """Adapts make_crafter_env() to Rainbow's Env interface.

    - action_space() returns int
    - reset() returns (1,H,W) float32 tensor
    - step(action) returns ((1,H,W) float32 tensor, reward, done)
    - train() / eval() control life-termination behaviour (noop here; Crafter has fixed episode length)
    """

    def __init__(self, args, seed=None):
        self.device = args.device
        self._env = make_crafter_env(seed=seed)
        self._info = {}

    def action_space(self):
        return self._env.action_space.n

    def reset(self):
        obs, info = self._env.reset()
        self._info = info
        return _obs_to_state(obs, self.device)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._info = info
        done = terminated or truncated
        return _obs_to_state(obs, self.device), reward, done

    def train(self):
        pass  # Crafter has no life-termination toggle

    def eval(self):
        pass

    def close(self):
        self._env.close()

    @property
    def info(self):
        return self._info


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt_dir(experiment_root):
    d = os.path.join(experiment_root, "checkpoints", "rainbow")
    os.makedirs(d, exist_ok=True)
    return d


def _save_live(dqn, experiment_root, args, global_step, episode_count):
    """Overwrite checkpoint_live.pt in-place (for training resume)."""
    dqn.save(_ckpt_dir(experiment_root), "checkpoint_live.pt",
             global_step=global_step, episode_count=episode_count, args=args)


def _save_analysis(dqn, experiment_root, args, global_step, episode_count):
    """Save a named checkpoint for analysis; return its path."""
    name = f"checkpoint_step{global_step}.pt"
    dqn.save(_ckpt_dir(experiment_root), name,
             global_step=global_step, episode_count=episode_count, args=args)
    return os.path.join(_ckpt_dir(experiment_root), name)


def _save_milestone(dqn, experiment_root, args, global_step, episode_count, achievement):
    """Save a named milestone checkpoint; return its path."""
    name = f"milestone_first_{achievement}_ep{episode_count}.pt"
    dqn.save(_ckpt_dir(experiment_root), name,
             global_step=global_step, episode_count=episode_count, args=args)
    return os.path.join(_ckpt_dir(experiment_root), name)


def _delete_ckpt(path):
    try:
        os.remove(path)
        print(f"  [ckpt] Deleted {os.path.basename(path)}")
    except OSError:
        pass


# ── Return log ─────────────────────────────────────────────────────────────────

def _log_return(experiment_root, ep_return):
    log_dir = os.path.join(experiment_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "rainbowreturnlog.txt"), "a") as f:
        f.write(f"{ep_return}\n")


# ── Main training function ─────────────────────────────────────────────────────

def main_rainbow(args, on_checkpoint_saved=None):
    """Train Rainbow DQN on Crafter.

    Args:
        args:                argparse.Namespace with all training hyperparameters.
        on_checkpoint_saved: optional callable(path, prev_path=None, is_milestone=False)
                             called when any checkpoint is saved.
                             - path:         path to the saved checkpoint
                             - prev_path:    path to the previous periodic checkpoint to use
                                            for weight-delta (Δθ) computation; None means
                                            skip weight-delta (early training or near-zero Δθ)
                             - is_milestone: True for achievement-triggered checkpoints,
                                            False for periodic step-interval checkpoints
                             Deletion is handled by the training loop, not the callback.

    Returns:
        (episode_count, saved_paths) — total episodes and list of analysis checkpoint paths.
    """
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = getattr(args, 'enable_cudnn', False)
    else:
        args.device = torch.device('cpu')

    env = CrafterEnvWrapper(args, seed=args.seed)
    action_space = env.action_space()

    dqn = Agent(args, env)

    # Optionally resume from a live checkpoint
    if getattr(args, 'model', None) and os.path.isfile(args.model):
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        dqn.online_net.load_state_dict(ckpt['online_net_state_dict'])
        dqn.target_net.load_state_dict(ckpt['target_net_state_dict'])
        dqn.optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        resume_step = ckpt.get('global_step', 0)
        resume_eps = ckpt.get('episode_count', 0)
        print(f"Resumed from {args.model} (step={resume_step}, eps={resume_eps})")
    else:
        resume_step = 0
        resume_eps = 0

    mem = ReplayMemory(args, args.memory_capacity)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state = env.reset()
        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, -1, 0.0, done)
        state = next_state
        T += 1

    # Training loop
    dqn.train()
    done = True
    episode_count = resume_eps
    ep_return = 0.0
    saved_paths = []

    # Two-pointer state for weight-delta preservation across milestone checkpoints
    seen_achievements = set()
    periodic_A = None        # penultimate periodic checkpoint path
    periodic_B = None        # last periodic checkpoint path
    last_periodic_step = 0   # training step at which periodic_B was saved

    for T in trange(1 + resume_step, args.T_max + 1):
        if done:
            state = env.reset()
            ep_return = 0.0

        if T % args.replay_frequency == 0:
            dqn.reset_noise()

        action = dqn.act(state)
        next_state, reward, done = env.step(action)
        ep_return += reward

        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)
        mem.append(state, action, reward, done)

        if done:
            episode_count += 1
            _log_return(args.experiment_root, ep_return)

            # Achievement milestone checkpointing
            cur_ach = {k: bool(v) for k, v in env.info.get("achievements", {}).items()}
            new_ach = {a for a, v in cur_ach.items() if v} - seen_achievements
            if new_ach and on_checkpoint_saved is not None:
                seen_achievements.update(new_ach)
                for ach in sorted(new_ach):
                    skip_delta = (periodic_B is None) or (T - last_periodic_step < 10_000)
                    m_path = _save_milestone(dqn, args.experiment_root, args, T,
                                             episode_count, ach)
                    on_checkpoint_saved(
                        m_path,
                        prev_path=None if skip_delta else periodic_B,
                        is_milestone=True,
                    )
                    _delete_ckpt(m_path)

        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                dqn.learn(mem)

            if T % args.target_update == 0:
                dqn.update_target_net()

            # Periodic checkpoint: save live + analysis, rotate two-pointer state
            if args.checkpoint_interval > 0 and T % args.checkpoint_interval == 0:
                _save_live(dqn, args.experiment_root, args, T, episode_count)

                ckpt_path = _save_analysis(dqn, args.experiment_root, args, T, episode_count)
                saved_paths.append(ckpt_path)

                # Rotate pointers: A ← B, B ← new; delete old A
                old_A = periodic_A
                periodic_A = periodic_B
                periodic_B = ckpt_path
                last_periodic_step = T

                if old_A and old_A != periodic_A and old_A != periodic_B:
                    _delete_ckpt(old_A)

                if on_checkpoint_saved is not None:
                    on_checkpoint_saved(ckpt_path, prev_path=periodic_A, is_milestone=False)

        state = next_state

    env.close()

    # Clean up the two held periodic checkpoints (A and B) that outlived training.
    # These were kept alive for weight-delta computation but no further checkpoints will arrive.
    for held in (periodic_A, periodic_B):
        if held and os.path.exists(held):
            _delete_ckpt(held)

    return episode_count, saved_paths


def main():
    parser = build_parser()
    args = parser.parse_args()
    main_rainbow(args)


if __name__ == "__main__":
    main()
