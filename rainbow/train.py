"""Rainbow DQN training on Crafter. Adapted from Rainbow/main.py."""

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


def _obs_to_state(obs, device):
    """Convert Crafter obs (H,W,3) uint8 → (3,H,W) float32 [0,1] on device."""
    import numpy as np
    arr = np.array(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)  # (3, H, W)



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


class CrafterEnvWrapper:
    """Adapts make_crafter_env() to Rainbow's (state, reward, done) interface."""

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


def _ckpt_dir(experiment_root):
    d = os.path.join(experiment_root, "checkpoints", "rainbow")
    os.makedirs(d, exist_ok=True)
    return d


def _buffer_live_path(experiment_root):
    return os.path.join(_ckpt_dir(experiment_root), "replay_buffer_live.npz")


def _save_live(dqn, experiment_root, args, global_step, episode_count, mem=None):
    """Overwrite checkpoint_live.pt (and optionally replay_buffer_live.npz) in-place."""
    dqn.save(_ckpt_dir(experiment_root), "checkpoint_live.pt",
             global_step=global_step, episode_count=episode_count, args=args)
    if mem is not None:
        buf_path = _buffer_live_path(experiment_root)
        mem.save_buffer(buf_path)
        print(f"  [buffer] Saved replay buffer → {os.path.basename(buf_path)}")


def _save_analysis(dqn, experiment_root, args, global_step, episode_count):
    """Save a named checkpoint for analysis; return its path.

    Consumes any accumulated training gradient from the agent's capture buffer
    and stores it in the checkpoint under 'training_gradient'.
    """
    name = f"checkpoint_step{global_step}.pt"
    training_gradient = dqn.consume_grad_capture()   # None if capture not enabled
    dqn.save(_ckpt_dir(experiment_root), name,
             global_step=global_step, episode_count=episode_count, args=args,
             training_gradient=training_gradient)
    if training_gradient is not None:
        print(f"  [grad capture] Stored training gradient ({len(training_gradient)} param tensors)")
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


def _log_return(experiment_root, ep_return, step=None):
    log_dir = os.path.join(experiment_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "rainbowreturnlog.txt"), "a") as f:
        if step is not None:
            f.write(f"{step},{ep_return}\n")
        else:
            f.write(f"{ep_return}\n")


def main_rainbow(args, on_checkpoint_saved=None, seen_achievements=None, mora_tracker=None,
                 outcome_tracker=None, capture_training_grads: bool = False):
    """Train Rainbow DQN on Crafter. Returns (episode_count, saved_paths)."""
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

    # Experiment 3: enable training gradient capture if requested.
    if capture_training_grads:
        dqn.enable_grad_capture()
        print("  [grad capture] Training gradient capture ENABLED.")

    # Optionally resume from a live checkpoint
    if getattr(args, 'model', None) and os.path.isfile(args.model):
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        # Bug 2: online_net already loaded by Agent.__init__ (rich checkpoint path);
        # only reload target_net + optimiser here to avoid a redundant double-copy.
        dqn.target_net.load_state_dict(ckpt['target_net_state_dict'])
        dqn.optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        resume_step = ckpt.get('global_step', 0)
        resume_eps = ckpt.get('episode_count', 0)
        print(f"Resumed from {args.model} (step={resume_step}, eps={resume_eps})")
    else:
        resume_step = 0
        resume_eps = 0

    mem = ReplayMemory(args, args.memory_capacity)

    # Reload saved buffer if available — avoids the learn_start dead zone on resume.
    buf_path = _buffer_live_path(args.experiment_root)
    if resume_step > 0 and os.path.exists(buf_path):
        print(f"  [buffer] Loading replay buffer from {os.path.basename(buf_path)} ...")
        mem.load_buffer(buf_path)
        print(f"  [buffer] Loaded — {mem.transitions.index} transitions, full={mem.transitions.full}")

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    # Bug 3: restore β to its correct annealed value at the resume step so that
    # the IS schedule continues smoothly rather than restarting from β_start=0.4.
    if resume_step > 0:
        anneal_frac = max(0.0, min(1.0,
            (resume_step - args.learn_start) / max(args.T_max - args.learn_start, 1)
        ))
        mem.priority_weight = args.priority_weight + (1 - args.priority_weight) * anneal_frac

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
    seen_achievements = set(seen_achievements) if seen_achievements else set()
    periodic_A = None        # penultimate periodic checkpoint path
    periodic_B = None        # last periodic checkpoint path
    last_periodic_step = 0   # training step at which periodic_B was saved

    _POSTFIX_INTERVAL = 500   # update tqdm postfix every N steps
    episode_bufpos: list[int] = []  # buffer positions for current episode (outcome weighting)
    pbar = trange(1 + resume_step, args.T_max + 1)
    for T in pbar:
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

        # Outcome weighting: record just-written buffer position for this episode
        if outcome_tracker is not None:
            episode_bufpos.append(mem.current_index())

        buf_size = mem.capacity if mem.transitions.full else mem.transitions.index
        if T % _POSTFIX_INTERVAL == 0:
            buf_pct  = 100.0 * buf_size / mem.capacity
            pbar.set_postfix(
                buf=f"{buf_size:,}/{mem.capacity:,} ({buf_pct:.1f}%)",
                learning=buf_size >= args.learn_start,
                ep=episode_count,
            )

        if done:
            episode_count += 1
            _log_return(args.experiment_root, ep_return, step=T)

            # MORA: update rolling gradient-coherence modifier and log
            if mora_tracker is not None:
                mora_tracker.episode_end(ep_return)
                mora_tracker.log_m(
                    os.path.join(args.experiment_root, "logs", "mora_modifier_log.csv"),
                    episode_count,
                )

            # Outcome weighting: classify episode, label buffer transitions, clear list
            if outcome_tracker is not None and episode_bufpos:
                label = outcome_tracker.classify(ep_return)
                if label != 0:
                    mem.set_outcome_label(np.array(episode_bufpos, dtype=np.int64), label)
                episode_bufpos.clear()

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

        if T >= args.learn_start and buf_size >= max(args.learn_start, args.batch_size * 10):
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                dqn.learn(mem, mora_tracker=mora_tracker)

            if T % args.target_update == 0:
                dqn.update_target_net()

            # Periodic checkpoint: save live + analysis, rotate two-pointer state
            if args.checkpoint_interval > 0 and T % args.checkpoint_interval == 0:
                _save_live(dqn, args.experiment_root, args, T, episode_count, mem=mem)

                ckpt_path = _save_analysis(dqn, args.experiment_root, args, T, episode_count)
                saved_paths.append(ckpt_path)

                # Rotate pointers: A ← B, B ← new; delete old A
                old_A = periodic_A
                periodic_A = periodic_B
                periodic_B = ckpt_path
                last_periodic_step = T

                keep = getattr(args, 'keep_checkpoints', False)
                if not keep and old_A and old_A != periodic_A and old_A != periodic_B:
                    _delete_ckpt(old_A)

                if on_checkpoint_saved is not None:
                    on_checkpoint_saved(ckpt_path, prev_path=periodic_A, is_milestone=False)

        state = next_state

    env.close()

    # Clean up the two held periodic checkpoints (A and B) that outlived training.
    # Skipped when keep_checkpoints=True — all step files are intentionally retained.
    keep = getattr(args, 'keep_checkpoints', False)
    if not keep:
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
