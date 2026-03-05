"""SAC with PER and CNN for MiniGrid environments.
Adapted from CleanRL's SAC Atari implementation.

Key design choices vs. original:
- CNN encoder matching PPO's ACModel for direct cross-algorithm comparison
- PER (Prioritized Experience Replay) via SumTree in TaggedPERBuffer
- Soft target updates (tau=0.005 every step) instead of hard copy every 8 steps
- autotune=True: entropy coefficient adjusts automatically to target entropy
- Observations stored as raw (H, W, C) image arrays; no OneHot or ImgObs wrappers
"""
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium

from wrappers import DoorKeyAchievementWrapper, KeyCorridorAchievementWrapper
from sac.per_buffer import TaggedPERBuffer
from sac.tagged_buffer import EpisodeStore
from shared.networks import SACCNNActor, SACCNNQNetwork


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "MiniGrid-DoorKey-8x8-v0"
    total_timesteps: int = 10_000_000
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.005          # soft target update rate (was 1.0 hard copy)
    batch_size: int = 256
    learning_starts_steps: int = 5000   # more random exploration before CNN training starts
    policy_lr: float = 1e-4    # lower than FC+ReLU; CNN+Tanh is more sensitive to large LR
    q_lr: float = 1e-4
    update_frequency: int = 4
    target_network_frequency: int = 1   # soft update every step (tau=0.005)
    alpha: float = 0.05         # initial entropy coefficient (autotune adjusts from here)
    autotune: bool = True       # auto-tune alpha to maintain target entropy
    target_entropy_scale: float = 0.5
    checkpoint_freq: int = 1000
    experiment_root: str = "experiment_root"

    # PER hyperparameters
    per_alpha: float = 0.6          # priority exponent (0=uniform, 1=greedy)
    per_beta_start: float = 0.4     # initial IS correction exponent
    per_beta_steps: int = 500_000   # steps to anneal beta from per_beta_start to 1.0


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        if "DoorKey" in env_id:
            env = DoorKeyAchievementWrapper(env)
        elif "KeyCorridor" in env_id:
            env = KeyCorridorAchievementWrapper(env)
        else:
            raise ValueError(f"No achievement wrapper for env_id: {env_id}")
        # No OneHotPartialObsWrapper / ImgObsWrapper — CNN handles raw 7x7x3 obs directly
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def save_checkpoint_sac(actor, qf1, qf2, qf1_target, qf2_target,
                        q_optimizer, actor_optimizer, global_step, episode_count,
                        path, log_alpha=None, a_optimizer=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "global_step": global_step,
        "episode_count": episode_count,
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if a_optimizer is not None:
        checkpoint["a_optimizer_state_dict"] = a_optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def main_sac(args, on_checkpoint_saved=None, should_stop=None):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(
            [f"|{key}|{value}|" for key, value in vars(args).items()]
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_actions = envs.single_action_space.n

    actor = SACCNNActor(n_actions).to(device)
    qf1 = SACCNNQNetwork(n_actions).to(device)
    qf2 = SACCNNQNetwork(n_actions).to(device)
    qf1_target = SACCNNQNetwork(n_actions).to(device)
    qf2_target = SACCNNQNetwork(n_actions).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(n_actions, dtype=torch.float32)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha
        log_alpha = None
        a_optimizer = None

    # Observation shape: raw MiniGrid image (H, W, C) — no wrappers modify this
    obs_shape = envs.single_observation_space["image"].shape   # (7, 7, 3) for standard envs

    rb = TaggedPERBuffer(
        args.buffer_size,
        obs_shape,
        n_actions,
        device,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=1.0,
        per_beta_steps=args.per_beta_steps,
        per_epsilon=1e-6,
    )
    episode_store = EpisodeStore()
    start_time = time.time()

    current_episode_id = 0
    ep_obs_buf, ep_act_buf, ep_rew_buf, ep_done_buf = [], [], [], []
    ep_return = 0.0

    global_step = 0
    obs, _ = envs.reset(seed=args.seed)   # obs: dict {"image": (1,H,W,C), "mission": ...}
    next_checkpoint = args.checkpoint_freq if args.checkpoint_freq > 0 else 0

    # Scalars set inside the update block, logged every 100 steps
    qf1_a_values = qf2_a_values = qf1_loss = qf2_loss = qf_loss = actor_loss = None
    alpha_loss = None
    mean_is_weight = 0.0
    current_beta = args.per_beta_start

    while global_step < args.total_timesteps:
        # Extract image from dict obs: (1, H, W, C)
        obs_image = obs["image"].astype(np.float32)

        if global_step < args.learning_starts_steps:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(
                    torch.tensor(obs_image, dtype=torch.float32).to(device)
                )
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        global_step += 1

        if "episode" in infos and infos["_episode"][0]:
            ep_return_logged = float(infos["episode"]["r"][0])
            ep_length_logged = int(infos["episode"]["l"][0])
            print(
                f"global_step={global_step}, episode={current_episode_id}, "
                f"episodic_return={ep_return_logged}"
            )
            writer.add_scalar("charts/episodic_return", ep_return_logged, current_episode_id)
            writer.add_scalar("charts/episodic_length", ep_length_logged, current_episode_id)

        # Handle terminal observation for truncated episodes
        real_next_obs_image = next_obs["image"].astype(np.float32)
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc and infos["_final_observation"][idx]:
                    real_next_obs_image[idx] = infos["final_observation"][idx]["image"].astype(np.float32)

        # Store single-env transition (strip leading batch dim)
        ep_obs_buf.append(obs_image[0])
        ep_act_buf.append(actions[0])
        ep_rew_buf.append(rewards[0])
        ep_return += rewards[0]

        rb.add(
            obs_image[0], real_next_obs_image[0],
            actions[0], rewards[0], terminations[0],
            current_episode_id,
        )

        done = terminations[0] or truncations[0]
        ep_done_buf.append(done)

        if done:
            ep_eps = float(infos["eps"][0]) if "eps" in infos else 0.0
            rb.backfill_episode(current_episode_id, ep_return, ep_eps)

            achievements = {}
            if "achievements" in infos:
                achievements = {
                    k: bool(v[0])
                    for k, v in infos["achievements"].items()
                    if not k.startswith("_")
                }
            episode_store.add_episode(
                ep_obs_buf, ep_act_buf, ep_rew_buf, ep_done_buf,
                ep_return, ep_eps, achievements,
            )

            current_episode_id += 1
            ep_obs_buf, ep_act_buf, ep_rew_buf, ep_done_buf = [], [], [], []
            ep_return = 0.0

            if args.checkpoint_freq > 0 and current_episode_id >= next_checkpoint:
                ckpt_base = (
                    f"{args.experiment_root}/checkpoints/sac/"
                    f"periodic_{current_episode_id // 1000}k_episodes"
                )
                save_checkpoint_sac(
                    actor, qf1, qf2, qf1_target, qf2_target,
                    q_optimizer, actor_optimizer, global_step,
                    current_episode_id, f"{ckpt_base}.pt",
                    log_alpha if args.autotune else None,
                    a_optimizer if args.autotune else None,
                )
                rb.save(f"{ckpt_base}_buffer.pkl")
                episode_store.save(f"{ckpt_base}_episodes.pkl")
                if on_checkpoint_saved:
                    on_checkpoint_saved(f"{ckpt_base}.pt")
                next_checkpoint += args.checkpoint_freq

            if should_stop and should_stop({"achievements": achievements}):
                break

        obs = next_obs

        # ----------------------------------------------------------------
        # Training updates
        # ----------------------------------------------------------------
        if global_step > args.learning_starts_steps:
            if global_step % args.update_frequency == 0:
                data, is_weights, leaf_indices = rb.sample(args.batch_size)
                mean_is_weight = is_weights.mean().item()
                t = min(1.0, rb._beta_step / rb.per_beta_steps)
                current_beta = rb.per_beta_start + t * (rb.per_beta_end - rb.per_beta_start)

                # -- Q-network update --
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(
                        data["next_observations"]
                    )
                    qf1_next_target = qf1_target(data["next_observations"])
                    qf2_next_target = qf2_target(data["next_observations"])
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                    )
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = (
                        data["rewards"].flatten()
                        + (1 - data["dones"].flatten()) * args.gamma * min_qf_next_target
                    )

                qf1_values = qf1(data["observations"])
                qf2_values = qf2(data["observations"])
                qf1_a_values = qf1_values.gather(1, data["actions"].long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data["actions"].long()).view(-1)

                # IS-weighted MSE: only Q-loss uses importance-sampling correction
                td1 = (qf1_a_values - next_q_value).detach()
                td2 = (qf2_a_values - next_q_value).detach()
                qf1_loss = (is_weights * (qf1_a_values - next_q_value) ** 2).mean()
                qf2_loss = (is_weights * (qf2_a_values - next_q_value) ** 2).mean()
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Update priorities with mean absolute TD error across both networks
                td_errors = ((td1.abs() + td2.abs()) / 2).cpu().numpy()
                rb.update_priorities(leaf_indices, td_errors)

                # -- Actor update (IS correction does NOT apply here) --
                _, log_pi, action_probs = actor.get_action(data["observations"])
                with torch.no_grad():
                    qf1_values = qf1(data["observations"])
                    qf2_values = qf2(data["observations"])
                    min_qf_values = torch.min(qf1_values, qf2_values)
                actor_loss = (
                    action_probs * ((alpha * log_pi) - min_qf_values)
                ).sum(dim=1).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # -- Alpha update (IS correction does NOT apply here) --
                if args.autotune:
                    alpha_loss = (
                        action_probs.detach()
                        * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                    ).sum(dim=1).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # Soft target network update every step (tau=0.005)
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0 and qf1_a_values is not None:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), current_episode_id)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), current_episode_id)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), current_episode_id)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), current_episode_id)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, current_episode_id)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), current_episode_id)
                writer.add_scalar("losses/alpha", alpha, current_episode_id)
                writer.add_scalar("losses/per_beta", current_beta, current_episode_id)
                writer.add_scalar("losses/mean_is_weight", mean_is_weight, current_episode_id)
                writer.add_scalar("charts/SPS",
                                  int(global_step / (time.time() - start_time)),
                                  current_episode_id)
                if args.autotune and alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), current_episode_id)

    # Final checkpoint
    if args.checkpoint_freq > 0:
        ckpt_base = (
            f"{args.experiment_root}/checkpoints/sac/"
            f"final_{current_episode_id // 1000}k_episodes"
        )
        save_checkpoint_sac(
            actor, qf1, qf2, qf1_target, qf2_target,
            q_optimizer, actor_optimizer, global_step,
            current_episode_id, f"{ckpt_base}.pt",
            log_alpha if args.autotune else None,
            a_optimizer if args.autotune else None,
        )
        rb.save(f"{ckpt_base}_buffer.pkl")
        episode_store.save(f"{ckpt_base}_episodes.pkl")
        if on_checkpoint_saved:
            on_checkpoint_saved(f"{ckpt_base}.pt")

    envs.close()
    writer.close()
    return current_episode_id, global_step


if __name__ == "__main__":
    args = tyro.cli(Args)
    main_sac(args)
