# Run a frozen PPO agent on a MiniGrid environment
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import minigrid
from minigrid.wrappers import ImgObsWrapper

#config
CHECKPOINT = "checkpoints/YOUR_RUN_NAME/final.pt"
EPISODES = 5
RENDER = True
DETERMINISTIC = True
DELAY = 0.1 


def get_env_id_from_checkpoint(checkpoint_path):
    """Extract env_id from checkpoint path.

    Expected format: checkpoints/{env_id}__{exp_name}__{seed}__{timestamp}/file.pt
    """
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))
    env_id = folder_name.split("__")[0]
    return env_id


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        obs_dim = np.array(obs_shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_action(self, x, deterministic=False):
        x = x.flatten(start_dim=1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(probs.probs, dim=1)
        else:
            action = probs.sample()
        return action


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract env_id from checkpoint path
    env_id = get_env_id_from_checkpoint(CHECKPOINT)

    # Create environment
    if RENDER:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)
    env = ImgObsWrapper(env)

    # Create agent with correct architecture
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(obs_shape, n_actions).to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    print(f"Loaded checkpoint from step {checkpoint['global_step']}")
    print(f"Environment: {env_id}")
    print(f"Running {EPISODES} episodes...")
    print("-" * 50)

    # Run episodes
    total_rewards = []
    total_lengths = []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=DETERMINISTIC)

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            obs = torch.Tensor(obs).unsqueeze(0).to(device)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

            if RENDER:
                time.sleep(DELAY)

        total_rewards.append(ep_reward)
        total_lengths.append(ep_length)
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_length}")

    print("-" * 50)
    print(f"Average reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"Average length: {np.mean(total_lengths):.2f} (+/- {np.std(total_lengths):.2f})")

    env.close()


if __name__ == "__main__":
    main()
