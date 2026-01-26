# Run a frozen SAC agent on a MiniGrid environment
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import minigrid
from minigrid.wrappers import ImgObsWrapper

# ============ CONFIGURATION ============
CHECKPOINT = "checkpoints/YOUR_RUN_NAME/final.pt"
EPISODES = 5
RENDER = True
DETERMINISTIC = False  # True = argmax, False = sample from policy
DELAY = 0.1  # Seconds between steps when rendering
# =======================================


def get_env_id_from_checkpoint(checkpoint_path):
    """Extract env_id from checkpoint path.

    Expected format: checkpoints/{env_id}__{exp_name}__{seed}__{timestamp}/file.pt
    """
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))
    env_id = folder_name.split("__")[0]
    return env_id


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        obs_dim = np.array(obs_shape).prod()
        self.fc1 = layer_init(nn.Linear(obs_dim, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc_logits = layer_init(nn.Linear(256, n_actions))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x, deterministic=False):
        logits = self(x)
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            policy_dist = Categorical(logits=logits)
            action = policy_dist.sample()
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

    # Create actor with correct architecture
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    actor = Actor(obs_shape, n_actions).to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

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
                action = actor.get_action(obs, deterministic=DETERMINISTIC)

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
