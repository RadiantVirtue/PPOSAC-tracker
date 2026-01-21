# PPOSAC-tracker

PPO and SAC implementations adapted for Minigrid environments, based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Installation

```bash
pip install gymnasium numpy torch tyro tensorboard minigrid cleanrl
```

## Files

| File | Description |
|------|-------------|
| `ppo.py` | Original CleanRL PPO for classic control (CartPole, etc.) |
| `ppo_minigrid.py` | PPO adapted for Minigrid grid environments |
| `sac_atari.py` | Original CleanRL SAC for Atari games |
| `sac_minigrid.py` | SAC adapted for Minigrid grid environments |

## Usage

### PPO on Minigrid

```bash
# Basic run
python ppo_minigrid.py

# Specify environment
python ppo_minigrid.py --env_id MiniGrid-Empty-8x8-v0

# With video capture
python ppo_minigrid.py --env_id MiniGrid-DoorKey-5x5-v0 --capture_video True

# With Weights & Biases tracking
python ppo_minigrid.py --track True --wandb_project_name my_project
```

### SAC on Minigrid

```bash
# Basic run
python sac_minigrid.py

# Specify environment
python sac_minigrid.py --env_id MiniGrid-FourRooms-v0

# With video capture
python sac_minigrid.py --env_id MiniGrid-LavaGapS5-v0 --capture_video True
```

## Common Minigrid Environments

| Environment | Description |
|-------------|-------------|
| `MiniGrid-Empty-5x5-v0` | Empty 5x5 grid, agent must reach goal |
| `MiniGrid-Empty-8x8-v0` | Empty 8x8 grid |
| `MiniGrid-Empty-16x16-v0` | Empty 16x16 grid |
| `MiniGrid-DoorKey-5x5-v0` | Pick up key, open door, reach goal |
| `MiniGrid-DoorKey-8x8-v0` | Larger door-key task |
| `MiniGrid-FourRooms-v0` | Navigate through four connected rooms |
| `MiniGrid-LavaGapS5-v0` | Cross a gap with lava |
| `MiniGrid-SimpleCrossingS9N1-v0` | Cross a room avoiding obstacles |

Full list: https://minigrid.farama.org/environments/minigrid/

## Hyperparameters

### PPO (`ppo_minigrid.py`)

```bash
--total_timesteps 500000    # Total training steps
--learning_rate 2.5e-4      # Learning rate
--num_envs 4                # Parallel environments
--num_steps 128             # Steps per rollout
--gamma 0.99                # Discount factor
--gae_lambda 0.95           # GAE lambda
--num_minibatches 4         # Minibatches per update
--update_epochs 4           # Epochs per update
--clip_coef 0.2             # PPO clip coefficient
--ent_coef 0.01             # Entropy bonus coefficient
--vf_coef 0.5               # Value function coefficient
```

### SAC (`sac_minigrid.py`)

```bash
--total_timesteps 500000    # Total training steps
--buffer_size 100000        # Replay buffer size
--batch_size 64             # Training batch size
--learning_starts 10000     # Steps before training starts
--policy_lr 3e-4            # Policy learning rate
--q_lr 3e-4                 # Q-network learning rate
--gamma 0.99                # Discount factor
--tau 1.0                   # Target network update rate
--alpha 0.2                 # Entropy coefficient (if autotune=False)
--autotune True             # Auto-tune entropy coefficient
```

## Monitoring

Training logs are saved to `runs/` directory. View with TensorBoard:

```bash
tensorboard --logdir runs
```

## Key Adaptations from Original

The Minigrid versions differ from the originals in:

1. **Observation handling**: Minigrid returns a Dict observation; we use `ImgObsWrapper` to extract just the 7x7x3 image grid
2. **Network architecture**: CNN instead of MLP to process spatial grid observations
3. **Channel ordering**: Permute from (H, W, C) to (C, H, W) for PyTorch Conv2d
4. **Removed Atari wrappers**: No frame stacking, grayscale, or reward clipping needed
