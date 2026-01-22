# PPOSAC-tracker

PPO and SAC implementations adapted for MiniGrid environments, based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Usage

Edit the configs at the top of `main.py`:

```python
# Select environment
ENV_ID = "MiniGrid-Empty-5x5-v0"

# Select algorithm: "ppo" or "sac"
ALGORITHM = "ppo"

# Training settings
TOTAL_TIMESTEPS = 500000
SEED = 1
CAPTURE_VIDEO = False
TRACK_WANDB = False
```
## RL Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point with configuration |
| `ppo_minigrid.py` | PPO implementation for MiniGrid |
| `sac_minigrid.py` | SAC implementation for MiniGrid |
| `ppo.py` | Original PPO for general environments |
| `sac_atari.py` | SAC adapted for Atari environments |

### Custom Environments

| File | Description |
|------|-------------|
| `doorkey.py` | DoorKey task implementation |
| `keycorridor.py` | KeyCorridor task implementation |
| `multiroom.py` | MultiRoom navigation task |
| `obstructedmaze.py` | ObstructedMaze task |

## Environments

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

## Monitoring

Training logs are saved to the `runs/` directory. View with TensorBoard:

```bash
tensorboard --logdir runs
```

1. **Observation handling**: MiniGrid returns a Dict observation; uses `ImgObsWrapper` to extract the 7x7x3 image grid
2. **Network architecture**: MLP (7x7x3 = 147 input features)
3. **Removed Atari wrappers**: No frame stacking, grayscale, or reward clipping
