# PPOSAC-tracker

PPO and SAC implementations adapted for Minigrid environments, based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

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

Then run:

```bash
python main.py
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point with configuration |
| `ppo_minigrid.py` | PPO implementation for Minigrid |
| `sac_minigrid.py` | SAC implementation for Minigrid |

## Available Environments

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

**Can just download file and save to folder, and use that instead**

## Monitoring

Training logs are saved to `runs/` directory. View with TensorBoard:

```bash
tensorboard --logdir runs
```
**FULL TRAINING LOGS + FROZEN AGENTS TBA**

## Key Adaptations from Original CleanRL

The Minigrid versions differ from the originals in:

1. **Observation handling**: Minigrid returns a Dict observation; I use `ImgObsWrapper` to just extract the 7x7x3 image grid
2. **Network architecture**: CNN instead of MLP
3. **Channel ordering**:  from (H, W, C) to (C, H, W) for PyTorch Conv2d
4. **Removed Atari wrappers**: No frame stacking, grayscale, or reward clipping
