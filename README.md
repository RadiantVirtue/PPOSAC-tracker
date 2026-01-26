# PPOSAC-tracker

PPO and SAC implementations adapted for MiniGrid environments, based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Installation

```bash
pip install gymnasium minigrid torch numpy tyro tensorboard
```

## Training

Edit the configs at the top of `main.py`:

```python
ENV_ID = "MiniGrid-Empty-5x5-v0"
ALGORITHM = "ppo"  # "ppo" or "sac"
TOTAL_TIMESTEPS = 500000
SEED = 1
CAPTURE_VIDEO = False
TRACK_WANDB = False
```

Then run: `python main.py`

**Checkpointing:** Pass `checkpoint_freq` to save periodic checkpoints. Set to `0` to disable (default).

```python
args = Args(
    env_id=ENV_ID,
    total_timesteps=TOTAL_TIMESTEPS,
    checkpoint_freq=50000,  # Save every 50k steps
)
```

Checkpoints saved to `checkpoints/{run_name}/step_{step}.pt` and `checkpoints/{run_name}/final.pt`

## Running Frozen Agents

Edit config at top of `run_ppo_agent.py` or `run_sac_agent.py`:

```python
CHECKPOINT = "checkpoints/MiniGrid-DoorKey-5x5-v0__ppo_minigrid__1__1234567890/final.pt"
EPISODES = 5
RENDER = True
DETERMINISTIC = True  # argmax vs sampling
DELAY = 0.1
```

Then run: `python run_ppo_agent.py` (or `run_sac_agent.py`)

Environment is auto-detected from checkpoint path.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main training entry point |
| `ppo_minigrid.py` | PPO implementation for MiniGrid |
| `sac_minigrid.py` | SAC implementation for MiniGrid |
| `run_ppo_agent.py` | Run frozen PPO agents |
| `run_sac_agent.py` | Run frozen SAC agents |
| `ppo.py` | Original PPO for general environments |
| `sac_atari.py` | SAC for Atari environments |

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

Training logs saved to `runs/`. View with TensorBoard:

```bash
python -m tensorboard.main --logdir runs
```