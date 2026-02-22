# PPOSAC-tracker

PPO and SAC implementations for MiniGrid environments, with achievement tracking and checkpoint analysis.

## Installation

```bash
pip install gymnasium minigrid torch numpy tyro tensorboard torch_ac
```

Also requires [rl-starter-files](https://github.com/lcswillems/rl-starter-files) cloned as a sibling directory (`../rl-starter-files`).

## Training

Both algorithms use [tyro](https://github.com/brentyi/tyro) for CLI configuration.

**PPO:**
```bash
python ppo/train.py --env-id MiniGrid-DoorKey-5x5-v0 --total-episodes 50000
```

**SAC:**
```bash
python sac/train.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 10000000
```

Key args (both): `--seed`, `--cuda`, `--track` (W&B), `--checkpoint-freq`, `--experiment-root`

Checkpoints saved to `{experiment_root}/checkpoints/{ppo|sac}/`.

## Analysis

```bash
python analyze_checkpoint.py
```

Runs gradient analysis, activation extraction, and RSA on a saved checkpoint.

## Monitoring

```bash
python -m tensorboard.main --logdir runs
```

## Project Structure

```
PPOSAC-tracker/
├── ppo/
│   ├── train.py            # PPO training (ACModel + torch_ac.PPOAlgo)
│   ├── sampling.py         # Evaluation episode rollouts
│   ├── activations.py      # Activation extraction
│   ├── gradients.py        # Gradient analysis
│   └── checkpoint_gen.py   # Milestone checkpoint saving
├── sac/
│   ├── train.py            # SAC training
│   ├── sampling.py         # Evaluation rollouts
│   ├── gradients.py        # Gradient analysis
│   ├── reward_moments.py   # Reward statistics
│   └── tagged_buffer.py    # Replay buffer with episode tagging
├── shared/
│   ├── networks.py         # ACModelWrapper, PPOAgent
│   ├── achievements.py     # Achievement definitions + eps scoring
│   ├── activation_utils.py # Shared activation helpers
│   ├── gradient_utils.py   # Shared gradient helpers
│   ├── metrics.py          # Evaluation metrics
│   ├── rsa.py              # Representational similarity analysis
│   ├── storage.py          # Saving analysis results
│   └── thresholding.py     # Episode partitioning
├── analysis/
│   ├── run_rsa.py          # RSA analysis runner
│   └── run_sac_analysis.py # SAC analysis runner
├── wrappers.py             # DoorKey + KeyCorridor achievement wrappers
├── analyze_checkpoint.py   # Analysis pipeline entry point
├── sweep.py                # Hyperparameter sweep
└── _rl_path.py             # Adds rl-starter-files to sys.path
```

## Environments

Supported environments with achievement tracking:

| Environment | Wrapper |
|-------------|---------|
| `MiniGrid-DoorKey-*` | `DoorKeyAchievementWrapper` |
| `MiniGrid-KeyCorridorS*` | `KeyCorridorAchievementWrapper` |

Both wrappers inject `info["achievements"]` (per-milestone flags) and `info["eps"]` (exploration progress score) into each step and reset.

Full environment list: https://minigrid.farama.org/environments/minigrid/
