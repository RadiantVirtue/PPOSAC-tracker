"""Batch Rainbow analysis: load networks, run metrics for every checkpoint.

Analogous to the former analysis/run_sac_analysis.py.

Public API
----------
load_rainbow_networks(checkpoint_path, n_actions, device)
    → (net, episode_count)

run_rainbow_analysis(experiment_root, env_id, device)
    → list of result dicts (one per checkpoint)
"""
import os

import gymnasium as gym
import minigrid  # noqa: F401
import torch

from rainbow.network import Dueling_CNN_Net, CNN_Net
from shared.storage import get_checkpoint_paths, save_analysis_results


# ── Network loader ────────────────────────────────────────────────────────────

def load_rainbow_networks(checkpoint_path: str, n_actions: int, device: str = "cuda"):
    """Reconstruct the online Rainbow network from a checkpoint.

    The checkpoint must contain 'obs_shape' (saved by rainbow/train.py).

    Args:
        checkpoint_path: path to a .pt checkpoint file.
        n_actions:       number of discrete actions (from env.action_space.n).
        device:          torch device string.

    Returns:
        (net, episode_count)
        net is a Dueling_CNN_Net (use_dueling=True by default for Rainbow).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    obs_shape  = ckpt.get("obs_shape", (7, 7, 3))
    action_dim = ckpt.get("action_dim", n_actions)

    # Reconstruct a minimal args namespace matching network constructor fields
    import argparse
    net_args = argparse.Namespace(
        obs_shape=obs_shape,
        action_dim=action_dim,
        use_noisy=True,     # Rainbow default
        use_dueling=True,   # Rainbow default
    )

    net = Dueling_CNN_Net(net_args)
    net.load_state_dict(ckpt["net_state_dict"])
    net.to(device)
    net.eval()

    episode_count = ckpt.get("episode_count", ckpt.get("global_step", 0))
    return net, episode_count


# ── Batch analysis runner ─────────────────────────────────────────────────────

def run_rainbow_analysis(
    experiment_root: str,
    env_id: str = "MiniGrid-DoorKey-8x8-v0",
    device: str = "cuda",
    n_episodes: int = 500,
    run_rsa: bool = True,
):
    """Analyse every Rainbow checkpoint under experiment_root.

    Calls analyze_checkpoint.analyze_checkpoint() for each .pt file found
    in <experiment_root>/checkpoints/rainbow/.

    Args:
        experiment_root: root directory of a sweep run.
        env_id:          environment used during training.
        device:          torch device for network inference.
        n_episodes:      used only for the PPO path (ignored here).
        run_rsa:         whether to run RSA analysis.

    Returns:
        list of result dicts (same structure as _analyze_rainbow output).
    """
    from analyze_checkpoint import analyze_checkpoint

    ckpt_paths = get_checkpoint_paths(experiment_root, "rainbow")
    if not ckpt_paths:
        print(f"No Rainbow checkpoints found under {experiment_root}/checkpoints/rainbow/")
        return []

    results = []
    for ckpt_path in ckpt_paths:
        episodes_path = ckpt_path.replace(".pt", "_episodes.pkl")
        if not os.path.exists(episodes_path):
            print(f"  Skipping {os.path.basename(ckpt_path)} — no episodes pkl found.")
            continue

        print(f"\nAnalysing: {os.path.basename(ckpt_path)}")
        analyze_checkpoint(
            algorithm="rainbow",
            checkpoint_path=ckpt_path,
            experiment_root=experiment_root,
            env_id=env_id,
            device=device,
            run_rsa=run_rsa,
        )

    return results
