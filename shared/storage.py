import json
import os
import re

import numpy as np


# convert numpy types to native Python for JSON serialisation
def _convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# save a metrics dict / list as JSON
def save_analysis_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)


# load a JSON analysis results file
def load_analysis_results(path):
    with open(path, "r") as f:
        return json.load(f)


# write metadata.json with training hyperparameters
def save_metadata(experiment_root, ppo_args=None, rainbow_args=None):
    meta = {}
    if ppo_args:
        meta["ppo"] = vars(ppo_args) if hasattr(ppo_args, "__dict__") else ppo_args
    if rainbow_args:
        meta["rainbow"] = vars(rainbow_args) if hasattr(rainbow_args, "__dict__") else rainbow_args
    save_analysis_results(meta, os.path.join(experiment_root, "metadata.json"))


# list checkpoint .pt files sorted by step number
def get_checkpoint_paths(experiment_root, algorithm):
    ckpt_dir = os.path.join(experiment_root, "checkpoints", algorithm)
    if not os.path.exists(ckpt_dir):
        return []
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]

    def sort_key(f):
        nums = re.findall(r"\d+", f)
        return int(nums[0]) if nums else 0

    return [os.path.join(ckpt_dir, f) for f in sorted(files, key=sort_key)]
