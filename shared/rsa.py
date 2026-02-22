import numpy as np
import torch
import torch.nn.functional as F

# MiniGrid object indices (from minigrid/core/constants.py)
DOORKEY_STIMULUS_SET = {
    "key": {"object_idx": 5, "color_idx": 4},  # yellow key
    "locked_door": {"object_idx": 4, "color_idx": 4, "state": 2},
    "open_door": {"object_idx": 4, "color_idx": 4, "state": 0},
    "goal": {"object_idx": 8},
}

# Ground-truth functional categories for RSA alignment
GROUND_TRUTH_GROUPS = {
    "key": "objective",
    "goal": "objective",
    "locked_door": "barrier",
    "open_door": "barrier",
}

# KeyCorridor stimulus set — 4 elements across 2 categories
# Key and target ball colors vary per episode, so match on object_idx only
KEYCORRIDOR_STIMULUS_SET = {
    "key":         {"object_idx": 5},              # key (color varies)
    "locked_door": {"object_idx": 4, "state": 2},
    "open_door":   {"object_idx": 4, "state": 0},
    "target_ball": {"object_idx": 6},              # target ball (color varies)
}

KEYCORRIDOR_GROUND_TRUTH_GROUPS = {
    "key":         "objective",
    "target_ball": "objective",
    "locked_door": "barrier",
    "open_door":   "barrier",
}


def get_stimulus_config(env_id):
    if "KeyCorridor" in env_id:
        return KEYCORRIDOR_STIMULUS_SET, KEYCORRIDOR_GROUND_TRUTH_GROUPS
    return DOORKEY_STIMULUS_SET, GROUND_TRUTH_GROUPS


# return indices of observations that contain the given element
def find_states_containing_element(observations, element):
    obj_match = observations[:, :, :, 0] == element["object_idx"]

    if "color_idx" in element:
        obj_match = obj_match & (observations[:, :, :, 1] == element["color_idx"])

    if "state" in element:
        obj_match = obj_match & (observations[:, :, :, 2] == element["state"])

    has_element = obj_match.any(axis=(1, 2))
    return np.where(has_element)[0]


# construct a cosine-dissimilarity RDM from per-element mean activations
def build_rdm(mean_activations):
    names = sorted(mean_activations.keys())
    K = len(names)
    rdm = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            vec_i = torch.tensor(mean_activations[names[i]], dtype=torch.float32)
            vec_j = torch.tensor(mean_activations[names[j]], dtype=torch.float32)
            cos_sim = F.cosine_similarity(
                vec_i.unsqueeze(0), vec_j.unsqueeze(0)
            ).item()
            rdm[i, j] = 1.0 - cos_sim

    return rdm, names
