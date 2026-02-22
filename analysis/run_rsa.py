import numpy as np
import torch

from shared.activation_utils import extract_activations
from shared.metrics import rsa_alignment
from shared.rsa import (
    DOORKEY_STIMULUS_SET,
    GROUND_TRUTH_GROUPS,
    build_rdm,
    find_states_containing_element,
)


# binary GT RDM: 0 if same category, 1 if different
def build_ground_truth_rdm(stimulus_set=None, gt_groups=None):
    if stimulus_set is None:
        stimulus_set = DOORKEY_STIMULUS_SET
    if gt_groups is None:
        gt_groups = GROUND_TRUTH_GROUPS
    names = sorted(stimulus_set.keys())
    K = len(names)
    gt = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if gt_groups[names[i]] != gt_groups[names[j]]:
                gt[i, j] = 1.0
    return gt


# build an RDM for one checkpoint and compare to ground truth
# returns None if too few observations for any stimulus element
def run_rsa_for_checkpoint(model, observations, layer_name, device="cuda",
                           stimulus_set=None, gt_groups=None):
    if stimulus_set is None:
        stimulus_set = DOORKEY_STIMULUS_SET
    if gt_groups is None:
        gt_groups = GROUND_TRUTH_GROUPS

    mean_activations = {}

    for elem_name, elem_def in stimulus_set.items():
        indices = find_states_containing_element(observations, elem_def)
        if len(indices) < 5:
            print(
                f"  Too few observations for {elem_name} "
                f"({len(indices)}), skipping RSA"
            )
            return None

        elem_obs = torch.tensor(
            observations[indices], dtype=torch.float32
        )
        acts = extract_activations(model, elem_obs, layer_name, device=device)
        mean_activations[elem_name] = acts.mean(axis=0)

    rdm, names = build_rdm(mean_activations)
    gt_rdm = build_ground_truth_rdm(stimulus_set, gt_groups)
    alignment = rsa_alignment(rdm, gt_rdm)

    return {
        "rdm": rdm.tolist(),
        "element_names": names,
        "alignment": alignment,
    }
