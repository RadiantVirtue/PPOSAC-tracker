"""rainbow/weight_delta.py — Weight-delta empirical validation for IS gradient reconstruction.

Computes cosine similarities between each offline gradient variant (G_uniform,
G_IS, G_reward) and the actual weight change Δθ = θ_{N+1} − θ_N between
consecutive analyzed checkpoints.

If cos(G_IS, Δθ) > cos(G_uniform, Δθ), the IS reconstruction better predicts
the direction weights actually moved — empirical validation of the stale-priority
approximation without needing the live replay buffer.

Key design choices:
- NoisyLinear sigma params (weight_sigma, bias_sigma) are excluded from Δθ and
  from the gradient dicts before comparison. Sigma params have zero analytical
  gradient in eval mode (noise is not sampled) but receive real weight updates
  during training. Including them would create systematic misalignment.
- cosine_similarity_flat() in shared/gradient_utils.py takes two {param_name: tensor}
  dicts and concatenates by sorted key order — both Δθ and the gradient dicts must
  have matching key sets.
- Absolute cosines will be low: Adam momentum accumulates many gradient steps between
  checkpoints, and gradients are computed at N+1's frozen weights rather than the
  intermediate weights that drove Δθ. Interpret *relative* differences (G_IS vs
  G_uniform), not absolute values.
- Training-stage caveat: relative ordering is most reliable at mid-training (large
  Δθ, meaningful signal). At late training Δθ is tiny and near-orthogonal to all
  variants — both cosines approach zero and ordering becomes unreliable. This is the
  opposite of where G_IS is most accurate. See Gradient Analysis v5.md §6.

Reference: Gradient Analysis v5.md, Section 6.
"""


def _non_sigma_keys(d: dict) -> list:
    """Return sorted param names that are not NoisyLinear sigma params.

    NoisyLinear stores 4 tensors per layer: weight_mu, weight_sigma,
    bias_mu, bias_sigma. Sigma params have zero analytical gradient in
    eval mode (noise is not sampled) but receive real weight updates
    during training. Excluding them makes Δθ and G_* comparable.
    Regular Conv and Linear layers have no sigma params — all their
    keys pass through unchanged.
    """
    return sorted(k for k in d if "weight_sigma" not in k and "bias_sigma" not in k)


def _state_delta_dict(prev_sd: dict, curr_sd: dict) -> dict:
    """Compute per-parameter weight change, excluding sigma params.

    Returns {param_name: (curr - prev) tensor} for all non-sigma params.
    """
    keys = _non_sigma_keys(prev_sd)
    return {k: (curr_sd[k] - prev_sd[k]).float() for k in keys}


def compute_weight_delta_metrics(
    prev_state_dict: dict,
    curr_state_dict: dict,
    grad_s: dict,
    grad_f: dict,
) -> dict:
    """Compute cosine similarities between gradient directions and Δθ.

    Δθ = θ_{N+1} − θ_N (non-sigma params only) is the cumulative weight
    change between the two analyzed checkpoints (covering all Adam updates
    in between; span may exceed analyze_every × checkpoint_interval if
    skip_existing caused some checkpoints to be skipped in analyze_range).

    Absolute cosines will be low: Adam momentum accumulates many gradient
    steps, and the gradients are computed at N+1's frozen weights rather
    than the intermediate weights that produced Δθ. Interpret *relative*
    differences (G_IS vs G_uniform) not absolute values.

    Training-stage caveat: relative ordering is most reliable at mid-training
    where Δθ is large. At late training (near convergence) Δθ is tiny and
    near-orthogonal to all gradient variants — both cosines approach zero and
    their ordering becomes unreliable. This is the opposite of where G_IS is
    most accurate (late training). Treat validation as strongest at mid-training
    checkpoints.

    Args:
        prev_state_dict: online_net.state_dict() from checkpoint N
        curr_state_dict: online_net.state_dict() from checkpoint N+1
        grad_s: return dict from compute_group_gradient_with_coherence (success)
        grad_f: return dict from compute_group_gradient_with_coherence (failure)

    Returns:
        dict with 6 float keys (or None where gradient variant unavailable):
            cos_uniform_success, cos_is_success, cos_reward_success,
            cos_uniform_failure, cos_is_failure, cos_reward_failure
    """
    from shared.gradient_utils import cosine_similarity_flat

    delta = _state_delta_dict(prev_state_dict, curr_state_dict)
    keep_keys = set(delta.keys())

    def _cos(raw_grad):
        """Cosine similarity between a gradient dict and Δθ, sigma-filtered.

        raw_grad comes from named_parameters() — parameters only.
        delta comes from state_dict() — parameters + registered buffers (e.g. support).
        Intersect to the keys present in both so both vectors have the same length.
        """
        if raw_grad is None:
            return None
        # common_keys = non-sigma params present in both raw_grad and delta
        common_keys = sorted(k for k in raw_grad if k in keep_keys)
        if not common_keys:
            return None
        filtered_grad  = {k: raw_grad[k].float() for k in common_keys}
        filtered_delta = {k: delta[k]            for k in common_keys}
        return cosine_similarity_flat(filtered_grad, filtered_delta)

    def _group(grad_dict, variant):
        # Use explicit key access rather than .get(variant, {}) to catch dict
        # structure mismatches early. If the variant key is absent the caller
        # has passed an incompatible grad dict — fail loudly rather than silently
        # returning None (which would look like a valid "first checkpoint" result).
        if variant not in grad_dict:
            raise KeyError(
                f"compute_weight_delta_metrics: expected variant '{variant}' in grad dict; "
                f"got keys {list(grad_dict.keys())}. "
                "Ensure compute_group_gradient_with_coherence return format is v4+."
            )
        raw = grad_dict[variant].get("raw")
        return _cos(raw)

    return {
        "cos_uniform_success": _group(grad_s, "uniform"),
        "cos_is_success":      _group(grad_s, "is_weighted"),
        "cos_reward_success":  _group(grad_s, "reward_weighted"),
        "cos_uniform_failure": _group(grad_f, "uniform"),
        "cos_is_failure":      _group(grad_f, "is_weighted"),
        "cos_reward_failure":  _group(grad_f, "reward_weighted"),
    }
