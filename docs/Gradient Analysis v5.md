# Gradient Analysis Methodology v5

Builds on v4. Three methodological additions: joint IS normalisation, diagnostic funnel reframing, and weight-delta empirical validation.

---

## Summary of Changes from v4

| Aspect | v4 | v5 |
|---|---|---|
| Partition | Percentile split (top/bottom K%) | Same — unchanged |
| Gradient variants | G_uniform, G_IS, G_reward | Same — unchanged |
| IS normalisation scope | Within-group (success separately, failure separately) | **Joint pool** (success + failure together) |
| Funnel framing | "Verification funnel" | **"Diagnostic funnel"** — cos_is_reward is a finding metric, not a gate |
| Empirical validation | None | **Weight-delta comparison** — cos(G, Δθ) per variant per group |

PPO analysis is unchanged. All changes apply to Rainbow only.

---

## 1. Partition Function: Percentile Split

Unchanged from v4. Reproduced here for completeness.

```
K = percentile_x (default 25)
Success = top-K% of episodes ranked by EPS
Failure = bottom-K% of episodes ranked by EPS
Middle (100 - 2K)% = discarded
```

Groups have equal, stable size at every checkpoint regardless of training stage. K=25 gives 125 episodes per group from 500 evaluation episodes.

**Threshold reported in JSON:**
- `threshold_lower`: EPS value at the K-th percentile (boundary of failure group)
- `threshold_upper`: EPS value at the (100−K)-th percentile (boundary of success group)

---

## 2. Three Gradient Variants

Unchanged from v4. Reproduced here for completeness.

### G_uniform (baseline, matches v3)

```
loss_uniform = (1/T) Σ_t L_t
```

All transitions contribute equally.

### G_IS (primary estimate)

Analytically reconstructs the IS-weighted gradient using the frozen network's own distributional Bellman loss as a proxy for the per-transition priorities that PER would assign.

```
priority_i = L_i ^ α                       α = priority_exponent (default 0.5)
P(i)       = priority_i / Σ_{k ∈ pool} priority_k
w_i        = (1 / (N · P(i))) ^ β
w_i        = w_i / max(w)                  normalize by batch max (matches training)

β = β_start + (1 − β_start) · clip((global_step − learn_start) / (T_max − learn_start), 0, 1)
```

In v5 the priority sum Σ_{k ∈ pool} is computed over the **joint success+failure pool** (see Section 3).

### G_reward (verification proxy)

```
w_t = |r_t| + ε      (ε = 0.01)
w_t = w_t / Σ w_t    (normalize to sum 1 per episode)
loss_reward = (1/T) Σ_t w_t · L_t
```

High-reward transitions receive more weight. Used as a diagnostic proxy — see Section 4.

---

## 3. Joint IS Normalisation (NEW in v5)

### v4 definition (replaced)

IS weights were normalised within each group separately:

```
P(i) = priority_i / Σ_{k ∈ success} priority_k   [success group]
P(i) = priority_i / Σ_{k ∈ failure} priority_k   [failure group]
```

**Problem:** the priority denominators differ between groups, so IS-weighted gradient magnitudes are on incomparable scales — `gradient_magnitude_success_is` vs `gradient_magnitude_failure_is` could not be compared.

### v5 definition (primary)

IS weights are normalised over the combined success+failure pool:

```
P(i) = priority_i / Σ_{k ∈ success ∪ failure} priority_k
```

Both groups share the same denominator. The max-normalisation `w_i / max(w)` is then applied globally over the joint pool before splitting back to per-group weight tensors.

**Implementation:** a single forward-only pass (`torch.no_grad()`) is run over all episodes (success + failure together) before any backward pass. IS weights are computed once over the full pool and split by group index before being passed to `compute_group_gradient_with_coherence`.

### Residual cross-group magnitude limitation

Joint normalisation does **not** make IS-weighted magnitudes directly comparable across groups. IS correction systematically upweights low-TD-error transitions (typical of success episodes where the network has mastered most subtasks) and downweights high-TD-error transitions (typical of failure episodes where learning is still active). The success IS-magnitude will tend to exceed the failure IS-magnitude as a mechanical consequence of the correction, not as a finding.

**Rule:** `gradient_magnitude_success_is` vs `gradient_magnitude_failure_is` must **not** be compared directly at a single checkpoint. Use `gradient_magnitude_success` vs `gradient_magnitude_failure` (uniform) for cross-group magnitude comparison. IS-weighted magnitudes are valid as **temporal trends within a single group** across checkpoints.

---

## 4. Diagnostic Funnel (reframed from v4)

Two cosine similarities reported per group (success, failure) per checkpoint:

```
cos(G_uniform, G_IS)    →  cos_uniform_is_{success,failure}
cos(G_IS, G_reward)     →  cos_is_reward_{success,failure}
```

`cos_uniform_is` quantifies how much the IS correction moves the gradient direction relative to the uniform baseline. `cos_is_reward` is a **finding metric** — its value characterises what PER is prioritising at this checkpoint. G_IS is the primary corrected estimate regardless of `cos_is_reward`.

### Interpretation table

| cos(G_uniform, G_IS) | cos(G_IS, G_reward) | Interpretation |
|:---:|:---:|---|
| High (> 0.8) | High (> 0.8) | IS correction is small; PER priorities nearly flat; reward magnitude correlates with priority; uniform analysis was sufficient |
| High (> 0.8) | Low | Correction is small; PER priorities not explained by reward magnitude at this stage |
| Low (< 0.8) | High (> 0.8) | Correction is material; PER priorities correlate with reward — achievement/death transitions drive learning |
| Low (< 0.8) | Low | Correction is material; PER emphasises zero-reward subtask prediction errors over reward transitions — report as finding |
| Low (< 0.8) | Negative | PER actively downweights mastered reward transitions; learning emphasis on zero-reward subtask errors — most informative case for the dissertation |

A low or negative `cos_is_reward` is not a problem with G_IS. It is a result about what PER is learning to prioritise in Crafter at this checkpoint.

---

## 5. Weight-Delta Empirical Validation (NEW in v5)

### Motivation

G_IS reconstructs IS weights from the frozen checkpoint network at analysis time. Training priorities were computed at each gradient step — potentially thousands of steps earlier (stale priorities, Limitation 1). There is no way to verify this reconstruction without the live replay buffer, which is not saved.

**The fix:** compare each offline gradient direction against the actual weight change Δθ = θ_{N+1} − θ_N between consecutive analyzed checkpoints. If `cos(G_IS, Δθ) > cos(G_uniform, Δθ)`, the IS reconstruction better predicted which direction weights actually moved — empirical validation without the buffer.

### Δθ definition

$$\Delta\theta = \theta_{N+1} - \theta_N \quad \text{(mu params only, online\_net state\_dict)}$$

where N and N+1 are consecutive analyzed checkpoints. Δθ covers all Adam updates between the two checkpoints (a span of `analyze_every × checkpoint_interval` training steps when no checkpoints are skipped; variable if `skip_existing` caused some to be skipped in `analyze_range.py`).

### Why sigma params are excluded

NoisyLinear stores four tensors per layer: `weight_mu`, `weight_sigma`, `bias_mu`, `bias_sigma`. In eval mode (used for offline analysis), NoisyLinear uses only `weight_mu` — sigma params have zero analytical gradient. But sigma params DO receive real weight updates during training (they control the noise scale). Including sigma params in Δθ would introduce a component absent from all gradient variants, creating systematic misalignment. All non-sigma params (mu params and Conv/Linear params) are included.

### Adam distortion caveat

Δθ accumulates many Adam gradient steps with momentum averaging. Even if G_IS perfectly reconstructed the training gradient direction, the cosine with the accumulated Δθ would be low. Interpret **relative** differences:

- `cos(G_IS, Δθ) > cos(G_uniform, Δθ)` → IS reconstruction is directionally superior
- `cos(G_IS, Δθ) < cos(G_uniform, Δθ)` → G_uniform better approximated training; treat G_uniform as primary for that checkpoint

### Training-stage tension

The metric is most diagnostic at mid-training (large Δθ, consistent gradient signal — relative ordering is reliable) but IS reconstruction is most accurate at late training (converged network, small priorities change slowly). At late training Δθ is tiny and near-orthogonal to all gradient variants — both cosines approach zero and their ordering becomes unreliable. This training-stage tension means the empirical validation signal is strongest precisely where the IS reconstruction quality is lower. Interpret accordingly: a relative win for G_IS at mid-training checkpoints provides stronger evidence than at late-training checkpoints.

### Temporal gradient mismatch

Gradients are computed at N+1's frozen weights, but the weight changes in Δθ were driven by intermediate network states from N to N+1. At late-training checkpoints where the network is mostly converged, the two network states are close. At early checkpoints the mismatch is larger. Absolute cosines are not meaningful; only relative differences within a checkpoint matter.

### Recovery path

If `cos(G_IS, Δθ) < cos(G_uniform, Δθ)` at most checkpoints, the IS reconstruction has not been validated and G_uniform should be treated as primary. The dissertation result is still valid — it simply reports uniform-weighted gradient analysis, as in v3.

### New JSON keys (6)

| Key | Definition | Notes |
|---|---|---|
| `cos_uniform_success_delta` | cos(G_uniform_success, Δθ_mu) | None for first checkpoint |
| `cos_is_success_delta` | cos(G_IS_success, Δθ_mu) | None for first checkpoint |
| `cos_reward_success_delta` | cos(G_reward_success, Δθ_mu) | None for first checkpoint |
| `cos_uniform_failure_delta` | cos(G_uniform_failure, Δθ_mu) | None for first checkpoint |
| `cos_is_failure_delta` | cos(G_IS_failure, Δθ_mu) | None for first checkpoint |
| `cos_reward_failure_delta` | cos(G_reward_failure, Δθ_mu) | None for first checkpoint |

Δθ_mu = {param_name: θ_{N+1}[param] − θ_N[param]} for all non-sigma params.

---

## 6. Computation Strategy

### Joint IS pass (Pass 0) — NEW in v5

Before any backward pass, a single forward-only pass (`torch.no_grad()`) runs over all episodes (success + failure combined) to compute per-transition losses and global IS weights. This replaces the separate within-group Pass 1 from v4.

### Three backward passes per episode (Pass 1, unchanged from v4)

For each episode, with the computation graph retained between backward calls:

```
zero_grad()
per_loss = forward(episode)                          # (T,) with gradient tracking

per_loss.mean().backward(retain_graph=True)
uniform_agg.accumulate()
zero_grad()

(w_is_ep * per_loss).mean().backward(retain_graph=True)
is_agg.accumulate()
zero_grad()

(w_rw_ep * per_loss).mean().backward()              # no retain_graph — free graph
rw_agg.accumulate()
zero_grad()
```

All three variants use `.mean()` (divide by T) for scale consistency.

### Weight-delta computation (Pass 2) — NEW in v5

After all backward passes (grad_s and grad_f still in memory), if `prev_checkpoint_path` is provided:

```
prev_sd = load_rainbow_nets(prev_checkpoint_path).online_net.state_dict()
curr_sd = online_net.state_dict()
Δθ = {k: curr_sd[k] - prev_sd[k] for k in non_sigma_keys}
cos_*_delta = cosine_similarity_flat(filtered_grad, Δθ)  # per variant, per group
```

**Compute cost:** weight-delta adds one checkpoint load and six cosine similarity computations per analysis. Negligible relative to the three-backward pass cost.

---

## 7. Checkpoint Lifecycle Change (train_and_analyze.py)

The previous lifecycle deleted checkpoint N immediately after its analysis. The new lifecycle:

1. Checkpoint N is analyzed with `prev_checkpoint_path = checkpoint_{N-1}`
2. Checkpoint N-1 is deleted (its delta has been computed)
3. Checkpoint N is kept until N+1's analysis completes

At most one extra `.pt` file (~60–120 MB) is on disk at any time. The final held checkpoint is deleted after training completes. `analyze_range.py` never deletes checkpoints, so no lifecycle changes are needed there.

---

## 8. Full JSON Output Metrics (v3 + v4 + v5)

### v3 keys (uniform baseline — all algorithms)

| Key | Description |
|-----|-------------|
| `opposition_score` | cos(G_uniform_success, G_uniform_failure) |
| `coherence_success` | Coherence within success group (uniform) |
| `coherence_failure` | Coherence within failure group (uniform) |
| `gradient_magnitude_success` | ‖G_uniform_success‖₂ |
| `gradient_magnitude_failure` | ‖G_uniform_failure‖₂ |
| `activation_separation` | Euclidean centroid distance |
| `activation_cosine_distance` | Cosine centroid distance |
| `cluster_stats` | UMAP+HDBSCAN cluster statistics |

### v4 keys (IS-corrected — Rainbow only)

| Key | Description |
|-----|-------------|
| `opposition_score_is` | cos(G_IS_success, G_IS_failure) |
| `coherence_success_is` | Coherence within success group (IS) |
| `coherence_failure_is` | Coherence within failure group (IS) |
| `gradient_magnitude_success_is` | ‖G_IS_success‖₂ |
| `gradient_magnitude_failure_is` | ‖G_IS_failure‖₂ |
| `coherence_success_reward` | Coherence within success group (reward proxy) |
| `coherence_failure_reward` | Coherence within failure group (reward proxy) |
| `cos_uniform_is_success` | cos(G_uniform, G_IS) for success group |
| `cos_uniform_is_failure` | cos(G_uniform, G_IS) for failure group |
| `cos_is_reward_success` | cos(G_IS, G_reward) for success group |
| `cos_is_reward_failure` | cos(G_IS, G_reward) for failure group |
| `beta_used` | Annealed β at this checkpoint's global_step |
| `n_transitions_success` | N used in joint IS normalisation (success episodes) |
| `n_transitions_failure` | N used in joint IS normalisation (failure episodes) |

### v5 keys (weight-delta validation — Rainbow only)

| Key | Definition | Notes |
|-----|------------|-------|
| `cos_uniform_success_delta` | cos(G_uniform_success, Δθ_mu) | None for first checkpoint |
| `cos_is_success_delta` | cos(G_IS_success, Δθ_mu) | None for first checkpoint |
| `cos_reward_success_delta` | cos(G_reward_success, Δθ_mu) | None for first checkpoint |
| `cos_uniform_failure_delta` | cos(G_uniform_failure, Δθ_mu) | None for first checkpoint |
| `cos_is_failure_delta` | cos(G_IS_failure, Δθ_mu) | None for first checkpoint |
| `cos_reward_failure_delta` | cos(G_reward_failure, Δθ_mu) | None for first checkpoint |

---

## 9. Limitations

1. **Stale priorities:** Analytical priorities reflect the network at checkpoint time; training priorities reflected the network at each gradient step (potentially many steps earlier). The approximation improves as training converges. The weight-delta comparison (Section 5) provides an empirical post-hoc check: if `cos(G_IS, Δθ) > cos(G_uniform, Δθ)` at mid-training checkpoints, the IS reconstruction is directionally validated.

2. **Max-priority initialisation:** New transitions receive `max_priority` regardless of TD error. Analytical priorities never apply this. Impact is largest for recently-collected transitions.

3. **Episode-level vs. buffer-level IS:** Training normalises priorities over the full buffer (N=500K). We normalise over the joint evaluation group (~25K–50K transitions). Absolute weight magnitudes differ, but relative ordering is preserved after max-normalisation. Gradient direction correction is valid; magnitude correction is not (not used for cross-group comparison, per Section 3).

4. **Reward proxy validity:** `|reward|` is a poor proxy for TD error when the agent has mastered reward-generating transitions (low TD error despite high reward) or when learning is driven by zero-reward subtask prediction (high TD error, low reward). The `cos_is_reward` metric directly quantifies this mismatch checkpoint by checkpoint.

5. **Gradient/delta temporal mismatch:** Gradients are computed at N+1's frozen weights, but Δθ was driven by N..N+1 intermediate network states. Relative comparison (G_IS vs G_uniform) is still valid since both use the same N+1 weights. Absolute cosines are not meaningful.

6. **Training-stage tension:** The weight-delta metric is most diagnostic at mid-training (large Δθ, reliable relative ordering) but G_IS is most accurate at late training (converged network, tiny Δθ, cosines → 0, ordering unreliable). These are inversely correlated — see Section 5 for interpretation guidance.
