# Gradient Analysis Methodology v4

Builds on v3. Two methodological changes: percentile partitioning and IS-corrected gradient weighting.

---

## Summary of Changes from v3

| Aspect | v3 | v4 |
|---|---|---|
| Partition threshold | Mean EPS (μ); all episodes assigned | Top/bottom K% by EPS; middle discarded |
| Gradient weighting (Rainbow) | Uniform mean | IS-corrected primary + reward proxy verification |
| Gradient variants reported | 1 (uniform) | 3 (uniform, IS-corrected, reward proxy) |
| New JSON metrics | — | `opposition_score_is`, `coherence_*_is`, `coherence_*_reward`, funnel cosines |

PPO analysis is unchanged. Changes apply to Rainbow only.

---

## 1. Partition Function: Percentile Split

### v3 definition (replaced)

```
μ = mean EPS across all evaluation episodes
Success = {ep : EPS(ep) ≥ μ}
Failure = {ep : EPS(ep) < μ}
```

**Problem:** when the agent is improving rapidly, μ shifts upward and can assign 80–90% of episodes to Failure. Both gradient estimates are polluted by near-average episodes that carry weak signal about either good or bad behaviour.

### v4 definition (primary)

```
K = percentile_x (default 25)
Success = top-K% of episodes ranked by EPS
Failure = bottom-K% of episodes ranked by EPS
Middle (100 - 2K)% = discarded
```

**Why this is better:**
- Groups have exactly equal and stable size at every checkpoint regardless of training stage
- The discarded middle removes ambiguous episodes from both gradient estimates
- The success/failure contrast is sharpest at the distributional extremes
- K=25 gives 125 episodes per group from 500 evaluation episodes — sufficient for reliable gradient aggregation

**Threshold reported in JSON:**
- `threshold_lower`: EPS value at the K-th percentile (boundary of failure group)
- `threshold_upper`: EPS value at the (100−K)-th percentile (boundary of success group)

---

## 2. Gradient Framework: Three Variants (Rainbow only)

### Background

Rainbow's training loss applies PER importance-sampling weights:

```
loss = (1/B) Σ_{i ∈ batch} w_i · L_i
```

where `L_i = -Σ_a m_a log p(s_i, a_i; θ)_a` is the per-transition distributional Bellman cross-entropy and `w_i` is the IS weight. Our offline analytical gradient uses `.mean()` — a uniform approximation — because PER priorities are not persisted to disk. v4 corrects this analytically.

### Variant 1 — G_uniform (baseline, matches v3)

```
loss_uniform = (1/T) Σ_t L_t
```

All transitions contribute equally. This is what the current code computes.

### Variant 2 — G_IS (primary estimate, new)

Analytically reconstructs the IS-weighted gradient using the frozen network's own distributional Bellman loss as a proxy for the per-transition priorities that PER would assign.

**IS weight derivation:**

```
priority_i = L_i ^ α                       α = priority_exponent (default 0.5)
P(i)       = priority_i / Σ_{k ∈ group} priority_k
w_i        = (1 / (N · P(i))) ^ β
w_i        = w_i / max(w)                  normalize by batch max (matches training)

N   = total transitions in the group (success or failure)
β   = β_start + (1 − β_start) · clip((global_step − learn_start) / (T_max − learn_start), 0, 1)
```

β is the exact annealed IS exponent at the checkpoint's training step. At early checkpoints β ≈ 0.4 (partial correction); at late checkpoints β ≈ 1.0 (full correction).

```
loss_IS = (1/T) Σ_t w_t · L_t
```

**Key approximation:** analytical priorities are computed at checkpoint evaluation time using the frozen network. Training priorities were computed at each gradient step and may reflect earlier network states (PER stores stale priorities; new transitions receive max priority). At late-training checkpoints where the network is mostly converged, the two are close. This approximation is acknowledged as a limitation.

### Variant 3 — G_reward (cheap verification proxy, new)

```
w_t = |r_t| + ε      (ε = 0.01)
w_t = w_t / Σ w_t    (normalize to sum 1 per episode)

loss_reward = (1/T) Σ_t w_t · L_t
```

High-reward transitions (achievement unlocks: r=+1, death: r=−1) receive more weight. This is a zero-information-cost proxy for IS: if G_IS and G_reward agree, the IS correction is supported by an independent weighting scheme. If they disagree, PER's priorities are driven by zero-reward prediction errors — itself a meaningful finding.

---

## 3. Diagnostic Funnel

Two cosine similarities reported per group (success, failure) per checkpoint:

```
cos(G_uniform, G_IS)    →  cos_uniform_is_{success,failure}
cos(G_IS, G_reward)     →  cos_is_reward_{success,failure}
```

`cos_uniform_is` quantifies how much the IS correction moves the gradient direction relative to the uniform baseline. `cos_is_reward` is a **finding metric**, not a verification gate — its value characterises the relationship between reward magnitude and PER's learning priority at this checkpoint. G_IS is the primary corrected estimate regardless of `cos_is_reward`.

**Interpretation table:**

| cos(G_uniform, G_IS) | cos(G_IS, G_reward) | Interpretation |
|---|---|---|
| High (>0.8) | High (>0.8) | IS correction is small; reward magnitude correlates with priority; uniform analysis was sufficient |
| High (>0.8) | Low | Correction is small; PER priorities are not well-explained by reward magnitude at this stage |
| Low (<0.8) | High (>0.8) | Correction is material; PER priorities correlate with reward — achievement/death transitions drive learning |
| Low (<0.8) | Low | Correction is material; PER emphasises zero-reward subtask prediction errors over reward transitions — report as finding |
| Low (<0.8) | Negative | PER actively downweights mastered reward transitions; learning emphasis is on zero-reward subtask errors — most informative case for the dissertation |

A low or negative `cos_is_reward` is not a problem with G_IS. It is a result about what PER is learning to prioritise in Crafter at this checkpoint.

---

## 4. Computation Strategy

### Two-pass design

Globally normalised IS weights require knowing the priority sum across all group transitions before any backward pass. This necessitates two passes:

**Pass 1 — forward only (`torch.no_grad()`):**
For each episode in the group, compute per-transition losses `L_t` via a full forward pass (no gradient tracking). Concatenate all `L_t` into `all_losses` (N_total,). Compute global IS weights `w_is` (N_total,). Split back to per-episode tensors.

**Pass 2 — three backward passes per episode:**
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

All three variants use `.mean()` (divide by T) for scale consistency. `OnlineGradientAggregator` increments `count` once per episode for all three, so `mean_gradient()` gives a consistent per-transition-scale mean across all variants.

**Compute cost:** ~3× the original single-backward design. Accepted since analysis runs are offline and infrequent.

---

## 5. New JSON Output Keys (Rainbow only)

| Key | Source | Notes |
|---|---|---|
| `opposition_score_is` | G_IS success vs failure | IS-corrected version of opposition_score |
| `coherence_success_is` | G_IS batch_grads, success | IS-corrected coherence |
| `coherence_failure_is` | G_IS batch_grads, failure | IS-corrected coherence |
| `coherence_success_reward` | G_reward batch_grads, success | Reward-proxy coherence |
| `coherence_failure_reward` | G_reward batch_grads, failure | Reward-proxy coherence |
| `gradient_magnitude_success_is` | G_IS raw mean, success | Same scale as uniform magnitude |
| `gradient_magnitude_failure_is` | G_IS raw mean, failure | Same scale as uniform magnitude |
| `cos_uniform_is_success` | flat cosine | How much IS moves success gradient |
| `cos_uniform_is_failure` | flat cosine | How much IS moves failure gradient |
| `cos_is_reward_success` | flat cosine | Proxy agreement for success |
| `cos_is_reward_failure` | flat cosine | Proxy agreement for failure |
| `beta_used` | annealed β at global_step | Reference: how strong the IS correction was |
| `n_transitions_success` | Σ episode lengths | N used in IS normalisation |
| `n_transitions_failure` | Σ episode lengths | N used in IS normalisation |

All existing v3 keys (`opposition_score`, `coherence_success`, `coherence_failure`, etc.) are retained unchanged and continue to report the uniform-weighted baseline.

**Cross-group magnitude comparison:** `gradient_magnitude_success_is` and `gradient_magnitude_failure_is` must **not** be compared directly against each other. IS correction systematically upweights low-TD-error transitions (typical of success episodes) and downweights high-TD-error transitions (typical of failure episodes), so the success IS-magnitude will tend to exceed the failure IS-magnitude as a mechanical consequence of the correction, not as a finding. For cross-group magnitude comparison use `gradient_magnitude_success` vs `gradient_magnitude_failure` (uniform). The IS-weighted magnitudes are meaningful as **temporal trends within a group** across checkpoints (is the IS-corrected success gradient growing over training?) but not as a cross-group comparison at a single checkpoint.

---

## 6. Limitations

1. **Stale priorities:** Analytical priorities reflect the network at checkpoint time; training priorities reflected the network at the time each transition was last sampled (which may be many gradient steps earlier). The approximation improves as training converges.

2. **Max-priority initialisation:** New transitions in PER receive `max_priority`, which inflates their effective sampling probability regardless of TD error. Our analytical priorities never apply this initialisation. Impact is largest for recently-collected transitions.

3. **Episode-level vs. buffer-level IS:** Training normalises priorities over the full buffer (N=500K). We normalise over the group (N ≈ 12,000–25,000 transitions from evaluation episodes). This changes absolute weight magnitudes but not relative direction after max-normalisation.

4. **Reward proxy validity:** `|reward|` is a poor proxy for TD error when the agent has mastered reward-generating transitions (low TD error despite high reward) or when learning is driven by zero-reward subtask prediction (high TD error, low reward). The `cos_is_reward` metric directly quantifies this mismatch checkpoint by checkpoint.
