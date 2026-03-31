# PER Gradient Bias: Problem, Analysis, and Correction

*A focused technical document on the methodological problem with offline gradient analysis of Rainbow DQN, and the v4 fix. Target audience: RL researcher familiar with DQN and importance sampling.*

---

## 1. Background: What PER Does During Training

Rainbow DQN uses **Prioritized Experience Replay (PER)**. Instead of sampling transitions uniformly from the replay buffer, PER samples proportionally to each transition's TD-error magnitude:

$$P(i) = \frac{\text{priority}_i}{\sum_k \text{priority}_k}, \qquad \text{priority}_i = |\delta_i|^\alpha$$

where $|\delta_i|$ is the TD error for transition $i$ and $\alpha = 0.5$ controls the degree of prioritisation (0 = uniform, 1 = fully greedy).

Because this biases the sampling distribution away from uniform, each gradient update is **not** a fair estimate of the true expected gradient over the data distribution. To correct for this, PER applies **importance-sampling (IS) weights** to each transition before averaging the loss:

$$\mathcal{L}_{\text{training}} = \frac{1}{B} \sum_{i \in \text{batch}} w_i \mathcal{L}_i$$

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\!\beta} \cdot \frac{1}{\max_j w_j}$$

where $N$ is the buffer size, $B$ is the batch size, and $\beta$ is a bias-correction exponent annealed from $\beta_0 = 0.4 \to 1.0$ over training (full unbiased correction at convergence). The max-normalisation keeps weights in $(0, 1]$.

**In short:** Rainbow's gradient at every training step is a reward-weighted, TD-error-prioritised, IS-corrected gradient — not a uniform mean.

---

## 2. The Problem: Offline Analysis Ignored All of This

The offline gradient analysis (v3 and earlier) computed:

$$G_{\text{uniform}} = \nabla_\theta \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_t$$

where the sum runs over all transitions in a group (success or failure episodes). This is a **uniform mean** — every transition contributes equally.

This is not the gradient Rainbow trained on. Rainbow never computed a uniform mean in its life.

### Why this matters

The whole point of the gradient analysis is to measure the *character of the learning signal* that Rainbow receives from success versus failure episodes. Specifically:

- **Opposition score:** does the success gradient point in the opposite direction to the failure gradient?
- **Coherence:** do multiple success episodes agree on gradient direction?

Both of these metrics are computed on $G_{\text{uniform}}$. But if PER is systematically upweighting certain transitions within success episodes (e.g., achievement-unlock transitions with high TD error) and downweighting others (e.g., mastered subtask transitions with low TD error), then the true training gradient has a different direction from $G_{\text{uniform}}$.

**Concretely:** suppose PER has learned to prioritise zero-reward preparatory transitions (high TD error because the agent is still learning fine-grained subtask prediction) over positive-reward achievement transitions (low TD error because those are now well-predicted). Then:
- $G_{\text{uniform}}$ is pulled strongly toward achievement-unlock frames (because they are salient, not because they are prioritised)
- The *actual* training gradient is pulled toward preparatory transitions

These could point in different directions. The v3 analysis would report the direction of $G_{\text{uniform}}$ and attribute it to Rainbow's learning signal — but Rainbow never saw that gradient.

---

## 3. The Fix: Analytical IS Weight Reconstruction (G_IS)

Training-time PER priorities are not saved to disk. They are computed dynamically as TD errors, updated per transition per sample, and stored only in the live segment tree. By the time a checkpoint is saved, the priorities that drove training are gone.

**The approximation:** use the frozen checkpoint network's own Bellman losses as proxies for the priorities that PER would have assigned.

$$\widehat{\text{priority}}_i = \mathcal{L}_i^\alpha$$

where $\mathcal{L}_i$ is the distributional cross-entropy loss computed by the frozen network on transition $i$ (the same loss used during training). The intuition is that, near convergence, the network's prediction errors at analysis time closely reflect the errors it had during its last few updates on each transition.

**Full derivation of analytical IS weights:**

$$\widehat{P}(i) = \frac{\widehat{\text{priority}}_i}{\sum_{k \in \text{group}} \widehat{\text{priority}}_k}$$

$$\hat{w}_i = \left(\frac{1}{N \cdot \widehat{P}(i)}\right)^{\!\beta} \cdot \frac{1}{\max_j \hat{w}_j}$$

where:
- $N$ = total transitions in the group (success or failure evaluation episodes, ~12k–25k)
- $\beta$ = the exact annealed value at the checkpoint's `global_step`:

$$\beta(\text{step}) = \beta_0 + (1 - \beta_0) \cdot \text{clip}\!\left(\frac{\text{step} - \text{learn\_start}}{T_{\max} - \text{learn\_start}},\; 0,\; 1\right)$$

with $\beta_0 = 0.4$, $\text{learn\_start} = 20{,}000$, $T_{\max} = 10{,}000{,}000$. This matches the exact β Rainbow was using at the time of the checkpoint.

The IS-corrected gradient estimate:

$$G_{\text{IS}} = \nabla_\theta \frac{1}{T} \sum_{t} \hat{w}_t \mathcal{L}_t$$

This is the primary corrected gradient estimate (v4 onwards).

---

## 4. The Diagnostic Proxy (G_reward)

IS weight reconstruction is an approximation. To verify whether the correction is principled, a second independent weighting scheme is computed at zero additional information cost:

$$w_t^{\text{reward}} = |r_t| + \varepsilon, \qquad w_t^{\text{reward}} \leftarrow w_t^{\text{reward}} / \sum_t w_t^{\text{reward}}$$

This upweights transitions with high absolute reward (achievement unlocks: $r = +1$; deaths: $r = -1$) and downweights zero-reward transitions.

The reward proxy is **not** the same as IS correction. It is a *prediction* about what IS correction would do, based on the assumption that "high reward = high TD error = high PER priority." This assumption is valid early in training but breaks down as the agent masters reward-generating transitions (TD error → 0 despite high reward).

The reward proxy gradient:

$$G_{\text{reward}} = \nabla_\theta \frac{1}{T} \sum_{t} w_t^{\text{reward}} \mathcal{L}_t$$

---

## 5. The Diagnostic Funnel

Two scalar cosine similarities are reported per group (success and failure) at each checkpoint:

$$\cos(G_{\text{uniform}},\; G_{\text{IS}}) \qquad \text{and} \qquad \cos(G_{\text{IS}},\; G_{\text{reward}})$$

These are **diagnostic metrics**, not validation gates. $G_{\text{IS}}$ is the primary estimate regardless of what the funnel shows — the funnel tells you what the correction means and how to interpret it.

### Interpretation Table

| $\cos(G_{\text{uniform}}, G_{\text{IS}})$ | $\cos(G_{\text{IS}}, G_{\text{reward}})$ | Interpretation |
|:---:|:---:|---|
| High (> 0.8) | High (> 0.8) | IS correction is small; PER priorities were nearly flat; reward magnitude correlated with priority; uniform analysis was sufficient |
| High (> 0.8) | Low | Correction is small; but PER priorities are not explained by reward magnitude — PER is prioritising something other than reward events |
| Low (< 0.8) | High (> 0.8) | Correction is material; PER priorities correlate with reward — achievement/death transitions dominate the gradient |
| Low (< 0.8) | Low | Correction is material; PER priorities driven by zero-reward subtask prediction errors — most informative case for the dissertation |
| Low (< 0.8) | Negative | PER is actively *downweighting* mastered reward transitions; learning emphasis is entirely on zero-reward subtask errors |

**The most theoretically interesting case is the bottom row.** A negative $\cos(G_{\text{IS}}, G_{\text{reward}})$ would mean that Rainbow's PER — trained on Crafter's sparse achievement rewards — has converged to a regime where it ignores the sparse reward events (low TD error, mastered) and focusses entirely on zero-reward preparatory transitions (high TD error, still learning). This would be direct evidence that Rainbow is learning the *structural* task graph, not just chasing rewards.

---

## 6. Two-Pass Computation Strategy

Global IS normalisation requires knowing $\sum_k \widehat{\text{priority}}_k$ over *all group transitions* before computing any individual weight. This forces a two-pass design:

### Pass 1 — Forward Only (`torch.no_grad()`)

For each episode in the group, run a full forward pass to compute per-transition losses $\mathcal{L}_t$, but do not track gradients. Concatenate all losses into a single tensor of shape $(N_{\text{total}},)$. Compute global IS weights $\hat{w}$ over the entire group. Split back into per-episode weight tensors.

**Cost:** one forward pass per episode, no backward. Cheap.

### Pass 2 — Three Backward Passes per Episode

For each episode, with the computation graph retained between backward calls:

1. Forward pass (with gradient tracking): compute per-transition losses $(T,)$
2. **Backward 1:** `(per_loss.mean()).backward(retain_graph=True)` → accumulate uniform gradient
3. Zero gradients
4. **Backward 2:** `((w_IS_ep * per_loss).mean()).backward(retain_graph=True)` → accumulate IS gradient
5. Zero gradients
6. **Backward 3:** `((w_rw_ep * per_loss).mean()).backward()` → accumulate reward gradient
7. Zero gradients, free computation graph

**Cost:** ~3× the original single-backward design. Accepted because analysis runs are offline and infrequent.

All three variants use `.mean()` (divide by $T$) for scale consistency. The `OnlineGradientAggregator` increments its count once per episode for all three variants, so `mean_gradient()` gives a consistent per-episode-scale mean across variants — enabling direct magnitude comparison between $G_{\text{uniform}}$ and $G_{\text{IS}}$ within the same group (though not across groups, see Section 7).

---

## 7. New Output Metrics (v4 JSON Keys)

All existing v3 keys are retained. New keys added for Rainbow only:

| Key | Description |
|-----|-------------|
| `opposition_score_is` | $\cos(G_{\text{IS, success}},\; G_{\text{IS, failure}})$ — IS-corrected opposition |
| `coherence_success_is` | Coherence within success group using IS-weighted gradients |
| `coherence_failure_is` | Coherence within failure group using IS-weighted gradients |
| `gradient_magnitude_success_is` | $\|G_{\text{IS, success}}\|_2$ |
| `gradient_magnitude_failure_is` | $\|G_{\text{IS, failure}}\|_2$ |
| `coherence_success_reward` | Coherence within success group using reward-proxy gradients |
| `coherence_failure_reward` | Coherence within failure group using reward-proxy gradients |
| `cos_uniform_is_success` | $\cos(G_{\text{uniform}}, G_{\text{IS}})$ for success group |
| `cos_uniform_is_failure` | $\cos(G_{\text{uniform}}, G_{\text{IS}})$ for failure group |
| `cos_is_reward_success` | $\cos(G_{\text{IS}}, G_{\text{reward}})$ for success group |
| `cos_is_reward_failure` | $\cos(G_{\text{IS}}, G_{\text{reward}})$ for failure group |
| `beta_used` | Exact annealed β at this checkpoint's `global_step` — reference for IS correction strength |
| `n_transitions_success` | $N$ used in IS normalisation for success group |
| `n_transitions_failure` | $N$ used in IS normalisation for failure group |

### Cross-Group Magnitude Warning

`gradient_magnitude_success_is` vs. `gradient_magnitude_failure_is` should **not** be compared directly. IS correction systematically upweights low-TD-error transitions (typical of well-performing success episodes where the network has mastered most subtasks) and downweights high-TD-error transitions (typical of failure episodes where learning is still active). This means the IS-corrected success magnitude will tend to exceed the IS-corrected failure magnitude as a mechanical artefact of the correction — not as a finding.

For cross-group magnitude comparison, use `gradient_magnitude_success` vs. `gradient_magnitude_failure` (the uniform baseline). IS-weighted magnitudes are valid only for *temporal trends within a single group* across checkpoints.

---

## 8. Code Changes Required

Five files require modification (all changes are additive — no existing behaviour is removed):

| File | Change |
|------|--------|
| `rainbow/agent.py` | Add `priority_exponent`, `priority_weight`, `T_max`, `learn_start` to `args_dict` in `save()` |
| `rainbow/sampling.py` | Return `global_step` from `load_rainbow_nets()` (5-tuple instead of 4-tuple); add new fields to `_build_args_from_dict()` |
| `rainbow/gradients.py` | Add `_forward_per_loss()`, `_compute_is_weights()`, `_compute_reward_weights()` helpers; refactor `compute_group_gradient_with_coherence()` to two-pass, three-backward design; change return type from 3-tuple to dict |
| `analyze_checkpoint.py` | Unpack 5-tuple from `load_rainbow_nets`; pass `global_step` to gradient functions; unpack dict return; extend `_build_result()` with new IS/reward metrics |
| `rainbow/train_and_analyze.py` | Change default `split_mode` from `"eps"` to `"percentile"` |

---

## 9. Acknowledged Limitations

These limitations are documented explicitly in `Gradient Analysis v4.md`:

### 1. Stale Priorities

Analytical priorities are computed using the frozen checkpoint network. Training priorities were computed at each gradient step using the network as it was *at that point in training* — which may be many gradient steps earlier. PER's segment tree stores stale priorities and updates them lazily (only when a transition is re-sampled).

**Impact:** At early checkpoints where the network changes rapidly, the analytical proxy can substantially diverge from the true training priorities. At late checkpoints where the network is converged, the proxy is close.

**Mitigation:** The proxy is most valid at the checkpoints that matter most for the dissertation (late training, where meaningful opposition/coherence signals are expected).

**Empirical check (v5):** Section 11 introduces a weight-delta comparison that provides a post-hoc empirical check without requiring the live replay buffer. If `cos(G_IS, Δθ) > cos(G_uniform, Δθ)` at mid-training checkpoints, the IS reconstruction is directionally validated. Note the training-stage tension: the check is most reliable at mid-training (large Δθ), but G_IS is most accurate at late training (converged network, Δθ tiny). If the check fails, G_uniform becomes primary — the result is still valid.

### 2. Max-Priority Initialisation

When a new transition is first inserted into the replay buffer, PER assigns it `max_priority` (the highest priority currently in the buffer), regardless of its TD error. This ensures new transitions get sampled at least once. The analytical reconstruction never applies this initialisation — it computes priorities from losses directly.

**Impact:** Recently-collected transitions (those from the last few thousand steps before the checkpoint) are systematically underweighted in the analytical reconstruction relative to training.

### 3. Episode-Level vs. Buffer-Level IS Normalisation

During training, $N$ in the IS weight formula is the full buffer size (500,000). In the analytical reconstruction, $N$ is the number of transitions in the evaluation group (~12,000–25,000). The absolute magnitudes of $\hat{w}_i$ are therefore different. However, because weights are then max-normalised (capped at 1.0), the *relative* ordering of weights within the group is preserved — which is what determines gradient direction. The correction to gradient direction is valid; the correction to gradient magnitude is not (and is not used for cross-group comparison, per Section 7).

### 4. Reward Proxy Validity Changes Over Training

`G_reward` is a useful proxy for `G_IS` only when high-reward transitions also have high TD errors. This is true early in training (the agent does not yet predict achievement outcomes well) but breaks down as training progresses:

- **Mastered reward transitions:** the network predicts achievement-unlock values well → TD error → 0 → PER priority → 0, even though $|r| = 1$. G_reward upweights these; G_IS downweights them.
- **Active zero-reward subtask learning:** fine-grained prediction of tool use or material collection produces high TD errors at zero reward. G_IS upweights these; G_reward ignores them.

`cos_is_reward` directly quantifies this mismatch at each checkpoint. A declining `cos_is_reward` over training is expected and interpretable — it shows the transition from reward-aligned priorities to structure-aligned priorities.

---

## 10. Summary

The core issue is simple: **Rainbow trained on a weighted loss; we were analysing an unweighted one.**

The fix reconstructs the weights analytically from the frozen network's own prediction errors, applies the exact β value that Rainbow was using at the checkpoint's training step, and normalises globally over the group — matching the structural form of PER IS weighting as closely as possible without access to the live priority queue.

The result is three gradient estimates per group at each checkpoint:

- **G_uniform** — what v3 computed; the baseline; wrong in principle but useful for comparison
- **G_IS** — the principled corrected estimate; primary
- **G_reward** — an independent verification proxy; its agreement or disagreement with G_IS is itself a finding about what PER is prioritising in Crafter

The diagnostic funnel (two cosine similarities) translates these three estimates into an interpretable statement about the character of Rainbow's learning signal at each stage of training.

---

## 11. Weight-Delta Empirical Validation (v5)

### Motivation

Section 9, Limitation 1 (Stale Priorities) identifies that G_IS reconstructs weights from the frozen checkpoint network rather than the live training network. This is unverifiable without the replay buffer. The weight-delta comparison provides a post-hoc empirical check.

### What Δθ Represents

$$\Delta\theta = \theta_{N+1} - \theta_N$$

where N and N+1 are consecutive analyzed checkpoints (online_net mu params only — sigma params excluded, see below). Δθ is the **cumulative** result of all Adam gradient updates between the two analysis points. It is not a single gradient step.

### Why Sigma Params are Excluded

NoisyLinear stores `weight_mu`, `weight_sigma`, `bias_mu`, `bias_sigma` per layer. In eval mode (used for offline analysis), sigma params have zero analytical gradient — noise is not sampled. But sigma params DO receive real weight updates during training (they control noise scale). Including them in Δθ would add a component absent from all gradient variants, creating systematic misalignment. Only non-sigma params (mu params and Conv/Linear params) are used.

### The Empirical Check

$$\cos(G_{\text{IS}}, \Delta\theta) \stackrel{?}{>} \cos(G_{\text{uniform}}, \Delta\theta)$$

If G_IS better predicts the direction weights actually moved, the IS reconstruction is directionally validated. Both cosines are expected to be low (Adam distortion); only the relative ordering matters.

### Adam Distortion

Adam momentum accumulates many gradient steps between checkpoints. Even a perfect gradient reconstruction would yield a low cosine against the cumulative Δθ. Do not interpret absolute cosines — only whether G_IS > G_uniform in relative terms.

### Training-Stage Tension

The check is most reliable at **mid-training** where Δθ is large and the signal-to-noise ratio is high. At **late training**, Δθ is tiny (near convergence) and near-orthogonal to all gradient variants — both cosines approach zero and their relative ordering becomes unreliable noise. This is the opposite of where G_IS is most accurate (late training, converged network). Interpret validation as strongest at mid-training checkpoints.

### Recovery Path

If `cos(G_IS, Δθ) < cos(G_uniform, Δθ)` at most checkpoints, the IS reconstruction is not validated. In this case, treat G_uniform as primary and report accordingly. The dissertation result is still valid — it simply uses uniform-weighted gradient analysis (as in v3), which is the principled baseline for comparison with PPO.

### New JSON Keys (6)

| Key | Definition |
|-----|------------|
| `cos_uniform_success_delta` | cos(G_uniform_success, Δθ_mu) — None for first checkpoint |
| `cos_is_success_delta` | cos(G_IS_success, Δθ_mu) — None for first checkpoint |
| `cos_reward_success_delta` | cos(G_reward_success, Δθ_mu) — None for first checkpoint |
| `cos_uniform_failure_delta` | cos(G_uniform_failure, Δθ_mu) — None for first checkpoint |
| `cos_is_failure_delta` | cos(G_IS_failure, Δθ_mu) — None for first checkpoint |
| `cos_reward_failure_delta` | cos(G_reward_failure, Δθ_mu) — None for first checkpoint |
