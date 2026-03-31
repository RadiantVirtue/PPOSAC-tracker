# Dissertation Structure - v2

## 3. Methodology

**Job of this section:** The lit review found the gaps. This section builds the analytical framework - the tools, the environment, the metrics - so that when the research questions arrive in Section 4, every term in them is already grounded. Audience: technical layman with RL knowledge.

---

### 3.1 Gradient Analysis as the Primary Method

- Keep short - the lit review already made this case (Section 5.1: saliency and Shapley explain the *product* of learning, not the process)
- The question is about what happens *during* training, so the gradient is the right thing to look at
- Methodological precedent: Szolnoky et al. (2022) use cosine similarity between gradients (Model Gradient Similarity) to check if two inputs push the network the same way. This framework extends that from pairwise inputs to partitioned outcome groups - instead of "do these two inputs produce similar gradients?" we're asking "do successful and failed episodes, as groups, push the policy in different directions?"
- RSA complements this, doesn't compete with it:
  - Gradients = direction of the learning signal
  - RSA = whether that signal builds structured representations
  - Together they connect process to product, which neither does alowne

---

### 3.2 Environments

**Crafter (primary testbed):**

- Justify Crafter as the testbed
- Key argument: the hierarchical achievement chain (wood → stone → iron → diamond) is a natural stress test for credit assignment
  - Big gap between *when an action is taken* and *when it pays off*
  - Directly exposes the GAE temporal heuristic and Bellman backup weaknesses from the lit review (Section 3.1, the ore/smelt example)

> **[Figure opportunity]** Crafter's achievement dependency tree -  full chain from basic survival to diamond. Annotate with approximate step distances between cause and effect. Core environmental justification, make it visual. draw.io vector.

**Minigrid (preliminary validation):**

- Shorter horizon, simpler - used to check the implementation works
- Too simple to really stress credit assignment
- Purely methodological role: confirms the pipeline before committing compute to Crafter

---

### 3.3 Algorithms

Keep short - lit review did the heavy lifting - PPO and Rainbow are the canonical deployed representatives of on-policy and off-policy families. PPO represents policy optimization, while Rainbow represents state-of-the-art value-based discrete RL. - Studying algorithms as actually used in practice, which rules out the learned credit assignment extensions (DAE, RUDDER, HCA) already dismissed in the review (Sections 3.2, 4.2)

---
### 3.4 Experimental Setup

- Shared infrastructure: Crafter config (observation space, action space, episode length, reward structure), hardware, training schedule
- 3M steps / every 50k (~60 analysis points)

**EPS - universal performance measure for both algorithms:**

$$EPS = \text{achievements} + 0.9 \cdot \frac{\text{materials\_acquired}}{\text{materials\_needed}}$$

*Materials needed = for next achievement. Use closest achievement.*

- The 0.9 multiplier matters:
  - Without it, collecting all required materials without completing the achievement scores the same as completing it
  - That's a misrepresentation
  - Partial progress counts, but can never be mistaken for completion
  - Matters in Crafter where most episodes end mid-chain
- Used both for partition thresholds and individual episode evaluation

---

### 3.5 The Partitioning Strategy

Needs more space than the other subsections - readers haven't seen the design process. The dual-axis plots in Section 6.2 overlay partition-derived metrics against return, which shows whether this strategy actually produces meaningful signals across training.

**The naive approach - fixed threshold:**

- Pick a global reference and calibrate once
- Breaks immediately: what counts as a good episode changes as the agent improves
- A threshold from early training labels nearly everything as success by the end
- Very early in training there might not be episodes on both sides at all

**The natural fix - mean/median split:**

- Use the checkpoint's own mean EPS as the boundary
- Adapts as the agent learns, always gives episodes on both sides
- Problem: forces a binary with no tolerance for ambiguity
  - An episode at 80% of the best run gets labelled Failure for falling below average
  - The Failure pool fills up with mid-range episodes that aren't meaningfully different from the Success pool
  - Their gradients end up looking similar -  the signal of genuine failure gets buried
  - You can't really defend the claim that a run just below average belongs in the same category as a genuinely poor run

**The chosen approach -  quartile split:**

	- Top 25% by EPS = Success
	- Bottom 25% by EPS = Failure
- Middle 50% thrown out
- Maximises contrast between groups
- The comparison only reflects episodes that are *unambiguously* strong or weak at that checkpoint
- Yes, half the data gets discarded -  worth it for clean signal, and there are enough evaluation episodes to absorb the loss

> **[Figure opportunity]** Diagram: the three partitioning strategies on the same hypothetical EPS distribution. Where the splits fall, which episodes end up where, the "ambiguous middle" the quartile approach discards. draw.io or Inkscape, vector.

---

### 3.6 Metrics

- Walk through each metric: what it measures and *why it matters* 
- Main point: Opposition Score is the headline metric but not sufficient on its own
- Coherence disambiguates this 
- Activation Separation and Cosine Distance check whether gradient-level differences actually manifest in the network's representations. 
- **Crucially, flag the dimensionality mismatch here:** PPO's encoder produces a 64-dimensional representation, while Rainbow's flattened convolutional output is 1024-dimensional. 
- Euclidean distance is highly sensitive to this volume difference, meaning Cosine Distance is the primary reliable metric for cross-algorithm representational comparisons.
- RSA Alignment checks if those representations are semantically structured, not just separated

| Metric                    | Scope          | Description                                                     | Interpretation                                                                                              |
| :------------------------ | :------------- | :-------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Opposition Score**      | PPO vs Rainbow | Cosine similarity between ∇_Success and ∇_Failure               | **-1.0:** Perfect distinction (opposites). **0.0:** Orthogonal (unrelated). **+1.0:** Confused (identical). |
| **Coherence**             | PPO vs Rainbow | Average cosine similarity *within* the Success gradient batch   | High = consistent "theory" of success.                                                                      |
| **Gradient Magnitude**    | Internal       | L2 norm of gradient vector                                      | Track signal fading over time. *Don't compare across algorithms.*                                           |
| **Activation Separation** | PPO vs Rainbow | Euclidean distance between Success/Failure activation centroids | How distinct are internal representations of winning vs. losing?                                            |
| **Cosine Distance**       | PPO vs Rainbow | Cosine distance between Success/Failure activation centroids    | Directional divergence in representation space, magnitude-independent.                                      |
| **RSA Alignment**         | PPO vs Rainbow | Correlation of RDM with ground-truth functional hierarchy       | Does the agent understand that Wood and Stone are similar (materials)?                                      |

- Reading order: Opposition Score → Coherence → Activation Separation + Cosine Distance → RSA Alignment

> **[Figure opportunity]** Flowchart showing the intended reading order and what each metric rules in/out. Opposition Score → Coherence → Activation + Cosine Separation → RSA Alignment, each node annotated with its interpretive role. draw.io vector.

---

### 3.7 Hyperparameters and Tuning

#### 3.7.1 PPO Config

- State hyperparameters: clipping epsilon, GAE lambda and gamma, entropy coefficient, network architecture, optimiser, learning rate schedule
- Flag lambda explicitly -  given the GAE critique in the lit review (Section 3.1), its value is theoretically significant. It determines the temporal decay rate that the whole analysis is ultimately probing

#### 3.7.2 Rainbow Config
- Distributional Q-learning (C51): 51 atoms over $V_{min} = -10$ to $V_{max} = 10$, resolution $\Delta z = 0.4$ — fine enough for Crafter's reward structure.
- Exploration: NoisyLinear layers ($\sigma_0 = 0.1$) replacing $\varepsilon$-greedy. In eval mode, NoisyLinear is deterministic (uses $\mu$ weights only; $\sigma$ parameters excluded from gradient and weight-delta analysis).
- Replay: 500K capacity, Prioritized Experience Replay (PER) with $\alpha = 0.5$, $\beta$ annealed from 0.4 to 1.0.
- **Flag PER explicitly:** prioritising high-TD-error transitions introduces an implicit form of credit assignment — experiences the model finds surprising receive disproportionate influence. This is the core empirical target of RQ2.
- Temporal: 3-step returns. Extends the credit horizon beyond single-step TD; spans roughly one Crafter action decision window.
- Target network updated every 8,000 steps (hard copy). Online and target networks both saved at each checkpoint — needed for offline distributional Bellman loss reconstruction during gradient analysis.

> **[Table]** PPO and Rainbow hyperparameters side by side.

---

### 3.8 Baselines

- Clarify what "baselines" means here: reference measurements at initialisation and random-policy level, *not* competing algorithms
- Purpose: anchor the gradient metrics
  - Opposition Score and Coherence have no intrinsic scale
  - "High coherence" means nothing without knowing what an untrained network produces
- Cover: random policy gradient baselines, initialisation checkpoint metrics, published Crafter benchmarks for contextualising learning curves

---

### 3.9 PPO: Checkpointing and Evaluation

#### 3.9.1 Checkpointing Strategy

- Two triggers:
  - Regular interval (every 50k steps)
  - Achievement-triggered: save whenever the agent earns a new achievement for the first time
- Storage is cheap, missing a transition point isn't
- Better to have redundant checkpoints than to reconstruct dynamics by interpolation
- Each checkpoint: policy network weights + value function weights

#### 3.9.2 Evaluation and Quartile Partitioning

- 1,000 evaluation episodes at each checkpoint, frozen policy, no updates
- Record (State, Action, Reward) tuples and Total Return
- Compute EPS per episode
- Quartile boundaries set dynamically at each checkpoint
- Success = top 25%, Failure = bottom 25%, middle discarded

#### 3.9.3 Gradient Computation

- Online aggregation to avoid OOM
  - Re-run forward passes, compute GAE advantages, accumulate mean policy loss gradient (∇θ L^CLIP) per group
  - Don't store per-step gradient vectors
- L2 normalise before any cosine similarity calculations
  - PPO and Rainbow have incomparable loss scales (policy gradient vs. KL divergence) — magnitude has to go, we only care about direction

#### 3.9.4 Activation Analysis

- Extract penultimate-layer activations
- Activation vectors for individual samples from both Success and Failure groups
- Visualise with UMAP or t-SNE
- HDBSCAN clustering, record stats (number of clusters, size, group composition)
- Store Success and Failure centroids (mean activation)
- `torch.no_grad()` throughout
- Whether you get coherent single clusters or fragmented ones matters -  fragmentation means the binary partition is hiding sub-structure in how the agent represents success internally

---

### 3.10 Rainbow: Checkpointing and Evaluation

#### 3.10.1 Checkpointing Strategy

- Two checkpoint types, mirroring PPO's two-trigger design:
  - **Periodic:** every 100,000 training steps (`checkpoint_live.pt` for resume; named `checkpoint_step{N}.pt` for analysis)
  - **Achievement-triggered:** saves a named milestone checkpoint (`milestone_first_{ach}_ep{E}.pt`) the first time each of the 22 Crafter achievements is unlocked — same logic as PPO's `MilestoneTracker`
- Checkpoint format: `{online_net_state_dict, target_net_state_dict, optimizer_state_dict, args_dict, global_step, episode_count}` — both networks saved because offline Bellman loss reconstruction requires the target network
- Disk management uses a **two-pointer system** to avoid accumulating checkpoints while preserving the weights needed for weight-delta computation (see 3.10.4):
  - Pointer A = penultimate periodic checkpoint; Pointer B = last periodic checkpoint
  - On each new periodic: rotate A ← B, B ← new, delete old A
  - Milestone checkpoints are deleted immediately after analysis; they never enter the pointer chain

#### 3.10.2 Evaluation and Quartile Partitioning

- 500 evaluation episodes at each checkpoint, frozen policy (`eval()` mode — NoisyLinear deterministic)
- Same quartile logic as PPO: compute EPS per episode, top/bottom 25% → Success/Failure; middle 50% discarded
- Episodes collected via 16 parallel gymnasium environments with `AsyncVectorEnv`

#### 3.10.3 Reinterpreted Gradient Computation

- Important to be precise about what this is:
  - *Not* the gradient applied when these transitions were originally collected
  - It is the gradient the *current* online network would produce under the distributional Bellman loss on these frozen episodes — a counterfactual reinterpretation of historical experience under current weights
  - That's the right quantity: it captures what the current network treats as its learning signal from past successes and failures, which is what the research question actually asks
- Three gradient variants computed at each checkpoint:
  - **G_uniform:** gradient under uniform replay (all episodes weighted equally)
  - **G_IS:** gradient corrected for PER's sampling bias via importance sampling weights $w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$, with $P(i)$ reconstructed analytically from loss magnitudes
  - **G_reward:** gradient weighted by $|r| + \varepsilon$ — a direct proxy for reward-signal magnitude
- IS weights computed over the joint success+failure pool so magnitudes are cross-comparable
- NoisyLinear $\sigma$ parameters excluded from all gradient computations (analytically zero in eval mode)
- L2 normalise before cosine similarity, same as PPO

#### 3.10.4 Weight-Delta Empirical Validation

- Compute $\cos(G_\text{variant},\, \Delta\theta)$ where $\Delta\theta = \theta_{t+1} - \theta_t$ between consecutive analyzed checkpoints
- Tests whether each gradient variant aligns with actual parameter movement direction — if $G_\text{IS}$ aligns better than $G_\text{uniform}$, PER's bias is rotating gradients (not just scaling them)
- Milestone checkpoints use the last periodic checkpoint as their weight-delta reference (Pointer B); weight-delta is skipped if the milestone fires within 10,000 steps of the last periodic (Δθ too small for stable cosine estimation)

#### 3.10.5 Moment of Reward Analysis

- Sub-partition the Rainbow Success pool by immediate reward sign:
  - r > 0: immediate positive reward (achievement or survival bonus)
  - r = 0: preparatory action, no immediate reward
  - r < 0: penalised transition within a successful episode
- Compare gradient magnitudes and directions across the three subgroups
- If PER's implicit credit assignment over-weights immediate reward, r > 0 transitions should dominate — measurable as disproportionate gradient magnitude in that subgroup
- Sub-partition applied to G_uniform and G_IS separately to isolate PER's contribution

#### 3.10.6 Activation Analysis

- Same procedure as PPO (Section 3.9.4), applied to Rainbow's flattened CNN output (1024-dimensional)
- **Dimensionality note (flagged in Section 3.6):** PPO's encoder produces 64-dim representations; Rainbow's produces 1024-dim. Euclidean distance is sensitive to this volume difference — cosine distance is the primary reliable metric for cross-algorithm representational comparisons

---

### 3.11 Representational Similarity Analysis (Shared)

#### 3.11.1 Stimulus Set Definition

- Key Crafter concepts the agent has *discovered*
  - Exclude undiscovered elements -  don't inject privileged information
- Categories: Resources (Wood, Stone, Iron, Coal), Threats (Zombie, Skeleton, Lava), Tools (Pickaxes if crafted)

#### 3.11.2 Activation Collection

- For each element in the stimulus set:
  - Scan evaluation episodes for states containing it (same source for both PPO and Rainbow — frozen policy rollouts, no buffer access)
  - Pass through frozen network
  - Record mean activation vector

#### 3.11.3 RDM Construction

- Representational Dissimilarity Matrix:

$$\text{RDM}_{i,j} = 1 - \text{CosineSimilarity}(\text{Act}_i, \text{Act}_j)$$

- Look at clustering patterns -  do "Zombie" and "Skeleton" cluster together as "Threats"?
- Compare against a ground-truth functional hierarchy (threats together, tools together, resources together)
- RSA Alignment = correlation between the learned RDM and the reference

---

## 4. Research Questions and Hypotheses

**Job of this section:** The reader now knows *how* the framework works. This section says *what it's being used to ask*. Each question is grounded in a gap from the lit review, and every term in the hypotheses (opposition score, coherence, moment of reward, RSA alignment) has already been defined in Section 3.

---

### 4.1 From Gaps to Questions

The lit review landed on four gaps. PPO and Rainbow both assign credit through mechanisms you can't directly inspect — temporal weighting and advantage aggregation for PPO, distributional Bellman backups under a priority-weighted sampling distribution for Rainbow (Sections 3-4). The interpretability methods that exist either explain what a trained policy does rather than how it got there, or haven't been used longitudinally across algorithm families (Section 5). Nobody has systematically compared how on-policy vs. off-policy methods handle successful and failed experience at the gradient level (Section 6). And the temporal distribution of the learning signal within a successful trajectory — which moments actually drive weight updates — has not been directly measured in a hierarchical environment. The framework in Section 3 was built to address these gaps. The following questions are what it's designed to answer:

---

### 4.2 RQ1: Do on-policy (PPO) and off-policy (Rainbow) credit assignment produce structurally different gradient signatures from successful and failed episodes?

- *Hypothesis:* PPO only ever learns from its own current behaviour (β = π), so its gradient signatures from successful episodes should be more internally coherent — a consistent "theory of success." Rainbow reinterprets stored experience under distributional Q-estimates that have moved on since the data was collected, with PER further distorting the sampling distribution. That retrospective reassessment under a non-uniform, priority-weighted distribution should produce noisier, less directionally consistent signals.
- *Grounding:* On-policy vs. off-policy credit assignment distinction (lit review Section 2.2). GAE processes current-policy experience immediately; distributional Bellman backups replay old data under new Q-estimates with PER-weighted sampling (Sections 3.1, 4.1).
- *Interpretation note:* Opposition Score near -1.0 is not the only valid outcome. If success and failure episodes involve qualitatively different behaviours (resource gathering vs. dying to threats) rather than opposite intensities of the same behaviour, orthogonality (~0.0) with high within-group coherence is equally meaningful. The headline claim is structured divergence, not a specific target value.
- *Metrics engaged:* Opposition Score (Section 3.6) as headline, Coherence for disambiguation, Cosine Distance as primary cross-algorithm representational metric (Euclidean distance not directly comparable due to 64 vs. 1024 dimensionality).

---

### 4.3 RQ2: Does Rainbow's PER-based replay mechanism implicitly over-weight immediate reward signals over preparatory actions — and does IS correction substantively correct this bias?

- *Hypothesis:* PER priority ∝ TD error, so PER does not merely scale the gradient — it rotates it toward high-surprise transitions. If IS correction changes the *direction* of the gradient (not just its magnitude), PER is introducing a systematic directional bias beyond sampling noise: G_IS should diverge from G_uniform, and this divergence should be empirically confirmed by G_IS aligning better with the actual weight movement Δθ than G_uniform does.
- *Grounding:* PER explicitly prioritises surprising experiences, compounding the delayed-reward propagation problem already present in single-step Bellman backups (lit review Section 4.1). The G_uniform vs. G_IS comparison (Section 3.10.3) and weight-delta validation (Section 3.10.4) both test this.
- *Metrics engaged:* cos(G_IS, G_uniform) as the primary direction-rotation measure; cos(G_IS, Δθ) vs cos(G_uniform, Δθ) for empirical validation; cos(G_IS, G_reward) as a diagnostic on what PER is prioritising.

---

### 4.4 RQ3: Do gradient-level differences between PPO and Rainbow manifest in representational structure?

- *Hypothesis:* If PPO produces more coherent learning signals (per RQ1), it should build more semantically organised internal representations — threats clustering with threats, resources with resources — as measured by RSA Alignment. Rainbow might still learn functional representations (the agent achieves tasks) but with messier semantic structure due to noisier, PER-distorted gradient signals.
- *Grounding:* Gradients track the learning signal's direction. RSA (Section 3.11) tracks whether that signal produces structured representations (lit review Section 5.3). RQ3 tests whether signal-level differences from RQ1 actually have representational consequences downstream.
- *Dimensionality note:* Cosine distance is the primary cross-algorithm metric; Euclidean separation is only meaningful within-algorithm.

---

### 4.5 RQ4: How is the learning signal distributed across the temporal phases of a successful trajectory, and does Rainbow's implicit credit assignment over-weight immediate reward feedback?

- *Hypothesis:* Within successful episodes, PER's priority mechanism (TD error ∝ reward surprise) should concentrate gradient mass on transitions where reward is received (r > 0) and on catastrophic failures avoided (r < 0), at the expense of the preparatory zero-reward transitions (r = 0) that make up the majority of a successful Crafter trajectory. Under G_uniform, gradient magnitude from the r > 0 sub-group should exceed that of r = 0. IS correction (G_IS) should partially restore the r = 0 contribution if PER is the source of the imbalance.
- *Grounding:* The credit assignment gap in hierarchical environments (lit review Section 4.1 — the Crafter ore/smelt example). Long causal chains mean most of the behaviourally important transitions are zero-reward. If those transitions are systematically under-weighted by PER, the agent learns *what the achievement is* but not *how to set it up*. This is a distinct question from RQ2: RQ2 asks whether PER rotates the aggregate gradient direction; RQ4 asks where within a trajectory that rotation is concentrated.
- *Novelty:* Directly measuring the gradient contribution of reward-signed sub-partitions of evaluation episodes — and comparing uniform vs. IS-corrected profiles across training — has not been done in prior credit assignment interpretability work. This is MORA (Moment of Reward Analysis), a novel analytical contribution of this dissertation.
- *Metrics engaged:* Per-subgroup gradient magnitude and coherence (r > 0, r = 0, r < 0 within success episodes), for both G_uniform and G_IS. The relative shift in magnitude profile between G_uniform and G_IS is the primary finding metric. cos(G_IS_r_zero, G_uniform_r_zero) as a directional check on whether IS changes where in the trajectory the gradient is pointing, not just how much.

> **[Figure opportunity]** Conceptual diagram: the four RQs and how they relate. RQ2 and RQ4 are Rainbow-specific; RQ1 and RQ3 are cross-algorithm. RQ2 (IS correction) and RQ4 (MORA) both feed into RQ1's interpretation of Rainbow's gradient structure. RQ1 (gradient-level) feeds RQ3 (representation-level). draw.io vector.

---

	## 5. Results

**Job of this section:** Answer the research questions one at a time. Validation first, then RQ1, RQ2, RQ3, RQ4. Put completed results before pending ones. Open with a short framing paragraph: what's done, what isn't, what the reader should expect.

---

### 5.1 Preliminary Validation

- Do both agents actually learn on Crafter?
- Learning curves against published baselines
- This isn't where the research question gets answered -  it's where the experimental setup gets validated
- Need non-trivial achievement acquisition before gradient analysis means anything
- Flag anomalies from preliminary runs

> **[Figure 5.0]** Learning curves for both algorithms overlaid, published Crafter baselines marked. Inkscape/draw.io vector.

---

### 5.2 RQ1: Do PPO and Rainbow Generate Structurally Different Gradient Signatures?

#### 5.2.1 Opposition Score Over Training

- Present trajectories across checkpoints, both algorithms side by side

**Figure 5.1: Return x Opposition Score over training (dual-axis).** Left y-axis: mean episode return. Right y-axis: Opposition Score. X-axis: training steps. Achievement milestones (wake_up, collect_stone, make_stone_sword, etc.) as vertical lines. *Headline figure for RQ1 -  shows whether improving performance comes with increasingly differentiated gradient signals from success vs. failure.*

- [PLACEHOLDER: Describe the observed trajectory. Phases, transitions, volatility. Does early opposition behaviour reflect a poorly calibrated value function (GAE critique, lit review)? Do achievement milestones line up with visible transitions in opposition?]
- Opposition score alone isn't enough -  connect to coherence (5.2.2) per the metric justification in Section 3.6

#### 5.2.2 Gradient Coherence and Signal Consistency

- Within-group coherence for the Success batch, both algorithms

**Figure 5.2: Return x Coherence (Success) over training (dual-axis).** Left y-axis: mean episode return. Right y-axis: Coherence (Success). X-axis: training steps. *Does improving performance come with an increasingly consistent "theory of success"? Or do they diverge -  meaning the Success partition is mixing qualitatively different episode types as the agent gets more capable?*

- [PLACEHOLDER: How do return and coherence relate? Track together, diverge, no pattern? Compare PPO vs. Rainbow.]
- Key for interpreting opposition score (Section 7.1): low coherence + high opposition could just be noise in the mean gradient directions, not a genuinely differentiated signal. Figure 5.2 is the main evidence for this.

#### 5.2.3 Activation Space Structure

**Figure 5.3: Return x Activation Separation over training (dual-axis).** Left y-axis: mean episode return. Right y-axis: Euclidean activation separation between Success/Failure centroids. X-axis: training steps. *Primary activation-space figure. Does the network build increasingly distinct representations of success vs. failure states as it learns?*

**Figure 5.4: Return x Cosine Distance over training (dual-axis).** Left y-axis: mean episode return. Right y-axis: Cosine distance between Success/Failure activation centroids. X-axis: training steps. *Backs up Figure 5.3. If both Euclidean and cosine track return, the separation is genuine directional divergence and not just magnitude scaling.*

- [PLACEHOLDER: Do activation metrics track return? Do Euclidean and cosine agree?]
- UMAP/t-SNE visualisations and HDBSCAN cluster stats at selected checkpoints (early, mid, late)
- [PLACEHOLDER: Fragmentation patterns. Single coherent clusters or fragmented? Fragmentation = the network has finer distinctions than the binary partition captures.]

---

### 5.3 RQ2: Does PER Rotate the Gradient Direction — and Does IS Correction Fix It?

#### 5.3.1 IS Correction and Directional Influence

- Primary measure: cos(G_IS, G_uniform) over training — does IS correction change gradient *direction*, or only scale?
- If cos(G_IS, G_uniform) is consistently below ~0.9, PER is rotating the gradient, not just reweighting magnitudes

> **[Figure 5.5]** cos(G_IS, G_uniform) and cos(G_IS, G_reward) over training for both success and failure groups. Dual-line plot per group. Shows whether IS correction is material and whether PER priorities are explained by reward magnitude.

- [PLACEHOLDER: Do cos values stay high (IS correction is minor) or drop (PER is directionally distorting)? Does the pattern differ between success and failure groups?]

#### 5.3.2 Weight-Delta Empirical Validation

- cos(G_IS, Δθ) vs cos(G_uniform, Δθ) across consecutive checkpoint pairs
- If G_IS more closely matches the actual direction of weight movement, the IS reconstruction is validated empirically

> **[Figure 5.6]** cos(G_IS, Δθ) vs cos(G_uniform, Δθ) over training, both groups. The relative ordering is the finding, not the absolute values (Adam distortion caveat, Section 3.10.4).

- [PLACEHOLDER: Does G_IS outperform G_uniform in mid-training? Does the signal collapse at late training as Δθ shrinks?]
- The reinterpreted gradient framing (Section 3.10.3) is essential context — these gradients are computed under the *current* network weights on frozen evaluation episodes, not whatever weights were active during training

---

### 5.4 RQ4: Moment of Reward Analysis (MORA)

#### 5.4.1 Gradient Distribution Across Temporal Phases

- Sub-partition the success group into three reward-signed sub-groups: r > 0, r = 0, r < 0
- Compute gradient magnitude and coherence for each sub-group under G_uniform and G_IS

> **[Figure 5.7]** Grouped bar chart or stacked area: gradient magnitude by moment-of-reward category (r > 0 / r = 0 / r < 0) across training checkpoints, for both G_uniform and G_IS. Standalone figure — the three-way decomposition and the IS vs. uniform shift are the point here.

- [PLACEHOLDER: Does r > 0 dominate under G_uniform? Does G_IS reduce or eliminate this dominance? What share of the gradient comes from zero-reward preparatory transitions?]

#### 5.4.2 Implications for Credit Assignment in Hierarchical Environments

- [PLACEHOLDER: If r = 0 transitions contribute disproportionately little under G_uniform, that is direct evidence that Rainbow's credit assignment under-weights the preparatory behaviours that make success possible in Crafter's achievement chain.]
- Connect to RQ2: if IS correction visibly restores r = 0 contribution, PER is the mechanism causing the imbalance. If IS correction leaves the profile unchanged, the imbalance is structural to distributional Bellman backups, not to PER sampling.
- This is MORA's unique contribution: not just that PER rotates the aggregate gradient (RQ2), but *where in the trajectory* that rotation concentrates the learning signal.

---

### 5.5 RQ3: Representational Similarity Analysis

- RDMs at early, mid, and late checkpoints, both algorithms
- Does semantic clustering emerge — threats together, tools together? When?
- RSA Alignment scores across algorithms:
  - Does PPO's on-policy signal give better-organised representations?
  - Or does Rainbow's PER-driven sample efficiency give it a representational edge despite noisier gradients?

> **[Figure 5.6]** RDM heatmaps, 2x3 grid: PPO/Rainbow x early/mid/late. Not a dual-axis plot — RDMs are best shown as matrices.

- [PLACEHOLDER: RSA patterns. When does semantic structure appear? How do the algorithms compare?]

---

### 5.6 Synthesis

- Pull it together before discussion
- What can actually be said at the final checkpoint about how PPO and Rainbow process success and failure differently?
- Figures 5.1-5.7 as evidence, noting which questions are resolved and which are still open
- Be straight about what's pending and what it would add
- Reader should arrive at Discussion with a clear picture, not needing it to retrospectively assemble one

---

## 6. Future Work and Discussion

**Job of this section:** Use findings as a launchpad. Every extension grounded in a specific result or limitation -  not a speculative wishlist.

---

### 6.1 Limitations

Each limitation motivates an extension.

**Quartile split discards the middle 50%:**
- Maximises contrast at the cost of coverage (Section 3.5)
- Boundary episodes (nearly succeeded, nearly failed) might contain relevant information -  e.g. the tipping point in gradient structure
- Defensible but not costless

**Reinterpreted gradient is a proxy:**
- For Rainbow (Section 3.10.3): what the current network *would* learn from these evaluation episodes, not what it *did* learn during training
- Actual training gradient was under different network weights, different Q-estimates, and a different PER priority distribution
- Right quantity for the research question, but not a record of what actually happened during training

**RSA Alignment depends on a hand-built hierarchy:**
- Ground-truth hierarchy (threats together, resources together, tools together) in Section 3.11.3 is one reasonable construction
- Could also group by required *action* (flee, collect, use) instead of semantic category
- Different reference RDMs, different alignment scores

**Crafter is one environment:**
- Good testbed for credit assignment because of the hierarchical achievement chain
- But it's a single environment with a particular reward structure
- No claims about generalising to denser rewards, continuous action spaces, or different temporal structures

**The opposition assumption:**
- Opposition Score treats -1.0 (perfect anti-correlation) as the ideal
- Assumes what you should learn from good episodes is the direct opposite of what you should learn from bad ones -  reinforcing and discouraging as mirror operations in parameter space
- That doesn't have to be true:
  - Good episodes might involve one set of behaviours (resource gathering, tool use) while bad episodes involve a *different* set (wandering, dying to threats) -  not the *opposite* set
  - Those gradient signals aren't opposed, they're just about different things
  - "Do more of X" and "do less of Y" only produce opposing gradients when X and Y are the same behaviour at different intensities
  - If best and worst episodes are qualitatively different, orthogonality (around 0.0) might be correct -  not a failure to distinguish
- The strength of the signal matters too:
  - Passive failure (just not doing the right thing) gives a weak gradient with little directional content
  - Active failure (doing something specifically bad) gives a strong, directed signal
- So Opposition Score has to be read alongside Coherence, not against a fixed -1.0 target
  - High coherence in both groups + near-zero opposition = the algorithm has clear but non-mirrored models of what success and failure look like
  - No strong opposition doesn't mean no credit assignment
  - Figure 5.2 is the key evidence: if coherence and return diverge, the Success partition has qualitatively different episode types in it and opposition score has to be read with that in mind

---

### 6.2 Extensions to Credit Assignment

- Proof-of-concept directions, grounded in specific results
- If RQ4 confirms Rainbow's immediate-reward bias (r > 0 dominance in MORA): sketch a modified replay sampling strategy that upweights preparatory zero-reward transitions based on gradient coherence with later successful outcomes
- If opposition score shows phase transitions at achievement milestones (Figure 5.1): propose adaptive checkpointing as a general training monitoring tool, not just for this analysis
- PoC-level ideas with clear links back to results, not speculation

---

### 6.3 Extensions to Hierarchical RL

- Opposition Score and Coherence map naturally onto HRL
- The termination function and intra-option policy are two separate learning signals
  - If they push gradients in opposite directions, the option is incoherent -  terminating in states where the intra-option policy still wants to keep going
- Coherence tracked over an option's gradient direction could be an environment-agnostic skill discovery criterion
  - Stable, high-coherence gradient directions = learned, internally-consistent skills

---

### 6.4 Extensions to Model-Based RL

- Compare gradient signatures of world model loss vs. actor loss in something like DreamerV3
- Hypothesis (from the deadly triad discussion, lit review Section 4): learning environment dynamics gives a more stable gradient substrate than algorithmic constraints (PPO's clipping, Rainbow's distributional loss and PER)
- RSA Alignment could test whether model-based agents develop cleaner semantic representations than model-free baselines -  a testable prediction from the framework developed here

---

### 6.5 Towards Deployment Auditing

- Back to the applied motivation from the intro (Amodei et al., Dulac-Arnold et al.)
- This framework could work as a training-time audit tool
  - Monitor Opposition Score and Coherence during training to flag policy collapse or bad credit assignment *before* deployment
- The Trim & Grayston maze agent that learned "go to the top-right corner" instead of "go to the goal" would probably have shown anomalous RSA Alignment during training
- Current evaluation catches failures after deployment -  this could catch them earlier

---

## 7. Conclusion

**Job of this section:** Tight, backward-looking. No new ideas. Someone who only read the introduction should feel the arc closed.

---

### 7.1 Summary of Contributions

Three things:

1. A gradient-based comparative framework for analysing credit assignment across algorithm families
2. Moment of Reward analysis as a novel diagnostic for implicit reward prioritisation in PER-based off-policy learning
3. Empirical results from Crafter characterising how PPO and Rainbow differ in gradient signatures across training

Keep proportionate to what was actually shown.

---

### 7.2 Closing Argument

- Back to the interpretability problem from the introduction
- Set out to look inside the learning process -  not the trained policy, but the signals shaping it in real time
- State what was found
- Strongest version of the contribution: applying gradient analysis and RSA longitudinally, across algorithm families, during training rather than after it, is a step toward interpretability that matters when it actually matters

---

## Appendix: Cross-Reference Notes

**Training steps and checkpoint intervals:** Methodology doc (v3) says 5M+ steps, checkpoints every 200k. Dissertation notes say 3M steps, every 50k (~60 checkpoints). Very different resolutions (25 vs. 60). Reconcile with whatever actually happened.

**Partitioning strategy:** Methodology doc (v3) uses mean split (EPS >= mu). Dissertation notes evolved to quartile split (top/bottom 25%, middle discarded). Quartile is the later thinking and is used throughout this structure. Update the methodology doc.

**Missing metric:** Cosine Distance/Cosine Separation appears in the dissertation notes and the metrics table (Section 3.6) but not in the methodology doc's comparative metrics table. Add it.

**v1 → v2 restructure:** Methodology moved from Section 4 to Section 3. Research Questions moved from Section 3.1 to Section 4. All internal cross-references updated. Content unchanged -  this is a reorder, not a rewrite.

---

## Figure Index

| Figure | Section | Type                       | Content                                                                                     |
| :----- | :------ | :------------------------- | :------------------------------------------------------------------------------------------ |
| 5.0    | 5.1     | Line chart                 | Learning curves for both algorithms, published baselines marked                             |
| 5.1    | 5.2.1   | Dual-axis line             | Return (left) x Opposition Score (right) over training steps, achievement milestones marked |
| 5.2    | 5.2.2   | Dual-axis line             | Return (left) x Coherence - Success (right) over training steps                             |
| 5.3    | 5.2.3   | Dual-axis line             | Return (left) x Activation Separation (right) over training steps                           |
| 5.4    | 5.2.3   | Dual-axis line             | Return (left) x Cosine Distance (right) over training steps                                 |
| 5.5    | 5.3.1   | Grouped bar / stacked area | Gradient magnitude by moment-of-reward category (r > 0, r = 0, r < 0) across checkpoints    |
| 5.6    | 5.4     | 2x3 heatmap grid           | RDM heatmaps: PPO/Rainbow x early/mid/late checkpoints                                          |

All dual-axis figures share the same x-axis (training steps) and left y-axis (mean episode return) so they can be compared visually across the series.

Conceptual/design figures in Section 3 (partitioning strategy, metric pipeline, Crafter dependency tree) and Section 4 (RQ diagram) aren't numbered here -  they're methodological illustrations, not results figures.
