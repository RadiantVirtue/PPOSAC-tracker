# Gradient Analysis Methodology

A comparative framework for analysing learning signals in PPO (on-policy) and SAC (off-policy) reinforcement learning algorithms.

---

## 1. Core Definitions & Experimental Controls

To ensure mathematical comparability between PPO and SAC, the following definitions are standardized across all analyses.

### 1.1 Dynamic Success Thresholding

For any given checkpoint or buffer sample:

1. Run evaluation episodes and record achievement counts
2. Compute **μ** = average $EPS^1$ across all episodes.
3. Partition:

| Group       | Definition                |
| :---------- | :------------------------ |
| **Success** | Episodes with EPS **≥ μ** |
| **Failure** | Episodes with EPS **< μ** |

This ensures the threshold adapts as the agent improves - what counts as "success" at 5M steps differs from success at 25M steps.

---

$^1$EPS (Episode progress score) is defined as: $$EPS = achievements + 0.9\frac{materials_{acquired}}{materials_{needed}*}$$
*Materials needed for next achievement
Used both in finding the partition threshold and individual episode evaluation. Use closest achievement.

---
### 1.2 Gradient Definitions

| Algorithm | Gradient Source                                                                    |
| :-------- | :--------------------------------------------------------------------------------- |
| **PPO**   | Policy loss gradient: ∇<sub>θ</sub> L<sup>CLIP</sup>                               |
| **SAC**   | Actor loss gradient: ∇<sub>θ</sub> L<sup>actor</sup> = ∇<sub>θ</sub> (α log π − Q) |

**Normalization:** All gradients are L2-normalized before cosine similarity calculations (layer-wise) This ensures analysis focuses on *direction* (semantics) rather than *magnitude* (scale), as PPO and SAC magnitudes are not directly comparable.

---

## 2. PPO Analysis Methodology

### Step 1: Checkpoint Generation

Train PPO on Crafter for 5M+ steps. Save frozen checkpoints at:
- Regular intervals (every 200k steps). Storage is cheap it is better to save checkpoints that turn out to be unnecessary than to risk missing a critical transition in learning dynamics.
- Every time the agent earns a **new achievement** for the first time. Additional milestone criteria (e.g., reaching a known achievement significantly faster than previous attempts) may be added as training behaviour becomes clearer.

Each checkpoint includes policy network weights and value function weights.

### Step 2: Sampling & Partitioning

At each checkpoint:

1. Load frozen policy π<sub>θ</sub>
2. Run **1,000 evaluation episodes** (no updates)
3. Record (State, Action, Reward) tuples and Total Return
4. Partition episodes into Success/Failure based on thresholding

### Step 3: Gradient Computation

To prevent OOM errors, gradients are computed via online aggregation rather than per-step storage.

```
For each group (Success / Failure):
    
    1. Initialize sum_gradient = 0
    
    2. For each episode in group:
        a. Re-compute forward passes → log_prob, Advantage (GAE)
        b. Compute ∇θ L^CLIP for the trajectory
        c. Accumulate: sum_gradient += current_gradient
    
    3. Store Mean Gradient Vector for the group
```

### Step 4: Activation Analysis

1. Extract activations from the policy network's penultimate layer
2. Generate activation vectors for individual samples from both Success and Failure groups
3. Visualise the activation space using dimensionality reduction (UMAP or t-SNE) to inspect whether Success and Failure form single coherent clusters or fragment into multiple sub-clusters
4. Apply density-based clustering (HDBSCAN) to the projected activations and compute statistics of the resulting clusters (number, size, composition by group label)
5. Store centroids (mean activation) for Success and Failure groups, alongside the cluster-level statistics

---

## 3. SAC Analysis Methodology

### Step 1: Tagged Replay Buffer

 The replay buffer is analyzed directly.

During training, tag every transition with episode outcome:
	
```
transition = {
    state:       s,
    action:      a,
    reward:      r,
    next_state:  s',
    episode_return: float    ← filled at episode end
}
```

In addition to the tagged replay buffer, store a separate copy of full episodes (ordered sequences of transitions) at each checkpoint. This avoids the need to reconstruct episode-level trajectories from the replay buffer after the fact, which becomes increasingly difficult as the buffer overwrites older data.

Save the Replay Buffer, the episode store, and network weights at every checkpoint.

### Step 2: Sampling & Partitioning

At each checkpoint:

1. Load the frozen Replay Buffer
2. Compute **μ** = mean episode return across all episodes in the buffer
3. Filter transitions into two pools:
   - **Success Pool:** Transitions from episodes with achievements ≥ μ
   - **Failure Pool:** Transitions from episodes with achievements < μ
4. Sample **N = 5,000 transitions** from each pool

### Step 3: Gradient Computation

Analyze the "Success Batch" vs. "Failure Batch" directly.

For each batch (Success / Failure):
 
1. Compute the **Reinterpreted Gradient** - the gradient the *current* actor π<sub>θ</sub> would generate if trained on this historical batch:

$$\nabla_\theta \mathbb{E}_{s \sim \mathcal{D}} [\alpha \log \pi_\theta(a|s) - Q_{\phi}(s, a)]$$
2. Compute the running mean gradient for the batch

### Step 4: Moment of Reward Analysis

Sub-partition the Success Pool into three categories to ensure generality across MDPs with different reward functions:

| Subgroup                  | Definition              | Question Addressed                                        |
| :------------------------ | :---------------------- | :-------------------------------------------------------- |
| **Positive Transitions**  | Transitions where r > 0 | How does SAC weight immediate positive reward?            |
| **Neutral Transitions**   | Transitions where r = 0 | How does SAC weight actions leading to reward?            |
| **Negative Transitions**  | Transitions where r < 0 | How does SAC weight penalised actions within success?     |

Compare mean gradients across these subgroups to determine if SAC over-prioritizes immediate reward over preparatory actions, and how negative reward signals within otherwise successful episodes influence the learning gradient.

---

## 4. Representational Similarity Analysis

*Verifying whether networks build structured representations of task elements.*

### Step 1: Stimulus Set Definition

Define key Crafter concepts. Only analyze elements the agent has **discovered** - undiscovered elements are excluded to avoid injecting privileged state information.

| Category      | Elements                |
| :------------ | :---------------------- |
| **Resources** | Wood, Stone, Iron, Coal |
| **Threats**   | Zombie, Skeleton, Lava  |
| **Tools**     | Pickaxes (if crafted)   |

### Step 2: Activation Collection

For each element in the stimulus set:

1. Scan the Replay Buffer (SAC) or Evaluation Episodes (PPO) for states containing the element
2. Pass states through the frozen network
3. Record the mean activation vector for that element

### Step 3: RDM Construction

Construct the Representational Dissimilarity Matrix:

$$\text{RDM}_{i,j} = 1 - \text{CosineSimilarity}(\text{Act}_i, \text{Act}_j)$$

Analyze clustering patterns - for example, do "Zombie" and "Skeleton" cluster together as "Threats"?

---

## 5. Comparative Metrics

For every checkpoint, generate the following comparison table.

| Metric                    | Scope      | Description                                                           | Interpretation                                                                                               |
| :------------------------ | :--------- | :-------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **Opposition Score**      | PPO vs SAC | Cosine similarity between ∇<sub>Success</sub> and ∇<sub>Failure</sub> | **−1.0:** Perfect distinction (opposites) ​ **0.0:** Orthogonal (unrelated) ​ **+1.0:** Confused (identical) |
| **Coherence**             | PPO vs SAC | Average cosine similarity *within* the Success gradient batch         | High = consistent "theory" of success                                                                        |
| **Gradient Magnitude**    | Internal   | L2 norm of gradient vector                                            | Track signal fading over time ​ *Do not compare across algorithms*                                           |
| **Activation Separation** | PPO vs SAC | Euclidean distance between Success/Failure activation centroids       | How distinct are internal representations of winning vs. losing?                                             |
| **RSA Alignment**         | PPO vs SAC | Correlation of RDM with ground-truth functional hierarchy             | Does the agent understand that Wood and Stone are similar (materials)?                                       |

---

## 6. Implementation Architecture

### 6.1 Data Storage Schema

```
experiment_root/
│
├── checkpoints/
│   ├── ppo_5M.pt
│   ├── sac_5M.pt
│   ├── sac_5M_buffer.pkl
│   ├── sac_5M_episodes.pkl
│   └── ...
│
├── analysis_logs/
│   ├── ppo_gradients.json      ← calculated metrics only
│   ├── sac_gradients.json      ← NOT raw vectors
│   └── rsa_matrices.npy
│
└── metadata.json               ← training hyperparameters
```

### 6.2 Compute Requirements

| Component        | Resource     | Notes                               |
| :--------------- | :----------- | :---------------------------------- |
| **Training**     | Standard GPU | Normal training loop                |
| **PPO Analysis** | CPU-heavy    | Requires re-simulation of episodes  |
| **SAC Analysis** | GPU + RAM    | Requires full replay buffer loading |

**Optimization Notes:**
- Use `torch.no_grad()` for all Activation Analysis
- Use `autograd` only for Gradient computation steps
- Store computed metrics, not raw gradient vectors
- Use TorchRL's replay buffer implementations for SAC - these are highly optimised and provide built-in features (e.g., prioritised sampling, efficient serialisation) that simplify checkpointing and analysis
- Keep replay buffer data on GPU where possible and minimise transfers between RAM and VRAM, as frequent data movement between devices is the primary bottleneck for wall-clock time during SAC analysis
