# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 2  
**Total episodes:** 100,000  
**Experiment root:** `Proof-of-Concept-Runs\seed_2`  
**Generated:** 2026-02-25 11:57

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1009 episodes) | 1,009 | 0.8685 | 0.3731 | 0.3059 | 0.0796 | 0.0677 | 0.0509 | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 2 — 2k (2004 episodes) | 2,004 | 0.5878 | 0.1187 | 0.2297 | 0.0487 | 0.0576 | 0.0542 | — |
| Checkpoint 3 — 3k (3011 episodes) | 3,011 | 0.7781 | 0.3677 | 0.3584 | 0.0694 | 0.0746 | 0.0372 | — |
| Checkpoint 4 — 4k (4008 episodes) | 4,008 | 0.5753 | 0.1033 | 0.2040 | 0.0736 | 0.0670 | 0.0609 | — |
| Checkpoint 5 — 5k (5001 episodes) | 5,001 | 0.4208 | 0.2664 | 0.1604 | 0.1457 | 0.0708 | 0.1503 | — |
| Checkpoint 6 — 6k (6002 episodes) | 6,002 | 0.5364 | 0.2963 | 0.0259 | 0.2063 | 0.0590 | 0.4978 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 7 — 7k (7008 episodes) | 7,008 | 0.3239 | 0.2068 | 0.0635 | 0.3191 | 0.0706 | 0.5175 | — |
| Checkpoint 8 — 8k (8002 episodes) | 8,002 | -0.2157 | 0.3543 | 0.1002 | 0.4593 | 0.1011 | 1.0988 | — |
| Checkpoint 9 — 9k (9010 episodes) | 9,010 | 0.0831 | 0.2128 | 0.0192 | 0.2459 | 0.0687 | 1.2633 | — |
| Checkpoint 100 — 100k (100000 episodes) | 100,000 | 0.2910 | 0.2260 | 0.2281 | 0.2859 | 0.2132 | 8.3123 | — |

---

## Checkpoint 1 — 1k (1009 episodes)

**Episodes:** 1,009  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8685 |
| Coherence (Success) | 0.3731 |
| Coherence (Failure) | 0.3059 |
| Gradient Magnitude (Success) | 0.0796 |
| Gradient Magnitude (Failure) | 0.0677 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0509 |
| Clusters | 2,135 |
| Noise Fraction | 0.0287 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.178 | 0.143 | 0.060 |
| locked_door | 0.178 | -0.000 | 0.166 | 0.285 |
| open_door | 0.143 | 0.166 | 0.000 | 0.303 |
| target_ball | 0.060 | 0.285 | 0.303 | 0.000 |

---

## Checkpoint 2 — 2k (2004 episodes)

**Episodes:** 2,004  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5878 |
| Coherence (Success) | 0.1187 |
| Coherence (Failure) | 0.2297 |
| Gradient Magnitude (Success) | 0.0487 |
| Gradient Magnitude (Failure) | 0.0576 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0542 |
| Clusters | 2,146 |
| Noise Fraction | 0.0331 |

---

## Checkpoint 3 — 3k (3011 episodes)

**Episodes:** 3,011  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7781 |
| Coherence (Success) | 0.3677 |
| Coherence (Failure) | 0.3584 |
| Gradient Magnitude (Success) | 0.0694 |
| Gradient Magnitude (Failure) | 0.0746 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0372 |
| Clusters | 2,128 |
| Noise Fraction | 0.0281 |

---

## Checkpoint 4 — 4k (4008 episodes)

**Episodes:** 4,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5753 |
| Coherence (Success) | 0.1033 |
| Coherence (Failure) | 0.2040 |
| Gradient Magnitude (Success) | 0.0736 |
| Gradient Magnitude (Failure) | 0.0670 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0609 |
| Clusters | 1,882 |
| Noise Fraction | 0.0240 |

---

## Checkpoint 5 — 5k (5001 episodes)

**Episodes:** 5,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4208 |
| Coherence (Success) | 0.2664 |
| Coherence (Failure) | 0.1604 |
| Gradient Magnitude (Success) | 0.1457 |
| Gradient Magnitude (Failure) | 0.0708 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1503 |
| Clusters | 1,764 |
| Noise Fraction | 0.0205 |

---

## Checkpoint 6 — 6k (6002 episodes)

**Episodes:** 6,002  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.630)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5364 |
| Coherence (Success) | 0.2963 |
| Coherence (Failure) | 0.0259 |
| Gradient Magnitude (Success) | 0.2063 |
| Gradient Magnitude (Failure) | 0.0590 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.4978 |
| Clusters | 1,341 |
| Noise Fraction | 0.0239 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.391 | 0.390 | 0.250 |
| locked_door | 0.391 | 0.000 | 0.069 | 0.899 |
| open_door | 0.390 | 0.069 | -0.000 | 0.826 |
| target_ball | 0.250 | 0.899 | 0.826 | 0.000 |

---

## Checkpoint 7 — 7k (7008 episodes)

**Episodes:** 7,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.903)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3239 |
| Coherence (Success) | 0.2068 |
| Coherence (Failure) | 0.0635 |
| Gradient Magnitude (Success) | 0.3191 |
| Gradient Magnitude (Failure) | 0.0706 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.5175 |
| Clusters | 1,236 |
| Noise Fraction | 0.0286 |

---

## Checkpoint 8 — 8k (8002 episodes)

**Episodes:** 8,002  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.153)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2157 |
| Coherence (Success) | 0.3543 |
| Coherence (Failure) | 0.1002 |
| Gradient Magnitude (Success) | 0.4593 |
| Gradient Magnitude (Failure) | 0.1011 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0988 |
| Clusters | 1,125 |
| Noise Fraction | 0.0256 |

---

## Checkpoint 9 — 9k (9010 episodes)

**Episodes:** 9,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.153)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0831 |
| Coherence (Success) | 0.2128 |
| Coherence (Failure) | 0.0192 |
| Gradient Magnitude (Success) | 0.2459 |
| Gradient Magnitude (Failure) | 0.0687 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.2633 |
| Clusters | 1,210 |
| Noise Fraction | 0.0333 |

---

## Checkpoint 100 — 100k (100000 episodes)

**Episodes:** 100,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2910 |
| Coherence (Success) | 0.2260 |
| Coherence (Failure) | 0.2281 |
| Gradient Magnitude (Success) | 0.2859 |
| Gradient Magnitude (Failure) | 0.2132 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.3123 |
| Clusters | 519 |
| Noise Fraction | 0.0864 |
