# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 1  
**Total episodes:** 100,061  
**Experiment root:** `Proof-of-Concept-Runs\seed_1`  
**Generated:** 2026-02-25 11:57

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1000 episodes) | 1,000 | 0.2424 | 0.1979 | 0.3319 | 0.0634 | 0.0786 | 0.0653 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2 — 2k (2010 episodes) | 2,010 | 0.9592 | 0.4664 | 0.3933 | 0.1063 | 0.0889 | 0.0357 | — |
| Checkpoint 3 — 3k (3006 episodes) | 3,006 | 0.6739 | 0.2495 | 0.1426 | 0.0875 | 0.0509 | 0.0649 | — |
| Checkpoint 4 — 4k (4009 episodes) | 4,009 | 0.0615 | 0.3577 | 0.1105 | 0.2757 | 0.0764 | 0.3750 | — |
| Checkpoint 5 — 5k (5005 episodes) | 5,005 | 0.0518 | 0.2922 | 0.0535 | 0.5618 | 0.1056 | 1.0008 | — |
| Checkpoint 6 — 6k (6016 episodes) | 6,016 | 0.0957 | 0.1729 | 0.1004 | 0.3079 | 0.1375 | 0.9080 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 7 — 7k (7010 episodes) | 7,010 | -0.2676 | 0.3239 | 0.0189 | 0.4858 | 0.0890 | 1.0657 | — |
| Checkpoint 8 — 8k (8018 episodes) | 8,018 | 0.4233 | 0.2396 | 0.2134 | 0.3688 | 0.1837 | 0.9578 | — |
| Checkpoint 9 — 9k (9003 episodes) | 9,003 | 0.0878 | 0.2599 | 0.0878 | 0.4104 | 0.1298 | 1.3200 | — |
| Checkpoint 100 — 100k (100061 episodes) | 100,061 | 0.5197 | 0.1636 | 0.1608 | 0.2463 | 0.1888 | 7.4301 | — |

---

## Checkpoint 1 — 1k (1000 episodes)

**Episodes:** 1,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2424 |
| Coherence (Success) | 0.1979 |
| Coherence (Failure) | 0.3319 |
| Gradient Magnitude (Success) | 0.0634 |
| Gradient Magnitude (Failure) | 0.0786 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0653 |
| Clusters | 2,123 |
| Noise Fraction | 0.0341 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.086 | 0.089 | 0.259 |
| locked_door | 0.086 | 0.000 | 0.053 | 0.509 |
| open_door | 0.089 | 0.053 | 0.000 | 0.564 |
| target_ball | 0.259 | 0.509 | 0.564 | 0.000 |

---

## Checkpoint 2 — 2k (2010 episodes)

**Episodes:** 2,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9592 |
| Coherence (Success) | 0.4664 |
| Coherence (Failure) | 0.3933 |
| Gradient Magnitude (Success) | 0.1063 |
| Gradient Magnitude (Failure) | 0.0889 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0357 |
| Clusters | 2,090 |
| Noise Fraction | 0.0278 |

---

## Checkpoint 3 — 3k (3006 episodes)

**Episodes:** 3,006  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6739 |
| Coherence (Success) | 0.2495 |
| Coherence (Failure) | 0.1426 |
| Gradient Magnitude (Success) | 0.0875 |
| Gradient Magnitude (Failure) | 0.0509 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0649 |
| Clusters | 2,085 |
| Noise Fraction | 0.0297 |

---

## Checkpoint 4 — 4k (4009 episodes)

**Episodes:** 4,009  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.650)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0615 |
| Coherence (Success) | 0.3577 |
| Coherence (Failure) | 0.1105 |
| Gradient Magnitude (Success) | 0.2757 |
| Gradient Magnitude (Failure) | 0.0764 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3750 |
| Clusters | 1,522 |
| Noise Fraction | 0.0387 |

---

## Checkpoint 5 — 5k (5005 episodes)

**Episodes:** 5,005  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.173)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0518 |
| Coherence (Success) | 0.2922 |
| Coherence (Failure) | 0.0535 |
| Gradient Magnitude (Success) | 0.5618 |
| Gradient Magnitude (Failure) | 0.1056 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0008 |
| Clusters | 1,272 |
| Noise Fraction | 0.0324 |

---

## Checkpoint 6 — 6k (6016 episodes)

**Episodes:** 6,016  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.187)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0957 |
| Coherence (Success) | 0.1729 |
| Coherence (Failure) | 0.1004 |
| Gradient Magnitude (Success) | 0.3079 |
| Gradient Magnitude (Failure) | 0.1375 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9080 |
| Clusters | 1,256 |
| Noise Fraction | 0.0381 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.561 | 0.411 | 0.227 |
| locked_door | 0.561 | 0.000 | 0.212 | 0.892 |
| open_door | 0.411 | 0.212 | 0.000 | 0.713 |
| target_ball | 0.227 | 0.892 | 0.713 | 0.000 |

---

## Checkpoint 7 — 7k (7010 episodes)

**Episodes:** 7,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.190)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2676 |
| Coherence (Success) | 0.3239 |
| Coherence (Failure) | 0.0189 |
| Gradient Magnitude (Success) | 0.4858 |
| Gradient Magnitude (Failure) | 0.0890 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0657 |
| Clusters | 1,261 |
| Noise Fraction | 0.0425 |

---

## Checkpoint 8 — 8k (8018 episodes)

**Episodes:** 8,018  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.777) | top 25% (≥ 2.213)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4233 |
| Coherence (Success) | 0.2396 |
| Coherence (Failure) | 0.2134 |
| Gradient Magnitude (Success) | 0.3688 |
| Gradient Magnitude (Failure) | 0.1837 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9578 |
| Clusters | 1,179 |
| Noise Fraction | 0.0490 |

---

## Checkpoint 9 — 9k (9003 episodes)

**Episodes:** 9,003  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.730) | top 25% (≥ 2.227)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0878 |
| Coherence (Success) | 0.2599 |
| Coherence (Failure) | 0.0878 |
| Gradient Magnitude (Success) | 0.4104 |
| Gradient Magnitude (Failure) | 0.1298 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3200 |
| Clusters | 1,216 |
| Noise Fraction | 0.0434 |

---

## Checkpoint 100 — 100k (100061 episodes)

**Episodes:** 100,061  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5197 |
| Coherence (Success) | 0.1636 |
| Coherence (Failure) | 0.1608 |
| Gradient Magnitude (Success) | 0.2463 |
| Gradient Magnitude (Failure) | 0.1888 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.4301 |
| Clusters | 549 |
| Noise Fraction | 0.0833 |
