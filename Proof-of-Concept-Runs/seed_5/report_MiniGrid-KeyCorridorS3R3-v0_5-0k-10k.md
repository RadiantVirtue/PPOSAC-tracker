# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 5  
**Total episodes:** 100,020  
**Experiment root:** `Proof-of-Concept-Runs\seed_5`  
**Generated:** 2026-02-25 13:20

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1010 episodes) | 1,010 | 0.8700 | 0.4090 | 0.3122 | 0.1248 | 0.0785 | 0.0619 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2 — 2k (2008 episodes) | 2,008 | 0.8784 | 0.2775 | 0.2924 | 0.0777 | 0.0768 | 0.0452 | — |
| Checkpoint 3 — 3k (3003 episodes) | 3,003 | 0.9404 | 0.2825 | 0.2880 | 0.1014 | 0.0920 | 0.0318 | — |
| Checkpoint 4 — 4k (4000 episodes) | 4,000 | 0.8845 | 0.1848 | 0.2059 | 0.0777 | 0.0790 | 0.0393 | — |
| Checkpoint 5 — 5k (5007 episodes) | 5,007 | 0.9283 | 0.2974 | 0.3404 | 0.0968 | 0.0957 | 0.0589 | — |
| Checkpoint 6 — 6k (6006 episodes) | 6,006 | 0.8048 | 0.3757 | 0.3400 | 0.1622 | 0.1215 | 0.0872 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7 — 7k (7001 episodes) | 7,001 | 0.7387 | 0.2182 | 0.3190 | 0.1948 | 0.1063 | 0.2369 | — |
| Checkpoint 8 — 8k (8009 episodes) | 8,009 | 0.6782 | 0.2091 | 0.0777 | 0.2689 | 0.0836 | 0.8688 | — |
| Checkpoint 9 — 9k (9014 episodes) | 9,014 | 0.6189 | 0.2749 | 0.1921 | 0.3973 | 0.1217 | 1.3195 | — |
| Checkpoint 100 — 100k (100020 episodes) | 100,020 | 0.5307 | 0.1765 | 0.1937 | 0.2336 | 0.2315 | 5.8891 | — |

---

## Checkpoint 1 — 1k (1010 episodes)

**Episodes:** 1,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8700 |
| Coherence (Success) | 0.4090 |
| Coherence (Failure) | 0.3122 |
| Gradient Magnitude (Success) | 0.1248 |
| Gradient Magnitude (Failure) | 0.0785 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0619 |
| Clusters | 2,127 |
| Noise Fraction | 0.0349 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.177 | 0.175 | 0.139 |
| locked_door | 0.177 | 0.000 | 0.221 | 0.449 |
| open_door | 0.175 | 0.221 | 0.000 | 0.323 |
| target_ball | 0.139 | 0.449 | 0.323 | 0.000 |

---

## Checkpoint 2 — 2k (2008 episodes)

**Episodes:** 2,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8784 |
| Coherence (Success) | 0.2775 |
| Coherence (Failure) | 0.2924 |
| Gradient Magnitude (Success) | 0.0777 |
| Gradient Magnitude (Failure) | 0.0768 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0452 |
| Clusters | 2,095 |
| Noise Fraction | 0.0326 |

---

## Checkpoint 3 — 3k (3003 episodes)

**Episodes:** 3,003  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9404 |
| Coherence (Success) | 0.2825 |
| Coherence (Failure) | 0.2880 |
| Gradient Magnitude (Success) | 0.1014 |
| Gradient Magnitude (Failure) | 0.0920 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0318 |
| Clusters | 2,088 |
| Noise Fraction | 0.0305 |

---

## Checkpoint 4 — 4k (4000 episodes)

**Episodes:** 4,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8845 |
| Coherence (Success) | 0.1848 |
| Coherence (Failure) | 0.2059 |
| Gradient Magnitude (Success) | 0.0777 |
| Gradient Magnitude (Failure) | 0.0790 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0393 |
| Clusters | 1,962 |
| Noise Fraction | 0.0235 |

---

## Checkpoint 5 — 5k (5007 episodes)

**Episodes:** 5,007  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9283 |
| Coherence (Success) | 0.2974 |
| Coherence (Failure) | 0.3404 |
| Gradient Magnitude (Success) | 0.0968 |
| Gradient Magnitude (Failure) | 0.0957 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0589 |
| Clusters | 2,048 |
| Noise Fraction | 0.0266 |

---

## Checkpoint 6 — 6k (6006 episodes)

**Episodes:** 6,006  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8048 |
| Coherence (Success) | 0.3757 |
| Coherence (Failure) | 0.3400 |
| Gradient Magnitude (Success) | 0.1622 |
| Gradient Magnitude (Failure) | 0.1215 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0872 |
| Clusters | 2,025 |
| Noise Fraction | 0.0329 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.008 | 0.204 |
| locked_door | 0.004 | 0.000 | 0.007 | 0.245 |
| open_door | 0.008 | 0.007 | 0.000 | 0.252 |
| target_ball | 0.204 | 0.245 | 0.252 | 0.000 |

---

## Checkpoint 7 — 7k (7001 episodes)

**Episodes:** 7,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7387 |
| Coherence (Success) | 0.2182 |
| Coherence (Failure) | 0.3190 |
| Gradient Magnitude (Success) | 0.1948 |
| Gradient Magnitude (Failure) | 0.1063 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2369 |
| Clusters | 1,852 |
| Noise Fraction | 0.0372 |

---

## Checkpoint 8 — 8k (8009 episodes)

**Episodes:** 8,009  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.967)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6782 |
| Coherence (Success) | 0.2091 |
| Coherence (Failure) | 0.0777 |
| Gradient Magnitude (Success) | 0.2689 |
| Gradient Magnitude (Failure) | 0.0836 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.8688 |
| Clusters | 1,358 |
| Noise Fraction | 0.0338 |

---

## Checkpoint 9 — 9k (9014 episodes)

**Episodes:** 9,014  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.187)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6189 |
| Coherence (Success) | 0.2749 |
| Coherence (Failure) | 0.1921 |
| Gradient Magnitude (Success) | 0.3973 |
| Gradient Magnitude (Failure) | 0.1217 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3195 |
| Clusters | 1,245 |
| Noise Fraction | 0.0328 |

---

## Checkpoint 100 — 100k (100020 episodes)

**Episodes:** 100,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.193) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5307 |
| Coherence (Success) | 0.1765 |
| Coherence (Failure) | 0.1937 |
| Gradient Magnitude (Success) | 0.2336 |
| Gradient Magnitude (Failure) | 0.2315 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.8891 |
| Clusters | 453 |
| Noise Fraction | 0.0858 |
