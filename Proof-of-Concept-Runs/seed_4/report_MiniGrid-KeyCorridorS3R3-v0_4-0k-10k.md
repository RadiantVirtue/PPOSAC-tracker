# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`
**Seed:** 4
**Total episodes:** 100,072
**Experiment root:** `Proof-of-Concept-Runs\seed_4`
**Generated:** 2026-02-25 12:42

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1009 episodes) | 1,009 | 0.8972 | 0.4063 | 0.2864 | 0.1031 | 0.0688 | 0.0764 | — | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 2 — 2k (2004 episodes) | 2,004 | 0.7458 | 0.4271 | 0.2544 | 0.0954 | 0.0735 | 0.0514 | 0.0514 | — |
| Checkpoint 3 — 3k (3001 episodes) | 3,001 | 0.7956 | 0.3304 | 0.1419 | 0.0947 | 0.0623 | 0.0447 | 0.0481 | — |
| Checkpoint 4 — 4k (4008 episodes) | 4,008 | 0.4162 | 0.3298 | 0.2446 | 0.2261 | 0.0836 | 0.1651 | 0.3493 | — |
| Checkpoint 5 — 5k (5005 episodes) | 5,005 | 0.0624 | 0.3600 | 0.1797 | 0.3643 | 0.0854 | 0.6027 | 1.0660 | — |
| Checkpoint 6 — 6k (6018 episodes) | 6,018 | -0.0342 | 0.4008 | 0.0764 | 0.4225 | 0.0898 | 0.8421 | — | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7 — 7k (7028 episodes) | 7,028 | -0.0765 | 0.1634 | 0.0071 | 0.3345 | 0.0750 | 1.0679 | 0.9165 | — |
| Checkpoint 8 — 8k (8000 episodes) | 8,000 | 0.3392 | 0.2083 | 0.0697 | 0.4655 | 0.1012 | 0.9136 | 0.7015 | — |
| Checkpoint 9 — 9k (9031 episodes) | 9,031 | 0.5232 | 0.1882 | 0.0552 | 0.3504 | 0.1150 | 1.0419 | 0.6739 | — |
| Checkpoint 100 — 100k (100072 episodes) | 100,072 | 0.4688 | 0.1255 | 0.1145 | 0.2110 | 0.1629 | 10.2518 | — | — |

---

## Checkpoint 1 — 1k (1009 episodes)

**Episodes:** 1,009
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8972 |
| Coherence (Success) | 0.4063 |
| Coherence (Failure) | 0.2864 |
| Gradient Magnitude (Success) | 0.1031 |
| Gradient Magnitude (Failure) | 0.0688 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0764 |
| Cosine Distance | — |
| Clusters | 2,097 |
| Noise Fraction | 0.0359 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.214 | 0.156 | 0.102 |
| locked_door | 0.214 | 0.000 | 0.208 | 0.304 |
| open_door | 0.156 | 0.208 | 0.000 | 0.432 |
| target_ball | 0.102 | 0.304 | 0.432 | 0.000 |

---

## Checkpoint 2 — 2k (2004 episodes)

**Episodes:** 2,004
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7458 |
| Coherence (Success) | 0.4271 |
| Coherence (Failure) | 0.2544 |
| Gradient Magnitude (Success) | 0.0954 |
| Gradient Magnitude (Failure) | 0.0735 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0514 |
| Cosine Distance | 0.0514 |
| Clusters | 2,139 |
| Noise Fraction | 0.0279 |

---

## Checkpoint 3 — 3k (3001 episodes)

**Episodes:** 3,001
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7956 |
| Coherence (Success) | 0.3304 |
| Coherence (Failure) | 0.1419 |
| Gradient Magnitude (Success) | 0.0947 |
| Gradient Magnitude (Failure) | 0.0623 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0447 |
| Cosine Distance | 0.0481 |
| Clusters | 1,968 |
| Noise Fraction | 0.0276 |

---

## Checkpoint 4 — 4k (4008 episodes)

**Episodes:** 4,008
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4162 |
| Coherence (Success) | 0.3298 |
| Coherence (Failure) | 0.2446 |
| Gradient Magnitude (Success) | 0.2261 |
| Gradient Magnitude (Failure) | 0.0836 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1651 |
| Cosine Distance | 0.3493 |
| Clusters | 1,586 |
| Noise Fraction | 0.0281 |

---

## Checkpoint 5 — 5k (5005 episodes)

**Episodes:** 5,005
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.867)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0624 |
| Coherence (Success) | 0.3600 |
| Coherence (Failure) | 0.1797 |
| Gradient Magnitude (Success) | 0.3643 |
| Gradient Magnitude (Failure) | 0.0854 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.6027 |
| Cosine Distance | 1.0660 |
| Clusters | 1,312 |
| Noise Fraction | 0.0312 |

---

## Checkpoint 6 — 6k (6018 episodes)

**Episodes:** 6,018
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.107)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0342 |
| Coherence (Success) | 0.4008 |
| Coherence (Failure) | 0.0764 |
| Gradient Magnitude (Success) | 0.4225 |
| Gradient Magnitude (Failure) | 0.0898 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.8421 |
| Cosine Distance | — |
| Clusters | 1,122 |
| Noise Fraction | 0.0357 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.224 | 0.080 | 0.175 |
| locked_door | 0.224 | 0.000 | 0.127 | 0.594 |
| open_door | 0.080 | 0.127 | 0.000 | 0.350 |
| target_ball | 0.175 | 0.594 | 0.350 | -0.000 |

---

## Checkpoint 7 — 7k (7028 episodes)

**Episodes:** 7,028
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.223)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0765 |
| Coherence (Success) | 0.1634 |
| Coherence (Failure) | 0.0071 |
| Gradient Magnitude (Success) | 0.3345 |
| Gradient Magnitude (Failure) | 0.0750 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0679 |
| Cosine Distance | 0.9165 |
| Clusters | 1,228 |
| Noise Fraction | 0.0314 |

---

## Checkpoint 8 — 8k (8000 episodes)

**Episodes:** 8,000
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.190)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3392 |
| Coherence (Success) | 0.2083 |
| Coherence (Failure) | 0.0697 |
| Gradient Magnitude (Success) | 0.4655 |
| Gradient Magnitude (Failure) | 0.1012 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9136 |
| Cosine Distance | 0.7015 |
| Clusters | 1,221 |
| Noise Fraction | 0.0441 |

---

## Checkpoint 9 — 9k (9031 episodes)

**Episodes:** 9,031
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 1.793) | top 25% (≥ 2.240)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5232 |
| Coherence (Success) | 0.1882 |
| Coherence (Failure) | 0.0552 |
| Gradient Magnitude (Success) | 0.3504 |
| Gradient Magnitude (Failure) | 0.1150 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0419 |
| Cosine Distance | 0.6739 |
| Clusters | 1,118 |
| Noise Fraction | 0.0450 |

---

## Checkpoint 100 — 100k (100072 episodes)

**Episodes:** 100,072
**Success:** 125
**Failure:** 125
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4688 |
| Coherence (Success) | 0.1255 |
| Coherence (Failure) | 0.1145 |
| Gradient Magnitude (Success) | 0.2110 |
| Gradient Magnitude (Failure) | 0.1629 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.2518 |
| Cosine Distance | — |
| Clusters | 524 |
| Noise Fraction | 0.0945 |
