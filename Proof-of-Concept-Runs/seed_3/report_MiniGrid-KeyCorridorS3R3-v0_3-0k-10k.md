# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 3  
**Total episodes:** 100,040  
**Experiment root:** `Proof-of-Concept-Runs\seed_3`  
**Generated:** 2026-02-25 11:57

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1010 episodes) | 1,010 | 0.8407 | 0.2874 | 0.3233 | 0.0697 | 0.0800 | 0.0637 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 2 — 2k (2008 episodes) | 2,008 | 0.8296 | 0.2943 | 0.1764 | 0.0715 | 0.0667 | 0.0823 | — |
| Checkpoint 3 — 3k (3008 episodes) | 3,008 | 0.8135 | 0.2880 | 0.3948 | 0.0973 | 0.0988 | 0.0563 | — |
| Checkpoint 4 — 4k (4002 episodes) | 4,002 | 0.4409 | 0.2938 | 0.3324 | 0.3080 | 0.1392 | 0.1602 | — |
| Checkpoint 5 — 5k (5015 episodes) | 5,015 | 0.1322 | 0.2271 | 0.0400 | 0.2722 | 0.0462 | 0.3155 | — |
| Checkpoint 6 — 6k (6019 episodes) | 6,019 | 0.3814 | 0.2535 | 0.1191 | 0.2837 | 0.0945 | 0.7106 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7 — 7k (7020 episodes) | 7,020 | 0.4464 | 0.1817 | 0.0096 | 0.2055 | 0.0680 | 0.7434 | — |
| Checkpoint 8 — 8k (8022 episodes) | 8,022 | -0.2938 | 0.3472 | 0.0842 | 0.4210 | 0.0700 | 1.1024 | — |
| Checkpoint 9 — 9k (9005 episodes) | 9,005 | 0.2370 | 0.1858 | -0.0071 | 0.4067 | 0.0505 | 1.3713 | — |
| Checkpoint 100 — 100k (100040 episodes) | 100,040 | 0.1653 | 0.0928 | 0.2036 | 0.1844 | 0.1717 | 10.8205 | — |

---

## Checkpoint 1 — 1k (1010 episodes)

**Episodes:** 1,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8407 |
| Coherence (Success) | 0.2874 |
| Coherence (Failure) | 0.3233 |
| Gradient Magnitude (Success) | 0.0697 |
| Gradient Magnitude (Failure) | 0.0800 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0637 |
| Clusters | 2,157 |
| Noise Fraction | 0.0347 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.250 | 0.209 | 0.122 |
| locked_door | 0.250 | 0.000 | 0.121 | 0.559 |
| open_door | 0.209 | 0.121 | 0.000 | 0.509 |
| target_ball | 0.122 | 0.559 | 0.509 | 0.000 |

---

## Checkpoint 2 — 2k (2008 episodes)

**Episodes:** 2,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8296 |
| Coherence (Success) | 0.2943 |
| Coherence (Failure) | 0.1764 |
| Gradient Magnitude (Success) | 0.0715 |
| Gradient Magnitude (Failure) | 0.0667 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0823 |
| Clusters | 2,054 |
| Noise Fraction | 0.0284 |

---

## Checkpoint 3 — 3k (3008 episodes)

**Episodes:** 3,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8135 |
| Coherence (Success) | 0.2880 |
| Coherence (Failure) | 0.3948 |
| Gradient Magnitude (Success) | 0.0973 |
| Gradient Magnitude (Failure) | 0.0988 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0563 |
| Clusters | 1,900 |
| Noise Fraction | 0.0227 |

---

## Checkpoint 4 — 4k (4002 episodes)

**Episodes:** 4,002  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4409 |
| Coherence (Success) | 0.2938 |
| Coherence (Failure) | 0.3324 |
| Gradient Magnitude (Success) | 0.3080 |
| Gradient Magnitude (Failure) | 0.1392 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1602 |
| Clusters | 1,568 |
| Noise Fraction | 0.0210 |

---

## Checkpoint 5 — 5k (5015 episodes)

**Episodes:** 5,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.587)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1322 |
| Coherence (Success) | 0.2271 |
| Coherence (Failure) | 0.0400 |
| Gradient Magnitude (Success) | 0.2722 |
| Gradient Magnitude (Failure) | 0.0462 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3155 |
| Clusters | 1,200 |
| Noise Fraction | 0.0196 |

---

## Checkpoint 6 — 6k (6019 episodes)

**Episodes:** 6,019  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.877)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3814 |
| Coherence (Success) | 0.2535 |
| Coherence (Failure) | 0.1191 |
| Gradient Magnitude (Success) | 0.2837 |
| Gradient Magnitude (Failure) | 0.0945 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.7106 |
| Clusters | 1,028 |
| Noise Fraction | 0.0161 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.478 | 0.522 | 0.614 |
| locked_door | 0.478 | 0.000 | 0.219 | 1.416 |
| open_door | 0.522 | 0.219 | 0.000 | 1.447 |
| target_ball | 0.614 | 1.416 | 1.447 | 0.000 |

---

## Checkpoint 7 — 7k (7020 episodes)

**Episodes:** 7,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.950)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4464 |
| Coherence (Success) | 0.1817 |
| Coherence (Failure) | 0.0096 |
| Gradient Magnitude (Success) | 0.2055 |
| Gradient Magnitude (Failure) | 0.0680 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.7434 |
| Clusters | 1,084 |
| Noise Fraction | 0.0192 |

---

## Checkpoint 8 — 8k (8022 episodes)

**Episodes:** 8,022  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.140)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2938 |
| Coherence (Success) | 0.3472 |
| Coherence (Failure) | 0.0842 |
| Gradient Magnitude (Success) | 0.4210 |
| Gradient Magnitude (Failure) | 0.0700 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.1024 |
| Clusters | 983 |
| Noise Fraction | 0.0178 |

---

## Checkpoint 9 — 9k (9005 episodes)

**Episodes:** 9,005  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.213)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2370 |
| Coherence (Success) | 0.1858 |
| Coherence (Failure) | -0.0071 |
| Gradient Magnitude (Success) | 0.4067 |
| Gradient Magnitude (Failure) | 0.0505 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3713 |
| Clusters | 1,052 |
| Noise Fraction | 0.0236 |

---

## Checkpoint 100 — 100k (100040 episodes)

**Episodes:** 100,040  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1653 |
| Coherence (Success) | 0.0928 |
| Coherence (Failure) | 0.2036 |
| Gradient Magnitude (Success) | 0.1844 |
| Gradient Magnitude (Failure) | 0.1717 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.8205 |
| Clusters | 553 |
| Noise Fraction | 0.0863 |
