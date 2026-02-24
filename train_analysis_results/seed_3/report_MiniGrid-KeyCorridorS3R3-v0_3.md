# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 3  
**Total episodes:** 30,005  
**Experiment root:** `train_analysis_results\seed_3`  
**Generated:** 2026-02-24 04:55

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 3k (3000 episodes) | 3,000 | 0.7753 | 0.4539 | 0.3719 | 0.0833 | 0.0567 | 0.0496 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 2 — 6k (6001 episodes) | 6,001 | 0.6964 | 0.3821 | 0.4286 | 0.1170 | 0.0785 | 0.0612 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 3 — 9k (9000 episodes) | 9,000 | 0.5016 | 0.5287 | 0.3919 | 0.1271 | 0.0718 | 0.0429 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4 — 12k (12002 episodes) | 12,002 | 0.2762 | 0.1795 | 0.1497 | 0.0646 | 0.0443 | 0.1324 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5 — 15k (15002 episodes) | 15,002 | 0.7864 | 0.1939 | 0.2043 | 0.0577 | 0.0659 | 0.0851 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6 — 18k (18005 episodes) | 18,005 | 0.7974 | 0.3649 | 0.2258 | 0.0936 | 0.0612 | 0.0730 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7 — 21k (21004 episodes) | 21,004 | 0.4125 | 0.4031 | 0.2462 | 0.1203 | 0.0525 | 0.0314 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8 — 24k (24007 episodes) | 24,007 | 0.4537 | 0.2658 | 0.2160 | 0.1016 | 0.0477 | 0.0560 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9 — 27k (27000 episodes) | 27,000 | 0.2195 | 0.3190 | 0.1987 | 0.1047 | 0.0412 | 0.0302 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 10 — 30k (30005 episodes) | 30,005 | 0.7521 | 0.2907 | 0.1737 | 0.0852 | 0.0769 | 0.0471 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Final — 30k episodes | 30,005 | 0.5987 | 0.3194 | 0.3344 | 0.1034 | 0.0756 | 0.0458 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

---

## Checkpoint 1 — 3k (3000 episodes)

**Episodes:** 3,000  
**Success:** 183  
**Failure:** 317  
**Threshold μ:** 3.706

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7753 |
| Coherence (Success) | 0.4539 |
| Coherence (Failure) | 0.3719 |
| Gradient Magnitude (Success) | 0.0833 |
| Gradient Magnitude (Failure) | 0.0567 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0496 |
| Clusters | 4,145 |
| Noise Fraction | 0.0297 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.404 | 0.096 | 0.171 |
| locked_door | 0.404 | 0.000 | 0.476 | 0.460 |
| open_door | 0.096 | 0.476 | -0.000 | 0.360 |
| target_ball | 0.171 | 0.460 | 0.360 | 0.000 |

---

## Checkpoint 2 — 6k (6001 episodes)

**Episodes:** 6,001  
**Success:** 224  
**Failure:** 276  
**Threshold μ:** 3.860

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6964 |
| Coherence (Success) | 0.3821 |
| Coherence (Failure) | 0.4286 |
| Gradient Magnitude (Success) | 0.1170 |
| Gradient Magnitude (Failure) | 0.0785 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0612 |
| Clusters | 3,346 |
| Noise Fraction | 0.0183 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.452 | 0.047 | 0.112 |
| locked_door | 0.452 | -0.000 | 0.410 | 0.321 |
| open_door | 0.047 | 0.410 | 0.000 | 0.103 |
| target_ball | 0.112 | 0.321 | 0.103 | 0.000 |

---

## Checkpoint 3 — 9k (9000 episodes)

**Episodes:** 9,000  
**Success:** 207  
**Failure:** 293  
**Threshold μ:** 3.792

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5016 |
| Coherence (Success) | 0.5287 |
| Coherence (Failure) | 0.3919 |
| Gradient Magnitude (Success) | 0.1271 |
| Gradient Magnitude (Failure) | 0.0718 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0429 |
| Clusters | 3,305 |
| Noise Fraction | 0.0192 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.215 | 0.073 | 0.054 |
| locked_door | 0.215 | 0.000 | 0.237 | 0.286 |
| open_door | 0.073 | 0.237 | -0.000 | 0.131 |
| target_ball | 0.054 | 0.286 | 0.131 | -0.000 |

---

## Checkpoint 4 — 12k (12002 episodes)

**Episodes:** 12,002  
**Success:** 254  
**Failure:** 246  
**Threshold μ:** 3.996

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2762 |
| Coherence (Success) | 0.1795 |
| Coherence (Failure) | 0.1497 |
| Gradient Magnitude (Success) | 0.0646 |
| Gradient Magnitude (Failure) | 0.0443 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1324 |
| Clusters | 3,300 |
| Noise Fraction | 0.0294 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.491 | 0.081 | 0.135 |
| locked_door | 0.491 | 0.000 | 0.463 | 0.602 |
| open_door | 0.081 | 0.463 | 0.000 | 0.330 |
| target_ball | 0.135 | 0.602 | 0.330 | -0.000 |

---

## Checkpoint 5 — 15k (15002 episodes)

**Episodes:** 15,002  
**Success:** 251  
**Failure:** 249  
**Threshold μ:** 4.070

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7864 |
| Coherence (Success) | 0.1939 |
| Coherence (Failure) | 0.2043 |
| Gradient Magnitude (Success) | 0.0577 |
| Gradient Magnitude (Failure) | 0.0659 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0851 |
| Clusters | 3,394 |
| Noise Fraction | 0.0178 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.256 | 0.101 | 0.578 |
| locked_door | 0.256 | 0.000 | 0.153 | 0.884 |
| open_door | 0.101 | 0.153 | 0.000 | 0.941 |
| target_ball | 0.578 | 0.884 | 0.941 | 0.000 |

---

## Checkpoint 6 — 18k (18005 episodes)

**Episodes:** 18,005  
**Success:** 205  
**Failure:** 295  
**Threshold μ:** 3.792

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7974 |
| Coherence (Success) | 0.3649 |
| Coherence (Failure) | 0.2258 |
| Gradient Magnitude (Success) | 0.0936 |
| Gradient Magnitude (Failure) | 0.0612 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0730 |
| Clusters | 3,227 |
| Noise Fraction | 0.0275 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.427 | 0.026 | 0.093 |
| locked_door | 0.427 | -0.000 | 0.385 | 0.437 |
| open_door | 0.026 | 0.385 | 0.000 | 0.176 |
| target_ball | 0.093 | 0.437 | 0.176 | -0.000 |

---

## Checkpoint 7 — 21k (21004 episodes)

**Episodes:** 21,004  
**Success:** 192  
**Failure:** 308  
**Threshold μ:** 3.746

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4125 |
| Coherence (Success) | 0.4031 |
| Coherence (Failure) | 0.2462 |
| Gradient Magnitude (Success) | 0.1203 |
| Gradient Magnitude (Failure) | 0.0525 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0314 |
| Clusters | 3,441 |
| Noise Fraction | 0.0117 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.186 | 0.013 | 0.880 |
| locked_door | 0.186 | 0.000 | 0.169 | 1.016 |
| open_door | 0.013 | 0.169 | 0.000 | 0.972 |
| target_ball | 0.880 | 1.016 | 0.972 | 0.000 |

---

## Checkpoint 8 — 24k (24007 episodes)

**Episodes:** 24,007  
**Success:** 162  
**Failure:** 338  
**Threshold μ:** 3.632

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4537 |
| Coherence (Success) | 0.2658 |
| Coherence (Failure) | 0.2160 |
| Gradient Magnitude (Success) | 0.1016 |
| Gradient Magnitude (Failure) | 0.0477 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0560 |
| Clusters | 3,380 |
| Noise Fraction | 0.0156 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.274 | 0.029 | 0.019 |
| locked_door | 0.274 | -0.000 | 0.311 | 0.282 |
| open_door | 0.029 | 0.311 | -0.000 | 0.031 |
| target_ball | 0.019 | 0.282 | 0.031 | 0.000 |

---

## Checkpoint 9 — 27k (27000 episodes)

**Episodes:** 27,000  
**Success:** 209  
**Failure:** 291  
**Threshold μ:** 3.782

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2195 |
| Coherence (Success) | 0.3190 |
| Coherence (Failure) | 0.1987 |
| Gradient Magnitude (Success) | 0.1047 |
| Gradient Magnitude (Failure) | 0.0412 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0302 |
| Clusters | 3,304 |
| Noise Fraction | 0.0208 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.318 | 0.023 | 0.192 |
| locked_door | 0.318 | 0.000 | 0.292 | 0.437 |
| open_door | 0.023 | 0.292 | 0.000 | 0.270 |
| target_ball | 0.192 | 0.437 | 0.270 | -0.000 |

---

## Checkpoint 10 — 30k (30005 episodes)

**Episodes:** 30,005  
**Success:** 207  
**Failure:** 293  
**Threshold μ:** 3.790

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7521 |
| Coherence (Success) | 0.2907 |
| Coherence (Failure) | 0.1737 |
| Gradient Magnitude (Success) | 0.0852 |
| Gradient Magnitude (Failure) | 0.0769 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0471 |
| Clusters | 3,278 |
| Noise Fraction | 0.0151 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.133 | 0.033 | 0.346 |
| locked_door | 0.133 | -0.000 | 0.116 | 0.568 |
| open_door | 0.033 | 0.116 | 0.000 | 0.535 |
| target_ball | 0.346 | 0.568 | 0.535 | 0.000 |

---

## Final — 30k episodes

**Episodes:** 30,005  
**Success:** 211  
**Failure:** 289  
**Threshold μ:** 3.818

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5987 |
| Coherence (Success) | 0.3194 |
| Coherence (Failure) | 0.3344 |
| Gradient Magnitude (Success) | 0.1034 |
| Gradient Magnitude (Failure) | 0.0756 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0458 |
| Clusters | 3,192 |
| Noise Fraction | 0.0099 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.123 | 0.032 | 0.199 |
| locked_door | 0.123 | 0.000 | 0.105 | 0.391 |
| open_door | 0.032 | 0.105 | 0.000 | 0.369 |
| target_ball | 0.199 | 0.391 | 0.369 | 0.000 |
