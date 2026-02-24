# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 5  
**Total episodes:** 30,002  
**Experiment root:** `train_analysis_results\seed_5`  
**Generated:** 2026-02-24 09:46

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 3k (3008 episodes) | 3,008 | 0.7969 | 0.3977 | 0.2297 | 0.0920 | 0.0597 | 0.0379 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2 — 6k (6007 episodes) | 6,007 | 0.7359 | 0.2588 | 0.4928 | 0.0810 | 0.0656 | 0.0420 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 3 — 9k (9001 episodes) | 9,001 | 0.8662 | 0.3895 | 0.3138 | 0.0898 | 0.0690 | 0.0717 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4 — 12k (12001 episodes) | 12,001 | 0.7739 | 0.2505 | 0.3291 | 0.0651 | 0.0583 | 0.0343 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5 — 15k (15005 episodes) | 15,005 | 0.6689 | 0.2702 | 0.2927 | 0.0713 | 0.0551 | 0.0373 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6 — 18k (18000 episodes) | 18,000 | 0.7534 | 0.2203 | 0.2293 | 0.0841 | 0.0486 | 0.0565 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7 — 21k (21002 episodes) | 21,002 | 0.6228 | 0.2798 | 0.2747 | 0.0741 | 0.0531 | 0.0454 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8 — 24k (24000 episodes) | 24,000 | 0.4422 | 0.3303 | 0.3278 | 0.0789 | 0.0644 | 0.0739 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9 — 27k (27000 episodes) | 27,000 | 0.6639 | 0.4134 | 0.1701 | 0.0677 | 0.0333 | 0.0489 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 10 — 30k (30002 episodes) | 30,002 | 0.5703 | 0.5368 | 0.2380 | 0.1105 | 0.0470 | 0.0433 | {'correlation': 0.0, 'p_value': 1.0} |
| Final — 30k episodes | 30,002 | 0.6984 | 0.5184 | 0.3301 | 0.0984 | 0.0749 | 0.0458 | {'correlation': 0.0, 'p_value': 1.0} |

---

## Checkpoint 1 — 3k (3008 episodes)

**Episodes:** 3,008  
**Success:** 229  
**Failure:** 271  
**Threshold μ:** 3.886

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7969 |
| Coherence (Success) | 0.3977 |
| Coherence (Failure) | 0.2297 |
| Gradient Magnitude (Success) | 0.0920 |
| Gradient Magnitude (Failure) | 0.0597 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0379 |
| Clusters | 4,176 |
| Noise Fraction | 0.0291 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.102 | 0.047 | 0.133 |
| locked_door | 0.102 | 0.000 | 0.141 | 0.149 |
| open_door | 0.047 | 0.141 | -0.000 | 0.200 |
| target_ball | 0.133 | 0.149 | 0.200 | 0.000 |

---

## Checkpoint 2 — 6k (6007 episodes)

**Episodes:** 6,007  
**Success:** 146  
**Failure:** 354  
**Threshold μ:** 3.530

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7359 |
| Coherence (Success) | 0.2588 |
| Coherence (Failure) | 0.4928 |
| Gradient Magnitude (Success) | 0.0810 |
| Gradient Magnitude (Failure) | 0.0656 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0420 |
| Clusters | 3,860 |
| Noise Fraction | 0.0258 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.615 | 0.017 | 0.192 |
| locked_door | 0.615 | 0.000 | 0.602 | 0.644 |
| open_door | 0.017 | 0.602 | 0.000 | 0.215 |
| target_ball | 0.192 | 0.644 | 0.215 | 0.000 |

---

## Checkpoint 3 — 9k (9001 episodes)

**Episodes:** 9,001  
**Success:** 248  
**Failure:** 252  
**Threshold μ:** 3.958

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8662 |
| Coherence (Success) | 0.3895 |
| Coherence (Failure) | 0.3138 |
| Gradient Magnitude (Success) | 0.0898 |
| Gradient Magnitude (Failure) | 0.0690 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0717 |
| Clusters | 3,342 |
| Noise Fraction | 0.0257 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.095 | 0.045 | 0.509 |
| locked_door | 0.095 | 0.000 | 0.086 | 0.704 |
| open_door | 0.045 | 0.086 | 0.000 | 0.654 |
| target_ball | 0.509 | 0.704 | 0.654 | 0.000 |

---

## Checkpoint 4 — 12k (12001 episodes)

**Episodes:** 12,001  
**Success:** 206  
**Failure:** 294  
**Threshold μ:** 3.802

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7739 |
| Coherence (Success) | 0.2505 |
| Coherence (Failure) | 0.3291 |
| Gradient Magnitude (Success) | 0.0651 |
| Gradient Magnitude (Failure) | 0.0583 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0343 |
| Clusters | 3,426 |
| Noise Fraction | 0.0156 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.305 | 0.061 | 0.792 |
| locked_door | 0.305 | -0.000 | 0.404 | 0.755 |
| open_door | 0.061 | 0.404 | 0.000 | 1.035 |
| target_ball | 0.792 | 0.755 | 1.035 | -0.000 |

---

## Checkpoint 5 — 15k (15005 episodes)

**Episodes:** 15,005  
**Success:** 167  
**Failure:** 333  
**Threshold μ:** 3.634

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6689 |
| Coherence (Success) | 0.2702 |
| Coherence (Failure) | 0.2927 |
| Gradient Magnitude (Success) | 0.0713 |
| Gradient Magnitude (Failure) | 0.0551 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0373 |
| Clusters | 3,185 |
| Noise Fraction | 0.0116 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.449 | 0.012 | 0.071 |
| locked_door | 0.449 | 0.000 | 0.433 | 0.516 |
| open_door | 0.012 | 0.433 | 0.000 | 0.115 |
| target_ball | 0.071 | 0.516 | 0.115 | -0.000 |

---

## Checkpoint 6 — 18k (18000 episodes)

**Episodes:** 18,000  
**Success:** 194  
**Failure:** 306  
**Threshold μ:** 3.738

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7534 |
| Coherence (Success) | 0.2203 |
| Coherence (Failure) | 0.2293 |
| Gradient Magnitude (Success) | 0.0841 |
| Gradient Magnitude (Failure) | 0.0486 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0565 |
| Clusters | 3,352 |
| Noise Fraction | 0.0143 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.263 | 0.072 | 0.149 |
| locked_door | 0.263 | 0.000 | 0.235 | 0.327 |
| open_door | 0.072 | 0.235 | 0.000 | 0.230 |
| target_ball | 0.149 | 0.327 | 0.230 | 0.000 |

---

## Checkpoint 7 — 21k (21002 episodes)

**Episodes:** 21,002  
**Success:** 162  
**Failure:** 338  
**Threshold μ:** 3.634

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6228 |
| Coherence (Success) | 0.2798 |
| Coherence (Failure) | 0.2747 |
| Gradient Magnitude (Success) | 0.0741 |
| Gradient Magnitude (Failure) | 0.0531 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0454 |
| Clusters | 3,326 |
| Noise Fraction | 0.0164 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.389 | 0.016 | 0.064 |
| locked_door | 0.389 | -0.000 | 0.357 | 0.466 |
| open_door | 0.016 | 0.357 | 0.000 | 0.090 |
| target_ball | 0.064 | 0.466 | 0.090 | 0.000 |

---

## Checkpoint 8 — 24k (24000 episodes)

**Episodes:** 24,000  
**Success:** 175  
**Failure:** 325  
**Threshold μ:** 3.688

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4422 |
| Coherence (Success) | 0.3303 |
| Coherence (Failure) | 0.3278 |
| Gradient Magnitude (Success) | 0.0789 |
| Gradient Magnitude (Failure) | 0.0644 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0739 |
| Clusters | 3,188 |
| Noise Fraction | 0.0148 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.250 | 0.024 | 0.078 |
| locked_door | 0.250 | -0.000 | 0.148 | 0.377 |
| open_door | 0.024 | 0.148 | 0.000 | 0.132 |
| target_ball | 0.078 | 0.377 | 0.132 | -0.000 |

---

## Checkpoint 9 — 27k (27000 episodes)

**Episodes:** 27,000  
**Success:** 199  
**Failure:** 301  
**Threshold μ:** 3.744

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6639 |
| Coherence (Success) | 0.4134 |
| Coherence (Failure) | 0.1701 |
| Gradient Magnitude (Success) | 0.0677 |
| Gradient Magnitude (Failure) | 0.0333 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0489 |
| Clusters | 3,315 |
| Noise Fraction | 0.0229 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.307 | 0.062 | 0.053 |
| locked_door | 0.307 | 0.000 | 0.232 | 0.339 |
| open_door | 0.062 | 0.232 | 0.000 | 0.108 |
| target_ball | 0.053 | 0.339 | 0.108 | -0.000 |

---

## Checkpoint 10 — 30k (30002 episodes)

**Episodes:** 30,002  
**Success:** 206  
**Failure:** 294  
**Threshold μ:** 3.804

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5703 |
| Coherence (Success) | 0.5368 |
| Coherence (Failure) | 0.2380 |
| Gradient Magnitude (Success) | 0.1105 |
| Gradient Magnitude (Failure) | 0.0470 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0433 |
| Clusters | 3,315 |
| Noise Fraction | 0.0128 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.232 | 0.025 | 0.274 |
| locked_door | 0.232 | 0.000 | 0.250 | 0.536 |
| open_door | 0.025 | 0.250 | 0.000 | 0.338 |
| target_ball | 0.274 | 0.536 | 0.338 | 0.000 |

---

## Final — 30k episodes

**Episodes:** 30,002  
**Success:** 238  
**Failure:** 262  
**Threshold μ:** 3.880

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6984 |
| Coherence (Success) | 0.5184 |
| Coherence (Failure) | 0.3301 |
| Gradient Magnitude (Success) | 0.0984 |
| Gradient Magnitude (Failure) | 0.0749 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0458 |
| Clusters | 3,283 |
| Noise Fraction | 0.0154 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.188 | 0.030 | 0.043 |
| locked_door | 0.188 | 0.000 | 0.202 | 0.291 |
| open_door | 0.030 | 0.202 | 0.000 | 0.050 |
| target_ball | 0.043 | 0.291 | 0.050 | 0.000 |
