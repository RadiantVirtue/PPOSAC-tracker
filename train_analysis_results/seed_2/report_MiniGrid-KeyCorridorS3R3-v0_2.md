# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 2  
**Total episodes:** 30,023  
**Experiment root:** `train_analysis_results\seed_2`  
**Generated:** 2026-02-24 01:39

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 3k (3000 episodes) | 3,000 | 0.7436 | 0.3297 | 0.2269 | 0.0866 | 0.0493 | 0.0406 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2 — 6k (6006 episodes) | 6,006 | 0.7084 | 0.4613 | 0.2562 | 0.1029 | 0.0784 | 0.0671 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 3 — 9k (9007 episodes) | 9,007 | 0.1667 | 0.3984 | 0.0698 | 0.1237 | 0.0749 | 0.3458 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 4 — 12k (12006 episodes) | 12,006 | 0.3519 | 0.3413 | 0.1614 | 0.2383 | 0.1063 | 0.6129 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 5 — 15k (15011 episodes) | 15,011 | -0.1122 | 0.3224 | 0.0175 | 0.3350 | 0.1292 | 0.8049 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6 — 18k (18028 episodes) | 18,028 | 0.1846 | 0.1706 | -0.0134 | 0.2331 | 0.2003 | 0.9598 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7 — 21k (21017 episodes) | 21,017 | 0.3553 | 0.3736 | -0.0013 | 0.3102 | 0.3310 | 1.2633 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8 — 24k (24036 episodes) | 24,036 | 0.1797 | 0.3581 | -0.1300 | 0.3265 | 0.2270 | 1.3718 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9 — 27k (27021 episodes) | 27,021 | -0.2799 | 0.2793 | -0.0604 | 0.2779 | 0.2696 | 1.7418 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 10 — 30k (30023 episodes) | 30,023 | — | 0.3595 | -0.0717 | 0.2648 | 0.3131 | 1.6692 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Final — 30k episodes | 30,023 | 0.0844 | 0.3668 | 0.1115 | 0.2828 | 0.2601 | 1.6301 | {'correlation': 0.0, 'p_value': 1.0} |

---

## Checkpoint 1 — 3k (3000 episodes)

**Episodes:** 3,000  
**Success:** 187  
**Failure:** 313  
**Threshold μ:** 3.724

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7436 |
| Coherence (Success) | 0.3297 |
| Coherence (Failure) | 0.2269 |
| Gradient Magnitude (Success) | 0.0866 |
| Gradient Magnitude (Failure) | 0.0493 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0406 |
| Clusters | 4,051 |
| Noise Fraction | 0.0302 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.234 | 0.017 | 0.142 |
| locked_door | 0.234 | 0.000 | 0.257 | 0.512 |
| open_door | 0.017 | 0.257 | -0.000 | 0.214 |
| target_ball | 0.142 | 0.512 | 0.214 | -0.000 |

---

## Checkpoint 2 — 6k (6006 episodes)

**Episodes:** 6,006  
**Success:** 220  
**Failure:** 280  
**Threshold μ:** 3.852

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7084 |
| Coherence (Success) | 0.4613 |
| Coherence (Failure) | 0.2562 |
| Gradient Magnitude (Success) | 0.1029 |
| Gradient Magnitude (Failure) | 0.0784 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0671 |
| Clusters | 3,965 |
| Noise Fraction | 0.0232 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.741 | 0.196 | 0.175 |
| locked_door | 0.741 | 0.000 | 0.752 | 0.933 |
| open_door | 0.196 | 0.752 | 0.000 | 0.449 |
| target_ball | 0.175 | 0.933 | 0.449 | 0.000 |

---

## Checkpoint 3 — 9k (9007 episodes)

**Episodes:** 9,007  
**Success:** 290  
**Failure:** 210  
**Threshold μ:** 4.500

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1667 |
| Coherence (Success) | 0.3984 |
| Coherence (Failure) | 0.0698 |
| Gradient Magnitude (Success) | 0.1237 |
| Gradient Magnitude (Failure) | 0.0749 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3458 |
| Clusters | 2,951 |
| Noise Fraction | 0.0323 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.695 | 0.464 | 0.153 |
| locked_door | 0.695 | 0.000 | 0.327 | 1.146 |
| open_door | 0.464 | 0.327 | 0.000 | 0.825 |
| target_ball | 0.153 | 1.146 | 0.825 | 0.000 |

---

## Checkpoint 4 — 12k (12006 episodes)

**Episodes:** 12,006  
**Success:** 351  
**Failure:** 149  
**Threshold μ:** 5.322

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3519 |
| Coherence (Success) | 0.3413 |
| Coherence (Failure) | 0.1614 |
| Gradient Magnitude (Success) | 0.2383 |
| Gradient Magnitude (Failure) | 0.1063 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.6129 |
| Clusters | 1,939 |
| Noise Fraction | 0.0411 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.661 | 0.455 | 0.156 |
| locked_door | 0.661 | 0.000 | 0.281 | 0.990 |
| open_door | 0.455 | 0.281 | -0.000 | 0.709 |
| target_ball | 0.156 | 0.990 | 0.709 | 0.000 |

---

## Checkpoint 5 — 15k (15011 episodes)

**Episodes:** 15,011  
**Success:** 433  
**Failure:** 67  
**Threshold μ:** 5.720

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1122 |
| Coherence (Success) | 0.3224 |
| Coherence (Failure) | 0.0175 |
| Gradient Magnitude (Success) | 0.3350 |
| Gradient Magnitude (Failure) | 0.1292 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.8049 |
| Clusters | 1,349 |
| Noise Fraction | 0.0525 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.535 | 0.189 | 0.239 |
| locked_door | 0.535 | 0.000 | 0.362 | 1.042 |
| open_door | 0.189 | 0.362 | -0.000 | 0.489 |
| target_ball | 0.239 | 1.042 | 0.489 | -0.000 |

---

## Checkpoint 6 — 18k (18028 episodes)

**Episodes:** 18,028  
**Success:** 455  
**Failure:** 45  
**Threshold μ:** 5.790

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1846 |
| Coherence (Success) | 0.1706 |
| Coherence (Failure) | -0.0134 |
| Gradient Magnitude (Success) | 0.2331 |
| Gradient Magnitude (Failure) | 0.2003 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9598 |
| Clusters | 1,161 |
| Noise Fraction | 0.0490 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.429 | 0.059 | 0.207 |
| locked_door | 0.429 | 0.000 | 0.426 | 0.955 |
| open_door | 0.059 | 0.426 | -0.000 | 0.313 |
| target_ball | 0.207 | 0.955 | 0.313 | 0.000 |

---

## Checkpoint 7 — 21k (21017 episodes)

**Episodes:** 21,017  
**Success:** 480  
**Failure:** 20  
**Threshold μ:** 5.906

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3553 |
| Coherence (Success) | 0.3736 |
| Coherence (Failure) | -0.0013 |
| Gradient Magnitude (Success) | 0.3102 |
| Gradient Magnitude (Failure) | 0.3310 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.2633 |
| Clusters | 959 |
| Noise Fraction | 0.0624 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.257 | 0.041 | 0.177 |
| locked_door | 0.257 | 0.000 | 0.287 | 0.683 |
| open_door | 0.041 | 0.287 | 0.000 | 0.248 |
| target_ball | 0.177 | 0.683 | 0.248 | 0.000 |

---

## Checkpoint 8 — 24k (24036 episodes)

**Episodes:** 24,036  
**Success:** 480  
**Failure:** 20  
**Threshold μ:** 5.944

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1797 |
| Coherence (Success) | 0.3581 |
| Coherence (Failure) | -0.1300 |
| Gradient Magnitude (Success) | 0.3265 |
| Gradient Magnitude (Failure) | 0.2270 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3718 |
| Clusters | 822 |
| Noise Fraction | 0.0602 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.344 | 0.050 | 0.101 |
| locked_door | 0.344 | 0.000 | 0.498 | 0.647 |
| open_door | 0.050 | 0.498 | -0.000 | 0.102 |
| target_ball | 0.101 | 0.647 | 0.102 | -0.000 |

---

## Checkpoint 9 — 27k (27021 episodes)

**Episodes:** 27,021  
**Success:** 485  
**Failure:** 15  
**Threshold μ:** 5.950

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2799 |
| Coherence (Success) | 0.2793 |
| Coherence (Failure) | -0.0604 |
| Gradient Magnitude (Success) | 0.2779 |
| Gradient Magnitude (Failure) | 0.2696 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7418 |
| Clusters | 815 |
| Noise Fraction | 0.0670 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.257 | 0.041 | 0.111 |
| locked_door | 0.257 | 0.000 | 0.321 | 0.540 |
| open_door | 0.041 | 0.321 | 0.000 | 0.167 |
| target_ball | 0.111 | 0.540 | 0.167 | 0.000 |

---

## Checkpoint 10 — 30k (30023 episodes)

**Episodes:** 30,023  
**Success:** 493  
**Failure:** 7  
**Threshold μ:** 5.978

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | — |
| Coherence (Success) | 0.3595 |
| Coherence (Failure) | -0.0717 |
| Gradient Magnitude (Success) | 0.2648 |
| Gradient Magnitude (Failure) | 0.3131 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.6692 |
| Clusters | 833 |
| Noise Fraction | 0.0639 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.267 | 0.031 | 0.104 |
| locked_door | 0.267 | 0.000 | 0.262 | 0.593 |
| open_door | 0.031 | 0.262 | 0.000 | 0.158 |
| target_ball | 0.104 | 0.593 | 0.158 | 0.000 |

---

## Final — 30k episodes

**Episodes:** 30,023  
**Success:** 484  
**Failure:** 16  
**Threshold μ:** 5.940

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0844 |
| Coherence (Success) | 0.3668 |
| Coherence (Failure) | 0.1115 |
| Gradient Magnitude (Success) | 0.2828 |
| Gradient Magnitude (Failure) | 0.2601 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.6301 |
| Clusters | 829 |
| Noise Fraction | 0.0727 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.308 | 0.026 | 0.086 |
| locked_door | 0.308 | 0.000 | 0.324 | 0.596 |
| open_door | 0.026 | 0.324 | 0.000 | 0.126 |
| target_ball | 0.086 | 0.596 | 0.126 | -0.000 |
