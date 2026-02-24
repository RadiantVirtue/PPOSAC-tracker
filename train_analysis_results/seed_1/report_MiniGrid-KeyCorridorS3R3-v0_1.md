# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 1  
**Total episodes:** 30,036  
**Experiment root:** `train_analysis_results\seed_1`  

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 3k (3003 episodes) | 3,003 | 0.8814 | 0.3823 | 0.4943 | 0.0745 | 0.1052 | 0.0352 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2 — 6k (6007 episodes) | 6,007 | 0.8778 | 0.2251 | 0.4346 | 0.0823 | 0.1061 | 0.1013 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 3 — 9k (9005 episodes) | 9,005 | 0.5426 | 0.3519 | 0.1906 | 0.1385 | 0.0787 | 0.2022 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4 — 12k (12012 episodes) | 12,012 | 0.5192 | 0.2170 | 0.1644 | 0.2135 | 0.1110 | 0.5452 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 5 — 15k (15015 episodes) | 15,015 | 0.2440 | 0.3075 | 0.0199 | 0.2255 | 0.1336 | 0.9786 | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 6 — 18k (18023 episodes) | 18,023 | -0.0965 | 0.2777 | 0.0499 | 0.2929 | 0.1399 | 1.0017 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7 — 21k (21023 episodes) | 21,023 | 0.1032 | 0.3818 | 0.0492 | 0.2589 | 0.3448 | 1.0810 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 8 — 24k (24015 episodes) | 24,015 | 0.2208 | 0.4201 | 0.1203 | 0.3086 | 0.4248 | 1.6308 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9 — 27k (27016 episodes) | 27,016 | 0.0003 | 0.2596 | 0.1114 | 0.2284 | 0.3260 | 1.9843 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 10 — 30k (30036 episodes) | 30,036 | 0.2372 | 0.4144 | -0.0057 | 0.2986 | 0.2035 | 2.0030 | {'correlation': 0.0, 'p_value': 1.0} |
| Final — 30k episodes | 30,036 | 0.1688 | 0.3321 | 0.0343 | 0.2692 | 0.3338 | 1.9631 | {'correlation': 0.0, 'p_value': 1.0} |

---

## Checkpoint 1 — 3k (3003 episodes)

**Episodes:** 3,003  
**Success:** 245  
**Failure:** 255  
**Threshold μ:** 4.080

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8814 |
| Coherence (Success) | 0.3823 |
| Coherence (Failure) | 0.4943 |
| Gradient Magnitude (Success) | 0.0745 |
| Gradient Magnitude (Failure) | 0.1052 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0352 |
| Clusters | 3,390 |
| Noise Fraction | 0.0206 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.177 | 0.047 | 0.443 |
| locked_door | 0.177 | 0.000 | 0.202 | 0.550 |
| open_door | 0.047 | 0.202 | 0.000 | 0.543 |
| target_ball | 0.443 | 0.550 | 0.543 | 0.000 |

---

## Checkpoint 2 — 6k (6007 episodes)

**Episodes:** 6,007  
**Success:** 227  
**Failure:** 273  
**Threshold μ:** 4.046

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8778 |
| Coherence (Success) | 0.2251 |
| Coherence (Failure) | 0.4346 |
| Gradient Magnitude (Success) | 0.0823 |
| Gradient Magnitude (Failure) | 0.1061 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1013 |
| Clusters | 3,351 |
| Noise Fraction | 0.0126 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.223 | 0.197 | 0.487 |
| locked_door | 0.223 | -0.000 | 0.042 | 1.061 |
| open_door | 0.197 | 0.042 | 0.000 | 1.019 |
| target_ball | 0.487 | 1.061 | 1.019 | 0.000 |

---

## Checkpoint 3 — 9k (9005 episodes)

**Episodes:** 9,005  
**Success:** 282  
**Failure:** 218  
**Threshold μ:** 4.458

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5426 |
| Coherence (Success) | 0.3519 |
| Coherence (Failure) | 0.1906 |
| Gradient Magnitude (Success) | 0.1385 |
| Gradient Magnitude (Failure) | 0.0787 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2022 |
| Clusters | 2,672 |
| Noise Fraction | 0.0166 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.490 | 0.175 | 0.582 |
| locked_door | 0.490 | 0.000 | 0.310 | 0.582 |
| open_door | 0.175 | 0.310 | 0.000 | 0.865 |
| target_ball | 0.582 | 0.582 | 0.865 | 0.000 |

---

## Checkpoint 4 — 12k (12012 episodes)

**Episodes:** 12,012  
**Success:** 292  
**Failure:** 208  
**Threshold μ:** 4.598

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5192 |
| Coherence (Success) | 0.2170 |
| Coherence (Failure) | 0.1644 |
| Gradient Magnitude (Success) | 0.2135 |
| Gradient Magnitude (Failure) | 0.1110 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.5452 |
| Clusters | 2,215 |
| Noise Fraction | 0.0222 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.332 | 0.272 | 0.168 |
| locked_door | 0.332 | 0.000 | 0.032 | 0.430 |
| open_door | 0.272 | 0.032 | 0.000 | 0.354 |
| target_ball | 0.168 | 0.430 | 0.354 | 0.000 |

---

## Checkpoint 5 — 15k (15015 episodes)

**Episodes:** 15,015  
**Success:** 414  
**Failure:** 86  
**Threshold μ:** 5.658

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2440 |
| Coherence (Success) | 0.3075 |
| Coherence (Failure) | 0.0199 |
| Gradient Magnitude (Success) | 0.2255 |
| Gradient Magnitude (Failure) | 0.1336 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9786 |
| Clusters | 1,584 |
| Noise Fraction | 0.0499 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.187 | 0.107 | 0.164 |
| locked_door | 0.187 | -0.000 | 0.064 | 0.425 |
| open_door | 0.107 | 0.064 | 0.000 | 0.343 |
| target_ball | 0.164 | 0.425 | 0.343 | 0.000 |

---

## Checkpoint 6 — 18k (18023 episodes)

**Episodes:** 18,023  
**Success:** 453  
**Failure:** 47  
**Threshold μ:** 5.848

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0965 |
| Coherence (Success) | 0.2777 |
| Coherence (Failure) | 0.0499 |
| Gradient Magnitude (Success) | 0.2929 |
| Gradient Magnitude (Failure) | 0.1399 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0017 |
| Clusters | 1,192 |
| Noise Fraction | 0.0613 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.227 | 0.066 | 0.183 |
| locked_door | 0.227 | 0.000 | 0.121 | 0.501 |
| open_door | 0.066 | 0.121 | 0.000 | 0.334 |
| target_ball | 0.183 | 0.501 | 0.334 | 0.000 |

---

## Checkpoint 7 — 21k (21023 episodes)

**Episodes:** 21,023  
**Success:** 475  
**Failure:** 25  
**Threshold μ:** 5.908

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1032 |
| Coherence (Success) | 0.3818 |
| Coherence (Failure) | 0.0492 |
| Gradient Magnitude (Success) | 0.2589 |
| Gradient Magnitude (Failure) | 0.3448 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0810 |
| Clusters | 1,029 |
| Noise Fraction | 0.0601 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.147 | 0.019 | 0.085 |
| locked_door | 0.147 | 0.000 | 0.118 | 0.351 |
| open_door | 0.019 | 0.118 | 0.000 | 0.152 |
| target_ball | 0.085 | 0.351 | 0.152 | -0.000 |

---

## Checkpoint 8 — 24k (24015 episodes)

**Episodes:** 24,015  
**Success:** 487  
**Failure:** 13  
**Threshold μ:** 5.950

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2208 |
| Coherence (Success) | 0.4201 |
| Coherence (Failure) | 0.1203 |
| Gradient Magnitude (Success) | 0.3086 |
| Gradient Magnitude (Failure) | 0.4248 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.6308 |
| Clusters | 972 |
| Noise Fraction | 0.0712 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.135 | 0.033 | 0.084 |
| locked_door | 0.135 | 0.000 | 0.141 | 0.367 |
| open_door | 0.033 | 0.141 | 0.000 | 0.152 |
| target_ball | 0.084 | 0.367 | 0.152 | 0.000 |

---

## Checkpoint 9 — 27k (27016 episodes)

**Episodes:** 27,016  
**Success:** 488  
**Failure:** 12  
**Threshold μ:** 5.952

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0003 |
| Coherence (Success) | 0.2596 |
| Coherence (Failure) | 0.1114 |
| Gradient Magnitude (Success) | 0.2284 |
| Gradient Magnitude (Failure) | 0.3260 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.9843 |
| Clusters | 865 |
| Noise Fraction | 0.0654 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.124 | 0.022 | 0.056 |
| locked_door | 0.124 | 0.000 | 0.115 | 0.279 |
| open_door | 0.022 | 0.115 | -0.000 | 0.101 |
| target_ball | 0.056 | 0.279 | 0.101 | 0.000 |

---

## Checkpoint 10 — 30k (30036 episodes)

**Episodes:** 30,036  
**Success:** 478  
**Failure:** 22  
**Threshold μ:** 5.918

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2372 |
| Coherence (Success) | 0.4144 |
| Coherence (Failure) | -0.0057 |
| Gradient Magnitude (Success) | 0.2986 |
| Gradient Magnitude (Failure) | 0.2035 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.0030 |
| Clusters | 839 |
| Noise Fraction | 0.0653 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.131 | 0.012 | 0.064 |
| locked_door | 0.131 | 0.000 | 0.135 | 0.253 |
| open_door | 0.012 | 0.135 | 0.000 | 0.097 |
| target_ball | 0.064 | 0.253 | 0.097 | 0.000 |

---

## Final — 30k episodes

**Episodes:** 30,036  
**Success:** 478  
**Failure:** 22  
**Threshold μ:** 5.932

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1688 |
| Coherence (Success) | 0.3321 |
| Coherence (Failure) | 0.0343 |
| Gradient Magnitude (Success) | 0.2692 |
| Gradient Magnitude (Failure) | 0.3338 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.9631 |
| Clusters | 864 |
| Noise Fraction | 0.0692 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.132 | 0.013 | 0.063 |
| locked_door | 0.132 | 0.000 | 0.139 | 0.262 |
| open_door | 0.013 | 0.139 | 0.000 | 0.100 |
| target_ball | 0.063 | 0.262 | 0.100 | 0.000 |
