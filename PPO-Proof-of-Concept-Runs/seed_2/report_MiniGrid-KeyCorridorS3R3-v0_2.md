# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 2  
**Total episodes:** 100,000  
**Experiment root:** `Proof-of-Concept-Runs\seed_2`  
**Generated:** 2026-02-24 21:11

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1009 episodes) | 1,009 | 0.8685 | 0.3731 | 0.3059 | 0.0796 | 0.0677 | 0.0509 | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 6 — 6k (6002 episodes) | 6,002 | 0.5364 | 0.2963 | 0.0259 | 0.2063 | 0.0590 | 0.4978 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 11 — 11k (11014 episodes) | 11,014 | 0.3417 | 0.2137 | 0.0311 | 0.2668 | 0.1075 | 1.2815 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 16 — 16k (16026 episodes) | 16,026 | 0.3193 | 0.1165 | 0.1261 | 0.2673 | 0.1838 | 1.5343 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 21 — 21k (21023 episodes) | 21,023 | 0.5375 | 0.2325 | 0.1634 | 0.3100 | 0.2248 | 2.1837 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 26 — 26k (26001 episodes) | 26,001 | 0.0802 | 0.1494 | 0.1546 | 0.2888 | 0.1808 | 2.9057 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 31 — 31k (31028 episodes) | 31,028 | 0.1918 | 0.2715 | 0.0865 | 0.2851 | 0.1682 | 3.4172 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 36 — 36k (36035 episodes) | 36,035 | 0.3277 | 0.1732 | 0.2218 | 0.1923 | 0.2104 | 3.9372 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 41 — 41k (41012 episodes) | 41,012 | 0.4925 | 0.2356 | 0.1676 | 0.2652 | 0.2057 | 3.8436 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 46 — 46k (46064 episodes) | 46,064 | 0.3665 | 0.2053 | 0.1757 | 0.2222 | 0.1582 | 4.3255 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 51 — 51k (51067 episodes) | 51,067 | 0.0485 | 0.2042 | 0.1900 | 0.2515 | 0.2061 | 4.5172 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 56 — 56k (56000 episodes) | 56,000 | 0.4670 | 0.2940 | 0.1664 | 0.2597 | 0.2391 | 4.8600 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 61 — 61k (61024 episodes) | 61,024 | 0.4833 | 0.1652 | 0.2252 | 0.2198 | 0.2609 | 5.1085 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 66 — 66k (66058 episodes) | 66,058 | 0.0344 | 0.1116 | 0.2953 | 0.1899 | 0.2238 | 5.7079 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 71 — 71k (71066 episodes) | 71,066 | 0.3979 | 0.1578 | 0.1401 | 0.2297 | 0.1563 | 6.5183 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 76 — 76k (76007 episodes) | 76,007 | 0.4967 | 0.2545 | 0.1517 | 0.2876 | 0.2032 | 7.5634 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 81 — 81k (81029 episodes) | 81,029 | 0.5701 | 0.2389 | 0.2292 | 0.2338 | 0.2142 | 6.4502 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 86 — 86k (86037 episodes) | 86,037 | 0.3899 | 0.2030 | 0.1422 | 0.2660 | 0.1692 | 6.7664 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 91 — 91k (91019 episodes) | 91,019 | 0.3904 | 0.2746 | 0.2011 | 0.3140 | 0.2059 | 6.7133 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 96 — 96k (96054 episodes) | 96,054 | 0.3987 | 0.1194 | 0.0937 | 0.2210 | 0.1943 | 8.4199 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,000 | 0.3759 | 0.2586 | 0.1069 | 0.3560 | 0.1871 | 8.8278 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

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

## Checkpoint 11 — 11k (11014 episodes)

**Episodes:** 11,014  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.920) | top 25% (≥ 2.243)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3417 |
| Coherence (Success) | 0.2137 |
| Coherence (Failure) | 0.0311 |
| Gradient Magnitude (Success) | 0.2668 |
| Gradient Magnitude (Failure) | 0.1075 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.2815 |
| Clusters | 1,095 |
| Noise Fraction | 0.0433 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.233 | 0.066 | 0.188 |
| locked_door | 0.233 | 0.000 | 0.336 | 0.592 |
| open_door | 0.066 | 0.336 | 0.000 | 0.183 |
| target_ball | 0.188 | 0.592 | 0.183 | 0.000 |

---

## Checkpoint 16 — 16k (16026 episodes)

**Episodes:** 16,026  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.040) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3193 |
| Coherence (Success) | 0.1165 |
| Coherence (Failure) | 0.1261 |
| Gradient Magnitude (Success) | 0.2673 |
| Gradient Magnitude (Failure) | 0.1838 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.5343 |
| Clusters | 965 |
| Noise Fraction | 0.0472 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.154 | 0.085 | 0.107 |
| locked_door | 0.154 | 0.000 | 0.311 | 0.366 |
| open_door | 0.085 | 0.311 | 0.000 | 0.151 |
| target_ball | 0.107 | 0.366 | 0.151 | 0.000 |

---

## Checkpoint 21 — 21k (21023 episodes)

**Episodes:** 21,023  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5375 |
| Coherence (Success) | 0.2325 |
| Coherence (Failure) | 0.1634 |
| Gradient Magnitude (Success) | 0.3100 |
| Gradient Magnitude (Failure) | 0.2248 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.1837 |
| Clusters | 659 |
| Noise Fraction | 0.0741 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.127 | 0.034 | 0.189 |
| locked_door | 0.127 | 0.000 | 0.223 | 0.547 |
| open_door | 0.034 | 0.223 | 0.000 | 0.133 |
| target_ball | 0.189 | 0.547 | 0.133 | -0.000 |

---

## Checkpoint 26 — 26k (26001 episodes)

**Episodes:** 26,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.103) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0802 |
| Coherence (Success) | 0.1494 |
| Coherence (Failure) | 0.1546 |
| Gradient Magnitude (Success) | 0.2888 |
| Gradient Magnitude (Failure) | 0.1808 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9057 |
| Clusters | 779 |
| Noise Fraction | 0.0679 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.089 | 0.035 | 0.168 |
| locked_door | 0.089 | 0.000 | 0.169 | 0.459 |
| open_door | 0.035 | 0.169 | 0.000 | 0.118 |
| target_ball | 0.168 | 0.459 | 0.118 | 0.000 |

---

## Checkpoint 31 — 31k (31028 episodes)

**Episodes:** 31,028  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1918 |
| Coherence (Success) | 0.2715 |
| Coherence (Failure) | 0.0865 |
| Gradient Magnitude (Success) | 0.2851 |
| Gradient Magnitude (Failure) | 0.1682 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.4172 |
| Clusters | 560 |
| Noise Fraction | 0.0775 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.074 | 0.037 | 0.154 |
| locked_door | 0.074 | -0.000 | 0.153 | 0.401 |
| open_door | 0.037 | 0.153 | 0.000 | 0.105 |
| target_ball | 0.154 | 0.401 | 0.105 | 0.000 |

---

## Checkpoint 36 — 36k (36035 episodes)

**Episodes:** 36,035  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.133) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3277 |
| Coherence (Success) | 0.1732 |
| Coherence (Failure) | 0.2218 |
| Gradient Magnitude (Success) | 0.1923 |
| Gradient Magnitude (Failure) | 0.2104 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.9372 |
| Clusters | 656 |
| Noise Fraction | 0.0720 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.081 | 0.036 | 0.138 |
| locked_door | 0.081 | -0.000 | 0.190 | 0.399 |
| open_door | 0.036 | 0.190 | 0.000 | 0.065 |
| target_ball | 0.138 | 0.399 | 0.065 | -0.000 |

---

## Checkpoint 41 — 41k (41012 episodes)

**Episodes:** 41,012  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4925 |
| Coherence (Success) | 0.2356 |
| Coherence (Failure) | 0.1676 |
| Gradient Magnitude (Success) | 0.2652 |
| Gradient Magnitude (Failure) | 0.2057 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8436 |
| Clusters | 574 |
| Noise Fraction | 0.0837 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.068 | 0.032 | 0.140 |
| locked_door | 0.068 | 0.000 | 0.166 | 0.377 |
| open_door | 0.032 | 0.166 | -0.000 | 0.069 |
| target_ball | 0.140 | 0.377 | 0.069 | 0.000 |

---

## Checkpoint 46 — 46k (46064 episodes)

**Episodes:** 46,064  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3665 |
| Coherence (Success) | 0.2053 |
| Coherence (Failure) | 0.1757 |
| Gradient Magnitude (Success) | 0.2222 |
| Gradient Magnitude (Failure) | 0.1582 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.3255 |
| Clusters | 572 |
| Noise Fraction | 0.0726 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.067 | 0.034 | 0.149 |
| locked_door | 0.067 | 0.000 | 0.163 | 0.383 |
| open_door | 0.034 | 0.163 | 0.000 | 0.070 |
| target_ball | 0.149 | 0.383 | 0.070 | 0.000 |

---

## Checkpoint 51 — 51k (51067 episodes)

**Episodes:** 51,067  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0485 |
| Coherence (Success) | 0.2042 |
| Coherence (Failure) | 0.1900 |
| Gradient Magnitude (Success) | 0.2515 |
| Gradient Magnitude (Failure) | 0.2061 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.5172 |
| Clusters | 508 |
| Noise Fraction | 0.0901 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.060 | 0.027 | 0.126 |
| locked_door | 0.060 | 0.000 | 0.137 | 0.333 |
| open_door | 0.027 | 0.137 | -0.000 | 0.080 |
| target_ball | 0.126 | 0.333 | 0.080 | 0.000 |

---

## Checkpoint 56 — 56k (56000 episodes)

**Episodes:** 56,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4670 |
| Coherence (Success) | 0.2940 |
| Coherence (Failure) | 0.1664 |
| Gradient Magnitude (Success) | 0.2597 |
| Gradient Magnitude (Failure) | 0.2391 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.8600 |
| Clusters | 504 |
| Noise Fraction | 0.0858 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.081 | 0.037 | 0.149 |
| locked_door | 0.081 | 0.000 | 0.194 | 0.413 |
| open_door | 0.037 | 0.194 | -0.000 | 0.072 |
| target_ball | 0.149 | 0.413 | 0.072 | 0.000 |

---

## Checkpoint 61 — 61k (61024 episodes)

**Episodes:** 61,024  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4833 |
| Coherence (Success) | 0.1652 |
| Coherence (Failure) | 0.2252 |
| Gradient Magnitude (Success) | 0.2198 |
| Gradient Magnitude (Failure) | 0.2609 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.1085 |
| Clusters | 492 |
| Noise Fraction | 0.0809 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.066 | 0.028 | 0.174 |
| locked_door | 0.066 | 0.000 | 0.148 | 0.417 |
| open_door | 0.028 | 0.148 | 0.000 | 0.100 |
| target_ball | 0.174 | 0.417 | 0.100 | 0.000 |

---

## Checkpoint 66 — 66k (66058 episodes)

**Episodes:** 66,058  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0344 |
| Coherence (Success) | 0.1116 |
| Coherence (Failure) | 0.2953 |
| Gradient Magnitude (Success) | 0.1899 |
| Gradient Magnitude (Failure) | 0.2238 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.7079 |
| Clusters | 533 |
| Noise Fraction | 0.0930 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.064 | 0.020 | 0.107 |
| locked_door | 0.064 | -0.000 | 0.121 | 0.311 |
| open_door | 0.020 | 0.121 | 0.000 | 0.067 |
| target_ball | 0.107 | 0.311 | 0.067 | 0.000 |

---

## Checkpoint 71 — 71k (71066 episodes)

**Episodes:** 71,066  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3979 |
| Coherence (Success) | 0.1578 |
| Coherence (Failure) | 0.1401 |
| Gradient Magnitude (Success) | 0.2297 |
| Gradient Magnitude (Failure) | 0.1563 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.5183 |
| Clusters | 517 |
| Noise Fraction | 0.0853 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.070 | 0.024 | 0.157 |
| locked_door | 0.070 | 0.000 | 0.147 | 0.399 |
| open_door | 0.024 | 0.147 | 0.000 | 0.084 |
| target_ball | 0.157 | 0.399 | 0.084 | 0.000 |

---

## Checkpoint 76 — 76k (76007 episodes)

**Episodes:** 76,007  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4967 |
| Coherence (Success) | 0.2545 |
| Coherence (Failure) | 0.1517 |
| Gradient Magnitude (Success) | 0.2876 |
| Gradient Magnitude (Failure) | 0.2032 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.5634 |
| Clusters | 550 |
| Noise Fraction | 0.0895 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.064 | 0.023 | 0.156 |
| locked_door | 0.064 | 0.000 | 0.135 | 0.386 |
| open_door | 0.023 | 0.135 | 0.000 | 0.086 |
| target_ball | 0.156 | 0.386 | 0.086 | -0.000 |

---

## Checkpoint 81 — 81k (81029 episodes)

**Episodes:** 81,029  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5701 |
| Coherence (Success) | 0.2389 |
| Coherence (Failure) | 0.2292 |
| Gradient Magnitude (Success) | 0.2338 |
| Gradient Magnitude (Failure) | 0.2142 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.4502 |
| Clusters | 465 |
| Noise Fraction | 0.1047 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.061 | 0.024 | 0.141 |
| locked_door | 0.061 | 0.000 | 0.129 | 0.353 |
| open_door | 0.024 | 0.129 | 0.000 | 0.080 |
| target_ball | 0.141 | 0.353 | 0.080 | 0.000 |

---

## Checkpoint 86 — 86k (86037 episodes)

**Episodes:** 86,037  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3899 |
| Coherence (Success) | 0.2030 |
| Coherence (Failure) | 0.1422 |
| Gradient Magnitude (Success) | 0.2660 |
| Gradient Magnitude (Failure) | 0.1692 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.7664 |
| Clusters | 470 |
| Noise Fraction | 0.0746 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.046 | 0.023 | 0.127 |
| locked_door | 0.046 | -0.000 | 0.099 | 0.296 |
| open_door | 0.023 | 0.099 | 0.000 | 0.071 |
| target_ball | 0.127 | 0.296 | 0.071 | 0.000 |

---

## Checkpoint 91 — 91k (91019 episodes)

**Episodes:** 91,019  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3904 |
| Coherence (Success) | 0.2746 |
| Coherence (Failure) | 0.2011 |
| Gradient Magnitude (Success) | 0.3140 |
| Gradient Magnitude (Failure) | 0.2059 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.7133 |
| Clusters | 498 |
| Noise Fraction | 0.1101 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.060 | 0.028 | 0.146 |
| locked_door | 0.060 | -0.000 | 0.132 | 0.362 |
| open_door | 0.028 | 0.132 | -0.000 | 0.080 |
| target_ball | 0.146 | 0.362 | 0.080 | 0.000 |

---

## Checkpoint 96 — 96k (96054 episodes)

**Episodes:** 96,054  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.187) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3987 |
| Coherence (Success) | 0.1194 |
| Coherence (Failure) | 0.0937 |
| Gradient Magnitude (Success) | 0.2210 |
| Gradient Magnitude (Failure) | 0.1943 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.4199 |
| Clusters | 451 |
| Noise Fraction | 0.0940 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.056 | 0.020 | 0.112 |
| locked_door | 0.056 | -0.000 | 0.110 | 0.298 |
| open_door | 0.020 | 0.110 | 0.000 | 0.064 |
| target_ball | 0.112 | 0.298 | 0.064 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3759 |
| Coherence (Success) | 0.2586 |
| Coherence (Failure) | 0.1069 |
| Gradient Magnitude (Success) | 0.3560 |
| Gradient Magnitude (Failure) | 0.1871 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.8278 |
| Clusters | 503 |
| Noise Fraction | 0.0794 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.053 | 0.021 | 0.120 |
| locked_door | 0.053 | 0.000 | 0.101 | 0.303 |
| open_door | 0.021 | 0.101 | 0.000 | 0.081 |
| target_ball | 0.120 | 0.303 | 0.081 | 0.000 |
