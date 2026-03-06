# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 5  
**Total episodes:** 100,020  
**Experiment root:** `Proof-of-Concept-Runs\seed_5`  
**Generated:** 2026-02-25 03:08

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1010 episodes) | 1,010 | 0.8700 | 0.4090 | 0.3122 | 0.1248 | 0.0785 | 0.0619 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6 — 6k (6006 episodes) | 6,006 | 0.8048 | 0.3757 | 0.3400 | 0.1622 | 0.1215 | 0.0872 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 11 — 11k (11022 episodes) | 11,022 | 0.2505 | 0.1984 | 0.0584 | 0.3024 | 0.1142 | 1.4923 | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 16 — 16k (16031 episodes) | 16,031 | 0.4995 | 0.2781 | 0.2121 | 0.4089 | 0.2254 | 1.7545 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 21 — 21k (21035 episodes) | 21,035 | 0.2503 | 0.2047 | 0.1990 | 0.3102 | 0.1854 | 2.0037 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 26 — 26k (26009 episodes) | 26,009 | 0.5217 | 0.2896 | 0.2423 | 0.3548 | 0.2618 | 1.6649 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 31 — 31k (31010 episodes) | 31,010 | 0.2362 | 0.3054 | 0.2534 | 0.2662 | 0.2496 | 1.9403 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 36 — 36k (36054 episodes) | 36,054 | 0.2023 | 0.2084 | 0.2201 | 0.2342 | 0.2214 | 2.9610 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 41 — 41k (41035 episodes) | 41,035 | 0.2020 | 0.1502 | 0.1586 | 0.2257 | 0.2259 | 2.2954 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 46 — 46k (46010 episodes) | 46,010 | 0.2670 | 0.1239 | 0.2310 | 0.2013 | 0.2283 | 2.8864 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 51 — 51k (51024 episodes) | 51,024 | 0.1751 | 0.1671 | 0.2160 | 0.1998 | 0.1756 | 3.9970 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 56 — 56k (56042 episodes) | 56,042 | 0.2673 | 0.3521 | 0.1883 | 0.3243 | 0.1845 | 3.8185 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 61 — 61k (61016 episodes) | 61,016 | 0.4914 | 0.2409 | 0.1493 | 0.2606 | 0.1960 | 3.5749 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 66 — 66k (66001 episodes) | 66,001 | 0.1650 | 0.2265 | 0.2703 | 0.2372 | 0.2384 | 4.0759 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 71 — 71k (71009 episodes) | 71,009 | 0.4102 | 0.2028 | 0.1200 | 0.2203 | 0.1682 | 4.5430 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 76 — 76k (76005 episodes) | 76,005 | 0.2843 | 0.1516 | 0.1526 | 0.2436 | 0.1615 | 4.0308 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 81 — 81k (81063 episodes) | 81,063 | 0.4261 | 0.1898 | 0.2428 | 0.2532 | 0.2186 | 4.8218 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 86 — 86k (86000 episodes) | 86,000 | 0.6566 | 0.1663 | 0.1454 | 0.2209 | 0.1613 | 3.7988 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 91 — 91k (91014 episodes) | 91,014 | 0.1915 | 0.1185 | 0.1323 | 0.1839 | 0.1824 | 5.3945 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 96 — 96k (96014 episodes) | 96,014 | 0.6716 | 0.1931 | 0.1473 | 0.2609 | 0.1768 | 4.7289 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,020 | 0.4319 | 0.2012 | 0.1352 | 0.2082 | 0.1776 | 5.3528 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

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

## Checkpoint 11 — 11k (11022 episodes)

**Episodes:** 11,022  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.857) | top 25% (≥ 2.253)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2505 |
| Coherence (Success) | 0.1984 |
| Coherence (Failure) | 0.0584 |
| Gradient Magnitude (Success) | 0.3024 |
| Gradient Magnitude (Failure) | 0.1142 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.4923 |
| Clusters | 1,102 |
| Noise Fraction | 0.0436 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.272 | 0.160 | 0.214 |
| locked_door | 0.272 | -0.000 | 0.124 | 0.741 |
| open_door | 0.160 | 0.124 | 0.000 | 0.402 |
| target_ball | 0.214 | 0.741 | 0.402 | 0.000 |

---

## Checkpoint 16 — 16k (16031 episodes)

**Episodes:** 16,031  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.073) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4995 |
| Coherence (Success) | 0.2781 |
| Coherence (Failure) | 0.2121 |
| Gradient Magnitude (Success) | 0.4089 |
| Gradient Magnitude (Failure) | 0.2254 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7545 |
| Clusters | 888 |
| Noise Fraction | 0.0525 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.342 | 0.068 | 0.142 |
| locked_door | 0.342 | -0.000 | 0.314 | 0.770 |
| open_door | 0.068 | 0.314 | -0.000 | 0.196 |
| target_ball | 0.142 | 0.770 | 0.196 | -0.000 |

---

## Checkpoint 21 — 21k (21035 episodes)

**Episodes:** 21,035  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.100) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2503 |
| Coherence (Success) | 0.2047 |
| Coherence (Failure) | 0.1990 |
| Gradient Magnitude (Success) | 0.3102 |
| Gradient Magnitude (Failure) | 0.1854 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.0037 |
| Clusters | 744 |
| Noise Fraction | 0.0706 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.252 | 0.036 | 0.128 |
| locked_door | 0.252 | 0.000 | 0.337 | 0.636 |
| open_door | 0.036 | 0.337 | -0.000 | 0.111 |
| target_ball | 0.128 | 0.636 | 0.111 | -0.000 |

---

## Checkpoint 26 — 26k (26009 episodes)

**Episodes:** 26,009  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.133) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5217 |
| Coherence (Success) | 0.2896 |
| Coherence (Failure) | 0.2423 |
| Gradient Magnitude (Success) | 0.3548 |
| Gradient Magnitude (Failure) | 0.2618 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.6649 |
| Clusters | 622 |
| Noise Fraction | 0.0814 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.164 | 0.038 | 0.215 |
| locked_door | 0.164 | 0.000 | 0.263 | 0.665 |
| open_door | 0.038 | 0.263 | 0.000 | 0.152 |
| target_ball | 0.215 | 0.665 | 0.152 | 0.000 |

---

## Checkpoint 31 — 31k (31010 episodes)

**Episodes:** 31,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2362 |
| Coherence (Success) | 0.3054 |
| Coherence (Failure) | 0.2534 |
| Gradient Magnitude (Success) | 0.2662 |
| Gradient Magnitude (Failure) | 0.2496 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.9403 |
| Clusters | 563 |
| Noise Fraction | 0.0711 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.157 | 0.028 | 0.211 |
| locked_door | 0.157 | -0.000 | 0.236 | 0.648 |
| open_door | 0.028 | 0.236 | 0.000 | 0.157 |
| target_ball | 0.211 | 0.648 | 0.157 | 0.000 |

---

## Checkpoint 36 — 36k (36054 episodes)

**Episodes:** 36,054  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2023 |
| Coherence (Success) | 0.2084 |
| Coherence (Failure) | 0.2201 |
| Gradient Magnitude (Success) | 0.2342 |
| Gradient Magnitude (Failure) | 0.2214 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9610 |
| Clusters | 546 |
| Noise Fraction | 0.0804 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.201 | 0.020 | 0.125 |
| locked_door | 0.201 | 0.000 | 0.279 | 0.573 |
| open_door | 0.020 | 0.279 | 0.000 | 0.078 |
| target_ball | 0.125 | 0.573 | 0.078 | 0.000 |

---

## Checkpoint 41 — 41k (41035 episodes)

**Episodes:** 41,035  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2020 |
| Coherence (Success) | 0.1502 |
| Coherence (Failure) | 0.1586 |
| Gradient Magnitude (Success) | 0.2257 |
| Gradient Magnitude (Failure) | 0.2259 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.2954 |
| Clusters | 485 |
| Noise Fraction | 0.0882 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.187 | 0.033 | 0.172 |
| locked_door | 0.187 | 0.000 | 0.306 | 0.637 |
| open_door | 0.033 | 0.306 | 0.000 | 0.106 |
| target_ball | 0.172 | 0.637 | 0.106 | -0.000 |

---

## Checkpoint 46 — 46k (46010 episodes)

**Episodes:** 46,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2670 |
| Coherence (Success) | 0.1239 |
| Coherence (Failure) | 0.2310 |
| Gradient Magnitude (Success) | 0.2013 |
| Gradient Magnitude (Failure) | 0.2283 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.8864 |
| Clusters | 517 |
| Noise Fraction | 0.0804 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.172 | 0.027 | 0.164 |
| locked_door | 0.172 | -0.000 | 0.262 | 0.598 |
| open_door | 0.027 | 0.262 | 0.000 | 0.109 |
| target_ball | 0.164 | 0.598 | 0.109 | 0.000 |

---

## Checkpoint 51 — 51k (51024 episodes)

**Episodes:** 51,024  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1751 |
| Coherence (Success) | 0.1671 |
| Coherence (Failure) | 0.2160 |
| Gradient Magnitude (Success) | 0.1998 |
| Gradient Magnitude (Failure) | 0.1756 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.9970 |
| Clusters | 605 |
| Noise Fraction | 0.0778 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.206 | 0.023 | 0.127 |
| locked_door | 0.206 | -0.000 | 0.290 | 0.588 |
| open_door | 0.023 | 0.290 | -0.000 | 0.090 |
| target_ball | 0.127 | 0.588 | 0.090 | -0.000 |

---

## Checkpoint 56 — 56k (56042 episodes)

**Episodes:** 56,042  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2673 |
| Coherence (Success) | 0.3521 |
| Coherence (Failure) | 0.1883 |
| Gradient Magnitude (Success) | 0.3243 |
| Gradient Magnitude (Failure) | 0.1845 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8185 |
| Clusters | 548 |
| Noise Fraction | 0.0796 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.164 | 0.028 | 0.154 |
| locked_door | 0.164 | 0.000 | 0.247 | 0.565 |
| open_door | 0.028 | 0.247 | 0.000 | 0.106 |
| target_ball | 0.154 | 0.565 | 0.106 | 0.000 |

---

## Checkpoint 61 — 61k (61016 episodes)

**Episodes:** 61,016  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4914 |
| Coherence (Success) | 0.2409 |
| Coherence (Failure) | 0.1493 |
| Gradient Magnitude (Success) | 0.2606 |
| Gradient Magnitude (Failure) | 0.1960 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.5749 |
| Clusters | 528 |
| Noise Fraction | 0.0936 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.143 | 0.024 | 0.183 |
| locked_door | 0.143 | 0.000 | 0.243 | 0.582 |
| open_door | 0.024 | 0.243 | 0.000 | 0.116 |
| target_ball | 0.183 | 0.582 | 0.116 | -0.000 |

---

## Checkpoint 66 — 66k (66001 episodes)

**Episodes:** 66,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1650 |
| Coherence (Success) | 0.2265 |
| Coherence (Failure) | 0.2703 |
| Gradient Magnitude (Success) | 0.2372 |
| Gradient Magnitude (Failure) | 0.2384 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.0759 |
| Clusters | 557 |
| Noise Fraction | 0.0778 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.151 | 0.038 | 0.205 |
| locked_door | 0.151 | 0.000 | 0.279 | 0.625 |
| open_door | 0.038 | 0.279 | -0.000 | 0.123 |
| target_ball | 0.205 | 0.625 | 0.123 | 0.000 |

---

## Checkpoint 71 — 71k (71009 episodes)

**Episodes:** 71,009  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4102 |
| Coherence (Success) | 0.2028 |
| Coherence (Failure) | 0.1200 |
| Gradient Magnitude (Success) | 0.2203 |
| Gradient Magnitude (Failure) | 0.1682 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.5430 |
| Clusters | 512 |
| Noise Fraction | 0.1011 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.123 | 0.035 | 0.215 |
| locked_door | 0.123 | 0.000 | 0.245 | 0.594 |
| open_door | 0.035 | 0.245 | 0.000 | 0.125 |
| target_ball | 0.215 | 0.594 | 0.125 | 0.000 |

---

## Checkpoint 76 — 76k (76005 episodes)

**Episodes:** 76,005  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2843 |
| Coherence (Success) | 0.1516 |
| Coherence (Failure) | 0.1526 |
| Gradient Magnitude (Success) | 0.2436 |
| Gradient Magnitude (Failure) | 0.1615 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.0308 |
| Clusters | 512 |
| Noise Fraction | 0.0861 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.131 | 0.027 | 0.186 |
| locked_door | 0.131 | -0.000 | 0.213 | 0.559 |
| open_door | 0.027 | 0.213 | 0.000 | 0.122 |
| target_ball | 0.186 | 0.559 | 0.122 | 0.000 |

---

## Checkpoint 81 — 81k (81063 episodes)

**Episodes:** 81,063  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4261 |
| Coherence (Success) | 0.1898 |
| Coherence (Failure) | 0.2428 |
| Gradient Magnitude (Success) | 0.2532 |
| Gradient Magnitude (Failure) | 0.2186 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.8218 |
| Clusters | 517 |
| Noise Fraction | 0.0869 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.134 | 0.040 | 0.191 |
| locked_door | 0.134 | 0.000 | 0.257 | 0.577 |
| open_door | 0.040 | 0.257 | 0.000 | 0.107 |
| target_ball | 0.191 | 0.577 | 0.107 | 0.000 |

---

## Checkpoint 86 — 86k (86000 episodes)

**Episodes:** 86,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6566 |
| Coherence (Success) | 0.1663 |
| Coherence (Failure) | 0.1454 |
| Gradient Magnitude (Success) | 0.2209 |
| Gradient Magnitude (Failure) | 0.1613 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.7988 |
| Clusters | 550 |
| Noise Fraction | 0.1159 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.112 | 0.040 | 0.256 |
| locked_door | 0.112 | -0.000 | 0.233 | 0.632 |
| open_door | 0.040 | 0.233 | 0.000 | 0.152 |
| target_ball | 0.256 | 0.632 | 0.152 | 0.000 |

---

## Checkpoint 91 — 91k (91014 episodes)

**Episodes:** 91,014  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1915 |
| Coherence (Success) | 0.1185 |
| Coherence (Failure) | 0.1323 |
| Gradient Magnitude (Success) | 0.1839 |
| Gradient Magnitude (Failure) | 0.1824 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.3945 |
| Clusters | 508 |
| Noise Fraction | 0.0842 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.142 | 0.033 | 0.175 |
| locked_door | 0.142 | 0.000 | 0.241 | 0.565 |
| open_door | 0.033 | 0.241 | 0.000 | 0.116 |
| target_ball | 0.175 | 0.565 | 0.116 | 0.000 |

---

## Checkpoint 96 — 96k (96014 episodes)

**Episodes:** 96,014  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6716 |
| Coherence (Success) | 0.1931 |
| Coherence (Failure) | 0.1473 |
| Gradient Magnitude (Success) | 0.2609 |
| Gradient Magnitude (Failure) | 0.1768 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.7289 |
| Clusters | 477 |
| Noise Fraction | 0.0837 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.140 | 0.028 | 0.188 |
| locked_door | 0.140 | 0.000 | 0.239 | 0.575 |
| open_door | 0.028 | 0.239 | 0.000 | 0.127 |
| target_ball | 0.188 | 0.575 | 0.127 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.187) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4319 |
| Coherence (Success) | 0.2012 |
| Coherence (Failure) | 0.1352 |
| Gradient Magnitude (Success) | 0.2082 |
| Gradient Magnitude (Failure) | 0.1776 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.3528 |
| Clusters | 472 |
| Noise Fraction | 0.0857 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.134 | 0.032 | 0.243 |
| locked_door | 0.134 | 0.000 | 0.253 | 0.660 |
| open_door | 0.032 | 0.253 | 0.000 | 0.146 |
| target_ball | 0.243 | 0.660 | 0.146 | -0.000 |
