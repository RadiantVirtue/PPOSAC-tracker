# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 3  
**Total episodes:** 100,040  
**Experiment root:** `Proof-of-Concept-Runs\seed_3`  
**Generated:** 2026-02-24 23:37

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1010 episodes) | 1,010 | 0.8407 | 0.2874 | 0.3233 | 0.0697 | 0.0800 | 0.0637 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 6 — 6k (6019 episodes) | 6,019 | 0.3814 | 0.2535 | 0.1191 | 0.2837 | 0.0945 | 0.7106 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 11 — 11k (11022 episodes) | 11,022 | -0.1190 | 0.2246 | 0.0298 | 0.3527 | 0.1172 | 1.5642 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 16 — 16k (16031 episodes) | 16,031 | -0.0473 | 0.2120 | 0.0558 | 0.3526 | 0.1241 | 2.2148 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 21 — 21k (21053 episodes) | 21,053 | 0.3563 | 0.1390 | 0.0739 | 0.2830 | 0.1614 | 2.9474 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 26 — 26k (26041 episodes) | 26,041 | 0.0191 | 0.0733 | 0.1347 | 0.2121 | 0.1559 | 3.0339 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 31 — 31k (31050 episodes) | 31,050 | 0.2098 | 0.1864 | 0.2108 | 0.3193 | 0.2312 | 3.4271 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 36 — 36k (36063 episodes) | 36,063 | 0.2396 | 0.2968 | 0.1692 | 0.3197 | 0.2129 | 3.8787 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 41 — 41k (41020 episodes) | 41,020 | 0.3938 | 0.1675 | 0.1220 | 0.2336 | 0.1649 | 4.2534 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 46 — 46k (46010 episodes) | 46,010 | 0.4089 | 0.3323 | 0.1419 | 0.3561 | 0.1657 | 4.3580 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 51 — 51k (51022 episodes) | 51,022 | -0.0498 | 0.2161 | 0.1625 | 0.2531 | 0.2244 | 4.9828 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 56 — 56k (56050 episodes) | 56,050 | 0.2059 | 0.2215 | 0.1520 | 0.2369 | 0.1932 | 5.6427 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 61 — 61k (61061 episodes) | 61,061 | 0.3774 | 0.1370 | 0.2141 | 0.2128 | 0.2157 | 7.2465 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 66 — 66k (66042 episodes) | 66,042 | 0.4688 | 0.2141 | 0.1568 | 0.2578 | 0.1853 | 7.2248 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 71 — 71k (71038 episodes) | 71,038 | 0.2562 | 0.2011 | 0.1132 | 0.2349 | 0.1731 | 7.0322 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 76 — 76k (76060 episodes) | 76,060 | 0.4041 | 0.1364 | 0.1837 | 0.1988 | 0.1880 | 7.5579 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 81 — 81k (81040 episodes) | 81,040 | -0.0332 | 0.1901 | 0.2420 | 0.2651 | 0.2760 | 9.1459 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 86 — 86k (86064 episodes) | 86,064 | 0.3605 | 0.2590 | 0.1844 | 0.2744 | 0.1823 | 8.2175 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 91 — 91k (91043 episodes) | 91,043 | 0.2477 | 0.1860 | 0.2133 | 0.2365 | 0.2003 | 9.4291 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 96 — 96k (96003 episodes) | 96,003 | 0.6665 | 0.3213 | 0.1308 | 0.2854 | 0.1867 | 11.2331 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,040 | 0.1597 | 0.0797 | 0.2134 | 0.1795 | 0.1851 | 10.7051 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

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

## Checkpoint 11 — 11k (11022 episodes)

**Episodes:** 11,022  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.237)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1190 |
| Coherence (Success) | 0.2246 |
| Coherence (Failure) | 0.0298 |
| Gradient Magnitude (Success) | 0.3527 |
| Gradient Magnitude (Failure) | 0.1172 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.5642 |
| Clusters | 1,139 |
| Noise Fraction | 0.0296 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.314 | 0.174 | 0.341 |
| locked_door | 0.314 | -0.000 | 0.340 | 0.933 |
| open_door | 0.174 | 0.340 | 0.000 | 0.533 |
| target_ball | 0.341 | 0.933 | 0.533 | 0.000 |

---

## Checkpoint 16 — 16k (16031 episodes)

**Episodes:** 16,031  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.073) | top 25% (≥ 2.253)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0473 |
| Coherence (Success) | 0.2120 |
| Coherence (Failure) | 0.0558 |
| Gradient Magnitude (Success) | 0.3526 |
| Gradient Magnitude (Failure) | 0.1241 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.2148 |
| Clusters | 896 |
| Noise Fraction | 0.0633 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.208 | 0.056 | 0.150 |
| locked_door | 0.208 | -0.000 | 0.237 | 0.620 |
| open_door | 0.056 | 0.237 | -0.000 | 0.210 |
| target_ball | 0.150 | 0.620 | 0.210 | -0.000 |

---

## Checkpoint 21 — 21k (21053 episodes)

**Episodes:** 21,053  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.110) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3563 |
| Coherence (Success) | 0.1390 |
| Coherence (Failure) | 0.0739 |
| Gradient Magnitude (Success) | 0.2830 |
| Gradient Magnitude (Failure) | 0.1614 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9474 |
| Clusters | 704 |
| Noise Fraction | 0.0827 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.103 | 0.015 | 0.151 |
| locked_door | 0.103 | -0.000 | 0.138 | 0.439 |
| open_door | 0.015 | 0.138 | 0.000 | 0.145 |
| target_ball | 0.151 | 0.439 | 0.145 | 0.000 |

---

## Checkpoint 26 — 26k (26041 episodes)

**Episodes:** 26,041  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.123) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0191 |
| Coherence (Success) | 0.0733 |
| Coherence (Failure) | 0.1347 |
| Gradient Magnitude (Success) | 0.2121 |
| Gradient Magnitude (Failure) | 0.1559 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.0339 |
| Clusters | 643 |
| Noise Fraction | 0.0835 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.081 | 0.014 | 0.108 |
| locked_door | 0.081 | 0.000 | 0.129 | 0.322 |
| open_door | 0.014 | 0.129 | -0.000 | 0.095 |
| target_ball | 0.108 | 0.322 | 0.095 | 0.000 |

---

## Checkpoint 31 — 31k (31050 episodes)

**Episodes:** 31,050  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2098 |
| Coherence (Success) | 0.1864 |
| Coherence (Failure) | 0.2108 |
| Gradient Magnitude (Success) | 0.3193 |
| Gradient Magnitude (Failure) | 0.2312 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.4271 |
| Clusters | 628 |
| Noise Fraction | 0.0834 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.063 | 0.019 | 0.115 |
| locked_door | 0.063 | -0.000 | 0.126 | 0.312 |
| open_door | 0.019 | 0.126 | -0.000 | 0.069 |
| target_ball | 0.115 | 0.312 | 0.069 | -0.000 |

---

## Checkpoint 36 — 36k (36063 episodes)

**Episodes:** 36,063  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2396 |
| Coherence (Success) | 0.2968 |
| Coherence (Failure) | 0.1692 |
| Gradient Magnitude (Success) | 0.3197 |
| Gradient Magnitude (Failure) | 0.2129 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8787 |
| Clusters | 583 |
| Noise Fraction | 0.0749 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.068 | 0.016 | 0.122 |
| locked_door | 0.068 | -0.000 | 0.120 | 0.314 |
| open_door | 0.016 | 0.120 | 0.000 | 0.115 |
| target_ball | 0.122 | 0.314 | 0.115 | 0.000 |

---

## Checkpoint 41 — 41k (41020 episodes)

**Episodes:** 41,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3938 |
| Coherence (Success) | 0.1675 |
| Coherence (Failure) | 0.1220 |
| Gradient Magnitude (Success) | 0.2336 |
| Gradient Magnitude (Failure) | 0.1649 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.2534 |
| Clusters | 517 |
| Noise Fraction | 0.0814 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.068 | 0.018 | 0.107 |
| locked_door | 0.068 | -0.000 | 0.138 | 0.309 |
| open_door | 0.018 | 0.138 | 0.000 | 0.075 |
| target_ball | 0.107 | 0.309 | 0.075 | 0.000 |

---

## Checkpoint 46 — 46k (46010 episodes)

**Episodes:** 46,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.130) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4089 |
| Coherence (Success) | 0.3323 |
| Coherence (Failure) | 0.1419 |
| Gradient Magnitude (Success) | 0.3561 |
| Gradient Magnitude (Failure) | 0.1657 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.3580 |
| Clusters | 591 |
| Noise Fraction | 0.0784 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.061 | 0.019 | 0.115 |
| locked_door | 0.061 | 0.000 | 0.117 | 0.311 |
| open_door | 0.019 | 0.117 | 0.000 | 0.087 |
| target_ball | 0.115 | 0.311 | 0.087 | -0.000 |

---

## Checkpoint 51 — 51k (51022 episodes)

**Episodes:** 51,022  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.187) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0498 |
| Coherence (Success) | 0.2161 |
| Coherence (Failure) | 0.1625 |
| Gradient Magnitude (Success) | 0.2531 |
| Gradient Magnitude (Failure) | 0.2244 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.9828 |
| Clusters | 463 |
| Noise Fraction | 0.0952 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.068 | 0.030 | 0.118 |
| locked_door | 0.068 | 0.000 | 0.160 | 0.324 |
| open_door | 0.030 | 0.160 | -0.000 | 0.093 |
| target_ball | 0.118 | 0.324 | 0.093 | 0.000 |

---

## Checkpoint 56 — 56k (56050 episodes)

**Episodes:** 56,050  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2059 |
| Coherence (Success) | 0.2215 |
| Coherence (Failure) | 0.1520 |
| Gradient Magnitude (Success) | 0.2369 |
| Gradient Magnitude (Failure) | 0.1932 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.6427 |
| Clusters | 592 |
| Noise Fraction | 0.0826 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.062 | 0.019 | 0.122 |
| locked_door | 0.062 | 0.000 | 0.126 | 0.320 |
| open_door | 0.019 | 0.126 | -0.000 | 0.091 |
| target_ball | 0.122 | 0.320 | 0.091 | 0.000 |

---

## Checkpoint 61 — 61k (61061 episodes)

**Episodes:** 61,061  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3774 |
| Coherence (Success) | 0.1370 |
| Coherence (Failure) | 0.2141 |
| Gradient Magnitude (Success) | 0.2128 |
| Gradient Magnitude (Failure) | 0.2157 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.2465 |
| Clusters | 510 |
| Noise Fraction | 0.0841 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.060 | 0.021 | 0.115 |
| locked_door | 0.060 | -0.000 | 0.127 | 0.311 |
| open_door | 0.021 | 0.127 | -0.000 | 0.084 |
| target_ball | 0.115 | 0.311 | 0.084 | 0.000 |

---

## Checkpoint 66 — 66k (66042 episodes)

**Episodes:** 66,042  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4688 |
| Coherence (Success) | 0.2141 |
| Coherence (Failure) | 0.1568 |
| Gradient Magnitude (Success) | 0.2578 |
| Gradient Magnitude (Failure) | 0.1853 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.2248 |
| Clusters | 558 |
| Noise Fraction | 0.0826 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.047 | 0.015 | 0.097 |
| locked_door | 0.047 | 0.000 | 0.092 | 0.252 |
| open_door | 0.015 | 0.092 | 0.000 | 0.083 |
| target_ball | 0.097 | 0.252 | 0.083 | 0.000 |

---

## Checkpoint 71 — 71k (71038 episodes)

**Episodes:** 71,038  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2562 |
| Coherence (Success) | 0.2011 |
| Coherence (Failure) | 0.1132 |
| Gradient Magnitude (Success) | 0.2349 |
| Gradient Magnitude (Failure) | 0.1731 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.0322 |
| Clusters | 610 |
| Noise Fraction | 0.0872 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.040 | 0.015 | 0.089 |
| locked_door | 0.040 | 0.000 | 0.076 | 0.226 |
| open_door | 0.015 | 0.076 | 0.000 | 0.090 |
| target_ball | 0.089 | 0.226 | 0.090 | 0.000 |

---

## Checkpoint 76 — 76k (76060 episodes)

**Episodes:** 76,060  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4041 |
| Coherence (Success) | 0.1364 |
| Coherence (Failure) | 0.1837 |
| Gradient Magnitude (Success) | 0.1988 |
| Gradient Magnitude (Failure) | 0.1880 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.5579 |
| Clusters | 511 |
| Noise Fraction | 0.1028 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.048 | 0.014 | 0.096 |
| locked_door | 0.048 | 0.000 | 0.095 | 0.251 |
| open_door | 0.014 | 0.095 | 0.000 | 0.070 |
| target_ball | 0.096 | 0.251 | 0.070 | 0.000 |

---

## Checkpoint 81 — 81k (81040 episodes)

**Episodes:** 81,040  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.183) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0332 |
| Coherence (Success) | 0.1901 |
| Coherence (Failure) | 0.2420 |
| Gradient Magnitude (Success) | 0.2651 |
| Gradient Magnitude (Failure) | 0.2760 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.1459 |
| Clusters | 511 |
| Noise Fraction | 0.0843 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.055 | 0.016 | 0.106 |
| locked_door | 0.055 | 0.000 | 0.112 | 0.291 |
| open_door | 0.016 | 0.112 | -0.000 | 0.068 |
| target_ball | 0.106 | 0.291 | 0.068 | 0.000 |

---

## Checkpoint 86 — 86k (86064 episodes)

**Episodes:** 86,064  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3605 |
| Coherence (Success) | 0.2590 |
| Coherence (Failure) | 0.1844 |
| Gradient Magnitude (Success) | 0.2744 |
| Gradient Magnitude (Failure) | 0.1823 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.2175 |
| Clusters | 511 |
| Noise Fraction | 0.1061 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.052 | 0.017 | 0.094 |
| locked_door | 0.052 | 0.000 | 0.108 | 0.264 |
| open_door | 0.017 | 0.108 | 0.000 | 0.056 |
| target_ball | 0.094 | 0.264 | 0.056 | 0.000 |

---

## Checkpoint 91 — 91k (91043 episodes)

**Episodes:** 91,043  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2477 |
| Coherence (Success) | 0.1860 |
| Coherence (Failure) | 0.2133 |
| Gradient Magnitude (Success) | 0.2365 |
| Gradient Magnitude (Failure) | 0.2003 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.4291 |
| Clusters | 524 |
| Noise Fraction | 0.0986 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.057 | 0.019 | 0.097 |
| locked_door | 0.057 | -0.000 | 0.111 | 0.279 |
| open_door | 0.019 | 0.111 | 0.000 | 0.071 |
| target_ball | 0.097 | 0.279 | 0.071 | 0.000 |

---

## Checkpoint 96 — 96k (96003 episodes)

**Episodes:** 96,003  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.183) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6665 |
| Coherence (Success) | 0.3213 |
| Coherence (Failure) | 0.1308 |
| Gradient Magnitude (Success) | 0.2854 |
| Gradient Magnitude (Failure) | 0.1867 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 11.2331 |
| Clusters | 483 |
| Noise Fraction | 0.0905 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.049 | 0.027 | 0.103 |
| locked_door | 0.049 | 0.000 | 0.123 | 0.263 |
| open_door | 0.027 | 0.123 | 0.000 | 0.068 |
| target_ball | 0.103 | 0.263 | 0.068 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,040  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1597 |
| Coherence (Success) | 0.0797 |
| Coherence (Failure) | 0.2134 |
| Gradient Magnitude (Success) | 0.1795 |
| Gradient Magnitude (Failure) | 0.1851 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.7051 |
| Clusters | 542 |
| Noise Fraction | 0.1005 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.074 | 0.014 | 0.097 |
| locked_door | 0.074 | 0.000 | 0.135 | 0.309 |
| open_door | 0.014 | 0.135 | 0.000 | 0.060 |
| target_ball | 0.097 | 0.309 | 0.060 | 0.000 |
