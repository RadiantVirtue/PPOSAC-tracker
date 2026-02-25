# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 4  
**Total episodes:** 100,072  
**Experiment root:** `Proof-of-Concept-Runs\seed_4`  
**Generated:** 2026-02-25 01:38

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1009 episodes) | 1,009 | 0.8972 | 0.4063 | 0.2864 | 0.1031 | 0.0688 | 0.0764 | {'correlation': 0.6210590034081188, 'p_value': 0.188187159830816} |
| Checkpoint 6 — 6k (6018 episodes) | 6,018 | -0.0342 | 0.4008 | 0.0764 | 0.4225 | 0.0898 | 0.8421 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 11 — 11k (11033 episodes) | 11,033 | 0.4333 | 0.1616 | 0.0704 | 0.3294 | 0.1235 | 0.9867 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 16 — 16k (16012 episodes) | 16,012 | 0.1397 | 0.2228 | 0.1411 | 0.4254 | 0.1582 | 1.7666 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 21 — 21k (21029 episodes) | 21,029 | 0.1345 | 0.2624 | 0.1821 | 0.4854 | 0.1634 | 1.8843 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 26 — 26k (26030 episodes) | 26,030 | 0.1586 | 0.1766 | 0.1803 | 0.2818 | 0.1943 | 2.1322 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 31 — 31k (31049 episodes) | 31,049 | 0.3637 | 0.1634 | 0.1345 | 0.2650 | 0.1610 | 3.3911 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 36 — 36k (36047 episodes) | 36,047 | 0.5325 | 0.3067 | 0.2471 | 0.3518 | 0.2330 | 3.0067 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 41 — 41k (41065 episodes) | 41,065 | 0.3670 | 0.1519 | 0.1061 | 0.2496 | 0.1392 | 3.8872 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 46 — 46k (46046 episodes) | 46,046 | 0.2528 | 0.2284 | 0.1608 | 0.3010 | 0.1697 | 4.0688 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 51 — 51k (51057 episodes) | 51,057 | 0.3303 | 0.2489 | 0.1555 | 0.2643 | 0.1587 | 4.1918 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 56 — 56k (56013 episodes) | 56,013 | 0.3170 | 0.2112 | 0.2615 | 0.2335 | 0.2062 | 4.8221 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 61 — 61k (61044 episodes) | 61,044 | -0.0002 | 0.2294 | 0.0952 | 0.2738 | 0.1669 | 5.5670 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 66 — 66k (66001 episodes) | 66,001 | 0.3348 | 0.2535 | 0.1475 | 0.2844 | 0.2030 | 6.0903 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 71 — 71k (71038 episodes) | 71,038 | 0.4000 | 0.2314 | 0.1245 | 0.2902 | 0.1579 | 5.0824 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 76 — 76k (76013 episodes) | 76,013 | 0.4753 | 0.2347 | 0.1906 | 0.2840 | 0.2302 | 6.2282 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 81 — 81k (81068 episodes) | 81,068 | 0.3979 | 0.1121 | 0.1130 | 0.2306 | 0.1597 | 6.6564 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 86 — 86k (86045 episodes) | 86,045 | 0.1109 | 0.1435 | 0.1502 | 0.2142 | 0.1707 | 8.5499 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 91 — 91k (91033 episodes) | 91,033 | 0.6191 | 0.1425 | 0.1292 | 0.2205 | 0.1758 | 8.9510 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 96 — 96k (96062 episodes) | 96,062 | 0.1391 | 0.1990 | 0.2186 | 0.2440 | 0.2008 | 8.7920 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,072 | 0.4537 | 0.2395 | 0.1197 | 0.3026 | 0.1654 | 10.2542 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

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

## Checkpoint 11 — 11k (11033 episodes)

**Episodes:** 11,033  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.957) | top 25% (≥ 2.223)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4333 |
| Coherence (Success) | 0.1616 |
| Coherence (Failure) | 0.0704 |
| Gradient Magnitude (Success) | 0.3294 |
| Gradient Magnitude (Failure) | 0.1235 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9867 |
| Clusters | 1,050 |
| Noise Fraction | 0.0494 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.313 | 0.047 | 0.201 |
| locked_door | 0.313 | -0.000 | 0.294 | 0.688 |
| open_door | 0.047 | 0.294 | 0.000 | 0.256 |
| target_ball | 0.201 | 0.688 | 0.256 | 0.000 |

---

## Checkpoint 16 — 16k (16012 episodes)

**Episodes:** 16,012  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.020) | top 25% (≥ 2.240)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1397 |
| Coherence (Success) | 0.2228 |
| Coherence (Failure) | 0.1411 |
| Gradient Magnitude (Success) | 0.4254 |
| Gradient Magnitude (Failure) | 0.1582 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7666 |
| Clusters | 1,012 |
| Noise Fraction | 0.0507 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.105 | 0.033 | 0.111 |
| locked_door | 0.105 | 0.000 | 0.185 | 0.259 |
| open_door | 0.033 | 0.185 | -0.000 | 0.108 |
| target_ball | 0.111 | 0.259 | 0.108 | 0.000 |

---

## Checkpoint 21 — 21k (21029 episodes)

**Episodes:** 21,029  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.107) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1345 |
| Coherence (Success) | 0.2624 |
| Coherence (Failure) | 0.1821 |
| Gradient Magnitude (Success) | 0.4854 |
| Gradient Magnitude (Failure) | 0.1634 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.8843 |
| Clusters | 691 |
| Noise Fraction | 0.0769 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.113 | 0.034 | 0.296 |
| locked_door | 0.113 | 0.000 | 0.215 | 0.681 |
| open_door | 0.034 | 0.215 | -0.000 | 0.191 |
| target_ball | 0.296 | 0.681 | 0.191 | 0.000 |

---

## Checkpoint 26 — 26k (26030 episodes)

**Episodes:** 26,030  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1586 |
| Coherence (Success) | 0.1766 |
| Coherence (Failure) | 0.1803 |
| Gradient Magnitude (Success) | 0.2818 |
| Gradient Magnitude (Failure) | 0.1943 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.1322 |
| Clusters | 503 |
| Noise Fraction | 0.1030 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.113 | 0.027 | 0.206 |
| locked_door | 0.113 | 0.000 | 0.192 | 0.559 |
| open_door | 0.027 | 0.192 | 0.000 | 0.145 |
| target_ball | 0.206 | 0.559 | 0.145 | -0.000 |

---

## Checkpoint 31 — 31k (31049 episodes)

**Episodes:** 31,049  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.127) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3637 |
| Coherence (Success) | 0.1634 |
| Coherence (Failure) | 0.1345 |
| Gradient Magnitude (Success) | 0.2650 |
| Gradient Magnitude (Failure) | 0.1610 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.3911 |
| Clusters | 609 |
| Noise Fraction | 0.0663 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.118 | 0.040 | 0.165 |
| locked_door | 0.118 | 0.000 | 0.223 | 0.496 |
| open_door | 0.040 | 0.223 | 0.000 | 0.109 |
| target_ball | 0.165 | 0.496 | 0.109 | 0.000 |

---

## Checkpoint 36 — 36k (36047 episodes)

**Episodes:** 36,047  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.133) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5325 |
| Coherence (Success) | 0.3067 |
| Coherence (Failure) | 0.2471 |
| Gradient Magnitude (Success) | 0.3518 |
| Gradient Magnitude (Failure) | 0.2330 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.0067 |
| Clusters | 664 |
| Noise Fraction | 0.0714 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.065 | 0.028 | 0.166 |
| locked_door | 0.065 | 0.000 | 0.140 | 0.388 |
| open_door | 0.028 | 0.140 | 0.000 | 0.100 |
| target_ball | 0.166 | 0.388 | 0.100 | 0.000 |

---

## Checkpoint 41 — 41k (41065 episodes)

**Episodes:** 41,065  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3670 |
| Coherence (Success) | 0.1519 |
| Coherence (Failure) | 0.1061 |
| Gradient Magnitude (Success) | 0.2496 |
| Gradient Magnitude (Failure) | 0.1392 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8872 |
| Clusters | 524 |
| Noise Fraction | 0.1020 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.085 | 0.027 | 0.127 |
| locked_door | 0.085 | 0.000 | 0.172 | 0.381 |
| open_door | 0.027 | 0.172 | 0.000 | 0.067 |
| target_ball | 0.127 | 0.381 | 0.067 | 0.000 |

---

## Checkpoint 46 — 46k (46046 episodes)

**Episodes:** 46,046  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2528 |
| Coherence (Success) | 0.2284 |
| Coherence (Failure) | 0.1608 |
| Gradient Magnitude (Success) | 0.3010 |
| Gradient Magnitude (Failure) | 0.1697 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.0688 |
| Clusters | 540 |
| Noise Fraction | 0.0873 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.048 | 0.020 | 0.100 |
| locked_door | 0.048 | 0.000 | 0.092 | 0.257 |
| open_door | 0.020 | 0.092 | -0.000 | 0.074 |
| target_ball | 0.100 | 0.257 | 0.074 | 0.000 |

---

## Checkpoint 51 — 51k (51057 episodes)

**Episodes:** 51,057  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3303 |
| Coherence (Success) | 0.2489 |
| Coherence (Failure) | 0.1555 |
| Gradient Magnitude (Success) | 0.2643 |
| Gradient Magnitude (Failure) | 0.1587 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.1918 |
| Clusters | 574 |
| Noise Fraction | 0.0786 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.064 | 0.020 | 0.158 |
| locked_door | 0.064 | 0.000 | 0.122 | 0.373 |
| open_door | 0.020 | 0.122 | -0.000 | 0.096 |
| target_ball | 0.158 | 0.373 | 0.096 | 0.000 |

---

## Checkpoint 56 — 56k (56013 episodes)

**Episodes:** 56,013  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3170 |
| Coherence (Success) | 0.2112 |
| Coherence (Failure) | 0.2615 |
| Gradient Magnitude (Success) | 0.2335 |
| Gradient Magnitude (Failure) | 0.2062 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.8221 |
| Clusters | 523 |
| Noise Fraction | 0.0759 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.070 | 0.019 | 0.117 |
| locked_door | 0.070 | -0.000 | 0.133 | 0.321 |
| open_door | 0.019 | 0.133 | -0.000 | 0.065 |
| target_ball | 0.117 | 0.321 | 0.065 | 0.000 |

---

## Checkpoint 61 — 61k (61044 episodes)

**Episodes:** 61,044  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.097) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0002 |
| Coherence (Success) | 0.2294 |
| Coherence (Failure) | 0.0952 |
| Gradient Magnitude (Success) | 0.2738 |
| Gradient Magnitude (Failure) | 0.1669 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.5670 |
| Clusters | 695 |
| Noise Fraction | 0.0764 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.046 | 0.015 | 0.130 |
| locked_door | 0.046 | 0.000 | 0.082 | 0.284 |
| open_door | 0.015 | 0.082 | 0.000 | 0.083 |
| target_ball | 0.130 | 0.284 | 0.083 | -0.000 |

---

## Checkpoint 66 — 66k (66001 episodes)

**Episodes:** 66,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3348 |
| Coherence (Success) | 0.2535 |
| Coherence (Failure) | 0.1475 |
| Gradient Magnitude (Success) | 0.2844 |
| Gradient Magnitude (Failure) | 0.2030 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.0903 |
| Clusters | 549 |
| Noise Fraction | 0.0846 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.080 | 0.027 | 0.111 |
| locked_door | 0.080 | 0.000 | 0.156 | 0.335 |
| open_door | 0.027 | 0.156 | 0.000 | 0.058 |
| target_ball | 0.111 | 0.335 | 0.058 | 0.000 |

---

## Checkpoint 71 — 71k (71038 episodes)

**Episodes:** 71,038  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.253)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4000 |
| Coherence (Success) | 0.2314 |
| Coherence (Failure) | 0.1245 |
| Gradient Magnitude (Success) | 0.2902 |
| Gradient Magnitude (Failure) | 0.1579 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.0824 |
| Clusters | 543 |
| Noise Fraction | 0.0861 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.037 | 0.020 | 0.126 |
| locked_door | 0.037 | -0.000 | 0.095 | 0.268 |
| open_door | 0.020 | 0.095 | 0.000 | 0.073 |
| target_ball | 0.126 | 0.268 | 0.073 | 0.000 |

---

## Checkpoint 76 — 76k (76013 episodes)

**Episodes:** 76,013  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4753 |
| Coherence (Success) | 0.2347 |
| Coherence (Failure) | 0.1906 |
| Gradient Magnitude (Success) | 0.2840 |
| Gradient Magnitude (Failure) | 0.2302 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.2282 |
| Clusters | 540 |
| Noise Fraction | 0.0957 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.055 | 0.020 | 0.129 |
| locked_door | 0.055 | -0.000 | 0.118 | 0.321 |
| open_door | 0.020 | 0.118 | 0.000 | 0.074 |
| target_ball | 0.129 | 0.321 | 0.074 | 0.000 |

---

## Checkpoint 81 — 81k (81068 episodes)

**Episodes:** 81,068  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.183) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3979 |
| Coherence (Success) | 0.1121 |
| Coherence (Failure) | 0.1130 |
| Gradient Magnitude (Success) | 0.2306 |
| Gradient Magnitude (Failure) | 0.1597 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.6564 |
| Clusters | 460 |
| Noise Fraction | 0.1061 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.071 | 0.021 | 0.140 |
| locked_door | 0.071 | 0.000 | 0.141 | 0.371 |
| open_door | 0.021 | 0.141 | -0.000 | 0.078 |
| target_ball | 0.140 | 0.371 | 0.078 | -0.000 |

---

## Checkpoint 86 — 86k (86045 episodes)

**Episodes:** 86,045  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1109 |
| Coherence (Success) | 0.1435 |
| Coherence (Failure) | 0.1502 |
| Gradient Magnitude (Success) | 0.2142 |
| Gradient Magnitude (Failure) | 0.1707 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.5499 |
| Clusters | 599 |
| Noise Fraction | 0.0827 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.053 | 0.020 | 0.120 |
| locked_door | 0.053 | 0.000 | 0.110 | 0.300 |
| open_door | 0.020 | 0.110 | 0.000 | 0.067 |
| target_ball | 0.120 | 0.300 | 0.067 | 0.000 |

---

## Checkpoint 91 — 91k (91033 episodes)

**Episodes:** 91,033  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6191 |
| Coherence (Success) | 0.1425 |
| Coherence (Failure) | 0.1292 |
| Gradient Magnitude (Success) | 0.2205 |
| Gradient Magnitude (Failure) | 0.1758 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.9510 |
| Clusters | 545 |
| Noise Fraction | 0.0911 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.054 | 0.016 | 0.108 |
| locked_door | 0.054 | 0.000 | 0.106 | 0.287 |
| open_door | 0.016 | 0.106 | 0.000 | 0.064 |
| target_ball | 0.108 | 0.287 | 0.064 | -0.000 |

---

## Checkpoint 96 — 96k (96062 episodes)

**Episodes:** 96,062  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1391 |
| Coherence (Success) | 0.1990 |
| Coherence (Failure) | 0.2186 |
| Gradient Magnitude (Success) | 0.2440 |
| Gradient Magnitude (Failure) | 0.2008 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.7920 |
| Clusters | 527 |
| Noise Fraction | 0.0947 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.068 | 0.017 | 0.105 |
| locked_door | 0.068 | 0.000 | 0.125 | 0.305 |
| open_door | 0.017 | 0.125 | -0.000 | 0.056 |
| target_ball | 0.105 | 0.305 | 0.056 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,072  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4537 |
| Coherence (Success) | 0.2395 |
| Coherence (Failure) | 0.1197 |
| Gradient Magnitude (Success) | 0.3026 |
| Gradient Magnitude (Failure) | 0.1654 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.2542 |
| Clusters | 487 |
| Noise Fraction | 0.0885 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.072 | 0.011 | 0.098 |
| locked_door | 0.072 | 0.000 | 0.115 | 0.311 |
| open_door | 0.011 | 0.115 | 0.000 | 0.066 |
| target_ball | 0.098 | 0.311 | 0.066 | -0.000 |
