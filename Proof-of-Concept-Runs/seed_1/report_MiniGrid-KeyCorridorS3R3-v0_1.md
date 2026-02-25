# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 1  
**Total episodes:** 100,061  
**Experiment root:** `Proof-of-Concept-Runs\seed_1`  
**Generated:** 2026-02-24 19:47

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 1k (1000 episodes) | 1,000 | 0.2424 | 0.1979 | 0.3319 | 0.0634 | 0.0786 | 0.0653 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6 — 6k (6016 episodes) | 6,016 | 0.0957 | 0.1729 | 0.1004 | 0.3079 | 0.1375 | 0.9080 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 11 — 11k (11008 episodes) | 11,008 | 0.0434 | 0.2350 | 0.1011 | 0.3998 | 0.1398 | 1.3230 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 16 — 16k (16047 episodes) | 16,047 | 0.3912 | 0.1320 | 0.2559 | 0.3301 | 0.2275 | 1.6173 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 21 — 21k (21058 episodes) | 21,058 | 0.1790 | 0.2016 | 0.3570 | 0.2957 | 0.2775 | 1.8881 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 26 — 26k (26025 episodes) | 26,025 | 0.5599 | 0.2410 | 0.2076 | 0.3795 | 0.2436 | 1.9709 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 31 — 31k (31017 episodes) | 31,017 | 0.4986 | 0.2669 | 0.1621 | 0.2849 | 0.2094 | 2.5519 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 36 — 36k (36060 episodes) | 36,060 | 0.5775 | 0.2181 | 0.1959 | 0.2645 | 0.2009 | 3.4615 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 41 — 41k (41004 episodes) | 41,004 | 0.1884 | 0.2142 | 0.2219 | 0.2514 | 0.2036 | 3.2886 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 46 — 46k (46024 episodes) | 46,024 | 0.3555 | 0.1563 | 0.1701 | 0.2338 | 0.1819 | 4.2847 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 51 — 51k (51062 episodes) | 51,062 | 0.3731 | 0.1581 | 0.1366 | 0.2300 | 0.1792 | 3.8735 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 56 — 56k (56015 episodes) | 56,015 | 0.4019 | 0.2386 | 0.2195 | 0.2238 | 0.1867 | 4.1893 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 61 — 61k (61048 episodes) | 61,048 | 0.3715 | 0.2028 | 0.2227 | 0.2487 | 0.2046 | 5.3783 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 66 — 66k (66034 episodes) | 66,034 | 0.1384 | 0.2190 | 0.1527 | 0.3235 | 0.1817 | 5.5966 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 71 — 71k (71043 episodes) | 71,043 | 0.5578 | 0.1499 | 0.1940 | 0.2068 | 0.1800 | 6.0023 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 76 — 76k (76033 episodes) | 76,033 | 0.5405 | 0.1772 | 0.1093 | 0.2768 | 0.1565 | 6.0231 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 81 — 81k (81020 episodes) | 81,020 | 0.4147 | 0.2112 | 0.1334 | 0.2471 | 0.1919 | 6.1094 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 86 — 86k (86048 episodes) | 86,048 | 0.5240 | 0.1936 | 0.1960 | 0.2418 | 0.1782 | 6.2631 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 91 — 91k (91036 episodes) | 91,036 | 0.2593 | 0.1513 | 0.1951 | 0.1876 | 0.1919 | 6.9887 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 96 — 96k (96007 episodes) | 96,007 | 0.5592 | 0.1360 | 0.2085 | 0.2293 | 0.2097 | 7.4583 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,061 | 0.5694 | 0.1425 | 0.1501 | 0.2419 | 0.1899 | 7.9328 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

---

## Checkpoint 1 — 1k (1000 episodes)

**Episodes:** 1,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2424 |
| Coherence (Success) | 0.1979 |
| Coherence (Failure) | 0.3319 |
| Gradient Magnitude (Success) | 0.0634 |
| Gradient Magnitude (Failure) | 0.0786 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0653 |
| Clusters | 2,123 |
| Noise Fraction | 0.0341 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.086 | 0.089 | 0.259 |
| locked_door | 0.086 | 0.000 | 0.053 | 0.509 |
| open_door | 0.089 | 0.053 | 0.000 | 0.564 |
| target_ball | 0.259 | 0.509 | 0.564 | 0.000 |

---

## Checkpoint 6 — 6k (6016 episodes)

**Episodes:** 6,016  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.187)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0957 |
| Coherence (Success) | 0.1729 |
| Coherence (Failure) | 0.1004 |
| Gradient Magnitude (Success) | 0.3079 |
| Gradient Magnitude (Failure) | 0.1375 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9080 |
| Clusters | 1,256 |
| Noise Fraction | 0.0381 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.561 | 0.411 | 0.227 |
| locked_door | 0.561 | 0.000 | 0.212 | 0.892 |
| open_door | 0.411 | 0.212 | 0.000 | 0.713 |
| target_ball | 0.227 | 0.892 | 0.713 | 0.000 |

---

## Checkpoint 11 — 11k (11008 episodes)

**Episodes:** 11,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.010) | top 25% (≥ 2.250)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0434 |
| Coherence (Success) | 0.2350 |
| Coherence (Failure) | 0.1011 |
| Gradient Magnitude (Success) | 0.3998 |
| Gradient Magnitude (Failure) | 0.1398 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3230 |
| Clusters | 957 |
| Noise Fraction | 0.0554 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.175 | 0.056 | 0.214 |
| locked_door | 0.175 | 0.000 | 0.157 | 0.633 |
| open_door | 0.056 | 0.157 | 0.000 | 0.258 |
| target_ball | 0.214 | 0.633 | 0.258 | 0.000 |

---

## Checkpoint 16 — 16k (16047 episodes)

**Episodes:** 16,047  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.097) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3912 |
| Coherence (Success) | 0.1320 |
| Coherence (Failure) | 0.2559 |
| Gradient Magnitude (Success) | 0.3301 |
| Gradient Magnitude (Failure) | 0.2275 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.6173 |
| Clusters | 743 |
| Noise Fraction | 0.0727 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.179 | 0.060 | 0.187 |
| locked_door | 0.179 | 0.000 | 0.291 | 0.603 |
| open_door | 0.060 | 0.291 | -0.000 | 0.113 |
| target_ball | 0.187 | 0.603 | 0.113 | 0.000 |

---

## Checkpoint 21 — 21k (21058 episodes)

**Episodes:** 21,058  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1790 |
| Coherence (Success) | 0.2016 |
| Coherence (Failure) | 0.3570 |
| Gradient Magnitude (Success) | 0.2957 |
| Gradient Magnitude (Failure) | 0.2775 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.8881 |
| Clusters | 582 |
| Noise Fraction | 0.1017 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.120 | 0.028 | 0.147 |
| locked_door | 0.120 | 0.000 | 0.216 | 0.469 |
| open_door | 0.028 | 0.216 | 0.000 | 0.082 |
| target_ball | 0.147 | 0.469 | 0.082 | 0.000 |

---

## Checkpoint 26 — 26k (26025 episodes)

**Episodes:** 26,025  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5599 |
| Coherence (Success) | 0.2410 |
| Coherence (Failure) | 0.2076 |
| Gradient Magnitude (Success) | 0.3795 |
| Gradient Magnitude (Failure) | 0.2436 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.9709 |
| Clusters | 568 |
| Noise Fraction | 0.0913 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.121 | 0.021 | 0.121 |
| locked_door | 0.121 | 0.000 | 0.178 | 0.430 |
| open_door | 0.021 | 0.178 | 0.000 | 0.085 |
| target_ball | 0.121 | 0.430 | 0.085 | 0.000 |

---

## Checkpoint 31 — 31k (31017 episodes)

**Episodes:** 31,017  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4986 |
| Coherence (Success) | 0.2669 |
| Coherence (Failure) | 0.1621 |
| Gradient Magnitude (Success) | 0.2849 |
| Gradient Magnitude (Failure) | 0.2094 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5519 |
| Clusters | 554 |
| Noise Fraction | 0.0761 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.124 | 0.032 | 0.152 |
| locked_door | 0.124 | 0.000 | 0.186 | 0.494 |
| open_door | 0.032 | 0.186 | 0.000 | 0.113 |
| target_ball | 0.152 | 0.494 | 0.113 | 0.000 |

---

## Checkpoint 36 — 36k (36060 episodes)

**Episodes:** 36,060  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5775 |
| Coherence (Success) | 0.2181 |
| Coherence (Failure) | 0.1959 |
| Gradient Magnitude (Success) | 0.2645 |
| Gradient Magnitude (Failure) | 0.2009 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.4615 |
| Clusters | 595 |
| Noise Fraction | 0.1008 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.069 | 0.027 | 0.096 |
| locked_door | 0.069 | 0.000 | 0.124 | 0.284 |
| open_door | 0.027 | 0.124 | 0.000 | 0.050 |
| target_ball | 0.096 | 0.284 | 0.050 | 0.000 |

---

## Checkpoint 41 — 41k (41004 episodes)

**Episodes:** 41,004  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1884 |
| Coherence (Success) | 0.2142 |
| Coherence (Failure) | 0.2219 |
| Gradient Magnitude (Success) | 0.2514 |
| Gradient Magnitude (Failure) | 0.2036 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.2886 |
| Clusters | 504 |
| Noise Fraction | 0.0933 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.088 | 0.015 | 0.076 |
| locked_door | 0.088 | -0.000 | 0.138 | 0.302 |
| open_door | 0.015 | 0.138 | 0.000 | 0.053 |
| target_ball | 0.076 | 0.302 | 0.053 | 0.000 |

---

## Checkpoint 46 — 46k (46024 episodes)

**Episodes:** 46,024  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3555 |
| Coherence (Success) | 0.1563 |
| Coherence (Failure) | 0.1701 |
| Gradient Magnitude (Success) | 0.2338 |
| Gradient Magnitude (Failure) | 0.1819 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.2847 |
| Clusters | 541 |
| Noise Fraction | 0.0792 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.101 | 0.017 | 0.111 |
| locked_door | 0.101 | 0.000 | 0.155 | 0.384 |
| open_door | 0.017 | 0.155 | -0.000 | 0.076 |
| target_ball | 0.111 | 0.384 | 0.076 | 0.000 |

---

## Checkpoint 51 — 51k (51062 episodes)

**Episodes:** 51,062  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3731 |
| Coherence (Success) | 0.1581 |
| Coherence (Failure) | 0.1366 |
| Gradient Magnitude (Success) | 0.2300 |
| Gradient Magnitude (Failure) | 0.1792 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8735 |
| Clusters | 498 |
| Noise Fraction | 0.0892 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.105 | 0.017 | 0.095 |
| locked_door | 0.105 | 0.000 | 0.152 | 0.359 |
| open_door | 0.017 | 0.152 | 0.000 | 0.071 |
| target_ball | 0.095 | 0.359 | 0.071 | -0.000 |

---

## Checkpoint 56 — 56k (56015 episodes)

**Episodes:** 56,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4019 |
| Coherence (Success) | 0.2386 |
| Coherence (Failure) | 0.2195 |
| Gradient Magnitude (Success) | 0.2238 |
| Gradient Magnitude (Failure) | 0.1867 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.1893 |
| Clusters | 598 |
| Noise Fraction | 0.0854 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.073 | 0.018 | 0.103 |
| locked_door | 0.073 | 0.000 | 0.125 | 0.315 |
| open_door | 0.018 | 0.125 | 0.000 | 0.070 |
| target_ball | 0.103 | 0.315 | 0.070 | 0.000 |

---

## Checkpoint 61 — 61k (61048 episodes)

**Episodes:** 61,048  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3715 |
| Coherence (Success) | 0.2028 |
| Coherence (Failure) | 0.2227 |
| Gradient Magnitude (Success) | 0.2487 |
| Gradient Magnitude (Failure) | 0.2046 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.3783 |
| Clusters | 529 |
| Noise Fraction | 0.0862 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.083 | 0.017 | 0.084 |
| locked_door | 0.083 | 0.000 | 0.130 | 0.302 |
| open_door | 0.017 | 0.130 | -0.000 | 0.060 |
| target_ball | 0.084 | 0.302 | 0.060 | -0.000 |

---

## Checkpoint 66 — 66k (66034 episodes)

**Episodes:** 66,034  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1384 |
| Coherence (Success) | 0.2190 |
| Coherence (Failure) | 0.1527 |
| Gradient Magnitude (Success) | 0.3235 |
| Gradient Magnitude (Failure) | 0.1817 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.5966 |
| Clusters | 524 |
| Noise Fraction | 0.0847 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.065 | 0.015 | 0.090 |
| locked_door | 0.065 | 0.000 | 0.106 | 0.281 |
| open_door | 0.015 | 0.106 | 0.000 | 0.062 |
| target_ball | 0.090 | 0.281 | 0.062 | 0.000 |

---

## Checkpoint 71 — 71k (71043 episodes)

**Episodes:** 71,043  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.183) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5578 |
| Coherence (Success) | 0.1499 |
| Coherence (Failure) | 0.1940 |
| Gradient Magnitude (Success) | 0.2068 |
| Gradient Magnitude (Failure) | 0.1800 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.0023 |
| Clusters | 466 |
| Noise Fraction | 0.0721 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.089 | 0.016 | 0.076 |
| locked_door | 0.089 | 0.000 | 0.132 | 0.291 |
| open_door | 0.016 | 0.132 | 0.000 | 0.051 |
| target_ball | 0.076 | 0.291 | 0.051 | 0.000 |

---

## Checkpoint 76 — 76k (76033 episodes)

**Episodes:** 76,033  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.130) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5405 |
| Coherence (Success) | 0.1772 |
| Coherence (Failure) | 0.1093 |
| Gradient Magnitude (Success) | 0.2768 |
| Gradient Magnitude (Failure) | 0.1565 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.0231 |
| Clusters | 605 |
| Noise Fraction | 0.0788 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.066 | 0.015 | 0.067 |
| locked_door | 0.066 | 0.000 | 0.096 | 0.234 |
| open_door | 0.015 | 0.096 | 0.000 | 0.047 |
| target_ball | 0.067 | 0.234 | 0.047 | 0.000 |

---

## Checkpoint 81 — 81k (81020 episodes)

**Episodes:** 81,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4147 |
| Coherence (Success) | 0.2112 |
| Coherence (Failure) | 0.1334 |
| Gradient Magnitude (Success) | 0.2471 |
| Gradient Magnitude (Failure) | 0.1919 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.1094 |
| Clusters | 485 |
| Noise Fraction | 0.0913 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.060 | 0.013 | 0.065 |
| locked_door | 0.060 | -0.000 | 0.095 | 0.221 |
| open_door | 0.013 | 0.095 | 0.000 | 0.045 |
| target_ball | 0.065 | 0.221 | 0.045 | 0.000 |

---

## Checkpoint 86 — 86k (86048 episodes)

**Episodes:** 86,048  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5240 |
| Coherence (Success) | 0.1936 |
| Coherence (Failure) | 0.1960 |
| Gradient Magnitude (Success) | 0.2418 |
| Gradient Magnitude (Failure) | 0.1782 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.2631 |
| Clusters | 521 |
| Noise Fraction | 0.0936 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.082 | 0.012 | 0.070 |
| locked_door | 0.082 | 0.000 | 0.125 | 0.274 |
| open_door | 0.012 | 0.125 | -0.000 | 0.041 |
| target_ball | 0.070 | 0.274 | 0.041 | 0.000 |

---

## Checkpoint 91 — 91k (91036 episodes)

**Episodes:** 91,036  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2593 |
| Coherence (Success) | 0.1513 |
| Coherence (Failure) | 0.1951 |
| Gradient Magnitude (Success) | 0.1876 |
| Gradient Magnitude (Failure) | 0.1919 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.9887 |
| Clusters | 490 |
| Noise Fraction | 0.0773 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.083 | 0.014 | 0.082 |
| locked_door | 0.083 | -0.000 | 0.133 | 0.300 |
| open_door | 0.014 | 0.133 | 0.000 | 0.049 |
| target_ball | 0.082 | 0.300 | 0.049 | 0.000 |

---

## Checkpoint 96 — 96k (96007 episodes)

**Episodes:** 96,007  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5592 |
| Coherence (Success) | 0.1360 |
| Coherence (Failure) | 0.2085 |
| Gradient Magnitude (Success) | 0.2293 |
| Gradient Magnitude (Failure) | 0.2097 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.4583 |
| Clusters | 564 |
| Noise Fraction | 0.1013 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.066 | 0.011 | 0.069 |
| locked_door | 0.066 | -0.000 | 0.101 | 0.243 |
| open_door | 0.011 | 0.101 | 0.000 | 0.046 |
| target_ball | 0.069 | 0.243 | 0.046 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,061  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5694 |
| Coherence (Success) | 0.1425 |
| Coherence (Failure) | 0.1501 |
| Gradient Magnitude (Success) | 0.2419 |
| Gradient Magnitude (Failure) | 0.1899 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.9328 |
| Clusters | 569 |
| Noise Fraction | 0.0911 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.070 | 0.018 | 0.114 |
| locked_door | 0.070 | 0.000 | 0.120 | 0.323 |
| open_door | 0.018 | 0.120 | 0.000 | 0.068 |
| target_ball | 0.114 | 0.323 | 0.068 | 0.000 |
