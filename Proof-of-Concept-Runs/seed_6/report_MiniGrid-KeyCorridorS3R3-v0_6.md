# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 6  
**Total episodes:** 100,015  
**Experiment root:** `Proof-of-Concept-Runs\seed_6`  
**Generated:** 2026-02-25 23:04

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reached_door @ ep0 | 0 | 0.7475 | 0.2740 | 0.1370 | 0.0439 | 0.0366 | 0.0761 | 0.0017 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| picked_up_target @ ep496 | 496 | 0.8718 | 0.6128 | 0.3752 | 0.1131 | 0.0817 | 0.0522 | 0.0072 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5 — 2k (2513 episodes) | 2,513 | 0.8379 | 0.2604 | 0.3622 | 0.0687 | 0.0775 | 0.0276 | 0.0167 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 10 — 5k (5010 episodes) | 5,010 | 0.4855 | 0.2004 | 0.2016 | 0.0576 | 0.0507 | 0.0378 | 0.0414 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 15 — 7k (7506 episodes) | 7,506 | 0.1317 | 0.3838 | 0.1864 | 0.3118 | 0.0933 | 0.4665 | 0.8373 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 20 — 10k (10010 episodes) | 10,010 | 0.2922 | 0.1709 | 0.1350 | 0.2518 | 0.0770 | 1.0800 | 0.9307 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 25 — 12k (12518 episodes) | 12,518 | 0.2045 | 0.1665 | 0.0575 | 0.2248 | 0.0759 | 1.3885 | 1.1561 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 30 — 15k (15015 episodes) | 15,015 | 0.1280 | 0.2923 | 0.0416 | 0.3382 | 0.1038 | 1.5166 | 0.6977 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 35 — 17k (17528 episodes) | 17,528 | 0.4095 | 0.2331 | 0.1134 | 0.3698 | 0.1025 | 1.5388 | 0.7541 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 40 — 20k (20017 episodes) | 20,017 | 0.2688 | 0.2517 | 0.1256 | 0.2946 | 0.1203 | 2.2515 | 0.4302 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 45 — 22k (22551 episodes) | 22,551 | 0.3139 | 0.1444 | 0.1491 | 0.2365 | 0.1813 | 2.5827 | 0.1450 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 50 — 25k (25032 episodes) | 25,032 | 0.1069 | 0.0624 | 0.2850 | 0.1987 | 0.2551 | 2.5093 | 0.1498 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 55 — 27k (27518 episodes) | 27,518 | 0.3750 | 0.1690 | 0.0777 | 0.2528 | 0.1490 | 2.9685 | 0.1544 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 60 — 30k (30045 episodes) | 30,045 | 0.3048 | 0.1947 | 0.1946 | 0.2454 | 0.1943 | 2.5132 | 0.0408 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 65 — 32k (32541 episodes) | 32,541 | 0.4357 | 0.1244 | 0.2126 | 0.2294 | 0.2117 | 2.9509 | 0.0598 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 70 — 35k (35021 episodes) | 35,021 | -0.1066 | 0.3028 | 0.2149 | 0.3333 | 0.2114 | 3.2635 | 0.0493 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 75 — 37k (37537 episodes) | 37,537 | 0.2129 | 0.1734 | 0.1676 | 0.2187 | 0.1473 | 3.8555 | 0.0698 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 80 — 40k (40000 episodes) | 40,000 | 0.2101 | 0.1776 | 0.1129 | 0.2288 | 0.1429 | 3.3774 | 0.0458 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 85 — 42k (42519 episodes) | 42,519 | 0.2215 | 0.1951 | 0.1246 | 0.2425 | 0.1925 | 4.3868 | 0.0426 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 90 — 45k (45059 episodes) | 45,059 | 0.3875 | 0.2458 | 0.2638 | 0.2573 | 0.2146 | 4.1618 | 0.0282 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 95 — 47k (47516 episodes) | 47,516 | 0.1865 | 0.2849 | 0.1523 | 0.2896 | 0.1592 | 3.9603 | 0.0274 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 100 — 50k (50028 episodes) | 50,028 | 0.1448 | 0.2046 | 0.1030 | 0.2049 | 0.1508 | 4.9647 | 0.0330 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 105 — 52k (52547 episodes) | 52,547 | 0.1946 | 0.1829 | 0.1218 | 0.2497 | 0.1547 | 4.8082 | 0.0219 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 110 — 55k (55062 episodes) | 55,062 | 0.2736 | 0.1867 | 0.2026 | 0.2262 | 0.1799 | 5.2843 | 0.0228 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 115 — 57k (57540 episodes) | 57,540 | 0.0776 | 0.2362 | 0.1553 | 0.2254 | 0.1616 | 6.7833 | 0.0241 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 120 — 60k (60060 episodes) | 60,060 | 0.3596 | 0.2338 | 0.1596 | 0.2590 | 0.1704 | 5.9759 | 0.0198 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 125 — 62k (62571 episodes) | 62,571 | 0.2907 | 0.2498 | 0.1675 | 0.2622 | 0.1673 | 6.1363 | 0.0204 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 130 — 65k (65051 episodes) | 65,051 | 0.2306 | 0.2772 | 0.1760 | 0.2913 | 0.1776 | 6.2874 | 0.0163 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 135 — 67k (67550 episodes) | 67,550 | 0.3824 | 0.1486 | 0.1677 | 0.2260 | 0.1758 | 6.3717 | 0.0155 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 140 — 70k (70020 episodes) | 70,020 | 0.4662 | 0.2614 | 0.2257 | 0.2686 | 0.1919 | 5.2902 | 0.0181 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 145 — 72k (72524 episodes) | 72,524 | 0.3546 | 0.2387 | 0.1460 | 0.2816 | 0.1437 | 7.6624 | 0.0251 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 150 — 75k (75028 episodes) | 75,028 | 0.3023 | 0.2735 | 0.2038 | 0.2880 | 0.2000 | 7.9492 | 0.0233 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 155 — 77k (77563 episodes) | 77,563 | 0.4584 | 0.1532 | 0.0941 | 0.2368 | 0.1503 | 7.0320 | 0.0223 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 160 — 80k (80007 episodes) | 80,007 | 0.3163 | 0.1568 | 0.1173 | 0.2430 | 0.1629 | 7.6442 | 0.0216 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 165 — 82k (82567 episodes) | 82,567 | 0.4196 | 0.2036 | 0.1468 | 0.2338 | 0.1713 | 7.3253 | 0.0248 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 170 — 85k (85007 episodes) | 85,007 | 0.4282 | 0.1916 | 0.1509 | 0.2416 | 0.1733 | 7.5423 | 0.0187 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 175 — 87k (87539 episodes) | 87,539 | 0.4010 | 0.1637 | 0.1731 | 0.2290 | 0.2092 | 6.2761 | 0.0185 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 180 — 90k (90057 episodes) | 90,057 | 0.1271 | 0.1685 | 0.1280 | 0.2287 | 0.1641 | 7.7864 | 0.0214 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 185 — 92k (92552 episodes) | 92,552 | 0.3095 | 0.1672 | 0.1389 | 0.2082 | 0.1601 | 8.7758 | 0.0174 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 190 — 95k (95017 episodes) | 95,017 | 0.4417 | 0.1615 | 0.0921 | 0.2120 | 0.1495 | 9.3657 | 0.0297 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 195 — 97k (97507 episodes) | 97,507 | 0.6653 | 0.2140 | 0.1784 | 0.2689 | 0.1566 | 9.6365 | 0.0245 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 200 — 100k (100015 episodes) | 100,015 | 0.3867 | 0.2931 | 0.1210 | 0.2935 | 0.1768 | 9.1582 | 0.0195 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,015 | 0.2290 | 0.2623 | 0.2080 | 0.2964 | 0.1790 | 9.7720 | 0.0204 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

---

## reached_door @ ep0

**Episodes:** 0  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7475 |
| Coherence (Success) | 0.2740 |
| Coherence (Failure) | 0.1370 |
| Gradient Magnitude (Success) | 0.0439 |
| Gradient Magnitude (Failure) | 0.0366 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0761 |
| Cosine Distance | 0.0017 |
| Clusters | 2,078 |
| Noise Fraction | 0.0211 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.008 |
| locked_door | 0.003 | 0.000 | 0.004 | 0.007 |
| open_door | 0.003 | 0.004 | -0.000 | 0.005 |
| target_ball | 0.008 | 0.007 | 0.005 | 0.000 |

---

## picked_up_target @ ep496

**Episodes:** 496  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8718 |
| Coherence (Success) | 0.6128 |
| Coherence (Failure) | 0.3752 |
| Gradient Magnitude (Success) | 0.1131 |
| Gradient Magnitude (Failure) | 0.0817 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0522 |
| Cosine Distance | 0.0072 |
| Clusters | 2,191 |
| Noise Fraction | 0.0313 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.064 | 0.020 | 0.038 |
| locked_door | 0.064 | 0.000 | 0.060 | 0.132 |
| open_door | 0.020 | 0.060 | -0.000 | 0.060 |
| target_ball | 0.038 | 0.132 | 0.060 | 0.000 |

---

## Checkpoint 5 — 2k (2513 episodes)

**Episodes:** 2,513  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8379 |
| Coherence (Success) | 0.2604 |
| Coherence (Failure) | 0.3622 |
| Gradient Magnitude (Success) | 0.0687 |
| Gradient Magnitude (Failure) | 0.0775 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0276 |
| Cosine Distance | 0.0167 |
| Clusters | 2,099 |
| Noise Fraction | 0.0286 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.127 | 0.069 | 0.286 |
| locked_door | 0.127 | -0.000 | 0.139 | 0.353 |
| open_door | 0.069 | 0.139 | 0.000 | 0.333 |
| target_ball | 0.286 | 0.353 | 0.333 | 0.000 |

---

## Checkpoint 10 — 5k (5010 episodes)

**Episodes:** 5,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4855 |
| Coherence (Success) | 0.2004 |
| Coherence (Failure) | 0.2016 |
| Gradient Magnitude (Success) | 0.0576 |
| Gradient Magnitude (Failure) | 0.0507 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0378 |
| Cosine Distance | 0.0414 |
| Clusters | 1,740 |
| Noise Fraction | 0.0245 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.224 | 0.095 | 0.959 |
| locked_door | 0.224 | 0.000 | 0.248 | 1.029 |
| open_door | 0.095 | 0.248 | -0.000 | 1.143 |
| target_ball | 0.959 | 1.029 | 1.143 | -0.000 |

---

## Checkpoint 15 — 7k (7506 episodes)

**Episodes:** 7,506  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1317 |
| Coherence (Success) | 0.3838 |
| Coherence (Failure) | 0.1864 |
| Gradient Magnitude (Success) | 0.3118 |
| Gradient Magnitude (Failure) | 0.0933 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.4665 |
| Cosine Distance | 0.8373 |
| Clusters | 1,225 |
| Noise Fraction | 0.0206 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.148 | 0.410 | 1.111 |
| locked_door | 0.148 | 0.000 | 0.172 | 1.138 |
| open_door | 0.410 | 0.172 | 0.000 | 1.088 |
| target_ball | 1.111 | 1.138 | 1.088 | 0.000 |

---

## Checkpoint 20 — 10k (10010 episodes)

**Episodes:** 10,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 2.167)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2922 |
| Coherence (Success) | 0.1709 |
| Coherence (Failure) | 0.1350 |
| Gradient Magnitude (Success) | 0.2518 |
| Gradient Magnitude (Failure) | 0.0770 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.0800 |
| Cosine Distance | 0.9307 |
| Clusters | 1,071 |
| Noise Fraction | 0.0260 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.389 | 0.368 | 0.328 |
| locked_door | 0.389 | 0.000 | 0.264 | 0.895 |
| open_door | 0.368 | 0.264 | 0.000 | 0.564 |
| target_ball | 0.328 | 0.895 | 0.564 | -0.000 |

---

## Checkpoint 25 — 12k (12518 episodes)

**Episodes:** 12,518  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.400) | top 25% (≥ 2.213)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2045 |
| Coherence (Success) | 0.1665 |
| Coherence (Failure) | 0.0575 |
| Gradient Magnitude (Success) | 0.2248 |
| Gradient Magnitude (Failure) | 0.0759 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.3885 |
| Cosine Distance | 1.1561 |
| Clusters | 1,162 |
| Noise Fraction | 0.0333 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.222 | 0.123 | 0.065 |
| locked_door | 0.222 | -0.000 | 0.316 | 0.377 |
| open_door | 0.123 | 0.316 | -0.000 | 0.132 |
| target_ball | 0.065 | 0.377 | 0.132 | 0.000 |

---

## Checkpoint 30 — 15k (15015 episodes)

**Episodes:** 15,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.863) | top 25% (≥ 2.220)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1280 |
| Coherence (Success) | 0.2923 |
| Coherence (Failure) | 0.0416 |
| Gradient Magnitude (Success) | 0.3382 |
| Gradient Magnitude (Failure) | 0.1038 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.5166 |
| Cosine Distance | 0.6977 |
| Clusters | 1,149 |
| Noise Fraction | 0.0441 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.243 | 0.194 | 0.114 |
| locked_door | 0.243 | 0.000 | 0.491 | 0.455 |
| open_door | 0.194 | 0.491 | 0.000 | 0.208 |
| target_ball | 0.114 | 0.455 | 0.208 | 0.000 |

---

## Checkpoint 35 — 17k (17528 episodes)

**Episodes:** 17,528  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.947) | top 25% (≥ 2.227)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4095 |
| Coherence (Success) | 0.2331 |
| Coherence (Failure) | 0.1134 |
| Gradient Magnitude (Success) | 0.3698 |
| Gradient Magnitude (Failure) | 0.1025 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.5388 |
| Cosine Distance | 0.7541 |
| Clusters | 1,118 |
| Noise Fraction | 0.0411 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.352 | 0.127 | 0.136 |
| locked_door | 0.352 | -0.000 | 0.504 | 0.684 |
| open_door | 0.127 | 0.504 | 0.000 | 0.189 |
| target_ball | 0.136 | 0.684 | 0.189 | 0.000 |

---

## Checkpoint 40 — 20k (20017 episodes)

**Episodes:** 20,017  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.030) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2688 |
| Coherence (Success) | 0.2517 |
| Coherence (Failure) | 0.1256 |
| Gradient Magnitude (Success) | 0.2946 |
| Gradient Magnitude (Failure) | 0.1203 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.2515 |
| Cosine Distance | 0.4302 |
| Clusters | 918 |
| Noise Fraction | 0.0576 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.222 | 0.068 | 0.206 |
| locked_door | 0.222 | 0.000 | 0.356 | 0.722 |
| open_door | 0.068 | 0.356 | -0.000 | 0.168 |
| target_ball | 0.206 | 0.722 | 0.168 | -0.000 |

---

## Checkpoint 45 — 22k (22551 episodes)

**Episodes:** 22,551  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.100) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3139 |
| Coherence (Success) | 0.1444 |
| Coherence (Failure) | 0.1491 |
| Gradient Magnitude (Success) | 0.2365 |
| Gradient Magnitude (Failure) | 0.1813 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5827 |
| Cosine Distance | 0.1450 |
| Clusters | 766 |
| Noise Fraction | 0.0620 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.182 | 0.039 | 0.220 |
| locked_door | 0.182 | 0.000 | 0.273 | 0.676 |
| open_door | 0.039 | 0.273 | 0.000 | 0.178 |
| target_ball | 0.220 | 0.676 | 0.178 | 0.000 |

---

## Checkpoint 50 — 25k (25032 episodes)

**Episodes:** 25,032  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.103) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1069 |
| Coherence (Success) | 0.0624 |
| Coherence (Failure) | 0.2850 |
| Gradient Magnitude (Success) | 0.1987 |
| Gradient Magnitude (Failure) | 0.2551 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5093 |
| Cosine Distance | 0.1498 |
| Clusters | 770 |
| Noise Fraction | 0.0682 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.209 | 0.033 | 0.222 |
| locked_door | 0.209 | 0.000 | 0.312 | 0.734 |
| open_door | 0.033 | 0.312 | 0.000 | 0.155 |
| target_ball | 0.222 | 0.734 | 0.155 | 0.000 |

---

## Checkpoint 55 — 27k (27518 episodes)

**Episodes:** 27,518  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3750 |
| Coherence (Success) | 0.1690 |
| Coherence (Failure) | 0.0777 |
| Gradient Magnitude (Success) | 0.2528 |
| Gradient Magnitude (Failure) | 0.1490 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9685 |
| Cosine Distance | 0.1544 |
| Clusters | 654 |
| Noise Fraction | 0.0640 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.169 | 0.035 | 0.241 |
| locked_door | 0.169 | 0.000 | 0.254 | 0.684 |
| open_door | 0.035 | 0.254 | 0.000 | 0.198 |
| target_ball | 0.241 | 0.684 | 0.198 | 0.000 |

---

## Checkpoint 60 — 30k (30045 episodes)

**Episodes:** 30,045  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3048 |
| Coherence (Success) | 0.1947 |
| Coherence (Failure) | 0.1946 |
| Gradient Magnitude (Success) | 0.2454 |
| Gradient Magnitude (Failure) | 0.1943 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5132 |
| Cosine Distance | 0.0408 |
| Clusters | 532 |
| Noise Fraction | 0.0874 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.126 | 0.026 | 0.177 |
| locked_door | 0.126 | 0.000 | 0.205 | 0.512 |
| open_door | 0.026 | 0.205 | -0.000 | 0.124 |
| target_ball | 0.177 | 0.512 | 0.124 | 0.000 |

---

## Checkpoint 65 — 32k (32541 episodes)

**Episodes:** 32,541  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4357 |
| Coherence (Success) | 0.1244 |
| Coherence (Failure) | 0.2126 |
| Gradient Magnitude (Success) | 0.2294 |
| Gradient Magnitude (Failure) | 0.2117 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9509 |
| Cosine Distance | 0.0598 |
| Clusters | 556 |
| Noise Fraction | 0.0793 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.116 | 0.028 | 0.150 |
| locked_door | 0.116 | 0.000 | 0.200 | 0.458 |
| open_door | 0.028 | 0.200 | 0.000 | 0.106 |
| target_ball | 0.150 | 0.458 | 0.106 | 0.000 |

---

## Checkpoint 70 — 35k (35021 episodes)

**Episodes:** 35,021  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1066 |
| Coherence (Success) | 0.3028 |
| Coherence (Failure) | 0.2149 |
| Gradient Magnitude (Success) | 0.3333 |
| Gradient Magnitude (Failure) | 0.2114 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.2635 |
| Cosine Distance | 0.0493 |
| Clusters | 499 |
| Noise Fraction | 0.0844 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.132 | 0.031 | 0.182 |
| locked_door | 0.132 | 0.000 | 0.230 | 0.545 |
| open_door | 0.031 | 0.230 | 0.000 | 0.115 |
| target_ball | 0.182 | 0.545 | 0.115 | 0.000 |

---

## Checkpoint 75 — 37k (37537 episodes)

**Episodes:** 37,537  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2129 |
| Coherence (Success) | 0.1734 |
| Coherence (Failure) | 0.1676 |
| Gradient Magnitude (Success) | 0.2187 |
| Gradient Magnitude (Failure) | 0.1473 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8555 |
| Cosine Distance | 0.0698 |
| Clusters | 589 |
| Noise Fraction | 0.0783 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.107 | 0.029 | 0.177 |
| locked_door | 0.107 | -0.000 | 0.181 | 0.479 |
| open_door | 0.029 | 0.181 | 0.000 | 0.127 |
| target_ball | 0.177 | 0.479 | 0.127 | 0.000 |

---

## Checkpoint 80 — 40k (40000 episodes)

**Episodes:** 40,000  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2101 |
| Coherence (Success) | 0.1776 |
| Coherence (Failure) | 0.1129 |
| Gradient Magnitude (Success) | 0.2288 |
| Gradient Magnitude (Failure) | 0.1429 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.3774 |
| Cosine Distance | 0.0458 |
| Clusters | 578 |
| Noise Fraction | 0.0850 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.120 | 0.034 | 0.187 |
| locked_door | 0.120 | 0.000 | 0.198 | 0.527 |
| open_door | 0.034 | 0.198 | 0.000 | 0.139 |
| target_ball | 0.187 | 0.527 | 0.139 | 0.000 |

---

## Checkpoint 85 — 42k (42519 episodes)

**Episodes:** 42,519  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2215 |
| Coherence (Success) | 0.1951 |
| Coherence (Failure) | 0.1246 |
| Gradient Magnitude (Success) | 0.2425 |
| Gradient Magnitude (Failure) | 0.1925 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.3868 |
| Cosine Distance | 0.0426 |
| Clusters | 517 |
| Noise Fraction | 0.0817 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.094 | 0.028 | 0.173 |
| locked_door | 0.094 | 0.000 | 0.179 | 0.466 |
| open_door | 0.028 | 0.179 | -0.000 | 0.106 |
| target_ball | 0.173 | 0.466 | 0.106 | -0.000 |

---

## Checkpoint 90 — 45k (45059 episodes)

**Episodes:** 45,059  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3875 |
| Coherence (Success) | 0.2458 |
| Coherence (Failure) | 0.2638 |
| Gradient Magnitude (Success) | 0.2573 |
| Gradient Magnitude (Failure) | 0.2146 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.1618 |
| Cosine Distance | 0.0282 |
| Clusters | 529 |
| Noise Fraction | 0.0827 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.091 | 0.028 | 0.141 |
| locked_door | 0.091 | 0.000 | 0.161 | 0.404 |
| open_door | 0.028 | 0.161 | 0.000 | 0.105 |
| target_ball | 0.141 | 0.404 | 0.105 | -0.000 |

---

## Checkpoint 95 — 47k (47516 episodes)

**Episodes:** 47,516  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1865 |
| Coherence (Success) | 0.2849 |
| Coherence (Failure) | 0.1523 |
| Gradient Magnitude (Success) | 0.2896 |
| Gradient Magnitude (Failure) | 0.1592 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.9603 |
| Cosine Distance | 0.0274 |
| Clusters | 540 |
| Noise Fraction | 0.0814 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.090 | 0.031 | 0.197 |
| locked_door | 0.090 | 0.000 | 0.160 | 0.475 |
| open_door | 0.031 | 0.160 | 0.000 | 0.159 |
| target_ball | 0.197 | 0.475 | 0.159 | 0.000 |

---

## Checkpoint 100 — 50k (50028 episodes)

**Episodes:** 50,028  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1448 |
| Coherence (Success) | 0.2046 |
| Coherence (Failure) | 0.1030 |
| Gradient Magnitude (Success) | 0.2049 |
| Gradient Magnitude (Failure) | 0.1508 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.9647 |
| Cosine Distance | 0.0330 |
| Clusters | 593 |
| Noise Fraction | 0.0742 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.097 | 0.022 | 0.137 |
| locked_door | 0.097 | 0.000 | 0.163 | 0.407 |
| open_door | 0.022 | 0.163 | -0.000 | 0.097 |
| target_ball | 0.137 | 0.407 | 0.097 | 0.000 |

---

## Checkpoint 105 — 52k (52547 episodes)

**Episodes:** 52,547  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1946 |
| Coherence (Success) | 0.1829 |
| Coherence (Failure) | 0.1218 |
| Gradient Magnitude (Success) | 0.2497 |
| Gradient Magnitude (Failure) | 0.1547 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.8082 |
| Cosine Distance | 0.0219 |
| Clusters | 510 |
| Noise Fraction | 0.0983 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.056 | 0.022 | 0.169 |
| locked_door | 0.056 | -0.000 | 0.104 | 0.368 |
| open_door | 0.022 | 0.104 | 0.000 | 0.131 |
| target_ball | 0.169 | 0.368 | 0.131 | 0.000 |

---

## Checkpoint 110 — 55k (55062 episodes)

**Episodes:** 55,062  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2736 |
| Coherence (Success) | 0.1867 |
| Coherence (Failure) | 0.2026 |
| Gradient Magnitude (Success) | 0.2262 |
| Gradient Magnitude (Failure) | 0.1799 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.2843 |
| Cosine Distance | 0.0228 |
| Clusters | 482 |
| Noise Fraction | 0.0857 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.073 | 0.029 | 0.143 |
| locked_door | 0.073 | 0.000 | 0.149 | 0.374 |
| open_door | 0.029 | 0.149 | 0.000 | 0.106 |
| target_ball | 0.143 | 0.374 | 0.106 | 0.000 |

---

## Checkpoint 115 — 57k (57540 episodes)

**Episodes:** 57,540  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.127) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0776 |
| Coherence (Success) | 0.2362 |
| Coherence (Failure) | 0.1553 |
| Gradient Magnitude (Success) | 0.2254 |
| Gradient Magnitude (Failure) | 0.1616 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.7833 |
| Cosine Distance | 0.0241 |
| Clusters | 669 |
| Noise Fraction | 0.0750 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.038 | 0.021 | 0.119 |
| locked_door | 0.038 | -0.000 | 0.085 | 0.260 |
| open_door | 0.021 | 0.085 | 0.000 | 0.086 |
| target_ball | 0.119 | 0.260 | 0.086 | 0.000 |

---

## Checkpoint 120 — 60k (60060 episodes)

**Episodes:** 60,060  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3596 |
| Coherence (Success) | 0.2338 |
| Coherence (Failure) | 0.1596 |
| Gradient Magnitude (Success) | 0.2590 |
| Gradient Magnitude (Failure) | 0.1704 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.9759 |
| Cosine Distance | 0.0198 |
| Clusters | 531 |
| Noise Fraction | 0.0885 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.049 | 0.027 | 0.179 |
| locked_door | 0.049 | 0.000 | 0.112 | 0.371 |
| open_door | 0.027 | 0.112 | 0.000 | 0.124 |
| target_ball | 0.179 | 0.371 | 0.124 | 0.000 |

---

## Checkpoint 125 — 62k (62571 episodes)

**Episodes:** 62,571  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2907 |
| Coherence (Success) | 0.2498 |
| Coherence (Failure) | 0.1675 |
| Gradient Magnitude (Success) | 0.2622 |
| Gradient Magnitude (Failure) | 0.1673 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.1363 |
| Cosine Distance | 0.0204 |
| Clusters | 531 |
| Noise Fraction | 0.1027 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.051 | 0.025 | 0.193 |
| locked_door | 0.051 | 0.000 | 0.115 | 0.395 |
| open_door | 0.025 | 0.115 | -0.000 | 0.129 |
| target_ball | 0.193 | 0.395 | 0.129 | 0.000 |

---

## Checkpoint 130 — 65k (65051 episodes)

**Episodes:** 65,051  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2306 |
| Coherence (Success) | 0.2772 |
| Coherence (Failure) | 0.1760 |
| Gradient Magnitude (Success) | 0.2913 |
| Gradient Magnitude (Failure) | 0.1776 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.2874 |
| Cosine Distance | 0.0163 |
| Clusters | 471 |
| Noise Fraction | 0.0762 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.058 | 0.030 | 0.185 |
| locked_door | 0.058 | -0.000 | 0.125 | 0.398 |
| open_door | 0.030 | 0.125 | 0.000 | 0.134 |
| target_ball | 0.185 | 0.398 | 0.134 | 0.000 |

---

## Checkpoint 135 — 67k (67550 episodes)

**Episodes:** 67,550  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3824 |
| Coherence (Success) | 0.1486 |
| Coherence (Failure) | 0.1677 |
| Gradient Magnitude (Success) | 0.2260 |
| Gradient Magnitude (Failure) | 0.1758 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.3717 |
| Cosine Distance | 0.0155 |
| Clusters | 530 |
| Noise Fraction | 0.0937 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.066 | 0.022 | 0.168 |
| locked_door | 0.066 | 0.000 | 0.121 | 0.396 |
| open_door | 0.022 | 0.121 | -0.000 | 0.121 |
| target_ball | 0.168 | 0.396 | 0.121 | 0.000 |

---

## Checkpoint 140 — 70k (70020 episodes)

**Episodes:** 70,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4662 |
| Coherence (Success) | 0.2614 |
| Coherence (Failure) | 0.2257 |
| Gradient Magnitude (Success) | 0.2686 |
| Gradient Magnitude (Failure) | 0.1919 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.2902 |
| Cosine Distance | 0.0181 |
| Clusters | 551 |
| Noise Fraction | 0.0780 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.092 | 0.034 | 0.209 |
| locked_door | 0.092 | 0.000 | 0.195 | 0.510 |
| open_door | 0.034 | 0.195 | -0.000 | 0.111 |
| target_ball | 0.209 | 0.510 | 0.111 | 0.000 |

---

## Checkpoint 145 — 72k (72524 episodes)

**Episodes:** 72,524  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.127) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3546 |
| Coherence (Success) | 0.2387 |
| Coherence (Failure) | 0.1460 |
| Gradient Magnitude (Success) | 0.2816 |
| Gradient Magnitude (Failure) | 0.1437 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.6624 |
| Cosine Distance | 0.0251 |
| Clusters | 647 |
| Noise Fraction | 0.0913 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.056 | 0.025 | 0.185 |
| locked_door | 0.056 | -0.000 | 0.116 | 0.402 |
| open_door | 0.025 | 0.116 | 0.000 | 0.123 |
| target_ball | 0.185 | 0.402 | 0.123 | -0.000 |

---

## Checkpoint 150 — 75k (75028 episodes)

**Episodes:** 75,028  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3023 |
| Coherence (Success) | 0.2735 |
| Coherence (Failure) | 0.2038 |
| Gradient Magnitude (Success) | 0.2880 |
| Gradient Magnitude (Failure) | 0.2000 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.9492 |
| Cosine Distance | 0.0233 |
| Clusters | 630 |
| Noise Fraction | 0.0752 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.057 | 0.037 | 0.242 |
| locked_door | 0.057 | 0.000 | 0.150 | 0.479 |
| open_door | 0.037 | 0.150 | -0.000 | 0.129 |
| target_ball | 0.242 | 0.479 | 0.129 | 0.000 |

---

## Checkpoint 155 — 77k (77563 episodes)

**Episodes:** 77,563  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4584 |
| Coherence (Success) | 0.1532 |
| Coherence (Failure) | 0.0941 |
| Gradient Magnitude (Success) | 0.2368 |
| Gradient Magnitude (Failure) | 0.1503 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.0320 |
| Cosine Distance | 0.0223 |
| Clusters | 526 |
| Noise Fraction | 0.0982 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.060 | 0.026 | 0.193 |
| locked_door | 0.060 | 0.000 | 0.134 | 0.419 |
| open_door | 0.026 | 0.134 | -0.000 | 0.114 |
| target_ball | 0.193 | 0.419 | 0.114 | 0.000 |

---

## Checkpoint 160 — 80k (80007 episodes)

**Episodes:** 80,007  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3163 |
| Coherence (Success) | 0.1568 |
| Coherence (Failure) | 0.1173 |
| Gradient Magnitude (Success) | 0.2430 |
| Gradient Magnitude (Failure) | 0.1629 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.6442 |
| Cosine Distance | 0.0216 |
| Clusters | 501 |
| Noise Fraction | 0.0697 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.060 | 0.024 | 0.159 |
| locked_door | 0.060 | 0.000 | 0.123 | 0.368 |
| open_door | 0.024 | 0.123 | 0.000 | 0.113 |
| target_ball | 0.159 | 0.368 | 0.113 | 0.000 |

---

## Checkpoint 165 — 82k (82567 episodes)

**Episodes:** 82,567  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4196 |
| Coherence (Success) | 0.2036 |
| Coherence (Failure) | 0.1468 |
| Gradient Magnitude (Success) | 0.2338 |
| Gradient Magnitude (Failure) | 0.1713 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.3253 |
| Cosine Distance | 0.0248 |
| Clusters | 556 |
| Noise Fraction | 0.0783 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.073 | 0.020 | 0.199 |
| locked_door | 0.073 | 0.000 | 0.126 | 0.448 |
| open_door | 0.020 | 0.126 | 0.000 | 0.134 |
| target_ball | 0.199 | 0.448 | 0.134 | 0.000 |

---

## Checkpoint 170 — 85k (85007 episodes)

**Episodes:** 85,007  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4282 |
| Coherence (Success) | 0.1916 |
| Coherence (Failure) | 0.1509 |
| Gradient Magnitude (Success) | 0.2416 |
| Gradient Magnitude (Failure) | 0.1733 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.5423 |
| Cosine Distance | 0.0187 |
| Clusters | 569 |
| Noise Fraction | 0.0732 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.061 | 0.025 | 0.200 |
| locked_door | 0.061 | 0.000 | 0.120 | 0.423 |
| open_door | 0.025 | 0.120 | 0.000 | 0.126 |
| target_ball | 0.200 | 0.423 | 0.126 | 0.000 |

---

## Checkpoint 175 — 87k (87539 episodes)

**Episodes:** 87,539  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4010 |
| Coherence (Success) | 0.1637 |
| Coherence (Failure) | 0.1731 |
| Gradient Magnitude (Success) | 0.2290 |
| Gradient Magnitude (Failure) | 0.2092 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.2761 |
| Cosine Distance | 0.0185 |
| Clusters | 468 |
| Noise Fraction | 0.1021 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.071 | 0.025 | 0.188 |
| locked_door | 0.071 | 0.000 | 0.143 | 0.431 |
| open_door | 0.025 | 0.143 | 0.000 | 0.117 |
| target_ball | 0.188 | 0.431 | 0.117 | -0.000 |

---

## Checkpoint 180 — 90k (90057 episodes)

**Episodes:** 90,057  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1271 |
| Coherence (Success) | 0.1685 |
| Coherence (Failure) | 0.1280 |
| Gradient Magnitude (Success) | 0.2287 |
| Gradient Magnitude (Failure) | 0.1641 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.7864 |
| Cosine Distance | 0.0214 |
| Clusters | 572 |
| Noise Fraction | 0.0850 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.075 | 0.024 | 0.186 |
| locked_door | 0.075 | 0.000 | 0.140 | 0.433 |
| open_door | 0.024 | 0.140 | 0.000 | 0.109 |
| target_ball | 0.186 | 0.433 | 0.109 | 0.000 |

---

## Checkpoint 185 — 92k (92552 episodes)

**Episodes:** 92,552  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3095 |
| Coherence (Success) | 0.1672 |
| Coherence (Failure) | 0.1389 |
| Gradient Magnitude (Success) | 0.2082 |
| Gradient Magnitude (Failure) | 0.1601 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.7758 |
| Cosine Distance | 0.0174 |
| Clusters | 476 |
| Noise Fraction | 0.0733 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.058 | 0.023 | 0.141 |
| locked_door | 0.058 | 0.000 | 0.129 | 0.343 |
| open_door | 0.023 | 0.129 | 0.000 | 0.078 |
| target_ball | 0.141 | 0.343 | 0.078 | 0.000 |

---

## Checkpoint 190 — 95k (95017 episodes)

**Episodes:** 95,017  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.143) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4417 |
| Coherence (Success) | 0.1615 |
| Coherence (Failure) | 0.0921 |
| Gradient Magnitude (Success) | 0.2120 |
| Gradient Magnitude (Failure) | 0.1495 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.3657 |
| Cosine Distance | 0.0297 |
| Clusters | 613 |
| Noise Fraction | 0.0900 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.066 | 0.015 | 0.165 |
| locked_door | 0.066 | -0.000 | 0.108 | 0.392 |
| open_door | 0.015 | 0.108 | 0.000 | 0.113 |
| target_ball | 0.165 | 0.392 | 0.113 | 0.000 |

---

## Checkpoint 195 — 97k (97507 episodes)

**Episodes:** 97,507  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6653 |
| Coherence (Success) | 0.2140 |
| Coherence (Failure) | 0.1784 |
| Gradient Magnitude (Success) | 0.2689 |
| Gradient Magnitude (Failure) | 0.1566 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.6365 |
| Cosine Distance | 0.0245 |
| Clusters | 550 |
| Noise Fraction | 0.0889 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.067 | 0.020 | 0.143 |
| locked_door | 0.067 | 0.000 | 0.121 | 0.370 |
| open_door | 0.020 | 0.121 | 0.000 | 0.096 |
| target_ball | 0.143 | 0.370 | 0.096 | 0.000 |

---

## Checkpoint 200 — 100k (100015 episodes)

**Episodes:** 100,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3867 |
| Coherence (Success) | 0.2931 |
| Coherence (Failure) | 0.1210 |
| Gradient Magnitude (Success) | 0.2935 |
| Gradient Magnitude (Failure) | 0.1768 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.1582 |
| Cosine Distance | 0.0195 |
| Clusters | 498 |
| Noise Fraction | 0.0936 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.062 | 0.027 | 0.155 |
| locked_door | 0.062 | 0.000 | 0.123 | 0.362 |
| open_door | 0.027 | 0.123 | 0.000 | 0.088 |
| target_ball | 0.155 | 0.362 | 0.088 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2290 |
| Coherence (Success) | 0.2623 |
| Coherence (Failure) | 0.2080 |
| Gradient Magnitude (Success) | 0.2964 |
| Gradient Magnitude (Failure) | 0.1790 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.7720 |
| Cosine Distance | 0.0204 |
| Clusters | 486 |
| Noise Fraction | 0.0679 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.061 | 0.029 | 0.151 |
| locked_door | 0.061 | 0.000 | 0.133 | 0.359 |
| open_door | 0.029 | 0.133 | 0.000 | 0.082 |
| target_ball | 0.151 | 0.359 | 0.082 | 0.000 |
