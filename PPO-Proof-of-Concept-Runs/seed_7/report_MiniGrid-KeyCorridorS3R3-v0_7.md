# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 7  
**Total episodes:** 100,015  
**Experiment root:** `Proof-of-Concept-Runs\seed_7`  
**Generated:** 2026-02-26 01:17

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| found_key @ ep0 | 0 | 0.8182 | 0.2957 | 0.2053 | 0.0501 | 0.0357 | 0.0444 | 0.0028 | {'correlation': 0.0, 'p_value': 1.0} |
| reached_door @ ep0 | 0 | 0.8512 | 0.3984 | 0.2007 | 0.0534 | 0.0364 | 0.0553 | 0.0043 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5 — 2k (2504 episodes) | 2,504 | 0.7143 | 0.1585 | 0.2078 | 0.1193 | 0.0731 | 0.0880 | 0.1856 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 10 — 5k (5001 episodes) | 5,001 | 0.2118 | 0.4128 | 0.1111 | 0.4824 | 0.0795 | 0.6345 | 1.0242 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 15 — 7k (7515 episodes) | 7,515 | 0.0962 | 0.1390 | 0.0641 | 0.2574 | 0.0987 | 1.2596 | 0.5938 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 20 — 10k (10003 episodes) | 10,003 | 0.1028 | 0.1623 | 0.0534 | 0.3064 | 0.1421 | 1.7628 | 0.2959 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 25 — 12k (12538 episodes) | 12,538 | -0.0033 | 0.1395 | 0.1554 | 0.2794 | 0.1613 | 1.8789 | 0.1904 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 30 — 15k (15033 episodes) | 15,033 | 0.0091 | 0.3161 | 0.0906 | 0.4181 | 0.1522 | 2.2547 | 0.0826 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 35 — 17k (17545 episodes) | 17,545 | 0.0153 | 0.2592 | 0.2224 | 0.3578 | 0.2269 | 2.1721 | 0.0667 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 40 — 20k (20023 episodes) | 20,023 | 0.1557 | 0.3314 | 0.1338 | 0.3324 | 0.1510 | 2.2484 | 0.0437 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 45 — 22k (22554 episodes) | 22,554 | 0.0234 | 0.4005 | 0.1139 | 0.5143 | 0.1862 | 2.9205 | 0.0413 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 50 — 25k (25046 episodes) | 25,046 | 0.2657 | 0.1401 | 0.0948 | 0.2218 | 0.1934 | 2.9902 | 0.0404 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 55 — 27k (27512 episodes) | 27,512 | 0.3322 | 0.2550 | 0.2735 | 0.2598 | 0.2502 | 3.0072 | 0.0354 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 60 — 30k (30016 episodes) | 30,016 | 0.3905 | 0.2422 | 0.2527 | 0.3157 | 0.2432 | 3.5437 | 0.0176 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 65 — 32k (32537 episodes) | 32,537 | 0.6579 | 0.2274 | 0.1989 | 0.2544 | 0.1828 | 3.0637 | 0.0123 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 70 — 35k (35070 episodes) | 35,070 | 0.2702 | 0.1467 | 0.1213 | 0.2256 | 0.1921 | 3.8818 | 0.0126 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 75 — 37k (37516 episodes) | 37,516 | 0.2880 | 0.3224 | 0.3230 | 0.4175 | 0.2487 | 4.1918 | 0.0253 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 80 — 40k (40056 episodes) | 40,056 | 0.2846 | 0.2755 | 0.1668 | 0.2810 | 0.2258 | 4.1713 | 0.0172 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 85 — 42k (42553 episodes) | 42,553 | 0.2673 | 0.1234 | 0.1063 | 0.2077 | 0.1627 | 4.5531 | 0.0251 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 90 — 45k (45020 episodes) | 45,020 | 0.0332 | 0.2482 | 0.2041 | 0.2985 | 0.2093 | 4.5417 | 0.0193 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 95 — 47k (47511 episodes) | 47,511 | 0.4725 | 0.2435 | 0.1281 | 0.2641 | 0.1654 | 4.5655 | 0.0247 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 100 — 50k (50015 episodes) | 50,015 | 0.3849 | 0.1872 | 0.1621 | 0.2358 | 0.1739 | 4.7871 | 0.0201 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 105 — 52k (52530 episodes) | 52,530 | 0.2221 | 0.1404 | 0.1705 | 0.2297 | 0.2021 | 5.5435 | 0.0183 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 110 — 55k (55028 episodes) | 55,028 | 0.1698 | 0.1554 | 0.1528 | 0.2625 | 0.2036 | 5.5896 | 0.0188 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 115 — 57k (57581 episodes) | 57,581 | 0.1738 | 0.1027 | 0.1198 | 0.1956 | 0.1679 | 5.4882 | 0.0174 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 120 — 60k (60035 episodes) | 60,035 | 0.5549 | 0.1882 | 0.1492 | 0.2598 | 0.2009 | 6.7637 | 0.0141 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 125 — 62k (62564 episodes) | 62,564 | 0.3823 | 0.1519 | 0.1169 | 0.2414 | 0.1728 | 7.1829 | 0.0198 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 130 — 65k (65010 episodes) | 65,010 | 0.4507 | 0.1582 | 0.1818 | 0.2508 | 0.1634 | 7.3734 | 0.0234 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 135 — 67k (67530 episodes) | 67,530 | 0.4022 | 0.1669 | 0.1874 | 0.2234 | 0.2171 | 7.5130 | 0.0182 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 140 — 70k (70050 episodes) | 70,050 | 0.2808 | 0.1628 | 0.1744 | 0.2518 | 0.2091 | 7.5954 | 0.0230 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 145 — 72k (72548 episodes) | 72,548 | 0.2196 | 0.1350 | 0.1849 | 0.2025 | 0.2544 | 8.0532 | 0.0189 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 150 — 75k (75031 episodes) | 75,031 | 0.4914 | 0.2454 | 0.2442 | 0.2682 | 0.2144 | 8.1442 | 0.0255 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 155 — 77k (77564 episodes) | 77,564 | 0.4038 | 0.1642 | 0.2221 | 0.2620 | 0.2058 | 8.0833 | 0.0147 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 160 — 80k (80055 episodes) | 80,055 | 0.3347 | 0.1951 | 0.2017 | 0.2773 | 0.2030 | 8.9095 | 0.0180 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 165 — 82k (82516 episodes) | 82,516 | 0.3852 | 0.1870 | 0.1924 | 0.2539 | 0.2157 | 8.5398 | 0.0237 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 170 — 85k (85008 episodes) | 85,008 | 0.5845 | 0.2119 | 0.1413 | 0.2478 | 0.1744 | 9.4189 | 0.0190 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 175 — 87k (87514 episodes) | 87,514 | 0.3771 | 0.1444 | 0.1218 | 0.2466 | 0.1691 | 11.1217 | 0.0279 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 180 — 90k (90023 episodes) | 90,023 | 0.4784 | 0.1642 | 0.1497 | 0.2666 | 0.2027 | 10.3820 | 0.0208 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 185 — 92k (92574 episodes) | 92,574 | 0.4180 | 0.1352 | 0.0929 | 0.2096 | 0.1734 | 9.1857 | 0.0155 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 190 — 95k (95019 episodes) | 95,019 | 0.3810 | 0.0975 | 0.1775 | 0.1981 | 0.2176 | 9.5489 | 0.0157 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 195 — 97k (97552 episodes) | 97,552 | 0.3056 | 0.0797 | 0.1860 | 0.1942 | 0.1908 | 10.0293 | 0.0213 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 200 — 100k (100015 episodes) | 100,015 | 0.4836 | 0.1676 | 0.1469 | 0.2276 | 0.1867 | 9.9386 | 0.0292 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Final — 100k episodes | 100,015 | 0.4786 | 0.1240 | 0.1537 | 0.2180 | 0.1805 | 10.3047 | 0.0232 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

---

## found_key @ ep0

**Episodes:** 0  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8182 |
| Coherence (Success) | 0.2957 |
| Coherence (Failure) | 0.2053 |
| Gradient Magnitude (Success) | 0.0501 |
| Gradient Magnitude (Failure) | 0.0357 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0444 |
| Cosine Distance | 0.0028 |
| Clusters | 2,127 |
| Noise Fraction | 0.0271 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.015 | 0.005 | 0.008 |
| locked_door | 0.015 | 0.000 | 0.020 | 0.036 |
| open_door | 0.005 | 0.020 | 0.000 | 0.020 |
| target_ball | 0.008 | 0.036 | 0.020 | 0.000 |

---

## reached_door @ ep0

**Episodes:** 0  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8512 |
| Coherence (Success) | 0.3984 |
| Coherence (Failure) | 0.2007 |
| Gradient Magnitude (Success) | 0.0534 |
| Gradient Magnitude (Failure) | 0.0364 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0553 |
| Cosine Distance | 0.0043 |
| Clusters | 2,128 |
| Noise Fraction | 0.0280 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.015 | 0.004 | 0.020 |
| locked_door | 0.015 | 0.000 | 0.016 | 0.055 |
| open_door | 0.004 | 0.016 | 0.000 | 0.026 |
| target_ball | 0.020 | 0.055 | 0.026 | 0.000 |

---

## Checkpoint 5 — 2k (2504 episodes)

**Episodes:** 2,504  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7143 |
| Coherence (Success) | 0.1585 |
| Coherence (Failure) | 0.2078 |
| Gradient Magnitude (Success) | 0.1193 |
| Gradient Magnitude (Failure) | 0.0731 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0880 |
| Cosine Distance | 0.1856 |
| Clusters | 1,766 |
| Noise Fraction | 0.0299 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.425 | 0.192 | 0.492 |
| locked_door | 0.425 | 0.000 | 0.494 | 1.013 |
| open_door | 0.192 | 0.494 | 0.000 | 0.703 |
| target_ball | 0.492 | 1.013 | 0.703 | -0.000 |

---

## Checkpoint 10 — 5k (5001 episodes)

**Episodes:** 5,001  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 1.937)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2118 |
| Coherence (Success) | 0.4128 |
| Coherence (Failure) | 0.1111 |
| Gradient Magnitude (Success) | 0.4824 |
| Gradient Magnitude (Failure) | 0.0795 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.6345 |
| Cosine Distance | 1.0242 |
| Clusters | 1,213 |
| Noise Fraction | 0.0327 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.428 | 0.438 | 0.310 |
| locked_door | 0.428 | 0.000 | 0.362 | 0.954 |
| open_door | 0.438 | 0.362 | 0.000 | 0.738 |
| target_ball | 0.310 | 0.954 | 0.738 | 0.000 |

---

## Checkpoint 15 — 7k (7515 episodes)

**Episodes:** 7,515  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 1.703) | top 25% (≥ 2.210)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0962 |
| Coherence (Success) | 0.1390 |
| Coherence (Failure) | 0.0641 |
| Gradient Magnitude (Success) | 0.2574 |
| Gradient Magnitude (Failure) | 0.0987 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.2596 |
| Cosine Distance | 0.5938 |
| Clusters | 1,229 |
| Noise Fraction | 0.0408 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.304 | 0.130 | 0.121 |
| locked_door | 0.304 | 0.000 | 0.380 | 0.691 |
| open_door | 0.130 | 0.380 | 0.000 | 0.249 |
| target_ball | 0.121 | 0.691 | 0.249 | 0.000 |

---

## Checkpoint 20 — 10k (10003 episodes)

**Episodes:** 10,003  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.010) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1028 |
| Coherence (Success) | 0.1623 |
| Coherence (Failure) | 0.0534 |
| Gradient Magnitude (Success) | 0.3064 |
| Gradient Magnitude (Failure) | 0.1421 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7628 |
| Cosine Distance | 0.2959 |
| Clusters | 962 |
| Noise Fraction | 0.0444 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.142 | 0.087 | 0.110 |
| locked_door | 0.142 | 0.000 | 0.222 | 0.430 |
| open_door | 0.087 | 0.222 | -0.000 | 0.179 |
| target_ball | 0.110 | 0.430 | 0.179 | 0.000 |

---

## Checkpoint 25 — 12k (12538 episodes)

**Episodes:** 12,538  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.077) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0033 |
| Coherence (Success) | 0.1395 |
| Coherence (Failure) | 0.1554 |
| Gradient Magnitude (Success) | 0.2794 |
| Gradient Magnitude (Failure) | 0.1613 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.8789 |
| Cosine Distance | 0.1904 |
| Clusters | 842 |
| Noise Fraction | 0.0514 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.158 | 0.060 | 0.137 |
| locked_door | 0.158 | 0.000 | 0.265 | 0.516 |
| open_door | 0.060 | 0.265 | 0.000 | 0.121 |
| target_ball | 0.137 | 0.516 | 0.121 | 0.000 |

---

## Checkpoint 30 — 15k (15033 episodes)

**Episodes:** 15,033  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.117) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0091 |
| Coherence (Success) | 0.3161 |
| Coherence (Failure) | 0.0906 |
| Gradient Magnitude (Success) | 0.4181 |
| Gradient Magnitude (Failure) | 0.1522 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.2547 |
| Cosine Distance | 0.0826 |
| Clusters | 707 |
| Noise Fraction | 0.0690 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.099 | 0.051 | 0.121 |
| locked_door | 0.099 | 0.000 | 0.222 | 0.394 |
| open_door | 0.051 | 0.222 | 0.000 | 0.092 |
| target_ball | 0.121 | 0.394 | 0.092 | -0.000 |

---

## Checkpoint 35 — 17k (17545 episodes)

**Episodes:** 17,545  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0153 |
| Coherence (Success) | 0.2592 |
| Coherence (Failure) | 0.2224 |
| Gradient Magnitude (Success) | 0.3578 |
| Gradient Magnitude (Failure) | 0.2269 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.1721 |
| Cosine Distance | 0.0667 |
| Clusters | 661 |
| Noise Fraction | 0.0816 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.089 | 0.061 | 0.149 |
| locked_door | 0.089 | 0.000 | 0.210 | 0.405 |
| open_door | 0.061 | 0.210 | -0.000 | 0.114 |
| target_ball | 0.149 | 0.405 | 0.114 | -0.000 |

---

## Checkpoint 40 — 20k (20023 episodes)

**Episodes:** 20,023  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.110) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1557 |
| Coherence (Success) | 0.3314 |
| Coherence (Failure) | 0.1338 |
| Gradient Magnitude (Success) | 0.3324 |
| Gradient Magnitude (Failure) | 0.1510 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.2484 |
| Cosine Distance | 0.0437 |
| Clusters | 722 |
| Noise Fraction | 0.0822 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.061 | 0.039 | 0.123 |
| locked_door | 0.061 | 0.000 | 0.132 | 0.300 |
| open_door | 0.039 | 0.132 | 0.000 | 0.107 |
| target_ball | 0.123 | 0.300 | 0.107 | 0.000 |

---

## Checkpoint 45 — 22k (22554 episodes)

**Episodes:** 22,554  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0234 |
| Coherence (Success) | 0.4005 |
| Coherence (Failure) | 0.1139 |
| Gradient Magnitude (Success) | 0.5143 |
| Gradient Magnitude (Failure) | 0.1862 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9205 |
| Cosine Distance | 0.0413 |
| Clusters | 530 |
| Noise Fraction | 0.0795 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.091 | 0.043 | 0.115 |
| locked_door | 0.091 | 0.000 | 0.184 | 0.370 |
| open_door | 0.043 | 0.184 | 0.000 | 0.098 |
| target_ball | 0.115 | 0.370 | 0.098 | -0.000 |

---

## Checkpoint 50 — 25k (25046 episodes)

**Episodes:** 25,046  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.113) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2657 |
| Coherence (Success) | 0.1401 |
| Coherence (Failure) | 0.0948 |
| Gradient Magnitude (Success) | 0.2218 |
| Gradient Magnitude (Failure) | 0.1934 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.9902 |
| Cosine Distance | 0.0404 |
| Clusters | 673 |
| Noise Fraction | 0.0757 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.114 | 0.037 | 0.094 |
| locked_door | 0.114 | 0.000 | 0.193 | 0.358 |
| open_door | 0.037 | 0.193 | 0.000 | 0.084 |
| target_ball | 0.094 | 0.358 | 0.084 | 0.000 |

---

## Checkpoint 55 — 27k (27512 episodes)

**Episodes:** 27,512  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3322 |
| Coherence (Success) | 0.2550 |
| Coherence (Failure) | 0.2735 |
| Gradient Magnitude (Success) | 0.2598 |
| Gradient Magnitude (Failure) | 0.2502 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.0072 |
| Cosine Distance | 0.0354 |
| Clusters | 598 |
| Noise Fraction | 0.0797 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.121 | 0.036 | 0.098 |
| locked_door | 0.121 | -0.000 | 0.206 | 0.395 |
| open_door | 0.036 | 0.206 | 0.000 | 0.075 |
| target_ball | 0.098 | 0.395 | 0.075 | 0.000 |

---

## Checkpoint 60 — 30k (30016 episodes)

**Episodes:** 30,016  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3905 |
| Coherence (Success) | 0.2422 |
| Coherence (Failure) | 0.2527 |
| Gradient Magnitude (Success) | 0.3157 |
| Gradient Magnitude (Failure) | 0.2432 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.5437 |
| Cosine Distance | 0.0176 |
| Clusters | 561 |
| Noise Fraction | 0.0873 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.082 | 0.027 | 0.088 |
| locked_door | 0.082 | 0.000 | 0.176 | 0.316 |
| open_door | 0.027 | 0.176 | 0.000 | 0.048 |
| target_ball | 0.088 | 0.316 | 0.048 | 0.000 |

---

## Checkpoint 65 — 32k (32537 episodes)

**Episodes:** 32,537  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6579 |
| Coherence (Success) | 0.2274 |
| Coherence (Failure) | 0.1989 |
| Gradient Magnitude (Success) | 0.2544 |
| Gradient Magnitude (Failure) | 0.1828 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.0637 |
| Cosine Distance | 0.0123 |
| Clusters | 560 |
| Noise Fraction | 0.0938 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.064 | 0.031 | 0.143 |
| locked_door | 0.064 | 0.000 | 0.157 | 0.366 |
| open_door | 0.031 | 0.157 | 0.000 | 0.088 |
| target_ball | 0.143 | 0.366 | 0.088 | 0.000 |

---

## Checkpoint 70 — 35k (35070 episodes)

**Episodes:** 35,070  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2702 |
| Coherence (Success) | 0.1467 |
| Coherence (Failure) | 0.1213 |
| Gradient Magnitude (Success) | 0.2256 |
| Gradient Magnitude (Failure) | 0.1921 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 3.8818 |
| Cosine Distance | 0.0126 |
| Clusters | 553 |
| Noise Fraction | 0.0798 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.061 | 0.022 | 0.129 |
| locked_door | 0.061 | 0.000 | 0.132 | 0.341 |
| open_door | 0.022 | 0.132 | 0.000 | 0.082 |
| target_ball | 0.129 | 0.341 | 0.082 | 0.000 |

---

## Checkpoint 75 — 37k (37516 episodes)

**Episodes:** 37,516  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2880 |
| Coherence (Success) | 0.3224 |
| Coherence (Failure) | 0.3230 |
| Gradient Magnitude (Success) | 0.4175 |
| Gradient Magnitude (Failure) | 0.2487 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.1918 |
| Cosine Distance | 0.0253 |
| Clusters | 535 |
| Noise Fraction | 0.0957 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.077 | 0.026 | 0.121 |
| locked_door | 0.077 | 0.000 | 0.153 | 0.354 |
| open_door | 0.026 | 0.153 | -0.000 | 0.091 |
| target_ball | 0.121 | 0.354 | 0.091 | 0.000 |

---

## Checkpoint 80 — 40k (40056 episodes)

**Episodes:** 40,056  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2846 |
| Coherence (Success) | 0.2755 |
| Coherence (Failure) | 0.1668 |
| Gradient Magnitude (Success) | 0.2810 |
| Gradient Magnitude (Failure) | 0.2258 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.1713 |
| Cosine Distance | 0.0172 |
| Clusters | 569 |
| Noise Fraction | 0.0956 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.093 | 0.020 | 0.101 |
| locked_door | 0.093 | -0.000 | 0.168 | 0.350 |
| open_door | 0.020 | 0.168 | -0.000 | 0.066 |
| target_ball | 0.101 | 0.350 | 0.066 | 0.000 |

---

## Checkpoint 85 — 42k (42553 episodes)

**Episodes:** 42,553  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.147) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2673 |
| Coherence (Success) | 0.1234 |
| Coherence (Failure) | 0.1063 |
| Gradient Magnitude (Success) | 0.2077 |
| Gradient Magnitude (Failure) | 0.1627 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.5531 |
| Cosine Distance | 0.0251 |
| Clusters | 567 |
| Noise Fraction | 0.0770 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.075 | 0.017 | 0.095 |
| locked_door | 0.075 | 0.000 | 0.125 | 0.306 |
| open_door | 0.017 | 0.125 | 0.000 | 0.077 |
| target_ball | 0.095 | 0.306 | 0.077 | 0.000 |

---

## Checkpoint 90 — 45k (45020 episodes)

**Episodes:** 45,020  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0332 |
| Coherence (Success) | 0.2482 |
| Coherence (Failure) | 0.2041 |
| Gradient Magnitude (Success) | 0.2985 |
| Gradient Magnitude (Failure) | 0.2093 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.5417 |
| Cosine Distance | 0.0193 |
| Clusters | 488 |
| Noise Fraction | 0.0802 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.078 | 0.019 | 0.126 |
| locked_door | 0.078 | 0.000 | 0.145 | 0.364 |
| open_door | 0.019 | 0.145 | 0.000 | 0.079 |
| target_ball | 0.126 | 0.364 | 0.079 | 0.000 |

---

## Checkpoint 95 — 47k (47511 episodes)

**Episodes:** 47,511  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.137) | top 25% (≥ 2.257)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4725 |
| Coherence (Success) | 0.2435 |
| Coherence (Failure) | 0.1281 |
| Gradient Magnitude (Success) | 0.2641 |
| Gradient Magnitude (Failure) | 0.1654 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.5655 |
| Cosine Distance | 0.0247 |
| Clusters | 594 |
| Noise Fraction | 0.0817 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.056 | 0.023 | 0.094 |
| locked_door | 0.056 | -0.000 | 0.115 | 0.269 |
| open_door | 0.023 | 0.115 | 0.000 | 0.074 |
| target_ball | 0.094 | 0.269 | 0.074 | 0.000 |

---

## Checkpoint 100 — 50k (50015 episodes)

**Episodes:** 50,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.163) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3849 |
| Coherence (Success) | 0.1872 |
| Coherence (Failure) | 0.1621 |
| Gradient Magnitude (Success) | 0.2358 |
| Gradient Magnitude (Failure) | 0.1739 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 4.7871 |
| Cosine Distance | 0.0201 |
| Clusters | 527 |
| Noise Fraction | 0.0827 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.056 | 0.017 | 0.090 |
| locked_door | 0.056 | 0.000 | 0.106 | 0.258 |
| open_door | 0.017 | 0.106 | 0.000 | 0.074 |
| target_ball | 0.090 | 0.258 | 0.074 | 0.000 |

---

## Checkpoint 105 — 52k (52530 episodes)

**Episodes:** 52,530  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.183) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2221 |
| Coherence (Success) | 0.1404 |
| Coherence (Failure) | 0.1705 |
| Gradient Magnitude (Success) | 0.2297 |
| Gradient Magnitude (Failure) | 0.2021 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.5435 |
| Cosine Distance | 0.0183 |
| Clusters | 490 |
| Noise Fraction | 0.0911 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.062 | 0.016 | 0.083 |
| locked_door | 0.062 | 0.000 | 0.116 | 0.262 |
| open_door | 0.016 | 0.116 | -0.000 | 0.056 |
| target_ball | 0.083 | 0.262 | 0.056 | 0.000 |

---

## Checkpoint 110 — 55k (55028 episodes)

**Episodes:** 55,028  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1698 |
| Coherence (Success) | 0.1554 |
| Coherence (Failure) | 0.1528 |
| Gradient Magnitude (Success) | 0.2625 |
| Gradient Magnitude (Failure) | 0.2036 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.5896 |
| Cosine Distance | 0.0188 |
| Clusters | 505 |
| Noise Fraction | 0.0917 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.060 | 0.013 | 0.069 |
| locked_door | 0.060 | 0.000 | 0.104 | 0.225 |
| open_door | 0.013 | 0.104 | 0.000 | 0.053 |
| target_ball | 0.069 | 0.225 | 0.053 | 0.000 |

---

## Checkpoint 115 — 57k (57581 episodes)

**Episodes:** 57,581  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1738 |
| Coherence (Success) | 0.1027 |
| Coherence (Failure) | 0.1198 |
| Gradient Magnitude (Success) | 0.1956 |
| Gradient Magnitude (Failure) | 0.1679 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 5.4882 |
| Cosine Distance | 0.0174 |
| Clusters | 456 |
| Noise Fraction | 0.0867 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.068 | 0.021 | 0.098 |
| locked_door | 0.068 | 0.000 | 0.133 | 0.300 |
| open_door | 0.021 | 0.133 | 0.000 | 0.065 |
| target_ball | 0.098 | 0.300 | 0.065 | 0.000 |

---

## Checkpoint 120 — 60k (60035 episodes)

**Episodes:** 60,035  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5549 |
| Coherence (Success) | 0.1882 |
| Coherence (Failure) | 0.1492 |
| Gradient Magnitude (Success) | 0.2598 |
| Gradient Magnitude (Failure) | 0.2009 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 6.7637 |
| Cosine Distance | 0.0141 |
| Clusters | 532 |
| Noise Fraction | 0.0956 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.061 | 0.014 | 0.073 |
| locked_door | 0.061 | 0.000 | 0.106 | 0.240 |
| open_door | 0.014 | 0.106 | 0.000 | 0.048 |
| target_ball | 0.073 | 0.240 | 0.048 | -0.000 |

---

## Checkpoint 125 — 62k (62564 episodes)

**Episodes:** 62,564  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.290)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3823 |
| Coherence (Success) | 0.1519 |
| Coherence (Failure) | 0.1169 |
| Gradient Magnitude (Success) | 0.2414 |
| Gradient Magnitude (Failure) | 0.1728 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.1829 |
| Cosine Distance | 0.0198 |
| Clusters | 474 |
| Noise Fraction | 0.0847 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.060 | 0.017 | 0.098 |
| locked_door | 0.060 | 0.000 | 0.115 | 0.278 |
| open_door | 0.017 | 0.115 | 0.000 | 0.059 |
| target_ball | 0.098 | 0.278 | 0.059 | -0.000 |

---

## Checkpoint 130 — 65k (65010 episodes)

**Episodes:** 65,010  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.110) | top 25% (≥ 2.260)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4507 |
| Coherence (Success) | 0.1582 |
| Coherence (Failure) | 0.1818 |
| Gradient Magnitude (Success) | 0.2508 |
| Gradient Magnitude (Failure) | 0.1634 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.3734 |
| Cosine Distance | 0.0234 |
| Clusters | 676 |
| Noise Fraction | 0.0829 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.041 | 0.014 | 0.082 |
| locked_door | 0.041 | -0.000 | 0.072 | 0.216 |
| open_door | 0.014 | 0.072 | 0.000 | 0.065 |
| target_ball | 0.082 | 0.216 | 0.065 | 0.000 |

---

## Checkpoint 135 — 67k (67530 episodes)

**Episodes:** 67,530  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.150) | top 25% (≥ 2.263)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4022 |
| Coherence (Success) | 0.1669 |
| Coherence (Failure) | 0.1874 |
| Gradient Magnitude (Success) | 0.2234 |
| Gradient Magnitude (Failure) | 0.2171 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.5130 |
| Cosine Distance | 0.0182 |
| Clusters | 550 |
| Noise Fraction | 0.0918 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.031 | 0.011 | 0.059 |
| locked_door | 0.031 | 0.000 | 0.061 | 0.157 |
| open_door | 0.011 | 0.061 | 0.000 | 0.041 |
| target_ball | 0.059 | 0.157 | 0.041 | 0.000 |

---

## Checkpoint 140 — 70k (70050 episodes)

**Episodes:** 70,050  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.160) | top 25% (≥ 2.277)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2808 |
| Coherence (Success) | 0.1628 |
| Coherence (Failure) | 0.1744 |
| Gradient Magnitude (Success) | 0.2518 |
| Gradient Magnitude (Failure) | 0.2091 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 7.5954 |
| Cosine Distance | 0.0230 |
| Clusters | 524 |
| Noise Fraction | 0.0734 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.035 | 0.014 | 0.071 |
| locked_door | 0.035 | 0.000 | 0.073 | 0.183 |
| open_door | 0.014 | 0.073 | 0.000 | 0.047 |
| target_ball | 0.071 | 0.183 | 0.047 | 0.000 |

---

## Checkpoint 145 — 72k (72548 episodes)

**Episodes:** 72,548  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.180) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2196 |
| Coherence (Success) | 0.1350 |
| Coherence (Failure) | 0.1849 |
| Gradient Magnitude (Success) | 0.2025 |
| Gradient Magnitude (Failure) | 0.2544 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.0532 |
| Cosine Distance | 0.0189 |
| Clusters | 489 |
| Noise Fraction | 0.0844 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.044 | 0.012 | 0.052 |
| locked_door | 0.044 | 0.000 | 0.080 | 0.171 |
| open_door | 0.012 | 0.080 | 0.000 | 0.036 |
| target_ball | 0.052 | 0.171 | 0.036 | -0.000 |

---

## Checkpoint 150 — 75k (75031 episodes)

**Episodes:** 75,031  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.140) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4914 |
| Coherence (Success) | 0.2454 |
| Coherence (Failure) | 0.2442 |
| Gradient Magnitude (Success) | 0.2682 |
| Gradient Magnitude (Failure) | 0.2144 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.1442 |
| Cosine Distance | 0.0255 |
| Clusters | 586 |
| Noise Fraction | 0.0844 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.046 | 0.015 | 0.072 |
| locked_door | 0.046 | 0.000 | 0.082 | 0.212 |
| open_door | 0.015 | 0.082 | 0.000 | 0.051 |
| target_ball | 0.072 | 0.212 | 0.051 | 0.000 |

---

## Checkpoint 155 — 77k (77564 episodes)

**Episodes:** 77,564  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.167) | top 25% (≥ 2.267)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4038 |
| Coherence (Success) | 0.1642 |
| Coherence (Failure) | 0.2221 |
| Gradient Magnitude (Success) | 0.2620 |
| Gradient Magnitude (Failure) | 0.2058 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.0833 |
| Cosine Distance | 0.0147 |
| Clusters | 511 |
| Noise Fraction | 0.0943 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.042 | 0.019 | 0.066 |
| locked_door | 0.042 | 0.000 | 0.095 | 0.198 |
| open_door | 0.019 | 0.095 | 0.000 | 0.040 |
| target_ball | 0.066 | 0.198 | 0.040 | 0.000 |

---

## Checkpoint 160 — 80k (80055 episodes)

**Episodes:** 80,055  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3347 |
| Coherence (Success) | 0.1951 |
| Coherence (Failure) | 0.2017 |
| Gradient Magnitude (Success) | 0.2773 |
| Gradient Magnitude (Failure) | 0.2030 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.9095 |
| Cosine Distance | 0.0180 |
| Clusters | 493 |
| Noise Fraction | 0.0989 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.047 | 0.015 | 0.051 |
| locked_door | 0.047 | 0.000 | 0.088 | 0.177 |
| open_door | 0.015 | 0.088 | 0.000 | 0.031 |
| target_ball | 0.051 | 0.177 | 0.031 | 0.000 |

---

## Checkpoint 165 — 82k (82516 episodes)

**Episodes:** 82,516  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3852 |
| Coherence (Success) | 0.1870 |
| Coherence (Failure) | 0.1924 |
| Gradient Magnitude (Success) | 0.2539 |
| Gradient Magnitude (Failure) | 0.2157 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 8.5398 |
| Cosine Distance | 0.0237 |
| Clusters | 502 |
| Noise Fraction | 0.0926 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.047 | 0.017 | 0.093 |
| locked_door | 0.047 | 0.000 | 0.095 | 0.255 |
| open_door | 0.017 | 0.095 | 0.000 | 0.059 |
| target_ball | 0.093 | 0.255 | 0.059 | 0.000 |

---

## Checkpoint 170 — 85k (85008 episodes)

**Episodes:** 85,008  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5845 |
| Coherence (Success) | 0.2119 |
| Coherence (Failure) | 0.1413 |
| Gradient Magnitude (Success) | 0.2478 |
| Gradient Magnitude (Failure) | 0.1744 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.4189 |
| Cosine Distance | 0.0190 |
| Clusters | 494 |
| Noise Fraction | 0.0976 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.045 | 0.016 | 0.082 |
| locked_door | 0.045 | 0.000 | 0.090 | 0.227 |
| open_door | 0.016 | 0.090 | 0.000 | 0.052 |
| target_ball | 0.082 | 0.227 | 0.052 | 0.000 |

---

## Checkpoint 175 — 87k (87514 episodes)

**Episodes:** 87,514  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.153) | top 25% (≥ 2.270)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3771 |
| Coherence (Success) | 0.1444 |
| Coherence (Failure) | 0.1218 |
| Gradient Magnitude (Success) | 0.2466 |
| Gradient Magnitude (Failure) | 0.1691 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 11.1217 |
| Cosine Distance | 0.0279 |
| Clusters | 537 |
| Noise Fraction | 0.0911 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.053 | 0.015 | 0.061 |
| locked_door | 0.053 | 0.000 | 0.092 | 0.208 |
| open_door | 0.015 | 0.092 | 0.000 | 0.043 |
| target_ball | 0.061 | 0.208 | 0.043 | -0.000 |

---

## Checkpoint 180 — 90k (90023 episodes)

**Episodes:** 90,023  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.157) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4784 |
| Coherence (Success) | 0.1642 |
| Coherence (Failure) | 0.1497 |
| Gradient Magnitude (Success) | 0.2666 |
| Gradient Magnitude (Failure) | 0.2027 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.3820 |
| Cosine Distance | 0.0208 |
| Clusters | 505 |
| Noise Fraction | 0.0919 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.032 | 0.015 | 0.083 |
| locked_door | 0.032 | -0.000 | 0.065 | 0.198 |
| open_door | 0.015 | 0.065 | 0.000 | 0.062 |
| target_ball | 0.083 | 0.198 | 0.062 | 0.000 |

---

## Checkpoint 185 — 92k (92574 episodes)

**Episodes:** 92,574  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.187) | top 25% (≥ 2.283)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4180 |
| Coherence (Success) | 0.1352 |
| Coherence (Failure) | 0.0929 |
| Gradient Magnitude (Success) | 0.2096 |
| Gradient Magnitude (Failure) | 0.1734 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.1857 |
| Cosine Distance | 0.0155 |
| Clusters | 437 |
| Noise Fraction | 0.0828 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.048 | 0.015 | 0.067 |
| locked_door | 0.048 | -0.000 | 0.090 | 0.206 |
| open_door | 0.015 | 0.090 | 0.000 | 0.047 |
| target_ball | 0.067 | 0.206 | 0.047 | 0.000 |

---

## Checkpoint 190 — 95k (95019 episodes)

**Episodes:** 95,019  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.197) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3810 |
| Coherence (Success) | 0.0975 |
| Coherence (Failure) | 0.1775 |
| Gradient Magnitude (Success) | 0.1981 |
| Gradient Magnitude (Failure) | 0.2176 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.5489 |
| Cosine Distance | 0.0157 |
| Clusters | 429 |
| Noise Fraction | 0.1021 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.036 | 0.014 | 0.063 |
| locked_door | 0.036 | -0.000 | 0.072 | 0.178 |
| open_door | 0.014 | 0.072 | -0.000 | 0.042 |
| target_ball | 0.063 | 0.178 | 0.042 | -0.000 |

---

## Checkpoint 195 — 97k (97552 episodes)

**Episodes:** 97,552  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.177) | top 25% (≥ 2.273)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3056 |
| Coherence (Success) | 0.0797 |
| Coherence (Failure) | 0.1860 |
| Gradient Magnitude (Success) | 0.1942 |
| Gradient Magnitude (Failure) | 0.1908 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.0293 |
| Cosine Distance | 0.0213 |
| Clusters | 501 |
| Noise Fraction | 0.0840 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.040 | 0.015 | 0.072 |
| locked_door | 0.040 | -0.000 | 0.080 | 0.202 |
| open_door | 0.015 | 0.080 | 0.000 | 0.053 |
| target_ball | 0.072 | 0.202 | 0.053 | 0.000 |

---

## Checkpoint 200 — 100k (100015 episodes)

**Episodes:** 100,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.173) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4836 |
| Coherence (Success) | 0.1676 |
| Coherence (Failure) | 0.1469 |
| Gradient Magnitude (Success) | 0.2276 |
| Gradient Magnitude (Failure) | 0.1867 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 9.9386 |
| Cosine Distance | 0.0292 |
| Clusters | 490 |
| Noise Fraction | 0.0997 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.052 | 0.014 | 0.071 |
| locked_door | 0.052 | -0.000 | 0.082 | 0.224 |
| open_door | 0.014 | 0.082 | -0.000 | 0.059 |
| target_ball | 0.071 | 0.224 | 0.059 | 0.000 |

---

## Final — 100k episodes

**Episodes:** 100,015  
**Success:** 125  
**Failure:** 125  
**Threshold:** bottom 25% (≤ 2.170) | top 25% (≥ 2.280)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4786 |
| Coherence (Success) | 0.1240 |
| Coherence (Failure) | 0.1537 |
| Gradient Magnitude (Success) | 0.2180 |
| Gradient Magnitude (Failure) | 0.1805 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 10.3047 |
| Cosine Distance | 0.0232 |
| Clusters | 491 |
| Noise Fraction | 0.0894 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.054 | 0.014 | 0.071 |
| locked_door | 0.054 | 0.000 | 0.087 | 0.229 |
| open_door | 0.014 | 0.087 | 0.000 | 0.056 |
| target_ball | 0.071 | 0.229 | 0.056 | -0.000 |
