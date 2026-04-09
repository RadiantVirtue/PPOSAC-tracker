# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 1  
**Total episodes:** 13,079  
**Experiment root:** `experiment_root\seed_1`  
**Generated:** 2026-04-03 00:38

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_drink @ ep131 | 131 | 0.9873 | 0.8195 | 0.9250 | 0.0472 | 0.0597 | 0.1025 | 0.0002 | — |
| collect_sapling @ ep140 | 140 | -0.8231 | 0.2769 | 0.8930 | 0.0598 | 0.0654 | 0.8529 | 0.0134 | — |
| collect_wood @ ep149 | 149 | -0.7830 | 0.9561 | 0.9694 | 0.1017 | 0.1305 | 1.3189 | 0.0194 | — |
| defeat_skeleton @ ep156 | 156 | -0.7362 | 0.9971 | 0.9504 | 0.4813 | 0.1334 | 1.6246 | 0.0292 | — |
| defeat_zombie @ ep157 | 157 | -0.7483 | 0.9950 | 0.9366 | 0.2953 | 0.0918 | 1.4012 | 0.0242 | — |
| eat_cow @ ep157 | 157 | -0.7606 | 0.9952 | 0.8990 | 0.2964 | 0.0915 | 1.4270 | 0.0251 | — |
| place_plant @ ep178 | 178 | -0.7809 | 0.8306 | 0.9698 | 0.0711 | 0.1735 | 1.3458 | 0.0295 | — |
| checkpoint_step50000 | 288 | 0.9873 | 0.9927 | 0.9755 | 0.3753 | 0.4091 | 0.5322 | 0.0012 | — |
| checkpoint_step100000 | 564 | 0.8526 | 0.9838 | 0.9819 | 0.3348 | 0.2109 | 0.3569 | 0.0012 | — |
| checkpoint_step150000 | 830 | 0.9625 | 0.9706 | 0.9680 | 0.3161 | 0.3028 | 0.5415 | 0.0027 | — |
| checkpoint_step200000 | 1,098 | 0.8097 | 0.8755 | 0.9256 | 0.1289 | 0.1848 | 0.4363 | 0.0025 | — |
| checkpoint_step250000 | 1,366 | 0.6837 | 0.8840 | 0.9399 | 0.1893 | 0.2090 | 0.5740 | 0.0055 | — |
| checkpoint_step300000 | 1,626 | 0.5445 | 0.9136 | 0.9347 | 0.1571 | 0.1887 | 0.5531 | 0.0062 | — |
| checkpoint_step350000 | 1,888 | 0.8381 | 0.9805 | 0.9786 | 0.4954 | 0.3245 | 0.6854 | 0.0089 | — |
| place_table @ ep1890 | 1,890 | 0.5772 | 0.9597 | 0.8804 | 0.2934 | 0.2072 | 0.6121 | 0.0074 | — |
| checkpoint_step400000 | 2,153 | 0.6783 | 0.9382 | 0.8807 | 0.2731 | 0.1980 | 0.5787 | 0.0077 | — |
| make_wood_sword @ ep2157 | 2,157 | 0.3961 | 0.8725 | 0.8883 | 0.2305 | 0.2723 | 0.7132 | 0.0120 | — |
| checkpoint_step450000 | 2,406 | 0.4701 | 0.9626 | 0.9154 | 0.3546 | 0.2832 | 0.7192 | 0.0142 | — |
| make_wood_pickaxe @ ep2420 | 2,420 | 0.6115 | 0.9401 | 0.9243 | 0.3368 | 0.2935 | 0.7481 | 0.0157 | — |
| wake_up @ ep2589 | 2,589 | 0.7443 | 0.9581 | 0.9151 | 0.4170 | 0.3056 | 0.9647 | 0.0264 | — |
| checkpoint_step500000 | 2,665 | 0.3715 | 0.9106 | 0.9112 | 0.2315 | 0.3020 | 0.9573 | 0.0263 | — |
| checkpoint_step550000 | 2,917 | 0.5210 | 0.9230 | 0.8824 | 0.3178 | 0.2266 | 1.1419 | 0.0400 | — |
| checkpoint_step600000 | 3,166 | 0.8255 | 0.9426 | 0.8644 | 0.3804 | 0.2363 | 0.6549 | 0.0133 | — |
| checkpoint_step650000 | 3,410 | 0.8550 | 0.9541 | 0.9117 | 0.4337 | 0.3978 | 0.4519 | 0.0067 | — |
| checkpoint_step700000 | 3,654 | 0.8512 | 0.9622 | 0.8795 | 0.3644 | 0.3165 | 0.4631 | 0.0078 | — |
| checkpoint_step750000 | 3,887 | 0.8226 | 0.8894 | 0.7985 | 0.2832 | 0.2183 | 0.4276 | 0.0067 | — |
| checkpoint_step800000 | 4,117 | 0.9124 | 0.9403 | 0.8603 | 0.3546 | 0.2898 | 0.4067 | 0.0067 | — |
| checkpoint_step850000 | 4,348 | 0.8607 | 0.9067 | 0.8958 | 0.2838 | 0.2901 | 0.3126 | 0.0046 | — |
| checkpoint_step900000 | 4,578 | 0.9072 | 0.9518 | 0.8238 | 0.3623 | 0.2512 | 0.3016 | 0.0039 | — |
| checkpoint_step950000 | 4,799 | 0.9221 | 0.9252 | 0.8123 | 0.2892 | 0.2454 | 0.3486 | 0.0053 | — |
| checkpoint_step1000000 | 5,020 | 0.8401 | 0.8993 | 0.9051 | 0.2749 | 0.2423 | 0.4212 | 0.0078 | — |
| collect_stone @ ep5154 | 5,154 | 0.9619 | 0.9420 | 0.9333 | 0.4294 | 0.4970 | 0.4273 | 0.0079 | — |
| checkpoint_step1050000 | 5,246 | 0.7199 | 0.8834 | 0.8050 | 0.1850 | 0.1767 | 0.4025 | 0.0077 | — |
| checkpoint_step1100000 | 5,468 | 0.9352 | 0.9435 | 0.9307 | 0.3932 | 0.3671 | 0.3879 | 0.0076 | — |
| checkpoint_step1150000 | 5,693 | 0.7922 | 0.8925 | 0.8390 | 0.2304 | 0.2510 | 0.3641 | 0.0066 | — |
| checkpoint_step1200000 | 5,908 | 0.8751 | 0.9211 | 0.8638 | 0.2573 | 0.2362 | 0.3133 | 0.0049 | — |
| checkpoint_step1250000 | 6,117 | 0.9358 | 0.9560 | 0.8882 | 0.4556 | 0.4704 | 0.3971 | 0.0077 | — |
| checkpoint_step1300000 | 6,331 | 0.7387 | 0.9003 | 0.8124 | 0.1903 | 0.1779 | 0.4745 | 0.0105 | — |
| checkpoint_step1350000 | 6,540 | 0.9602 | 0.9543 | 0.9046 | 0.4321 | 0.4431 | 0.4650 | 0.0113 | — |
| checkpoint_step1400000 | 6,749 | 0.9119 | 0.9441 | 0.8594 | 0.3263 | 0.2809 | 0.5227 | 0.0132 | — |
| checkpoint_step1450000 | 6,952 | 0.8539 | 0.9353 | 0.8296 | 0.2893 | 0.2029 | 0.5614 | 0.0149 | — |
| collect_coal @ ep7002 | 7,002 | 0.8058 | 0.9402 | 0.8436 | 0.2959 | 0.2314 | 0.6286 | 0.0186 | — |
| checkpoint_step1500000 | 7,161 | 0.9721 | 0.9430 | 0.9290 | 0.5317 | 0.5625 | 0.5449 | 0.0138 | — |
| checkpoint_step1550000 | 7,373 | 0.8052 | 0.9171 | 0.8003 | 0.2440 | 0.1917 | 0.5978 | 0.0143 | — |
| checkpoint_step1600000 | 7,573 | 0.9222 | 0.9259 | 0.8204 | 0.3421 | 0.3364 | 0.6346 | 0.0170 | — |
| checkpoint_step1650000 | 7,779 | 0.9291 | 0.9413 | 0.8551 | 0.4375 | 0.4749 | 0.5623 | 0.0124 | — |
| checkpoint_step1700000 | 7,984 | 0.9253 | 0.8867 | 0.8545 | 0.3231 | 0.3298 | 0.4530 | 0.0086 | — |
| checkpoint_step1750000 | 8,188 | 0.8925 | 0.9052 | 0.7832 | 0.2404 | 0.1740 | 0.5153 | 0.0106 | — |
| place_stone @ ep8238 | 8,238 | 0.9013 | 0.9055 | 0.8431 | 0.2856 | 0.3126 | 0.4883 | 0.0093 | — |
| make_stone_pickaxe @ ep8332 | 8,332 | 0.8370 | 0.9105 | 0.8263 | 0.2804 | 0.3238 | 0.5359 | 0.0109 | — |
| checkpoint_step1800000 | 8,382 | 0.9063 | 0.9316 | 0.8701 | 0.3103 | 0.3066 | 0.5151 | 0.0099 | — |
| checkpoint_step1850000 | 8,579 | 0.8615 | 0.9463 | 0.8657 | 0.3343 | 0.2695 | 0.7527 | 0.0197 | — |
| checkpoint_step1900000 | 8,777 | 0.9634 | 0.9402 | 0.9208 | 0.5162 | 0.6350 | 0.7494 | 0.0188 | — |
| checkpoint_step1950000 | 8,969 | 0.5141 | 0.8851 | 0.7195 | 0.2420 | 0.2068 | 0.9956 | 0.0344 | — |
| checkpoint_step2000000 | 9,157 | 0.7608 | 0.9323 | 0.8464 | 0.3090 | 0.3322 | 1.1460 | 0.0448 | — |
| checkpoint_step2050000 | 9,345 | 0.7881 | 0.9050 | 0.8449 | 0.2810 | 0.4530 | 1.1456 | 0.0431 | — |
| make_stone_sword @ ep9479 | 9,479 | 0.6572 | 0.9092 | 0.7813 | 0.2820 | 0.3538 | 1.1388 | 0.0445 | — |
| checkpoint_step2100000 | 9,543 | 0.7561 | 0.9118 | 0.8001 | 0.2624 | 0.2486 | 1.0965 | 0.0404 | — |
| checkpoint_step2150000 | 9,742 | 0.7140 | 0.8877 | 0.7725 | 0.2317 | 0.2201 | 1.0932 | 0.0406 | — |
| checkpoint_step2200000 | 9,938 | 0.6794 | 0.8914 | 0.7698 | 0.2316 | 0.3004 | 1.3403 | 0.0610 | — |
| checkpoint_step2250000 | 10,136 | 0.7465 | 0.9274 | 0.8165 | 0.2805 | 0.3034 | 1.2479 | 0.0523 | — |
| eat_plant @ ep10244 | 10,244 | 0.8603 | 0.8920 | 0.8131 | 0.2314 | 0.3021 | 1.2900 | 0.0581 | — |
| checkpoint_step2300000 | 10,329 | 0.7450 | 0.8526 | 0.7704 | 0.1868 | 0.2342 | 1.2114 | 0.0490 | — |
| checkpoint_step2350000 | 10,528 | 0.5484 | 0.8866 | 0.7757 | 0.2337 | 0.2251 | 1.1801 | 0.0454 | — |
| checkpoint_step2400000 | 10,729 | 0.8413 | 0.8983 | 0.7466 | 0.2112 | 0.2378 | 1.1338 | 0.0452 | — |
| checkpoint_step2450000 | 10,927 | 0.8224 | 0.8618 | 0.7694 | 0.2323 | 0.2460 | 1.0904 | 0.0390 | — |
| checkpoint_step2500000 | 11,126 | 0.8784 | 0.8149 | 0.7836 | 0.2412 | 0.3356 | 1.1222 | 0.0422 | — |
| checkpoint_step2550000 | 11,314 | 0.7691 | 0.8646 | 0.7765 | 0.1807 | 0.2476 | 1.1166 | 0.0436 | — |
| checkpoint_step2600000 | 11,503 | 0.6995 | 0.8664 | 0.7787 | 0.1623 | 0.1999 | 1.1768 | 0.0417 | — |
| checkpoint_step2650000 | 11,700 | 0.6543 | 0.8062 | 0.6433 | 0.1464 | 0.1465 | 1.2329 | 0.0498 | — |
| place_furnace @ ep11719 | 11,719 | 0.3595 | 0.8430 | 0.6976 | 0.1870 | 0.2419 | 1.1938 | 0.0468 | — |
| checkpoint_step2700000 | 11,898 | 0.9392 | 0.9011 | 0.8394 | 0.4390 | 0.3688 | 1.0356 | 0.0353 | — |
| collect_iron @ ep11997 | 11,997 | 0.8559 | 0.8922 | 0.7445 | 0.2617 | 0.1893 | 0.9026 | 0.0243 | — |
| checkpoint_step2750000 | 12,099 | 0.8822 | 0.8923 | 0.8105 | 0.2669 | 0.4302 | 1.2531 | 0.0514 | — |
| checkpoint_step2800000 | 12,299 | 0.5417 | 0.8833 | 0.7110 | 0.1867 | 0.1967 | 1.1481 | 0.0417 | — |
| checkpoint_step2850000 | 12,498 | 0.5386 | 0.8745 | 0.6681 | 0.2156 | 0.1825 | 1.1914 | 0.0495 | — |
| checkpoint_step2900000 | 12,688 | 0.8671 | 0.8873 | 0.8050 | 0.2531 | 0.3720 | 1.0897 | 0.0371 | — |
| checkpoint_step2950000 | 12,890 | 0.9668 | 0.8902 | 0.8198 | 0.2738 | 0.2603 | 1.2310 | 0.0445 | — |
| checkpoint_step3000000 | 13,079 | 0.6664 | 0.8790 | 0.7552 | 0.1923 | 0.1600 | 1.1551 | 0.0386 | — |

---

## collect_drink_ep131_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9873 |
| Coherence (Success) | 0.8195 |
| Coherence (Failure) | 0.9250 |
| Gradient Magnitude (Success) | 0.0472 |
| Gradient Magnitude (Failure) | 0.0597 |
| Activation Separation | 0.1025 |
| Cosine Distance | 0.0002 |
| Clusters | 1,691 |
| Noise Fraction | 0.1730 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 202 | 15.8763 | 0.9870 |
| Neutral  (r = 0) | 42,499 | 0.0503 | 0.9812 |
| Negative (r < 0) | 1,304 | 2.5383 | 0.9951 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6833 |
| Pos vs Negative  | -0.6577 |
| Neutral vs Neg.  | 0.1618 |
| Pos vs Failure   | -0.8102 |
| Neutral vs Fail. | 0.5537 |
| Neg. vs Failure  | 0.8509 |

---

## collect_sapling_ep140_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.8231 |
| Coherence (Success) | 0.2769 |
| Coherence (Failure) | 0.8930 |
| Gradient Magnitude (Success) | 0.0598 |
| Gradient Magnitude (Failure) | 0.0654 |
| Activation Separation | 0.8529 |
| Cosine Distance | 0.0134 |
| Clusters | 1,642 |
| Noise Fraction | 0.1631 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 330 | 15.3480 | 0.9208 |
| Neutral  (r = 0) | 51,420 | 0.0700 | 0.4262 |
| Negative (r < 0) | 1,360 | 2.8090 | 0.9457 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.7272 |
| Pos vs Negative  | -0.7650 |
| Neutral vs Neg.  | -0.9618 |
| Pos vs Failure   | -0.7433 |
| Neutral vs Fail. | -0.9427 |
| Neg. vs Failure  | 0.9423 |

---

## collect_wood_ep149_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7830 |
| Coherence (Success) | 0.9561 |
| Coherence (Failure) | 0.9694 |
| Gradient Magnitude (Success) | 0.1017 |
| Gradient Magnitude (Failure) | 0.1305 |
| Activation Separation | 1.3189 |
| Cosine Distance | 0.0194 |
| Clusters | 1,627 |
| Noise Fraction | 0.1733 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 465 | 28.6750 | 0.9906 |
| Neutral  (r = 0) | 51,564 | 0.0964 | 0.9719 |
| Negative (r < 0) | 1,325 | 3.7723 | 0.9970 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.6642 |
| Pos vs Negative  | -0.8119 |
| Neutral vs Neg.  | -0.8135 |
| Pos vs Failure   | -0.8194 |
| Neutral vs Fail. | -0.7515 |
| Neg. vs Failure  | 0.7345 |

---

## defeat_skeleton_ep156_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7362 |
| Coherence (Success) | 0.9971 |
| Coherence (Failure) | 0.9504 |
| Gradient Magnitude (Success) | 0.4813 |
| Gradient Magnitude (Failure) | 0.1334 |
| Activation Separation | 1.6246 |
| Cosine Distance | 0.0292 |
| Clusters | 1,513 |
| Noise Fraction | 0.2172 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 632 | 17.0731 | 0.7818 |
| Neutral  (r = 0) | 51,602 | 0.4054 | 0.9979 |
| Negative (r < 0) | 1,344 | 3.4930 | 0.9953 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.9527 |
| Pos vs Negative  | -0.6701 |
| Neutral vs Neg.  | -0.6998 |
| Pos vs Failure   | -0.8140 |
| Neutral vs Fail. | -0.7756 |
| Neg. vs Failure  | 0.7077 |

---

## defeat_zombie_ep157_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7483 |
| Coherence (Success) | 0.9950 |
| Coherence (Failure) | 0.9366 |
| Gradient Magnitude (Success) | 0.2953 |
| Gradient Magnitude (Failure) | 0.0918 |
| Activation Separation | 1.4012 |
| Cosine Distance | 0.0242 |
| Clusters | 1,544 |
| Noise Fraction | 0.1940 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 673 | 8.0339 | 0.8290 |
| Neutral  (r = 0) | 51,445 | 0.2428 | 0.9975 |
| Negative (r < 0) | 1,312 | 3.7099 | 0.9953 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.8969 |
| Pos vs Negative  | -0.7355 |
| Neutral vs Neg.  | -0.7985 |
| Pos vs Failure   | -0.7583 |
| Neutral vs Fail. | -0.8252 |
| Neg. vs Failure  | 0.8105 |

---

## eat_cow_ep157_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7606 |
| Coherence (Success) | 0.9952 |
| Coherence (Failure) | 0.8990 |
| Gradient Magnitude (Success) | 0.2964 |
| Gradient Magnitude (Failure) | 0.0915 |
| Activation Separation | 1.4270 |
| Cosine Distance | 0.0251 |
| Clusters | 1,582 |
| Noise Fraction | 0.2138 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 653 | 13.2846 | 0.7629 |
| Neutral  (r = 0) | 51,561 | 0.2516 | 0.9938 |
| Negative (r < 0) | 1,335 | 3.6366 | 0.9977 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.9352 |
| Pos vs Negative  | -0.7523 |
| Neutral vs Neg.  | -0.7915 |
| Pos vs Failure   | -0.7988 |
| Neutral vs Fail. | -0.8290 |
| Neg. vs Failure  | 0.7920 |

---

## place_plant_ep178_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7809 |
| Coherence (Success) | 0.8306 |
| Coherence (Failure) | 0.9698 |
| Gradient Magnitude (Success) | 0.0711 |
| Gradient Magnitude (Failure) | 0.1735 |
| Activation Separation | 1.3458 |
| Cosine Distance | 0.0295 |
| Clusters | 1,575 |
| Noise Fraction | 0.1718 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 553 | 14.9081 | 0.7917 |
| Neutral  (r = 0) | 51,513 | 0.0678 | 0.8054 |
| Negative (r < 0) | 1,353 | 3.7636 | 0.9947 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.7218 |
| Pos vs Negative  | -0.8515 |
| Neutral vs Neg.  | -0.9551 |
| Pos vs Failure   | -0.8036 |
| Neutral vs Fail. | -0.6833 |
| Neg. vs Failure  | 0.7598 |

---

## ep288_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9873 |
| Coherence (Success) | 0.9927 |
| Coherence (Failure) | 0.9755 |
| Gradient Magnitude (Success) | 0.3753 |
| Gradient Magnitude (Failure) | 0.4091 |
| Activation Separation | 0.5322 |
| Cosine Distance | 0.0012 |
| Clusters | 1,966 |
| Noise Fraction | 0.1483 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,047 | 2.4861 | 0.9615 |
| Neutral  (r = 0) | 47,279 | 0.2698 | 0.9877 |
| Negative (r < 0) | 1,371 | 4.1859 | 0.9970 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.0225 |
| Pos vs Negative  | -0.1977 |
| Neutral vs Neg.  | -0.7893 |
| Pos vs Failure   | 0.0042 |
| Neutral vs Fail. | 0.9000 |
| Neg. vs Failure  | -0.4854 |

---

## ep564_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8526 |
| Coherence (Success) | 0.9838 |
| Coherence (Failure) | 0.9819 |
| Gradient Magnitude (Success) | 0.3348 |
| Gradient Magnitude (Failure) | 0.2109 |
| Activation Separation | 0.3569 |
| Cosine Distance | 0.0012 |
| Clusters | 1,756 |
| Noise Fraction | 0.1630 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 998 | 5.8479 | 0.9852 |
| Neutral  (r = 0) | 47,903 | 0.2441 | 0.9952 |
| Negative (r < 0) | 1,363 | 3.6849 | 0.9942 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.3270 |
| Pos vs Negative  | -0.6166 |
| Neutral vs Neg.  | -0.8642 |
| Pos vs Failure   | 0.3955 |
| Neutral vs Fail. | 0.5648 |
| Neg. vs Failure  | -0.5449 |

---

## ep830_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9625 |
| Coherence (Success) | 0.9706 |
| Coherence (Failure) | 0.9680 |
| Gradient Magnitude (Success) | 0.3161 |
| Gradient Magnitude (Failure) | 0.3028 |
| Activation Separation | 0.5415 |
| Cosine Distance | 0.0027 |
| Clusters | 2,219 |
| Noise Fraction | 0.1440 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 973 | 3.9757 | 0.9655 |
| Neutral  (r = 0) | 48,085 | 0.2319 | 0.9884 |
| Negative (r < 0) | 1,360 | 3.7473 | 0.5448 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.2856 |
| Pos vs Negative  | -0.0972 |
| Neutral vs Neg.  | 0.0169 |
| Pos vs Failure   | 0.1753 |
| Neutral vs Fail. | 0.6045 |
| Neg. vs Failure  | 0.5420 |

---

## ep1098_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8097 |
| Coherence (Success) | 0.8755 |
| Coherence (Failure) | 0.9256 |
| Gradient Magnitude (Success) | 0.1289 |
| Gradient Magnitude (Failure) | 0.1848 |
| Activation Separation | 0.4363 |
| Cosine Distance | 0.0025 |
| Clusters | 2,183 |
| Noise Fraction | 0.1700 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 944 | 3.9946 | 0.9140 |
| Neutral  (r = 0) | 46,700 | 0.2494 | 0.9864 |
| Negative (r < 0) | 1,333 | 3.7858 | 0.9830 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.0171 |
| Pos vs Negative  | -0.6027 |
| Neutral vs Neg.  | -0.3874 |
| Pos vs Failure   | -0.4035 |
| Neutral vs Fail. | 0.1635 |
| Neg. vs Failure  | 0.5909 |

---

## ep1366_lower2.000_upper3.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6837 |
| Coherence (Success) | 0.8840 |
| Coherence (Failure) | 0.9399 |
| Gradient Magnitude (Success) | 0.1893 |
| Gradient Magnitude (Failure) | 0.2090 |
| Activation Separation | 0.5740 |
| Cosine Distance | 0.0055 |
| Clusters | 2,299 |
| Noise Fraction | 0.1431 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,209 | 5.0247 | 0.9735 |
| Neutral  (r = 0) | 47,160 | 0.2412 | 0.9793 |
| Negative (r < 0) | 1,389 | 4.0615 | 0.9868 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4248 |
| Pos vs Negative  | -0.4110 |
| Neutral vs Neg.  | -0.4333 |
| Pos vs Failure   | -0.3692 |
| Neutral vs Fail. | 0.1066 |
| Neg. vs Failure  | 0.6377 |

---

## ep1626_lower3.000_upper3.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5445 |
| Coherence (Success) | 0.9136 |
| Coherence (Failure) | 0.9347 |
| Gradient Magnitude (Success) | 0.1571 |
| Gradient Magnitude (Failure) | 0.1887 |
| Activation Separation | 0.5531 |
| Cosine Distance | 0.0062 |
| Clusters | 2,216 |
| Noise Fraction | 0.1571 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,354 | 5.9384 | 0.9895 |
| Neutral  (r = 0) | 47,918 | 0.2108 | 0.9872 |
| Negative (r < 0) | 1,366 | 3.9458 | 0.9887 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4684 |
| Pos vs Negative  | -0.4030 |
| Neutral vs Neg.  | -0.3693 |
| Pos vs Failure   | -0.0115 |
| Neutral vs Fail. | -0.2560 |
| Neg. vs Failure  | 0.6685 |

---

## ep1888_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8381 |
| Coherence (Success) | 0.9805 |
| Coherence (Failure) | 0.9786 |
| Gradient Magnitude (Success) | 0.4954 |
| Gradient Magnitude (Failure) | 0.3245 |
| Activation Separation | 0.6854 |
| Cosine Distance | 0.0089 |
| Clusters | 1,907 |
| Noise Fraction | 0.1716 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,395 | 8.7147 | 0.9934 |
| Neutral  (r = 0) | 48,419 | 0.3781 | 0.9940 |
| Negative (r < 0) | 1,386 | 7.8206 | 0.6228 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.2721 |
| Pos vs Negative  | 0.1674 |
| Neutral vs Neg.  | -0.0489 |
| Pos vs Failure   | 0.7905 |
| Neutral vs Fail. | 0.3541 |
| Neg. vs Failure  | 0.2507 |

---

## place_table_ep1890_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5772 |
| Coherence (Success) | 0.9597 |
| Coherence (Failure) | 0.8804 |
| Gradient Magnitude (Success) | 0.2934 |
| Gradient Magnitude (Failure) | 0.2072 |
| Activation Separation | 0.6121 |
| Cosine Distance | 0.0074 |
| Clusters | 2,013 |
| Noise Fraction | 0.1689 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,365 | 8.2562 | 0.9933 |
| Neutral  (r = 0) | 49,228 | 0.2980 | 0.9905 |
| Negative (r < 0) | 1,378 | 3.4698 | 0.9799 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1370 |
| Pos vs Negative  | -0.2791 |
| Neutral vs Neg.  | -0.7684 |
| Pos vs Failure   | 0.6451 |
| Neutral vs Fail. | -0.1274 |
| Neg. vs Failure  | 0.0763 |

---

## ep2153_lower3.900_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6783 |
| Coherence (Success) | 0.9382 |
| Coherence (Failure) | 0.8807 |
| Gradient Magnitude (Success) | 0.2731 |
| Gradient Magnitude (Failure) | 0.1980 |
| Activation Separation | 0.5787 |
| Cosine Distance | 0.0077 |
| Clusters | 2,290 |
| Noise Fraction | 0.1678 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,449 | 9.7139 | 0.9839 |
| Neutral  (r = 0) | 48,046 | 0.3041 | 0.9912 |
| Negative (r < 0) | 1,353 | 8.2837 | 0.5704 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4173 |
| Pos vs Negative  | 0.2210 |
| Neutral vs Neg.  | -0.0910 |
| Pos vs Failure   | 0.6704 |
| Neutral vs Fail. | -0.3365 |
| Neg. vs Failure  | 0.2756 |

---

## make_wood_sword_ep2157_lower3.900_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3961 |
| Coherence (Success) | 0.8725 |
| Coherence (Failure) | 0.8883 |
| Gradient Magnitude (Success) | 0.2305 |
| Gradient Magnitude (Failure) | 0.2723 |
| Activation Separation | 0.7132 |
| Cosine Distance | 0.0120 |
| Clusters | 2,306 |
| Noise Fraction | 0.1824 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,496 | 9.6285 | 0.9773 |
| Neutral  (r = 0) | 50,003 | 0.3370 | 0.9852 |
| Negative (r < 0) | 1,379 | 8.8365 | 0.6182 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4580 |
| Pos vs Negative  | 0.2752 |
| Neutral vs Neg.  | -0.2397 |
| Pos vs Failure   | 0.2518 |
| Neutral vs Fail. | 0.1326 |
| Neg. vs Failure  | 0.0591 |

---

## ep2406_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4701 |
| Coherence (Success) | 0.9626 |
| Coherence (Failure) | 0.9154 |
| Gradient Magnitude (Success) | 0.3546 |
| Gradient Magnitude (Failure) | 0.2832 |
| Activation Separation | 0.7192 |
| Cosine Distance | 0.0142 |
| Clusters | 2,200 |
| Noise Fraction | 0.1922 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,755 | 9.8666 | 0.9920 |
| Neutral  (r = 0) | 50,270 | 0.2873 | 0.9876 |
| Negative (r < 0) | 1,428 | 3.4269 | 0.9736 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3669 |
| Pos vs Negative  | -0.2319 |
| Neutral vs Neg.  | -0.5591 |
| Pos vs Failure   | 0.5873 |
| Neutral vs Fail. | -0.3050 |
| Neg. vs Failure  | 0.3420 |

---

## make_wood_pickaxe_ep2420_lower3.900_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6115 |
| Coherence (Success) | 0.9401 |
| Coherence (Failure) | 0.9243 |
| Gradient Magnitude (Success) | 0.3368 |
| Gradient Magnitude (Failure) | 0.2935 |
| Activation Separation | 0.7481 |
| Cosine Distance | 0.0157 |
| Clusters | 1,981 |
| Noise Fraction | 0.2049 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,656 | 9.7094 | 0.9844 |
| Neutral  (r = 0) | 48,323 | 0.2694 | 0.9797 |
| Negative (r < 0) | 1,397 | 3.6359 | 0.9756 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4068 |
| Pos vs Negative  | -0.2372 |
| Neutral vs Neg.  | -0.5006 |
| Pos vs Failure   | 0.6706 |
| Neutral vs Fail. | -0.3227 |
| Neg. vs Failure  | 0.2502 |

---

## wake_up_ep2589_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7443 |
| Coherence (Success) | 0.9581 |
| Coherence (Failure) | 0.9151 |
| Gradient Magnitude (Success) | 0.4170 |
| Gradient Magnitude (Failure) | 0.3056 |
| Activation Separation | 0.9647 |
| Cosine Distance | 0.0264 |
| Clusters | 1,995 |
| Noise Fraction | 0.2133 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,848 | 11.0677 | 0.9904 |
| Neutral  (r = 0) | 48,230 | 0.2558 | 0.9856 |
| Negative (r < 0) | 1,394 | 4.2433 | 0.9662 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4906 |
| Pos vs Negative  | -0.2103 |
| Neutral vs Neg.  | -0.3770 |
| Pos vs Failure   | 0.7314 |
| Neutral vs Fail. | -0.4025 |
| Neg. vs Failure  | 0.3067 |

---

## ep2665_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3715 |
| Coherence (Success) | 0.9106 |
| Coherence (Failure) | 0.9112 |
| Gradient Magnitude (Success) | 0.2315 |
| Gradient Magnitude (Failure) | 0.3020 |
| Activation Separation | 0.9573 |
| Cosine Distance | 0.0263 |
| Clusters | 2,168 |
| Noise Fraction | 0.1921 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,785 | 9.9213 | 0.9887 |
| Neutral  (r = 0) | 49,240 | 0.4017 | 0.9886 |
| Negative (r < 0) | 1,395 | 3.8118 | 0.9705 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5902 |
| Pos vs Negative  | -0.1868 |
| Neutral vs Neg.  | -0.2727 |
| Pos vs Failure   | 0.1537 |
| Neutral vs Fail. | 0.3032 |
| Neg. vs Failure  | 0.3207 |

---

## ep2917_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5210 |
| Coherence (Success) | 0.9230 |
| Coherence (Failure) | 0.8824 |
| Gradient Magnitude (Success) | 0.3178 |
| Gradient Magnitude (Failure) | 0.2266 |
| Activation Separation | 1.1419 |
| Cosine Distance | 0.0400 |
| Clusters | 2,424 |
| Noise Fraction | 0.1883 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,875 | 9.5501 | 0.9856 |
| Neutral  (r = 0) | 52,371 | 0.3309 | 0.9862 |
| Negative (r < 0) | 1,510 | 3.7725 | 0.9625 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4523 |
| Pos vs Negative  | -0.2179 |
| Neutral vs Neg.  | -0.5302 |
| Pos vs Failure   | 0.4895 |
| Neutral vs Fail. | -0.0754 |
| Neg. vs Failure  | 0.1742 |

---

## ep3166_lower4.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8255 |
| Coherence (Success) | 0.9426 |
| Coherence (Failure) | 0.8644 |
| Gradient Magnitude (Success) | 0.3804 |
| Gradient Magnitude (Failure) | 0.2363 |
| Activation Separation | 0.6549 |
| Cosine Distance | 0.0133 |
| Clusters | 2,331 |
| Noise Fraction | 0.1823 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,972 | 11.2152 | 0.9875 |
| Neutral  (r = 0) | 51,266 | 0.3170 | 0.9863 |
| Negative (r < 0) | 1,499 | 3.7491 | 0.9595 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5610 |
| Pos vs Negative  | -0.1352 |
| Neutral vs Neg.  | -0.4622 |
| Pos vs Failure   | 0.7944 |
| Neutral vs Fail. | -0.3678 |
| Neg. vs Failure  | 0.0896 |

---

## ep3410_lower4.900_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8550 |
| Coherence (Success) | 0.9541 |
| Coherence (Failure) | 0.9117 |
| Gradient Magnitude (Success) | 0.4337 |
| Gradient Magnitude (Failure) | 0.3978 |
| Activation Separation | 0.4519 |
| Cosine Distance | 0.0067 |
| Clusters | 2,141 |
| Noise Fraction | 0.1924 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,039 | 11.6831 | 0.9902 |
| Neutral  (r = 0) | 51,659 | 0.1958 | 0.9711 |
| Negative (r < 0) | 1,459 | 3.8816 | 0.9682 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4909 |
| Pos vs Negative  | -0.1055 |
| Neutral vs Neg.  | -0.2345 |
| Pos vs Failure   | 0.8167 |
| Neutral vs Fail. | -0.3508 |
| Neg. vs Failure  | 0.3065 |

---

## ep3654_lower5.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8512 |
| Coherence (Success) | 0.9622 |
| Coherence (Failure) | 0.8795 |
| Gradient Magnitude (Success) | 0.3644 |
| Gradient Magnitude (Failure) | 0.3165 |
| Activation Separation | 0.4631 |
| Cosine Distance | 0.0078 |
| Clusters | 2,165 |
| Noise Fraction | 0.2135 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,228 | 9.7044 | 0.9891 |
| Neutral  (r = 0) | 54,005 | 0.2331 | 0.9775 |
| Negative (r < 0) | 1,489 | 3.6526 | 0.9696 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4298 |
| Pos vs Negative  | -0.2730 |
| Neutral vs Neg.  | -0.3398 |
| Pos vs Failure   | 0.8832 |
| Neutral vs Fail. | -0.3038 |
| Neg. vs Failure  | -0.0860 |

---

## ep3887_lower5.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8226 |
| Coherence (Success) | 0.8894 |
| Coherence (Failure) | 0.7985 |
| Gradient Magnitude (Success) | 0.2832 |
| Gradient Magnitude (Failure) | 0.2183 |
| Activation Separation | 0.4276 |
| Cosine Distance | 0.0067 |
| Clusters | 2,018 |
| Noise Fraction | 0.2294 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,352 | 11.0423 | 0.9923 |
| Neutral  (r = 0) | 53,577 | 0.3934 | 0.9863 |
| Negative (r < 0) | 1,480 | 3.6691 | 0.9710 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7048 |
| Pos vs Negative  | -0.1698 |
| Neutral vs Neg.  | -0.1613 |
| Pos vs Failure   | 0.6379 |
| Neutral vs Fail. | -0.1748 |
| Neg. vs Failure  | -0.0097 |

---

## ep4117_lower5.900_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9124 |
| Coherence (Success) | 0.9403 |
| Coherence (Failure) | 0.8603 |
| Gradient Magnitude (Success) | 0.3546 |
| Gradient Magnitude (Failure) | 0.2898 |
| Activation Separation | 0.4067 |
| Cosine Distance | 0.0067 |
| Clusters | 1,867 |
| Noise Fraction | 0.2559 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,349 | 10.9460 | 0.9861 |
| Neutral  (r = 0) | 54,114 | 0.2576 | 0.9871 |
| Negative (r < 0) | 1,493 | 3.8063 | 0.9765 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6094 |
| Pos vs Negative  | -0.2045 |
| Neutral vs Neg.  | -0.1890 |
| Pos vs Failure   | 0.9038 |
| Neutral vs Fail. | -0.5211 |
| Neg. vs Failure  | -0.0512 |

---

## ep4348_lower6.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8607 |
| Coherence (Success) | 0.9067 |
| Coherence (Failure) | 0.8958 |
| Gradient Magnitude (Success) | 0.2838 |
| Gradient Magnitude (Failure) | 0.2901 |
| Activation Separation | 0.3126 |
| Cosine Distance | 0.0046 |
| Clusters | 1,735 |
| Noise Fraction | 0.2893 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,340 | 9.7617 | 0.9881 |
| Neutral  (r = 0) | 54,079 | 0.3042 | 0.9832 |
| Negative (r < 0) | 1,481 | 4.0143 | 0.9714 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5708 |
| Pos vs Negative  | -0.2001 |
| Neutral vs Neg.  | 0.2801 |
| Pos vs Failure   | 0.6167 |
| Neutral vs Fail. | 0.0592 |
| Neg. vs Failure  | 0.3624 |

---

## ep4578_lower6.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9072 |
| Coherence (Success) | 0.9518 |
| Coherence (Failure) | 0.8238 |
| Gradient Magnitude (Success) | 0.3623 |
| Gradient Magnitude (Failure) | 0.2512 |
| Activation Separation | 0.3016 |
| Cosine Distance | 0.0039 |
| Clusters | 1,498 |
| Noise Fraction | 0.3196 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,365 | 11.0251 | 0.9918 |
| Neutral  (r = 0) | 52,213 | 0.2862 | 0.9816 |
| Negative (r < 0) | 1,494 | 3.7247 | 0.9686 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6861 |
| Pos vs Negative  | -0.1393 |
| Neutral vs Neg.  | -0.2435 |
| Pos vs Failure   | 0.8705 |
| Neutral vs Fail. | -0.4399 |
| Neg. vs Failure  | -0.1656 |

---

## ep4799_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9221 |
| Coherence (Success) | 0.9252 |
| Coherence (Failure) | 0.8123 |
| Gradient Magnitude (Success) | 0.2892 |
| Gradient Magnitude (Failure) | 0.2454 |
| Activation Separation | 0.3486 |
| Cosine Distance | 0.0053 |
| Clusters | 1,417 |
| Noise Fraction | 0.2837 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,411 | 9.2535 | 0.9798 |
| Neutral  (r = 0) | 52,518 | 0.3385 | 0.9822 |
| Negative (r < 0) | 1,476 | 3.8926 | 0.9749 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6710 |
| Pos vs Negative  | -0.2156 |
| Neutral vs Neg.  | -0.1405 |
| Pos vs Failure   | 0.7537 |
| Neutral vs Fail. | -0.2508 |
| Neg. vs Failure  | -0.1794 |

---

## ep5020_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8401 |
| Coherence (Success) | 0.8993 |
| Coherence (Failure) | 0.9051 |
| Gradient Magnitude (Success) | 0.2749 |
| Gradient Magnitude (Failure) | 0.2423 |
| Activation Separation | 0.4212 |
| Cosine Distance | 0.0078 |
| Clusters | 2,001 |
| Noise Fraction | 0.2615 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,465 | 8.2585 | 0.9780 |
| Neutral  (r = 0) | 56,219 | 0.2336 | 0.9792 |
| Negative (r < 0) | 1,554 | 7.3248 | 0.6323 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5013 |
| Pos vs Negative  | 0.1821 |
| Neutral vs Neg.  | 0.0575 |
| Pos vs Failure   | 0.8622 |
| Neutral vs Fail. | -0.3059 |
| Neg. vs Failure  | 0.3405 |

---

## collect_stone_ep5154_lower6.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9619 |
| Coherence (Success) | 0.9420 |
| Coherence (Failure) | 0.9333 |
| Gradient Magnitude (Success) | 0.4294 |
| Gradient Magnitude (Failure) | 0.4970 |
| Activation Separation | 0.4273 |
| Cosine Distance | 0.0079 |
| Clusters | 1,905 |
| Noise Fraction | 0.2775 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,472 | 9.3184 | 0.9817 |
| Neutral  (r = 0) | 56,284 | 0.1758 | 0.9647 |
| Negative (r < 0) | 1,536 | 3.5285 | 0.9726 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1146 |
| Pos vs Negative  | -0.0907 |
| Neutral vs Neg.  | 0.1498 |
| Pos vs Failure   | 0.9135 |
| Neutral vs Fail. | 0.0519 |
| Neg. vs Failure  | 0.1631 |

---

## ep5246_lower6.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7199 |
| Coherence (Success) | 0.8834 |
| Coherence (Failure) | 0.8050 |
| Gradient Magnitude (Success) | 0.1850 |
| Gradient Magnitude (Failure) | 0.1767 |
| Activation Separation | 0.4025 |
| Cosine Distance | 0.0077 |
| Clusters | 2,184 |
| Noise Fraction | 0.2536 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,488 | 7.2689 | 0.9861 |
| Neutral  (r = 0) | 56,213 | 0.4431 | 0.9880 |
| Negative (r < 0) | 1,547 | 3.3396 | 0.9766 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.8299 |
| Pos vs Negative  | -0.1804 |
| Neutral vs Neg.  | 0.0072 |
| Pos vs Failure   | -0.3124 |
| Neutral vs Fail. | 0.5738 |
| Neg. vs Failure  | 0.1904 |

---

## ep5468_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9352 |
| Coherence (Success) | 0.9435 |
| Coherence (Failure) | 0.9307 |
| Gradient Magnitude (Success) | 0.3932 |
| Gradient Magnitude (Failure) | 0.3671 |
| Activation Separation | 0.3879 |
| Cosine Distance | 0.0076 |
| Clusters | 1,767 |
| Noise Fraction | 0.2790 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,606 | 7.8883 | 0.9749 |
| Neutral  (r = 0) | 56,680 | 0.2134 | 0.9744 |
| Negative (r < 0) | 1,586 | 3.2173 | 0.9569 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4302 |
| Pos vs Negative  | -0.1713 |
| Neutral vs Neg.  | -0.4409 |
| Pos vs Failure   | 0.9208 |
| Neutral vs Fail. | -0.3125 |
| Neg. vs Failure  | -0.1284 |

---

## ep5693_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7922 |
| Coherence (Success) | 0.8925 |
| Coherence (Failure) | 0.8390 |
| Gradient Magnitude (Success) | 0.2304 |
| Gradient Magnitude (Failure) | 0.2510 |
| Activation Separation | 0.3641 |
| Cosine Distance | 0.0066 |
| Clusters | 1,839 |
| Noise Fraction | 0.2941 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,627 | 6.9308 | 0.9694 |
| Neutral  (r = 0) | 56,666 | 0.5299 | 0.9860 |
| Negative (r < 0) | 1,587 | 3.0788 | 0.9658 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7990 |
| Pos vs Negative  | -0.2177 |
| Neutral vs Neg.  | -0.0586 |
| Pos vs Failure   | -0.4407 |
| Neutral vs Fail. | 0.7728 |
| Neg. vs Failure  | 0.0940 |

---

## ep5908_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8751 |
| Coherence (Success) | 0.9211 |
| Coherence (Failure) | 0.8638 |
| Gradient Magnitude (Success) | 0.2573 |
| Gradient Magnitude (Failure) | 0.2362 |
| Activation Separation | 0.3133 |
| Cosine Distance | 0.0049 |
| Clusters | 2,083 |
| Noise Fraction | 0.2520 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,646 | 7.8858 | 0.9766 |
| Neutral  (r = 0) | 58,573 | 0.2915 | 0.9786 |
| Negative (r < 0) | 1,621 | 6.3846 | 0.6191 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7268 |
| Pos vs Negative  | 0.1848 |
| Neutral vs Neg.  | 0.0092 |
| Pos vs Failure   | 0.8596 |
| Neutral vs Fail. | -0.5728 |
| Neg. vs Failure  | 0.3564 |

---

## ep6117_lower7.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9358 |
| Coherence (Success) | 0.9560 |
| Coherence (Failure) | 0.8882 |
| Gradient Magnitude (Success) | 0.4556 |
| Gradient Magnitude (Failure) | 0.4704 |
| Activation Separation | 0.3971 |
| Cosine Distance | 0.0077 |
| Clusters | 1,977 |
| Noise Fraction | 0.2720 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,801 | 9.2599 | 0.9853 |
| Neutral  (r = 0) | 59,190 | 0.2038 | 0.9745 |
| Negative (r < 0) | 1,641 | 3.3731 | 0.9668 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4652 |
| Pos vs Negative  | -0.0955 |
| Neutral vs Neg.  | -0.3387 |
| Pos vs Failure   | 0.9317 |
| Neutral vs Fail. | -0.4328 |
| Neg. vs Failure  | 0.1185 |

---

## ep6331_lower7.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7387 |
| Coherence (Success) | 0.9003 |
| Coherence (Failure) | 0.8124 |
| Gradient Magnitude (Success) | 0.1903 |
| Gradient Magnitude (Failure) | 0.1779 |
| Activation Separation | 0.4745 |
| Cosine Distance | 0.0105 |
| Clusters | 1,958 |
| Noise Fraction | 0.2714 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,691 | 7.7066 | 0.9791 |
| Neutral  (r = 0) | 59,308 | 0.4936 | 0.9813 |
| Negative (r < 0) | 1,604 | 3.1151 | 0.9520 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.8383 |
| Pos vs Negative  | -0.1219 |
| Neutral vs Neg.  | -0.0085 |
| Pos vs Failure   | -0.1467 |
| Neutral vs Fail. | 0.4629 |
| Neg. vs Failure  | 0.3749 |

---

## ep6540_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9602 |
| Coherence (Success) | 0.9543 |
| Coherence (Failure) | 0.9046 |
| Gradient Magnitude (Success) | 0.4321 |
| Gradient Magnitude (Failure) | 0.4431 |
| Activation Separation | 0.4650 |
| Cosine Distance | 0.0113 |
| Clusters | 2,127 |
| Noise Fraction | 0.2700 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,827 | 8.9175 | 0.9799 |
| Neutral  (r = 0) | 60,777 | 0.1800 | 0.9689 |
| Negative (r < 0) | 1,651 | 2.9158 | 0.9586 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5607 |
| Pos vs Negative  | -0.0419 |
| Neutral vs Neg.  | -0.3204 |
| Pos vs Failure   | 0.9438 |
| Neutral vs Fail. | -0.5374 |
| Neg. vs Failure  | 0.1229 |

---

## ep6749_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9119 |
| Coherence (Success) | 0.9441 |
| Coherence (Failure) | 0.8594 |
| Gradient Magnitude (Success) | 0.3263 |
| Gradient Magnitude (Failure) | 0.2809 |
| Activation Separation | 0.5227 |
| Cosine Distance | 0.0132 |
| Clusters | 2,164 |
| Noise Fraction | 0.2567 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,918 | 9.2501 | 0.9800 |
| Neutral  (r = 0) | 59,160 | 0.3263 | 0.9821 |
| Negative (r < 0) | 1,608 | 3.0905 | 0.9509 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7901 |
| Pos vs Negative  | -0.1320 |
| Neutral vs Neg.  | -0.1631 |
| Pos vs Failure   | 0.9247 |
| Neutral vs Fail. | -0.7153 |
| Neg. vs Failure  | 0.0363 |

---

## ep6952_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8539 |
| Coherence (Success) | 0.9353 |
| Coherence (Failure) | 0.8296 |
| Gradient Magnitude (Success) | 0.2893 |
| Gradient Magnitude (Failure) | 0.2029 |
| Activation Separation | 0.5614 |
| Cosine Distance | 0.0149 |
| Clusters | 2,171 |
| Noise Fraction | 0.2491 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,958 | 8.3266 | 0.9854 |
| Neutral  (r = 0) | 59,284 | 0.3778 | 0.9817 |
| Negative (r < 0) | 1,646 | 3.2636 | 0.9652 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.8008 |
| Pos vs Negative  | -0.0699 |
| Neutral vs Neg.  | -0.2289 |
| Pos vs Failure   | 0.8155 |
| Neutral vs Fail. | -0.6364 |
| Neg. vs Failure  | 0.1702 |

---

## collect_coal_ep7002_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8058 |
| Coherence (Success) | 0.9402 |
| Coherence (Failure) | 0.8436 |
| Gradient Magnitude (Success) | 0.2959 |
| Gradient Magnitude (Failure) | 0.2314 |
| Activation Separation | 0.6286 |
| Cosine Distance | 0.0186 |
| Clusters | 2,255 |
| Noise Fraction | 0.2686 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,038 | 8.4396 | 0.9881 |
| Neutral  (r = 0) | 64,728 | 0.3287 | 0.9798 |
| Negative (r < 0) | 1,774 | 3.2082 | 0.9594 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7370 |
| Pos vs Negative  | -0.0235 |
| Neutral vs Neg.  | -0.3421 |
| Pos vs Failure   | 0.8763 |
| Neutral vs Fail. | -0.6141 |
| Neg. vs Failure  | 0.1270 |

---

## ep7161_lower7.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9721 |
| Coherence (Success) | 0.9430 |
| Coherence (Failure) | 0.9290 |
| Gradient Magnitude (Success) | 0.5317 |
| Gradient Magnitude (Failure) | 0.5625 |
| Activation Separation | 0.5449 |
| Cosine Distance | 0.0138 |
| Clusters | 1,874 |
| Noise Fraction | 0.3048 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,042 | 10.2104 | 0.9757 |
| Neutral  (r = 0) | 60,944 | 0.2272 | 0.9705 |
| Negative (r < 0) | 1,668 | 3.2579 | 0.9657 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4549 |
| Pos vs Negative  | -0.0473 |
| Neutral vs Neg.  | -0.2090 |
| Pos vs Failure   | 0.9329 |
| Neutral vs Fail. | -0.3341 |
| Neg. vs Failure  | 0.1219 |

---

## ep7373_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8052 |
| Coherence (Success) | 0.9171 |
| Coherence (Failure) | 0.8003 |
| Gradient Magnitude (Success) | 0.2440 |
| Gradient Magnitude (Failure) | 0.1917 |
| Activation Separation | 0.5978 |
| Cosine Distance | 0.0143 |
| Clusters | 1,946 |
| Noise Fraction | 0.2975 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,042 | 9.3470 | 0.9839 |
| Neutral  (r = 0) | 59,529 | 0.4691 | 0.9837 |
| Negative (r < 0) | 1,694 | 3.1095 | 0.9615 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.8345 |
| Pos vs Negative  | -0.0730 |
| Neutral vs Neg.  | -0.1681 |
| Pos vs Failure   | 0.6716 |
| Neutral vs Fail. | -0.3996 |
| Neg. vs Failure  | 0.1120 |

---

## ep7573_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9222 |
| Coherence (Success) | 0.9259 |
| Coherence (Failure) | 0.8204 |
| Gradient Magnitude (Success) | 0.3421 |
| Gradient Magnitude (Failure) | 0.3364 |
| Activation Separation | 0.6346 |
| Cosine Distance | 0.0170 |
| Clusters | 2,320 |
| Noise Fraction | 0.2597 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,151 | 10.0609 | 0.9646 |
| Neutral  (r = 0) | 66,676 | 0.2946 | 0.9800 |
| Negative (r < 0) | 1,825 | 7.2952 | 0.6618 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7975 |
| Pos vs Negative  | 0.1428 |
| Neutral vs Neg.  | -0.0499 |
| Pos vs Failure   | 0.9328 |
| Neutral vs Fail. | -0.7376 |
| Neg. vs Failure  | 0.2623 |

---

## ep7779_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9291 |
| Coherence (Success) | 0.9413 |
| Coherence (Failure) | 0.8551 |
| Gradient Magnitude (Success) | 0.4375 |
| Gradient Magnitude (Failure) | 0.4749 |
| Activation Separation | 0.5623 |
| Cosine Distance | 0.0124 |
| Clusters | 2,142 |
| Noise Fraction | 0.2681 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,094 | 11.0137 | 0.9773 |
| Neutral  (r = 0) | 64,215 | 0.2859 | 0.9705 |
| Negative (r < 0) | 1,731 | 2.8562 | 0.9461 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6177 |
| Pos vs Negative  | -0.0201 |
| Neutral vs Neg.  | -0.3568 |
| Pos vs Failure   | 0.9437 |
| Neutral vs Fail. | -0.5223 |
| Neg. vs Failure  | 0.0891 |

---

## ep7984_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9253 |
| Coherence (Success) | 0.8867 |
| Coherence (Failure) | 0.8545 |
| Gradient Magnitude (Success) | 0.3231 |
| Gradient Magnitude (Failure) | 0.3298 |
| Activation Separation | 0.4530 |
| Cosine Distance | 0.0086 |
| Clusters | 1,907 |
| Noise Fraction | 0.2735 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,228 | 9.5004 | 0.9704 |
| Neutral  (r = 0) | 62,632 | 0.3380 | 0.9723 |
| Negative (r < 0) | 1,764 | 3.0733 | 0.9401 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6946 |
| Pos vs Negative  | 0.0143 |
| Neutral vs Neg.  | -0.2427 |
| Pos vs Failure   | 0.8853 |
| Neutral vs Fail. | -0.4953 |
| Neg. vs Failure  | 0.2114 |

---

## ep8188_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8925 |
| Coherence (Success) | 0.9052 |
| Coherence (Failure) | 0.7832 |
| Gradient Magnitude (Success) | 0.2404 |
| Gradient Magnitude (Failure) | 0.1740 |
| Activation Separation | 0.5153 |
| Cosine Distance | 0.0106 |
| Clusters | 2,128 |
| Noise Fraction | 0.2713 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,203 | 7.4258 | 0.9531 |
| Neutral  (r = 0) | 62,792 | 0.4897 | 0.9802 |
| Negative (r < 0) | 1,790 | 2.8123 | 0.9419 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7624 |
| Pos vs Negative  | -0.0162 |
| Neutral vs Neg.  | -0.2466 |
| Pos vs Failure   | 0.0233 |
| Neutral vs Fail. | 0.4383 |
| Neg. vs Failure  | -0.0816 |

---

## place_stone_ep8238_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9013 |
| Coherence (Success) | 0.9055 |
| Coherence (Failure) | 0.8431 |
| Gradient Magnitude (Success) | 0.2856 |
| Gradient Magnitude (Failure) | 0.3126 |
| Activation Separation | 0.4883 |
| Cosine Distance | 0.0093 |
| Clusters | 1,981 |
| Noise Fraction | 0.2806 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,266 | 7.6525 | 0.9738 |
| Neutral  (r = 0) | 62,160 | 0.2498 | 0.9711 |
| Negative (r < 0) | 1,762 | 7.6417 | 0.6601 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6402 |
| Pos vs Negative  | 0.1638 |
| Neutral vs Neg.  | 0.0366 |
| Pos vs Failure   | 0.8806 |
| Neutral vs Fail. | -0.4772 |
| Neg. vs Failure  | 0.3046 |

---

## make_stone_pickaxe_ep8332_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8370 |
| Coherence (Success) | 0.9105 |
| Coherence (Failure) | 0.8263 |
| Gradient Magnitude (Success) | 0.2804 |
| Gradient Magnitude (Failure) | 0.3238 |
| Activation Separation | 0.5359 |
| Cosine Distance | 0.0109 |
| Clusters | 2,120 |
| Noise Fraction | 0.2688 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,115 | 8.7115 | 0.9817 |
| Neutral  (r = 0) | 62,336 | 0.2814 | 0.9686 |
| Negative (r < 0) | 1,664 | 12.8637 | 0.5112 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7421 |
| Pos vs Negative  | 0.2152 |
| Neutral vs Neg.  | 0.0163 |
| Pos vs Failure   | 0.7521 |
| Neutral vs Fail. | -0.3526 |
| Neg. vs Failure  | 0.2750 |

---

## ep8382_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9063 |
| Coherence (Success) | 0.9316 |
| Coherence (Failure) | 0.8701 |
| Gradient Magnitude (Success) | 0.3103 |
| Gradient Magnitude (Failure) | 0.3066 |
| Activation Separation | 0.5151 |
| Cosine Distance | 0.0099 |
| Clusters | 2,169 |
| Noise Fraction | 0.2681 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,214 | 8.1360 | 0.9666 |
| Neutral  (r = 0) | 63,241 | 0.2497 | 0.9708 |
| Negative (r < 0) | 1,761 | 3.1166 | 0.9426 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6023 |
| Pos vs Negative  | -0.0344 |
| Neutral vs Neg.  | -0.3263 |
| Pos vs Failure   | 0.9240 |
| Neutral vs Fail. | -0.5432 |
| Neg. vs Failure  | 0.1230 |

---

## ep8579_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8615 |
| Coherence (Success) | 0.9463 |
| Coherence (Failure) | 0.8657 |
| Gradient Magnitude (Success) | 0.3343 |
| Gradient Magnitude (Failure) | 0.2695 |
| Activation Separation | 0.7527 |
| Cosine Distance | 0.0197 |
| Clusters | 2,415 |
| Noise Fraction | 0.2564 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,416 | 6.8137 | 0.9844 |
| Neutral  (r = 0) | 64,820 | 0.2810 | 0.9739 |
| Negative (r < 0) | 1,793 | 3.1452 | 0.9554 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5699 |
| Pos vs Negative  | -0.0350 |
| Neutral vs Neg.  | -0.3690 |
| Pos vs Failure   | 0.8417 |
| Neutral vs Fail. | -0.3726 |
| Neg. vs Failure  | 0.0860 |

---

## ep8777_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9634 |
| Coherence (Success) | 0.9402 |
| Coherence (Failure) | 0.9208 |
| Gradient Magnitude (Success) | 0.5162 |
| Gradient Magnitude (Failure) | 0.6350 |
| Activation Separation | 0.7494 |
| Cosine Distance | 0.0188 |
| Clusters | 2,407 |
| Noise Fraction | 0.2523 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,418 | 8.1021 | 0.9861 |
| Neutral  (r = 0) | 65,517 | 0.2414 | 0.9720 |
| Negative (r < 0) | 1,781 | 6.1778 | 0.6446 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.0843 |
| Pos vs Negative  | 0.1823 |
| Neutral vs Neg.  | 0.0578 |
| Pos vs Failure   | 0.8896 |
| Neutral vs Fail. | 0.3434 |
| Neg. vs Failure  | 0.1796 |

---

## ep8969_lower8.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5141 |
| Coherence (Success) | 0.8851 |
| Coherence (Failure) | 0.7195 |
| Gradient Magnitude (Success) | 0.2420 |
| Gradient Magnitude (Failure) | 0.2068 |
| Activation Separation | 0.9956 |
| Cosine Distance | 0.0344 |
| Clusters | 2,542 |
| Noise Fraction | 0.2234 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,440 | 6.5527 | 0.9804 |
| Neutral  (r = 0) | 65,376 | 0.4572 | 0.9773 |
| Negative (r < 0) | 1,810 | 2.9845 | 0.9454 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7152 |
| Pos vs Negative  | -0.1275 |
| Neutral vs Neg.  | -0.1007 |
| Pos vs Failure   | -0.0147 |
| Neutral vs Fail. | 0.4722 |
| Neg. vs Failure  | 0.2212 |

---

## ep9157_lower9.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7608 |
| Coherence (Success) | 0.9323 |
| Coherence (Failure) | 0.8464 |
| Gradient Magnitude (Success) | 0.3090 |
| Gradient Magnitude (Failure) | 0.3322 |
| Activation Separation | 1.1460 |
| Cosine Distance | 0.0448 |
| Clusters | 2,406 |
| Noise Fraction | 0.2187 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,555 | 7.7721 | 0.9815 |
| Neutral  (r = 0) | 63,699 | 0.2716 | 0.9789 |
| Negative (r < 0) | 1,775 | 3.1305 | 0.9483 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6416 |
| Pos vs Negative  | -0.0517 |
| Neutral vs Neg.  | -0.1880 |
| Pos vs Failure   | 0.8710 |
| Neutral vs Fail. | -0.5394 |
| Neg. vs Failure  | 0.1955 |

---

## ep9345_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7881 |
| Coherence (Success) | 0.9050 |
| Coherence (Failure) | 0.8449 |
| Gradient Magnitude (Success) | 0.2810 |
| Gradient Magnitude (Failure) | 0.4530 |
| Activation Separation | 1.1456 |
| Cosine Distance | 0.0431 |
| Clusters | 2,476 |
| Noise Fraction | 0.2249 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,600 | 6.6298 | 0.9845 |
| Neutral  (r = 0) | 61,361 | 0.3011 | 0.9743 |
| Negative (r < 0) | 1,729 | 3.0195 | 0.9472 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4054 |
| Pos vs Negative  | -0.0124 |
| Neutral vs Neg.  | 0.1811 |
| Pos vs Failure   | 0.6646 |
| Neutral vs Fail. | 0.1288 |
| Neg. vs Failure  | 0.3063 |

---

## make_stone_sword_ep9479_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6572 |
| Coherence (Success) | 0.9092 |
| Coherence (Failure) | 0.7813 |
| Gradient Magnitude (Success) | 0.2820 |
| Gradient Magnitude (Failure) | 0.3538 |
| Activation Separation | 1.1388 |
| Cosine Distance | 0.0445 |
| Clusters | 2,537 |
| Noise Fraction | 0.2281 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,640 | 6.7490 | 0.9824 |
| Neutral  (r = 0) | 63,087 | 0.2545 | 0.9639 |
| Negative (r < 0) | 1,760 | 3.0141 | 0.9380 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5817 |
| Pos vs Negative  | 0.0046 |
| Neutral vs Neg.  | -0.0896 |
| Pos vs Failure   | 0.7485 |
| Neutral vs Fail. | -0.3983 |
| Neg. vs Failure  | 0.2943 |

---

## ep9543_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7561 |
| Coherence (Success) | 0.9118 |
| Coherence (Failure) | 0.8001 |
| Gradient Magnitude (Success) | 0.2624 |
| Gradient Magnitude (Failure) | 0.2486 |
| Activation Separation | 1.0965 |
| Cosine Distance | 0.0404 |
| Clusters | 2,308 |
| Noise Fraction | 0.2434 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,585 | 6.5216 | 0.9845 |
| Neutral  (r = 0) | 61,850 | 0.2987 | 0.9683 |
| Negative (r < 0) | 1,753 | 2.9669 | 0.9437 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6593 |
| Pos vs Negative  | -0.0375 |
| Neutral vs Neg.  | -0.0909 |
| Pos vs Failure   | 0.8165 |
| Neutral vs Fail. | -0.4171 |
| Neg. vs Failure  | 0.2192 |

---

## ep9742_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7140 |
| Coherence (Success) | 0.8877 |
| Coherence (Failure) | 0.7725 |
| Gradient Magnitude (Success) | 0.2317 |
| Gradient Magnitude (Failure) | 0.2201 |
| Activation Separation | 1.0932 |
| Cosine Distance | 0.0406 |
| Clusters | 2,474 |
| Noise Fraction | 0.2064 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,562 | 6.1287 | 0.9818 |
| Neutral  (r = 0) | 63,434 | 0.3155 | 0.9699 |
| Negative (r < 0) | 1,770 | 2.8648 | 0.9267 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6159 |
| Pos vs Negative  | -0.0029 |
| Neutral vs Neg.  | -0.1892 |
| Pos vs Failure   | 0.8149 |
| Neutral vs Fail. | -0.3565 |
| Neg. vs Failure  | 0.1718 |

---

## ep9938_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6794 |
| Coherence (Success) | 0.8914 |
| Coherence (Failure) | 0.7698 |
| Gradient Magnitude (Success) | 0.2316 |
| Gradient Magnitude (Failure) | 0.3004 |
| Activation Separation | 1.3403 |
| Cosine Distance | 0.0610 |
| Clusters | 2,741 |
| Noise Fraction | 0.2147 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,473 | 6.4324 | 0.9808 |
| Neutral  (r = 0) | 63,603 | 0.2386 | 0.9621 |
| Negative (r < 0) | 1,746 | 2.8452 | 0.9184 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6131 |
| Pos vs Negative  | -0.0080 |
| Neutral vs Neg.  | -0.2630 |
| Pos vs Failure   | 0.8272 |
| Neutral vs Fail. | -0.5156 |
| Neg. vs Failure  | 0.1412 |

---

## ep10136_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7465 |
| Coherence (Success) | 0.9274 |
| Coherence (Failure) | 0.8165 |
| Gradient Magnitude (Success) | 0.2805 |
| Gradient Magnitude (Failure) | 0.3034 |
| Activation Separation | 1.2479 |
| Cosine Distance | 0.0523 |
| Clusters | 2,704 |
| Noise Fraction | 0.2130 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,599 | 6.4631 | 0.9806 |
| Neutral  (r = 0) | 63,457 | 0.2184 | 0.9589 |
| Negative (r < 0) | 1,791 | 2.9795 | 0.9321 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4960 |
| Pos vs Negative  | 0.0735 |
| Neutral vs Neg.  | -0.3794 |
| Pos vs Failure   | 0.8699 |
| Neutral vs Fail. | -0.4653 |
| Neg. vs Failure  | 0.1992 |

---

## eat_plant_ep10244_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8603 |
| Coherence (Success) | 0.8920 |
| Coherence (Failure) | 0.8131 |
| Gradient Magnitude (Success) | 0.2314 |
| Gradient Magnitude (Failure) | 0.3021 |
| Activation Separation | 1.2900 |
| Cosine Distance | 0.0581 |
| Clusters | 2,463 |
| Noise Fraction | 0.2569 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,631 | 6.1969 | 0.9833 |
| Neutral  (r = 0) | 64,537 | 0.2596 | 0.9614 |
| Negative (r < 0) | 1,785 | 2.7785 | 0.9200 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4539 |
| Pos vs Negative  | 0.0582 |
| Neutral vs Neg.  | -0.2743 |
| Pos vs Failure   | 0.7784 |
| Neutral vs Fail. | -0.0658 |
| Neg. vs Failure  | 0.1872 |

---

## ep10329_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7450 |
| Coherence (Success) | 0.8526 |
| Coherence (Failure) | 0.7704 |
| Gradient Magnitude (Success) | 0.1868 |
| Gradient Magnitude (Failure) | 0.2342 |
| Activation Separation | 1.2114 |
| Cosine Distance | 0.0490 |
| Clusters | 2,639 |
| Noise Fraction | 0.2062 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,617 | 6.1866 | 0.9812 |
| Neutral  (r = 0) | 66,476 | 0.3140 | 0.9758 |
| Negative (r < 0) | 1,802 | 2.9594 | 0.9388 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5777 |
| Pos vs Negative  | 0.0664 |
| Neutral vs Neg.  | 0.1792 |
| Pos vs Failure   | 0.5735 |
| Neutral vs Fail. | 0.0710 |
| Neg. vs Failure  | 0.4810 |

---

## ep10528_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5484 |
| Coherence (Success) | 0.8866 |
| Coherence (Failure) | 0.7757 |
| Gradient Magnitude (Success) | 0.2337 |
| Gradient Magnitude (Failure) | 0.2251 |
| Activation Separation | 1.1801 |
| Cosine Distance | 0.0454 |
| Clusters | 2,456 |
| Noise Fraction | 0.2240 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,630 | 6.2824 | 0.9818 |
| Neutral  (r = 0) | 63,249 | 0.2976 | 0.9709 |
| Negative (r < 0) | 1,786 | 2.7798 | 0.9315 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6607 |
| Pos vs Negative  | 0.0853 |
| Neutral vs Neg.  | -0.4221 |
| Pos vs Failure   | 0.8040 |
| Neutral vs Fail. | -0.4664 |
| Neg. vs Failure  | 0.2481 |

---

## ep10729_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8413 |
| Coherence (Success) | 0.8983 |
| Coherence (Failure) | 0.7466 |
| Gradient Magnitude (Success) | 0.2112 |
| Gradient Magnitude (Failure) | 0.2378 |
| Activation Separation | 1.1338 |
| Cosine Distance | 0.0452 |
| Clusters | 2,435 |
| Noise Fraction | 0.2287 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,602 | 6.0961 | 0.9833 |
| Neutral  (r = 0) | 65,579 | 0.2233 | 0.9662 |
| Negative (r < 0) | 1,802 | 2.5972 | 0.9232 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6346 |
| Pos vs Negative  | 0.0110 |
| Neutral vs Neg.  | -0.2041 |
| Pos vs Failure   | 0.8939 |
| Neutral vs Fail. | -0.5052 |
| Neg. vs Failure  | 0.1701 |

---

## ep10927_lower9.000_upper11.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8224 |
| Coherence (Success) | 0.8618 |
| Coherence (Failure) | 0.7694 |
| Gradient Magnitude (Success) | 0.2323 |
| Gradient Magnitude (Failure) | 0.2460 |
| Activation Separation | 1.0904 |
| Cosine Distance | 0.0390 |
| Clusters | 2,323 |
| Noise Fraction | 0.2616 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,728 | 5.7570 | 0.9778 |
| Neutral  (r = 0) | 60,707 | 0.2446 | 0.9686 |
| Negative (r < 0) | 1,729 | 2.8520 | 0.9402 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6104 |
| Pos vs Negative  | -0.0173 |
| Neutral vs Neg.  | -0.2698 |
| Pos vs Failure   | 0.8699 |
| Neutral vs Fail. | -0.4653 |
| Neg. vs Failure  | 0.1343 |

---

## ep11126_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8784 |
| Coherence (Success) | 0.8149 |
| Coherence (Failure) | 0.7836 |
| Gradient Magnitude (Success) | 0.2412 |
| Gradient Magnitude (Failure) | 0.3356 |
| Activation Separation | 1.1222 |
| Cosine Distance | 0.0422 |
| Clusters | 2,622 |
| Noise Fraction | 0.2392 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,660 | 5.3297 | 0.9813 |
| Neutral  (r = 0) | 64,755 | 0.2144 | 0.9613 |
| Negative (r < 0) | 1,806 | 2.7990 | 0.9311 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4033 |
| Pos vs Negative  | -0.0538 |
| Neutral vs Neg.  | -0.1341 |
| Pos vs Failure   | 0.7144 |
| Neutral vs Fail. | 0.0290 |
| Neg. vs Failure  | 0.1357 |

---

## ep11314_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7691 |
| Coherence (Success) | 0.8646 |
| Coherence (Failure) | 0.7765 |
| Gradient Magnitude (Success) | 0.1807 |
| Gradient Magnitude (Failure) | 0.2476 |
| Activation Separation | 1.1166 |
| Cosine Distance | 0.0436 |
| Clusters | 2,550 |
| Noise Fraction | 0.2314 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,689 | 5.3482 | 0.9811 |
| Neutral  (r = 0) | 65,838 | 0.2312 | 0.9519 |
| Negative (r < 0) | 1,820 | 2.6237 | 0.9084 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7024 |
| Pos vs Negative  | 0.0059 |
| Neutral vs Neg.  | -0.0609 |
| Pos vs Failure   | 0.7708 |
| Neutral vs Fail. | -0.4767 |
| Neg. vs Failure  | 0.3240 |

---

## ep11503_lower9.450_upper11.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6995 |
| Coherence (Success) | 0.8664 |
| Coherence (Failure) | 0.7787 |
| Gradient Magnitude (Success) | 0.1623 |
| Gradient Magnitude (Failure) | 0.1999 |
| Activation Separation | 1.1768 |
| Cosine Distance | 0.0417 |
| Clusters | 2,509 |
| Noise Fraction | 0.2244 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,864 | 3.9818 | 0.9743 |
| Neutral  (r = 0) | 68,573 | 0.1709 | 0.9620 |
| Negative (r < 0) | 1,907 | 2.5717 | 0.9251 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4618 |
| Pos vs Negative  | -0.0415 |
| Neutral vs Neg.  | -0.2101 |
| Pos vs Failure   | 0.7520 |
| Neutral vs Fail. | -0.2996 |
| Neg. vs Failure  | 0.2371 |

---

## ep11700_lower9.000_upper11.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6543 |
| Coherence (Success) | 0.8062 |
| Coherence (Failure) | 0.6433 |
| Gradient Magnitude (Success) | 0.1464 |
| Gradient Magnitude (Failure) | 0.1465 |
| Activation Separation | 1.2329 |
| Cosine Distance | 0.0498 |
| Clusters | 2,443 |
| Noise Fraction | 0.2009 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,913 | 4.1781 | 0.9767 |
| Neutral  (r = 0) | 70,052 | 0.2042 | 0.9589 |
| Negative (r < 0) | 1,837 | 6.2459 | 0.6171 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6216 |
| Pos vs Negative  | 0.2683 |
| Neutral vs Neg.  | -0.0025 |
| Pos vs Failure   | 0.6764 |
| Neutral vs Fail. | -0.2026 |
| Neg. vs Failure  | 0.5384 |

---

## place_furnace_ep11719_lower9.000_upper11.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3595 |
| Coherence (Success) | 0.8430 |
| Coherence (Failure) | 0.6976 |
| Gradient Magnitude (Success) | 0.1870 |
| Gradient Magnitude (Failure) | 0.2419 |
| Activation Separation | 1.1938 |
| Cosine Distance | 0.0468 |
| Clusters | 2,567 |
| Noise Fraction | 0.2332 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,932 | 4.2818 | 0.9788 |
| Neutral  (r = 0) | 69,413 | 0.1982 | 0.9609 |
| Negative (r < 0) | 1,870 | 2.7195 | 0.9166 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4429 |
| Pos vs Negative  | -0.0130 |
| Neutral vs Neg.  | -0.1434 |
| Pos vs Failure   | 0.5362 |
| Neutral vs Fail. | 0.0155 |
| Neg. vs Failure  | 0.4086 |

---

## ep11898_lower9.900_upper12.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9392 |
| Coherence (Success) | 0.9011 |
| Coherence (Failure) | 0.8394 |
| Gradient Magnitude (Success) | 0.4390 |
| Gradient Magnitude (Failure) | 0.3688 |
| Activation Separation | 1.0356 |
| Cosine Distance | 0.0353 |
| Clusters | 2,392 |
| Noise Fraction | 0.2268 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 4,030 | 4.0759 | 0.9752 |
| Neutral  (r = 0) | 67,267 | 0.3227 | 0.9430 |
| Negative (r < 0) | 1,673 | 3.0608 | 0.9071 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0147 |
| Pos vs Negative  | -0.1532 |
| Neutral vs Neg.  | -0.2974 |
| Pos vs Failure   | 0.6794 |
| Neutral vs Fail. | 0.5985 |
| Neg. vs Failure  | -0.1070 |

---

## collect_iron_ep11997_lower10.000_upper12.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8559 |
| Coherence (Success) | 0.8922 |
| Coherence (Failure) | 0.7445 |
| Gradient Magnitude (Success) | 0.2617 |
| Gradient Magnitude (Failure) | 0.1893 |
| Activation Separation | 0.9026 |
| Cosine Distance | 0.0243 |
| Clusters | 2,642 |
| Noise Fraction | 0.2252 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,977 | 4.0504 | 0.9778 |
| Neutral  (r = 0) | 68,329 | 0.1715 | 0.9425 |
| Negative (r < 0) | 1,742 | 6.2531 | 0.5580 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4049 |
| Pos vs Negative  | 0.2753 |
| Neutral vs Neg.  | -0.0688 |
| Pos vs Failure   | 0.7804 |
| Neutral vs Fail. | -0.1263 |
| Neg. vs Failure  | 0.3675 |

---

## ep12099_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8822 |
| Coherence (Success) | 0.8923 |
| Coherence (Failure) | 0.8105 |
| Gradient Magnitude (Success) | 0.2669 |
| Gradient Magnitude (Failure) | 0.4302 |
| Activation Separation | 1.2531 |
| Cosine Distance | 0.0514 |
| Clusters | 2,682 |
| Noise Fraction | 0.2274 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,777 | 3.7522 | 0.9722 |
| Neutral  (r = 0) | 66,049 | 0.3345 | 0.9554 |
| Negative (r < 0) | 1,806 | 2.7721 | 0.8934 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.2209 |
| Pos vs Negative  | -0.0339 |
| Neutral vs Neg.  | 0.1531 |
| Pos vs Failure   | 0.2949 |
| Neutral vs Fail. | 0.7625 |
| Neg. vs Failure  | 0.3731 |

---

## ep12299_lower9.900_upper12.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5417 |
| Coherence (Success) | 0.8833 |
| Coherence (Failure) | 0.7110 |
| Gradient Magnitude (Success) | 0.1867 |
| Gradient Magnitude (Failure) | 0.1967 |
| Activation Separation | 1.1481 |
| Cosine Distance | 0.0417 |
| Clusters | 2,303 |
| Noise Fraction | 0.2334 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,879 | 3.8098 | 0.9757 |
| Neutral  (r = 0) | 64,715 | 0.1909 | 0.9545 |
| Negative (r < 0) | 1,775 | 2.6125 | 0.9086 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4929 |
| Pos vs Negative  | -0.1840 |
| Neutral vs Neg.  | -0.2152 |
| Pos vs Failure   | 0.5138 |
| Neutral vs Fail. | -0.0202 |
| Neg. vs Failure  | 0.1281 |

---

## ep12498_lower10.000_upper12.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5386 |
| Coherence (Success) | 0.8745 |
| Coherence (Failure) | 0.6681 |
| Gradient Magnitude (Success) | 0.2156 |
| Gradient Magnitude (Failure) | 0.1825 |
| Activation Separation | 1.1914 |
| Cosine Distance | 0.0495 |
| Clusters | 2,689 |
| Noise Fraction | 0.2077 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,957 | 3.3768 | 0.9713 |
| Neutral  (r = 0) | 66,710 | 0.1988 | 0.9547 |
| Negative (r < 0) | 1,804 | 2.6990 | 0.8913 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4510 |
| Pos vs Negative  | -0.1891 |
| Neutral vs Neg.  | -0.3815 |
| Pos vs Failure   | 0.4172 |
| Neutral vs Fail. | -0.0339 |
| Neg. vs Failure  | 0.1316 |

---

## ep12688_lower10.000_upper12.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8671 |
| Coherence (Success) | 0.8873 |
| Coherence (Failure) | 0.8050 |
| Gradient Magnitude (Success) | 0.2531 |
| Gradient Magnitude (Failure) | 0.3720 |
| Activation Separation | 1.0897 |
| Cosine Distance | 0.0371 |
| Clusters | 2,392 |
| Noise Fraction | 0.2352 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,999 | 3.7010 | 0.9738 |
| Neutral  (r = 0) | 65,569 | 0.2509 | 0.9581 |
| Negative (r < 0) | 1,783 | 2.8412 | 0.9234 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1795 |
| Pos vs Negative  | -0.1517 |
| Neutral vs Neg.  | 0.1055 |
| Pos vs Failure   | 0.5041 |
| Neutral vs Fail. | 0.5609 |
| Neg. vs Failure  | 0.2710 |

---

## ep12890_lower10.000_upper12.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9668 |
| Coherence (Success) | 0.8902 |
| Coherence (Failure) | 0.8198 |
| Gradient Magnitude (Success) | 0.2738 |
| Gradient Magnitude (Failure) | 0.2603 |
| Activation Separation | 1.2310 |
| Cosine Distance | 0.0445 |
| Clusters | 2,173 |
| Noise Fraction | 0.2352 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,769 | 3.7107 | 0.9691 |
| Neutral  (r = 0) | 59,847 | 0.2323 | 0.9549 |
| Negative (r < 0) | 1,662 | 2.8487 | 0.8960 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3419 |
| Pos vs Negative  | -0.1474 |
| Neutral vs Neg.  | -0.2509 |
| Pos vs Failure   | 0.5761 |
| Neutral vs Fail. | 0.2634 |
| Neg. vs Failure  | -0.0325 |

---

## ep13079_lower10.000_upper12.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6664 |
| Coherence (Success) | 0.8790 |
| Coherence (Failure) | 0.7552 |
| Gradient Magnitude (Success) | 0.1923 |
| Gradient Magnitude (Failure) | 0.1600 |
| Activation Separation | 1.1551 |
| Cosine Distance | 0.0386 |
| Clusters | 2,330 |
| Noise Fraction | 0.2371 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (0) | — |

### Moment of Reward (SAC)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,885 | 3.6029 | 0.9756 |
| Neutral  (r = 0) | 62,978 | 0.2036 | 0.9593 |
| Negative (r < 0) | 1,741 | 2.8954 | 0.9276 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4452 |
| Pos vs Negative  | -0.1605 |
| Neutral vs Neg.  | -0.1878 |
| Pos vs Failure   | 0.6050 |
| Neutral vs Fail. | -0.0657 |
| Neg. vs Failure  | 0.1931 |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_drink @ ep131 | 131 | 0.9873 | 0.8195 | 0.9250 | 0.0472 | 0.0597 | 0.1025 | 0.0002 | — |
| collect_sapling @ ep140 | 140 | -0.8231 | 0.2769 | 0.8930 | 0.0598 | 0.0654 | 0.8529 | 0.0134 | — |
| collect_wood @ ep149 | 149 | -0.7830 | 0.9561 | 0.9694 | 0.1017 | 0.1305 | 1.3189 | 0.0194 | — |
| defeat_skeleton @ ep156 | 156 | -0.7362 | 0.9971 | 0.9504 | 0.4813 | 0.1334 | 1.6246 | 0.0292 | — |
| defeat_zombie @ ep157 | 157 | -0.7483 | 0.9950 | 0.9366 | 0.2953 | 0.0918 | 1.4012 | 0.0242 | — |
| eat_cow @ ep157 | 157 | -0.7606 | 0.9952 | 0.8990 | 0.2964 | 0.0915 | 1.4270 | 0.0251 | — |
| place_plant @ ep178 | 178 | -0.7809 | 0.8306 | 0.9698 | 0.0711 | 0.1735 | 1.3458 | 0.0295 | — |
| place_table @ ep1890 | 1,890 | 0.5772 | 0.9597 | 0.8804 | 0.2934 | 0.2072 | 0.6121 | 0.0074 | — |
| make_wood_sword @ ep2157 | 2,157 | 0.3961 | 0.8725 | 0.8883 | 0.2305 | 0.2723 | 0.7132 | 0.0120 | — |
| make_wood_pickaxe @ ep2420 | 2,420 | 0.6115 | 0.9401 | 0.9243 | 0.3368 | 0.2935 | 0.7481 | 0.0157 | — |
| wake_up @ ep2589 | 2,589 | 0.7443 | 0.9581 | 0.9151 | 0.4170 | 0.3056 | 0.9647 | 0.0264 | — |
| collect_stone @ ep5154 | 5,154 | 0.9619 | 0.9420 | 0.9333 | 0.4294 | 0.4970 | 0.4273 | 0.0079 | — |
| collect_coal @ ep7002 | 7,002 | 0.8058 | 0.9402 | 0.8436 | 0.2959 | 0.2314 | 0.6286 | 0.0186 | — |
| place_stone @ ep8238 | 8,238 | 0.9013 | 0.9055 | 0.8431 | 0.2856 | 0.3126 | 0.4883 | 0.0093 | — |
| make_stone_pickaxe @ ep8332 | 8,332 | 0.8370 | 0.9105 | 0.8263 | 0.2804 | 0.3238 | 0.5359 | 0.0109 | — |
| make_stone_sword @ ep9479 | 9,479 | 0.6572 | 0.9092 | 0.7813 | 0.2820 | 0.3538 | 1.1388 | 0.0445 | — |
| eat_plant @ ep10244 | 10,244 | 0.8603 | 0.8920 | 0.8131 | 0.2314 | 0.3021 | 1.2900 | 0.0581 | — |
| place_furnace @ ep11719 | 11,719 | 0.3595 | 0.8430 | 0.6976 | 0.1870 | 0.2419 | 1.1938 | 0.0468 | — |
| collect_iron @ ep11997 | 11,997 | 0.8559 | 0.8922 | 0.7445 | 0.2617 | 0.1893 | 0.9026 | 0.0243 | — |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint_step50000 | 288 | 0.9873 | 0.9927 | 0.9755 | 0.3753 | 0.4091 | 0.5322 | 0.0012 | — |
| checkpoint_step100000 | 564 | 0.8526 | 0.9838 | 0.9819 | 0.3348 | 0.2109 | 0.3569 | 0.0012 | — |
| checkpoint_step150000 | 830 | 0.9625 | 0.9706 | 0.9680 | 0.3161 | 0.3028 | 0.5415 | 0.0027 | — |
| checkpoint_step200000 | 1,098 | 0.8097 | 0.8755 | 0.9256 | 0.1289 | 0.1848 | 0.4363 | 0.0025 | — |
| checkpoint_step250000 | 1,366 | 0.6837 | 0.8840 | 0.9399 | 0.1893 | 0.2090 | 0.5740 | 0.0055 | — |
| checkpoint_step300000 | 1,626 | 0.5445 | 0.9136 | 0.9347 | 0.1571 | 0.1887 | 0.5531 | 0.0062 | — |
| checkpoint_step350000 | 1,888 | 0.8381 | 0.9805 | 0.9786 | 0.4954 | 0.3245 | 0.6854 | 0.0089 | — |
| checkpoint_step400000 | 2,153 | 0.6783 | 0.9382 | 0.8807 | 0.2731 | 0.1980 | 0.5787 | 0.0077 | — |
| checkpoint_step450000 | 2,406 | 0.4701 | 0.9626 | 0.9154 | 0.3546 | 0.2832 | 0.7192 | 0.0142 | — |
| checkpoint_step500000 | 2,665 | 0.3715 | 0.9106 | 0.9112 | 0.2315 | 0.3020 | 0.9573 | 0.0263 | — |
| checkpoint_step550000 | 2,917 | 0.5210 | 0.9230 | 0.8824 | 0.3178 | 0.2266 | 1.1419 | 0.0400 | — |
| checkpoint_step600000 | 3,166 | 0.8255 | 0.9426 | 0.8644 | 0.3804 | 0.2363 | 0.6549 | 0.0133 | — |
| checkpoint_step650000 | 3,410 | 0.8550 | 0.9541 | 0.9117 | 0.4337 | 0.3978 | 0.4519 | 0.0067 | — |
| checkpoint_step700000 | 3,654 | 0.8512 | 0.9622 | 0.8795 | 0.3644 | 0.3165 | 0.4631 | 0.0078 | — |
| checkpoint_step750000 | 3,887 | 0.8226 | 0.8894 | 0.7985 | 0.2832 | 0.2183 | 0.4276 | 0.0067 | — |
| checkpoint_step800000 | 4,117 | 0.9124 | 0.9403 | 0.8603 | 0.3546 | 0.2898 | 0.4067 | 0.0067 | — |
| checkpoint_step850000 | 4,348 | 0.8607 | 0.9067 | 0.8958 | 0.2838 | 0.2901 | 0.3126 | 0.0046 | — |
| checkpoint_step900000 | 4,578 | 0.9072 | 0.9518 | 0.8238 | 0.3623 | 0.2512 | 0.3016 | 0.0039 | — |
| checkpoint_step950000 | 4,799 | 0.9221 | 0.9252 | 0.8123 | 0.2892 | 0.2454 | 0.3486 | 0.0053 | — |
| checkpoint_step1000000 | 5,020 | 0.8401 | 0.8993 | 0.9051 | 0.2749 | 0.2423 | 0.4212 | 0.0078 | — |
| checkpoint_step1050000 | 5,246 | 0.7199 | 0.8834 | 0.8050 | 0.1850 | 0.1767 | 0.4025 | 0.0077 | — |
| checkpoint_step1100000 | 5,468 | 0.9352 | 0.9435 | 0.9307 | 0.3932 | 0.3671 | 0.3879 | 0.0076 | — |
| checkpoint_step1150000 | 5,693 | 0.7922 | 0.8925 | 0.8390 | 0.2304 | 0.2510 | 0.3641 | 0.0066 | — |
| checkpoint_step1200000 | 5,908 | 0.8751 | 0.9211 | 0.8638 | 0.2573 | 0.2362 | 0.3133 | 0.0049 | — |
| checkpoint_step1250000 | 6,117 | 0.9358 | 0.9560 | 0.8882 | 0.4556 | 0.4704 | 0.3971 | 0.0077 | — |
| checkpoint_step1300000 | 6,331 | 0.7387 | 0.9003 | 0.8124 | 0.1903 | 0.1779 | 0.4745 | 0.0105 | — |
| checkpoint_step1350000 | 6,540 | 0.9602 | 0.9543 | 0.9046 | 0.4321 | 0.4431 | 0.4650 | 0.0113 | — |
| checkpoint_step1400000 | 6,749 | 0.9119 | 0.9441 | 0.8594 | 0.3263 | 0.2809 | 0.5227 | 0.0132 | — |
| checkpoint_step1450000 | 6,952 | 0.8539 | 0.9353 | 0.8296 | 0.2893 | 0.2029 | 0.5614 | 0.0149 | — |
| checkpoint_step1500000 | 7,161 | 0.9721 | 0.9430 | 0.9290 | 0.5317 | 0.5625 | 0.5449 | 0.0138 | — |
| checkpoint_step1550000 | 7,373 | 0.8052 | 0.9171 | 0.8003 | 0.2440 | 0.1917 | 0.5978 | 0.0143 | — |
| checkpoint_step1600000 | 7,573 | 0.9222 | 0.9259 | 0.8204 | 0.3421 | 0.3364 | 0.6346 | 0.0170 | — |
| checkpoint_step1650000 | 7,779 | 0.9291 | 0.9413 | 0.8551 | 0.4375 | 0.4749 | 0.5623 | 0.0124 | — |
| checkpoint_step1700000 | 7,984 | 0.9253 | 0.8867 | 0.8545 | 0.3231 | 0.3298 | 0.4530 | 0.0086 | — |
| checkpoint_step1750000 | 8,188 | 0.8925 | 0.9052 | 0.7832 | 0.2404 | 0.1740 | 0.5153 | 0.0106 | — |
| checkpoint_step1800000 | 8,382 | 0.9063 | 0.9316 | 0.8701 | 0.3103 | 0.3066 | 0.5151 | 0.0099 | — |
| checkpoint_step1850000 | 8,579 | 0.8615 | 0.9463 | 0.8657 | 0.3343 | 0.2695 | 0.7527 | 0.0197 | — |
| checkpoint_step1900000 | 8,777 | 0.9634 | 0.9402 | 0.9208 | 0.5162 | 0.6350 | 0.7494 | 0.0188 | — |
| checkpoint_step1950000 | 8,969 | 0.5141 | 0.8851 | 0.7195 | 0.2420 | 0.2068 | 0.9956 | 0.0344 | — |
| checkpoint_step2000000 | 9,157 | 0.7608 | 0.9323 | 0.8464 | 0.3090 | 0.3322 | 1.1460 | 0.0448 | — |
| checkpoint_step2050000 | 9,345 | 0.7881 | 0.9050 | 0.8449 | 0.2810 | 0.4530 | 1.1456 | 0.0431 | — |
| checkpoint_step2100000 | 9,543 | 0.7561 | 0.9118 | 0.8001 | 0.2624 | 0.2486 | 1.0965 | 0.0404 | — |
| checkpoint_step2150000 | 9,742 | 0.7140 | 0.8877 | 0.7725 | 0.2317 | 0.2201 | 1.0932 | 0.0406 | — |
| checkpoint_step2200000 | 9,938 | 0.6794 | 0.8914 | 0.7698 | 0.2316 | 0.3004 | 1.3403 | 0.0610 | — |
| checkpoint_step2250000 | 10,136 | 0.7465 | 0.9274 | 0.8165 | 0.2805 | 0.3034 | 1.2479 | 0.0523 | — |
| checkpoint_step2300000 | 10,329 | 0.7450 | 0.8526 | 0.7704 | 0.1868 | 0.2342 | 1.2114 | 0.0490 | — |
| checkpoint_step2350000 | 10,528 | 0.5484 | 0.8866 | 0.7757 | 0.2337 | 0.2251 | 1.1801 | 0.0454 | — |
| checkpoint_step2400000 | 10,729 | 0.8413 | 0.8983 | 0.7466 | 0.2112 | 0.2378 | 1.1338 | 0.0452 | — |
| checkpoint_step2450000 | 10,927 | 0.8224 | 0.8618 | 0.7694 | 0.2323 | 0.2460 | 1.0904 | 0.0390 | — |
| checkpoint_step2500000 | 11,126 | 0.8784 | 0.8149 | 0.7836 | 0.2412 | 0.3356 | 1.1222 | 0.0422 | — |
| checkpoint_step2550000 | 11,314 | 0.7691 | 0.8646 | 0.7765 | 0.1807 | 0.2476 | 1.1166 | 0.0436 | — |
| checkpoint_step2600000 | 11,503 | 0.6995 | 0.8664 | 0.7787 | 0.1623 | 0.1999 | 1.1768 | 0.0417 | — |
| checkpoint_step2650000 | 11,700 | 0.6543 | 0.8062 | 0.6433 | 0.1464 | 0.1465 | 1.2329 | 0.0498 | — |
| checkpoint_step2700000 | 11,898 | 0.9392 | 0.9011 | 0.8394 | 0.4390 | 0.3688 | 1.0356 | 0.0353 | — |
| checkpoint_step2750000 | 12,099 | 0.8822 | 0.8923 | 0.8105 | 0.2669 | 0.4302 | 1.2531 | 0.0514 | — |
| checkpoint_step2800000 | 12,299 | 0.5417 | 0.8833 | 0.7110 | 0.1867 | 0.1967 | 1.1481 | 0.0417 | — |
| checkpoint_step2850000 | 12,498 | 0.5386 | 0.8745 | 0.6681 | 0.2156 | 0.1825 | 1.1914 | 0.0495 | — |
| checkpoint_step2900000 | 12,688 | 0.8671 | 0.8873 | 0.8050 | 0.2531 | 0.3720 | 1.0897 | 0.0371 | — |
| checkpoint_step2950000 | 12,890 | 0.9668 | 0.8902 | 0.8198 | 0.2738 | 0.2603 | 1.2310 | 0.0445 | — |
| checkpoint_step3000000 | 13,079 | 0.6664 | 0.8790 | 0.7552 | 0.1923 | 0.1600 | 1.1551 | 0.0386 | — |
