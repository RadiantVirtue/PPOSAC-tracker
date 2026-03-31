# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 1  
**Total episodes:** 16,885  
**Experiment root:** `ppo_experiment_root\seed_1`  
**Generated:** 2026-03-24 18:00

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.0764 | 0.3413 | 0.0017 | 0.0881 | 0.0357 | 0.1101 | 0.0004 | — |
| wake_up @ ep1 | 1 | 0.0024 | 0.5272 | 0.0726 | 0.0831 | 0.0307 | 0.1286 | 0.0005 | — |
| collect_drink @ ep2 | 2 | -0.0618 | 0.6532 | 0.0339 | 0.0824 | 0.0278 | 0.1106 | 0.0004 | — |
| collect_sapling @ ep2 | 2 | -0.0299 | 0.4805 | -0.0089 | 0.0853 | 0.0254 | 0.1104 | 0.0004 | -0.6547 |
| place_table @ ep3 | 3 | -0.1649 | 0.6488 | 0.1342 | 0.0858 | 0.0283 | 0.1088 | 0.0004 | nan |
| place_plant @ ep5 | 5 | 0.2189 | 0.4367 | 0.0407 | 0.0777 | 0.0426 | 0.1234 | 0.0004 | nan |
| eat_cow @ ep21 | 21 | -0.0271 | 0.8967 | 0.1441 | 0.1329 | 0.0549 | 0.1600 | 0.0003 | nan |
| make_wood_sword @ ep56 | 56 | 0.6410 | 0.9103 | 0.2535 | 0.2837 | 0.1203 | 0.3821 | 0.0003 | -0.2659 |
| make_wood_pickaxe @ ep83 | 83 | 0.8639 | 0.9207 | 0.6938 | 0.2824 | 0.1077 | 0.2873 | 0.0002 | -0.4179 |
| defeat_zombie @ ep265 | 265 | 0.5193 | 0.4068 | 0.7628 | 0.1894 | 0.2536 | 1.0845 | 0.0002 | -0.2791 |
| Step 50,000 | 290 | 0.4436 | 0.0868 | 0.4354 | 0.1573 | 0.3648 | 0.7086 | 0.0002 | -0.5698 |
| collect_stone @ ep306 | 306 | -0.1293 | 0.0790 | 0.7781 | 0.1743 | 0.1969 | 0.9992 | 0.0003 | -0.0698 |
| defeat_skeleton @ ep512 | 512 | -0.8133 | 0.0176 | 0.3134 | 0.1298 | 0.2231 | 1.0048 | 0.0002 | -0.1157 |
| Step 100,000 | 570 | 0.3109 | 0.2723 | 0.0520 | 0.2047 | 0.2271 | 0.9330 | 0.0002 | -0.1741 |
| collect_coal @ ep601 | 601 | -0.1406 | 0.1107 | 0.2783 | 0.2005 | 0.2385 | 0.6883 | 0.0002 | -0.1396 |
| place_stone @ ep601 | 601 | -0.1105 | 0.1041 | -0.0059 | 0.2333 | 0.1427 | 0.6225 | 0.0002 | -0.2611 |
| make_stone_pickaxe @ ep628 | 628 | -0.0210 | 0.1004 | 0.3274 | 0.1333 | 0.3028 | 0.5745 | 0.0003 | -0.1047 |
| Step 150,000 | 841 | -0.3122 | 0.0847 | 0.5356 | 0.1324 | 0.3725 | 0.8318 | 0.0007 | -0.2443 |
| Step 200,000 | 1,111 | -0.1803 | 0.0324 | 0.5524 | 0.1517 | 0.3294 | 0.5286 | 0.0003 | -0.2791 |
| Step 250,000 | 1,389 | -0.4295 | 0.1421 | 0.5146 | 0.1136 | 0.2795 | 0.6767 | 0.0006 | -0.1745 |
| make_stone_sword @ ep1482 | 1,482 | -0.3215 | 0.1420 | 0.0434 | 0.1563 | 0.1457 | 0.8883 | 0.0009 | -0.1745 |
| Step 300,000 | 1,658 | -0.7052 | 0.2533 | 0.2482 | 0.1420 | 0.3517 | 0.6653 | 0.0007 | -0.1396 |
| Step 350,000 | 1,930 | 0.0104 | 0.1886 | 0.4089 | 0.1093 | 0.2527 | 0.8907 | 0.0021 | 0.0000 |
| place_furnace @ ep2196 | 2,196 | 0.4665 | 0.2188 | 0.0594 | 0.1672 | 0.1417 | 0.9477 | 0.0028 | 0.0000 |
| Step 400,000 | 2,203 | 0.5945 | 0.2227 | 0.3188 | 0.1405 | 0.2174 | 0.8892 | 0.0020 | -0.1396 |
| Step 450,000 | 2,474 | 0.2218 | -0.0073 | 0.0348 | 0.1244 | 0.2375 | 1.1174 | 0.0033 | -0.1396 |
| Step 500,000 | 2,746 | -0.3237 | 0.1578 | 0.2505 | 0.1457 | 0.2274 | 1.2832 | 0.0028 | -0.1396 |
| Step 550,000 | 2,994 | 0.5768 | 0.2253 | 0.3746 | 0.1618 | 0.2831 | 1.3402 | 0.0050 | -0.2094 |
| Step 600,000 | 3,250 | 0.7671 | 0.4675 | 0.4173 | 0.2727 | 0.3399 | 2.2119 | 0.0150 | -0.1108 |
| Step 650,000 | 3,513 | 0.6180 | 0.2194 | 0.3244 | 0.1707 | 0.3033 | 2.5758 | 0.0184 | -0.1292 |
| Step 700,000 | 3,779 | 0.3251 | 0.0940 | 0.0089 | 0.1746 | 0.2221 | 3.2783 | 0.0197 | -0.1468 |
| Step 750,000 | 4,036 | 0.5970 | 0.1525 | 0.0884 | 0.1500 | 0.1822 | 2.5115 | 0.0119 | 0.0000 |
| Step 800,000 | 4,292 | 0.2161 | 0.0869 | 0.0227 | 0.1197 | 0.1253 | 2.9597 | 0.0158 | -0.0349 |
| Step 850,000 | 4,557 | 0.7397 | 0.3354 | 0.4060 | 0.2503 | 0.3858 | 3.1373 | 0.0243 | -0.0554 |
| Step 900,000 | 4,815 | 0.3285 | 0.1598 | 0.1435 | 0.1697 | 0.2034 | 3.2333 | 0.0168 | -0.0349 |
| Step 950,000 | 5,083 | 0.5856 | 0.2422 | 0.1556 | 0.2994 | 0.5337 | 3.2284 | 0.0177 | -0.0923 |
| Step 1,000,000 | 5,339 | 0.7568 | 0.3222 | 0.1917 | 0.2013 | 0.2567 | 3.4388 | 0.0281 | -0.0685 |
| Step 1,050,000 | 5,610 | 0.2045 | 0.0861 | -0.0071 | 0.1430 | 0.1273 | 3.2802 | 0.0180 | -0.0923 |
| Step 1,100,000 | 5,865 | 0.3577 | 0.0091 | 0.0059 | 0.1050 | 0.1217 | 4.1131 | 0.0234 | -0.1108 |
| Step 1,150,000 | 6,125 | 0.1880 | 0.0986 | 0.0553 | 0.1256 | 0.1685 | 4.2865 | 0.0316 | -0.0098 |
| Step 1,200,000 | 6,386 | 0.1206 | 0.0354 | 0.0077 | 0.1741 | 0.2281 | 4.3535 | 0.0279 | 0.0698 |
| Step 1,250,000 | 6,643 | 0.5407 | 0.3234 | 0.1675 | 0.2373 | 0.2616 | 4.2952 | 0.0265 | -0.0185 |
| Step 1,300,000 | 6,899 | 0.4888 | 0.0474 | -0.0078 | 0.1200 | 0.1330 | 3.8932 | 0.0270 | 0.0185 |
| Step 1,350,000 | 7,160 | 0.1066 | 0.0741 | 0.0516 | 0.1294 | 0.1655 | 3.9415 | 0.0277 | 0.0185 |
| Step 1,400,000 | 7,421 | 0.6411 | 0.1733 | 0.1113 | 0.1858 | 0.2039 | 3.4932 | 0.0195 | 0.0739 |
| Step 1,450,000 | 7,682 | 0.1535 | 0.0486 | 0.0381 | 0.1679 | 0.2398 | 3.6155 | 0.0251 | 0.1292 |
| Step 1,500,000 | 7,931 | 0.7460 | 0.1584 | 0.0625 | 0.2070 | 0.2102 | 3.5872 | 0.0203 | 0.0369 |
| Step 1,550,000 | 8,186 | 0.4243 | 0.1551 | 0.0598 | 0.2178 | 0.2185 | 3.8553 | 0.0186 | 0.0923 |
| Step 1,600,000 | 8,432 | 0.5287 | 0.2651 | 0.1337 | 0.2002 | 0.2383 | 3.9360 | 0.0316 | 0.2094 |
| Step 1,650,000 | 8,686 | 0.3776 | 0.1267 | 0.0949 | 0.1688 | 0.2055 | 3.6363 | 0.0224 | 0.3140 |
| Step 1,700,000 | 8,943 | 0.4419 | 0.0304 | 0.0826 | 0.2275 | 0.3219 | 4.4494 | 0.0234 | 0.0923 |
| Step 1,750,000 | 9,187 | 0.6159 | 0.1033 | 0.0913 | 0.1553 | 0.2018 | 4.9887 | 0.0329 | 0.0369 |
| Step 1,800,000 | 9,439 | -0.0560 | 0.0954 | 0.0283 | 0.1798 | 0.1761 | 4.4295 | 0.0259 | -0.0685 |
| Step 1,850,000 | 9,700 | 0.3838 | 0.1162 | 0.0612 | 0.1879 | 0.2127 | 4.0413 | 0.0191 | 0.0369 |
| Step 1,900,000 | 9,957 | 0.2905 | 0.0969 | 0.0198 | 0.1641 | 0.1815 | 4.3567 | 0.0293 | 0.2443 |
| Step 1,950,000 | 10,219 | -0.1528 | 0.1292 | 0.0123 | 0.2647 | 0.2488 | 4.0464 | 0.0242 | 0.2443 |
| Step 2,000,000 | 10,465 | 0.1885 | 0.0304 | 0.0253 | 0.1331 | 0.1924 | 4.3853 | 0.0260 | 0.3140 |
| eat_plant @ ep10599 | 10,599 | 0.3757 | 0.1194 | 0.0162 | 0.1607 | 0.2352 | 4.2313 | 0.0272 | 0.2791 |
| Step 2,050,000 | 10,699 | 0.2391 | 0.0310 | 0.0110 | 0.1143 | 0.1618 | 4.3927 | 0.0301 | 0.2443 |
| Step 2,100,000 | 10,940 | 0.2928 | -0.0009 | 0.0718 | 0.1020 | 0.1943 | 3.9084 | 0.0237 | 0.1477 |
| Step 2,150,000 | 11,192 | 0.4901 | 0.1313 | 0.0753 | 0.2358 | 0.3613 | 3.6750 | 0.0204 | 0.2791 |
| Step 2,200,000 | 11,444 | 0.7249 | 0.3443 | 0.1340 | 0.2813 | 0.3513 | 4.2718 | 0.0210 | 0.2443 |
| Step 2,250,000 | 11,704 | 0.6994 | 0.2234 | 0.1444 | 0.2387 | 0.3276 | 4.3943 | 0.0226 | 0.2791 |
| Step 2,300,000 | 11,946 | 0.6320 | 0.0929 | 0.0335 | 0.1692 | 0.2017 | 3.3700 | 0.0157 | 0.1662 |
| Step 2,350,000 | 12,193 | 0.3592 | 0.0944 | 0.1033 | 0.1612 | 0.2202 | 3.7535 | 0.0188 | 0.2791 |
| Step 2,400,000 | 12,436 | -0.0389 | 0.0662 | 0.0399 | 0.2188 | 0.3200 | 5.2122 | 0.0356 | 0.3140 |
| Step 2,450,000 | 12,667 | 0.1266 | 0.1152 | -0.0213 | 0.1569 | 0.1424 | 3.7629 | 0.0164 | 0.2443 |
| Step 2,500,000 | 12,906 | 0.0376 | 0.0650 | 0.0698 | 0.1539 | 0.2306 | 5.2738 | 0.0319 | 0.3489 |
| Step 2,550,000 | 13,146 | 0.1349 | 0.1786 | 0.0570 | 0.2035 | 0.2144 | 4.6839 | 0.0313 | 0.3140 |
| Step 2,600,000 | 13,379 | 0.0522 | 0.0045 | 0.0170 | 0.1159 | 0.1841 | 4.2460 | 0.0223 | 0.2031 |
| Step 2,650,000 | 13,610 | 0.3510 | 0.1117 | -0.0150 | 0.2604 | 0.2718 | 3.7915 | 0.0186 | 0.3140 |
| Step 2,700,000 | 13,839 | 0.2488 | 0.0457 | -0.0110 | 0.1441 | 0.1440 | 3.7545 | 0.0183 | 0.1292 |
| Step 2,750,000 | 14,071 | 0.4804 | 0.0965 | 0.0694 | 0.1797 | 0.2239 | 3.5066 | 0.0205 | 0.2216 |
| Step 2,800,000 | 14,301 | 0.4222 | 0.2225 | 0.1180 | 0.2077 | 0.2856 | 3.8227 | 0.0254 | 0.1846 |
| Step 2,850,000 | 14,521 | 0.2454 | 0.2136 | 0.0398 | 0.1999 | 0.1909 | 4.1124 | 0.0292 | 0.1477 |
| Step 2,900,000 | 14,753 | 0.3565 | 0.1043 | 0.0334 | 0.2247 | 0.2801 | 4.0283 | 0.0289 | 0.0554 |
| Step 2,950,000 | 14,970 | 0.7458 | 0.2180 | 0.0643 | 0.2250 | 0.2472 | 3.7517 | 0.0308 | 0.1477 |
| Step 3,000,000 | 15,208 | 0.6507 | 0.1905 | 0.1743 | 0.2293 | 0.3444 | 3.9948 | 0.0275 | -0.0391 |
| collect_iron @ ep16885 | 16,885 | 0.4463 | 0.0857 | 0.0769 | 0.2347 | 0.2850 | 4.1343 | 0.0258 | 0.1662 |

---

## collect_wood_ep1_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0764 |
| Coherence (Success) | 0.3413 |
| Coherence (Failure) | 0.0017 |
| Gradient Magnitude (Success) | 0.0881 |
| Gradient Magnitude (Failure) | 0.0357 |
| Activation Separation | 0.1101 |
| Cosine Distance | 0.0004 |
| Clusters | 766 |
| Noise Fraction | 0.2634 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## wake_up_ep1_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0024 |
| Coherence (Success) | 0.5272 |
| Coherence (Failure) | 0.0726 |
| Gradient Magnitude (Success) | 0.0831 |
| Gradient Magnitude (Failure) | 0.0307 |
| Activation Separation | 0.1286 |
| Cosine Distance | 0.0005 |
| Clusters | 1,465 |
| Noise Fraction | 0.2480 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## collect_drink_ep2_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0618 |
| Coherence (Success) | 0.6532 |
| Coherence (Failure) | 0.0339 |
| Gradient Magnitude (Success) | 0.0824 |
| Gradient Magnitude (Failure) | 0.0278 |
| Activation Separation | 0.1106 |
| Cosine Distance | 0.0004 |
| Clusters | 1,499 |
| Noise Fraction | 0.2430 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## collect_sapling_ep2_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0299 |
| Coherence (Success) | 0.4805 |
| Coherence (Failure) | -0.0089 |
| Gradient Magnitude (Success) | 0.0853 |
| Gradient Magnitude (Failure) | 0.0254 |
| Activation Separation | 0.1104 |
| Cosine Distance | 0.0004 |
| Clusters | 1,440 |
| Noise Fraction | 0.2476 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep3_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1649 |
| Coherence (Success) | 0.6488 |
| Coherence (Failure) | 0.1342 |
| Gradient Magnitude (Success) | 0.0858 |
| Gradient Magnitude (Failure) | 0.0283 |
| Activation Separation | 0.1088 |
| Cosine Distance | 0.0004 |
| Clusters | 1,499 |
| Noise Fraction | 0.2312 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep5_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2189 |
| Coherence (Success) | 0.4367 |
| Coherence (Failure) | 0.0407 |
| Gradient Magnitude (Success) | 0.0777 |
| Gradient Magnitude (Failure) | 0.0426 |
| Activation Separation | 0.1234 |
| Cosine Distance | 0.0004 |
| Clusters | 790 |
| Noise Fraction | 0.2487 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep21_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0271 |
| Coherence (Success) | 0.8967 |
| Coherence (Failure) | 0.1441 |
| Gradient Magnitude (Success) | 0.1329 |
| Gradient Magnitude (Failure) | 0.0549 |
| Activation Separation | 0.1600 |
| Cosine Distance | 0.0003 |
| Clusters | 1,508 |
| Noise Fraction | 0.2454 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep56_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6410 |
| Coherence (Success) | 0.9103 |
| Coherence (Failure) | 0.2535 |
| Gradient Magnitude (Success) | 0.2837 |
| Gradient Magnitude (Failure) | 0.1203 |
| Activation Separation | 0.3821 |
| Cosine Distance | 0.0003 |
| Clusters | 1,773 |
| Noise Fraction | 0.2172 |
| RSA Alignment (ρ) | -0.2659 |
| RSA Stimuli (5) | Coal, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep83_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8639 |
| Coherence (Success) | 0.9207 |
| Coherence (Failure) | 0.6938 |
| Gradient Magnitude (Success) | 0.2824 |
| Gradient Magnitude (Failure) | 0.1077 |
| Activation Separation | 0.2873 |
| Cosine Distance | 0.0002 |
| Clusters | 1,769 |
| Noise Fraction | 0.2297 |
| RSA Alignment (ρ) | -0.4179 |
| RSA Stimuli (5) | Coal, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep265_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5193 |
| Coherence (Success) | 0.4068 |
| Coherence (Failure) | 0.7628 |
| Gradient Magnitude (Success) | 0.1894 |
| Gradient Magnitude (Failure) | 0.2536 |
| Activation Separation | 1.0845 |
| Cosine Distance | 0.0002 |
| Clusters | 1,524 |
| Noise Fraction | 0.2913 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep290_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4436 |
| Coherence (Success) | 0.0868 |
| Coherence (Failure) | 0.4354 |
| Gradient Magnitude (Success) | 0.1573 |
| Gradient Magnitude (Failure) | 0.3648 |
| Activation Separation | 0.7086 |
| Cosine Distance | 0.0002 |
| Clusters | 799 |
| Noise Fraction | 0.2650 |
| RSA Alignment (ρ) | -0.5698 |
| RSA Stimuli (5) | Coal, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep306_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1293 |
| Coherence (Success) | 0.0790 |
| Coherence (Failure) | 0.7781 |
| Gradient Magnitude (Success) | 0.1743 |
| Gradient Magnitude (Failure) | 0.1969 |
| Activation Separation | 0.9992 |
| Cosine Distance | 0.0003 |
| Clusters | 1,474 |
| Noise Fraction | 0.2589 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep512_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.8133 |
| Coherence (Success) | 0.0176 |
| Coherence (Failure) | 0.3134 |
| Gradient Magnitude (Success) | 0.1298 |
| Gradient Magnitude (Failure) | 0.2231 |
| Activation Separation | 1.0048 |
| Cosine Distance | 0.0002 |
| Clusters | 1,766 |
| Noise Fraction | 0.2548 |
| RSA Alignment (ρ) | -0.1157 |
| RSA Stimuli (6) | Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep570_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3109 |
| Coherence (Success) | 0.2723 |
| Coherence (Failure) | 0.0520 |
| Gradient Magnitude (Success) | 0.2047 |
| Gradient Magnitude (Failure) | 0.2271 |
| Activation Separation | 0.9330 |
| Cosine Distance | 0.0002 |
| Clusters | 1,668 |
| Noise Fraction | 0.2556 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep601_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1406 |
| Coherence (Success) | 0.1107 |
| Coherence (Failure) | 0.2783 |
| Gradient Magnitude (Success) | 0.2005 |
| Gradient Magnitude (Failure) | 0.2385 |
| Activation Separation | 0.6883 |
| Cosine Distance | 0.0002 |
| Clusters | 1,854 |
| Noise Fraction | 0.2374 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep601_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1105 |
| Coherence (Success) | 0.1041 |
| Coherence (Failure) | -0.0059 |
| Gradient Magnitude (Success) | 0.2333 |
| Gradient Magnitude (Failure) | 0.1427 |
| Activation Separation | 0.6225 |
| Cosine Distance | 0.0002 |
| Clusters | 977 |
| Noise Fraction | 0.2044 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep628_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0210 |
| Coherence (Success) | 0.1004 |
| Coherence (Failure) | 0.3274 |
| Gradient Magnitude (Success) | 0.1333 |
| Gradient Magnitude (Failure) | 0.3028 |
| Activation Separation | 0.5745 |
| Cosine Distance | 0.0003 |
| Clusters | 1,812 |
| Noise Fraction | 0.2462 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep841_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3122 |
| Coherence (Success) | 0.0847 |
| Coherence (Failure) | 0.5356 |
| Gradient Magnitude (Success) | 0.1324 |
| Gradient Magnitude (Failure) | 0.3725 |
| Activation Separation | 0.8318 |
| Cosine Distance | 0.0007 |
| Clusters | 1,773 |
| Noise Fraction | 0.2173 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1111_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1803 |
| Coherence (Success) | 0.0324 |
| Coherence (Failure) | 0.5524 |
| Gradient Magnitude (Success) | 0.1517 |
| Gradient Magnitude (Failure) | 0.3294 |
| Activation Separation | 0.5286 |
| Cosine Distance | 0.0003 |
| Clusters | 1,777 |
| Noise Fraction | 0.2417 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1389_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4295 |
| Coherence (Success) | 0.1421 |
| Coherence (Failure) | 0.5146 |
| Gradient Magnitude (Success) | 0.1136 |
| Gradient Magnitude (Failure) | 0.2795 |
| Activation Separation | 0.6767 |
| Cosine Distance | 0.0006 |
| Clusters | 1,800 |
| Noise Fraction | 0.2474 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep1482_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3215 |
| Coherence (Success) | 0.1420 |
| Coherence (Failure) | 0.0434 |
| Gradient Magnitude (Success) | 0.1563 |
| Gradient Magnitude (Failure) | 0.1457 |
| Activation Separation | 0.8883 |
| Cosine Distance | 0.0009 |
| Clusters | 848 |
| Noise Fraction | 0.2703 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1658_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7052 |
| Coherence (Success) | 0.2533 |
| Coherence (Failure) | 0.2482 |
| Gradient Magnitude (Success) | 0.1420 |
| Gradient Magnitude (Failure) | 0.3517 |
| Activation Separation | 0.6653 |
| Cosine Distance | 0.0007 |
| Clusters | 1,883 |
| Noise Fraction | 0.2452 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1930_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0104 |
| Coherence (Success) | 0.1886 |
| Coherence (Failure) | 0.4089 |
| Gradient Magnitude (Success) | 0.1093 |
| Gradient Magnitude (Failure) | 0.2527 |
| Activation Separation | 0.8907 |
| Cosine Distance | 0.0021 |
| Clusters | 1,686 |
| Noise Fraction | 0.2759 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2196_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4665 |
| Coherence (Success) | 0.2188 |
| Coherence (Failure) | 0.0594 |
| Gradient Magnitude (Success) | 0.1672 |
| Gradient Magnitude (Failure) | 0.1417 |
| Activation Separation | 0.9477 |
| Cosine Distance | 0.0028 |
| Clusters | 1,518 |
| Noise Fraction | 0.2876 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2203_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5945 |
| Coherence (Success) | 0.2227 |
| Coherence (Failure) | 0.3188 |
| Gradient Magnitude (Success) | 0.1405 |
| Gradient Magnitude (Failure) | 0.2174 |
| Activation Separation | 0.8892 |
| Cosine Distance | 0.0020 |
| Clusters | 1,564 |
| Noise Fraction | 0.2648 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2474_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2218 |
| Coherence (Success) | -0.0073 |
| Coherence (Failure) | 0.0348 |
| Gradient Magnitude (Success) | 0.1244 |
| Gradient Magnitude (Failure) | 0.2375 |
| Activation Separation | 1.1174 |
| Cosine Distance | 0.0033 |
| Clusters | 913 |
| Noise Fraction | 0.2196 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2746_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3237 |
| Coherence (Success) | 0.1578 |
| Coherence (Failure) | 0.2505 |
| Gradient Magnitude (Success) | 0.1457 |
| Gradient Magnitude (Failure) | 0.2274 |
| Activation Separation | 1.2832 |
| Cosine Distance | 0.0028 |
| Clusters | 1,733 |
| Noise Fraction | 0.2738 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2994_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5768 |
| Coherence (Success) | 0.2253 |
| Coherence (Failure) | 0.3746 |
| Gradient Magnitude (Success) | 0.1618 |
| Gradient Magnitude (Failure) | 0.2831 |
| Activation Separation | 1.3402 |
| Cosine Distance | 0.0050 |
| Clusters | 1,696 |
| Noise Fraction | 0.2962 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3250_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7671 |
| Coherence (Success) | 0.4675 |
| Coherence (Failure) | 0.4173 |
| Gradient Magnitude (Success) | 0.2727 |
| Gradient Magnitude (Failure) | 0.3399 |
| Activation Separation | 2.2119 |
| Cosine Distance | 0.0150 |
| Clusters | 1,413 |
| Noise Fraction | 0.3288 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3513_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6180 |
| Coherence (Success) | 0.2194 |
| Coherence (Failure) | 0.3244 |
| Gradient Magnitude (Success) | 0.1707 |
| Gradient Magnitude (Failure) | 0.3033 |
| Activation Separation | 2.5758 |
| Cosine Distance | 0.0184 |
| Clusters | 1,346 |
| Noise Fraction | 0.3141 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3779_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3251 |
| Coherence (Success) | 0.0940 |
| Coherence (Failure) | 0.0089 |
| Gradient Magnitude (Success) | 0.1746 |
| Gradient Magnitude (Failure) | 0.2221 |
| Activation Separation | 3.2783 |
| Cosine Distance | 0.0197 |
| Clusters | 806 |
| Noise Fraction | 0.2681 |
| RSA Alignment (ρ) | -0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4036_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5970 |
| Coherence (Success) | 0.1525 |
| Coherence (Failure) | 0.0884 |
| Gradient Magnitude (Success) | 0.1500 |
| Gradient Magnitude (Failure) | 0.1822 |
| Activation Separation | 2.5115 |
| Cosine Distance | 0.0119 |
| Clusters | 1,622 |
| Noise Fraction | 0.3102 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4292_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2161 |
| Coherence (Success) | 0.0869 |
| Coherence (Failure) | 0.0227 |
| Gradient Magnitude (Success) | 0.1197 |
| Gradient Magnitude (Failure) | 0.1253 |
| Activation Separation | 2.9597 |
| Cosine Distance | 0.0158 |
| Clusters | 1,877 |
| Noise Fraction | 0.2761 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4557_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7397 |
| Coherence (Success) | 0.3354 |
| Coherence (Failure) | 0.4060 |
| Gradient Magnitude (Success) | 0.2503 |
| Gradient Magnitude (Failure) | 0.3858 |
| Activation Separation | 3.1373 |
| Cosine Distance | 0.0243 |
| Clusters | 1,476 |
| Noise Fraction | 0.3003 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4815_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3285 |
| Coherence (Success) | 0.1598 |
| Coherence (Failure) | 0.1435 |
| Gradient Magnitude (Success) | 0.1697 |
| Gradient Magnitude (Failure) | 0.2034 |
| Activation Separation | 3.2333 |
| Cosine Distance | 0.0168 |
| Clusters | 1,624 |
| Noise Fraction | 0.3091 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5083_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5856 |
| Coherence (Success) | 0.2422 |
| Coherence (Failure) | 0.1556 |
| Gradient Magnitude (Success) | 0.2994 |
| Gradient Magnitude (Failure) | 0.5337 |
| Activation Separation | 3.2284 |
| Cosine Distance | 0.0177 |
| Clusters | 903 |
| Noise Fraction | 0.3032 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5339_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7568 |
| Coherence (Success) | 0.3222 |
| Coherence (Failure) | 0.1917 |
| Gradient Magnitude (Success) | 0.2013 |
| Gradient Magnitude (Failure) | 0.2567 |
| Activation Separation | 3.4388 |
| Cosine Distance | 0.0281 |
| Clusters | 1,474 |
| Noise Fraction | 0.3448 |
| RSA Alignment (ρ) | -0.0685 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5610_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2045 |
| Coherence (Success) | 0.0861 |
| Coherence (Failure) | -0.0071 |
| Gradient Magnitude (Success) | 0.1430 |
| Gradient Magnitude (Failure) | 0.1273 |
| Activation Separation | 3.2802 |
| Cosine Distance | 0.0180 |
| Clusters | 1,471 |
| Noise Fraction | 0.3311 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5865_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3577 |
| Coherence (Success) | 0.0091 |
| Coherence (Failure) | 0.0059 |
| Gradient Magnitude (Success) | 0.1050 |
| Gradient Magnitude (Failure) | 0.1217 |
| Activation Separation | 4.1131 |
| Cosine Distance | 0.0234 |
| Clusters | 1,515 |
| Noise Fraction | 0.3052 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6125_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1880 |
| Coherence (Success) | 0.0986 |
| Coherence (Failure) | 0.0553 |
| Gradient Magnitude (Success) | 0.1256 |
| Gradient Magnitude (Failure) | 0.1685 |
| Activation Separation | 4.2865 |
| Cosine Distance | 0.0316 |
| Clusters | 1,587 |
| Noise Fraction | 0.3029 |
| RSA Alignment (ρ) | -0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6386_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1206 |
| Coherence (Success) | 0.0354 |
| Coherence (Failure) | 0.0077 |
| Gradient Magnitude (Success) | 0.1741 |
| Gradient Magnitude (Failure) | 0.2281 |
| Activation Separation | 4.3535 |
| Cosine Distance | 0.0279 |
| Clusters | 841 |
| Noise Fraction | 0.2990 |
| RSA Alignment (ρ) | 0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6643_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5407 |
| Coherence (Success) | 0.3234 |
| Coherence (Failure) | 0.1675 |
| Gradient Magnitude (Success) | 0.2373 |
| Gradient Magnitude (Failure) | 0.2616 |
| Activation Separation | 4.2952 |
| Cosine Distance | 0.0265 |
| Clusters | 1,477 |
| Noise Fraction | 0.3153 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6899_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4888 |
| Coherence (Success) | 0.0474 |
| Coherence (Failure) | -0.0078 |
| Gradient Magnitude (Success) | 0.1200 |
| Gradient Magnitude (Failure) | 0.1330 |
| Activation Separation | 3.8932 |
| Cosine Distance | 0.0270 |
| Clusters | 1,373 |
| Noise Fraction | 0.3295 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7160_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1066 |
| Coherence (Success) | 0.0741 |
| Coherence (Failure) | 0.0516 |
| Gradient Magnitude (Success) | 0.1294 |
| Gradient Magnitude (Failure) | 0.1655 |
| Activation Separation | 3.9415 |
| Cosine Distance | 0.0277 |
| Clusters | 1,387 |
| Noise Fraction | 0.3317 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7421_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6411 |
| Coherence (Success) | 0.1733 |
| Coherence (Failure) | 0.1113 |
| Gradient Magnitude (Success) | 0.1858 |
| Gradient Magnitude (Failure) | 0.2039 |
| Activation Separation | 3.4932 |
| Cosine Distance | 0.0195 |
| Clusters | 1,345 |
| Noise Fraction | 0.3779 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7682_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1535 |
| Coherence (Success) | 0.0486 |
| Coherence (Failure) | 0.0381 |
| Gradient Magnitude (Success) | 0.1679 |
| Gradient Magnitude (Failure) | 0.2398 |
| Activation Separation | 3.6155 |
| Cosine Distance | 0.0251 |
| Clusters | 753 |
| Noise Fraction | 0.3048 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7931_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7460 |
| Coherence (Success) | 0.1584 |
| Coherence (Failure) | 0.0625 |
| Gradient Magnitude (Success) | 0.2070 |
| Gradient Magnitude (Failure) | 0.2102 |
| Activation Separation | 3.5872 |
| Cosine Distance | 0.0203 |
| Clusters | 1,293 |
| Noise Fraction | 0.3628 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8186_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4243 |
| Coherence (Success) | 0.1551 |
| Coherence (Failure) | 0.0598 |
| Gradient Magnitude (Success) | 0.2178 |
| Gradient Magnitude (Failure) | 0.2185 |
| Activation Separation | 3.8553 |
| Cosine Distance | 0.0186 |
| Clusters | 1,236 |
| Noise Fraction | 0.3298 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8432_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5287 |
| Coherence (Success) | 0.2651 |
| Coherence (Failure) | 0.1337 |
| Gradient Magnitude (Success) | 0.2002 |
| Gradient Magnitude (Failure) | 0.2383 |
| Activation Separation | 3.9360 |
| Cosine Distance | 0.0316 |
| Clusters | 1,459 |
| Noise Fraction | 0.3513 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8686_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3776 |
| Coherence (Success) | 0.1267 |
| Coherence (Failure) | 0.0949 |
| Gradient Magnitude (Success) | 0.1688 |
| Gradient Magnitude (Failure) | 0.2055 |
| Activation Separation | 3.6363 |
| Cosine Distance | 0.0224 |
| Clusters | 1,291 |
| Noise Fraction | 0.3260 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8943_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4419 |
| Coherence (Success) | 0.0304 |
| Coherence (Failure) | 0.0826 |
| Gradient Magnitude (Success) | 0.2275 |
| Gradient Magnitude (Failure) | 0.3219 |
| Activation Separation | 4.4494 |
| Cosine Distance | 0.0234 |
| Clusters | 827 |
| Noise Fraction | 0.2918 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9187_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6159 |
| Coherence (Success) | 0.1033 |
| Coherence (Failure) | 0.0913 |
| Gradient Magnitude (Success) | 0.1553 |
| Gradient Magnitude (Failure) | 0.2018 |
| Activation Separation | 4.9887 |
| Cosine Distance | 0.0329 |
| Clusters | 1,513 |
| Noise Fraction | 0.3217 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9439_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0560 |
| Coherence (Success) | 0.0954 |
| Coherence (Failure) | 0.0283 |
| Gradient Magnitude (Success) | 0.1798 |
| Gradient Magnitude (Failure) | 0.1761 |
| Activation Separation | 4.4295 |
| Cosine Distance | 0.0259 |
| Clusters | 1,430 |
| Noise Fraction | 0.3164 |
| RSA Alignment (ρ) | -0.0685 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9700_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3838 |
| Coherence (Success) | 0.1162 |
| Coherence (Failure) | 0.0612 |
| Gradient Magnitude (Success) | 0.1879 |
| Gradient Magnitude (Failure) | 0.2127 |
| Activation Separation | 4.0413 |
| Cosine Distance | 0.0191 |
| Clusters | 1,327 |
| Noise Fraction | 0.3006 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9957_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2905 |
| Coherence (Success) | 0.0969 |
| Coherence (Failure) | 0.0198 |
| Gradient Magnitude (Success) | 0.1641 |
| Gradient Magnitude (Failure) | 0.1815 |
| Activation Separation | 4.3567 |
| Cosine Distance | 0.0293 |
| Clusters | 1,355 |
| Noise Fraction | 0.3261 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10219_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1528 |
| Coherence (Success) | 0.1292 |
| Coherence (Failure) | 0.0123 |
| Gradient Magnitude (Success) | 0.2647 |
| Gradient Magnitude (Failure) | 0.2488 |
| Activation Separation | 4.0464 |
| Cosine Distance | 0.0242 |
| Clusters | 818 |
| Noise Fraction | 0.2968 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10465_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1885 |
| Coherence (Success) | 0.0304 |
| Coherence (Failure) | 0.0253 |
| Gradient Magnitude (Success) | 0.1331 |
| Gradient Magnitude (Failure) | 0.1924 |
| Activation Separation | 4.3853 |
| Cosine Distance | 0.0260 |
| Clusters | 1,403 |
| Noise Fraction | 0.2623 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep10599_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3757 |
| Coherence (Success) | 0.1194 |
| Coherence (Failure) | 0.0162 |
| Gradient Magnitude (Success) | 0.1607 |
| Gradient Magnitude (Failure) | 0.2352 |
| Activation Separation | 4.2313 |
| Cosine Distance | 0.0272 |
| Clusters | 1,475 |
| Noise Fraction | 0.2890 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10699_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2391 |
| Coherence (Success) | 0.0310 |
| Coherence (Failure) | 0.0110 |
| Gradient Magnitude (Success) | 0.1143 |
| Gradient Magnitude (Failure) | 0.1618 |
| Activation Separation | 4.3927 |
| Cosine Distance | 0.0301 |
| Clusters | 1,420 |
| Noise Fraction | 0.2789 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10940_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2928 |
| Coherence (Success) | -0.0009 |
| Coherence (Failure) | 0.0718 |
| Gradient Magnitude (Success) | 0.1020 |
| Gradient Magnitude (Failure) | 0.1943 |
| Activation Separation | 3.9084 |
| Cosine Distance | 0.0237 |
| Clusters | 1,251 |
| Noise Fraction | 0.2466 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11192_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4901 |
| Coherence (Success) | 0.1313 |
| Coherence (Failure) | 0.0753 |
| Gradient Magnitude (Success) | 0.2358 |
| Gradient Magnitude (Failure) | 0.3613 |
| Activation Separation | 3.6750 |
| Cosine Distance | 0.0204 |
| Clusters | 666 |
| Noise Fraction | 0.2440 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11444_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7249 |
| Coherence (Success) | 0.3443 |
| Coherence (Failure) | 0.1340 |
| Gradient Magnitude (Success) | 0.2813 |
| Gradient Magnitude (Failure) | 0.3513 |
| Activation Separation | 4.2718 |
| Cosine Distance | 0.0210 |
| Clusters | 1,141 |
| Noise Fraction | 0.3206 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11704_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6994 |
| Coherence (Success) | 0.2234 |
| Coherence (Failure) | 0.1444 |
| Gradient Magnitude (Success) | 0.2387 |
| Gradient Magnitude (Failure) | 0.3276 |
| Activation Separation | 4.3943 |
| Cosine Distance | 0.0226 |
| Clusters | 1,379 |
| Noise Fraction | 0.2883 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11946_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6320 |
| Coherence (Success) | 0.0929 |
| Coherence (Failure) | 0.0335 |
| Gradient Magnitude (Success) | 0.1692 |
| Gradient Magnitude (Failure) | 0.2017 |
| Activation Separation | 3.3700 |
| Cosine Distance | 0.0157 |
| Clusters | 1,359 |
| Noise Fraction | 0.3009 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12193_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3592 |
| Coherence (Success) | 0.0944 |
| Coherence (Failure) | 0.1033 |
| Gradient Magnitude (Success) | 0.1612 |
| Gradient Magnitude (Failure) | 0.2202 |
| Activation Separation | 3.7535 |
| Cosine Distance | 0.0188 |
| Clusters | 1,459 |
| Noise Fraction | 0.3207 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12436_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0389 |
| Coherence (Success) | 0.0662 |
| Coherence (Failure) | 0.0399 |
| Gradient Magnitude (Success) | 0.2188 |
| Gradient Magnitude (Failure) | 0.3200 |
| Activation Separation | 5.2122 |
| Cosine Distance | 0.0356 |
| Clusters | 817 |
| Noise Fraction | 0.3035 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12667_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1266 |
| Coherence (Success) | 0.1152 |
| Coherence (Failure) | -0.0213 |
| Gradient Magnitude (Success) | 0.1569 |
| Gradient Magnitude (Failure) | 0.1424 |
| Activation Separation | 3.7629 |
| Cosine Distance | 0.0164 |
| Clusters | 1,456 |
| Noise Fraction | 0.3102 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12906_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0376 |
| Coherence (Success) | 0.0650 |
| Coherence (Failure) | 0.0698 |
| Gradient Magnitude (Success) | 0.1539 |
| Gradient Magnitude (Failure) | 0.2306 |
| Activation Separation | 5.2738 |
| Cosine Distance | 0.0319 |
| Clusters | 1,363 |
| Noise Fraction | 0.3256 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13146_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1349 |
| Coherence (Success) | 0.1786 |
| Coherence (Failure) | 0.0570 |
| Gradient Magnitude (Success) | 0.2035 |
| Gradient Magnitude (Failure) | 0.2144 |
| Activation Separation | 4.6839 |
| Cosine Distance | 0.0313 |
| Clusters | 1,173 |
| Noise Fraction | 0.3102 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13379_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0522 |
| Coherence (Success) | 0.0045 |
| Coherence (Failure) | 0.0170 |
| Gradient Magnitude (Success) | 0.1159 |
| Gradient Magnitude (Failure) | 0.1841 |
| Activation Separation | 4.2460 |
| Cosine Distance | 0.0223 |
| Clusters | 1,271 |
| Noise Fraction | 0.3321 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13610_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3510 |
| Coherence (Success) | 0.1117 |
| Coherence (Failure) | -0.0150 |
| Gradient Magnitude (Success) | 0.2604 |
| Gradient Magnitude (Failure) | 0.2718 |
| Activation Separation | 3.7915 |
| Cosine Distance | 0.0186 |
| Clusters | 711 |
| Noise Fraction | 0.3098 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13839_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2488 |
| Coherence (Success) | 0.0457 |
| Coherence (Failure) | -0.0110 |
| Gradient Magnitude (Success) | 0.1441 |
| Gradient Magnitude (Failure) | 0.1440 |
| Activation Separation | 3.7545 |
| Cosine Distance | 0.0183 |
| Clusters | 1,299 |
| Noise Fraction | 0.3562 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14071_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4804 |
| Coherence (Success) | 0.0965 |
| Coherence (Failure) | 0.0694 |
| Gradient Magnitude (Success) | 0.1797 |
| Gradient Magnitude (Failure) | 0.2239 |
| Activation Separation | 3.5066 |
| Cosine Distance | 0.0205 |
| Clusters | 1,473 |
| Noise Fraction | 0.3781 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14301_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4222 |
| Coherence (Success) | 0.2225 |
| Coherence (Failure) | 0.1180 |
| Gradient Magnitude (Success) | 0.2077 |
| Gradient Magnitude (Failure) | 0.2856 |
| Activation Separation | 3.8227 |
| Cosine Distance | 0.0254 |
| Clusters | 1,283 |
| Noise Fraction | 0.3272 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14521_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2454 |
| Coherence (Success) | 0.2136 |
| Coherence (Failure) | 0.0398 |
| Gradient Magnitude (Success) | 0.1999 |
| Gradient Magnitude (Failure) | 0.1909 |
| Activation Separation | 4.1124 |
| Cosine Distance | 0.0292 |
| Clusters | 1,392 |
| Noise Fraction | 0.3128 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14753_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3565 |
| Coherence (Success) | 0.1043 |
| Coherence (Failure) | 0.0334 |
| Gradient Magnitude (Success) | 0.2247 |
| Gradient Magnitude (Failure) | 0.2801 |
| Activation Separation | 4.0283 |
| Cosine Distance | 0.0289 |
| Clusters | 785 |
| Noise Fraction | 0.3233 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14970_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7458 |
| Coherence (Success) | 0.2180 |
| Coherence (Failure) | 0.0643 |
| Gradient Magnitude (Success) | 0.2250 |
| Gradient Magnitude (Failure) | 0.2472 |
| Activation Separation | 3.7517 |
| Cosine Distance | 0.0308 |
| Clusters | 1,361 |
| Noise Fraction | 0.3565 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15208_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6507 |
| Coherence (Success) | 0.1905 |
| Coherence (Failure) | 0.1743 |
| Gradient Magnitude (Success) | 0.2293 |
| Gradient Magnitude (Failure) | 0.3444 |
| Activation Separation | 3.9948 |
| Cosine Distance | 0.0275 |
| Clusters | 1,492 |
| Noise Fraction | 0.3365 |
| RSA Alignment (ρ) | -0.0391 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep16885_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4463 |
| Coherence (Success) | 0.0857 |
| Coherence (Failure) | 0.0769 |
| Gradient Magnitude (Success) | 0.2347 |
| Gradient Magnitude (Failure) | 0.2850 |
| Activation Separation | 4.1343 |
| Cosine Distance | 0.0258 |
| Clusters | 781 |
| Noise Fraction | 0.3037 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.0764 | 0.3413 | 0.0017 | 0.0881 | 0.0357 | 0.1101 | 0.0004 | — |
| wake_up @ ep1 | 1 | 0.0024 | 0.5272 | 0.0726 | 0.0831 | 0.0307 | 0.1286 | 0.0005 | — |
| collect_drink @ ep2 | 2 | -0.0618 | 0.6532 | 0.0339 | 0.0824 | 0.0278 | 0.1106 | 0.0004 | — |
| collect_sapling @ ep2 | 2 | -0.0299 | 0.4805 | -0.0089 | 0.0853 | 0.0254 | 0.1104 | 0.0004 | -0.6547 |
| place_table @ ep3 | 3 | -0.1649 | 0.6488 | 0.1342 | 0.0858 | 0.0283 | 0.1088 | 0.0004 | nan |
| place_plant @ ep5 | 5 | 0.2189 | 0.4367 | 0.0407 | 0.0777 | 0.0426 | 0.1234 | 0.0004 | nan |
| eat_cow @ ep21 | 21 | -0.0271 | 0.8967 | 0.1441 | 0.1329 | 0.0549 | 0.1600 | 0.0003 | nan |
| make_wood_sword @ ep56 | 56 | 0.6410 | 0.9103 | 0.2535 | 0.2837 | 0.1203 | 0.3821 | 0.0003 | -0.2659 |
| make_wood_pickaxe @ ep83 | 83 | 0.8639 | 0.9207 | 0.6938 | 0.2824 | 0.1077 | 0.2873 | 0.0002 | -0.4179 |
| defeat_zombie @ ep265 | 265 | 0.5193 | 0.4068 | 0.7628 | 0.1894 | 0.2536 | 1.0845 | 0.0002 | -0.2791 |
| collect_stone @ ep306 | 306 | -0.1293 | 0.0790 | 0.7781 | 0.1743 | 0.1969 | 0.9992 | 0.0003 | -0.0698 |
| defeat_skeleton @ ep512 | 512 | -0.8133 | 0.0176 | 0.3134 | 0.1298 | 0.2231 | 1.0048 | 0.0002 | -0.1157 |
| collect_coal @ ep601 | 601 | -0.1406 | 0.1107 | 0.2783 | 0.2005 | 0.2385 | 0.6883 | 0.0002 | -0.1396 |
| place_stone @ ep601 | 601 | -0.1105 | 0.1041 | -0.0059 | 0.2333 | 0.1427 | 0.6225 | 0.0002 | -0.2611 |
| make_stone_pickaxe @ ep628 | 628 | -0.0210 | 0.1004 | 0.3274 | 0.1333 | 0.3028 | 0.5745 | 0.0003 | -0.1047 |
| make_stone_sword @ ep1482 | 1,482 | -0.3215 | 0.1420 | 0.0434 | 0.1563 | 0.1457 | 0.8883 | 0.0009 | -0.1745 |
| place_furnace @ ep2196 | 2,196 | 0.4665 | 0.2188 | 0.0594 | 0.1672 | 0.1417 | 0.9477 | 0.0028 | 0.0000 |
| eat_plant @ ep10599 | 10,599 | 0.3757 | 0.1194 | 0.0162 | 0.1607 | 0.2352 | 4.2313 | 0.0272 | 0.2791 |
| collect_iron @ ep16885 | 16,885 | 0.4463 | 0.0857 | 0.0769 | 0.2347 | 0.2850 | 4.1343 | 0.0258 | 0.1662 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 290 | 0.4436 | 0.0868 | 0.4354 | 0.1573 | 0.3648 | 0.7086 | 0.0002 | -0.5698 |
| Step 100,000 | 570 | 0.3109 | 0.2723 | 0.0520 | 0.2047 | 0.2271 | 0.9330 | 0.0002 | -0.1741 |
| Step 150,000 | 841 | -0.3122 | 0.0847 | 0.5356 | 0.1324 | 0.3725 | 0.8318 | 0.0007 | -0.2443 |
| Step 200,000 | 1,111 | -0.1803 | 0.0324 | 0.5524 | 0.1517 | 0.3294 | 0.5286 | 0.0003 | -0.2791 |
| Step 250,000 | 1,389 | -0.4295 | 0.1421 | 0.5146 | 0.1136 | 0.2795 | 0.6767 | 0.0006 | -0.1745 |
| Step 300,000 | 1,658 | -0.7052 | 0.2533 | 0.2482 | 0.1420 | 0.3517 | 0.6653 | 0.0007 | -0.1396 |
| Step 350,000 | 1,930 | 0.0104 | 0.1886 | 0.4089 | 0.1093 | 0.2527 | 0.8907 | 0.0021 | 0.0000 |
| Step 400,000 | 2,203 | 0.5945 | 0.2227 | 0.3188 | 0.1405 | 0.2174 | 0.8892 | 0.0020 | -0.1396 |
| Step 450,000 | 2,474 | 0.2218 | -0.0073 | 0.0348 | 0.1244 | 0.2375 | 1.1174 | 0.0033 | -0.1396 |
| Step 500,000 | 2,746 | -0.3237 | 0.1578 | 0.2505 | 0.1457 | 0.2274 | 1.2832 | 0.0028 | -0.1396 |
| Step 550,000 | 2,994 | 0.5768 | 0.2253 | 0.3746 | 0.1618 | 0.2831 | 1.3402 | 0.0050 | -0.2094 |
| Step 600,000 | 3,250 | 0.7671 | 0.4675 | 0.4173 | 0.2727 | 0.3399 | 2.2119 | 0.0150 | -0.1108 |
| Step 650,000 | 3,513 | 0.6180 | 0.2194 | 0.3244 | 0.1707 | 0.3033 | 2.5758 | 0.0184 | -0.1292 |
| Step 700,000 | 3,779 | 0.3251 | 0.0940 | 0.0089 | 0.1746 | 0.2221 | 3.2783 | 0.0197 | -0.1468 |
| Step 750,000 | 4,036 | 0.5970 | 0.1525 | 0.0884 | 0.1500 | 0.1822 | 2.5115 | 0.0119 | 0.0000 |
| Step 800,000 | 4,292 | 0.2161 | 0.0869 | 0.0227 | 0.1197 | 0.1253 | 2.9597 | 0.0158 | -0.0349 |
| Step 850,000 | 4,557 | 0.7397 | 0.3354 | 0.4060 | 0.2503 | 0.3858 | 3.1373 | 0.0243 | -0.0554 |
| Step 900,000 | 4,815 | 0.3285 | 0.1598 | 0.1435 | 0.1697 | 0.2034 | 3.2333 | 0.0168 | -0.0349 |
| Step 950,000 | 5,083 | 0.5856 | 0.2422 | 0.1556 | 0.2994 | 0.5337 | 3.2284 | 0.0177 | -0.0923 |
| Step 1,000,000 | 5,339 | 0.7568 | 0.3222 | 0.1917 | 0.2013 | 0.2567 | 3.4388 | 0.0281 | -0.0685 |
| Step 1,050,000 | 5,610 | 0.2045 | 0.0861 | -0.0071 | 0.1430 | 0.1273 | 3.2802 | 0.0180 | -0.0923 |
| Step 1,100,000 | 5,865 | 0.3577 | 0.0091 | 0.0059 | 0.1050 | 0.1217 | 4.1131 | 0.0234 | -0.1108 |
| Step 1,150,000 | 6,125 | 0.1880 | 0.0986 | 0.0553 | 0.1256 | 0.1685 | 4.2865 | 0.0316 | -0.0098 |
| Step 1,200,000 | 6,386 | 0.1206 | 0.0354 | 0.0077 | 0.1741 | 0.2281 | 4.3535 | 0.0279 | 0.0698 |
| Step 1,250,000 | 6,643 | 0.5407 | 0.3234 | 0.1675 | 0.2373 | 0.2616 | 4.2952 | 0.0265 | -0.0185 |
| Step 1,300,000 | 6,899 | 0.4888 | 0.0474 | -0.0078 | 0.1200 | 0.1330 | 3.8932 | 0.0270 | 0.0185 |
| Step 1,350,000 | 7,160 | 0.1066 | 0.0741 | 0.0516 | 0.1294 | 0.1655 | 3.9415 | 0.0277 | 0.0185 |
| Step 1,400,000 | 7,421 | 0.6411 | 0.1733 | 0.1113 | 0.1858 | 0.2039 | 3.4932 | 0.0195 | 0.0739 |
| Step 1,450,000 | 7,682 | 0.1535 | 0.0486 | 0.0381 | 0.1679 | 0.2398 | 3.6155 | 0.0251 | 0.1292 |
| Step 1,500,000 | 7,931 | 0.7460 | 0.1584 | 0.0625 | 0.2070 | 0.2102 | 3.5872 | 0.0203 | 0.0369 |
| Step 1,550,000 | 8,186 | 0.4243 | 0.1551 | 0.0598 | 0.2178 | 0.2185 | 3.8553 | 0.0186 | 0.0923 |
| Step 1,600,000 | 8,432 | 0.5287 | 0.2651 | 0.1337 | 0.2002 | 0.2383 | 3.9360 | 0.0316 | 0.2094 |
| Step 1,650,000 | 8,686 | 0.3776 | 0.1267 | 0.0949 | 0.1688 | 0.2055 | 3.6363 | 0.0224 | 0.3140 |
| Step 1,700,000 | 8,943 | 0.4419 | 0.0304 | 0.0826 | 0.2275 | 0.3219 | 4.4494 | 0.0234 | 0.0923 |
| Step 1,750,000 | 9,187 | 0.6159 | 0.1033 | 0.0913 | 0.1553 | 0.2018 | 4.9887 | 0.0329 | 0.0369 |
| Step 1,800,000 | 9,439 | -0.0560 | 0.0954 | 0.0283 | 0.1798 | 0.1761 | 4.4295 | 0.0259 | -0.0685 |
| Step 1,850,000 | 9,700 | 0.3838 | 0.1162 | 0.0612 | 0.1879 | 0.2127 | 4.0413 | 0.0191 | 0.0369 |
| Step 1,900,000 | 9,957 | 0.2905 | 0.0969 | 0.0198 | 0.1641 | 0.1815 | 4.3567 | 0.0293 | 0.2443 |
| Step 1,950,000 | 10,219 | -0.1528 | 0.1292 | 0.0123 | 0.2647 | 0.2488 | 4.0464 | 0.0242 | 0.2443 |
| Step 2,000,000 | 10,465 | 0.1885 | 0.0304 | 0.0253 | 0.1331 | 0.1924 | 4.3853 | 0.0260 | 0.3140 |
| Step 2,050,000 | 10,699 | 0.2391 | 0.0310 | 0.0110 | 0.1143 | 0.1618 | 4.3927 | 0.0301 | 0.2443 |
| Step 2,100,000 | 10,940 | 0.2928 | -0.0009 | 0.0718 | 0.1020 | 0.1943 | 3.9084 | 0.0237 | 0.1477 |
| Step 2,150,000 | 11,192 | 0.4901 | 0.1313 | 0.0753 | 0.2358 | 0.3613 | 3.6750 | 0.0204 | 0.2791 |
| Step 2,200,000 | 11,444 | 0.7249 | 0.3443 | 0.1340 | 0.2813 | 0.3513 | 4.2718 | 0.0210 | 0.2443 |
| Step 2,250,000 | 11,704 | 0.6994 | 0.2234 | 0.1444 | 0.2387 | 0.3276 | 4.3943 | 0.0226 | 0.2791 |
| Step 2,300,000 | 11,946 | 0.6320 | 0.0929 | 0.0335 | 0.1692 | 0.2017 | 3.3700 | 0.0157 | 0.1662 |
| Step 2,350,000 | 12,193 | 0.3592 | 0.0944 | 0.1033 | 0.1612 | 0.2202 | 3.7535 | 0.0188 | 0.2791 |
| Step 2,400,000 | 12,436 | -0.0389 | 0.0662 | 0.0399 | 0.2188 | 0.3200 | 5.2122 | 0.0356 | 0.3140 |
| Step 2,450,000 | 12,667 | 0.1266 | 0.1152 | -0.0213 | 0.1569 | 0.1424 | 3.7629 | 0.0164 | 0.2443 |
| Step 2,500,000 | 12,906 | 0.0376 | 0.0650 | 0.0698 | 0.1539 | 0.2306 | 5.2738 | 0.0319 | 0.3489 |
| Step 2,550,000 | 13,146 | 0.1349 | 0.1786 | 0.0570 | 0.2035 | 0.2144 | 4.6839 | 0.0313 | 0.3140 |
| Step 2,600,000 | 13,379 | 0.0522 | 0.0045 | 0.0170 | 0.1159 | 0.1841 | 4.2460 | 0.0223 | 0.2031 |
| Step 2,650,000 | 13,610 | 0.3510 | 0.1117 | -0.0150 | 0.2604 | 0.2718 | 3.7915 | 0.0186 | 0.3140 |
| Step 2,700,000 | 13,839 | 0.2488 | 0.0457 | -0.0110 | 0.1441 | 0.1440 | 3.7545 | 0.0183 | 0.1292 |
| Step 2,750,000 | 14,071 | 0.4804 | 0.0965 | 0.0694 | 0.1797 | 0.2239 | 3.5066 | 0.0205 | 0.2216 |
| Step 2,800,000 | 14,301 | 0.4222 | 0.2225 | 0.1180 | 0.2077 | 0.2856 | 3.8227 | 0.0254 | 0.1846 |
| Step 2,850,000 | 14,521 | 0.2454 | 0.2136 | 0.0398 | 0.1999 | 0.1909 | 4.1124 | 0.0292 | 0.1477 |
| Step 2,900,000 | 14,753 | 0.3565 | 0.1043 | 0.0334 | 0.2247 | 0.2801 | 4.0283 | 0.0289 | 0.0554 |
| Step 2,950,000 | 14,970 | 0.7458 | 0.2180 | 0.0643 | 0.2250 | 0.2472 | 3.7517 | 0.0308 | 0.1477 |
| Step 3,000,000 | 15,208 | 0.6507 | 0.1905 | 0.1743 | 0.2293 | 0.3444 | 3.9948 | 0.0275 | -0.0391 |
