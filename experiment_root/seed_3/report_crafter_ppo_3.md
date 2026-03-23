# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 3  
**Total episodes:** 15,954  
**Experiment root:** `experiment_root\seed_3`  
**Generated:** 2026-03-22 08:22

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep3 | 3 | 0.2299 | 0.8689 | 0.4556 | 0.0893 | 0.0351 | 0.1624 | 0.0010 | — |
| collect_sapling @ ep4 | 4 | 0.1078 | 0.8868 | 0.2095 | 0.0861 | 0.0366 | 0.1859 | 0.0012 | 0.0000 |
| place_plant @ ep4 | 4 | 0.0644 | 0.8561 | 0.1817 | 0.0815 | 0.0370 | 0.1727 | 0.0009 | nan |
| collect_wood @ ep17 | 17 | 0.2779 | 0.9183 | 0.7394 | 0.0895 | 0.0439 | 0.1553 | 0.0009 | nan |
| place_table @ ep17 | 17 | 0.0767 | 0.9202 | 0.1961 | 0.0866 | 0.0352 | 0.1506 | 0.0007 | nan |
| collect_drink @ ep19 | 19 | 0.3079 | 0.9084 | -0.0201 | 0.1467 | 0.0387 | 0.2817 | 0.0007 | -0.6547 |
| make_wood_sword @ ep34 | 34 | 0.5666 | 0.9439 | 0.2863 | 0.2146 | 0.0879 | 0.3303 | 0.0005 | -0.6547 |
| make_wood_pickaxe @ ep50 | 50 | 0.6616 | 0.9350 | 0.3729 | 0.2528 | 0.0725 | 0.4286 | 0.0003 | -0.4352 |
| defeat_zombie @ ep76 | 76 | 0.8228 | 0.9173 | 0.7367 | 0.2570 | 0.1449 | 0.5409 | 0.0002 | 0.0000 |
| eat_cow @ ep158 | 158 | 0.6217 | 0.7229 | 0.4277 | 0.2653 | 0.2397 | 0.8008 | 0.0002 | -0.4352 |
| collect_stone @ ep215 | 215 | -0.1831 | 0.0012 | 0.0195 | 0.1413 | 0.1578 | 1.2858 | 0.0001 | -0.1745 |
| periodic_step50000_ep291 | 291 | 0.1000 | 0.2593 | 0.0574 | 0.2196 | 0.1342 | 1.3244 | 0.0001 | -0.4352 |
| periodic_step100000_ep576 | 576 | 0.4186 | 0.0304 | 0.3007 | 0.1466 | 0.1710 | 1.5008 | 0.0002 | -0.1047 |
| defeat_skeleton @ ep707 | 707 | -0.4419 | 0.2309 | 0.1662 | 0.1810 | 0.1708 | 1.2939 | 0.0003 | -0.1108 |
| periodic_step150000_ep862 | 862 | 0.3293 | 0.2447 | 0.4175 | 0.2120 | 0.2775 | 1.5970 | 0.0006 | -0.1741 |
| eat_plant @ ep1138 | 1,138 | -0.7358 | 0.1504 | 0.1424 | 0.2421 | 0.1834 | 1.0028 | 0.0008 | -0.0870 |
| periodic_step200000_ep1142 | 1,142 | -0.4306 | 0.2715 | 0.0636 | 0.1884 | 0.1023 | 1.1240 | 0.0014 | -0.2094 |
| make_stone_pickaxe @ ep1202 | 1,202 | -0.3702 | -0.0154 | 0.3743 | 0.0647 | 0.2183 | 0.7943 | 0.0003 | -0.3482 |
| periodic_step250000_ep1422 | 1,422 | 0.2588 | 0.4969 | 0.1612 | 0.1708 | 0.1511 | 0.4801 | 0.0006 | -0.4352 |
| periodic_step300000_ep1707 | 1,707 | -0.3561 | 0.3763 | 0.2096 | 0.1786 | 0.1802 | 0.5087 | 0.0009 | -0.2094 |
| place_stone @ ep1716 | 1,716 | 0.5038 | 0.1161 | 0.3617 | 0.0917 | 0.2072 | 0.5041 | 0.0010 | -0.1745 |
| make_stone_sword @ ep1895 | 1,895 | 0.5022 | 0.2314 | 0.3301 | 0.1570 | 0.3205 | 1.0515 | 0.0037 | -0.2791 |
| collect_coal @ ep1898 | 1,898 | 0.5807 | 0.3724 | 0.2984 | 0.1634 | 0.2574 | 0.7331 | 0.0019 | -0.2443 |
| periodic_step350000_ep2000 | 2,000 | -0.1553 | 0.0927 | 0.0758 | 0.1168 | 0.2284 | 0.8877 | 0.0028 | -0.2094 |
| periodic_step400000_ep2273 | 2,273 | 0.2813 | 0.0089 | 0.1542 | 0.1015 | 0.2131 | 1.2697 | 0.0027 | 0.0489 |
| periodic_step450000_ep2554 | 2,554 | 0.0268 | 0.1808 | -0.0237 | 0.1190 | 0.0715 | 1.1807 | 0.0035 | -0.1477 |
| collect_iron @ ep2571 | 2,571 | 0.1505 | 0.3086 | 0.0321 | 0.1653 | 0.0923 | 1.5415 | 0.0062 | -0.1292 |
| periodic_step500000_ep2835 | 2,835 | 0.1260 | 0.1060 | 0.1902 | 0.1059 | 0.2007 | 1.3341 | 0.0044 | -0.0923 |
| place_furnace @ ep2988 | 2,988 | 0.5970 | 0.1711 | 0.1163 | 0.1315 | 0.1439 | 1.3567 | 0.0031 | 0.0185 |
| periodic_step550000_ep3099 | 3,099 | -0.6095 | 0.0504 | 0.0253 | 0.1335 | 0.1837 | 1.6569 | 0.0048 | 0.1662 |
| periodic_step600000_ep3365 | 3,365 | 0.3271 | 0.1586 | 0.0907 | 0.1602 | 0.1838 | 1.9544 | 0.0049 | 0.0000 |
| periodic_step650000_ep3625 | 3,625 | -0.3885 | 0.1608 | 0.1017 | 0.1415 | 0.1699 | 2.1696 | 0.0086 | 0.0000 |
| periodic_step700000_ep3894 | 3,894 | 0.0308 | 0.0372 | 0.0184 | 0.1343 | 0.1377 | 1.7564 | 0.0034 | 0.0000 |
| periodic_step750000_ep4172 | 4,172 | 0.2495 | 0.1759 | 0.0160 | 0.1547 | 0.1402 | 2.6410 | 0.0109 | -0.1047 |
| periodic_step800000_ep4435 | 4,435 | 0.4308 | 0.1816 | 0.0135 | 0.1941 | 0.1437 | 2.4620 | 0.0096 | 0.0185 |
| periodic_step850000_ep4702 | 4,702 | 0.4297 | 0.0900 | 0.1876 | 0.1391 | 0.2038 | 2.3362 | 0.0100 | 0.0923 |
| periodic_step900000_ep4975 | 4,975 | 0.3476 | 0.2021 | 0.1118 | 0.1735 | 0.2035 | 2.7955 | 0.0123 | -0.0369 |
| periodic_step950000_ep5233 | 5,233 | 0.1645 | 0.1949 | 0.0126 | 0.1582 | 0.1273 | 2.3783 | 0.0095 | 0.0185 |
| periodic_step1000000_ep5503 | 5,503 | -0.1115 | 0.1484 | 0.0110 | 0.1508 | 0.1138 | 2.3243 | 0.0101 | 0.0923 |
| periodic_step1050000_ep5780 | 5,780 | 0.6801 | 0.3161 | 0.1209 | 0.2798 | 0.2235 | 3.1478 | 0.0129 | -0.1477 |
| periodic_step1100000_ep6058 | 6,058 | 0.6218 | 0.2729 | 0.0943 | 0.2159 | 0.1950 | 2.8083 | 0.0145 | -0.0923 |
| periodic_step1150000_ep6336 | 6,336 | 0.1169 | 0.1242 | 0.0967 | 0.1496 | 0.1650 | 2.8634 | 0.0165 | -0.0923 |
| periodic_step1200000_ep6602 | 6,602 | 0.8315 | 0.3092 | 0.1912 | 0.2678 | 0.3231 | 3.3159 | 0.0206 | -0.0923 |
| periodic_step1250000_ep6872 | 6,872 | 0.2527 | 0.0823 | 0.0946 | 0.1449 | 0.1985 | 3.2089 | 0.0208 | -0.0554 |
| periodic_step1300000_ep7137 | 7,137 | 0.1392 | 0.0254 | -0.0104 | 0.1091 | 0.1176 | 3.4716 | 0.0170 | 0.2094 |
| periodic_step1350000_ep7403 | 7,403 | 0.5658 | 0.1662 | 0.0516 | 0.1812 | 0.1824 | 3.8130 | 0.0241 | 0.0698 |
| periodic_step1400000_ep7676 | 7,676 | 0.5607 | 0.1295 | 0.1921 | 0.1655 | 0.2809 | 3.6741 | 0.0217 | 0.2443 |
| periodic_step1450000_ep7950 | 7,950 | 0.5975 | 0.2134 | 0.1345 | 0.2056 | 0.2394 | 3.4319 | 0.0256 | 0.2443 |
| periodic_step1500000_ep8228 | 8,228 | 0.4416 | 0.1334 | 0.0868 | 0.1685 | 0.1761 | 3.3323 | 0.0167 | 0.3838 |
| periodic_step1550000_ep8502 | 8,502 | -0.0438 | 0.0963 | 0.0424 | 0.1596 | 0.2013 | 3.6967 | 0.0195 | 0.1745 |
| periodic_step1600000_ep8781 | 8,781 | 0.8057 | 0.4570 | 0.1611 | 0.3623 | 0.3023 | 4.9855 | 0.0324 | 0.1468 |
| periodic_step1650000_ep9045 | 9,045 | 0.3354 | 0.0802 | -0.0021 | 0.1385 | 0.1328 | 5.2518 | 0.0404 | 0.1292 |
| periodic_step1700000_ep9321 | 9,321 | 0.5214 | 0.1494 | 0.1940 | 0.1575 | 0.2516 | 5.1845 | 0.0400 | 0.1662 |
| periodic_step1750000_ep9599 | 9,599 | 0.2442 | 0.0440 | 0.0688 | 0.1320 | 0.1880 | 5.3505 | 0.0325 | 0.3140 |
| periodic_step1800000_ep9876 | 9,876 | 0.4155 | 0.0299 | 0.0248 | 0.1175 | 0.1803 | 4.7865 | 0.0306 | 0.0923 |
| periodic_step1850000_ep10151 | 10,151 | 0.1026 | 0.1125 | -0.0248 | 0.1497 | 0.1320 | 5.4388 | 0.0398 | 0.1662 |
| periodic_step1900000_ep10418 | 10,418 | 0.0930 | 0.0605 | 0.0403 | 0.1395 | 0.1619 | 5.3784 | 0.0388 | 0.0923 |
| periodic_step1950000_ep10683 | 10,683 | 0.5810 | 0.1474 | 0.0965 | 0.1821 | 0.3260 | 4.6181 | 0.0293 | 0.1662 |
| periodic_step2000000_ep10934 | 10,934 | 0.1488 | 0.2331 | -0.0140 | 0.2804 | 0.1512 | 4.3192 | 0.0245 | 0.0979 |
| periodic_step2050000_ep11185 | 11,185 | 0.1551 | 0.0305 | 0.0343 | 0.1205 | 0.1547 | 4.2872 | 0.0243 | 0.2216 |
| periodic_step2100000_ep11442 | 11,442 | 0.3026 | 0.0701 | 0.0378 | 0.1407 | 0.1743 | 4.9053 | 0.0291 | 0.0685 |
| periodic_step2150000_ep11702 | 11,702 | 0.4186 | 0.1527 | 0.0441 | 0.1699 | 0.1864 | 4.7274 | 0.0323 | 0.1174 |
| periodic_step2200000_ep11958 | 11,958 | 0.3223 | 0.1554 | 0.0120 | 0.1776 | 0.1704 | 4.8911 | 0.0384 | 0.1662 |
| periodic_step2250000_ep12223 | 12,223 | 0.2247 | 0.1523 | 0.0459 | 0.1819 | 0.1840 | 4.4895 | 0.0300 | 0.1477 |
| periodic_step2300000_ep12486 | 12,486 | 0.5989 | 0.2116 | 0.0443 | 0.2025 | 0.1638 | 4.1003 | 0.0243 | 0.1292 |
| periodic_step2350000_ep12740 | 12,740 | 0.7570 | 0.1153 | 0.1464 | 0.1892 | 0.2520 | 3.6838 | 0.0192 | 0.3140 |
| periodic_step2400000_ep12995 | 12,995 | 0.1815 | 0.1186 | 0.0789 | 0.1504 | 0.1912 | 3.9960 | 0.0284 | 0.1846 |
| periodic_step2450000_ep13252 | 13,252 | 0.3812 | 0.1111 | 0.0190 | 0.1767 | 0.1869 | 4.0630 | 0.0174 | 0.2031 |
| periodic_step2500000_ep13506 | 13,506 | 0.4656 | 0.0773 | 0.0862 | 0.1665 | 0.2630 | 3.8888 | 0.0199 | 0.1846 |
| periodic_step2550000_ep13776 | 13,776 | 0.4918 | 0.2948 | 0.0359 | 0.2806 | 0.1941 | 3.7805 | 0.0210 | 0.1957 |
| periodic_step2600000_ep14032 | 14,032 | 0.2942 | 0.1041 | 0.0592 | 0.1808 | 0.2096 | 3.5910 | 0.0154 | 0.2031 |
| periodic_step2650000_ep14270 | 14,270 | 0.0596 | 0.1318 | 0.0256 | 0.1678 | 0.1608 | 3.6763 | 0.0181 | 0.2216 |
| periodic_step2700000_ep14515 | 14,515 | 0.5698 | 0.2640 | 0.0980 | 0.2759 | 0.2595 | 4.7524 | 0.0279 | 0.1292 |
| periodic_step2750000_ep14755 | 14,755 | 0.1043 | 0.0421 | 0.0258 | 0.1471 | 0.1815 | 4.0581 | 0.0201 | 0.2216 |
| periodic_step2800000_ep14998 | 14,998 | 0.2954 | 0.0868 | -0.0167 | 0.1548 | 0.1452 | 3.9083 | 0.0183 | 0.2031 |
| periodic_step2850000_ep15241 | 15,241 | -0.0783 | 0.0797 | 0.0774 | 0.1534 | 0.2073 | 4.2013 | 0.0223 | 0.2031 |
| periodic_step2900000_ep15483 | 15,483 | 0.0634 | 0.0397 | 0.0776 | 0.1350 | 0.1940 | 3.7338 | 0.0217 | 0.2031 |
| periodic_step2950000_ep15719 | 15,719 | 0.5158 | 0.0384 | 0.0458 | 0.1600 | 0.2240 | 3.6664 | 0.0191 | 0.2216 |
| periodic_step3000000_ep15953 | 15,953 | 0.1579 | 0.0791 | 0.0129 | 0.1760 | 0.1747 | 3.8672 | 0.0205 | 0.1846 |
| final_step3000320_ep15954 | 15,954 | 0.6631 | 0.1251 | 0.0370 | 0.1934 | 0.1900 | 3.8896 | 0.0219 | 0.1846 |

---

## wake_up_ep3_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2299 |
| Coherence (Success) | 0.8689 |
| Coherence (Failure) | 0.4556 |
| Gradient Magnitude (Success) | 0.0893 |
| Gradient Magnitude (Failure) | 0.0351 |
| Activation Separation | 0.1624 |
| Cosine Distance | 0.0010 |
| Clusters | 1,439 |
| Noise Fraction | 0.2327 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## collect_sapling_ep4_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1078 |
| Coherence (Success) | 0.8868 |
| Coherence (Failure) | 0.2095 |
| Gradient Magnitude (Success) | 0.0861 |
| Gradient Magnitude (Failure) | 0.0366 |
| Activation Separation | 0.1859 |
| Cosine Distance | 0.0012 |
| Clusters | 1,411 |
| Noise Fraction | 0.2202 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (3) | Skeleton, Wood, Zombie |

---

## place_plant_ep4_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0644 |
| Coherence (Success) | 0.8561 |
| Coherence (Failure) | 0.1817 |
| Gradient Magnitude (Success) | 0.0815 |
| Gradient Magnitude (Failure) | 0.0370 |
| Activation Separation | 0.1727 |
| Cosine Distance | 0.0009 |
| Clusters | 1,423 |
| Noise Fraction | 0.2615 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_wood_ep17_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2779 |
| Coherence (Success) | 0.9183 |
| Coherence (Failure) | 0.7394 |
| Gradient Magnitude (Success) | 0.0895 |
| Gradient Magnitude (Failure) | 0.0439 |
| Activation Separation | 0.1553 |
| Cosine Distance | 0.0009 |
| Clusters | 1,441 |
| Noise Fraction | 0.2448 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Skeleton, Wood, Wood Pickaxe |

---

## place_table_ep17_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0767 |
| Coherence (Success) | 0.9202 |
| Coherence (Failure) | 0.1961 |
| Gradient Magnitude (Success) | 0.0866 |
| Gradient Magnitude (Failure) | 0.0352 |
| Activation Separation | 0.1506 |
| Cosine Distance | 0.0007 |
| Clusters | 1,409 |
| Noise Fraction | 0.2581 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep19_lower1.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3079 |
| Coherence (Success) | 0.9084 |
| Coherence (Failure) | -0.0201 |
| Gradient Magnitude (Success) | 0.1467 |
| Gradient Magnitude (Failure) | 0.0387 |
| Activation Separation | 0.2817 |
| Cosine Distance | 0.0007 |
| Clusters | 1,428 |
| Noise Fraction | 0.2575 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep34_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5666 |
| Coherence (Success) | 0.9439 |
| Coherence (Failure) | 0.2863 |
| Gradient Magnitude (Success) | 0.2146 |
| Gradient Magnitude (Failure) | 0.0879 |
| Activation Separation | 0.3303 |
| Cosine Distance | 0.0005 |
| Clusters | 1,366 |
| Noise Fraction | 0.2478 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep50_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6616 |
| Coherence (Success) | 0.9350 |
| Coherence (Failure) | 0.3729 |
| Gradient Magnitude (Success) | 0.2528 |
| Gradient Magnitude (Failure) | 0.0725 |
| Activation Separation | 0.4286 |
| Cosine Distance | 0.0003 |
| Clusters | 1,476 |
| Noise Fraction | 0.2370 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep76_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8228 |
| Coherence (Success) | 0.9173 |
| Coherence (Failure) | 0.7367 |
| Gradient Magnitude (Success) | 0.2570 |
| Gradient Magnitude (Failure) | 0.1449 |
| Activation Separation | 0.5409 |
| Cosine Distance | 0.0002 |
| Clusters | 1,601 |
| Noise Fraction | 0.2665 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep158_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6217 |
| Coherence (Success) | 0.7229 |
| Coherence (Failure) | 0.4277 |
| Gradient Magnitude (Success) | 0.2653 |
| Gradient Magnitude (Failure) | 0.2397 |
| Activation Separation | 0.8008 |
| Cosine Distance | 0.0002 |
| Clusters | 1,606 |
| Noise Fraction | 0.2612 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep215_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1831 |
| Coherence (Success) | 0.0012 |
| Coherence (Failure) | 0.0195 |
| Gradient Magnitude (Success) | 0.1413 |
| Gradient Magnitude (Failure) | 0.1578 |
| Activation Separation | 1.2858 |
| Cosine Distance | 0.0001 |
| Clusters | 1,485 |
| Noise Fraction | 0.2690 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep291_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1000 |
| Coherence (Success) | 0.2593 |
| Coherence (Failure) | 0.0574 |
| Gradient Magnitude (Success) | 0.2196 |
| Gradient Magnitude (Failure) | 0.1342 |
| Activation Separation | 1.3244 |
| Cosine Distance | 0.0001 |
| Clusters | 1,377 |
| Noise Fraction | 0.3197 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep576_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4186 |
| Coherence (Success) | 0.0304 |
| Coherence (Failure) | 0.3007 |
| Gradient Magnitude (Success) | 0.1466 |
| Gradient Magnitude (Failure) | 0.1710 |
| Activation Separation | 1.5008 |
| Cosine Distance | 0.0002 |
| Clusters | 1,363 |
| Noise Fraction | 0.3103 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep707_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4419 |
| Coherence (Success) | 0.2309 |
| Coherence (Failure) | 0.1662 |
| Gradient Magnitude (Success) | 0.1810 |
| Gradient Magnitude (Failure) | 0.1708 |
| Activation Separation | 1.2939 |
| Cosine Distance | 0.0003 |
| Clusters | 1,421 |
| Noise Fraction | 0.3251 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep862_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3293 |
| Coherence (Success) | 0.2447 |
| Coherence (Failure) | 0.4175 |
| Gradient Magnitude (Success) | 0.2120 |
| Gradient Magnitude (Failure) | 0.2775 |
| Activation Separation | 1.5970 |
| Cosine Distance | 0.0006 |
| Clusters | 1,347 |
| Noise Fraction | 0.3140 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep1138_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7358 |
| Coherence (Success) | 0.1504 |
| Coherence (Failure) | 0.1424 |
| Gradient Magnitude (Success) | 0.2421 |
| Gradient Magnitude (Failure) | 0.1834 |
| Activation Separation | 1.0028 |
| Cosine Distance | 0.0008 |
| Clusters | 1,543 |
| Noise Fraction | 0.2886 |
| RSA Alignment (ρ) | -0.0870 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1142_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4306 |
| Coherence (Success) | 0.2715 |
| Coherence (Failure) | 0.0636 |
| Gradient Magnitude (Success) | 0.1884 |
| Gradient Magnitude (Failure) | 0.1023 |
| Activation Separation | 1.1240 |
| Cosine Distance | 0.0014 |
| Clusters | 1,591 |
| Noise Fraction | 0.2677 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep1202_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3702 |
| Coherence (Success) | -0.0154 |
| Coherence (Failure) | 0.3743 |
| Gradient Magnitude (Success) | 0.0647 |
| Gradient Magnitude (Failure) | 0.2183 |
| Activation Separation | 0.7943 |
| Cosine Distance | 0.0003 |
| Clusters | 1,453 |
| Noise Fraction | 0.2748 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1422_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2588 |
| Coherence (Success) | 0.4969 |
| Coherence (Failure) | 0.1612 |
| Gradient Magnitude (Success) | 0.1708 |
| Gradient Magnitude (Failure) | 0.1511 |
| Activation Separation | 0.4801 |
| Cosine Distance | 0.0006 |
| Clusters | 1,528 |
| Noise Fraction | 0.2836 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1707_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3561 |
| Coherence (Success) | 0.3763 |
| Coherence (Failure) | 0.2096 |
| Gradient Magnitude (Success) | 0.1786 |
| Gradient Magnitude (Failure) | 0.1802 |
| Activation Separation | 0.5087 |
| Cosine Distance | 0.0009 |
| Clusters | 1,305 |
| Noise Fraction | 0.2875 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep1716_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5038 |
| Coherence (Success) | 0.1161 |
| Coherence (Failure) | 0.3617 |
| Gradient Magnitude (Success) | 0.0917 |
| Gradient Magnitude (Failure) | 0.2072 |
| Activation Separation | 0.5041 |
| Cosine Distance | 0.0010 |
| Clusters | 1,378 |
| Noise Fraction | 0.3052 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep1895_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5022 |
| Coherence (Success) | 0.2314 |
| Coherence (Failure) | 0.3301 |
| Gradient Magnitude (Success) | 0.1570 |
| Gradient Magnitude (Failure) | 0.3205 |
| Activation Separation | 1.0515 |
| Cosine Distance | 0.0037 |
| Clusters | 1,297 |
| Noise Fraction | 0.2845 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep1898_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5807 |
| Coherence (Success) | 0.3724 |
| Coherence (Failure) | 0.2984 |
| Gradient Magnitude (Success) | 0.1634 |
| Gradient Magnitude (Failure) | 0.2574 |
| Activation Separation | 0.7331 |
| Cosine Distance | 0.0019 |
| Clusters | 1,292 |
| Noise Fraction | 0.3082 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2000_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1553 |
| Coherence (Success) | 0.0927 |
| Coherence (Failure) | 0.0758 |
| Gradient Magnitude (Success) | 0.1168 |
| Gradient Magnitude (Failure) | 0.2284 |
| Activation Separation | 0.8877 |
| Cosine Distance | 0.0028 |
| Clusters | 1,193 |
| Noise Fraction | 0.2581 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2273_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2813 |
| Coherence (Success) | 0.0089 |
| Coherence (Failure) | 0.1542 |
| Gradient Magnitude (Success) | 0.1015 |
| Gradient Magnitude (Failure) | 0.2131 |
| Activation Separation | 1.2697 |
| Cosine Distance | 0.0027 |
| Clusters | 1,409 |
| Noise Fraction | 0.2696 |
| RSA Alignment (ρ) | 0.0489 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2554_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0268 |
| Coherence (Success) | 0.1808 |
| Coherence (Failure) | -0.0237 |
| Gradient Magnitude (Success) | 0.1190 |
| Gradient Magnitude (Failure) | 0.0715 |
| Activation Separation | 1.1807 |
| Cosine Distance | 0.0035 |
| Clusters | 1,365 |
| Noise Fraction | 0.2458 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep2571_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1505 |
| Coherence (Success) | 0.3086 |
| Coherence (Failure) | 0.0321 |
| Gradient Magnitude (Success) | 0.1653 |
| Gradient Magnitude (Failure) | 0.0923 |
| Activation Separation | 1.5415 |
| Cosine Distance | 0.0062 |
| Clusters | 1,337 |
| Noise Fraction | 0.2675 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2835_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1260 |
| Coherence (Success) | 0.1060 |
| Coherence (Failure) | 0.1902 |
| Gradient Magnitude (Success) | 0.1059 |
| Gradient Magnitude (Failure) | 0.2007 |
| Activation Separation | 1.3341 |
| Cosine Distance | 0.0044 |
| Clusters | 1,465 |
| Noise Fraction | 0.2411 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2988_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5970 |
| Coherence (Success) | 0.1711 |
| Coherence (Failure) | 0.1163 |
| Gradient Magnitude (Success) | 0.1315 |
| Gradient Magnitude (Failure) | 0.1439 |
| Activation Separation | 1.3567 |
| Cosine Distance | 0.0031 |
| Clusters | 1,479 |
| Noise Fraction | 0.2657 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3099_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.6095 |
| Coherence (Success) | 0.0504 |
| Coherence (Failure) | 0.0253 |
| Gradient Magnitude (Success) | 0.1335 |
| Gradient Magnitude (Failure) | 0.1837 |
| Activation Separation | 1.6569 |
| Cosine Distance | 0.0048 |
| Clusters | 1,518 |
| Noise Fraction | 0.2563 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3365_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3271 |
| Coherence (Success) | 0.1586 |
| Coherence (Failure) | 0.0907 |
| Gradient Magnitude (Success) | 0.1602 |
| Gradient Magnitude (Failure) | 0.1838 |
| Activation Separation | 1.9544 |
| Cosine Distance | 0.0049 |
| Clusters | 1,598 |
| Noise Fraction | 0.2504 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3625_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3885 |
| Coherence (Success) | 0.1608 |
| Coherence (Failure) | 0.1017 |
| Gradient Magnitude (Success) | 0.1415 |
| Gradient Magnitude (Failure) | 0.1699 |
| Activation Separation | 2.1696 |
| Cosine Distance | 0.0086 |
| Clusters | 1,435 |
| Noise Fraction | 0.2832 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3894_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0308 |
| Coherence (Success) | 0.0372 |
| Coherence (Failure) | 0.0184 |
| Gradient Magnitude (Success) | 0.1343 |
| Gradient Magnitude (Failure) | 0.1377 |
| Activation Separation | 1.7564 |
| Cosine Distance | 0.0034 |
| Clusters | 1,419 |
| Noise Fraction | 0.2926 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4172_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2495 |
| Coherence (Success) | 0.1759 |
| Coherence (Failure) | 0.0160 |
| Gradient Magnitude (Success) | 0.1547 |
| Gradient Magnitude (Failure) | 0.1402 |
| Activation Separation | 2.6410 |
| Cosine Distance | 0.0109 |
| Clusters | 1,409 |
| Noise Fraction | 0.2618 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4435_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4308 |
| Coherence (Success) | 0.1816 |
| Coherence (Failure) | 0.0135 |
| Gradient Magnitude (Success) | 0.1941 |
| Gradient Magnitude (Failure) | 0.1437 |
| Activation Separation | 2.4620 |
| Cosine Distance | 0.0096 |
| Clusters | 1,512 |
| Noise Fraction | 0.2984 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4702_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4297 |
| Coherence (Success) | 0.0900 |
| Coherence (Failure) | 0.1876 |
| Gradient Magnitude (Success) | 0.1391 |
| Gradient Magnitude (Failure) | 0.2038 |
| Activation Separation | 2.3362 |
| Cosine Distance | 0.0100 |
| Clusters | 1,381 |
| Noise Fraction | 0.3351 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4975_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3476 |
| Coherence (Success) | 0.2021 |
| Coherence (Failure) | 0.1118 |
| Gradient Magnitude (Success) | 0.1735 |
| Gradient Magnitude (Failure) | 0.2035 |
| Activation Separation | 2.7955 |
| Cosine Distance | 0.0123 |
| Clusters | 1,288 |
| Noise Fraction | 0.3194 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5233_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1645 |
| Coherence (Success) | 0.1949 |
| Coherence (Failure) | 0.0126 |
| Gradient Magnitude (Success) | 0.1582 |
| Gradient Magnitude (Failure) | 0.1273 |
| Activation Separation | 2.3783 |
| Cosine Distance | 0.0095 |
| Clusters | 1,273 |
| Noise Fraction | 0.3087 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5503_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1115 |
| Coherence (Success) | 0.1484 |
| Coherence (Failure) | 0.0110 |
| Gradient Magnitude (Success) | 0.1508 |
| Gradient Magnitude (Failure) | 0.1138 |
| Activation Separation | 2.3243 |
| Cosine Distance | 0.0101 |
| Clusters | 1,447 |
| Noise Fraction | 0.2972 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5780_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6801 |
| Coherence (Success) | 0.3161 |
| Coherence (Failure) | 0.1209 |
| Gradient Magnitude (Success) | 0.2798 |
| Gradient Magnitude (Failure) | 0.2235 |
| Activation Separation | 3.1478 |
| Cosine Distance | 0.0129 |
| Clusters | 1,284 |
| Noise Fraction | 0.3392 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6058_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6218 |
| Coherence (Success) | 0.2729 |
| Coherence (Failure) | 0.0943 |
| Gradient Magnitude (Success) | 0.2159 |
| Gradient Magnitude (Failure) | 0.1950 |
| Activation Separation | 2.8083 |
| Cosine Distance | 0.0145 |
| Clusters | 1,289 |
| Noise Fraction | 0.3614 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6336_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1169 |
| Coherence (Success) | 0.1242 |
| Coherence (Failure) | 0.0967 |
| Gradient Magnitude (Success) | 0.1496 |
| Gradient Magnitude (Failure) | 0.1650 |
| Activation Separation | 2.8634 |
| Cosine Distance | 0.0165 |
| Clusters | 1,222 |
| Noise Fraction | 0.3678 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6602_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8315 |
| Coherence (Success) | 0.3092 |
| Coherence (Failure) | 0.1912 |
| Gradient Magnitude (Success) | 0.2678 |
| Gradient Magnitude (Failure) | 0.3231 |
| Activation Separation | 3.3159 |
| Cosine Distance | 0.0206 |
| Clusters | 1,334 |
| Noise Fraction | 0.3449 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6872_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2527 |
| Coherence (Success) | 0.0823 |
| Coherence (Failure) | 0.0946 |
| Gradient Magnitude (Success) | 0.1449 |
| Gradient Magnitude (Failure) | 0.1985 |
| Activation Separation | 3.2089 |
| Cosine Distance | 0.0208 |
| Clusters | 1,451 |
| Noise Fraction | 0.3464 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7137_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1392 |
| Coherence (Success) | 0.0254 |
| Coherence (Failure) | -0.0104 |
| Gradient Magnitude (Success) | 0.1091 |
| Gradient Magnitude (Failure) | 0.1176 |
| Activation Separation | 3.4716 |
| Cosine Distance | 0.0170 |
| Clusters | 1,364 |
| Noise Fraction | 0.3215 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7403_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5658 |
| Coherence (Success) | 0.1662 |
| Coherence (Failure) | 0.0516 |
| Gradient Magnitude (Success) | 0.1812 |
| Gradient Magnitude (Failure) | 0.1824 |
| Activation Separation | 3.8130 |
| Cosine Distance | 0.0241 |
| Clusters | 1,437 |
| Noise Fraction | 0.3343 |
| RSA Alignment (ρ) | 0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7676_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5607 |
| Coherence (Success) | 0.1295 |
| Coherence (Failure) | 0.1921 |
| Gradient Magnitude (Success) | 0.1655 |
| Gradient Magnitude (Failure) | 0.2809 |
| Activation Separation | 3.6741 |
| Cosine Distance | 0.0217 |
| Clusters | 1,290 |
| Noise Fraction | 0.3170 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7950_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5975 |
| Coherence (Success) | 0.2134 |
| Coherence (Failure) | 0.1345 |
| Gradient Magnitude (Success) | 0.2056 |
| Gradient Magnitude (Failure) | 0.2394 |
| Activation Separation | 3.4319 |
| Cosine Distance | 0.0256 |
| Clusters | 1,364 |
| Noise Fraction | 0.3131 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8228_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4416 |
| Coherence (Success) | 0.1334 |
| Coherence (Failure) | 0.0868 |
| Gradient Magnitude (Success) | 0.1685 |
| Gradient Magnitude (Failure) | 0.1761 |
| Activation Separation | 3.3323 |
| Cosine Distance | 0.0167 |
| Clusters | 1,321 |
| Noise Fraction | 0.3251 |
| RSA Alignment (ρ) | 0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8502_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0438 |
| Coherence (Success) | 0.0963 |
| Coherence (Failure) | 0.0424 |
| Gradient Magnitude (Success) | 0.1596 |
| Gradient Magnitude (Failure) | 0.2013 |
| Activation Separation | 3.6967 |
| Cosine Distance | 0.0195 |
| Clusters | 1,439 |
| Noise Fraction | 0.3016 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8781_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8057 |
| Coherence (Success) | 0.4570 |
| Coherence (Failure) | 0.1611 |
| Gradient Magnitude (Success) | 0.3623 |
| Gradient Magnitude (Failure) | 0.3023 |
| Activation Separation | 4.9855 |
| Cosine Distance | 0.0324 |
| Clusters | 1,245 |
| Noise Fraction | 0.2962 |
| RSA Alignment (ρ) | 0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9045_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3354 |
| Coherence (Success) | 0.0802 |
| Coherence (Failure) | -0.0021 |
| Gradient Magnitude (Success) | 0.1385 |
| Gradient Magnitude (Failure) | 0.1328 |
| Activation Separation | 5.2518 |
| Cosine Distance | 0.0404 |
| Clusters | 1,538 |
| Noise Fraction | 0.2755 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9321_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5214 |
| Coherence (Success) | 0.1494 |
| Coherence (Failure) | 0.1940 |
| Gradient Magnitude (Success) | 0.1575 |
| Gradient Magnitude (Failure) | 0.2516 |
| Activation Separation | 5.1845 |
| Cosine Distance | 0.0400 |
| Clusters | 1,577 |
| Noise Fraction | 0.2810 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9599_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2442 |
| Coherence (Success) | 0.0440 |
| Coherence (Failure) | 0.0688 |
| Gradient Magnitude (Success) | 0.1320 |
| Gradient Magnitude (Failure) | 0.1880 |
| Activation Separation | 5.3505 |
| Cosine Distance | 0.0325 |
| Clusters | 1,284 |
| Noise Fraction | 0.2803 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9876_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4155 |
| Coherence (Success) | 0.0299 |
| Coherence (Failure) | 0.0248 |
| Gradient Magnitude (Success) | 0.1175 |
| Gradient Magnitude (Failure) | 0.1803 |
| Activation Separation | 4.7865 |
| Cosine Distance | 0.0306 |
| Clusters | 1,322 |
| Noise Fraction | 0.2741 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10151_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1026 |
| Coherence (Success) | 0.1125 |
| Coherence (Failure) | -0.0248 |
| Gradient Magnitude (Success) | 0.1497 |
| Gradient Magnitude (Failure) | 0.1320 |
| Activation Separation | 5.4388 |
| Cosine Distance | 0.0398 |
| Clusters | 1,252 |
| Noise Fraction | 0.2791 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10418_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0930 |
| Coherence (Success) | 0.0605 |
| Coherence (Failure) | 0.0403 |
| Gradient Magnitude (Success) | 0.1395 |
| Gradient Magnitude (Failure) | 0.1619 |
| Activation Separation | 5.3784 |
| Cosine Distance | 0.0388 |
| Clusters | 1,326 |
| Noise Fraction | 0.2868 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10683_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5810 |
| Coherence (Success) | 0.1474 |
| Coherence (Failure) | 0.0965 |
| Gradient Magnitude (Success) | 0.1821 |
| Gradient Magnitude (Failure) | 0.3260 |
| Activation Separation | 4.6181 |
| Cosine Distance | 0.0293 |
| Clusters | 1,196 |
| Noise Fraction | 0.3117 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10934_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1488 |
| Coherence (Success) | 0.2331 |
| Coherence (Failure) | -0.0140 |
| Gradient Magnitude (Success) | 0.2804 |
| Gradient Magnitude (Failure) | 0.1512 |
| Activation Separation | 4.3192 |
| Cosine Distance | 0.0245 |
| Clusters | 1,135 |
| Noise Fraction | 0.3257 |
| RSA Alignment (ρ) | 0.0979 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11185_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1551 |
| Coherence (Success) | 0.0305 |
| Coherence (Failure) | 0.0343 |
| Gradient Magnitude (Success) | 0.1205 |
| Gradient Magnitude (Failure) | 0.1547 |
| Activation Separation | 4.2872 |
| Cosine Distance | 0.0243 |
| Clusters | 1,212 |
| Noise Fraction | 0.3082 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11442_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3026 |
| Coherence (Success) | 0.0701 |
| Coherence (Failure) | 0.0378 |
| Gradient Magnitude (Success) | 0.1407 |
| Gradient Magnitude (Failure) | 0.1743 |
| Activation Separation | 4.9053 |
| Cosine Distance | 0.0291 |
| Clusters | 1,216 |
| Noise Fraction | 0.2907 |
| RSA Alignment (ρ) | 0.0685 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11702_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4186 |
| Coherence (Success) | 0.1527 |
| Coherence (Failure) | 0.0441 |
| Gradient Magnitude (Success) | 0.1699 |
| Gradient Magnitude (Failure) | 0.1864 |
| Activation Separation | 4.7274 |
| Cosine Distance | 0.0323 |
| Clusters | 1,358 |
| Noise Fraction | 0.3223 |
| RSA Alignment (ρ) | 0.1174 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11958_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3223 |
| Coherence (Success) | 0.1554 |
| Coherence (Failure) | 0.0120 |
| Gradient Magnitude (Success) | 0.1776 |
| Gradient Magnitude (Failure) | 0.1704 |
| Activation Separation | 4.8911 |
| Cosine Distance | 0.0384 |
| Clusters | 1,177 |
| Noise Fraction | 0.3022 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12223_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2247 |
| Coherence (Success) | 0.1523 |
| Coherence (Failure) | 0.0459 |
| Gradient Magnitude (Success) | 0.1819 |
| Gradient Magnitude (Failure) | 0.1840 |
| Activation Separation | 4.4895 |
| Cosine Distance | 0.0300 |
| Clusters | 1,352 |
| Noise Fraction | 0.3254 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12486_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5989 |
| Coherence (Success) | 0.2116 |
| Coherence (Failure) | 0.0443 |
| Gradient Magnitude (Success) | 0.2025 |
| Gradient Magnitude (Failure) | 0.1638 |
| Activation Separation | 4.1003 |
| Cosine Distance | 0.0243 |
| Clusters | 1,271 |
| Noise Fraction | 0.3321 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12740_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7570 |
| Coherence (Success) | 0.1153 |
| Coherence (Failure) | 0.1464 |
| Gradient Magnitude (Success) | 0.1892 |
| Gradient Magnitude (Failure) | 0.2520 |
| Activation Separation | 3.6838 |
| Cosine Distance | 0.0192 |
| Clusters | 1,169 |
| Noise Fraction | 0.3597 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12995_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1815 |
| Coherence (Success) | 0.1186 |
| Coherence (Failure) | 0.0789 |
| Gradient Magnitude (Success) | 0.1504 |
| Gradient Magnitude (Failure) | 0.1912 |
| Activation Separation | 3.9960 |
| Cosine Distance | 0.0284 |
| Clusters | 1,031 |
| Noise Fraction | 0.3042 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13252_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3812 |
| Coherence (Success) | 0.1111 |
| Coherence (Failure) | 0.0190 |
| Gradient Magnitude (Success) | 0.1767 |
| Gradient Magnitude (Failure) | 0.1869 |
| Activation Separation | 4.0630 |
| Cosine Distance | 0.0174 |
| Clusters | 1,145 |
| Noise Fraction | 0.3571 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13506_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4656 |
| Coherence (Success) | 0.0773 |
| Coherence (Failure) | 0.0862 |
| Gradient Magnitude (Success) | 0.1665 |
| Gradient Magnitude (Failure) | 0.2630 |
| Activation Separation | 3.8888 |
| Cosine Distance | 0.0199 |
| Clusters | 1,230 |
| Noise Fraction | 0.3527 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13776_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4918 |
| Coherence (Success) | 0.2948 |
| Coherence (Failure) | 0.0359 |
| Gradient Magnitude (Success) | 0.2806 |
| Gradient Magnitude (Failure) | 0.1941 |
| Activation Separation | 3.7805 |
| Cosine Distance | 0.0210 |
| Clusters | 1,123 |
| Noise Fraction | 0.3361 |
| RSA Alignment (ρ) | 0.1957 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14032_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2942 |
| Coherence (Success) | 0.1041 |
| Coherence (Failure) | 0.0592 |
| Gradient Magnitude (Success) | 0.1808 |
| Gradient Magnitude (Failure) | 0.2096 |
| Activation Separation | 3.5910 |
| Cosine Distance | 0.0154 |
| Clusters | 1,144 |
| Noise Fraction | 0.3699 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14270_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0596 |
| Coherence (Success) | 0.1318 |
| Coherence (Failure) | 0.0256 |
| Gradient Magnitude (Success) | 0.1678 |
| Gradient Magnitude (Failure) | 0.1608 |
| Activation Separation | 3.6763 |
| Cosine Distance | 0.0181 |
| Clusters | 1,233 |
| Noise Fraction | 0.3425 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14515_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5698 |
| Coherence (Success) | 0.2640 |
| Coherence (Failure) | 0.0980 |
| Gradient Magnitude (Success) | 0.2759 |
| Gradient Magnitude (Failure) | 0.2595 |
| Activation Separation | 4.7524 |
| Cosine Distance | 0.0279 |
| Clusters | 1,162 |
| Noise Fraction | 0.3598 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14755_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1043 |
| Coherence (Success) | 0.0421 |
| Coherence (Failure) | 0.0258 |
| Gradient Magnitude (Success) | 0.1471 |
| Gradient Magnitude (Failure) | 0.1815 |
| Activation Separation | 4.0581 |
| Cosine Distance | 0.0201 |
| Clusters | 1,238 |
| Noise Fraction | 0.3572 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14998_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2954 |
| Coherence (Success) | 0.0868 |
| Coherence (Failure) | -0.0167 |
| Gradient Magnitude (Success) | 0.1548 |
| Gradient Magnitude (Failure) | 0.1452 |
| Activation Separation | 3.9083 |
| Cosine Distance | 0.0183 |
| Clusters | 1,308 |
| Noise Fraction | 0.3490 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15241_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0783 |
| Coherence (Success) | 0.0797 |
| Coherence (Failure) | 0.0774 |
| Gradient Magnitude (Success) | 0.1534 |
| Gradient Magnitude (Failure) | 0.2073 |
| Activation Separation | 4.2013 |
| Cosine Distance | 0.0223 |
| Clusters | 1,263 |
| Noise Fraction | 0.3327 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15483_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0634 |
| Coherence (Success) | 0.0397 |
| Coherence (Failure) | 0.0776 |
| Gradient Magnitude (Success) | 0.1350 |
| Gradient Magnitude (Failure) | 0.1940 |
| Activation Separation | 3.7338 |
| Cosine Distance | 0.0217 |
| Clusters | 1,131 |
| Noise Fraction | 0.3410 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15719_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5158 |
| Coherence (Success) | 0.0384 |
| Coherence (Failure) | 0.0458 |
| Gradient Magnitude (Success) | 0.1600 |
| Gradient Magnitude (Failure) | 0.2240 |
| Activation Separation | 3.6664 |
| Cosine Distance | 0.0191 |
| Clusters | 1,148 |
| Noise Fraction | 0.3310 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15953_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1579 |
| Coherence (Success) | 0.0791 |
| Coherence (Failure) | 0.0129 |
| Gradient Magnitude (Success) | 0.1760 |
| Gradient Magnitude (Failure) | 0.1747 |
| Activation Separation | 3.8672 |
| Cosine Distance | 0.0205 |
| Clusters | 1,221 |
| Noise Fraction | 0.3717 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15954_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6631 |
| Coherence (Success) | 0.1251 |
| Coherence (Failure) | 0.0370 |
| Gradient Magnitude (Success) | 0.1934 |
| Gradient Magnitude (Failure) | 0.1900 |
| Activation Separation | 3.8896 |
| Cosine Distance | 0.0219 |
| Clusters | 1,104 |
| Noise Fraction | 0.3439 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep3 | 3 | 0.2299 | 0.8689 | 0.4556 | 0.0893 | 0.0351 | 0.1624 | 0.0010 | — |
| collect_sapling @ ep4 | 4 | 0.1078 | 0.8868 | 0.2095 | 0.0861 | 0.0366 | 0.1859 | 0.0012 | 0.0000 |
| place_plant @ ep4 | 4 | 0.0644 | 0.8561 | 0.1817 | 0.0815 | 0.0370 | 0.1727 | 0.0009 | nan |
| collect_wood @ ep17 | 17 | 0.2779 | 0.9183 | 0.7394 | 0.0895 | 0.0439 | 0.1553 | 0.0009 | nan |
| place_table @ ep17 | 17 | 0.0767 | 0.9202 | 0.1961 | 0.0866 | 0.0352 | 0.1506 | 0.0007 | nan |
| collect_drink @ ep19 | 19 | 0.3079 | 0.9084 | -0.0201 | 0.1467 | 0.0387 | 0.2817 | 0.0007 | -0.6547 |
| make_wood_sword @ ep34 | 34 | 0.5666 | 0.9439 | 0.2863 | 0.2146 | 0.0879 | 0.3303 | 0.0005 | -0.6547 |
| make_wood_pickaxe @ ep50 | 50 | 0.6616 | 0.9350 | 0.3729 | 0.2528 | 0.0725 | 0.4286 | 0.0003 | -0.4352 |
| defeat_zombie @ ep76 | 76 | 0.8228 | 0.9173 | 0.7367 | 0.2570 | 0.1449 | 0.5409 | 0.0002 | 0.0000 |
| eat_cow @ ep158 | 158 | 0.6217 | 0.7229 | 0.4277 | 0.2653 | 0.2397 | 0.8008 | 0.0002 | -0.4352 |
| collect_stone @ ep215 | 215 | -0.1831 | 0.0012 | 0.0195 | 0.1413 | 0.1578 | 1.2858 | 0.0001 | -0.1745 |
| defeat_skeleton @ ep707 | 707 | -0.4419 | 0.2309 | 0.1662 | 0.1810 | 0.1708 | 1.2939 | 0.0003 | -0.1108 |
| eat_plant @ ep1138 | 1,138 | -0.7358 | 0.1504 | 0.1424 | 0.2421 | 0.1834 | 1.0028 | 0.0008 | -0.0870 |
| make_stone_pickaxe @ ep1202 | 1,202 | -0.3702 | -0.0154 | 0.3743 | 0.0647 | 0.2183 | 0.7943 | 0.0003 | -0.3482 |
| place_stone @ ep1716 | 1,716 | 0.5038 | 0.1161 | 0.3617 | 0.0917 | 0.2072 | 0.5041 | 0.0010 | -0.1745 |
| make_stone_sword @ ep1895 | 1,895 | 0.5022 | 0.2314 | 0.3301 | 0.1570 | 0.3205 | 1.0515 | 0.0037 | -0.2791 |
| collect_coal @ ep1898 | 1,898 | 0.5807 | 0.3724 | 0.2984 | 0.1634 | 0.2574 | 0.7331 | 0.0019 | -0.2443 |
| collect_iron @ ep2571 | 2,571 | 0.1505 | 0.3086 | 0.0321 | 0.1653 | 0.0923 | 1.5415 | 0.0062 | -0.1292 |
| place_furnace @ ep2988 | 2,988 | 0.5970 | 0.1711 | 0.1163 | 0.1315 | 0.1439 | 1.3567 | 0.0031 | 0.0185 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic_step50000_ep291 | 291 | 0.1000 | 0.2593 | 0.0574 | 0.2196 | 0.1342 | 1.3244 | 0.0001 | -0.4352 |
| periodic_step100000_ep576 | 576 | 0.4186 | 0.0304 | 0.3007 | 0.1466 | 0.1710 | 1.5008 | 0.0002 | -0.1047 |
| periodic_step150000_ep862 | 862 | 0.3293 | 0.2447 | 0.4175 | 0.2120 | 0.2775 | 1.5970 | 0.0006 | -0.1741 |
| periodic_step200000_ep1142 | 1,142 | -0.4306 | 0.2715 | 0.0636 | 0.1884 | 0.1023 | 1.1240 | 0.0014 | -0.2094 |
| periodic_step250000_ep1422 | 1,422 | 0.2588 | 0.4969 | 0.1612 | 0.1708 | 0.1511 | 0.4801 | 0.0006 | -0.4352 |
| periodic_step300000_ep1707 | 1,707 | -0.3561 | 0.3763 | 0.2096 | 0.1786 | 0.1802 | 0.5087 | 0.0009 | -0.2094 |
| periodic_step350000_ep2000 | 2,000 | -0.1553 | 0.0927 | 0.0758 | 0.1168 | 0.2284 | 0.8877 | 0.0028 | -0.2094 |
| periodic_step400000_ep2273 | 2,273 | 0.2813 | 0.0089 | 0.1542 | 0.1015 | 0.2131 | 1.2697 | 0.0027 | 0.0489 |
| periodic_step450000_ep2554 | 2,554 | 0.0268 | 0.1808 | -0.0237 | 0.1190 | 0.0715 | 1.1807 | 0.0035 | -0.1477 |
| periodic_step500000_ep2835 | 2,835 | 0.1260 | 0.1060 | 0.1902 | 0.1059 | 0.2007 | 1.3341 | 0.0044 | -0.0923 |
| periodic_step550000_ep3099 | 3,099 | -0.6095 | 0.0504 | 0.0253 | 0.1335 | 0.1837 | 1.6569 | 0.0048 | 0.1662 |
| periodic_step600000_ep3365 | 3,365 | 0.3271 | 0.1586 | 0.0907 | 0.1602 | 0.1838 | 1.9544 | 0.0049 | 0.0000 |
| periodic_step650000_ep3625 | 3,625 | -0.3885 | 0.1608 | 0.1017 | 0.1415 | 0.1699 | 2.1696 | 0.0086 | 0.0000 |
| periodic_step700000_ep3894 | 3,894 | 0.0308 | 0.0372 | 0.0184 | 0.1343 | 0.1377 | 1.7564 | 0.0034 | 0.0000 |
| periodic_step750000_ep4172 | 4,172 | 0.2495 | 0.1759 | 0.0160 | 0.1547 | 0.1402 | 2.6410 | 0.0109 | -0.1047 |
| periodic_step800000_ep4435 | 4,435 | 0.4308 | 0.1816 | 0.0135 | 0.1941 | 0.1437 | 2.4620 | 0.0096 | 0.0185 |
| periodic_step850000_ep4702 | 4,702 | 0.4297 | 0.0900 | 0.1876 | 0.1391 | 0.2038 | 2.3362 | 0.0100 | 0.0923 |
| periodic_step900000_ep4975 | 4,975 | 0.3476 | 0.2021 | 0.1118 | 0.1735 | 0.2035 | 2.7955 | 0.0123 | -0.0369 |
| periodic_step950000_ep5233 | 5,233 | 0.1645 | 0.1949 | 0.0126 | 0.1582 | 0.1273 | 2.3783 | 0.0095 | 0.0185 |
| periodic_step1000000_ep5503 | 5,503 | -0.1115 | 0.1484 | 0.0110 | 0.1508 | 0.1138 | 2.3243 | 0.0101 | 0.0923 |
| periodic_step1050000_ep5780 | 5,780 | 0.6801 | 0.3161 | 0.1209 | 0.2798 | 0.2235 | 3.1478 | 0.0129 | -0.1477 |
| periodic_step1100000_ep6058 | 6,058 | 0.6218 | 0.2729 | 0.0943 | 0.2159 | 0.1950 | 2.8083 | 0.0145 | -0.0923 |
| periodic_step1150000_ep6336 | 6,336 | 0.1169 | 0.1242 | 0.0967 | 0.1496 | 0.1650 | 2.8634 | 0.0165 | -0.0923 |
| periodic_step1200000_ep6602 | 6,602 | 0.8315 | 0.3092 | 0.1912 | 0.2678 | 0.3231 | 3.3159 | 0.0206 | -0.0923 |
| periodic_step1250000_ep6872 | 6,872 | 0.2527 | 0.0823 | 0.0946 | 0.1449 | 0.1985 | 3.2089 | 0.0208 | -0.0554 |
| periodic_step1300000_ep7137 | 7,137 | 0.1392 | 0.0254 | -0.0104 | 0.1091 | 0.1176 | 3.4716 | 0.0170 | 0.2094 |
| periodic_step1350000_ep7403 | 7,403 | 0.5658 | 0.1662 | 0.0516 | 0.1812 | 0.1824 | 3.8130 | 0.0241 | 0.0698 |
| periodic_step1400000_ep7676 | 7,676 | 0.5607 | 0.1295 | 0.1921 | 0.1655 | 0.2809 | 3.6741 | 0.0217 | 0.2443 |
| periodic_step1450000_ep7950 | 7,950 | 0.5975 | 0.2134 | 0.1345 | 0.2056 | 0.2394 | 3.4319 | 0.0256 | 0.2443 |
| periodic_step1500000_ep8228 | 8,228 | 0.4416 | 0.1334 | 0.0868 | 0.1685 | 0.1761 | 3.3323 | 0.0167 | 0.3838 |
| periodic_step1550000_ep8502 | 8,502 | -0.0438 | 0.0963 | 0.0424 | 0.1596 | 0.2013 | 3.6967 | 0.0195 | 0.1745 |
| periodic_step1600000_ep8781 | 8,781 | 0.8057 | 0.4570 | 0.1611 | 0.3623 | 0.3023 | 4.9855 | 0.0324 | 0.1468 |
| periodic_step1650000_ep9045 | 9,045 | 0.3354 | 0.0802 | -0.0021 | 0.1385 | 0.1328 | 5.2518 | 0.0404 | 0.1292 |
| periodic_step1700000_ep9321 | 9,321 | 0.5214 | 0.1494 | 0.1940 | 0.1575 | 0.2516 | 5.1845 | 0.0400 | 0.1662 |
| periodic_step1750000_ep9599 | 9,599 | 0.2442 | 0.0440 | 0.0688 | 0.1320 | 0.1880 | 5.3505 | 0.0325 | 0.3140 |
| periodic_step1800000_ep9876 | 9,876 | 0.4155 | 0.0299 | 0.0248 | 0.1175 | 0.1803 | 4.7865 | 0.0306 | 0.0923 |
| periodic_step1850000_ep10151 | 10,151 | 0.1026 | 0.1125 | -0.0248 | 0.1497 | 0.1320 | 5.4388 | 0.0398 | 0.1662 |
| periodic_step1900000_ep10418 | 10,418 | 0.0930 | 0.0605 | 0.0403 | 0.1395 | 0.1619 | 5.3784 | 0.0388 | 0.0923 |
| periodic_step1950000_ep10683 | 10,683 | 0.5810 | 0.1474 | 0.0965 | 0.1821 | 0.3260 | 4.6181 | 0.0293 | 0.1662 |
| periodic_step2000000_ep10934 | 10,934 | 0.1488 | 0.2331 | -0.0140 | 0.2804 | 0.1512 | 4.3192 | 0.0245 | 0.0979 |
| periodic_step2050000_ep11185 | 11,185 | 0.1551 | 0.0305 | 0.0343 | 0.1205 | 0.1547 | 4.2872 | 0.0243 | 0.2216 |
| periodic_step2100000_ep11442 | 11,442 | 0.3026 | 0.0701 | 0.0378 | 0.1407 | 0.1743 | 4.9053 | 0.0291 | 0.0685 |
| periodic_step2150000_ep11702 | 11,702 | 0.4186 | 0.1527 | 0.0441 | 0.1699 | 0.1864 | 4.7274 | 0.0323 | 0.1174 |
| periodic_step2200000_ep11958 | 11,958 | 0.3223 | 0.1554 | 0.0120 | 0.1776 | 0.1704 | 4.8911 | 0.0384 | 0.1662 |
| periodic_step2250000_ep12223 | 12,223 | 0.2247 | 0.1523 | 0.0459 | 0.1819 | 0.1840 | 4.4895 | 0.0300 | 0.1477 |
| periodic_step2300000_ep12486 | 12,486 | 0.5989 | 0.2116 | 0.0443 | 0.2025 | 0.1638 | 4.1003 | 0.0243 | 0.1292 |
| periodic_step2350000_ep12740 | 12,740 | 0.7570 | 0.1153 | 0.1464 | 0.1892 | 0.2520 | 3.6838 | 0.0192 | 0.3140 |
| periodic_step2400000_ep12995 | 12,995 | 0.1815 | 0.1186 | 0.0789 | 0.1504 | 0.1912 | 3.9960 | 0.0284 | 0.1846 |
| periodic_step2450000_ep13252 | 13,252 | 0.3812 | 0.1111 | 0.0190 | 0.1767 | 0.1869 | 4.0630 | 0.0174 | 0.2031 |
| periodic_step2500000_ep13506 | 13,506 | 0.4656 | 0.0773 | 0.0862 | 0.1665 | 0.2630 | 3.8888 | 0.0199 | 0.1846 |
| periodic_step2550000_ep13776 | 13,776 | 0.4918 | 0.2948 | 0.0359 | 0.2806 | 0.1941 | 3.7805 | 0.0210 | 0.1957 |
| periodic_step2600000_ep14032 | 14,032 | 0.2942 | 0.1041 | 0.0592 | 0.1808 | 0.2096 | 3.5910 | 0.0154 | 0.2031 |
| periodic_step2650000_ep14270 | 14,270 | 0.0596 | 0.1318 | 0.0256 | 0.1678 | 0.1608 | 3.6763 | 0.0181 | 0.2216 |
| periodic_step2700000_ep14515 | 14,515 | 0.5698 | 0.2640 | 0.0980 | 0.2759 | 0.2595 | 4.7524 | 0.0279 | 0.1292 |
| periodic_step2750000_ep14755 | 14,755 | 0.1043 | 0.0421 | 0.0258 | 0.1471 | 0.1815 | 4.0581 | 0.0201 | 0.2216 |
| periodic_step2800000_ep14998 | 14,998 | 0.2954 | 0.0868 | -0.0167 | 0.1548 | 0.1452 | 3.9083 | 0.0183 | 0.2031 |
| periodic_step2850000_ep15241 | 15,241 | -0.0783 | 0.0797 | 0.0774 | 0.1534 | 0.2073 | 4.2013 | 0.0223 | 0.2031 |
| periodic_step2900000_ep15483 | 15,483 | 0.0634 | 0.0397 | 0.0776 | 0.1350 | 0.1940 | 3.7338 | 0.0217 | 0.2031 |
| periodic_step2950000_ep15719 | 15,719 | 0.5158 | 0.0384 | 0.0458 | 0.1600 | 0.2240 | 3.6664 | 0.0191 | 0.2216 |
| periodic_step3000000_ep15953 | 15,953 | 0.1579 | 0.0791 | 0.0129 | 0.1760 | 0.1747 | 3.8672 | 0.0205 | 0.1846 |
| final_step3000320_ep15954 | 15,954 | 0.6631 | 0.1251 | 0.0370 | 0.1934 | 0.1900 | 3.8896 | 0.0219 | 0.1846 |
