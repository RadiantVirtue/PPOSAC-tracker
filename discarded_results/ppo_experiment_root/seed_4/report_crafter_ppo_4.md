# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 4  
**Total episodes:** 15,067  
**Experiment root:** `experiment_root\seed_4`  
**Generated:** 2026-03-22 18:08

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep2 | 2 | 0.2564 | 0.6709 | 0.1906 | 0.0819 | 0.0467 | 0.1432 | 0.0006 | nan |
| collect_sapling @ ep4 | 4 | 0.0775 | 0.9053 | 0.2802 | 0.1029 | 0.0458 | 0.1936 | 0.0012 | nan |
| place_plant @ ep4 | 4 | 0.0947 | 0.8852 | 0.3782 | 0.1023 | 0.0450 | 0.1745 | 0.0009 | nan |
| collect_wood @ ep7 | 7 | -0.0822 | 0.8830 | 0.4171 | 0.1019 | 0.0430 | 0.2234 | 0.0013 | — |
| eat_cow @ ep9 | 9 | 0.1532 | 0.8305 | 0.4948 | 0.0956 | 0.0476 | 0.1857 | 0.0011 | 0.0000 |
| collect_drink @ ep14 | 14 | 0.2326 | 0.8907 | 0.5863 | 0.0962 | 0.0534 | 0.2056 | 0.0012 | -0.6547 |
| place_table @ ep48 | 48 | 0.4195 | 0.9329 | 0.0273 | 0.2295 | 0.0638 | 0.4353 | 0.0012 | nan |
| make_wood_sword @ ep66 | 66 | 0.8175 | 0.9129 | 0.4397 | 0.2473 | 0.1070 | 0.4324 | 0.0005 | nan |
| defeat_zombie @ ep120 | 120 | 0.7394 | 0.8315 | 0.3137 | 0.2312 | 0.1762 | 0.4990 | 0.0005 | -0.4352 |
| make_wood_pickaxe @ ep135 | 135 | 0.7881 | 0.8016 | 0.7356 | 0.2139 | 0.2153 | 0.4844 | 0.0003 | -0.3928 |
| defeat_skeleton @ ep235 | 235 | 0.0139 | 0.0817 | -0.0109 | 0.1662 | 0.1462 | 0.8094 | 0.0003 | -0.1396 |
| periodic_step50000_ep290 | 290 | 0.5173 | 0.1494 | 0.0342 | 0.1599 | 0.1924 | 0.6082 | 0.0002 | -0.2740 |
| collect_stone @ ep436 | 436 | 0.3255 | 0.3392 | 0.2373 | 0.1862 | 0.1555 | 0.4600 | 0.0002 | -0.1745 |
| periodic_step100000_ep574 | 574 | -0.2612 | 0.1619 | 0.1364 | 0.1327 | 0.2329 | 0.6573 | 0.0007 | -0.1047 |
| make_stone_pickaxe @ ep788 | 788 | 0.7166 | 0.2125 | 0.2015 | 0.1873 | 0.2424 | 0.5583 | 0.0003 | 0.0000 |
| periodic_step150000_ep847 | 847 | 0.5337 | 0.2173 | 0.0462 | 0.2003 | 0.0993 | 1.7051 | 0.0024 | 0.0000 |
| collect_coal @ ep990 | 990 | -0.2752 | 0.1203 | 0.2580 | 0.1183 | 0.2202 | 0.3440 | 0.0002 | -0.4187 |
| periodic_step200000_ep1132 | 1,132 | 0.3334 | 0.1396 | 0.5108 | 0.0955 | 0.3552 | 0.7913 | 0.0004 | -0.3838 |
| eat_plant @ ep1194 | 1,194 | -0.4643 | 0.1694 | 0.1560 | 0.1108 | 0.2457 | 1.2762 | 0.0008 | -0.2443 |
| place_stone @ ep1333 | 1,333 | 0.3159 | 0.2365 | 0.0801 | 0.1230 | 0.1262 | 0.6799 | 0.0004 | -0.1662 |
| periodic_step250000_ep1407 | 1,407 | 0.2551 | 0.0358 | 0.3606 | 0.1003 | 0.2527 | 0.8756 | 0.0006 | -0.0698 |
| periodic_step300000_ep1687 | 1,687 | -0.0448 | 0.2392 | 0.0041 | 0.1372 | 0.1321 | 0.8710 | 0.0008 | -0.2791 |
| periodic_step350000_ep1967 | 1,967 | 0.5712 | 0.3474 | 0.2748 | 0.1624 | 0.2836 | 0.9789 | 0.0019 | -0.1396 |
| periodic_step400000_ep2245 | 2,245 | -0.0462 | 0.1780 | 0.2832 | 0.2611 | 0.2694 | 1.3384 | 0.0018 | -0.2791 |
| place_furnace @ ep2357 | 2,357 | 0.0545 | 0.1027 | 0.0409 | 0.1342 | 0.1673 | 1.1679 | 0.0015 | -0.2791 |
| periodic_step450000_ep2527 | 2,527 | 0.6360 | 0.3114 | 0.2995 | 0.1698 | 0.3740 | 1.3472 | 0.0017 | -0.1292 |
| periodic_step500000_ep2818 | 2,818 | -0.4294 | 0.0284 | 0.0810 | 0.1172 | 0.1850 | 1.7981 | 0.0055 | -0.3489 |
| periodic_step550000_ep3084 | 3,084 | -0.3945 | 0.0264 | 0.2211 | 0.1185 | 0.2689 | 1.8233 | 0.0042 | 0.0000 |
| periodic_step600000_ep3343 | 3,343 | -0.1108 | 0.1199 | 0.1573 | 0.1167 | 0.1946 | 1.4024 | 0.0036 | -0.0349 |
| periodic_step650000_ep3610 | 3,610 | 0.7166 | 0.1749 | 0.2929 | 0.1529 | 0.3717 | 2.1466 | 0.0081 | -0.1396 |
| periodic_step700000_ep3880 | 3,880 | 0.7624 | 0.5032 | 0.1965 | 0.3141 | 0.2816 | 3.6270 | 0.0180 | -0.2443 |
| periodic_step750000_ep4146 | 4,146 | 0.5472 | 0.0712 | 0.2388 | 0.1239 | 0.2984 | 2.1844 | 0.0059 | -0.2791 |
| periodic_step800000_ep4420 | 4,420 | 0.1319 | 0.1209 | 0.0407 | 0.1627 | 0.1554 | 1.5482 | 0.0032 | -0.1396 |
| periodic_step850000_ep4690 | 4,690 | 0.3728 | 0.2324 | 0.1445 | 0.2009 | 0.2450 | 3.1699 | 0.0165 | -0.0739 |
| periodic_step900000_ep4948 | 4,948 | 0.6059 | 0.2939 | 0.1199 | 0.2267 | 0.2525 | 2.4538 | 0.0097 | -0.2216 |
| periodic_step950000_ep5226 | 5,226 | 0.7981 | 0.1562 | 0.3029 | 0.1732 | 0.3152 | 1.8955 | 0.0046 | 0.0000 |
| periodic_step1000000_ep5480 | 5,480 | 0.3638 | 0.2822 | 0.1527 | 0.2082 | 0.2489 | 2.6252 | 0.0093 | 0.0000 |
| periodic_step1050000_ep5721 | 5,721 | 0.8794 | 0.4286 | 0.2194 | 0.2743 | 0.3071 | 2.9189 | 0.0111 | -0.0923 |
| periodic_step1100000_ep5982 | 5,982 | 0.4848 | 0.1496 | 0.1718 | 0.1658 | 0.2657 | 2.7209 | 0.0093 | 0.0000 |
| make_stone_sword @ ep6079 | 6,079 | -0.1982 | 0.1151 | 0.0159 | 0.1526 | 0.1370 | 2.4905 | 0.0089 | 0.3508 |
| periodic_step1150000_ep6234 | 6,234 | 0.7171 | 0.2424 | 0.1114 | 0.1972 | 0.2004 | 2.9001 | 0.0123 | 0.0554 |
| periodic_step1200000_ep6494 | 6,494 | 0.2283 | 0.0669 | 0.1909 | 0.1152 | 0.2898 | 3.6075 | 0.0223 | -0.0739 |
| periodic_step1250000_ep6755 | 6,755 | 0.0030 | 0.0013 | 0.0924 | 0.0975 | 0.1909 | 3.9879 | 0.0252 | -0.0554 |
| periodic_step1300000_ep7011 | 7,011 | -0.0188 | 0.0428 | 0.0449 | 0.1184 | 0.1739 | 4.2614 | 0.0206 | 0.1047 |
| periodic_step1350000_ep7271 | 7,271 | 0.5210 | 0.1161 | 0.2047 | 0.1440 | 0.2924 | 3.2006 | 0.0142 | 0.3489 |
| periodic_step1400000_ep7530 | 7,530 | 0.5683 | 0.1259 | 0.1368 | 0.1539 | 0.2286 | 3.8229 | 0.0206 | 0.0185 |
| periodic_step1450000_ep7786 | 7,786 | 0.6799 | 0.3173 | 0.1615 | 0.2896 | 0.3565 | 4.2283 | 0.0216 | 0.3140 |
| periodic_step1500000_ep8053 | 8,053 | 0.2491 | 0.1053 | 0.0412 | 0.1365 | 0.1633 | 3.9179 | 0.0192 | 0.3693 |
| periodic_step1550000_ep8308 | 8,308 | 0.2697 | 0.1029 | -0.0047 | 0.1358 | 0.1391 | 4.0048 | 0.0236 | 0.2443 |
| periodic_step1600000_ep8557 | 8,557 | 0.1953 | 0.1327 | 0.0912 | 0.1768 | 0.2440 | 4.8026 | 0.0347 | 0.0000 |
| periodic_step1650000_ep8797 | 8,797 | 0.0732 | 0.0885 | 0.0049 | 0.1490 | 0.1478 | 4.2420 | 0.0275 | 0.2443 |
| periodic_step1700000_ep9046 | 9,046 | 0.5423 | 0.2748 | 0.0355 | 0.2455 | 0.2103 | 3.8829 | 0.0240 | 0.2094 |
| periodic_step1750000_ep9290 | 9,290 | 0.4770 | 0.0980 | -0.0164 | 0.1663 | 0.1598 | 4.1354 | 0.0219 | 0.2094 |
| periodic_step1800000_ep9536 | 9,536 | 0.8261 | 0.2378 | 0.1188 | 0.2438 | 0.2459 | 3.5912 | 0.0233 | 0.0185 |
| periodic_step1850000_ep9782 | 9,782 | 0.2335 | 0.0980 | 0.0520 | 0.1541 | 0.2045 | 4.2677 | 0.0274 | 0.3140 |
| periodic_step1900000_ep10030 | 10,030 | 0.2766 | 0.3281 | 0.0040 | 0.2778 | 0.1657 | 3.4292 | 0.0173 | 0.1292 |
| periodic_step1950000_ep10273 | 10,273 | 0.7465 | 0.0813 | 0.2007 | 0.1700 | 0.3390 | 4.6274 | 0.0300 | 0.1745 |
| periodic_step2000000_ep10525 | 10,525 | 0.5032 | 0.1838 | 0.0453 | 0.1952 | 0.1992 | 4.9164 | 0.0265 | 0.2443 |
| periodic_step2050000_ep10759 | 10,759 | 0.5590 | 0.2321 | 0.0209 | 0.2195 | 0.1957 | 4.0056 | 0.0244 | 0.2791 |
| periodic_step2100000_ep11027 | 11,027 | 0.1028 | 0.0504 | 0.0307 | 0.1222 | 0.1804 | 4.2031 | 0.0279 | 0.2094 |
| periodic_step2150000_ep11269 | 11,269 | 0.2340 | 0.0626 | 0.0817 | 0.1475 | 0.2240 | 3.9944 | 0.0232 | 0.2954 |
| periodic_step2200000_ep11497 | 11,497 | 0.6003 | 0.3862 | 0.1439 | 0.3308 | 0.2918 | 4.2398 | 0.0209 | 0.2791 |
| periodic_step2250000_ep11718 | 11,718 | 0.7705 | 0.3187 | 0.0360 | 0.2613 | 0.2180 | 3.8330 | 0.0189 | 0.0185 |
| periodic_step2300000_ep11949 | 11,949 | 0.4204 | 0.1907 | 0.0686 | 0.1987 | 0.2490 | 4.1480 | 0.0168 | 0.2400 |
| periodic_step2350000_ep12164 | 12,164 | -0.0861 | 0.1112 | 0.0349 | 0.1448 | 0.1834 | 4.8070 | 0.0285 | 0.1292 |
| periodic_step2400000_ep12367 | 12,367 | 0.5831 | 0.0904 | 0.0928 | 0.1654 | 0.2305 | 3.5086 | 0.0174 | 0.1846 |
| periodic_step2450000_ep12596 | 12,596 | 0.2521 | 0.2739 | 0.0188 | 0.2262 | 0.1686 | 3.9673 | 0.0235 | 0.2031 |
| periodic_step2500000_ep12823 | 12,823 | 0.7245 | 0.2734 | 0.0879 | 0.2603 | 0.2517 | 3.3367 | 0.0160 | 0.2770 |
| periodic_step2550000_ep13047 | 13,047 | 0.4111 | 0.3216 | 0.1203 | 0.2677 | 0.2508 | 3.4645 | 0.0181 | 0.3230 |
| periodic_step2600000_ep13278 | 13,278 | 0.4214 | 0.2687 | 0.1099 | 0.2317 | 0.2347 | 3.9337 | 0.0231 | 0.2400 |
| periodic_step2650000_ep13516 | 13,516 | 0.5829 | 0.2424 | 0.2337 | 0.2171 | 0.3378 | 4.7152 | 0.0279 | 0.0587 |
| periodic_step2700000_ep13759 | 13,759 | 0.1987 | 0.0352 | 0.0628 | 0.1239 | 0.2236 | 4.6019 | 0.0344 | 0.1846 |
| collect_iron @ ep13849 | 13,849 | 0.3191 | 0.1383 | 0.0120 | 0.1683 | 0.1520 | 3.7950 | 0.0199 | 0.2216 |
| periodic_step2750000_ep13994 | 13,994 | 0.4067 | 0.2607 | 0.0356 | 0.2395 | 0.1694 | 4.5376 | 0.0334 | 0.3230 |
| periodic_step2800000_ep14217 | 14,217 | 0.1310 | 0.0239 | 0.0034 | 0.1182 | 0.1325 | 4.3590 | 0.0299 | 0.2216 |
| periodic_step2850000_ep14436 | 14,436 | 0.4291 | 0.1103 | 0.1220 | 0.1565 | 0.2380 | 4.4715 | 0.0337 | 0.1860 |
| periodic_step2900000_ep14654 | 14,654 | 0.2181 | 0.1303 | 0.0715 | 0.1450 | 0.2282 | 4.5782 | 0.0329 | 0.2447 |
| periodic_step2950000_ep14855 | 14,855 | 0.5805 | 0.1072 | 0.0298 | 0.1710 | 0.1969 | 3.9585 | 0.0252 | 0.0000 |
| periodic_step3000000_ep15067 | 15,067 | 0.7307 | 0.1853 | 0.1712 | 0.2299 | 0.3172 | 3.7326 | 0.0200 | 0.1860 |
| final_step3000320_ep15067 | 15,067 | 0.5955 | 0.4167 | 0.0568 | 0.3236 | 0.2202 | 3.2625 | 0.0145 | 0.1077 |

---

## wake_up_ep2_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2564 |
| Coherence (Success) | 0.6709 |
| Coherence (Failure) | 0.1906 |
| Gradient Magnitude (Success) | 0.0819 |
| Gradient Magnitude (Failure) | 0.0467 |
| Activation Separation | 0.1432 |
| Cosine Distance | 0.0006 |
| Clusters | 1,402 |
| Noise Fraction | 0.2735 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep4_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0775 |
| Coherence (Success) | 0.9053 |
| Coherence (Failure) | 0.2802 |
| Gradient Magnitude (Success) | 0.1029 |
| Gradient Magnitude (Failure) | 0.0458 |
| Activation Separation | 0.1936 |
| Cosine Distance | 0.0012 |
| Clusters | 1,399 |
| Noise Fraction | 0.2650 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep4_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0947 |
| Coherence (Success) | 0.8852 |
| Coherence (Failure) | 0.3782 |
| Gradient Magnitude (Success) | 0.1023 |
| Gradient Magnitude (Failure) | 0.0450 |
| Activation Separation | 0.1745 |
| Cosine Distance | 0.0009 |
| Clusters | 1,337 |
| Noise Fraction | 0.2491 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_wood_ep7_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0822 |
| Coherence (Success) | 0.8830 |
| Coherence (Failure) | 0.4171 |
| Gradient Magnitude (Success) | 0.1019 |
| Gradient Magnitude (Failure) | 0.0430 |
| Activation Separation | 0.2234 |
| Cosine Distance | 0.0013 |
| Clusters | 1,289 |
| Noise Fraction | 0.2589 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## eat_cow_ep9_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1532 |
| Coherence (Success) | 0.8305 |
| Coherence (Failure) | 0.4948 |
| Gradient Magnitude (Success) | 0.0956 |
| Gradient Magnitude (Failure) | 0.0476 |
| Activation Separation | 0.1857 |
| Cosine Distance | 0.0011 |
| Clusters | 1,405 |
| Noise Fraction | 0.2749 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (3) | Stone, Wood, Wood Pickaxe |

---

## collect_drink_ep14_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2326 |
| Coherence (Success) | 0.8907 |
| Coherence (Failure) | 0.5863 |
| Gradient Magnitude (Success) | 0.0962 |
| Gradient Magnitude (Failure) | 0.0534 |
| Activation Separation | 0.2056 |
| Cosine Distance | 0.0012 |
| Clusters | 1,353 |
| Noise Fraction | 0.2707 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep48_lower1.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4195 |
| Coherence (Success) | 0.9329 |
| Coherence (Failure) | 0.0273 |
| Gradient Magnitude (Success) | 0.2295 |
| Gradient Magnitude (Failure) | 0.0638 |
| Activation Separation | 0.4353 |
| Cosine Distance | 0.0012 |
| Clusters | 1,361 |
| Noise Fraction | 0.2332 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep66_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8175 |
| Coherence (Success) | 0.9129 |
| Coherence (Failure) | 0.4397 |
| Gradient Magnitude (Success) | 0.2473 |
| Gradient Magnitude (Failure) | 0.1070 |
| Activation Separation | 0.4324 |
| Cosine Distance | 0.0005 |
| Clusters | 1,430 |
| Noise Fraction | 0.2370 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep120_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7394 |
| Coherence (Success) | 0.8315 |
| Coherence (Failure) | 0.3137 |
| Gradient Magnitude (Success) | 0.2312 |
| Gradient Magnitude (Failure) | 0.1762 |
| Activation Separation | 0.4990 |
| Cosine Distance | 0.0005 |
| Clusters | 1,520 |
| Noise Fraction | 0.2233 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep135_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7881 |
| Coherence (Success) | 0.8016 |
| Coherence (Failure) | 0.7356 |
| Gradient Magnitude (Success) | 0.2139 |
| Gradient Magnitude (Failure) | 0.2153 |
| Activation Separation | 0.4844 |
| Cosine Distance | 0.0003 |
| Clusters | 1,609 |
| Noise Fraction | 0.2140 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep235_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0139 |
| Coherence (Success) | 0.0817 |
| Coherence (Failure) | -0.0109 |
| Gradient Magnitude (Success) | 0.1662 |
| Gradient Magnitude (Failure) | 0.1462 |
| Activation Separation | 0.8094 |
| Cosine Distance | 0.0003 |
| Clusters | 1,627 |
| Noise Fraction | 0.2534 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep290_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5173 |
| Coherence (Success) | 0.1494 |
| Coherence (Failure) | 0.0342 |
| Gradient Magnitude (Success) | 0.1599 |
| Gradient Magnitude (Failure) | 0.1924 |
| Activation Separation | 0.6082 |
| Cosine Distance | 0.0002 |
| Clusters | 1,800 |
| Noise Fraction | 0.2254 |
| RSA Alignment (ρ) | -0.2740 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep436_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3255 |
| Coherence (Success) | 0.3392 |
| Coherence (Failure) | 0.2373 |
| Gradient Magnitude (Success) | 0.1862 |
| Gradient Magnitude (Failure) | 0.1555 |
| Activation Separation | 0.4600 |
| Cosine Distance | 0.0002 |
| Clusters | 1,678 |
| Noise Fraction | 0.2445 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep574_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2612 |
| Coherence (Success) | 0.1619 |
| Coherence (Failure) | 0.1364 |
| Gradient Magnitude (Success) | 0.1327 |
| Gradient Magnitude (Failure) | 0.2329 |
| Activation Separation | 0.6573 |
| Cosine Distance | 0.0007 |
| Clusters | 1,628 |
| Noise Fraction | 0.2593 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep788_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7166 |
| Coherence (Success) | 0.2125 |
| Coherence (Failure) | 0.2015 |
| Gradient Magnitude (Success) | 0.1873 |
| Gradient Magnitude (Failure) | 0.2424 |
| Activation Separation | 0.5583 |
| Cosine Distance | 0.0003 |
| Clusters | 1,552 |
| Noise Fraction | 0.2676 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep847_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5337 |
| Coherence (Success) | 0.2173 |
| Coherence (Failure) | 0.0462 |
| Gradient Magnitude (Success) | 0.2003 |
| Gradient Magnitude (Failure) | 0.0993 |
| Activation Separation | 1.7051 |
| Cosine Distance | 0.0024 |
| Clusters | 1,446 |
| Noise Fraction | 0.2989 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep990_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2752 |
| Coherence (Success) | 0.1203 |
| Coherence (Failure) | 0.2580 |
| Gradient Magnitude (Success) | 0.1183 |
| Gradient Magnitude (Failure) | 0.2202 |
| Activation Separation | 0.3440 |
| Cosine Distance | 0.0002 |
| Clusters | 1,636 |
| Noise Fraction | 0.2651 |
| RSA Alignment (ρ) | -0.4187 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1132_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3334 |
| Coherence (Success) | 0.1396 |
| Coherence (Failure) | 0.5108 |
| Gradient Magnitude (Success) | 0.0955 |
| Gradient Magnitude (Failure) | 0.3552 |
| Activation Separation | 0.7913 |
| Cosine Distance | 0.0004 |
| Clusters | 1,493 |
| Noise Fraction | 0.2801 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep1194_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4643 |
| Coherence (Success) | 0.1694 |
| Coherence (Failure) | 0.1560 |
| Gradient Magnitude (Success) | 0.1108 |
| Gradient Magnitude (Failure) | 0.2457 |
| Activation Separation | 1.2762 |
| Cosine Distance | 0.0008 |
| Clusters | 1,565 |
| Noise Fraction | 0.2760 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep1333_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3159 |
| Coherence (Success) | 0.2365 |
| Coherence (Failure) | 0.0801 |
| Gradient Magnitude (Success) | 0.1230 |
| Gradient Magnitude (Failure) | 0.1262 |
| Activation Separation | 0.6799 |
| Cosine Distance | 0.0004 |
| Clusters | 1,549 |
| Noise Fraction | 0.2745 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1407_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2551 |
| Coherence (Success) | 0.0358 |
| Coherence (Failure) | 0.3606 |
| Gradient Magnitude (Success) | 0.1003 |
| Gradient Magnitude (Failure) | 0.2527 |
| Activation Separation | 0.8756 |
| Cosine Distance | 0.0006 |
| Clusters | 1,471 |
| Noise Fraction | 0.2793 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1687_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0448 |
| Coherence (Success) | 0.2392 |
| Coherence (Failure) | 0.0041 |
| Gradient Magnitude (Success) | 0.1372 |
| Gradient Magnitude (Failure) | 0.1321 |
| Activation Separation | 0.8710 |
| Cosine Distance | 0.0008 |
| Clusters | 1,384 |
| Noise Fraction | 0.2835 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1967_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5712 |
| Coherence (Success) | 0.3474 |
| Coherence (Failure) | 0.2748 |
| Gradient Magnitude (Success) | 0.1624 |
| Gradient Magnitude (Failure) | 0.2836 |
| Activation Separation | 0.9789 |
| Cosine Distance | 0.0019 |
| Clusters | 1,557 |
| Noise Fraction | 0.2798 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2245_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0462 |
| Coherence (Success) | 0.1780 |
| Coherence (Failure) | 0.2832 |
| Gradient Magnitude (Success) | 0.2611 |
| Gradient Magnitude (Failure) | 0.2694 |
| Activation Separation | 1.3384 |
| Cosine Distance | 0.0018 |
| Clusters | 1,519 |
| Noise Fraction | 0.2789 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2357_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0545 |
| Coherence (Success) | 0.1027 |
| Coherence (Failure) | 0.0409 |
| Gradient Magnitude (Success) | 0.1342 |
| Gradient Magnitude (Failure) | 0.1673 |
| Activation Separation | 1.1679 |
| Cosine Distance | 0.0015 |
| Clusters | 1,355 |
| Noise Fraction | 0.3108 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2527_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6360 |
| Coherence (Success) | 0.3114 |
| Coherence (Failure) | 0.2995 |
| Gradient Magnitude (Success) | 0.1698 |
| Gradient Magnitude (Failure) | 0.3740 |
| Activation Separation | 1.3472 |
| Cosine Distance | 0.0017 |
| Clusters | 1,241 |
| Noise Fraction | 0.3441 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2818_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4294 |
| Coherence (Success) | 0.0284 |
| Coherence (Failure) | 0.0810 |
| Gradient Magnitude (Success) | 0.1172 |
| Gradient Magnitude (Failure) | 0.1850 |
| Activation Separation | 1.7981 |
| Cosine Distance | 0.0055 |
| Clusters | 1,537 |
| Noise Fraction | 0.2884 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3084_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3945 |
| Coherence (Success) | 0.0264 |
| Coherence (Failure) | 0.2211 |
| Gradient Magnitude (Success) | 0.1185 |
| Gradient Magnitude (Failure) | 0.2689 |
| Activation Separation | 1.8233 |
| Cosine Distance | 0.0042 |
| Clusters | 1,369 |
| Noise Fraction | 0.3228 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3343_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1108 |
| Coherence (Success) | 0.1199 |
| Coherence (Failure) | 0.1573 |
| Gradient Magnitude (Success) | 0.1167 |
| Gradient Magnitude (Failure) | 0.1946 |
| Activation Separation | 1.4024 |
| Cosine Distance | 0.0036 |
| Clusters | 1,516 |
| Noise Fraction | 0.2731 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3610_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7166 |
| Coherence (Success) | 0.1749 |
| Coherence (Failure) | 0.2929 |
| Gradient Magnitude (Success) | 0.1529 |
| Gradient Magnitude (Failure) | 0.3717 |
| Activation Separation | 2.1466 |
| Cosine Distance | 0.0081 |
| Clusters | 1,422 |
| Noise Fraction | 0.3064 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3880_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7624 |
| Coherence (Success) | 0.5032 |
| Coherence (Failure) | 0.1965 |
| Gradient Magnitude (Success) | 0.3141 |
| Gradient Magnitude (Failure) | 0.2816 |
| Activation Separation | 3.6270 |
| Cosine Distance | 0.0180 |
| Clusters | 1,470 |
| Noise Fraction | 0.3075 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4146_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5472 |
| Coherence (Success) | 0.0712 |
| Coherence (Failure) | 0.2388 |
| Gradient Magnitude (Success) | 0.1239 |
| Gradient Magnitude (Failure) | 0.2984 |
| Activation Separation | 2.1844 |
| Cosine Distance | 0.0059 |
| Clusters | 1,526 |
| Noise Fraction | 0.3042 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4420_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1319 |
| Coherence (Success) | 0.1209 |
| Coherence (Failure) | 0.0407 |
| Gradient Magnitude (Success) | 0.1627 |
| Gradient Magnitude (Failure) | 0.1554 |
| Activation Separation | 1.5482 |
| Cosine Distance | 0.0032 |
| Clusters | 1,461 |
| Noise Fraction | 0.2990 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4690_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3728 |
| Coherence (Success) | 0.2324 |
| Coherence (Failure) | 0.1445 |
| Gradient Magnitude (Success) | 0.2009 |
| Gradient Magnitude (Failure) | 0.2450 |
| Activation Separation | 3.1699 |
| Cosine Distance | 0.0165 |
| Clusters | 1,354 |
| Noise Fraction | 0.3419 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4948_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6059 |
| Coherence (Success) | 0.2939 |
| Coherence (Failure) | 0.1199 |
| Gradient Magnitude (Success) | 0.2267 |
| Gradient Magnitude (Failure) | 0.2525 |
| Activation Separation | 2.4538 |
| Cosine Distance | 0.0097 |
| Clusters | 1,350 |
| Noise Fraction | 0.3644 |
| RSA Alignment (ρ) | -0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5226_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7981 |
| Coherence (Success) | 0.1562 |
| Coherence (Failure) | 0.3029 |
| Gradient Magnitude (Success) | 0.1732 |
| Gradient Magnitude (Failure) | 0.3152 |
| Activation Separation | 1.8955 |
| Cosine Distance | 0.0046 |
| Clusters | 1,452 |
| Noise Fraction | 0.3144 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5480_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3638 |
| Coherence (Success) | 0.2822 |
| Coherence (Failure) | 0.1527 |
| Gradient Magnitude (Success) | 0.2082 |
| Gradient Magnitude (Failure) | 0.2489 |
| Activation Separation | 2.6252 |
| Cosine Distance | 0.0093 |
| Clusters | 1,560 |
| Noise Fraction | 0.3133 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5721_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8794 |
| Coherence (Success) | 0.4286 |
| Coherence (Failure) | 0.2194 |
| Gradient Magnitude (Success) | 0.2743 |
| Gradient Magnitude (Failure) | 0.3071 |
| Activation Separation | 2.9189 |
| Cosine Distance | 0.0111 |
| Clusters | 1,457 |
| Noise Fraction | 0.3295 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5982_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4848 |
| Coherence (Success) | 0.1496 |
| Coherence (Failure) | 0.1718 |
| Gradient Magnitude (Success) | 0.1658 |
| Gradient Magnitude (Failure) | 0.2657 |
| Activation Separation | 2.7209 |
| Cosine Distance | 0.0093 |
| Clusters | 1,395 |
| Noise Fraction | 0.3433 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep6079_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1982 |
| Coherence (Success) | 0.1151 |
| Coherence (Failure) | 0.0159 |
| Gradient Magnitude (Success) | 0.1526 |
| Gradient Magnitude (Failure) | 0.1370 |
| Activation Separation | 2.4905 |
| Cosine Distance | 0.0089 |
| Clusters | 1,450 |
| Noise Fraction | 0.3315 |
| RSA Alignment (ρ) | 0.3508 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6234_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7171 |
| Coherence (Success) | 0.2424 |
| Coherence (Failure) | 0.1114 |
| Gradient Magnitude (Success) | 0.1972 |
| Gradient Magnitude (Failure) | 0.2004 |
| Activation Separation | 2.9001 |
| Cosine Distance | 0.0123 |
| Clusters | 1,544 |
| Noise Fraction | 0.3324 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6494_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2283 |
| Coherence (Success) | 0.0669 |
| Coherence (Failure) | 0.1909 |
| Gradient Magnitude (Success) | 0.1152 |
| Gradient Magnitude (Failure) | 0.2898 |
| Activation Separation | 3.6075 |
| Cosine Distance | 0.0223 |
| Clusters | 1,394 |
| Noise Fraction | 0.3582 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6755_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0030 |
| Coherence (Success) | 0.0013 |
| Coherence (Failure) | 0.0924 |
| Gradient Magnitude (Success) | 0.0975 |
| Gradient Magnitude (Failure) | 0.1909 |
| Activation Separation | 3.9879 |
| Cosine Distance | 0.0252 |
| Clusters | 1,631 |
| Noise Fraction | 0.3346 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7011_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0188 |
| Coherence (Success) | 0.0428 |
| Coherence (Failure) | 0.0449 |
| Gradient Magnitude (Success) | 0.1184 |
| Gradient Magnitude (Failure) | 0.1739 |
| Activation Separation | 4.2614 |
| Cosine Distance | 0.0206 |
| Clusters | 1,480 |
| Noise Fraction | 0.3504 |
| RSA Alignment (ρ) | 0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7271_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5210 |
| Coherence (Success) | 0.1161 |
| Coherence (Failure) | 0.2047 |
| Gradient Magnitude (Success) | 0.1440 |
| Gradient Magnitude (Failure) | 0.2924 |
| Activation Separation | 3.2006 |
| Cosine Distance | 0.0142 |
| Clusters | 1,535 |
| Noise Fraction | 0.3205 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7530_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5683 |
| Coherence (Success) | 0.1259 |
| Coherence (Failure) | 0.1368 |
| Gradient Magnitude (Success) | 0.1539 |
| Gradient Magnitude (Failure) | 0.2286 |
| Activation Separation | 3.8229 |
| Cosine Distance | 0.0206 |
| Clusters | 1,225 |
| Noise Fraction | 0.3052 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7786_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6799 |
| Coherence (Success) | 0.3173 |
| Coherence (Failure) | 0.1615 |
| Gradient Magnitude (Success) | 0.2896 |
| Gradient Magnitude (Failure) | 0.3565 |
| Activation Separation | 4.2283 |
| Cosine Distance | 0.0216 |
| Clusters | 1,306 |
| Noise Fraction | 0.2943 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8053_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2491 |
| Coherence (Success) | 0.1053 |
| Coherence (Failure) | 0.0412 |
| Gradient Magnitude (Success) | 0.1365 |
| Gradient Magnitude (Failure) | 0.1633 |
| Activation Separation | 3.9179 |
| Cosine Distance | 0.0192 |
| Clusters | 1,374 |
| Noise Fraction | 0.3382 |
| RSA Alignment (ρ) | 0.3693 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8308_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2697 |
| Coherence (Success) | 0.1029 |
| Coherence (Failure) | -0.0047 |
| Gradient Magnitude (Success) | 0.1358 |
| Gradient Magnitude (Failure) | 0.1391 |
| Activation Separation | 4.0048 |
| Cosine Distance | 0.0236 |
| Clusters | 1,421 |
| Noise Fraction | 0.3318 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8557_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1953 |
| Coherence (Success) | 0.1327 |
| Coherence (Failure) | 0.0912 |
| Gradient Magnitude (Success) | 0.1768 |
| Gradient Magnitude (Failure) | 0.2440 |
| Activation Separation | 4.8026 |
| Cosine Distance | 0.0347 |
| Clusters | 1,416 |
| Noise Fraction | 0.3169 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8797_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0732 |
| Coherence (Success) | 0.0885 |
| Coherence (Failure) | 0.0049 |
| Gradient Magnitude (Success) | 0.1490 |
| Gradient Magnitude (Failure) | 0.1478 |
| Activation Separation | 4.2420 |
| Cosine Distance | 0.0275 |
| Clusters | 1,448 |
| Noise Fraction | 0.3342 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9046_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5423 |
| Coherence (Success) | 0.2748 |
| Coherence (Failure) | 0.0355 |
| Gradient Magnitude (Success) | 0.2455 |
| Gradient Magnitude (Failure) | 0.2103 |
| Activation Separation | 3.8829 |
| Cosine Distance | 0.0240 |
| Clusters | 1,352 |
| Noise Fraction | 0.3239 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9290_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4770 |
| Coherence (Success) | 0.0980 |
| Coherence (Failure) | -0.0164 |
| Gradient Magnitude (Success) | 0.1663 |
| Gradient Magnitude (Failure) | 0.1598 |
| Activation Separation | 4.1354 |
| Cosine Distance | 0.0219 |
| Clusters | 1,487 |
| Noise Fraction | 0.3258 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9536_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8261 |
| Coherence (Success) | 0.2378 |
| Coherence (Failure) | 0.1188 |
| Gradient Magnitude (Success) | 0.2438 |
| Gradient Magnitude (Failure) | 0.2459 |
| Activation Separation | 3.5912 |
| Cosine Distance | 0.0233 |
| Clusters | 1,403 |
| Noise Fraction | 0.3165 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9782_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2335 |
| Coherence (Success) | 0.0980 |
| Coherence (Failure) | 0.0520 |
| Gradient Magnitude (Success) | 0.1541 |
| Gradient Magnitude (Failure) | 0.2045 |
| Activation Separation | 4.2677 |
| Cosine Distance | 0.0274 |
| Clusters | 1,454 |
| Noise Fraction | 0.3219 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10030_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2766 |
| Coherence (Success) | 0.3281 |
| Coherence (Failure) | 0.0040 |
| Gradient Magnitude (Success) | 0.2778 |
| Gradient Magnitude (Failure) | 0.1657 |
| Activation Separation | 3.4292 |
| Cosine Distance | 0.0173 |
| Clusters | 1,364 |
| Noise Fraction | 0.3519 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10273_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7465 |
| Coherence (Success) | 0.0813 |
| Coherence (Failure) | 0.2007 |
| Gradient Magnitude (Success) | 0.1700 |
| Gradient Magnitude (Failure) | 0.3390 |
| Activation Separation | 4.6274 |
| Cosine Distance | 0.0300 |
| Clusters | 1,507 |
| Noise Fraction | 0.3176 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10525_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5032 |
| Coherence (Success) | 0.1838 |
| Coherence (Failure) | 0.0453 |
| Gradient Magnitude (Success) | 0.1952 |
| Gradient Magnitude (Failure) | 0.1992 |
| Activation Separation | 4.9164 |
| Cosine Distance | 0.0265 |
| Clusters | 1,510 |
| Noise Fraction | 0.3339 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10759_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5590 |
| Coherence (Success) | 0.2321 |
| Coherence (Failure) | 0.0209 |
| Gradient Magnitude (Success) | 0.2195 |
| Gradient Magnitude (Failure) | 0.1957 |
| Activation Separation | 4.0056 |
| Cosine Distance | 0.0244 |
| Clusters | 1,439 |
| Noise Fraction | 0.3303 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11027_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1028 |
| Coherence (Success) | 0.0504 |
| Coherence (Failure) | 0.0307 |
| Gradient Magnitude (Success) | 0.1222 |
| Gradient Magnitude (Failure) | 0.1804 |
| Activation Separation | 4.2031 |
| Cosine Distance | 0.0279 |
| Clusters | 1,322 |
| Noise Fraction | 0.3411 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11269_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2340 |
| Coherence (Success) | 0.0626 |
| Coherence (Failure) | 0.0817 |
| Gradient Magnitude (Success) | 0.1475 |
| Gradient Magnitude (Failure) | 0.2240 |
| Activation Separation | 3.9944 |
| Cosine Distance | 0.0232 |
| Clusters | 1,412 |
| Noise Fraction | 0.3506 |
| RSA Alignment (ρ) | 0.2954 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11497_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6003 |
| Coherence (Success) | 0.3862 |
| Coherence (Failure) | 0.1439 |
| Gradient Magnitude (Success) | 0.3308 |
| Gradient Magnitude (Failure) | 0.2918 |
| Activation Separation | 4.2398 |
| Cosine Distance | 0.0209 |
| Clusters | 1,328 |
| Noise Fraction | 0.3563 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11718_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7705 |
| Coherence (Success) | 0.3187 |
| Coherence (Failure) | 0.0360 |
| Gradient Magnitude (Success) | 0.2613 |
| Gradient Magnitude (Failure) | 0.2180 |
| Activation Separation | 3.8330 |
| Cosine Distance | 0.0189 |
| Clusters | 1,395 |
| Noise Fraction | 0.3458 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11949_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4204 |
| Coherence (Success) | 0.1907 |
| Coherence (Failure) | 0.0686 |
| Gradient Magnitude (Success) | 0.1987 |
| Gradient Magnitude (Failure) | 0.2490 |
| Activation Separation | 4.1480 |
| Cosine Distance | 0.0168 |
| Clusters | 1,535 |
| Noise Fraction | 0.3414 |
| RSA Alignment (ρ) | 0.2400 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12164_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0861 |
| Coherence (Success) | 0.1112 |
| Coherence (Failure) | 0.0349 |
| Gradient Magnitude (Success) | 0.1448 |
| Gradient Magnitude (Failure) | 0.1834 |
| Activation Separation | 4.8070 |
| Cosine Distance | 0.0285 |
| Clusters | 1,348 |
| Noise Fraction | 0.3430 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12367_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5831 |
| Coherence (Success) | 0.0904 |
| Coherence (Failure) | 0.0928 |
| Gradient Magnitude (Success) | 0.1654 |
| Gradient Magnitude (Failure) | 0.2305 |
| Activation Separation | 3.5086 |
| Cosine Distance | 0.0174 |
| Clusters | 1,426 |
| Noise Fraction | 0.3561 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12596_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2521 |
| Coherence (Success) | 0.2739 |
| Coherence (Failure) | 0.0188 |
| Gradient Magnitude (Success) | 0.2262 |
| Gradient Magnitude (Failure) | 0.1686 |
| Activation Separation | 3.9673 |
| Cosine Distance | 0.0235 |
| Clusters | 1,240 |
| Noise Fraction | 0.3403 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12823_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7245 |
| Coherence (Success) | 0.2734 |
| Coherence (Failure) | 0.0879 |
| Gradient Magnitude (Success) | 0.2603 |
| Gradient Magnitude (Failure) | 0.2517 |
| Activation Separation | 3.3367 |
| Cosine Distance | 0.0160 |
| Clusters | 1,149 |
| Noise Fraction | 0.3389 |
| RSA Alignment (ρ) | 0.2770 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13047_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4111 |
| Coherence (Success) | 0.3216 |
| Coherence (Failure) | 0.1203 |
| Gradient Magnitude (Success) | 0.2677 |
| Gradient Magnitude (Failure) | 0.2508 |
| Activation Separation | 3.4645 |
| Cosine Distance | 0.0181 |
| Clusters | 1,113 |
| Noise Fraction | 0.3056 |
| RSA Alignment (ρ) | 0.3230 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13278_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4214 |
| Coherence (Success) | 0.2687 |
| Coherence (Failure) | 0.1099 |
| Gradient Magnitude (Success) | 0.2317 |
| Gradient Magnitude (Failure) | 0.2347 |
| Activation Separation | 3.9337 |
| Cosine Distance | 0.0231 |
| Clusters | 1,222 |
| Noise Fraction | 0.3421 |
| RSA Alignment (ρ) | 0.2400 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13516_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5829 |
| Coherence (Success) | 0.2424 |
| Coherence (Failure) | 0.2337 |
| Gradient Magnitude (Success) | 0.2171 |
| Gradient Magnitude (Failure) | 0.3378 |
| Activation Separation | 4.7152 |
| Cosine Distance | 0.0279 |
| Clusters | 1,379 |
| Noise Fraction | 0.3372 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13759_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1987 |
| Coherence (Success) | 0.0352 |
| Coherence (Failure) | 0.0628 |
| Gradient Magnitude (Success) | 0.1239 |
| Gradient Magnitude (Failure) | 0.2236 |
| Activation Separation | 4.6019 |
| Cosine Distance | 0.0344 |
| Clusters | 1,456 |
| Noise Fraction | 0.3281 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep13849_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3191 |
| Coherence (Success) | 0.1383 |
| Coherence (Failure) | 0.0120 |
| Gradient Magnitude (Success) | 0.1683 |
| Gradient Magnitude (Failure) | 0.1520 |
| Activation Separation | 3.7950 |
| Cosine Distance | 0.0199 |
| Clusters | 1,313 |
| Noise Fraction | 0.3177 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13994_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4067 |
| Coherence (Success) | 0.2607 |
| Coherence (Failure) | 0.0356 |
| Gradient Magnitude (Success) | 0.2395 |
| Gradient Magnitude (Failure) | 0.1694 |
| Activation Separation | 4.5376 |
| Cosine Distance | 0.0334 |
| Clusters | 1,450 |
| Noise Fraction | 0.3371 |
| RSA Alignment (ρ) | 0.3230 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14217_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1310 |
| Coherence (Success) | 0.0239 |
| Coherence (Failure) | 0.0034 |
| Gradient Magnitude (Success) | 0.1182 |
| Gradient Magnitude (Failure) | 0.1325 |
| Activation Separation | 4.3590 |
| Cosine Distance | 0.0299 |
| Clusters | 1,380 |
| Noise Fraction | 0.3374 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14436_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4291 |
| Coherence (Success) | 0.1103 |
| Coherence (Failure) | 0.1220 |
| Gradient Magnitude (Success) | 0.1565 |
| Gradient Magnitude (Failure) | 0.2380 |
| Activation Separation | 4.4715 |
| Cosine Distance | 0.0337 |
| Clusters | 1,471 |
| Noise Fraction | 0.3157 |
| RSA Alignment (ρ) | 0.1860 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14654_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2181 |
| Coherence (Success) | 0.1303 |
| Coherence (Failure) | 0.0715 |
| Gradient Magnitude (Success) | 0.1450 |
| Gradient Magnitude (Failure) | 0.2282 |
| Activation Separation | 4.5782 |
| Cosine Distance | 0.0329 |
| Clusters | 1,671 |
| Noise Fraction | 0.3272 |
| RSA Alignment (ρ) | 0.2447 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14855_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5805 |
| Coherence (Success) | 0.1072 |
| Coherence (Failure) | 0.0298 |
| Gradient Magnitude (Success) | 0.1710 |
| Gradient Magnitude (Failure) | 0.1969 |
| Activation Separation | 3.9585 |
| Cosine Distance | 0.0252 |
| Clusters | 1,561 |
| Noise Fraction | 0.3401 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15067_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7307 |
| Coherence (Success) | 0.1853 |
| Coherence (Failure) | 0.1712 |
| Gradient Magnitude (Success) | 0.2299 |
| Gradient Magnitude (Failure) | 0.3172 |
| Activation Separation | 3.7326 |
| Cosine Distance | 0.0200 |
| Clusters | 1,476 |
| Noise Fraction | 0.3591 |
| RSA Alignment (ρ) | 0.1860 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15067_lower8.100_upper11.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5955 |
| Coherence (Success) | 0.4167 |
| Coherence (Failure) | 0.0568 |
| Gradient Magnitude (Success) | 0.3236 |
| Gradient Magnitude (Failure) | 0.2202 |
| Activation Separation | 3.2625 |
| Cosine Distance | 0.0145 |
| Clusters | 1,403 |
| Noise Fraction | 0.3424 |
| RSA Alignment (ρ) | 0.1077 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep2 | 2 | 0.2564 | 0.6709 | 0.1906 | 0.0819 | 0.0467 | 0.1432 | 0.0006 | nan |
| collect_sapling @ ep4 | 4 | 0.0775 | 0.9053 | 0.2802 | 0.1029 | 0.0458 | 0.1936 | 0.0012 | nan |
| place_plant @ ep4 | 4 | 0.0947 | 0.8852 | 0.3782 | 0.1023 | 0.0450 | 0.1745 | 0.0009 | nan |
| collect_wood @ ep7 | 7 | -0.0822 | 0.8830 | 0.4171 | 0.1019 | 0.0430 | 0.2234 | 0.0013 | — |
| eat_cow @ ep9 | 9 | 0.1532 | 0.8305 | 0.4948 | 0.0956 | 0.0476 | 0.1857 | 0.0011 | 0.0000 |
| collect_drink @ ep14 | 14 | 0.2326 | 0.8907 | 0.5863 | 0.0962 | 0.0534 | 0.2056 | 0.0012 | -0.6547 |
| place_table @ ep48 | 48 | 0.4195 | 0.9329 | 0.0273 | 0.2295 | 0.0638 | 0.4353 | 0.0012 | nan |
| make_wood_sword @ ep66 | 66 | 0.8175 | 0.9129 | 0.4397 | 0.2473 | 0.1070 | 0.4324 | 0.0005 | nan |
| defeat_zombie @ ep120 | 120 | 0.7394 | 0.8315 | 0.3137 | 0.2312 | 0.1762 | 0.4990 | 0.0005 | -0.4352 |
| make_wood_pickaxe @ ep135 | 135 | 0.7881 | 0.8016 | 0.7356 | 0.2139 | 0.2153 | 0.4844 | 0.0003 | -0.3928 |
| defeat_skeleton @ ep235 | 235 | 0.0139 | 0.0817 | -0.0109 | 0.1662 | 0.1462 | 0.8094 | 0.0003 | -0.1396 |
| collect_stone @ ep436 | 436 | 0.3255 | 0.3392 | 0.2373 | 0.1862 | 0.1555 | 0.4600 | 0.0002 | -0.1745 |
| make_stone_pickaxe @ ep788 | 788 | 0.7166 | 0.2125 | 0.2015 | 0.1873 | 0.2424 | 0.5583 | 0.0003 | 0.0000 |
| collect_coal @ ep990 | 990 | -0.2752 | 0.1203 | 0.2580 | 0.1183 | 0.2202 | 0.3440 | 0.0002 | -0.4187 |
| eat_plant @ ep1194 | 1,194 | -0.4643 | 0.1694 | 0.1560 | 0.1108 | 0.2457 | 1.2762 | 0.0008 | -0.2443 |
| place_stone @ ep1333 | 1,333 | 0.3159 | 0.2365 | 0.0801 | 0.1230 | 0.1262 | 0.6799 | 0.0004 | -0.1662 |
| place_furnace @ ep2357 | 2,357 | 0.0545 | 0.1027 | 0.0409 | 0.1342 | 0.1673 | 1.1679 | 0.0015 | -0.2791 |
| make_stone_sword @ ep6079 | 6,079 | -0.1982 | 0.1151 | 0.0159 | 0.1526 | 0.1370 | 2.4905 | 0.0089 | 0.3508 |
| collect_iron @ ep13849 | 13,849 | 0.3191 | 0.1383 | 0.0120 | 0.1683 | 0.1520 | 3.7950 | 0.0199 | 0.2216 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic_step50000_ep290 | 290 | 0.5173 | 0.1494 | 0.0342 | 0.1599 | 0.1924 | 0.6082 | 0.0002 | -0.2740 |
| periodic_step100000_ep574 | 574 | -0.2612 | 0.1619 | 0.1364 | 0.1327 | 0.2329 | 0.6573 | 0.0007 | -0.1047 |
| periodic_step150000_ep847 | 847 | 0.5337 | 0.2173 | 0.0462 | 0.2003 | 0.0993 | 1.7051 | 0.0024 | 0.0000 |
| periodic_step200000_ep1132 | 1,132 | 0.3334 | 0.1396 | 0.5108 | 0.0955 | 0.3552 | 0.7913 | 0.0004 | -0.3838 |
| periodic_step250000_ep1407 | 1,407 | 0.2551 | 0.0358 | 0.3606 | 0.1003 | 0.2527 | 0.8756 | 0.0006 | -0.0698 |
| periodic_step300000_ep1687 | 1,687 | -0.0448 | 0.2392 | 0.0041 | 0.1372 | 0.1321 | 0.8710 | 0.0008 | -0.2791 |
| periodic_step350000_ep1967 | 1,967 | 0.5712 | 0.3474 | 0.2748 | 0.1624 | 0.2836 | 0.9789 | 0.0019 | -0.1396 |
| periodic_step400000_ep2245 | 2,245 | -0.0462 | 0.1780 | 0.2832 | 0.2611 | 0.2694 | 1.3384 | 0.0018 | -0.2791 |
| periodic_step450000_ep2527 | 2,527 | 0.6360 | 0.3114 | 0.2995 | 0.1698 | 0.3740 | 1.3472 | 0.0017 | -0.1292 |
| periodic_step500000_ep2818 | 2,818 | -0.4294 | 0.0284 | 0.0810 | 0.1172 | 0.1850 | 1.7981 | 0.0055 | -0.3489 |
| periodic_step550000_ep3084 | 3,084 | -0.3945 | 0.0264 | 0.2211 | 0.1185 | 0.2689 | 1.8233 | 0.0042 | 0.0000 |
| periodic_step600000_ep3343 | 3,343 | -0.1108 | 0.1199 | 0.1573 | 0.1167 | 0.1946 | 1.4024 | 0.0036 | -0.0349 |
| periodic_step650000_ep3610 | 3,610 | 0.7166 | 0.1749 | 0.2929 | 0.1529 | 0.3717 | 2.1466 | 0.0081 | -0.1396 |
| periodic_step700000_ep3880 | 3,880 | 0.7624 | 0.5032 | 0.1965 | 0.3141 | 0.2816 | 3.6270 | 0.0180 | -0.2443 |
| periodic_step750000_ep4146 | 4,146 | 0.5472 | 0.0712 | 0.2388 | 0.1239 | 0.2984 | 2.1844 | 0.0059 | -0.2791 |
| periodic_step800000_ep4420 | 4,420 | 0.1319 | 0.1209 | 0.0407 | 0.1627 | 0.1554 | 1.5482 | 0.0032 | -0.1396 |
| periodic_step850000_ep4690 | 4,690 | 0.3728 | 0.2324 | 0.1445 | 0.2009 | 0.2450 | 3.1699 | 0.0165 | -0.0739 |
| periodic_step900000_ep4948 | 4,948 | 0.6059 | 0.2939 | 0.1199 | 0.2267 | 0.2525 | 2.4538 | 0.0097 | -0.2216 |
| periodic_step950000_ep5226 | 5,226 | 0.7981 | 0.1562 | 0.3029 | 0.1732 | 0.3152 | 1.8955 | 0.0046 | 0.0000 |
| periodic_step1000000_ep5480 | 5,480 | 0.3638 | 0.2822 | 0.1527 | 0.2082 | 0.2489 | 2.6252 | 0.0093 | 0.0000 |
| periodic_step1050000_ep5721 | 5,721 | 0.8794 | 0.4286 | 0.2194 | 0.2743 | 0.3071 | 2.9189 | 0.0111 | -0.0923 |
| periodic_step1100000_ep5982 | 5,982 | 0.4848 | 0.1496 | 0.1718 | 0.1658 | 0.2657 | 2.7209 | 0.0093 | 0.0000 |
| periodic_step1150000_ep6234 | 6,234 | 0.7171 | 0.2424 | 0.1114 | 0.1972 | 0.2004 | 2.9001 | 0.0123 | 0.0554 |
| periodic_step1200000_ep6494 | 6,494 | 0.2283 | 0.0669 | 0.1909 | 0.1152 | 0.2898 | 3.6075 | 0.0223 | -0.0739 |
| periodic_step1250000_ep6755 | 6,755 | 0.0030 | 0.0013 | 0.0924 | 0.0975 | 0.1909 | 3.9879 | 0.0252 | -0.0554 |
| periodic_step1300000_ep7011 | 7,011 | -0.0188 | 0.0428 | 0.0449 | 0.1184 | 0.1739 | 4.2614 | 0.0206 | 0.1047 |
| periodic_step1350000_ep7271 | 7,271 | 0.5210 | 0.1161 | 0.2047 | 0.1440 | 0.2924 | 3.2006 | 0.0142 | 0.3489 |
| periodic_step1400000_ep7530 | 7,530 | 0.5683 | 0.1259 | 0.1368 | 0.1539 | 0.2286 | 3.8229 | 0.0206 | 0.0185 |
| periodic_step1450000_ep7786 | 7,786 | 0.6799 | 0.3173 | 0.1615 | 0.2896 | 0.3565 | 4.2283 | 0.0216 | 0.3140 |
| periodic_step1500000_ep8053 | 8,053 | 0.2491 | 0.1053 | 0.0412 | 0.1365 | 0.1633 | 3.9179 | 0.0192 | 0.3693 |
| periodic_step1550000_ep8308 | 8,308 | 0.2697 | 0.1029 | -0.0047 | 0.1358 | 0.1391 | 4.0048 | 0.0236 | 0.2443 |
| periodic_step1600000_ep8557 | 8,557 | 0.1953 | 0.1327 | 0.0912 | 0.1768 | 0.2440 | 4.8026 | 0.0347 | 0.0000 |
| periodic_step1650000_ep8797 | 8,797 | 0.0732 | 0.0885 | 0.0049 | 0.1490 | 0.1478 | 4.2420 | 0.0275 | 0.2443 |
| periodic_step1700000_ep9046 | 9,046 | 0.5423 | 0.2748 | 0.0355 | 0.2455 | 0.2103 | 3.8829 | 0.0240 | 0.2094 |
| periodic_step1750000_ep9290 | 9,290 | 0.4770 | 0.0980 | -0.0164 | 0.1663 | 0.1598 | 4.1354 | 0.0219 | 0.2094 |
| periodic_step1800000_ep9536 | 9,536 | 0.8261 | 0.2378 | 0.1188 | 0.2438 | 0.2459 | 3.5912 | 0.0233 | 0.0185 |
| periodic_step1850000_ep9782 | 9,782 | 0.2335 | 0.0980 | 0.0520 | 0.1541 | 0.2045 | 4.2677 | 0.0274 | 0.3140 |
| periodic_step1900000_ep10030 | 10,030 | 0.2766 | 0.3281 | 0.0040 | 0.2778 | 0.1657 | 3.4292 | 0.0173 | 0.1292 |
| periodic_step1950000_ep10273 | 10,273 | 0.7465 | 0.0813 | 0.2007 | 0.1700 | 0.3390 | 4.6274 | 0.0300 | 0.1745 |
| periodic_step2000000_ep10525 | 10,525 | 0.5032 | 0.1838 | 0.0453 | 0.1952 | 0.1992 | 4.9164 | 0.0265 | 0.2443 |
| periodic_step2050000_ep10759 | 10,759 | 0.5590 | 0.2321 | 0.0209 | 0.2195 | 0.1957 | 4.0056 | 0.0244 | 0.2791 |
| periodic_step2100000_ep11027 | 11,027 | 0.1028 | 0.0504 | 0.0307 | 0.1222 | 0.1804 | 4.2031 | 0.0279 | 0.2094 |
| periodic_step2150000_ep11269 | 11,269 | 0.2340 | 0.0626 | 0.0817 | 0.1475 | 0.2240 | 3.9944 | 0.0232 | 0.2954 |
| periodic_step2200000_ep11497 | 11,497 | 0.6003 | 0.3862 | 0.1439 | 0.3308 | 0.2918 | 4.2398 | 0.0209 | 0.2791 |
| periodic_step2250000_ep11718 | 11,718 | 0.7705 | 0.3187 | 0.0360 | 0.2613 | 0.2180 | 3.8330 | 0.0189 | 0.0185 |
| periodic_step2300000_ep11949 | 11,949 | 0.4204 | 0.1907 | 0.0686 | 0.1987 | 0.2490 | 4.1480 | 0.0168 | 0.2400 |
| periodic_step2350000_ep12164 | 12,164 | -0.0861 | 0.1112 | 0.0349 | 0.1448 | 0.1834 | 4.8070 | 0.0285 | 0.1292 |
| periodic_step2400000_ep12367 | 12,367 | 0.5831 | 0.0904 | 0.0928 | 0.1654 | 0.2305 | 3.5086 | 0.0174 | 0.1846 |
| periodic_step2450000_ep12596 | 12,596 | 0.2521 | 0.2739 | 0.0188 | 0.2262 | 0.1686 | 3.9673 | 0.0235 | 0.2031 |
| periodic_step2500000_ep12823 | 12,823 | 0.7245 | 0.2734 | 0.0879 | 0.2603 | 0.2517 | 3.3367 | 0.0160 | 0.2770 |
| periodic_step2550000_ep13047 | 13,047 | 0.4111 | 0.3216 | 0.1203 | 0.2677 | 0.2508 | 3.4645 | 0.0181 | 0.3230 |
| periodic_step2600000_ep13278 | 13,278 | 0.4214 | 0.2687 | 0.1099 | 0.2317 | 0.2347 | 3.9337 | 0.0231 | 0.2400 |
| periodic_step2650000_ep13516 | 13,516 | 0.5829 | 0.2424 | 0.2337 | 0.2171 | 0.3378 | 4.7152 | 0.0279 | 0.0587 |
| periodic_step2700000_ep13759 | 13,759 | 0.1987 | 0.0352 | 0.0628 | 0.1239 | 0.2236 | 4.6019 | 0.0344 | 0.1846 |
| periodic_step2750000_ep13994 | 13,994 | 0.4067 | 0.2607 | 0.0356 | 0.2395 | 0.1694 | 4.5376 | 0.0334 | 0.3230 |
| periodic_step2800000_ep14217 | 14,217 | 0.1310 | 0.0239 | 0.0034 | 0.1182 | 0.1325 | 4.3590 | 0.0299 | 0.2216 |
| periodic_step2850000_ep14436 | 14,436 | 0.4291 | 0.1103 | 0.1220 | 0.1565 | 0.2380 | 4.4715 | 0.0337 | 0.1860 |
| periodic_step2900000_ep14654 | 14,654 | 0.2181 | 0.1303 | 0.0715 | 0.1450 | 0.2282 | 4.5782 | 0.0329 | 0.2447 |
| periodic_step2950000_ep14855 | 14,855 | 0.5805 | 0.1072 | 0.0298 | 0.1710 | 0.1969 | 3.9585 | 0.0252 | 0.0000 |
| periodic_step3000000_ep15067 | 15,067 | 0.7307 | 0.1853 | 0.1712 | 0.2299 | 0.3172 | 3.7326 | 0.0200 | 0.1860 |
| final_step3000320_ep15067 | 15,067 | 0.5955 | 0.4167 | 0.0568 | 0.3236 | 0.2202 | 3.2625 | 0.0145 | 0.1077 |
