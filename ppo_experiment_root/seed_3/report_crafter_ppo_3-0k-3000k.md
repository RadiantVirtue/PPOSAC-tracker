# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 3  
**Total episodes:** 14,913  
**Experiment root:** `ppo_experiment_root\seed_3`  
**Generated:** 2026-04-01 00:33

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep3 | 3 | 0.1125 | 0.8712 | 0.2822 | 0.1039 | 0.0412 | 0.1691 | 0.0007 | -0.2611 |
| collect_sapling @ ep4 | 4 | -0.0701 | 0.9016 | 0.3411 | 0.0922 | 0.0462 | 0.1970 | 0.0012 | -0.3928 |
| place_plant @ ep4 | 4 | 0.2482 | 0.8362 | 0.4257 | 0.0957 | 0.0448 | 0.2029 | 0.0009 | nan |
| collect_wood @ ep15 | 15 | -0.2225 | 0.9189 | 0.0844 | 0.0979 | 0.0407 | 0.2061 | 0.0009 | -0.6547 |
| place_table @ ep15 | 15 | -0.1358 | 0.6361 | 0.2619 | 0.0881 | 0.0374 | 0.1964 | 0.0010 | -0.2611 |
| collect_drink @ ep20 | 20 | 0.0212 | 0.7412 | 0.2092 | 0.1365 | 0.0571 | 0.2390 | 0.0006 | -0.6547 |
| eat_cow @ ep111 | 111 | 0.8540 | 0.8148 | 0.7263 | 0.2327 | 0.2217 | 0.6872 | 0.0003 | nan |
| make_wood_pickaxe @ ep205 | 205 | 0.7719 | 0.4323 | 0.6089 | 0.2644 | 0.1889 | 1.2835 | 0.0004 | -0.4352 |
| defeat_zombie @ ep209 | 209 | 0.5205 | 0.2012 | 0.1599 | 0.2397 | 0.1559 | 1.3374 | 0.0004 | -0.2791 |
| Step 50,000 | 290 | -0.3284 | 0.1061 | 0.1093 | 0.1377 | 0.1447 | 1.9804 | 0.0007 | 0.0000 |
| make_wood_sword @ ep291 | 291 | 0.1528 | 0.0112 | 0.1550 | 0.1381 | 0.1441 | 2.1228 | 0.0008 | -0.3489 |
| Step 100,000 | 568 | 0.4600 | 0.7907 | 0.4027 | 0.2100 | 0.2370 | 0.7776 | 0.0003 | -0.3140 |
| defeat_skeleton @ ep792 | 792 | 0.0717 | 0.1002 | 0.1369 | 0.1121 | 0.1632 | 0.8617 | 0.0009 | -0.1047 |
| Step 150,000 | 864 | 0.3356 | 0.2503 | 0.3307 | 0.1643 | 0.2289 | 0.8011 | 0.0010 | -0.0698 |
| collect_stone @ ep995 | 995 | 0.9303 | 0.3996 | 0.7388 | 0.1664 | 0.2860 | 0.3543 | 0.0004 | -0.1396 |
| place_furnace @ ep995 | 995 | 0.8340 | 0.5528 | 0.2889 | 0.1903 | 0.1889 | 0.3410 | 0.0004 | -0.1846 |
| place_stone @ ep995 | 995 | 0.8946 | 0.4135 | 0.6493 | 0.1683 | 0.2918 | 0.3790 | 0.0006 | -0.3140 |
| Step 200,000 | 1,156 | -0.0569 | 0.0919 | 0.1170 | 0.0716 | 0.1398 | 0.9710 | 0.0015 | -0.1047 |
| collect_coal @ ep1320 | 1,320 | 0.6980 | 0.2017 | 0.3923 | 0.1491 | 0.2756 | 0.8411 | 0.0022 | -0.3482 |
| Step 250,000 | 1,439 | 0.0375 | 0.2263 | 0.0360 | 0.1263 | 0.1278 | 0.9433 | 0.0015 | 0.0000 |
| Step 300,000 | 1,724 | 0.2578 | 0.1005 | 0.0903 | 0.1056 | 0.1508 | 1.0031 | 0.0025 | -0.2443 |
| Step 350,000 | 2,009 | -0.1139 | 0.2065 | 0.0300 | 0.1409 | 0.1474 | 1.0888 | 0.0047 | -0.2094 |
| make_stone_sword @ ep2126 | 2,126 | 0.4122 | 0.1204 | 0.2045 | 0.0876 | 0.1845 | 1.2968 | 0.0069 | -0.1662 |
| Step 400,000 | 2,295 | 0.7079 | 0.4312 | 0.4258 | 0.1764 | 0.3090 | 1.2532 | 0.0057 | -0.2443 |
| Step 450,000 | 2,562 | 0.7069 | 0.1108 | 0.1425 | 0.1138 | 0.1618 | 1.2795 | 0.0048 | -0.3838 |
| Step 500,000 | 2,848 | 0.9021 | 0.4584 | 0.4386 | 0.2475 | 0.3888 | 1.5526 | 0.0073 | -0.1396 |
| Step 550,000 | 3,128 | 0.5381 | 0.2485 | 0.1744 | 0.1533 | 0.3254 | 1.9856 | 0.0105 | -0.0698 |
| Step 600,000 | 3,402 | 0.6506 | 0.3052 | 0.1581 | 0.2141 | 0.2329 | 1.9335 | 0.0071 | -0.0698 |
| Step 650,000 | 3,688 | 0.7251 | 0.2275 | 0.1766 | 0.1903 | 0.2269 | 2.0900 | 0.0092 | -0.3140 |
| Step 700,000 | 3,961 | 0.8679 | 0.5167 | 0.4529 | 0.3157 | 0.4505 | 2.0059 | 0.0089 | -0.0185 |
| Step 750,000 | 4,228 | 0.2772 | 0.1884 | 0.1497 | 0.1483 | 0.2584 | 1.7875 | 0.0038 | -0.2094 |
| Step 800,000 | 4,506 | 0.8163 | 0.2743 | 0.3100 | 0.2237 | 0.3853 | 2.3697 | 0.0078 | 0.0000 |
| Step 850,000 | 4,773 | 0.5300 | 0.1654 | 0.0800 | 0.1625 | 0.1773 | 2.0842 | 0.0063 | -0.1108 |
| Step 900,000 | 5,043 | -0.0283 | 0.1535 | 0.0436 | 0.1605 | 0.1788 | 2.0600 | 0.0071 | -0.1745 |
| Step 950,000 | 5,306 | 0.2657 | 0.1180 | 0.0048 | 0.1336 | 0.1370 | 2.4453 | 0.0080 | -0.1662 |
| make_stone_pickaxe @ ep5445 | 5,445 | 0.7015 | 0.3474 | 0.0976 | 0.2254 | 0.2435 | 1.9739 | 0.0066 | -0.1396 |
| Step 1,000,000 | 5,574 | 0.0295 | 0.0756 | 0.0470 | 0.1365 | 0.1612 | 2.5074 | 0.0055 | -0.1047 |
| Step 1,050,000 | 5,838 | 0.5666 | 0.1805 | 0.0986 | 0.2105 | 0.2226 | 2.8800 | 0.0073 | -0.1108 |
| Step 1,100,000 | 6,090 | 0.1859 | 0.2365 | 0.0535 | 0.2115 | 0.1730 | 2.3809 | 0.0078 | -0.1396 |
| Step 1,150,000 | 6,342 | 0.7271 | 0.5098 | 0.0782 | 0.3804 | 0.2511 | 3.0110 | 0.0095 | 0.0000 |
| Step 1,200,000 | 6,598 | 0.5069 | 0.2087 | 0.0420 | 0.1983 | 0.1580 | 2.7352 | 0.0073 | -0.0698 |
| Step 1,250,000 | 6,847 | 0.7028 | 0.4747 | 0.2341 | 0.3625 | 0.3416 | 2.9972 | 0.0083 | -0.0349 |
| Step 1,300,000 | 7,095 | 0.3378 | 0.1958 | 0.0202 | 0.2173 | 0.2106 | 3.4043 | 0.0107 | -0.0349 |
| Step 1,350,000 | 7,337 | 0.2962 | 0.1195 | 0.0184 | 0.1711 | 0.1659 | 3.8081 | 0.0153 | -0.0698 |
| Step 1,400,000 | 7,579 | 0.6457 | 0.3970 | 0.0244 | 0.3104 | 0.2065 | 2.9062 | 0.0091 | -0.0698 |
| Step 1,450,000 | 7,832 | 0.7059 | 0.1873 | 0.1722 | 0.1987 | 0.2510 | 3.5628 | 0.0143 | 0.0000 |
| Step 1,500,000 | 8,073 | 0.5597 | 0.2468 | 0.0332 | 0.2433 | 0.1943 | 3.5224 | 0.0128 | -0.0698 |
| Step 1,550,000 | 8,315 | 0.0377 | 0.3524 | 0.0372 | 0.3560 | 0.1852 | 3.0869 | 0.0108 | -0.1477 |
| Step 1,600,000 | 8,559 | 0.4212 | 0.1337 | 0.0711 | 0.1861 | 0.2970 | 4.0563 | 0.0184 | -0.0923 |
| Step 1,650,000 | 8,797 | 0.1547 | 0.1572 | 0.0316 | 0.1856 | 0.1892 | 4.9656 | 0.0286 | -0.0098 |
| Step 1,700,000 | 9,038 | 0.6628 | 0.1537 | 0.0567 | 0.1892 | 0.2178 | 4.3629 | 0.0229 | -0.1108 |
| Step 1,750,000 | 9,277 | 0.3242 | 0.1595 | 0.0894 | 0.2194 | 0.1853 | 3.4637 | 0.0139 | -0.0739 |
| Step 1,800,000 | 9,503 | 0.7226 | 0.3961 | 0.1655 | 0.2969 | 0.2700 | 3.3921 | 0.0138 | 0.0881 |
| Step 1,850,000 | 9,741 | 0.3049 | 0.1645 | 0.0800 | 0.2208 | 0.2780 | 5.2098 | 0.0265 | -0.0739 |
| Step 1,900,000 | 9,982 | 0.5784 | 0.2618 | 0.0609 | 0.2487 | 0.2023 | 3.2961 | 0.0119 | -0.0923 |
| Step 1,950,000 | 10,208 | 0.4691 | 0.2759 | 0.0624 | 0.2738 | 0.2180 | 3.6958 | 0.0164 | -0.0923 |
| Step 2,000,000 | 10,434 | 0.1233 | 0.1426 | -0.0171 | 0.1741 | 0.1545 | 4.2209 | 0.0204 | -0.0739 |
| Step 2,050,000 | 10,662 | 0.6633 | 0.1249 | 0.0791 | 0.1895 | 0.2524 | 2.9866 | 0.0120 | 0.1292 |
| Step 2,100,000 | 10,903 | 0.2207 | 0.1778 | 0.0308 | 0.2124 | 0.1820 | 4.2004 | 0.0176 | 0.0185 |
| Step 2,150,000 | 11,129 | 0.2343 | 0.0337 | 0.0384 | 0.1310 | 0.1820 | 3.7495 | 0.0149 | 0.1745 |
| Step 2,200,000 | 11,352 | 0.6181 | 0.2752 | 0.1873 | 0.2582 | 0.2746 | 4.4227 | 0.0232 | -0.0185 |
| Step 2,250,000 | 11,585 | 0.6457 | 0.1517 | 0.2848 | 0.2170 | 0.4320 | 3.2546 | 0.0135 | 0.0369 |
| Step 2,300,000 | 11,801 | 0.4385 | 0.3140 | -0.0062 | 0.2573 | 0.1701 | 2.8290 | 0.0084 | 0.1745 |
| Step 2,350,000 | 12,034 | 0.6609 | 0.3486 | 0.2502 | 0.3137 | 0.3219 | 3.1397 | 0.0137 | 0.0369 |
| Step 2,400,000 | 12,261 | 0.5937 | 0.0382 | 0.1432 | 0.1526 | 0.2155 | 4.8954 | 0.0326 | 0.0000 |
| Step 2,450,000 | 12,488 | 0.4313 | 0.2208 | 0.0594 | 0.2147 | 0.2125 | 4.1231 | 0.0200 | 0.0185 |
| Step 2,500,000 | 12,703 | 0.5202 | 0.1214 | 0.0070 | 0.1851 | 0.1583 | 4.2201 | 0.0202 | 0.1396 |
| Step 2,550,000 | 12,923 | 0.0858 | 0.0577 | -0.0211 | 0.1377 | 0.1281 | 5.0520 | 0.0327 | 0.0698 |
| Step 2,600,000 | 13,148 | 0.7334 | 0.3067 | 0.1526 | 0.2657 | 0.2898 | 3.2656 | 0.0129 | 0.1047 |
| Step 2,650,000 | 13,380 | 0.1744 | 0.0040 | 0.0432 | 0.1176 | 0.1843 | 5.0316 | 0.0342 | 0.0554 |
| Step 2,700,000 | 13,598 | 0.3117 | 0.1488 | 0.0462 | 0.1808 | 0.1757 | 3.7519 | 0.0206 | 0.1396 |
| Step 2,750,000 | 13,819 | 0.5402 | 0.0954 | 0.0440 | 0.1719 | 0.2249 | 4.1130 | 0.0265 | 0.2094 |
| Step 2,800,000 | 14,048 | 0.4689 | 0.1779 | 0.1156 | 0.2520 | 0.2433 | 3.8133 | 0.0183 | 0.2443 |
| Step 2,850,000 | 14,263 | 0.4426 | 0.1446 | 0.0125 | 0.1860 | 0.1638 | 3.6654 | 0.0215 | 0.0349 |
| Step 2,900,000 | 14,476 | 0.1489 | 0.1410 | 0.0663 | 0.1844 | 0.1715 | 3.4752 | 0.0207 | 0.1396 |
| Step 2,950,000 | 14,696 | 0.5510 | 0.1857 | 0.0910 | 0.2058 | 0.2110 | 3.3381 | 0.0202 | 0.2791 |
| Step 3,000,000 | 14,911 | 0.3418 | 0.0738 | 0.0109 | 0.1720 | 0.1316 | 2.2883 | 0.0092 | 0.1745 |
| Final | 14,913 | 0.4190 | 0.0293 | 0.1112 | 0.1382 | 0.1951 | 2.6706 | 0.0135 | 0.1745 |

---

## wake_up_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1125 |
| Coherence (Success) | 0.8712 |
| Coherence (Failure) | 0.2822 |
| Gradient Magnitude (Success) | 0.1039 |
| Gradient Magnitude (Failure) | 0.0412 |
| Activation Separation | 0.1691 |
| Cosine Distance | 0.0007 |
| Clusters | 1,422 |
| Noise Fraction | 0.2661 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep4_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0701 |
| Coherence (Success) | 0.9016 |
| Coherence (Failure) | 0.3411 |
| Gradient Magnitude (Success) | 0.0922 |
| Gradient Magnitude (Failure) | 0.0462 |
| Activation Separation | 0.1970 |
| Cosine Distance | 0.0012 |
| Clusters | 1,601 |
| Noise Fraction | 0.2717 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep4_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2482 |
| Coherence (Success) | 0.8362 |
| Coherence (Failure) | 0.4257 |
| Gradient Magnitude (Success) | 0.0957 |
| Gradient Magnitude (Failure) | 0.0448 |
| Activation Separation | 0.2029 |
| Cosine Distance | 0.0009 |
| Clusters | 1,470 |
| Noise Fraction | 0.2548 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Skeleton, Wood, Wood Pickaxe |

---

## collect_wood_ep15_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2225 |
| Coherence (Success) | 0.9189 |
| Coherence (Failure) | 0.0844 |
| Gradient Magnitude (Success) | 0.0979 |
| Gradient Magnitude (Failure) | 0.0407 |
| Activation Separation | 0.2061 |
| Cosine Distance | 0.0009 |
| Clusters | 1,449 |
| Noise Fraction | 0.2632 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep15_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1358 |
| Coherence (Success) | 0.6361 |
| Coherence (Failure) | 0.2619 |
| Gradient Magnitude (Success) | 0.0881 |
| Gradient Magnitude (Failure) | 0.0374 |
| Activation Separation | 0.1964 |
| Cosine Distance | 0.0010 |
| Clusters | 1,499 |
| Noise Fraction | 0.2530 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep20_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0212 |
| Coherence (Success) | 0.7412 |
| Coherence (Failure) | 0.2092 |
| Gradient Magnitude (Success) | 0.1365 |
| Gradient Magnitude (Failure) | 0.0571 |
| Activation Separation | 0.2390 |
| Cosine Distance | 0.0006 |
| Clusters | 1,411 |
| Noise Fraction | 0.2523 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep111_lower3.000_upper4.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8540 |
| Coherence (Success) | 0.8148 |
| Coherence (Failure) | 0.7263 |
| Gradient Magnitude (Success) | 0.2327 |
| Gradient Magnitude (Failure) | 0.2217 |
| Activation Separation | 0.6872 |
| Cosine Distance | 0.0003 |
| Clusters | 1,360 |
| Noise Fraction | 0.2417 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep205_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7719 |
| Coherence (Success) | 0.4323 |
| Coherence (Failure) | 0.6089 |
| Gradient Magnitude (Success) | 0.2644 |
| Gradient Magnitude (Failure) | 0.1889 |
| Activation Separation | 1.2835 |
| Cosine Distance | 0.0004 |
| Clusters | 1,697 |
| Noise Fraction | 0.2455 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep209_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5205 |
| Coherence (Success) | 0.2012 |
| Coherence (Failure) | 0.1599 |
| Gradient Magnitude (Success) | 0.2397 |
| Gradient Magnitude (Failure) | 0.1559 |
| Activation Separation | 1.3374 |
| Cosine Distance | 0.0004 |
| Clusters | 1,738 |
| Noise Fraction | 0.2626 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep290_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3284 |
| Coherence (Success) | 0.1061 |
| Coherence (Failure) | 0.1093 |
| Gradient Magnitude (Success) | 0.1377 |
| Gradient Magnitude (Failure) | 0.1447 |
| Activation Separation | 1.9804 |
| Cosine Distance | 0.0007 |
| Clusters | 1,518 |
| Noise Fraction | 0.3014 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep291_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1528 |
| Coherence (Success) | 0.0112 |
| Coherence (Failure) | 0.1550 |
| Gradient Magnitude (Success) | 0.1381 |
| Gradient Magnitude (Failure) | 0.1441 |
| Activation Separation | 2.1228 |
| Cosine Distance | 0.0008 |
| Clusters | 1,595 |
| Noise Fraction | 0.2486 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep568_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4600 |
| Coherence (Success) | 0.7907 |
| Coherence (Failure) | 0.4027 |
| Gradient Magnitude (Success) | 0.2100 |
| Gradient Magnitude (Failure) | 0.2370 |
| Activation Separation | 0.7776 |
| Cosine Distance | 0.0003 |
| Clusters | 1,509 |
| Noise Fraction | 0.2711 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep792_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0717 |
| Coherence (Success) | 0.1002 |
| Coherence (Failure) | 0.1369 |
| Gradient Magnitude (Success) | 0.1121 |
| Gradient Magnitude (Failure) | 0.1632 |
| Activation Separation | 0.8617 |
| Cosine Distance | 0.0009 |
| Clusters | 1,594 |
| Noise Fraction | 0.2793 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep864_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3356 |
| Coherence (Success) | 0.2503 |
| Coherence (Failure) | 0.3307 |
| Gradient Magnitude (Success) | 0.1643 |
| Gradient Magnitude (Failure) | 0.2289 |
| Activation Separation | 0.8011 |
| Cosine Distance | 0.0010 |
| Clusters | 1,633 |
| Noise Fraction | 0.2739 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep995_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9303 |
| Coherence (Success) | 0.3996 |
| Coherence (Failure) | 0.7388 |
| Gradient Magnitude (Success) | 0.1664 |
| Gradient Magnitude (Failure) | 0.2860 |
| Activation Separation | 0.3543 |
| Cosine Distance | 0.0004 |
| Clusters | 1,701 |
| Noise Fraction | 0.2703 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep995_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8340 |
| Coherence (Success) | 0.5528 |
| Coherence (Failure) | 0.2889 |
| Gradient Magnitude (Success) | 0.1903 |
| Gradient Magnitude (Failure) | 0.1889 |
| Activation Separation | 0.3410 |
| Cosine Distance | 0.0004 |
| Clusters | 1,677 |
| Noise Fraction | 0.2718 |
| RSA Alignment (ρ) | -0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep995_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8946 |
| Coherence (Success) | 0.4135 |
| Coherence (Failure) | 0.6493 |
| Gradient Magnitude (Success) | 0.1683 |
| Gradient Magnitude (Failure) | 0.2918 |
| Activation Separation | 0.3790 |
| Cosine Distance | 0.0006 |
| Clusters | 1,688 |
| Noise Fraction | 0.2488 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1156_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0569 |
| Coherence (Success) | 0.0919 |
| Coherence (Failure) | 0.1170 |
| Gradient Magnitude (Success) | 0.0716 |
| Gradient Magnitude (Failure) | 0.1398 |
| Activation Separation | 0.9710 |
| Cosine Distance | 0.0015 |
| Clusters | 1,566 |
| Noise Fraction | 0.2788 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep1320_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6980 |
| Coherence (Success) | 0.2017 |
| Coherence (Failure) | 0.3923 |
| Gradient Magnitude (Success) | 0.1491 |
| Gradient Magnitude (Failure) | 0.2756 |
| Activation Separation | 0.8411 |
| Cosine Distance | 0.0022 |
| Clusters | 1,370 |
| Noise Fraction | 0.2961 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1439_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0375 |
| Coherence (Success) | 0.2263 |
| Coherence (Failure) | 0.0360 |
| Gradient Magnitude (Success) | 0.1263 |
| Gradient Magnitude (Failure) | 0.1278 |
| Activation Separation | 0.9433 |
| Cosine Distance | 0.0015 |
| Clusters | 1,500 |
| Noise Fraction | 0.3087 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1724_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2578 |
| Coherence (Success) | 0.1005 |
| Coherence (Failure) | 0.0903 |
| Gradient Magnitude (Success) | 0.1056 |
| Gradient Magnitude (Failure) | 0.1508 |
| Activation Separation | 1.0031 |
| Cosine Distance | 0.0025 |
| Clusters | 1,481 |
| Noise Fraction | 0.2536 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2009_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1139 |
| Coherence (Success) | 0.2065 |
| Coherence (Failure) | 0.0300 |
| Gradient Magnitude (Success) | 0.1409 |
| Gradient Magnitude (Failure) | 0.1474 |
| Activation Separation | 1.0888 |
| Cosine Distance | 0.0047 |
| Clusters | 1,685 |
| Noise Fraction | 0.2559 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep2126_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4122 |
| Coherence (Success) | 0.1204 |
| Coherence (Failure) | 0.2045 |
| Gradient Magnitude (Success) | 0.0876 |
| Gradient Magnitude (Failure) | 0.1845 |
| Activation Separation | 1.2968 |
| Cosine Distance | 0.0069 |
| Clusters | 1,617 |
| Noise Fraction | 0.2539 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2295_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7079 |
| Coherence (Success) | 0.4312 |
| Coherence (Failure) | 0.4258 |
| Gradient Magnitude (Success) | 0.1764 |
| Gradient Magnitude (Failure) | 0.3090 |
| Activation Separation | 1.2532 |
| Cosine Distance | 0.0057 |
| Clusters | 1,575 |
| Noise Fraction | 0.2451 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2562_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7069 |
| Coherence (Success) | 0.1108 |
| Coherence (Failure) | 0.1425 |
| Gradient Magnitude (Success) | 0.1138 |
| Gradient Magnitude (Failure) | 0.1618 |
| Activation Separation | 1.2795 |
| Cosine Distance | 0.0048 |
| Clusters | 1,629 |
| Noise Fraction | 0.2827 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2848_lower4.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9021 |
| Coherence (Success) | 0.4584 |
| Coherence (Failure) | 0.4386 |
| Gradient Magnitude (Success) | 0.2475 |
| Gradient Magnitude (Failure) | 0.3888 |
| Activation Separation | 1.5526 |
| Cosine Distance | 0.0073 |
| Clusters | 1,773 |
| Noise Fraction | 0.2658 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3128_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5381 |
| Coherence (Success) | 0.2485 |
| Coherence (Failure) | 0.1744 |
| Gradient Magnitude (Success) | 0.1533 |
| Gradient Magnitude (Failure) | 0.3254 |
| Activation Separation | 1.9856 |
| Cosine Distance | 0.0105 |
| Clusters | 1,515 |
| Noise Fraction | 0.3050 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3402_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6506 |
| Coherence (Success) | 0.3052 |
| Coherence (Failure) | 0.1581 |
| Gradient Magnitude (Success) | 0.2141 |
| Gradient Magnitude (Failure) | 0.2329 |
| Activation Separation | 1.9335 |
| Cosine Distance | 0.0071 |
| Clusters | 1,299 |
| Noise Fraction | 0.3108 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3688_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7251 |
| Coherence (Success) | 0.2275 |
| Coherence (Failure) | 0.1766 |
| Gradient Magnitude (Success) | 0.1903 |
| Gradient Magnitude (Failure) | 0.2269 |
| Activation Separation | 2.0900 |
| Cosine Distance | 0.0092 |
| Clusters | 1,295 |
| Noise Fraction | 0.3475 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3961_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8679 |
| Coherence (Success) | 0.5167 |
| Coherence (Failure) | 0.4529 |
| Gradient Magnitude (Success) | 0.3157 |
| Gradient Magnitude (Failure) | 0.4505 |
| Activation Separation | 2.0059 |
| Cosine Distance | 0.0089 |
| Clusters | 1,392 |
| Noise Fraction | 0.2994 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4228_lower5.900_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2772 |
| Coherence (Success) | 0.1884 |
| Coherence (Failure) | 0.1497 |
| Gradient Magnitude (Success) | 0.1483 |
| Gradient Magnitude (Failure) | 0.2584 |
| Activation Separation | 1.7875 |
| Cosine Distance | 0.0038 |
| Clusters | 1,285 |
| Noise Fraction | 0.3247 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4506_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8163 |
| Coherence (Success) | 0.2743 |
| Coherence (Failure) | 0.3100 |
| Gradient Magnitude (Success) | 0.2237 |
| Gradient Magnitude (Failure) | 0.3853 |
| Activation Separation | 2.3697 |
| Cosine Distance | 0.0078 |
| Clusters | 1,206 |
| Noise Fraction | 0.3461 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4773_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5300 |
| Coherence (Success) | 0.1654 |
| Coherence (Failure) | 0.0800 |
| Gradient Magnitude (Success) | 0.1625 |
| Gradient Magnitude (Failure) | 0.1773 |
| Activation Separation | 2.0842 |
| Cosine Distance | 0.0063 |
| Clusters | 1,335 |
| Noise Fraction | 0.3295 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5043_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0283 |
| Coherence (Success) | 0.1535 |
| Coherence (Failure) | 0.0436 |
| Gradient Magnitude (Success) | 0.1605 |
| Gradient Magnitude (Failure) | 0.1788 |
| Activation Separation | 2.0600 |
| Cosine Distance | 0.0071 |
| Clusters | 1,319 |
| Noise Fraction | 0.3382 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5306_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2657 |
| Coherence (Success) | 0.1180 |
| Coherence (Failure) | 0.0048 |
| Gradient Magnitude (Success) | 0.1336 |
| Gradient Magnitude (Failure) | 0.1370 |
| Activation Separation | 2.4453 |
| Cosine Distance | 0.0080 |
| Clusters | 1,279 |
| Noise Fraction | 0.3464 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep5445_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7015 |
| Coherence (Success) | 0.3474 |
| Coherence (Failure) | 0.0976 |
| Gradient Magnitude (Success) | 0.2254 |
| Gradient Magnitude (Failure) | 0.2435 |
| Activation Separation | 1.9739 |
| Cosine Distance | 0.0066 |
| Clusters | 1,367 |
| Noise Fraction | 0.3461 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5574_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0295 |
| Coherence (Success) | 0.0756 |
| Coherence (Failure) | 0.0470 |
| Gradient Magnitude (Success) | 0.1365 |
| Gradient Magnitude (Failure) | 0.1612 |
| Activation Separation | 2.5074 |
| Cosine Distance | 0.0055 |
| Clusters | 1,275 |
| Noise Fraction | 0.3642 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5838_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5666 |
| Coherence (Success) | 0.1805 |
| Coherence (Failure) | 0.0986 |
| Gradient Magnitude (Success) | 0.2105 |
| Gradient Magnitude (Failure) | 0.2226 |
| Activation Separation | 2.8800 |
| Cosine Distance | 0.0073 |
| Clusters | 1,351 |
| Noise Fraction | 0.3736 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6090_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1859 |
| Coherence (Success) | 0.2365 |
| Coherence (Failure) | 0.0535 |
| Gradient Magnitude (Success) | 0.2115 |
| Gradient Magnitude (Failure) | 0.1730 |
| Activation Separation | 2.3809 |
| Cosine Distance | 0.0078 |
| Clusters | 1,490 |
| Noise Fraction | 0.3279 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6342_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7271 |
| Coherence (Success) | 0.5098 |
| Coherence (Failure) | 0.0782 |
| Gradient Magnitude (Success) | 0.3804 |
| Gradient Magnitude (Failure) | 0.2511 |
| Activation Separation | 3.0110 |
| Cosine Distance | 0.0095 |
| Clusters | 1,447 |
| Noise Fraction | 0.3783 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6598_lower7.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5069 |
| Coherence (Success) | 0.2087 |
| Coherence (Failure) | 0.0420 |
| Gradient Magnitude (Success) | 0.1983 |
| Gradient Magnitude (Failure) | 0.1580 |
| Activation Separation | 2.7352 |
| Cosine Distance | 0.0073 |
| Clusters | 1,318 |
| Noise Fraction | 0.3558 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6847_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7028 |
| Coherence (Success) | 0.4747 |
| Coherence (Failure) | 0.2341 |
| Gradient Magnitude (Success) | 0.3625 |
| Gradient Magnitude (Failure) | 0.3416 |
| Activation Separation | 2.9972 |
| Cosine Distance | 0.0083 |
| Clusters | 1,324 |
| Noise Fraction | 0.3509 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7095_lower7.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3378 |
| Coherence (Success) | 0.1958 |
| Coherence (Failure) | 0.0202 |
| Gradient Magnitude (Success) | 0.2173 |
| Gradient Magnitude (Failure) | 0.2106 |
| Activation Separation | 3.4043 |
| Cosine Distance | 0.0107 |
| Clusters | 1,281 |
| Noise Fraction | 0.3301 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7337_lower7.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2962 |
| Coherence (Success) | 0.1195 |
| Coherence (Failure) | 0.0184 |
| Gradient Magnitude (Success) | 0.1711 |
| Gradient Magnitude (Failure) | 0.1659 |
| Activation Separation | 3.8081 |
| Cosine Distance | 0.0153 |
| Clusters | 1,325 |
| Noise Fraction | 0.3738 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7579_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6457 |
| Coherence (Success) | 0.3970 |
| Coherence (Failure) | 0.0244 |
| Gradient Magnitude (Success) | 0.3104 |
| Gradient Magnitude (Failure) | 0.2065 |
| Activation Separation | 2.9062 |
| Cosine Distance | 0.0091 |
| Clusters | 1,273 |
| Noise Fraction | 0.3615 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7832_lower7.000_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7059 |
| Coherence (Success) | 0.1873 |
| Coherence (Failure) | 0.1722 |
| Gradient Magnitude (Success) | 0.1987 |
| Gradient Magnitude (Failure) | 0.2510 |
| Activation Separation | 3.5628 |
| Cosine Distance | 0.0143 |
| Clusters | 1,173 |
| Noise Fraction | 0.3850 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8073_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5597 |
| Coherence (Success) | 0.2468 |
| Coherence (Failure) | 0.0332 |
| Gradient Magnitude (Success) | 0.2433 |
| Gradient Magnitude (Failure) | 0.1943 |
| Activation Separation | 3.5224 |
| Cosine Distance | 0.0128 |
| Clusters | 1,239 |
| Noise Fraction | 0.3851 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8315_lower7.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0377 |
| Coherence (Success) | 0.3524 |
| Coherence (Failure) | 0.0372 |
| Gradient Magnitude (Success) | 0.3560 |
| Gradient Magnitude (Failure) | 0.1852 |
| Activation Separation | 3.0869 |
| Cosine Distance | 0.0108 |
| Clusters | 1,265 |
| Noise Fraction | 0.3890 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8559_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4212 |
| Coherence (Success) | 0.1337 |
| Coherence (Failure) | 0.0711 |
| Gradient Magnitude (Success) | 0.1861 |
| Gradient Magnitude (Failure) | 0.2970 |
| Activation Separation | 4.0563 |
| Cosine Distance | 0.0184 |
| Clusters | 1,276 |
| Noise Fraction | 0.3821 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8797_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1547 |
| Coherence (Success) | 0.1572 |
| Coherence (Failure) | 0.0316 |
| Gradient Magnitude (Success) | 0.1856 |
| Gradient Magnitude (Failure) | 0.1892 |
| Activation Separation | 4.9656 |
| Cosine Distance | 0.0286 |
| Clusters | 1,367 |
| Noise Fraction | 0.3580 |
| RSA Alignment (ρ) | -0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9038_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6628 |
| Coherence (Success) | 0.1537 |
| Coherence (Failure) | 0.0567 |
| Gradient Magnitude (Success) | 0.1892 |
| Gradient Magnitude (Failure) | 0.2178 |
| Activation Separation | 4.3629 |
| Cosine Distance | 0.0229 |
| Clusters | 1,439 |
| Noise Fraction | 0.3552 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9277_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3242 |
| Coherence (Success) | 0.1595 |
| Coherence (Failure) | 0.0894 |
| Gradient Magnitude (Success) | 0.2194 |
| Gradient Magnitude (Failure) | 0.1853 |
| Activation Separation | 3.4637 |
| Cosine Distance | 0.0139 |
| Clusters | 1,436 |
| Noise Fraction | 0.3407 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9503_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7226 |
| Coherence (Success) | 0.3961 |
| Coherence (Failure) | 0.1655 |
| Gradient Magnitude (Success) | 0.2969 |
| Gradient Magnitude (Failure) | 0.2700 |
| Activation Separation | 3.3921 |
| Cosine Distance | 0.0138 |
| Clusters | 1,301 |
| Noise Fraction | 0.3566 |
| RSA Alignment (ρ) | 0.0881 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9741_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3049 |
| Coherence (Success) | 0.1645 |
| Coherence (Failure) | 0.0800 |
| Gradient Magnitude (Success) | 0.2208 |
| Gradient Magnitude (Failure) | 0.2780 |
| Activation Separation | 5.2098 |
| Cosine Distance | 0.0265 |
| Clusters | 1,332 |
| Noise Fraction | 0.3408 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9982_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5784 |
| Coherence (Success) | 0.2618 |
| Coherence (Failure) | 0.0609 |
| Gradient Magnitude (Success) | 0.2487 |
| Gradient Magnitude (Failure) | 0.2023 |
| Activation Separation | 3.2961 |
| Cosine Distance | 0.0119 |
| Clusters | 1,387 |
| Noise Fraction | 0.3680 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10208_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4691 |
| Coherence (Success) | 0.2759 |
| Coherence (Failure) | 0.0624 |
| Gradient Magnitude (Success) | 0.2738 |
| Gradient Magnitude (Failure) | 0.2180 |
| Activation Separation | 3.6958 |
| Cosine Distance | 0.0164 |
| Clusters | 1,303 |
| Noise Fraction | 0.3599 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10434_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1233 |
| Coherence (Success) | 0.1426 |
| Coherence (Failure) | -0.0171 |
| Gradient Magnitude (Success) | 0.1741 |
| Gradient Magnitude (Failure) | 0.1545 |
| Activation Separation | 4.2209 |
| Cosine Distance | 0.0204 |
| Clusters | 1,225 |
| Noise Fraction | 0.3283 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10662_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6633 |
| Coherence (Success) | 0.1249 |
| Coherence (Failure) | 0.0791 |
| Gradient Magnitude (Success) | 0.1895 |
| Gradient Magnitude (Failure) | 0.2524 |
| Activation Separation | 2.9866 |
| Cosine Distance | 0.0120 |
| Clusters | 1,042 |
| Noise Fraction | 0.3478 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10903_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2207 |
| Coherence (Success) | 0.1778 |
| Coherence (Failure) | 0.0308 |
| Gradient Magnitude (Success) | 0.2124 |
| Gradient Magnitude (Failure) | 0.1820 |
| Activation Separation | 4.2004 |
| Cosine Distance | 0.0176 |
| Clusters | 1,273 |
| Noise Fraction | 0.3823 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11129_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2343 |
| Coherence (Success) | 0.0337 |
| Coherence (Failure) | 0.0384 |
| Gradient Magnitude (Success) | 0.1310 |
| Gradient Magnitude (Failure) | 0.1820 |
| Activation Separation | 3.7495 |
| Cosine Distance | 0.0149 |
| Clusters | 1,211 |
| Noise Fraction | 0.3526 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11352_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6181 |
| Coherence (Success) | 0.2752 |
| Coherence (Failure) | 0.1873 |
| Gradient Magnitude (Success) | 0.2582 |
| Gradient Magnitude (Failure) | 0.2746 |
| Activation Separation | 4.4227 |
| Cosine Distance | 0.0232 |
| Clusters | 1,161 |
| Noise Fraction | 0.4060 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11585_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6457 |
| Coherence (Success) | 0.1517 |
| Coherence (Failure) | 0.2848 |
| Gradient Magnitude (Success) | 0.2170 |
| Gradient Magnitude (Failure) | 0.4320 |
| Activation Separation | 3.2546 |
| Cosine Distance | 0.0135 |
| Clusters | 1,233 |
| Noise Fraction | 0.3968 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11801_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4385 |
| Coherence (Success) | 0.3140 |
| Coherence (Failure) | -0.0062 |
| Gradient Magnitude (Success) | 0.2573 |
| Gradient Magnitude (Failure) | 0.1701 |
| Activation Separation | 2.8290 |
| Cosine Distance | 0.0084 |
| Clusters | 1,113 |
| Noise Fraction | 0.3784 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12034_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6609 |
| Coherence (Success) | 0.3486 |
| Coherence (Failure) | 0.2502 |
| Gradient Magnitude (Success) | 0.3137 |
| Gradient Magnitude (Failure) | 0.3219 |
| Activation Separation | 3.1397 |
| Cosine Distance | 0.0137 |
| Clusters | 1,127 |
| Noise Fraction | 0.3874 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12261_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5937 |
| Coherence (Success) | 0.0382 |
| Coherence (Failure) | 0.1432 |
| Gradient Magnitude (Success) | 0.1526 |
| Gradient Magnitude (Failure) | 0.2155 |
| Activation Separation | 4.8954 |
| Cosine Distance | 0.0326 |
| Clusters | 1,356 |
| Noise Fraction | 0.3490 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12488_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4313 |
| Coherence (Success) | 0.2208 |
| Coherence (Failure) | 0.0594 |
| Gradient Magnitude (Success) | 0.2147 |
| Gradient Magnitude (Failure) | 0.2125 |
| Activation Separation | 4.1231 |
| Cosine Distance | 0.0200 |
| Clusters | 1,157 |
| Noise Fraction | 0.3447 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12703_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5202 |
| Coherence (Success) | 0.1214 |
| Coherence (Failure) | 0.0070 |
| Gradient Magnitude (Success) | 0.1851 |
| Gradient Magnitude (Failure) | 0.1583 |
| Activation Separation | 4.2201 |
| Cosine Distance | 0.0202 |
| Clusters | 1,443 |
| Noise Fraction | 0.3602 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12923_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0858 |
| Coherence (Success) | 0.0577 |
| Coherence (Failure) | -0.0211 |
| Gradient Magnitude (Success) | 0.1377 |
| Gradient Magnitude (Failure) | 0.1281 |
| Activation Separation | 5.0520 |
| Cosine Distance | 0.0327 |
| Clusters | 1,417 |
| Noise Fraction | 0.3684 |
| RSA Alignment (ρ) | 0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13148_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7334 |
| Coherence (Success) | 0.3067 |
| Coherence (Failure) | 0.1526 |
| Gradient Magnitude (Success) | 0.2657 |
| Gradient Magnitude (Failure) | 0.2898 |
| Activation Separation | 3.2656 |
| Cosine Distance | 0.0129 |
| Clusters | 1,277 |
| Noise Fraction | 0.3840 |
| RSA Alignment (ρ) | 0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13380_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1744 |
| Coherence (Success) | 0.0040 |
| Coherence (Failure) | 0.0432 |
| Gradient Magnitude (Success) | 0.1176 |
| Gradient Magnitude (Failure) | 0.1843 |
| Activation Separation | 5.0316 |
| Cosine Distance | 0.0342 |
| Clusters | 1,426 |
| Noise Fraction | 0.3622 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13598_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3117 |
| Coherence (Success) | 0.1488 |
| Coherence (Failure) | 0.0462 |
| Gradient Magnitude (Success) | 0.1808 |
| Gradient Magnitude (Failure) | 0.1757 |
| Activation Separation | 3.7519 |
| Cosine Distance | 0.0206 |
| Clusters | 1,404 |
| Noise Fraction | 0.3162 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13819_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5402 |
| Coherence (Success) | 0.0954 |
| Coherence (Failure) | 0.0440 |
| Gradient Magnitude (Success) | 0.1719 |
| Gradient Magnitude (Failure) | 0.2249 |
| Activation Separation | 4.1130 |
| Cosine Distance | 0.0265 |
| Clusters | 1,511 |
| Noise Fraction | 0.3154 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14048_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4689 |
| Coherence (Success) | 0.1779 |
| Coherence (Failure) | 0.1156 |
| Gradient Magnitude (Success) | 0.2520 |
| Gradient Magnitude (Failure) | 0.2433 |
| Activation Separation | 3.8133 |
| Cosine Distance | 0.0183 |
| Clusters | 1,335 |
| Noise Fraction | 0.3616 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14263_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4426 |
| Coherence (Success) | 0.1446 |
| Coherence (Failure) | 0.0125 |
| Gradient Magnitude (Success) | 0.1860 |
| Gradient Magnitude (Failure) | 0.1638 |
| Activation Separation | 3.6654 |
| Cosine Distance | 0.0215 |
| Clusters | 1,420 |
| Noise Fraction | 0.3557 |
| RSA Alignment (ρ) | 0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14476_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1489 |
| Coherence (Success) | 0.1410 |
| Coherence (Failure) | 0.0663 |
| Gradient Magnitude (Success) | 0.1844 |
| Gradient Magnitude (Failure) | 0.1715 |
| Activation Separation | 3.4752 |
| Cosine Distance | 0.0207 |
| Clusters | 1,519 |
| Noise Fraction | 0.3126 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14696_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5510 |
| Coherence (Success) | 0.1857 |
| Coherence (Failure) | 0.0910 |
| Gradient Magnitude (Success) | 0.2058 |
| Gradient Magnitude (Failure) | 0.2110 |
| Activation Separation | 3.3381 |
| Cosine Distance | 0.0202 |
| Clusters | 1,449 |
| Noise Fraction | 0.3461 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14911_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3418 |
| Coherence (Success) | 0.0738 |
| Coherence (Failure) | 0.0109 |
| Gradient Magnitude (Success) | 0.1720 |
| Gradient Magnitude (Failure) | 0.1316 |
| Activation Separation | 2.2883 |
| Cosine Distance | 0.0092 |
| Clusters | 1,447 |
| Noise Fraction | 0.3539 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14913_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4190 |
| Coherence (Success) | 0.0293 |
| Coherence (Failure) | 0.1112 |
| Gradient Magnitude (Success) | 0.1382 |
| Gradient Magnitude (Failure) | 0.1951 |
| Activation Separation | 2.6706 |
| Cosine Distance | 0.0135 |
| Clusters | 1,361 |
| Noise Fraction | 0.3920 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep3 | 3 | 0.1125 | 0.8712 | 0.2822 | 0.1039 | 0.0412 | 0.1691 | 0.0007 | -0.2611 |
| collect_sapling @ ep4 | 4 | -0.0701 | 0.9016 | 0.3411 | 0.0922 | 0.0462 | 0.1970 | 0.0012 | -0.3928 |
| place_plant @ ep4 | 4 | 0.2482 | 0.8362 | 0.4257 | 0.0957 | 0.0448 | 0.2029 | 0.0009 | nan |
| collect_wood @ ep15 | 15 | -0.2225 | 0.9189 | 0.0844 | 0.0979 | 0.0407 | 0.2061 | 0.0009 | -0.6547 |
| place_table @ ep15 | 15 | -0.1358 | 0.6361 | 0.2619 | 0.0881 | 0.0374 | 0.1964 | 0.0010 | -0.2611 |
| collect_drink @ ep20 | 20 | 0.0212 | 0.7412 | 0.2092 | 0.1365 | 0.0571 | 0.2390 | 0.0006 | -0.6547 |
| eat_cow @ ep111 | 111 | 0.8540 | 0.8148 | 0.7263 | 0.2327 | 0.2217 | 0.6872 | 0.0003 | nan |
| make_wood_pickaxe @ ep205 | 205 | 0.7719 | 0.4323 | 0.6089 | 0.2644 | 0.1889 | 1.2835 | 0.0004 | -0.4352 |
| defeat_zombie @ ep209 | 209 | 0.5205 | 0.2012 | 0.1599 | 0.2397 | 0.1559 | 1.3374 | 0.0004 | -0.2791 |
| make_wood_sword @ ep291 | 291 | 0.1528 | 0.0112 | 0.1550 | 0.1381 | 0.1441 | 2.1228 | 0.0008 | -0.3489 |
| defeat_skeleton @ ep792 | 792 | 0.0717 | 0.1002 | 0.1369 | 0.1121 | 0.1632 | 0.8617 | 0.0009 | -0.1047 |
| collect_stone @ ep995 | 995 | 0.9303 | 0.3996 | 0.7388 | 0.1664 | 0.2860 | 0.3543 | 0.0004 | -0.1396 |
| place_furnace @ ep995 | 995 | 0.8340 | 0.5528 | 0.2889 | 0.1903 | 0.1889 | 0.3410 | 0.0004 | -0.1846 |
| place_stone @ ep995 | 995 | 0.8946 | 0.4135 | 0.6493 | 0.1683 | 0.2918 | 0.3790 | 0.0006 | -0.3140 |
| collect_coal @ ep1320 | 1,320 | 0.6980 | 0.2017 | 0.3923 | 0.1491 | 0.2756 | 0.8411 | 0.0022 | -0.3482 |
| make_stone_sword @ ep2126 | 2,126 | 0.4122 | 0.1204 | 0.2045 | 0.0876 | 0.1845 | 1.2968 | 0.0069 | -0.1662 |
| make_stone_pickaxe @ ep5445 | 5,445 | 0.7015 | 0.3474 | 0.0976 | 0.2254 | 0.2435 | 1.9739 | 0.0066 | -0.1396 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 290 | -0.3284 | 0.1061 | 0.1093 | 0.1377 | 0.1447 | 1.9804 | 0.0007 | 0.0000 |
| Step 100,000 | 568 | 0.4600 | 0.7907 | 0.4027 | 0.2100 | 0.2370 | 0.7776 | 0.0003 | -0.3140 |
| Step 150,000 | 864 | 0.3356 | 0.2503 | 0.3307 | 0.1643 | 0.2289 | 0.8011 | 0.0010 | -0.0698 |
| Step 200,000 | 1,156 | -0.0569 | 0.0919 | 0.1170 | 0.0716 | 0.1398 | 0.9710 | 0.0015 | -0.1047 |
| Step 250,000 | 1,439 | 0.0375 | 0.2263 | 0.0360 | 0.1263 | 0.1278 | 0.9433 | 0.0015 | 0.0000 |
| Step 300,000 | 1,724 | 0.2578 | 0.1005 | 0.0903 | 0.1056 | 0.1508 | 1.0031 | 0.0025 | -0.2443 |
| Step 350,000 | 2,009 | -0.1139 | 0.2065 | 0.0300 | 0.1409 | 0.1474 | 1.0888 | 0.0047 | -0.2094 |
| Step 400,000 | 2,295 | 0.7079 | 0.4312 | 0.4258 | 0.1764 | 0.3090 | 1.2532 | 0.0057 | -0.2443 |
| Step 450,000 | 2,562 | 0.7069 | 0.1108 | 0.1425 | 0.1138 | 0.1618 | 1.2795 | 0.0048 | -0.3838 |
| Step 500,000 | 2,848 | 0.9021 | 0.4584 | 0.4386 | 0.2475 | 0.3888 | 1.5526 | 0.0073 | -0.1396 |
| Step 550,000 | 3,128 | 0.5381 | 0.2485 | 0.1744 | 0.1533 | 0.3254 | 1.9856 | 0.0105 | -0.0698 |
| Step 600,000 | 3,402 | 0.6506 | 0.3052 | 0.1581 | 0.2141 | 0.2329 | 1.9335 | 0.0071 | -0.0698 |
| Step 650,000 | 3,688 | 0.7251 | 0.2275 | 0.1766 | 0.1903 | 0.2269 | 2.0900 | 0.0092 | -0.3140 |
| Step 700,000 | 3,961 | 0.8679 | 0.5167 | 0.4529 | 0.3157 | 0.4505 | 2.0059 | 0.0089 | -0.0185 |
| Step 750,000 | 4,228 | 0.2772 | 0.1884 | 0.1497 | 0.1483 | 0.2584 | 1.7875 | 0.0038 | -0.2094 |
| Step 800,000 | 4,506 | 0.8163 | 0.2743 | 0.3100 | 0.2237 | 0.3853 | 2.3697 | 0.0078 | 0.0000 |
| Step 850,000 | 4,773 | 0.5300 | 0.1654 | 0.0800 | 0.1625 | 0.1773 | 2.0842 | 0.0063 | -0.1108 |
| Step 900,000 | 5,043 | -0.0283 | 0.1535 | 0.0436 | 0.1605 | 0.1788 | 2.0600 | 0.0071 | -0.1745 |
| Step 950,000 | 5,306 | 0.2657 | 0.1180 | 0.0048 | 0.1336 | 0.1370 | 2.4453 | 0.0080 | -0.1662 |
| Step 1,000,000 | 5,574 | 0.0295 | 0.0756 | 0.0470 | 0.1365 | 0.1612 | 2.5074 | 0.0055 | -0.1047 |
| Step 1,050,000 | 5,838 | 0.5666 | 0.1805 | 0.0986 | 0.2105 | 0.2226 | 2.8800 | 0.0073 | -0.1108 |
| Step 1,100,000 | 6,090 | 0.1859 | 0.2365 | 0.0535 | 0.2115 | 0.1730 | 2.3809 | 0.0078 | -0.1396 |
| Step 1,150,000 | 6,342 | 0.7271 | 0.5098 | 0.0782 | 0.3804 | 0.2511 | 3.0110 | 0.0095 | 0.0000 |
| Step 1,200,000 | 6,598 | 0.5069 | 0.2087 | 0.0420 | 0.1983 | 0.1580 | 2.7352 | 0.0073 | -0.0698 |
| Step 1,250,000 | 6,847 | 0.7028 | 0.4747 | 0.2341 | 0.3625 | 0.3416 | 2.9972 | 0.0083 | -0.0349 |
| Step 1,300,000 | 7,095 | 0.3378 | 0.1958 | 0.0202 | 0.2173 | 0.2106 | 3.4043 | 0.0107 | -0.0349 |
| Step 1,350,000 | 7,337 | 0.2962 | 0.1195 | 0.0184 | 0.1711 | 0.1659 | 3.8081 | 0.0153 | -0.0698 |
| Step 1,400,000 | 7,579 | 0.6457 | 0.3970 | 0.0244 | 0.3104 | 0.2065 | 2.9062 | 0.0091 | -0.0698 |
| Step 1,450,000 | 7,832 | 0.7059 | 0.1873 | 0.1722 | 0.1987 | 0.2510 | 3.5628 | 0.0143 | 0.0000 |
| Step 1,500,000 | 8,073 | 0.5597 | 0.2468 | 0.0332 | 0.2433 | 0.1943 | 3.5224 | 0.0128 | -0.0698 |
| Step 1,550,000 | 8,315 | 0.0377 | 0.3524 | 0.0372 | 0.3560 | 0.1852 | 3.0869 | 0.0108 | -0.1477 |
| Step 1,600,000 | 8,559 | 0.4212 | 0.1337 | 0.0711 | 0.1861 | 0.2970 | 4.0563 | 0.0184 | -0.0923 |
| Step 1,650,000 | 8,797 | 0.1547 | 0.1572 | 0.0316 | 0.1856 | 0.1892 | 4.9656 | 0.0286 | -0.0098 |
| Step 1,700,000 | 9,038 | 0.6628 | 0.1537 | 0.0567 | 0.1892 | 0.2178 | 4.3629 | 0.0229 | -0.1108 |
| Step 1,750,000 | 9,277 | 0.3242 | 0.1595 | 0.0894 | 0.2194 | 0.1853 | 3.4637 | 0.0139 | -0.0739 |
| Step 1,800,000 | 9,503 | 0.7226 | 0.3961 | 0.1655 | 0.2969 | 0.2700 | 3.3921 | 0.0138 | 0.0881 |
| Step 1,850,000 | 9,741 | 0.3049 | 0.1645 | 0.0800 | 0.2208 | 0.2780 | 5.2098 | 0.0265 | -0.0739 |
| Step 1,900,000 | 9,982 | 0.5784 | 0.2618 | 0.0609 | 0.2487 | 0.2023 | 3.2961 | 0.0119 | -0.0923 |
| Step 1,950,000 | 10,208 | 0.4691 | 0.2759 | 0.0624 | 0.2738 | 0.2180 | 3.6958 | 0.0164 | -0.0923 |
| Step 2,000,000 | 10,434 | 0.1233 | 0.1426 | -0.0171 | 0.1741 | 0.1545 | 4.2209 | 0.0204 | -0.0739 |
| Step 2,050,000 | 10,662 | 0.6633 | 0.1249 | 0.0791 | 0.1895 | 0.2524 | 2.9866 | 0.0120 | 0.1292 |
| Step 2,100,000 | 10,903 | 0.2207 | 0.1778 | 0.0308 | 0.2124 | 0.1820 | 4.2004 | 0.0176 | 0.0185 |
| Step 2,150,000 | 11,129 | 0.2343 | 0.0337 | 0.0384 | 0.1310 | 0.1820 | 3.7495 | 0.0149 | 0.1745 |
| Step 2,200,000 | 11,352 | 0.6181 | 0.2752 | 0.1873 | 0.2582 | 0.2746 | 4.4227 | 0.0232 | -0.0185 |
| Step 2,250,000 | 11,585 | 0.6457 | 0.1517 | 0.2848 | 0.2170 | 0.4320 | 3.2546 | 0.0135 | 0.0369 |
| Step 2,300,000 | 11,801 | 0.4385 | 0.3140 | -0.0062 | 0.2573 | 0.1701 | 2.8290 | 0.0084 | 0.1745 |
| Step 2,350,000 | 12,034 | 0.6609 | 0.3486 | 0.2502 | 0.3137 | 0.3219 | 3.1397 | 0.0137 | 0.0369 |
| Step 2,400,000 | 12,261 | 0.5937 | 0.0382 | 0.1432 | 0.1526 | 0.2155 | 4.8954 | 0.0326 | 0.0000 |
| Step 2,450,000 | 12,488 | 0.4313 | 0.2208 | 0.0594 | 0.2147 | 0.2125 | 4.1231 | 0.0200 | 0.0185 |
| Step 2,500,000 | 12,703 | 0.5202 | 0.1214 | 0.0070 | 0.1851 | 0.1583 | 4.2201 | 0.0202 | 0.1396 |
| Step 2,550,000 | 12,923 | 0.0858 | 0.0577 | -0.0211 | 0.1377 | 0.1281 | 5.0520 | 0.0327 | 0.0698 |
| Step 2,600,000 | 13,148 | 0.7334 | 0.3067 | 0.1526 | 0.2657 | 0.2898 | 3.2656 | 0.0129 | 0.1047 |
| Step 2,650,000 | 13,380 | 0.1744 | 0.0040 | 0.0432 | 0.1176 | 0.1843 | 5.0316 | 0.0342 | 0.0554 |
| Step 2,700,000 | 13,598 | 0.3117 | 0.1488 | 0.0462 | 0.1808 | 0.1757 | 3.7519 | 0.0206 | 0.1396 |
| Step 2,750,000 | 13,819 | 0.5402 | 0.0954 | 0.0440 | 0.1719 | 0.2249 | 4.1130 | 0.0265 | 0.2094 |
| Step 2,800,000 | 14,048 | 0.4689 | 0.1779 | 0.1156 | 0.2520 | 0.2433 | 3.8133 | 0.0183 | 0.2443 |
| Step 2,850,000 | 14,263 | 0.4426 | 0.1446 | 0.0125 | 0.1860 | 0.1638 | 3.6654 | 0.0215 | 0.0349 |
| Step 2,900,000 | 14,476 | 0.1489 | 0.1410 | 0.0663 | 0.1844 | 0.1715 | 3.4752 | 0.0207 | 0.1396 |
| Step 2,950,000 | 14,696 | 0.5510 | 0.1857 | 0.0910 | 0.2058 | 0.2110 | 3.3381 | 0.0202 | 0.2791 |
| Step 3,000,000 | 14,911 | 0.3418 | 0.0738 | 0.0109 | 0.1720 | 0.1316 | 2.2883 | 0.0092 | 0.1745 |
| Final | 14,913 | 0.4190 | 0.0293 | 0.1112 | 0.1382 | 0.1951 | 2.6706 | 0.0135 | 0.1745 |
