# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 5  
**Total episodes:** 14,207  
**Experiment root:** `experiment_root\seed_5`  
**Generated:** 2026-03-23 06:32

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_sapling @ ep1 | 1 | 0.4216 | 0.7357 | -0.0019 | 0.0611 | 0.0184 | 0.1170 | 0.0004 | nan |
| wake_up @ ep2 | 2 | 0.3386 | 0.8784 | 0.0399 | 0.0836 | 0.0361 | 0.1215 | 0.0005 | nan |
| place_plant @ ep3 | 3 | -0.0465 | 0.8908 | 0.0662 | 0.0862 | 0.0323 | 0.1690 | 0.0010 | -0.3928 |
| collect_wood @ ep10 | 10 | 0.2345 | 0.9056 | -0.0158 | 0.0851 | 0.0359 | 0.1865 | 0.0012 | -0.6547 |
| place_table @ ep38 | 38 | 0.3987 | 0.9226 | 0.4167 | 0.1964 | 0.0906 | 0.3518 | 0.0010 | -0.6547 |
| collect_drink @ ep40 | 40 | 0.2523 | 0.9212 | 0.0092 | 0.1949 | 0.0816 | 0.3750 | 0.0010 | nan |
| make_wood_pickaxe @ ep58 | 58 | 0.8139 | 0.8469 | 0.4860 | 0.2888 | 0.1569 | 0.5289 | 0.0002 | -0.3482 |
| make_wood_sword @ ep63 | 63 | 0.6543 | 0.9200 | -0.0174 | 0.3055 | 0.1266 | 0.4471 | 0.0002 | -0.3928 |
| eat_cow @ ep77 | 77 | 0.7703 | 0.8977 | 0.3913 | 0.2481 | 0.1360 | 0.5027 | 0.0003 | -0.1741 |
| defeat_zombie @ ep97 | 97 | 0.8603 | 0.9262 | 0.6708 | 0.3492 | 0.1935 | 0.5279 | 0.0002 | -0.2791 |
| defeat_skeleton @ ep149 | 149 | 0.7919 | 0.7289 | 0.5031 | 0.2867 | 0.2594 | 0.5528 | 0.0002 | -0.1662 |
| eat_plant @ ep187 | 187 | 0.4691 | 0.4923 | 0.1067 | 0.1995 | 0.2011 | 0.6865 | 0.0002 | -0.3482 |
| collect_stone @ ep230 | 230 | -0.0442 | 0.1333 | 0.0992 | 0.1598 | 0.1234 | 1.5420 | 0.0006 | -0.1745 |
| make_stone_sword @ ep230 | 230 | 0.6929 | 0.6002 | 0.1491 | 0.2098 | 0.1361 | 1.4299 | 0.0006 | -0.6093 |
| periodic_step50000_ep273 | 273 | 0.5798 | 0.7141 | 0.6616 | 0.2132 | 0.2064 | 1.0396 | 0.0006 | -0.3140 |
| collect_coal @ ep372 | 372 | -0.1821 | 0.1648 | 0.2245 | 0.1456 | 0.1516 | 0.8727 | 0.0005 | -0.1745 |
| periodic_step100000_ep555 | 555 | -0.3212 | -0.0088 | -0.0125 | 0.0977 | 0.1531 | 1.1580 | 0.0005 | -0.1741 |
| periodic_step150000_ep842 | 842 | 0.2191 | 0.3429 | 0.1903 | 0.1855 | 0.1847 | 0.5917 | 0.0006 | 0.0000 |
| periodic_step200000_ep1124 | 1,124 | 0.4007 | 0.0529 | 0.3027 | 0.1043 | 0.2267 | 0.8247 | 0.0011 | -0.2443 |
| place_stone @ ep1136 | 1,136 | -0.3174 | 0.0048 | 0.1544 | 0.0810 | 0.1934 | 0.7219 | 0.0007 | -0.1047 |
| periodic_step250000_ep1407 | 1,407 | 0.5241 | 0.3719 | 0.1426 | 0.1885 | 0.1417 | 0.9513 | 0.0015 | -0.4536 |
| periodic_step300000_ep1682 | 1,682 | 0.0515 | 0.1162 | 0.1622 | 0.1755 | 0.1853 | 0.9874 | 0.0016 | -0.2094 |
| periodic_step350000_ep1942 | 1,942 | 0.6765 | 0.3232 | 0.1953 | 0.1616 | 0.3111 | 1.4238 | 0.0027 | -0.3140 |
| periodic_step400000_ep2203 | 2,203 | -0.1425 | 0.1664 | 0.1719 | 0.1264 | 0.3039 | 1.5074 | 0.0026 | -0.2791 |
| place_furnace @ ep2239 | 2,239 | -0.2746 | 0.1340 | 0.0921 | 0.1667 | 0.2022 | 1.4610 | 0.0018 | -0.2443 |
| periodic_step450000_ep2462 | 2,462 | 0.3118 | 0.1234 | 0.2077 | 0.1418 | 0.2764 | 1.1835 | 0.0023 | -0.2443 |
| periodic_step500000_ep2722 | 2,722 | -0.4254 | 0.0633 | 0.1034 | 0.1104 | 0.2074 | 1.3647 | 0.0026 | -0.1292 |
| periodic_step550000_ep2967 | 2,967 | 0.8818 | 0.2940 | 0.2626 | 0.2172 | 0.3277 | 1.6941 | 0.0053 | -0.1047 |
| periodic_step600000_ep3223 | 3,223 | 0.3701 | 0.0978 | 0.2015 | 0.1479 | 0.3294 | 2.1769 | 0.0058 | -0.2443 |
| periodic_step650000_ep3465 | 3,465 | 0.3805 | 0.1817 | 0.2394 | 0.1737 | 0.3913 | 2.1327 | 0.0067 | -0.0739 |
| periodic_step700000_ep3716 | 3,716 | 0.6588 | 0.1104 | 0.1671 | 0.1641 | 0.2317 | 2.3627 | 0.0076 | -0.1047 |
| periodic_step750000_ep3974 | 3,974 | 0.8102 | 0.4583 | 0.1519 | 0.2912 | 0.3199 | 3.3868 | 0.0128 | 0.1396 |
| periodic_step800000_ep4227 | 4,227 | -0.0110 | 0.1635 | 0.0195 | 0.1435 | 0.1539 | 2.5864 | 0.0117 | -0.1108 |
| periodic_step850000_ep4481 | 4,481 | 0.5435 | 0.1789 | 0.1974 | 0.1563 | 0.2192 | 3.0324 | 0.0153 | 0.0000 |
| make_stone_pickaxe @ ep4500 | 4,500 | 0.6190 | 0.0703 | 0.2228 | 0.1430 | 0.2709 | 3.2488 | 0.0176 | -0.0349 |
| periodic_step900000_ep4744 | 4,744 | 0.7829 | 0.4308 | 0.4054 | 0.2953 | 0.4174 | 2.6386 | 0.0114 | 0.0000 |
| periodic_step950000_ep5011 | 5,011 | 0.8032 | 0.3272 | 0.1826 | 0.2337 | 0.2344 | 3.0654 | 0.0158 | -0.0739 |
| periodic_step1000000_ep5271 | 5,271 | 0.6896 | 0.2356 | 0.2304 | 0.1855 | 0.3225 | 3.0937 | 0.0131 | 0.0000 |
| periodic_step1050000_ep5531 | 5,531 | 0.3162 | 0.4402 | -0.0205 | 0.3140 | 0.1322 | 2.9710 | 0.0154 | 0.0185 |
| periodic_step1100000_ep5775 | 5,775 | 0.8454 | 0.4648 | 0.2305 | 0.3675 | 0.3324 | 3.1769 | 0.0178 | -0.0369 |
| periodic_step1150000_ep6030 | 6,030 | 0.7539 | 0.2805 | 0.2576 | 0.2398 | 0.3546 | 3.1996 | 0.0171 | -0.1108 |
| periodic_step1200000_ep6276 | 6,276 | 0.1902 | 0.0955 | 0.0288 | 0.1230 | 0.1292 | 2.8489 | 0.0168 | -0.0923 |
| periodic_step1250000_ep6534 | 6,534 | 0.5521 | 0.0892 | 0.0408 | 0.1460 | 0.1453 | 2.1523 | 0.0068 | 0.1396 |
| periodic_step1300000_ep6782 | 6,782 | 0.5239 | 0.1263 | 0.1300 | 0.1803 | 0.2429 | 2.8625 | 0.0132 | 0.1292 |
| periodic_step1350000_ep7022 | 7,022 | 0.2232 | 0.1960 | -0.0347 | 0.2050 | 0.1137 | 3.0457 | 0.0115 | 0.2443 |
| periodic_step1400000_ep7271 | 7,271 | 0.6883 | 0.2423 | 0.1289 | 0.2255 | 0.3218 | 2.9328 | 0.0095 | 0.1047 |
| periodic_step1450000_ep7501 | 7,501 | 0.4050 | 0.1617 | 0.0386 | 0.1630 | 0.1879 | 3.7414 | 0.0196 | 0.1396 |
| periodic_step1500000_ep7726 | 7,726 | 0.1910 | 0.0044 | 0.0271 | 0.1164 | 0.1473 | 3.4756 | 0.0183 | 0.1745 |
| periodic_step1550000_ep7949 | 7,949 | 0.3950 | 0.0062 | 0.0784 | 0.0935 | 0.2192 | 3.4846 | 0.0164 | 0.1047 |
| periodic_step1600000_ep8168 | 8,168 | 0.6576 | 0.3460 | 0.2266 | 0.2921 | 0.3228 | 3.9383 | 0.0210 | 0.0185 |
| periodic_step1650000_ep8387 | 8,387 | 0.5040 | 0.1355 | 0.1097 | 0.1938 | 0.2216 | 4.5266 | 0.0308 | 0.0391 |
| periodic_step1700000_ep8616 | 8,616 | 0.5320 | 0.2843 | 0.1735 | 0.2637 | 0.3444 | 4.5275 | 0.0247 | -0.0554 |
| periodic_step1750000_ep8837 | 8,837 | 0.3433 | 0.0748 | -0.0171 | 0.1626 | 0.1381 | 3.8065 | 0.0179 | 0.0739 |
| periodic_step1800000_ep9069 | 9,069 | 0.4459 | 0.1751 | 0.0947 | 0.2013 | 0.2594 | 3.5217 | 0.0180 | 0.0185 |
| periodic_step1850000_ep9287 | 9,287 | 0.3620 | 0.2040 | 0.0683 | 0.1881 | 0.2174 | 4.6580 | 0.0369 | 0.0739 |
| periodic_step1900000_ep9503 | 9,503 | 0.4913 | 0.1886 | 0.2023 | 0.1890 | 0.3195 | 4.7327 | 0.0377 | 0.1272 |
| periodic_step1950000_ep9719 | 9,719 | 0.6941 | 0.2347 | 0.1550 | 0.2446 | 0.3434 | 3.6153 | 0.0181 | 0.0783 |
| periodic_step2000000_ep9936 | 9,936 | 0.1911 | 0.2306 | 0.0981 | 0.2175 | 0.2019 | 3.6957 | 0.0198 | 0.0000 |
| periodic_step2050000_ep10156 | 10,156 | 0.7703 | 0.3852 | 0.1098 | 0.2892 | 0.2732 | 4.4300 | 0.0277 | 0.1292 |
| periodic_step2100000_ep10376 | 10,376 | 0.6327 | 0.1007 | 0.0212 | 0.1805 | 0.2018 | 4.1301 | 0.0187 | 0.1292 |
| periodic_step2150000_ep10600 | 10,600 | 0.6010 | 0.2467 | 0.0761 | 0.2100 | 0.2027 | 3.8551 | 0.0193 | 0.0391 |
| periodic_step2200000_ep10814 | 10,814 | 0.5278 | 0.3100 | 0.0408 | 0.2447 | 0.1754 | 4.1254 | 0.0322 | 0.0587 |
| periodic_step2250000_ep11017 | 11,017 | 0.6940 | 0.3485 | 0.0759 | 0.2548 | 0.2221 | 3.4802 | 0.0165 | -0.0979 |
| periodic_step2300000_ep11228 | 11,228 | 0.6423 | 0.3350 | 0.1583 | 0.2540 | 0.3624 | 3.9974 | 0.0210 | 0.0923 |
| periodic_step2350000_ep11441 | 11,441 | 0.7664 | 0.3380 | 0.1891 | 0.2771 | 0.3236 | 3.0522 | 0.0144 | 0.2031 |
| periodic_step2400000_ep11672 | 11,672 | 0.4282 | 0.0856 | 0.1246 | 0.1431 | 0.2461 | 3.6775 | 0.0178 | 0.1468 |
| collect_iron @ ep11871 | 11,871 | 0.5993 | 0.1725 | 0.1488 | 0.1874 | 0.2647 | 4.2518 | 0.0327 | 0.0739 |
| periodic_step2450000_ep11894 | 11,894 | 0.5396 | 0.0951 | 0.0550 | 0.1575 | 0.1987 | 4.2626 | 0.0302 | 0.1860 |
| periodic_step2500000_ep12108 | 12,108 | 0.6134 | 0.1974 | 0.0550 | 0.2185 | 0.1885 | 2.8471 | 0.0146 | 0.0196 |
| periodic_step2550000_ep12311 | 12,311 | 0.5234 | 0.1668 | 0.1274 | 0.1857 | 0.2397 | 3.8372 | 0.0268 | 0.0587 |
| periodic_step2600000_ep12521 | 12,521 | 0.4529 | 0.3092 | 0.1107 | 0.2565 | 0.2603 | 4.3720 | 0.0269 | 0.2055 |
| periodic_step2650000_ep12733 | 12,733 | 0.4848 | 0.1278 | 0.0473 | 0.1683 | 0.1963 | 4.1121 | 0.0267 | 0.0739 |
| periodic_step2700000_ep12939 | 12,939 | 0.1755 | 0.3287 | 0.0688 | 0.2433 | 0.1910 | 3.2650 | 0.0205 | 0.0000 |
| periodic_step2750000_ep13153 | 13,153 | 0.4202 | 0.2731 | 0.0240 | 0.2461 | 0.1686 | 4.3009 | 0.0319 | 0.0369 |
| periodic_step2800000_ep13354 | 13,354 | 0.5869 | 0.2507 | 0.1531 | 0.2488 | 0.3437 | 4.8023 | 0.0299 | 0.0000 |
| periodic_step2850000_ep13562 | 13,562 | 0.5104 | 0.1998 | 0.0757 | 0.2437 | 0.2357 | 3.7459 | 0.0202 | 0.1108 |
| periodic_step2900000_ep13770 | 13,770 | 0.2798 | 0.2212 | 0.0259 | 0.2097 | 0.1611 | 4.2740 | 0.0303 | 0.0369 |
| periodic_step2950000_ep13984 | 13,984 | 0.7564 | 0.1412 | 0.0740 | 0.2024 | 0.2656 | 4.1299 | 0.0283 | 0.0923 |
| periodic_step3000000_ep14206 | 14,206 | 0.2496 | 0.0970 | 0.0810 | 0.1699 | 0.2267 | 4.1801 | 0.0310 | 0.2031 |
| final_step3000320_ep14207 | 14,207 | 0.2959 | 0.1679 | 0.0680 | 0.2031 | 0.2054 | 4.8714 | 0.0412 | 0.2216 |

---

## collect_sapling_ep1_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4216 |
| Coherence (Success) | 0.7357 |
| Coherence (Failure) | -0.0019 |
| Gradient Magnitude (Success) | 0.0611 |
| Gradient Magnitude (Failure) | 0.0184 |
| Activation Separation | 0.1170 |
| Cosine Distance | 0.0004 |
| Clusters | 1,410 |
| Noise Fraction | 0.2576 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## wake_up_ep2_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3386 |
| Coherence (Success) | 0.8784 |
| Coherence (Failure) | 0.0399 |
| Gradient Magnitude (Success) | 0.0836 |
| Gradient Magnitude (Failure) | 0.0361 |
| Activation Separation | 0.1215 |
| Cosine Distance | 0.0005 |
| Clusters | 1,497 |
| Noise Fraction | 0.2660 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep3_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0465 |
| Coherence (Success) | 0.8908 |
| Coherence (Failure) | 0.0662 |
| Gradient Magnitude (Success) | 0.0862 |
| Gradient Magnitude (Failure) | 0.0323 |
| Activation Separation | 0.1690 |
| Cosine Distance | 0.0010 |
| Clusters | 1,415 |
| Noise Fraction | 0.2589 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_wood_ep10_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2345 |
| Coherence (Success) | 0.9056 |
| Coherence (Failure) | -0.0158 |
| Gradient Magnitude (Success) | 0.0851 |
| Gradient Magnitude (Failure) | 0.0359 |
| Activation Separation | 0.1865 |
| Cosine Distance | 0.0012 |
| Clusters | 1,468 |
| Noise Fraction | 0.2751 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep38_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3987 |
| Coherence (Success) | 0.9226 |
| Coherence (Failure) | 0.4167 |
| Gradient Magnitude (Success) | 0.1964 |
| Gradient Magnitude (Failure) | 0.0906 |
| Activation Separation | 0.3518 |
| Cosine Distance | 0.0010 |
| Clusters | 1,465 |
| Noise Fraction | 0.2832 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep40_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2523 |
| Coherence (Success) | 0.9212 |
| Coherence (Failure) | 0.0092 |
| Gradient Magnitude (Success) | 0.1949 |
| Gradient Magnitude (Failure) | 0.0816 |
| Activation Separation | 0.3750 |
| Cosine Distance | 0.0010 |
| Clusters | 1,426 |
| Noise Fraction | 0.2939 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep58_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8139 |
| Coherence (Success) | 0.8469 |
| Coherence (Failure) | 0.4860 |
| Gradient Magnitude (Success) | 0.2888 |
| Gradient Magnitude (Failure) | 0.1569 |
| Activation Separation | 0.5289 |
| Cosine Distance | 0.0002 |
| Clusters | 1,532 |
| Noise Fraction | 0.2726 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep63_lower1.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6543 |
| Coherence (Success) | 0.9200 |
| Coherence (Failure) | -0.0174 |
| Gradient Magnitude (Success) | 0.3055 |
| Gradient Magnitude (Failure) | 0.1266 |
| Activation Separation | 0.4471 |
| Cosine Distance | 0.0002 |
| Clusters | 1,556 |
| Noise Fraction | 0.2655 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep77_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7703 |
| Coherence (Success) | 0.8977 |
| Coherence (Failure) | 0.3913 |
| Gradient Magnitude (Success) | 0.2481 |
| Gradient Magnitude (Failure) | 0.1360 |
| Activation Separation | 0.5027 |
| Cosine Distance | 0.0003 |
| Clusters | 1,667 |
| Noise Fraction | 0.2605 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Coal, Skeleton, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep97_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8603 |
| Coherence (Success) | 0.9262 |
| Coherence (Failure) | 0.6708 |
| Gradient Magnitude (Success) | 0.3492 |
| Gradient Magnitude (Failure) | 0.1935 |
| Activation Separation | 0.5279 |
| Cosine Distance | 0.0002 |
| Clusters | 1,647 |
| Noise Fraction | 0.2651 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep149_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7919 |
| Coherence (Success) | 0.7289 |
| Coherence (Failure) | 0.5031 |
| Gradient Magnitude (Success) | 0.2867 |
| Gradient Magnitude (Failure) | 0.2594 |
| Activation Separation | 0.5528 |
| Cosine Distance | 0.0002 |
| Clusters | 1,734 |
| Noise Fraction | 0.2730 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep187_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4691 |
| Coherence (Success) | 0.4923 |
| Coherence (Failure) | 0.1067 |
| Gradient Magnitude (Success) | 0.1995 |
| Gradient Magnitude (Failure) | 0.2011 |
| Activation Separation | 0.6865 |
| Cosine Distance | 0.0002 |
| Clusters | 1,664 |
| Noise Fraction | 0.2410 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep230_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0442 |
| Coherence (Success) | 0.1333 |
| Coherence (Failure) | 0.0992 |
| Gradient Magnitude (Success) | 0.1598 |
| Gradient Magnitude (Failure) | 0.1234 |
| Activation Separation | 1.5420 |
| Cosine Distance | 0.0006 |
| Clusters | 1,461 |
| Noise Fraction | 0.2781 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep230_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6929 |
| Coherence (Success) | 0.6002 |
| Coherence (Failure) | 0.1491 |
| Gradient Magnitude (Success) | 0.2098 |
| Gradient Magnitude (Failure) | 0.1361 |
| Activation Separation | 1.4299 |
| Cosine Distance | 0.0006 |
| Clusters | 1,502 |
| Noise Fraction | 0.2943 |
| RSA Alignment (ρ) | -0.6093 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep273_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5798 |
| Coherence (Success) | 0.7141 |
| Coherence (Failure) | 0.6616 |
| Gradient Magnitude (Success) | 0.2132 |
| Gradient Magnitude (Failure) | 0.2064 |
| Activation Separation | 1.0396 |
| Cosine Distance | 0.0006 |
| Clusters | 1,559 |
| Noise Fraction | 0.2745 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep372_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1821 |
| Coherence (Success) | 0.1648 |
| Coherence (Failure) | 0.2245 |
| Gradient Magnitude (Success) | 0.1456 |
| Gradient Magnitude (Failure) | 0.1516 |
| Activation Separation | 0.8727 |
| Cosine Distance | 0.0005 |
| Clusters | 1,615 |
| Noise Fraction | 0.2792 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep555_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3212 |
| Coherence (Success) | -0.0088 |
| Coherence (Failure) | -0.0125 |
| Gradient Magnitude (Success) | 0.0977 |
| Gradient Magnitude (Failure) | 0.1531 |
| Activation Separation | 1.1580 |
| Cosine Distance | 0.0005 |
| Clusters | 1,589 |
| Noise Fraction | 0.3046 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep842_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2191 |
| Coherence (Success) | 0.3429 |
| Coherence (Failure) | 0.1903 |
| Gradient Magnitude (Success) | 0.1855 |
| Gradient Magnitude (Failure) | 0.1847 |
| Activation Separation | 0.5917 |
| Cosine Distance | 0.0006 |
| Clusters | 1,611 |
| Noise Fraction | 0.2852 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1124_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4007 |
| Coherence (Success) | 0.0529 |
| Coherence (Failure) | 0.3027 |
| Gradient Magnitude (Success) | 0.1043 |
| Gradient Magnitude (Failure) | 0.2267 |
| Activation Separation | 0.8247 |
| Cosine Distance | 0.0011 |
| Clusters | 1,553 |
| Noise Fraction | 0.2740 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep1136_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3174 |
| Coherence (Success) | 0.0048 |
| Coherence (Failure) | 0.1544 |
| Gradient Magnitude (Success) | 0.0810 |
| Gradient Magnitude (Failure) | 0.1934 |
| Activation Separation | 0.7219 |
| Cosine Distance | 0.0007 |
| Clusters | 1,685 |
| Noise Fraction | 0.2716 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1407_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5241 |
| Coherence (Success) | 0.3719 |
| Coherence (Failure) | 0.1426 |
| Gradient Magnitude (Success) | 0.1885 |
| Gradient Magnitude (Failure) | 0.1417 |
| Activation Separation | 0.9513 |
| Cosine Distance | 0.0015 |
| Clusters | 1,446 |
| Noise Fraction | 0.3041 |
| RSA Alignment (ρ) | -0.4536 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1682_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0515 |
| Coherence (Success) | 0.1162 |
| Coherence (Failure) | 0.1622 |
| Gradient Magnitude (Success) | 0.1755 |
| Gradient Magnitude (Failure) | 0.1853 |
| Activation Separation | 0.9874 |
| Cosine Distance | 0.0016 |
| Clusters | 1,614 |
| Noise Fraction | 0.2607 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1942_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6765 |
| Coherence (Success) | 0.3232 |
| Coherence (Failure) | 0.1953 |
| Gradient Magnitude (Success) | 0.1616 |
| Gradient Magnitude (Failure) | 0.3111 |
| Activation Separation | 1.4238 |
| Cosine Distance | 0.0027 |
| Clusters | 1,792 |
| Noise Fraction | 0.2569 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2203_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1425 |
| Coherence (Success) | 0.1664 |
| Coherence (Failure) | 0.1719 |
| Gradient Magnitude (Success) | 0.1264 |
| Gradient Magnitude (Failure) | 0.3039 |
| Activation Separation | 1.5074 |
| Cosine Distance | 0.0026 |
| Clusters | 1,739 |
| Noise Fraction | 0.2642 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2239_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2746 |
| Coherence (Success) | 0.1340 |
| Coherence (Failure) | 0.0921 |
| Gradient Magnitude (Success) | 0.1667 |
| Gradient Magnitude (Failure) | 0.2022 |
| Activation Separation | 1.4610 |
| Cosine Distance | 0.0018 |
| Clusters | 1,610 |
| Noise Fraction | 0.2915 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2462_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3118 |
| Coherence (Success) | 0.1234 |
| Coherence (Failure) | 0.2077 |
| Gradient Magnitude (Success) | 0.1418 |
| Gradient Magnitude (Failure) | 0.2764 |
| Activation Separation | 1.1835 |
| Cosine Distance | 0.0023 |
| Clusters | 1,726 |
| Noise Fraction | 0.2931 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2722_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4254 |
| Coherence (Success) | 0.0633 |
| Coherence (Failure) | 0.1034 |
| Gradient Magnitude (Success) | 0.1104 |
| Gradient Magnitude (Failure) | 0.2074 |
| Activation Separation | 1.3647 |
| Cosine Distance | 0.0026 |
| Clusters | 1,788 |
| Noise Fraction | 0.2784 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2967_lower3.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8818 |
| Coherence (Success) | 0.2940 |
| Coherence (Failure) | 0.2626 |
| Gradient Magnitude (Success) | 0.2172 |
| Gradient Magnitude (Failure) | 0.3277 |
| Activation Separation | 1.6941 |
| Cosine Distance | 0.0053 |
| Clusters | 1,912 |
| Noise Fraction | 0.2630 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3223_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3701 |
| Coherence (Success) | 0.0978 |
| Coherence (Failure) | 0.2015 |
| Gradient Magnitude (Success) | 0.1479 |
| Gradient Magnitude (Failure) | 0.3294 |
| Activation Separation | 2.1769 |
| Cosine Distance | 0.0058 |
| Clusters | 1,903 |
| Noise Fraction | 0.2593 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3465_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3805 |
| Coherence (Success) | 0.1817 |
| Coherence (Failure) | 0.2394 |
| Gradient Magnitude (Success) | 0.1737 |
| Gradient Magnitude (Failure) | 0.3913 |
| Activation Separation | 2.1327 |
| Cosine Distance | 0.0067 |
| Clusters | 1,866 |
| Noise Fraction | 0.2807 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3716_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6588 |
| Coherence (Success) | 0.1104 |
| Coherence (Failure) | 0.1671 |
| Gradient Magnitude (Success) | 0.1641 |
| Gradient Magnitude (Failure) | 0.2317 |
| Activation Separation | 2.3627 |
| Cosine Distance | 0.0076 |
| Clusters | 1,754 |
| Noise Fraction | 0.3007 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3974_lower3.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8102 |
| Coherence (Success) | 0.4583 |
| Coherence (Failure) | 0.1519 |
| Gradient Magnitude (Success) | 0.2912 |
| Gradient Magnitude (Failure) | 0.3199 |
| Activation Separation | 3.3868 |
| Cosine Distance | 0.0128 |
| Clusters | 1,825 |
| Noise Fraction | 0.2852 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4227_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0110 |
| Coherence (Success) | 0.1635 |
| Coherence (Failure) | 0.0195 |
| Gradient Magnitude (Success) | 0.1435 |
| Gradient Magnitude (Failure) | 0.1539 |
| Activation Separation | 2.5864 |
| Cosine Distance | 0.0117 |
| Clusters | 1,517 |
| Noise Fraction | 0.3207 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4481_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5435 |
| Coherence (Success) | 0.1789 |
| Coherence (Failure) | 0.1974 |
| Gradient Magnitude (Success) | 0.1563 |
| Gradient Magnitude (Failure) | 0.2192 |
| Activation Separation | 3.0324 |
| Cosine Distance | 0.0153 |
| Clusters | 1,616 |
| Noise Fraction | 0.3076 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep4500_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6190 |
| Coherence (Success) | 0.0703 |
| Coherence (Failure) | 0.2228 |
| Gradient Magnitude (Success) | 0.1430 |
| Gradient Magnitude (Failure) | 0.2709 |
| Activation Separation | 3.2488 |
| Cosine Distance | 0.0176 |
| Clusters | 1,593 |
| Noise Fraction | 0.3032 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4744_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7829 |
| Coherence (Success) | 0.4308 |
| Coherence (Failure) | 0.4054 |
| Gradient Magnitude (Success) | 0.2953 |
| Gradient Magnitude (Failure) | 0.4174 |
| Activation Separation | 2.6386 |
| Cosine Distance | 0.0114 |
| Clusters | 1,444 |
| Noise Fraction | 0.3534 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5011_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8032 |
| Coherence (Success) | 0.3272 |
| Coherence (Failure) | 0.1826 |
| Gradient Magnitude (Success) | 0.2337 |
| Gradient Magnitude (Failure) | 0.2344 |
| Activation Separation | 3.0654 |
| Cosine Distance | 0.0158 |
| Clusters | 1,350 |
| Noise Fraction | 0.3706 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5271_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6896 |
| Coherence (Success) | 0.2356 |
| Coherence (Failure) | 0.2304 |
| Gradient Magnitude (Success) | 0.1855 |
| Gradient Magnitude (Failure) | 0.3225 |
| Activation Separation | 3.0937 |
| Cosine Distance | 0.0131 |
| Clusters | 1,532 |
| Noise Fraction | 0.3492 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5531_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3162 |
| Coherence (Success) | 0.4402 |
| Coherence (Failure) | -0.0205 |
| Gradient Magnitude (Success) | 0.3140 |
| Gradient Magnitude (Failure) | 0.1322 |
| Activation Separation | 2.9710 |
| Cosine Distance | 0.0154 |
| Clusters | 1,418 |
| Noise Fraction | 0.3467 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5775_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8454 |
| Coherence (Success) | 0.4648 |
| Coherence (Failure) | 0.2305 |
| Gradient Magnitude (Success) | 0.3675 |
| Gradient Magnitude (Failure) | 0.3324 |
| Activation Separation | 3.1769 |
| Cosine Distance | 0.0178 |
| Clusters | 1,421 |
| Noise Fraction | 0.3628 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6030_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7539 |
| Coherence (Success) | 0.2805 |
| Coherence (Failure) | 0.2576 |
| Gradient Magnitude (Success) | 0.2398 |
| Gradient Magnitude (Failure) | 0.3546 |
| Activation Separation | 3.1996 |
| Cosine Distance | 0.0171 |
| Clusters | 1,408 |
| Noise Fraction | 0.3577 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6276_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1902 |
| Coherence (Success) | 0.0955 |
| Coherence (Failure) | 0.0288 |
| Gradient Magnitude (Success) | 0.1230 |
| Gradient Magnitude (Failure) | 0.1292 |
| Activation Separation | 2.8489 |
| Cosine Distance | 0.0168 |
| Clusters | 1,442 |
| Noise Fraction | 0.3633 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6534_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5521 |
| Coherence (Success) | 0.0892 |
| Coherence (Failure) | 0.0408 |
| Gradient Magnitude (Success) | 0.1460 |
| Gradient Magnitude (Failure) | 0.1453 |
| Activation Separation | 2.1523 |
| Cosine Distance | 0.0068 |
| Clusters | 1,308 |
| Noise Fraction | 0.4022 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6782_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5239 |
| Coherence (Success) | 0.1263 |
| Coherence (Failure) | 0.1300 |
| Gradient Magnitude (Success) | 0.1803 |
| Gradient Magnitude (Failure) | 0.2429 |
| Activation Separation | 2.8625 |
| Cosine Distance | 0.0132 |
| Clusters | 1,332 |
| Noise Fraction | 0.3856 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7022_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2232 |
| Coherence (Success) | 0.1960 |
| Coherence (Failure) | -0.0347 |
| Gradient Magnitude (Success) | 0.2050 |
| Gradient Magnitude (Failure) | 0.1137 |
| Activation Separation | 3.0457 |
| Cosine Distance | 0.0115 |
| Clusters | 1,634 |
| Noise Fraction | 0.3581 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7271_lower5.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6883 |
| Coherence (Success) | 0.2423 |
| Coherence (Failure) | 0.1289 |
| Gradient Magnitude (Success) | 0.2255 |
| Gradient Magnitude (Failure) | 0.3218 |
| Activation Separation | 2.9328 |
| Cosine Distance | 0.0095 |
| Clusters | 1,546 |
| Noise Fraction | 0.3489 |
| RSA Alignment (ρ) | 0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7501_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4050 |
| Coherence (Success) | 0.1617 |
| Coherence (Failure) | 0.0386 |
| Gradient Magnitude (Success) | 0.1630 |
| Gradient Magnitude (Failure) | 0.1879 |
| Activation Separation | 3.7414 |
| Cosine Distance | 0.0196 |
| Clusters | 1,463 |
| Noise Fraction | 0.3730 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7726_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1910 |
| Coherence (Success) | 0.0044 |
| Coherence (Failure) | 0.0271 |
| Gradient Magnitude (Success) | 0.1164 |
| Gradient Magnitude (Failure) | 0.1473 |
| Activation Separation | 3.4756 |
| Cosine Distance | 0.0183 |
| Clusters | 1,517 |
| Noise Fraction | 0.3350 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7949_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3950 |
| Coherence (Success) | 0.0062 |
| Coherence (Failure) | 0.0784 |
| Gradient Magnitude (Success) | 0.0935 |
| Gradient Magnitude (Failure) | 0.2192 |
| Activation Separation | 3.4846 |
| Cosine Distance | 0.0164 |
| Clusters | 1,566 |
| Noise Fraction | 0.3502 |
| RSA Alignment (ρ) | 0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8168_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6576 |
| Coherence (Success) | 0.3460 |
| Coherence (Failure) | 0.2266 |
| Gradient Magnitude (Success) | 0.2921 |
| Gradient Magnitude (Failure) | 0.3228 |
| Activation Separation | 3.9383 |
| Cosine Distance | 0.0210 |
| Clusters | 1,584 |
| Noise Fraction | 0.3415 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8387_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5040 |
| Coherence (Success) | 0.1355 |
| Coherence (Failure) | 0.1097 |
| Gradient Magnitude (Success) | 0.1938 |
| Gradient Magnitude (Failure) | 0.2216 |
| Activation Separation | 4.5266 |
| Cosine Distance | 0.0308 |
| Clusters | 1,346 |
| Noise Fraction | 0.3500 |
| RSA Alignment (ρ) | 0.0391 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8616_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5320 |
| Coherence (Success) | 0.2843 |
| Coherence (Failure) | 0.1735 |
| Gradient Magnitude (Success) | 0.2637 |
| Gradient Magnitude (Failure) | 0.3444 |
| Activation Separation | 4.5275 |
| Cosine Distance | 0.0247 |
| Clusters | 1,412 |
| Noise Fraction | 0.3704 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8837_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3433 |
| Coherence (Success) | 0.0748 |
| Coherence (Failure) | -0.0171 |
| Gradient Magnitude (Success) | 0.1626 |
| Gradient Magnitude (Failure) | 0.1381 |
| Activation Separation | 3.8065 |
| Cosine Distance | 0.0179 |
| Clusters | 1,596 |
| Noise Fraction | 0.3424 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9069_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4459 |
| Coherence (Success) | 0.1751 |
| Coherence (Failure) | 0.0947 |
| Gradient Magnitude (Success) | 0.2013 |
| Gradient Magnitude (Failure) | 0.2594 |
| Activation Separation | 3.5217 |
| Cosine Distance | 0.0180 |
| Clusters | 1,444 |
| Noise Fraction | 0.3826 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9287_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3620 |
| Coherence (Success) | 0.2040 |
| Coherence (Failure) | 0.0683 |
| Gradient Magnitude (Success) | 0.1881 |
| Gradient Magnitude (Failure) | 0.2174 |
| Activation Separation | 4.6580 |
| Cosine Distance | 0.0369 |
| Clusters | 1,339 |
| Noise Fraction | 0.3461 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9503_lower6.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4913 |
| Coherence (Success) | 0.1886 |
| Coherence (Failure) | 0.2023 |
| Gradient Magnitude (Success) | 0.1890 |
| Gradient Magnitude (Failure) | 0.3195 |
| Activation Separation | 4.7327 |
| Cosine Distance | 0.0377 |
| Clusters | 1,474 |
| Noise Fraction | 0.3664 |
| RSA Alignment (ρ) | 0.1272 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9719_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6941 |
| Coherence (Success) | 0.2347 |
| Coherence (Failure) | 0.1550 |
| Gradient Magnitude (Success) | 0.2446 |
| Gradient Magnitude (Failure) | 0.3434 |
| Activation Separation | 3.6153 |
| Cosine Distance | 0.0181 |
| Clusters | 1,248 |
| Noise Fraction | 0.3969 |
| RSA Alignment (ρ) | 0.0783 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9936_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1911 |
| Coherence (Success) | 0.2306 |
| Coherence (Failure) | 0.0981 |
| Gradient Magnitude (Success) | 0.2175 |
| Gradient Magnitude (Failure) | 0.2019 |
| Activation Separation | 3.6957 |
| Cosine Distance | 0.0198 |
| Clusters | 1,454 |
| Noise Fraction | 0.3839 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10156_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7703 |
| Coherence (Success) | 0.3852 |
| Coherence (Failure) | 0.1098 |
| Gradient Magnitude (Success) | 0.2892 |
| Gradient Magnitude (Failure) | 0.2732 |
| Activation Separation | 4.4300 |
| Cosine Distance | 0.0277 |
| Clusters | 1,348 |
| Noise Fraction | 0.3680 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10376_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6327 |
| Coherence (Success) | 0.1007 |
| Coherence (Failure) | 0.0212 |
| Gradient Magnitude (Success) | 0.1805 |
| Gradient Magnitude (Failure) | 0.2018 |
| Activation Separation | 4.1301 |
| Cosine Distance | 0.0187 |
| Clusters | 1,490 |
| Noise Fraction | 0.3731 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10600_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6010 |
| Coherence (Success) | 0.2467 |
| Coherence (Failure) | 0.0761 |
| Gradient Magnitude (Success) | 0.2100 |
| Gradient Magnitude (Failure) | 0.2027 |
| Activation Separation | 3.8551 |
| Cosine Distance | 0.0193 |
| Clusters | 1,318 |
| Noise Fraction | 0.3650 |
| RSA Alignment (ρ) | 0.0391 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10814_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5278 |
| Coherence (Success) | 0.3100 |
| Coherence (Failure) | 0.0408 |
| Gradient Magnitude (Success) | 0.2447 |
| Gradient Magnitude (Failure) | 0.1754 |
| Activation Separation | 4.1254 |
| Cosine Distance | 0.0322 |
| Clusters | 1,436 |
| Noise Fraction | 0.3864 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11017_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6940 |
| Coherence (Success) | 0.3485 |
| Coherence (Failure) | 0.0759 |
| Gradient Magnitude (Success) | 0.2548 |
| Gradient Magnitude (Failure) | 0.2221 |
| Activation Separation | 3.4802 |
| Cosine Distance | 0.0165 |
| Clusters | 1,487 |
| Noise Fraction | 0.3785 |
| RSA Alignment (ρ) | -0.0979 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11228_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6423 |
| Coherence (Success) | 0.3350 |
| Coherence (Failure) | 0.1583 |
| Gradient Magnitude (Success) | 0.2540 |
| Gradient Magnitude (Failure) | 0.3624 |
| Activation Separation | 3.9974 |
| Cosine Distance | 0.0210 |
| Clusters | 1,387 |
| Noise Fraction | 0.3656 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11441_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7664 |
| Coherence (Success) | 0.3380 |
| Coherence (Failure) | 0.1891 |
| Gradient Magnitude (Success) | 0.2771 |
| Gradient Magnitude (Failure) | 0.3236 |
| Activation Separation | 3.0522 |
| Cosine Distance | 0.0144 |
| Clusters | 1,249 |
| Noise Fraction | 0.3946 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11672_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4282 |
| Coherence (Success) | 0.0856 |
| Coherence (Failure) | 0.1246 |
| Gradient Magnitude (Success) | 0.1431 |
| Gradient Magnitude (Failure) | 0.2461 |
| Activation Separation | 3.6775 |
| Cosine Distance | 0.0178 |
| Clusters | 1,481 |
| Noise Fraction | 0.4020 |
| RSA Alignment (ρ) | 0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep11871_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5993 |
| Coherence (Success) | 0.1725 |
| Coherence (Failure) | 0.1488 |
| Gradient Magnitude (Success) | 0.1874 |
| Gradient Magnitude (Failure) | 0.2647 |
| Activation Separation | 4.2518 |
| Cosine Distance | 0.0327 |
| Clusters | 1,495 |
| Noise Fraction | 0.3654 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11894_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5396 |
| Coherence (Success) | 0.0951 |
| Coherence (Failure) | 0.0550 |
| Gradient Magnitude (Success) | 0.1575 |
| Gradient Magnitude (Failure) | 0.1987 |
| Activation Separation | 4.2626 |
| Cosine Distance | 0.0302 |
| Clusters | 1,561 |
| Noise Fraction | 0.3715 |
| RSA Alignment (ρ) | 0.1860 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12108_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6134 |
| Coherence (Success) | 0.1974 |
| Coherence (Failure) | 0.0550 |
| Gradient Magnitude (Success) | 0.2185 |
| Gradient Magnitude (Failure) | 0.1885 |
| Activation Separation | 2.8471 |
| Cosine Distance | 0.0146 |
| Clusters | 1,441 |
| Noise Fraction | 0.3982 |
| RSA Alignment (ρ) | 0.0196 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12311_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5234 |
| Coherence (Success) | 0.1668 |
| Coherence (Failure) | 0.1274 |
| Gradient Magnitude (Success) | 0.1857 |
| Gradient Magnitude (Failure) | 0.2397 |
| Activation Separation | 3.8372 |
| Cosine Distance | 0.0268 |
| Clusters | 1,490 |
| Noise Fraction | 0.3778 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12521_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4529 |
| Coherence (Success) | 0.3092 |
| Coherence (Failure) | 0.1107 |
| Gradient Magnitude (Success) | 0.2565 |
| Gradient Magnitude (Failure) | 0.2603 |
| Activation Separation | 4.3720 |
| Cosine Distance | 0.0269 |
| Clusters | 1,424 |
| Noise Fraction | 0.3721 |
| RSA Alignment (ρ) | 0.2055 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12733_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4848 |
| Coherence (Success) | 0.1278 |
| Coherence (Failure) | 0.0473 |
| Gradient Magnitude (Success) | 0.1683 |
| Gradient Magnitude (Failure) | 0.1963 |
| Activation Separation | 4.1121 |
| Cosine Distance | 0.0267 |
| Clusters | 1,464 |
| Noise Fraction | 0.3805 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12939_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1755 |
| Coherence (Success) | 0.3287 |
| Coherence (Failure) | 0.0688 |
| Gradient Magnitude (Success) | 0.2433 |
| Gradient Magnitude (Failure) | 0.1910 |
| Activation Separation | 3.2650 |
| Cosine Distance | 0.0205 |
| Clusters | 1,449 |
| Noise Fraction | 0.3510 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13153_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4202 |
| Coherence (Success) | 0.2731 |
| Coherence (Failure) | 0.0240 |
| Gradient Magnitude (Success) | 0.2461 |
| Gradient Magnitude (Failure) | 0.1686 |
| Activation Separation | 4.3009 |
| Cosine Distance | 0.0319 |
| Clusters | 1,471 |
| Noise Fraction | 0.3526 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13354_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5869 |
| Coherence (Success) | 0.2507 |
| Coherence (Failure) | 0.1531 |
| Gradient Magnitude (Success) | 0.2488 |
| Gradient Magnitude (Failure) | 0.3437 |
| Activation Separation | 4.8023 |
| Cosine Distance | 0.0299 |
| Clusters | 1,420 |
| Noise Fraction | 0.3224 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13562_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5104 |
| Coherence (Success) | 0.1998 |
| Coherence (Failure) | 0.0757 |
| Gradient Magnitude (Success) | 0.2437 |
| Gradient Magnitude (Failure) | 0.2357 |
| Activation Separation | 3.7459 |
| Cosine Distance | 0.0202 |
| Clusters | 1,451 |
| Noise Fraction | 0.3133 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13770_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2798 |
| Coherence (Success) | 0.2212 |
| Coherence (Failure) | 0.0259 |
| Gradient Magnitude (Success) | 0.2097 |
| Gradient Magnitude (Failure) | 0.1611 |
| Activation Separation | 4.2740 |
| Cosine Distance | 0.0303 |
| Clusters | 1,513 |
| Noise Fraction | 0.3145 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13984_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7564 |
| Coherence (Success) | 0.1412 |
| Coherence (Failure) | 0.0740 |
| Gradient Magnitude (Success) | 0.2024 |
| Gradient Magnitude (Failure) | 0.2656 |
| Activation Separation | 4.1299 |
| Cosine Distance | 0.0283 |
| Clusters | 1,368 |
| Noise Fraction | 0.3448 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14206_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2496 |
| Coherence (Success) | 0.0970 |
| Coherence (Failure) | 0.0810 |
| Gradient Magnitude (Success) | 0.1699 |
| Gradient Magnitude (Failure) | 0.2267 |
| Activation Separation | 4.1801 |
| Cosine Distance | 0.0310 |
| Clusters | 1,521 |
| Noise Fraction | 0.3485 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14207_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2959 |
| Coherence (Success) | 0.1679 |
| Coherence (Failure) | 0.0680 |
| Gradient Magnitude (Success) | 0.2031 |
| Gradient Magnitude (Failure) | 0.2054 |
| Activation Separation | 4.8714 |
| Cosine Distance | 0.0412 |
| Clusters | 1,545 |
| Noise Fraction | 0.3476 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_sapling @ ep1 | 1 | 0.4216 | 0.7357 | -0.0019 | 0.0611 | 0.0184 | 0.1170 | 0.0004 | nan |
| wake_up @ ep2 | 2 | 0.3386 | 0.8784 | 0.0399 | 0.0836 | 0.0361 | 0.1215 | 0.0005 | nan |
| place_plant @ ep3 | 3 | -0.0465 | 0.8908 | 0.0662 | 0.0862 | 0.0323 | 0.1690 | 0.0010 | -0.3928 |
| collect_wood @ ep10 | 10 | 0.2345 | 0.9056 | -0.0158 | 0.0851 | 0.0359 | 0.1865 | 0.0012 | -0.6547 |
| place_table @ ep38 | 38 | 0.3987 | 0.9226 | 0.4167 | 0.1964 | 0.0906 | 0.3518 | 0.0010 | -0.6547 |
| collect_drink @ ep40 | 40 | 0.2523 | 0.9212 | 0.0092 | 0.1949 | 0.0816 | 0.3750 | 0.0010 | nan |
| make_wood_pickaxe @ ep58 | 58 | 0.8139 | 0.8469 | 0.4860 | 0.2888 | 0.1569 | 0.5289 | 0.0002 | -0.3482 |
| make_wood_sword @ ep63 | 63 | 0.6543 | 0.9200 | -0.0174 | 0.3055 | 0.1266 | 0.4471 | 0.0002 | -0.3928 |
| eat_cow @ ep77 | 77 | 0.7703 | 0.8977 | 0.3913 | 0.2481 | 0.1360 | 0.5027 | 0.0003 | -0.1741 |
| defeat_zombie @ ep97 | 97 | 0.8603 | 0.9262 | 0.6708 | 0.3492 | 0.1935 | 0.5279 | 0.0002 | -0.2791 |
| defeat_skeleton @ ep149 | 149 | 0.7919 | 0.7289 | 0.5031 | 0.2867 | 0.2594 | 0.5528 | 0.0002 | -0.1662 |
| eat_plant @ ep187 | 187 | 0.4691 | 0.4923 | 0.1067 | 0.1995 | 0.2011 | 0.6865 | 0.0002 | -0.3482 |
| collect_stone @ ep230 | 230 | -0.0442 | 0.1333 | 0.0992 | 0.1598 | 0.1234 | 1.5420 | 0.0006 | -0.1745 |
| make_stone_sword @ ep230 | 230 | 0.6929 | 0.6002 | 0.1491 | 0.2098 | 0.1361 | 1.4299 | 0.0006 | -0.6093 |
| collect_coal @ ep372 | 372 | -0.1821 | 0.1648 | 0.2245 | 0.1456 | 0.1516 | 0.8727 | 0.0005 | -0.1745 |
| place_stone @ ep1136 | 1,136 | -0.3174 | 0.0048 | 0.1544 | 0.0810 | 0.1934 | 0.7219 | 0.0007 | -0.1047 |
| place_furnace @ ep2239 | 2,239 | -0.2746 | 0.1340 | 0.0921 | 0.1667 | 0.2022 | 1.4610 | 0.0018 | -0.2443 |
| make_stone_pickaxe @ ep4500 | 4,500 | 0.6190 | 0.0703 | 0.2228 | 0.1430 | 0.2709 | 3.2488 | 0.0176 | -0.0349 |
| collect_iron @ ep11871 | 11,871 | 0.5993 | 0.1725 | 0.1488 | 0.1874 | 0.2647 | 4.2518 | 0.0327 | 0.0739 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic_step50000_ep273 | 273 | 0.5798 | 0.7141 | 0.6616 | 0.2132 | 0.2064 | 1.0396 | 0.0006 | -0.3140 |
| periodic_step100000_ep555 | 555 | -0.3212 | -0.0088 | -0.0125 | 0.0977 | 0.1531 | 1.1580 | 0.0005 | -0.1741 |
| periodic_step150000_ep842 | 842 | 0.2191 | 0.3429 | 0.1903 | 0.1855 | 0.1847 | 0.5917 | 0.0006 | 0.0000 |
| periodic_step200000_ep1124 | 1,124 | 0.4007 | 0.0529 | 0.3027 | 0.1043 | 0.2267 | 0.8247 | 0.0011 | -0.2443 |
| periodic_step250000_ep1407 | 1,407 | 0.5241 | 0.3719 | 0.1426 | 0.1885 | 0.1417 | 0.9513 | 0.0015 | -0.4536 |
| periodic_step300000_ep1682 | 1,682 | 0.0515 | 0.1162 | 0.1622 | 0.1755 | 0.1853 | 0.9874 | 0.0016 | -0.2094 |
| periodic_step350000_ep1942 | 1,942 | 0.6765 | 0.3232 | 0.1953 | 0.1616 | 0.3111 | 1.4238 | 0.0027 | -0.3140 |
| periodic_step400000_ep2203 | 2,203 | -0.1425 | 0.1664 | 0.1719 | 0.1264 | 0.3039 | 1.5074 | 0.0026 | -0.2791 |
| periodic_step450000_ep2462 | 2,462 | 0.3118 | 0.1234 | 0.2077 | 0.1418 | 0.2764 | 1.1835 | 0.0023 | -0.2443 |
| periodic_step500000_ep2722 | 2,722 | -0.4254 | 0.0633 | 0.1034 | 0.1104 | 0.2074 | 1.3647 | 0.0026 | -0.1292 |
| periodic_step550000_ep2967 | 2,967 | 0.8818 | 0.2940 | 0.2626 | 0.2172 | 0.3277 | 1.6941 | 0.0053 | -0.1047 |
| periodic_step600000_ep3223 | 3,223 | 0.3701 | 0.0978 | 0.2015 | 0.1479 | 0.3294 | 2.1769 | 0.0058 | -0.2443 |
| periodic_step650000_ep3465 | 3,465 | 0.3805 | 0.1817 | 0.2394 | 0.1737 | 0.3913 | 2.1327 | 0.0067 | -0.0739 |
| periodic_step700000_ep3716 | 3,716 | 0.6588 | 0.1104 | 0.1671 | 0.1641 | 0.2317 | 2.3627 | 0.0076 | -0.1047 |
| periodic_step750000_ep3974 | 3,974 | 0.8102 | 0.4583 | 0.1519 | 0.2912 | 0.3199 | 3.3868 | 0.0128 | 0.1396 |
| periodic_step800000_ep4227 | 4,227 | -0.0110 | 0.1635 | 0.0195 | 0.1435 | 0.1539 | 2.5864 | 0.0117 | -0.1108 |
| periodic_step850000_ep4481 | 4,481 | 0.5435 | 0.1789 | 0.1974 | 0.1563 | 0.2192 | 3.0324 | 0.0153 | 0.0000 |
| periodic_step900000_ep4744 | 4,744 | 0.7829 | 0.4308 | 0.4054 | 0.2953 | 0.4174 | 2.6386 | 0.0114 | 0.0000 |
| periodic_step950000_ep5011 | 5,011 | 0.8032 | 0.3272 | 0.1826 | 0.2337 | 0.2344 | 3.0654 | 0.0158 | -0.0739 |
| periodic_step1000000_ep5271 | 5,271 | 0.6896 | 0.2356 | 0.2304 | 0.1855 | 0.3225 | 3.0937 | 0.0131 | 0.0000 |
| periodic_step1050000_ep5531 | 5,531 | 0.3162 | 0.4402 | -0.0205 | 0.3140 | 0.1322 | 2.9710 | 0.0154 | 0.0185 |
| periodic_step1100000_ep5775 | 5,775 | 0.8454 | 0.4648 | 0.2305 | 0.3675 | 0.3324 | 3.1769 | 0.0178 | -0.0369 |
| periodic_step1150000_ep6030 | 6,030 | 0.7539 | 0.2805 | 0.2576 | 0.2398 | 0.3546 | 3.1996 | 0.0171 | -0.1108 |
| periodic_step1200000_ep6276 | 6,276 | 0.1902 | 0.0955 | 0.0288 | 0.1230 | 0.1292 | 2.8489 | 0.0168 | -0.0923 |
| periodic_step1250000_ep6534 | 6,534 | 0.5521 | 0.0892 | 0.0408 | 0.1460 | 0.1453 | 2.1523 | 0.0068 | 0.1396 |
| periodic_step1300000_ep6782 | 6,782 | 0.5239 | 0.1263 | 0.1300 | 0.1803 | 0.2429 | 2.8625 | 0.0132 | 0.1292 |
| periodic_step1350000_ep7022 | 7,022 | 0.2232 | 0.1960 | -0.0347 | 0.2050 | 0.1137 | 3.0457 | 0.0115 | 0.2443 |
| periodic_step1400000_ep7271 | 7,271 | 0.6883 | 0.2423 | 0.1289 | 0.2255 | 0.3218 | 2.9328 | 0.0095 | 0.1047 |
| periodic_step1450000_ep7501 | 7,501 | 0.4050 | 0.1617 | 0.0386 | 0.1630 | 0.1879 | 3.7414 | 0.0196 | 0.1396 |
| periodic_step1500000_ep7726 | 7,726 | 0.1910 | 0.0044 | 0.0271 | 0.1164 | 0.1473 | 3.4756 | 0.0183 | 0.1745 |
| periodic_step1550000_ep7949 | 7,949 | 0.3950 | 0.0062 | 0.0784 | 0.0935 | 0.2192 | 3.4846 | 0.0164 | 0.1047 |
| periodic_step1600000_ep8168 | 8,168 | 0.6576 | 0.3460 | 0.2266 | 0.2921 | 0.3228 | 3.9383 | 0.0210 | 0.0185 |
| periodic_step1650000_ep8387 | 8,387 | 0.5040 | 0.1355 | 0.1097 | 0.1938 | 0.2216 | 4.5266 | 0.0308 | 0.0391 |
| periodic_step1700000_ep8616 | 8,616 | 0.5320 | 0.2843 | 0.1735 | 0.2637 | 0.3444 | 4.5275 | 0.0247 | -0.0554 |
| periodic_step1750000_ep8837 | 8,837 | 0.3433 | 0.0748 | -0.0171 | 0.1626 | 0.1381 | 3.8065 | 0.0179 | 0.0739 |
| periodic_step1800000_ep9069 | 9,069 | 0.4459 | 0.1751 | 0.0947 | 0.2013 | 0.2594 | 3.5217 | 0.0180 | 0.0185 |
| periodic_step1850000_ep9287 | 9,287 | 0.3620 | 0.2040 | 0.0683 | 0.1881 | 0.2174 | 4.6580 | 0.0369 | 0.0739 |
| periodic_step1900000_ep9503 | 9,503 | 0.4913 | 0.1886 | 0.2023 | 0.1890 | 0.3195 | 4.7327 | 0.0377 | 0.1272 |
| periodic_step1950000_ep9719 | 9,719 | 0.6941 | 0.2347 | 0.1550 | 0.2446 | 0.3434 | 3.6153 | 0.0181 | 0.0783 |
| periodic_step2000000_ep9936 | 9,936 | 0.1911 | 0.2306 | 0.0981 | 0.2175 | 0.2019 | 3.6957 | 0.0198 | 0.0000 |
| periodic_step2050000_ep10156 | 10,156 | 0.7703 | 0.3852 | 0.1098 | 0.2892 | 0.2732 | 4.4300 | 0.0277 | 0.1292 |
| periodic_step2100000_ep10376 | 10,376 | 0.6327 | 0.1007 | 0.0212 | 0.1805 | 0.2018 | 4.1301 | 0.0187 | 0.1292 |
| periodic_step2150000_ep10600 | 10,600 | 0.6010 | 0.2467 | 0.0761 | 0.2100 | 0.2027 | 3.8551 | 0.0193 | 0.0391 |
| periodic_step2200000_ep10814 | 10,814 | 0.5278 | 0.3100 | 0.0408 | 0.2447 | 0.1754 | 4.1254 | 0.0322 | 0.0587 |
| periodic_step2250000_ep11017 | 11,017 | 0.6940 | 0.3485 | 0.0759 | 0.2548 | 0.2221 | 3.4802 | 0.0165 | -0.0979 |
| periodic_step2300000_ep11228 | 11,228 | 0.6423 | 0.3350 | 0.1583 | 0.2540 | 0.3624 | 3.9974 | 0.0210 | 0.0923 |
| periodic_step2350000_ep11441 | 11,441 | 0.7664 | 0.3380 | 0.1891 | 0.2771 | 0.3236 | 3.0522 | 0.0144 | 0.2031 |
| periodic_step2400000_ep11672 | 11,672 | 0.4282 | 0.0856 | 0.1246 | 0.1431 | 0.2461 | 3.6775 | 0.0178 | 0.1468 |
| periodic_step2450000_ep11894 | 11,894 | 0.5396 | 0.0951 | 0.0550 | 0.1575 | 0.1987 | 4.2626 | 0.0302 | 0.1860 |
| periodic_step2500000_ep12108 | 12,108 | 0.6134 | 0.1974 | 0.0550 | 0.2185 | 0.1885 | 2.8471 | 0.0146 | 0.0196 |
| periodic_step2550000_ep12311 | 12,311 | 0.5234 | 0.1668 | 0.1274 | 0.1857 | 0.2397 | 3.8372 | 0.0268 | 0.0587 |
| periodic_step2600000_ep12521 | 12,521 | 0.4529 | 0.3092 | 0.1107 | 0.2565 | 0.2603 | 4.3720 | 0.0269 | 0.2055 |
| periodic_step2650000_ep12733 | 12,733 | 0.4848 | 0.1278 | 0.0473 | 0.1683 | 0.1963 | 4.1121 | 0.0267 | 0.0739 |
| periodic_step2700000_ep12939 | 12,939 | 0.1755 | 0.3287 | 0.0688 | 0.2433 | 0.1910 | 3.2650 | 0.0205 | 0.0000 |
| periodic_step2750000_ep13153 | 13,153 | 0.4202 | 0.2731 | 0.0240 | 0.2461 | 0.1686 | 4.3009 | 0.0319 | 0.0369 |
| periodic_step2800000_ep13354 | 13,354 | 0.5869 | 0.2507 | 0.1531 | 0.2488 | 0.3437 | 4.8023 | 0.0299 | 0.0000 |
| periodic_step2850000_ep13562 | 13,562 | 0.5104 | 0.1998 | 0.0757 | 0.2437 | 0.2357 | 3.7459 | 0.0202 | 0.1108 |
| periodic_step2900000_ep13770 | 13,770 | 0.2798 | 0.2212 | 0.0259 | 0.2097 | 0.1611 | 4.2740 | 0.0303 | 0.0369 |
| periodic_step2950000_ep13984 | 13,984 | 0.7564 | 0.1412 | 0.0740 | 0.2024 | 0.2656 | 4.1299 | 0.0283 | 0.0923 |
| periodic_step3000000_ep14206 | 14,206 | 0.2496 | 0.0970 | 0.0810 | 0.1699 | 0.2267 | 4.1801 | 0.0310 | 0.2031 |
| final_step3000320_ep14207 | 14,207 | 0.2959 | 0.1679 | 0.0680 | 0.2031 | 0.2054 | 4.8714 | 0.0412 | 0.2216 |
