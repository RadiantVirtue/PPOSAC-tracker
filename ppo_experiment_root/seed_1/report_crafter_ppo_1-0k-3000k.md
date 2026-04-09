# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 1  
**Total episodes:** 15,343  
**Experiment root:** `ppo_experiment_root\seed_1`  
**Generated:** 2026-04-01 00:33

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep2 | 2 | 0.0624 | 0.3394 | 0.1928 | 0.0929 | 0.0344 | 0.1352 | 0.0005 | nan |
| wake_up @ ep2 | 2 | 0.1040 | 0.3315 | 0.1525 | 0.0910 | 0.0365 | 0.1190 | 0.0004 | — |
| collect_drink @ ep3 | 3 | -0.0126 | 0.3954 | 0.1264 | 0.0828 | 0.0341 | 0.1318 | 0.0005 | nan |
| place_table @ ep5 | 5 | -0.1738 | 0.3320 | 0.3555 | 0.0835 | 0.0363 | 0.1326 | 0.0005 | nan |
| collect_sapling @ ep10 | 10 | 0.0504 | 0.2699 | 0.2862 | 0.0813 | 0.0355 | 0.1418 | 0.0005 | — |
| place_plant @ ep10 | 10 | 0.3617 | 0.2866 | 0.2073 | 0.0808 | 0.0408 | 0.1317 | 0.0005 | — |
| eat_cow @ ep50 | 50 | 0.6053 | 0.8578 | 0.3081 | 0.1914 | 0.0799 | 0.2884 | 0.0005 | 0.1309 |
| make_wood_sword @ ep55 | 55 | 0.4523 | 0.9257 | 0.7070 | 0.2091 | 0.1047 | 0.3242 | 0.0005 | -0.3928 |
| defeat_zombie @ ep83 | 83 | 0.8639 | 0.9316 | 0.2742 | 0.3459 | 0.1655 | 0.4379 | 0.0001 | -0.4352 |
| make_wood_pickaxe @ ep114 | 114 | 0.7621 | 0.8947 | 0.0840 | 0.2930 | 0.1976 | 0.3724 | 0.0001 | -0.6093 |
| collect_stone @ ep125 | 125 | 0.8134 | 0.7710 | 0.4167 | 0.3007 | 0.2002 | 0.5255 | 0.0001 | -0.2094 |
| defeat_skeleton @ ep208 | 208 | 0.0490 | -0.0242 | 0.0990 | 0.0921 | 0.2036 | 1.2855 | 0.0002 | -0.3489 |
| Step 50,000 | 280 | -0.2196 | 0.1228 | 0.1229 | 0.1497 | 0.2348 | 0.9005 | 0.0002 | -0.1047 |
| collect_coal @ ep337 | 337 | 0.1742 | 0.2763 | 0.4460 | 0.1228 | 0.2190 | 0.7789 | 0.0002 | -0.2791 |
| place_stone @ ep480 | 480 | 0.7402 | 0.4421 | 0.2267 | 0.1833 | 0.1967 | 0.5958 | 0.0004 | -0.2611 |
| Step 100,000 | 569 | 0.2167 | 0.0852 | 0.1471 | 0.1026 | 0.1283 | 0.7116 | 0.0007 | -0.5222 |
| Step 150,000 | 854 | 0.2693 | 0.4447 | 0.2924 | 0.1706 | 0.1969 | 0.7448 | 0.0004 | -0.3489 |
| Step 200,000 | 1,125 | -0.2890 | 0.2722 | 0.0491 | 0.1194 | 0.1070 | 0.7027 | 0.0005 | -0.3482 |
| Step 250,000 | 1,407 | -0.0683 | 0.1637 | 0.1796 | 0.0996 | 0.1532 | 0.6588 | 0.0011 | -0.5222 |
| Step 300,000 | 1,684 | -0.7424 | 0.2677 | 0.3167 | 0.1019 | 0.1732 | 0.6607 | 0.0018 | -0.1745 |
| place_furnace @ ep1781 | 1,781 | 0.0486 | 0.1696 | 0.3123 | 0.0959 | 0.1570 | 0.7958 | 0.0026 | -0.0698 |
| Step 350,000 | 1,948 | -0.4100 | 0.1530 | 0.1028 | 0.0970 | 0.1195 | 0.8618 | 0.0037 | -0.0698 |
| Step 400,000 | 2,229 | -0.4373 | 0.1294 | 0.3975 | 0.0879 | 0.2113 | 0.9325 | 0.0065 | -0.2094 |
| Step 450,000 | 2,504 | -0.2500 | 0.3296 | 0.3032 | 0.1432 | 0.1701 | 1.5241 | 0.0130 | -0.1292 |
| Step 500,000 | 2,777 | 0.0459 | 0.4196 | 0.1329 | 0.1845 | 0.1593 | 1.6503 | 0.0081 | -0.2791 |
| Step 550,000 | 3,050 | 0.0972 | 0.3543 | 0.2381 | 0.1783 | 0.2264 | 2.0617 | 0.0110 | -0.2094 |
| Step 600,000 | 3,321 | 0.4959 | 0.1404 | 0.1726 | 0.1231 | 0.2046 | 2.2040 | 0.0121 | -0.0739 |
| Step 650,000 | 3,592 | 0.0394 | 0.1423 | 0.0573 | 0.1421 | 0.1397 | 2.5916 | 0.0118 | -0.1292 |
| Step 700,000 | 3,853 | 0.3683 | 0.0375 | 0.2032 | 0.1348 | 0.2103 | 2.8900 | 0.0120 | -0.0349 |
| make_stone_sword @ ep4113 | 4,113 | 0.3869 | 0.2586 | 0.1872 | 0.1800 | 0.2452 | 3.1901 | 0.0143 | -0.0698 |
| Step 750,000 | 4,124 | 0.6584 | 0.2352 | 0.0890 | 0.2016 | 0.1907 | 2.6577 | 0.0107 | -0.0698 |
| Step 800,000 | 4,379 | 0.5489 | 0.1850 | 0.0492 | 0.1777 | 0.1511 | 3.0503 | 0.0160 | -0.0349 |
| Step 850,000 | 4,652 | 0.3158 | 0.1156 | 0.0522 | 0.1551 | 0.1659 | 3.3906 | 0.0190 | -0.1662 |
| make_stone_pickaxe @ ep4919 | 4,919 | 0.3184 | 0.1492 | 0.1243 | 0.1625 | 0.1926 | 3.8244 | 0.0243 | 0.0000 |
| Step 900,000 | 4,925 | 0.3241 | 0.0985 | 0.0669 | 0.1324 | 0.1547 | 3.6890 | 0.0242 | -0.0185 |
| Step 950,000 | 5,187 | -0.1858 | 0.0841 | 0.0060 | 0.1224 | 0.1234 | 4.5680 | 0.0282 | -0.1047 |
| Step 1,000,000 | 5,447 | 0.4896 | 0.1724 | 0.0668 | 0.1611 | 0.2032 | 3.6312 | 0.0205 | -0.0349 |
| Step 1,050,000 | 5,703 | 0.5672 | 0.1675 | 0.1688 | 0.2076 | 0.2624 | 4.4506 | 0.0233 | 0.0000 |
| Step 1,100,000 | 5,977 | 0.1536 | 0.2865 | -0.0086 | 0.2347 | 0.1464 | 3.9591 | 0.0173 | 0.0349 |
| Step 1,150,000 | 6,250 | 0.7399 | 0.1862 | 0.0635 | 0.1974 | 0.1884 | 3.8028 | 0.0291 | 0.1396 |
| Step 1,200,000 | 6,523 | -0.0035 | 0.0729 | 0.0624 | 0.1408 | 0.1759 | 4.2199 | 0.0305 | 0.0349 |
| Step 1,250,000 | 6,782 | 0.5825 | 0.1106 | 0.1826 | 0.1467 | 0.2851 | 4.8867 | 0.0318 | 0.0000 |
| Step 1,300,000 | 7,045 | 0.6133 | 0.1690 | 0.0670 | 0.1730 | 0.2184 | 4.3985 | 0.0305 | 0.0739 |
| Step 1,350,000 | 7,306 | 0.5680 | 0.2749 | 0.1702 | 0.2231 | 0.2353 | 4.2895 | 0.0234 | 0.0369 |
| Step 1,400,000 | 7,562 | 0.4695 | 0.2328 | 0.0763 | 0.2432 | 0.2054 | 5.3444 | 0.0306 | -0.0739 |
| Step 1,450,000 | 7,807 | -0.3296 | 0.0104 | -0.0116 | 0.1066 | 0.1209 | 5.1402 | 0.0450 | 0.0185 |
| Step 1,500,000 | 8,065 | 0.6255 | 0.0886 | 0.1033 | 0.1542 | 0.1991 | 5.1954 | 0.0395 | 0.1047 |
| Step 1,550,000 | 8,328 | 0.7256 | 0.2290 | 0.2448 | 0.2273 | 0.3001 | 4.3364 | 0.0270 | -0.1108 |
| Step 1,600,000 | 8,581 | 0.6326 | 0.3949 | 0.1295 | 0.2970 | 0.2331 | 3.9010 | 0.0233 | 0.0554 |
| Step 1,650,000 | 8,844 | 0.6192 | 0.2182 | 0.0803 | 0.2061 | 0.1925 | 4.8971 | 0.0400 | 0.2443 |
| Step 1,700,000 | 9,103 | 0.7183 | 0.2638 | 0.2405 | 0.2420 | 0.2989 | 4.0125 | 0.0238 | 0.1292 |
| Step 1,750,000 | 9,363 | 0.3926 | 0.1230 | 0.0776 | 0.2043 | 0.2107 | 5.2068 | 0.0350 | 0.1477 |
| Step 1,800,000 | 9,613 | 0.6008 | 0.2867 | 0.1488 | 0.2543 | 0.2716 | 4.3421 | 0.0255 | 0.3140 |
| Step 1,850,000 | 9,864 | 0.2716 | 0.2313 | 0.0062 | 0.1965 | 0.1468 | 4.2221 | 0.0273 | 0.2791 |
| Step 1,900,000 | 10,116 | 0.4418 | 0.0308 | 0.0701 | 0.1334 | 0.1837 | 4.3408 | 0.0269 | 0.1846 |
| Step 1,950,000 | 10,371 | 0.6407 | 0.2464 | 0.0254 | 0.2376 | 0.1982 | 4.0931 | 0.0197 | 0.2791 |
| Step 2,000,000 | 10,618 | 0.7778 | 0.1322 | 0.0521 | 0.1905 | 0.2105 | 3.3368 | 0.0163 | 0.1477 |
| Step 2,050,000 | 10,868 | 0.1186 | 0.0175 | -0.0032 | 0.1431 | 0.1662 | 4.3868 | 0.0211 | 0.2791 |
| Step 2,100,000 | 11,116 | 0.4864 | 0.0770 | 0.1501 | 0.1426 | 0.3098 | 4.6313 | 0.0263 | 0.2791 |
| Step 2,150,000 | 11,370 | 0.5824 | 0.0777 | 0.0303 | 0.1618 | 0.1959 | 4.2640 | 0.0197 | 0.2791 |
| Step 2,200,000 | 11,602 | 0.1462 | 0.0983 | 0.0622 | 0.1800 | 0.1960 | 4.9921 | 0.0297 | 0.3489 |
| Step 2,250,000 | 11,831 | 0.3705 | 0.0599 | 0.0709 | 0.1376 | 0.1683 | 4.4891 | 0.0287 | 0.2791 |
| Step 2,300,000 | 12,064 | 0.0673 | 0.0479 | -0.0032 | 0.1247 | 0.1335 | 4.3075 | 0.0318 | 0.2791 |
| Step 2,350,000 | 12,301 | 0.5198 | 0.0864 | 0.0397 | 0.1663 | 0.1999 | 4.0913 | 0.0246 | 0.2791 |
| Step 2,400,000 | 12,546 | 0.1352 | 0.1596 | -0.0098 | 0.1802 | 0.1353 | 3.5992 | 0.0206 | 0.3140 |
| Step 2,450,000 | 12,780 | 0.1233 | 0.0482 | 0.0167 | 0.1540 | 0.1546 | 4.7105 | 0.0332 | 0.2791 |
| Step 2,500,000 | 13,012 | 0.2810 | 0.0484 | 0.0684 | 0.1305 | 0.1808 | 4.3058 | 0.0297 | 0.3489 |
| Step 2,550,000 | 13,242 | 0.1452 | 0.0156 | 0.0919 | 0.1119 | 0.1946 | 5.0119 | 0.0517 | 0.2791 |
| Step 2,600,000 | 13,474 | 0.2097 | 0.0378 | 0.0497 | 0.1163 | 0.1485 | 3.7186 | 0.0289 | 0.3139 |
| Step 2,650,000 | 13,716 | 0.6711 | 0.1186 | 0.1192 | 0.1765 | 0.2230 | 4.7698 | 0.0372 | 0.2031 |
| Step 2,700,000 | 13,950 | 0.5542 | 0.3785 | 0.0495 | 0.3377 | 0.1971 | 3.5415 | 0.0173 | 0.2791 |
| Step 2,750,000 | 14,179 | 0.6845 | 0.2559 | 0.1095 | 0.2172 | 0.2727 | 3.6768 | 0.0236 | 0.3877 |
| Step 2,800,000 | 14,412 | 0.5751 | 0.0942 | 0.0989 | 0.1415 | 0.2030 | 3.5418 | 0.0226 | 0.2791 |
| Step 2,850,000 | 14,644 | -0.1397 | 0.1009 | 0.0702 | 0.1595 | 0.2012 | 3.8330 | 0.0254 | 0.2031 |
| Step 2,900,000 | 14,875 | 0.1150 | 0.0360 | 0.0152 | 0.1296 | 0.1502 | 3.5138 | 0.0197 | 0.2216 |
| Step 2,950,000 | 15,115 | 0.0454 | 0.1531 | 0.1238 | 0.1676 | 0.1851 | 3.9964 | 0.0327 | 0.3489 |
| Step 3,000,000 | 15,342 | 0.2338 | 0.1238 | 0.0347 | 0.1649 | 0.1484 | 3.9306 | 0.0274 | 0.0923 |
| Final | 15,343 | 0.0288 | 0.0523 | 0.0178 | 0.1353 | 0.1570 | 3.9587 | 0.0277 | 0.2031 |

---

## collect_wood_ep2_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0624 |
| Coherence (Success) | 0.3394 |
| Coherence (Failure) | 0.1928 |
| Gradient Magnitude (Success) | 0.0929 |
| Gradient Magnitude (Failure) | 0.0344 |
| Activation Separation | 0.1352 |
| Cosine Distance | 0.0005 |
| Clusters | 1,504 |
| Noise Fraction | 0.2465 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## wake_up_ep2_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1040 |
| Coherence (Success) | 0.3315 |
| Coherence (Failure) | 0.1525 |
| Gradient Magnitude (Success) | 0.0910 |
| Gradient Magnitude (Failure) | 0.0365 |
| Activation Separation | 0.1190 |
| Cosine Distance | 0.0004 |
| Clusters | 1,562 |
| Noise Fraction | 0.2438 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## collect_drink_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0126 |
| Coherence (Success) | 0.3954 |
| Coherence (Failure) | 0.1264 |
| Gradient Magnitude (Success) | 0.0828 |
| Gradient Magnitude (Failure) | 0.0341 |
| Activation Separation | 0.1318 |
| Cosine Distance | 0.0005 |
| Clusters | 1,555 |
| Noise Fraction | 0.2544 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_table_ep5_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1738 |
| Coherence (Success) | 0.3320 |
| Coherence (Failure) | 0.3555 |
| Gradient Magnitude (Success) | 0.0835 |
| Gradient Magnitude (Failure) | 0.0363 |
| Activation Separation | 0.1326 |
| Cosine Distance | 0.0005 |
| Clusters | 1,528 |
| Noise Fraction | 0.2540 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep10_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0504 |
| Coherence (Success) | 0.2699 |
| Coherence (Failure) | 0.2862 |
| Gradient Magnitude (Success) | 0.0813 |
| Gradient Magnitude (Failure) | 0.0355 |
| Activation Separation | 0.1418 |
| Cosine Distance | 0.0005 |
| Clusters | 1,575 |
| Noise Fraction | 0.2295 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## place_plant_ep10_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3617 |
| Coherence (Success) | 0.2866 |
| Coherence (Failure) | 0.2073 |
| Gradient Magnitude (Success) | 0.0808 |
| Gradient Magnitude (Failure) | 0.0408 |
| Activation Separation | 0.1317 |
| Cosine Distance | 0.0005 |
| Clusters | 1,508 |
| Noise Fraction | 0.2418 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## eat_cow_ep50_lower2.900_upper4.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6053 |
| Coherence (Success) | 0.8578 |
| Coherence (Failure) | 0.3081 |
| Gradient Magnitude (Success) | 0.1914 |
| Gradient Magnitude (Failure) | 0.0799 |
| Activation Separation | 0.2884 |
| Cosine Distance | 0.0005 |
| Clusters | 1,482 |
| Noise Fraction | 0.2447 |
| RSA Alignment (ρ) | 0.1309 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep55_lower2.000_upper4.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4523 |
| Coherence (Success) | 0.9257 |
| Coherence (Failure) | 0.7070 |
| Gradient Magnitude (Success) | 0.2091 |
| Gradient Magnitude (Failure) | 0.1047 |
| Activation Separation | 0.3242 |
| Cosine Distance | 0.0005 |
| Clusters | 1,499 |
| Noise Fraction | 0.2415 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep83_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8639 |
| Coherence (Success) | 0.9316 |
| Coherence (Failure) | 0.2742 |
| Gradient Magnitude (Success) | 0.3459 |
| Gradient Magnitude (Failure) | 0.1655 |
| Activation Separation | 0.4379 |
| Cosine Distance | 0.0001 |
| Clusters | 1,409 |
| Noise Fraction | 0.2631 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep114_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7621 |
| Coherence (Success) | 0.8947 |
| Coherence (Failure) | 0.0840 |
| Gradient Magnitude (Success) | 0.2930 |
| Gradient Magnitude (Failure) | 0.1976 |
| Activation Separation | 0.3724 |
| Cosine Distance | 0.0001 |
| Clusters | 1,479 |
| Noise Fraction | 0.2461 |
| RSA Alignment (ρ) | -0.6093 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep125_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8134 |
| Coherence (Success) | 0.7710 |
| Coherence (Failure) | 0.4167 |
| Gradient Magnitude (Success) | 0.3007 |
| Gradient Magnitude (Failure) | 0.2002 |
| Activation Separation | 0.5255 |
| Cosine Distance | 0.0001 |
| Clusters | 1,661 |
| Noise Fraction | 0.2551 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep208_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0490 |
| Coherence (Success) | -0.0242 |
| Coherence (Failure) | 0.0990 |
| Gradient Magnitude (Success) | 0.0921 |
| Gradient Magnitude (Failure) | 0.2036 |
| Activation Separation | 1.2855 |
| Cosine Distance | 0.0002 |
| Clusters | 1,501 |
| Noise Fraction | 0.2928 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep280_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2196 |
| Coherence (Success) | 0.1228 |
| Coherence (Failure) | 0.1229 |
| Gradient Magnitude (Success) | 0.1497 |
| Gradient Magnitude (Failure) | 0.2348 |
| Activation Separation | 0.9005 |
| Cosine Distance | 0.0002 |
| Clusters | 1,607 |
| Noise Fraction | 0.2607 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep337_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1742 |
| Coherence (Success) | 0.2763 |
| Coherence (Failure) | 0.4460 |
| Gradient Magnitude (Success) | 0.1228 |
| Gradient Magnitude (Failure) | 0.2190 |
| Activation Separation | 0.7789 |
| Cosine Distance | 0.0002 |
| Clusters | 1,469 |
| Noise Fraction | 0.2583 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep480_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7402 |
| Coherence (Success) | 0.4421 |
| Coherence (Failure) | 0.2267 |
| Gradient Magnitude (Success) | 0.1833 |
| Gradient Magnitude (Failure) | 0.1967 |
| Activation Separation | 0.5958 |
| Cosine Distance | 0.0004 |
| Clusters | 1,660 |
| Noise Fraction | 0.2462 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep569_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2167 |
| Coherence (Success) | 0.0852 |
| Coherence (Failure) | 0.1471 |
| Gradient Magnitude (Success) | 0.1026 |
| Gradient Magnitude (Failure) | 0.1283 |
| Activation Separation | 0.7116 |
| Cosine Distance | 0.0007 |
| Clusters | 1,521 |
| Noise Fraction | 0.2638 |
| RSA Alignment (ρ) | -0.5222 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep854_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2693 |
| Coherence (Success) | 0.4447 |
| Coherence (Failure) | 0.2924 |
| Gradient Magnitude (Success) | 0.1706 |
| Gradient Magnitude (Failure) | 0.1969 |
| Activation Separation | 0.7448 |
| Cosine Distance | 0.0004 |
| Clusters | 1,601 |
| Noise Fraction | 0.2867 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1125_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2890 |
| Coherence (Success) | 0.2722 |
| Coherence (Failure) | 0.0491 |
| Gradient Magnitude (Success) | 0.1194 |
| Gradient Magnitude (Failure) | 0.1070 |
| Activation Separation | 0.7027 |
| Cosine Distance | 0.0005 |
| Clusters | 1,616 |
| Noise Fraction | 0.2661 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1407_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0683 |
| Coherence (Success) | 0.1637 |
| Coherence (Failure) | 0.1796 |
| Gradient Magnitude (Success) | 0.0996 |
| Gradient Magnitude (Failure) | 0.1532 |
| Activation Separation | 0.6588 |
| Cosine Distance | 0.0011 |
| Clusters | 1,737 |
| Noise Fraction | 0.2477 |
| RSA Alignment (ρ) | -0.5222 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1684_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.7424 |
| Coherence (Success) | 0.2677 |
| Coherence (Failure) | 0.3167 |
| Gradient Magnitude (Success) | 0.1019 |
| Gradient Magnitude (Failure) | 0.1732 |
| Activation Separation | 0.6607 |
| Cosine Distance | 0.0018 |
| Clusters | 1,716 |
| Noise Fraction | 0.2503 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep1781_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0486 |
| Coherence (Success) | 0.1696 |
| Coherence (Failure) | 0.3123 |
| Gradient Magnitude (Success) | 0.0959 |
| Gradient Magnitude (Failure) | 0.1570 |
| Activation Separation | 0.7958 |
| Cosine Distance | 0.0026 |
| Clusters | 1,745 |
| Noise Fraction | 0.2132 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1948_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4100 |
| Coherence (Success) | 0.1530 |
| Coherence (Failure) | 0.1028 |
| Gradient Magnitude (Success) | 0.0970 |
| Gradient Magnitude (Failure) | 0.1195 |
| Activation Separation | 0.8618 |
| Cosine Distance | 0.0037 |
| Clusters | 1,776 |
| Noise Fraction | 0.2282 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2229_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4373 |
| Coherence (Success) | 0.1294 |
| Coherence (Failure) | 0.3975 |
| Gradient Magnitude (Success) | 0.0879 |
| Gradient Magnitude (Failure) | 0.2113 |
| Activation Separation | 0.9325 |
| Cosine Distance | 0.0065 |
| Clusters | 1,486 |
| Noise Fraction | 0.2632 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2504_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2500 |
| Coherence (Success) | 0.3296 |
| Coherence (Failure) | 0.3032 |
| Gradient Magnitude (Success) | 0.1432 |
| Gradient Magnitude (Failure) | 0.1701 |
| Activation Separation | 1.5241 |
| Cosine Distance | 0.0130 |
| Clusters | 1,489 |
| Noise Fraction | 0.2434 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2777_lower5.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0459 |
| Coherence (Success) | 0.4196 |
| Coherence (Failure) | 0.1329 |
| Gradient Magnitude (Success) | 0.1845 |
| Gradient Magnitude (Failure) | 0.1593 |
| Activation Separation | 1.6503 |
| Cosine Distance | 0.0081 |
| Clusters | 1,482 |
| Noise Fraction | 0.2604 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3050_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0972 |
| Coherence (Success) | 0.3543 |
| Coherence (Failure) | 0.2381 |
| Gradient Magnitude (Success) | 0.1783 |
| Gradient Magnitude (Failure) | 0.2264 |
| Activation Separation | 2.0617 |
| Cosine Distance | 0.0110 |
| Clusters | 1,440 |
| Noise Fraction | 0.2814 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3321_lower5.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4959 |
| Coherence (Success) | 0.1404 |
| Coherence (Failure) | 0.1726 |
| Gradient Magnitude (Success) | 0.1231 |
| Gradient Magnitude (Failure) | 0.2046 |
| Activation Separation | 2.2040 |
| Cosine Distance | 0.0121 |
| Clusters | 1,548 |
| Noise Fraction | 0.2890 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3592_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0394 |
| Coherence (Success) | 0.1423 |
| Coherence (Failure) | 0.0573 |
| Gradient Magnitude (Success) | 0.1421 |
| Gradient Magnitude (Failure) | 0.1397 |
| Activation Separation | 2.5916 |
| Cosine Distance | 0.0118 |
| Clusters | 1,674 |
| Noise Fraction | 0.2764 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3853_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3683 |
| Coherence (Success) | 0.0375 |
| Coherence (Failure) | 0.2032 |
| Gradient Magnitude (Success) | 0.1348 |
| Gradient Magnitude (Failure) | 0.2103 |
| Activation Separation | 2.8900 |
| Cosine Distance | 0.0120 |
| Clusters | 1,464 |
| Noise Fraction | 0.2491 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep4113_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3869 |
| Coherence (Success) | 0.2586 |
| Coherence (Failure) | 0.1872 |
| Gradient Magnitude (Success) | 0.1800 |
| Gradient Magnitude (Failure) | 0.2452 |
| Activation Separation | 3.1901 |
| Cosine Distance | 0.0143 |
| Clusters | 1,358 |
| Noise Fraction | 0.3072 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4124_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6584 |
| Coherence (Success) | 0.2352 |
| Coherence (Failure) | 0.0890 |
| Gradient Magnitude (Success) | 0.2016 |
| Gradient Magnitude (Failure) | 0.1907 |
| Activation Separation | 2.6577 |
| Cosine Distance | 0.0107 |
| Clusters | 1,413 |
| Noise Fraction | 0.3385 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4379_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5489 |
| Coherence (Success) | 0.1850 |
| Coherence (Failure) | 0.0492 |
| Gradient Magnitude (Success) | 0.1777 |
| Gradient Magnitude (Failure) | 0.1511 |
| Activation Separation | 3.0503 |
| Cosine Distance | 0.0160 |
| Clusters | 1,392 |
| Noise Fraction | 0.3263 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4652_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3158 |
| Coherence (Success) | 0.1156 |
| Coherence (Failure) | 0.0522 |
| Gradient Magnitude (Success) | 0.1551 |
| Gradient Magnitude (Failure) | 0.1659 |
| Activation Separation | 3.3906 |
| Cosine Distance | 0.0190 |
| Clusters | 1,339 |
| Noise Fraction | 0.2844 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep4919_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3184 |
| Coherence (Success) | 0.1492 |
| Coherence (Failure) | 0.1243 |
| Gradient Magnitude (Success) | 0.1625 |
| Gradient Magnitude (Failure) | 0.1926 |
| Activation Separation | 3.8244 |
| Cosine Distance | 0.0243 |
| Clusters | 1,608 |
| Noise Fraction | 0.2919 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4925_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3241 |
| Coherence (Success) | 0.0985 |
| Coherence (Failure) | 0.0669 |
| Gradient Magnitude (Success) | 0.1324 |
| Gradient Magnitude (Failure) | 0.1547 |
| Activation Separation | 3.6890 |
| Cosine Distance | 0.0242 |
| Clusters | 1,712 |
| Noise Fraction | 0.2866 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5187_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1858 |
| Coherence (Success) | 0.0841 |
| Coherence (Failure) | 0.0060 |
| Gradient Magnitude (Success) | 0.1224 |
| Gradient Magnitude (Failure) | 0.1234 |
| Activation Separation | 4.5680 |
| Cosine Distance | 0.0282 |
| Clusters | 1,421 |
| Noise Fraction | 0.3285 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5447_lower7.000_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4896 |
| Coherence (Success) | 0.1724 |
| Coherence (Failure) | 0.0668 |
| Gradient Magnitude (Success) | 0.1611 |
| Gradient Magnitude (Failure) | 0.2032 |
| Activation Separation | 3.6312 |
| Cosine Distance | 0.0205 |
| Clusters | 1,517 |
| Noise Fraction | 0.2787 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5703_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5672 |
| Coherence (Success) | 0.1675 |
| Coherence (Failure) | 0.1688 |
| Gradient Magnitude (Success) | 0.2076 |
| Gradient Magnitude (Failure) | 0.2624 |
| Activation Separation | 4.4506 |
| Cosine Distance | 0.0233 |
| Clusters | 1,347 |
| Noise Fraction | 0.3175 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5977_lower6.000_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1536 |
| Coherence (Success) | 0.2865 |
| Coherence (Failure) | -0.0086 |
| Gradient Magnitude (Success) | 0.2347 |
| Gradient Magnitude (Failure) | 0.1464 |
| Activation Separation | 3.9591 |
| Cosine Distance | 0.0173 |
| Clusters | 1,470 |
| Noise Fraction | 0.3236 |
| RSA Alignment (ρ) | 0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6250_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7399 |
| Coherence (Success) | 0.1862 |
| Coherence (Failure) | 0.0635 |
| Gradient Magnitude (Success) | 0.1974 |
| Gradient Magnitude (Failure) | 0.1884 |
| Activation Separation | 3.8028 |
| Cosine Distance | 0.0291 |
| Clusters | 1,371 |
| Noise Fraction | 0.3065 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6523_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0035 |
| Coherence (Success) | 0.0729 |
| Coherence (Failure) | 0.0624 |
| Gradient Magnitude (Success) | 0.1408 |
| Gradient Magnitude (Failure) | 0.1759 |
| Activation Separation | 4.2199 |
| Cosine Distance | 0.0305 |
| Clusters | 1,401 |
| Noise Fraction | 0.2950 |
| RSA Alignment (ρ) | 0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6782_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5825 |
| Coherence (Success) | 0.1106 |
| Coherence (Failure) | 0.1826 |
| Gradient Magnitude (Success) | 0.1467 |
| Gradient Magnitude (Failure) | 0.2851 |
| Activation Separation | 4.8867 |
| Cosine Distance | 0.0318 |
| Clusters | 1,381 |
| Noise Fraction | 0.3255 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7045_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6133 |
| Coherence (Success) | 0.1690 |
| Coherence (Failure) | 0.0670 |
| Gradient Magnitude (Success) | 0.1730 |
| Gradient Magnitude (Failure) | 0.2184 |
| Activation Separation | 4.3985 |
| Cosine Distance | 0.0305 |
| Clusters | 1,443 |
| Noise Fraction | 0.3248 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7306_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5680 |
| Coherence (Success) | 0.2749 |
| Coherence (Failure) | 0.1702 |
| Gradient Magnitude (Success) | 0.2231 |
| Gradient Magnitude (Failure) | 0.2353 |
| Activation Separation | 4.2895 |
| Cosine Distance | 0.0234 |
| Clusters | 1,433 |
| Noise Fraction | 0.3443 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7562_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4695 |
| Coherence (Success) | 0.2328 |
| Coherence (Failure) | 0.0763 |
| Gradient Magnitude (Success) | 0.2432 |
| Gradient Magnitude (Failure) | 0.2054 |
| Activation Separation | 5.3444 |
| Cosine Distance | 0.0306 |
| Clusters | 1,468 |
| Noise Fraction | 0.3040 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7807_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3296 |
| Coherence (Success) | 0.0104 |
| Coherence (Failure) | -0.0116 |
| Gradient Magnitude (Success) | 0.1066 |
| Gradient Magnitude (Failure) | 0.1209 |
| Activation Separation | 5.1402 |
| Cosine Distance | 0.0450 |
| Clusters | 1,442 |
| Noise Fraction | 0.3262 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8065_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6255 |
| Coherence (Success) | 0.0886 |
| Coherence (Failure) | 0.1033 |
| Gradient Magnitude (Success) | 0.1542 |
| Gradient Magnitude (Failure) | 0.1991 |
| Activation Separation | 5.1954 |
| Cosine Distance | 0.0395 |
| Clusters | 1,400 |
| Noise Fraction | 0.2951 |
| RSA Alignment (ρ) | 0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8328_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7256 |
| Coherence (Success) | 0.2290 |
| Coherence (Failure) | 0.2448 |
| Gradient Magnitude (Success) | 0.2273 |
| Gradient Magnitude (Failure) | 0.3001 |
| Activation Separation | 4.3364 |
| Cosine Distance | 0.0270 |
| Clusters | 1,315 |
| Noise Fraction | 0.2717 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8581_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6326 |
| Coherence (Success) | 0.3949 |
| Coherence (Failure) | 0.1295 |
| Gradient Magnitude (Success) | 0.2970 |
| Gradient Magnitude (Failure) | 0.2331 |
| Activation Separation | 3.9010 |
| Cosine Distance | 0.0233 |
| Clusters | 1,302 |
| Noise Fraction | 0.3029 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8844_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6192 |
| Coherence (Success) | 0.2182 |
| Coherence (Failure) | 0.0803 |
| Gradient Magnitude (Success) | 0.2061 |
| Gradient Magnitude (Failure) | 0.1925 |
| Activation Separation | 4.8971 |
| Cosine Distance | 0.0400 |
| Clusters | 1,369 |
| Noise Fraction | 0.2852 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9103_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7183 |
| Coherence (Success) | 0.2638 |
| Coherence (Failure) | 0.2405 |
| Gradient Magnitude (Success) | 0.2420 |
| Gradient Magnitude (Failure) | 0.2989 |
| Activation Separation | 4.0125 |
| Cosine Distance | 0.0238 |
| Clusters | 1,420 |
| Noise Fraction | 0.3108 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9363_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3926 |
| Coherence (Success) | 0.1230 |
| Coherence (Failure) | 0.0776 |
| Gradient Magnitude (Success) | 0.2043 |
| Gradient Magnitude (Failure) | 0.2107 |
| Activation Separation | 5.2068 |
| Cosine Distance | 0.0350 |
| Clusters | 1,328 |
| Noise Fraction | 0.3192 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9613_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6008 |
| Coherence (Success) | 0.2867 |
| Coherence (Failure) | 0.1488 |
| Gradient Magnitude (Success) | 0.2543 |
| Gradient Magnitude (Failure) | 0.2716 |
| Activation Separation | 4.3421 |
| Cosine Distance | 0.0255 |
| Clusters | 1,321 |
| Noise Fraction | 0.3780 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9864_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2716 |
| Coherence (Success) | 0.2313 |
| Coherence (Failure) | 0.0062 |
| Gradient Magnitude (Success) | 0.1965 |
| Gradient Magnitude (Failure) | 0.1468 |
| Activation Separation | 4.2221 |
| Cosine Distance | 0.0273 |
| Clusters | 1,166 |
| Noise Fraction | 0.3276 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10116_lower8.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4418 |
| Coherence (Success) | 0.0308 |
| Coherence (Failure) | 0.0701 |
| Gradient Magnitude (Success) | 0.1334 |
| Gradient Magnitude (Failure) | 0.1837 |
| Activation Separation | 4.3408 |
| Cosine Distance | 0.0269 |
| Clusters | 1,420 |
| Noise Fraction | 0.3487 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10371_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6407 |
| Coherence (Success) | 0.2464 |
| Coherence (Failure) | 0.0254 |
| Gradient Magnitude (Success) | 0.2376 |
| Gradient Magnitude (Failure) | 0.1982 |
| Activation Separation | 4.0931 |
| Cosine Distance | 0.0197 |
| Clusters | 1,210 |
| Noise Fraction | 0.2963 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10618_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7778 |
| Coherence (Success) | 0.1322 |
| Coherence (Failure) | 0.0521 |
| Gradient Magnitude (Success) | 0.1905 |
| Gradient Magnitude (Failure) | 0.2105 |
| Activation Separation | 3.3368 |
| Cosine Distance | 0.0163 |
| Clusters | 1,312 |
| Noise Fraction | 0.3519 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10868_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1186 |
| Coherence (Success) | 0.0175 |
| Coherence (Failure) | -0.0032 |
| Gradient Magnitude (Success) | 0.1431 |
| Gradient Magnitude (Failure) | 0.1662 |
| Activation Separation | 4.3868 |
| Cosine Distance | 0.0211 |
| Clusters | 1,435 |
| Noise Fraction | 0.3509 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11116_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4864 |
| Coherence (Success) | 0.0770 |
| Coherence (Failure) | 0.1501 |
| Gradient Magnitude (Success) | 0.1426 |
| Gradient Magnitude (Failure) | 0.3098 |
| Activation Separation | 4.6313 |
| Cosine Distance | 0.0263 |
| Clusters | 1,258 |
| Noise Fraction | 0.3431 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11370_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5824 |
| Coherence (Success) | 0.0777 |
| Coherence (Failure) | 0.0303 |
| Gradient Magnitude (Success) | 0.1618 |
| Gradient Magnitude (Failure) | 0.1959 |
| Activation Separation | 4.2640 |
| Cosine Distance | 0.0197 |
| Clusters | 1,238 |
| Noise Fraction | 0.3077 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11602_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1462 |
| Coherence (Success) | 0.0983 |
| Coherence (Failure) | 0.0622 |
| Gradient Magnitude (Success) | 0.1800 |
| Gradient Magnitude (Failure) | 0.1960 |
| Activation Separation | 4.9921 |
| Cosine Distance | 0.0297 |
| Clusters | 1,380 |
| Noise Fraction | 0.3308 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep11831_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3705 |
| Coherence (Success) | 0.0599 |
| Coherence (Failure) | 0.0709 |
| Gradient Magnitude (Success) | 0.1376 |
| Gradient Magnitude (Failure) | 0.1683 |
| Activation Separation | 4.4891 |
| Cosine Distance | 0.0287 |
| Clusters | 1,431 |
| Noise Fraction | 0.3279 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12064_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0673 |
| Coherence (Success) | 0.0479 |
| Coherence (Failure) | -0.0032 |
| Gradient Magnitude (Success) | 0.1247 |
| Gradient Magnitude (Failure) | 0.1335 |
| Activation Separation | 4.3075 |
| Cosine Distance | 0.0318 |
| Clusters | 1,264 |
| Noise Fraction | 0.3365 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12301_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5198 |
| Coherence (Success) | 0.0864 |
| Coherence (Failure) | 0.0397 |
| Gradient Magnitude (Success) | 0.1663 |
| Gradient Magnitude (Failure) | 0.1999 |
| Activation Separation | 4.0913 |
| Cosine Distance | 0.0246 |
| Clusters | 1,218 |
| Noise Fraction | 0.3381 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12546_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1352 |
| Coherence (Success) | 0.1596 |
| Coherence (Failure) | -0.0098 |
| Gradient Magnitude (Success) | 0.1802 |
| Gradient Magnitude (Failure) | 0.1353 |
| Activation Separation | 3.5992 |
| Cosine Distance | 0.0206 |
| Clusters | 1,211 |
| Noise Fraction | 0.3829 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep12780_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1233 |
| Coherence (Success) | 0.0482 |
| Coherence (Failure) | 0.0167 |
| Gradient Magnitude (Success) | 0.1540 |
| Gradient Magnitude (Failure) | 0.1546 |
| Activation Separation | 4.7105 |
| Cosine Distance | 0.0332 |
| Clusters | 1,268 |
| Noise Fraction | 0.3185 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13012_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2810 |
| Coherence (Success) | 0.0484 |
| Coherence (Failure) | 0.0684 |
| Gradient Magnitude (Success) | 0.1305 |
| Gradient Magnitude (Failure) | 0.1808 |
| Activation Separation | 4.3058 |
| Cosine Distance | 0.0297 |
| Clusters | 1,310 |
| Noise Fraction | 0.3045 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13242_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1452 |
| Coherence (Success) | 0.0156 |
| Coherence (Failure) | 0.0919 |
| Gradient Magnitude (Success) | 0.1119 |
| Gradient Magnitude (Failure) | 0.1946 |
| Activation Separation | 5.0119 |
| Cosine Distance | 0.0517 |
| Clusters | 1,477 |
| Noise Fraction | 0.3299 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13474_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2097 |
| Coherence (Success) | 0.0378 |
| Coherence (Failure) | 0.0497 |
| Gradient Magnitude (Success) | 0.1163 |
| Gradient Magnitude (Failure) | 0.1485 |
| Activation Separation | 3.7186 |
| Cosine Distance | 0.0289 |
| Clusters | 1,320 |
| Noise Fraction | 0.3109 |
| RSA Alignment (ρ) | 0.3139 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13716_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6711 |
| Coherence (Success) | 0.1186 |
| Coherence (Failure) | 0.1192 |
| Gradient Magnitude (Success) | 0.1765 |
| Gradient Magnitude (Failure) | 0.2230 |
| Activation Separation | 4.7698 |
| Cosine Distance | 0.0372 |
| Clusters | 1,355 |
| Noise Fraction | 0.3388 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13950_lower8.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5542 |
| Coherence (Success) | 0.3785 |
| Coherence (Failure) | 0.0495 |
| Gradient Magnitude (Success) | 0.3377 |
| Gradient Magnitude (Failure) | 0.1971 |
| Activation Separation | 3.5415 |
| Cosine Distance | 0.0173 |
| Clusters | 1,348 |
| Noise Fraction | 0.3471 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14179_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6845 |
| Coherence (Success) | 0.2559 |
| Coherence (Failure) | 0.1095 |
| Gradient Magnitude (Success) | 0.2172 |
| Gradient Magnitude (Failure) | 0.2727 |
| Activation Separation | 3.6768 |
| Cosine Distance | 0.0236 |
| Clusters | 1,270 |
| Noise Fraction | 0.3630 |
| RSA Alignment (ρ) | 0.3877 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14412_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5751 |
| Coherence (Success) | 0.0942 |
| Coherence (Failure) | 0.0989 |
| Gradient Magnitude (Success) | 0.1415 |
| Gradient Magnitude (Failure) | 0.2030 |
| Activation Separation | 3.5418 |
| Cosine Distance | 0.0226 |
| Clusters | 1,350 |
| Noise Fraction | 0.3176 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14644_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1397 |
| Coherence (Success) | 0.1009 |
| Coherence (Failure) | 0.0702 |
| Gradient Magnitude (Success) | 0.1595 |
| Gradient Magnitude (Failure) | 0.2012 |
| Activation Separation | 3.8330 |
| Cosine Distance | 0.0254 |
| Clusters | 1,357 |
| Noise Fraction | 0.3282 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14875_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1150 |
| Coherence (Success) | 0.0360 |
| Coherence (Failure) | 0.0152 |
| Gradient Magnitude (Success) | 0.1296 |
| Gradient Magnitude (Failure) | 0.1502 |
| Activation Separation | 3.5138 |
| Cosine Distance | 0.0197 |
| Clusters | 1,419 |
| Noise Fraction | 0.3433 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15115_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0454 |
| Coherence (Success) | 0.1531 |
| Coherence (Failure) | 0.1238 |
| Gradient Magnitude (Success) | 0.1676 |
| Gradient Magnitude (Failure) | 0.1851 |
| Activation Separation | 3.9964 |
| Cosine Distance | 0.0327 |
| Clusters | 1,282 |
| Noise Fraction | 0.3207 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep15342_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2338 |
| Coherence (Success) | 0.1238 |
| Coherence (Failure) | 0.0347 |
| Gradient Magnitude (Success) | 0.1649 |
| Gradient Magnitude (Failure) | 0.1484 |
| Activation Separation | 3.9306 |
| Cosine Distance | 0.0274 |
| Clusters | 1,407 |
| Noise Fraction | 0.3397 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15343_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0288 |
| Coherence (Success) | 0.0523 |
| Coherence (Failure) | 0.0178 |
| Gradient Magnitude (Success) | 0.1353 |
| Gradient Magnitude (Failure) | 0.1570 |
| Activation Separation | 3.9587 |
| Cosine Distance | 0.0277 |
| Clusters | 1,384 |
| Noise Fraction | 0.3479 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep2 | 2 | 0.0624 | 0.3394 | 0.1928 | 0.0929 | 0.0344 | 0.1352 | 0.0005 | nan |
| wake_up @ ep2 | 2 | 0.1040 | 0.3315 | 0.1525 | 0.0910 | 0.0365 | 0.1190 | 0.0004 | — |
| collect_drink @ ep3 | 3 | -0.0126 | 0.3954 | 0.1264 | 0.0828 | 0.0341 | 0.1318 | 0.0005 | nan |
| place_table @ ep5 | 5 | -0.1738 | 0.3320 | 0.3555 | 0.0835 | 0.0363 | 0.1326 | 0.0005 | nan |
| collect_sapling @ ep10 | 10 | 0.0504 | 0.2699 | 0.2862 | 0.0813 | 0.0355 | 0.1418 | 0.0005 | — |
| place_plant @ ep10 | 10 | 0.3617 | 0.2866 | 0.2073 | 0.0808 | 0.0408 | 0.1317 | 0.0005 | — |
| eat_cow @ ep50 | 50 | 0.6053 | 0.8578 | 0.3081 | 0.1914 | 0.0799 | 0.2884 | 0.0005 | 0.1309 |
| make_wood_sword @ ep55 | 55 | 0.4523 | 0.9257 | 0.7070 | 0.2091 | 0.1047 | 0.3242 | 0.0005 | -0.3928 |
| defeat_zombie @ ep83 | 83 | 0.8639 | 0.9316 | 0.2742 | 0.3459 | 0.1655 | 0.4379 | 0.0001 | -0.4352 |
| make_wood_pickaxe @ ep114 | 114 | 0.7621 | 0.8947 | 0.0840 | 0.2930 | 0.1976 | 0.3724 | 0.0001 | -0.6093 |
| collect_stone @ ep125 | 125 | 0.8134 | 0.7710 | 0.4167 | 0.3007 | 0.2002 | 0.5255 | 0.0001 | -0.2094 |
| defeat_skeleton @ ep208 | 208 | 0.0490 | -0.0242 | 0.0990 | 0.0921 | 0.2036 | 1.2855 | 0.0002 | -0.3489 |
| collect_coal @ ep337 | 337 | 0.1742 | 0.2763 | 0.4460 | 0.1228 | 0.2190 | 0.7789 | 0.0002 | -0.2791 |
| place_stone @ ep480 | 480 | 0.7402 | 0.4421 | 0.2267 | 0.1833 | 0.1967 | 0.5958 | 0.0004 | -0.2611 |
| place_furnace @ ep1781 | 1,781 | 0.0486 | 0.1696 | 0.3123 | 0.0959 | 0.1570 | 0.7958 | 0.0026 | -0.0698 |
| make_stone_sword @ ep4113 | 4,113 | 0.3869 | 0.2586 | 0.1872 | 0.1800 | 0.2452 | 3.1901 | 0.0143 | -0.0698 |
| make_stone_pickaxe @ ep4919 | 4,919 | 0.3184 | 0.1492 | 0.1243 | 0.1625 | 0.1926 | 3.8244 | 0.0243 | 0.0000 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 280 | -0.2196 | 0.1228 | 0.1229 | 0.1497 | 0.2348 | 0.9005 | 0.0002 | -0.1047 |
| Step 100,000 | 569 | 0.2167 | 0.0852 | 0.1471 | 0.1026 | 0.1283 | 0.7116 | 0.0007 | -0.5222 |
| Step 150,000 | 854 | 0.2693 | 0.4447 | 0.2924 | 0.1706 | 0.1969 | 0.7448 | 0.0004 | -0.3489 |
| Step 200,000 | 1,125 | -0.2890 | 0.2722 | 0.0491 | 0.1194 | 0.1070 | 0.7027 | 0.0005 | -0.3482 |
| Step 250,000 | 1,407 | -0.0683 | 0.1637 | 0.1796 | 0.0996 | 0.1532 | 0.6588 | 0.0011 | -0.5222 |
| Step 300,000 | 1,684 | -0.7424 | 0.2677 | 0.3167 | 0.1019 | 0.1732 | 0.6607 | 0.0018 | -0.1745 |
| Step 350,000 | 1,948 | -0.4100 | 0.1530 | 0.1028 | 0.0970 | 0.1195 | 0.8618 | 0.0037 | -0.0698 |
| Step 400,000 | 2,229 | -0.4373 | 0.1294 | 0.3975 | 0.0879 | 0.2113 | 0.9325 | 0.0065 | -0.2094 |
| Step 450,000 | 2,504 | -0.2500 | 0.3296 | 0.3032 | 0.1432 | 0.1701 | 1.5241 | 0.0130 | -0.1292 |
| Step 500,000 | 2,777 | 0.0459 | 0.4196 | 0.1329 | 0.1845 | 0.1593 | 1.6503 | 0.0081 | -0.2791 |
| Step 550,000 | 3,050 | 0.0972 | 0.3543 | 0.2381 | 0.1783 | 0.2264 | 2.0617 | 0.0110 | -0.2094 |
| Step 600,000 | 3,321 | 0.4959 | 0.1404 | 0.1726 | 0.1231 | 0.2046 | 2.2040 | 0.0121 | -0.0739 |
| Step 650,000 | 3,592 | 0.0394 | 0.1423 | 0.0573 | 0.1421 | 0.1397 | 2.5916 | 0.0118 | -0.1292 |
| Step 700,000 | 3,853 | 0.3683 | 0.0375 | 0.2032 | 0.1348 | 0.2103 | 2.8900 | 0.0120 | -0.0349 |
| Step 750,000 | 4,124 | 0.6584 | 0.2352 | 0.0890 | 0.2016 | 0.1907 | 2.6577 | 0.0107 | -0.0698 |
| Step 800,000 | 4,379 | 0.5489 | 0.1850 | 0.0492 | 0.1777 | 0.1511 | 3.0503 | 0.0160 | -0.0349 |
| Step 850,000 | 4,652 | 0.3158 | 0.1156 | 0.0522 | 0.1551 | 0.1659 | 3.3906 | 0.0190 | -0.1662 |
| Step 900,000 | 4,925 | 0.3241 | 0.0985 | 0.0669 | 0.1324 | 0.1547 | 3.6890 | 0.0242 | -0.0185 |
| Step 950,000 | 5,187 | -0.1858 | 0.0841 | 0.0060 | 0.1224 | 0.1234 | 4.5680 | 0.0282 | -0.1047 |
| Step 1,000,000 | 5,447 | 0.4896 | 0.1724 | 0.0668 | 0.1611 | 0.2032 | 3.6312 | 0.0205 | -0.0349 |
| Step 1,050,000 | 5,703 | 0.5672 | 0.1675 | 0.1688 | 0.2076 | 0.2624 | 4.4506 | 0.0233 | 0.0000 |
| Step 1,100,000 | 5,977 | 0.1536 | 0.2865 | -0.0086 | 0.2347 | 0.1464 | 3.9591 | 0.0173 | 0.0349 |
| Step 1,150,000 | 6,250 | 0.7399 | 0.1862 | 0.0635 | 0.1974 | 0.1884 | 3.8028 | 0.0291 | 0.1396 |
| Step 1,200,000 | 6,523 | -0.0035 | 0.0729 | 0.0624 | 0.1408 | 0.1759 | 4.2199 | 0.0305 | 0.0349 |
| Step 1,250,000 | 6,782 | 0.5825 | 0.1106 | 0.1826 | 0.1467 | 0.2851 | 4.8867 | 0.0318 | 0.0000 |
| Step 1,300,000 | 7,045 | 0.6133 | 0.1690 | 0.0670 | 0.1730 | 0.2184 | 4.3985 | 0.0305 | 0.0739 |
| Step 1,350,000 | 7,306 | 0.5680 | 0.2749 | 0.1702 | 0.2231 | 0.2353 | 4.2895 | 0.0234 | 0.0369 |
| Step 1,400,000 | 7,562 | 0.4695 | 0.2328 | 0.0763 | 0.2432 | 0.2054 | 5.3444 | 0.0306 | -0.0739 |
| Step 1,450,000 | 7,807 | -0.3296 | 0.0104 | -0.0116 | 0.1066 | 0.1209 | 5.1402 | 0.0450 | 0.0185 |
| Step 1,500,000 | 8,065 | 0.6255 | 0.0886 | 0.1033 | 0.1542 | 0.1991 | 5.1954 | 0.0395 | 0.1047 |
| Step 1,550,000 | 8,328 | 0.7256 | 0.2290 | 0.2448 | 0.2273 | 0.3001 | 4.3364 | 0.0270 | -0.1108 |
| Step 1,600,000 | 8,581 | 0.6326 | 0.3949 | 0.1295 | 0.2970 | 0.2331 | 3.9010 | 0.0233 | 0.0554 |
| Step 1,650,000 | 8,844 | 0.6192 | 0.2182 | 0.0803 | 0.2061 | 0.1925 | 4.8971 | 0.0400 | 0.2443 |
| Step 1,700,000 | 9,103 | 0.7183 | 0.2638 | 0.2405 | 0.2420 | 0.2989 | 4.0125 | 0.0238 | 0.1292 |
| Step 1,750,000 | 9,363 | 0.3926 | 0.1230 | 0.0776 | 0.2043 | 0.2107 | 5.2068 | 0.0350 | 0.1477 |
| Step 1,800,000 | 9,613 | 0.6008 | 0.2867 | 0.1488 | 0.2543 | 0.2716 | 4.3421 | 0.0255 | 0.3140 |
| Step 1,850,000 | 9,864 | 0.2716 | 0.2313 | 0.0062 | 0.1965 | 0.1468 | 4.2221 | 0.0273 | 0.2791 |
| Step 1,900,000 | 10,116 | 0.4418 | 0.0308 | 0.0701 | 0.1334 | 0.1837 | 4.3408 | 0.0269 | 0.1846 |
| Step 1,950,000 | 10,371 | 0.6407 | 0.2464 | 0.0254 | 0.2376 | 0.1982 | 4.0931 | 0.0197 | 0.2791 |
| Step 2,000,000 | 10,618 | 0.7778 | 0.1322 | 0.0521 | 0.1905 | 0.2105 | 3.3368 | 0.0163 | 0.1477 |
| Step 2,050,000 | 10,868 | 0.1186 | 0.0175 | -0.0032 | 0.1431 | 0.1662 | 4.3868 | 0.0211 | 0.2791 |
| Step 2,100,000 | 11,116 | 0.4864 | 0.0770 | 0.1501 | 0.1426 | 0.3098 | 4.6313 | 0.0263 | 0.2791 |
| Step 2,150,000 | 11,370 | 0.5824 | 0.0777 | 0.0303 | 0.1618 | 0.1959 | 4.2640 | 0.0197 | 0.2791 |
| Step 2,200,000 | 11,602 | 0.1462 | 0.0983 | 0.0622 | 0.1800 | 0.1960 | 4.9921 | 0.0297 | 0.3489 |
| Step 2,250,000 | 11,831 | 0.3705 | 0.0599 | 0.0709 | 0.1376 | 0.1683 | 4.4891 | 0.0287 | 0.2791 |
| Step 2,300,000 | 12,064 | 0.0673 | 0.0479 | -0.0032 | 0.1247 | 0.1335 | 4.3075 | 0.0318 | 0.2791 |
| Step 2,350,000 | 12,301 | 0.5198 | 0.0864 | 0.0397 | 0.1663 | 0.1999 | 4.0913 | 0.0246 | 0.2791 |
| Step 2,400,000 | 12,546 | 0.1352 | 0.1596 | -0.0098 | 0.1802 | 0.1353 | 3.5992 | 0.0206 | 0.3140 |
| Step 2,450,000 | 12,780 | 0.1233 | 0.0482 | 0.0167 | 0.1540 | 0.1546 | 4.7105 | 0.0332 | 0.2791 |
| Step 2,500,000 | 13,012 | 0.2810 | 0.0484 | 0.0684 | 0.1305 | 0.1808 | 4.3058 | 0.0297 | 0.3489 |
| Step 2,550,000 | 13,242 | 0.1452 | 0.0156 | 0.0919 | 0.1119 | 0.1946 | 5.0119 | 0.0517 | 0.2791 |
| Step 2,600,000 | 13,474 | 0.2097 | 0.0378 | 0.0497 | 0.1163 | 0.1485 | 3.7186 | 0.0289 | 0.3139 |
| Step 2,650,000 | 13,716 | 0.6711 | 0.1186 | 0.1192 | 0.1765 | 0.2230 | 4.7698 | 0.0372 | 0.2031 |
| Step 2,700,000 | 13,950 | 0.5542 | 0.3785 | 0.0495 | 0.3377 | 0.1971 | 3.5415 | 0.0173 | 0.2791 |
| Step 2,750,000 | 14,179 | 0.6845 | 0.2559 | 0.1095 | 0.2172 | 0.2727 | 3.6768 | 0.0236 | 0.3877 |
| Step 2,800,000 | 14,412 | 0.5751 | 0.0942 | 0.0989 | 0.1415 | 0.2030 | 3.5418 | 0.0226 | 0.2791 |
| Step 2,850,000 | 14,644 | -0.1397 | 0.1009 | 0.0702 | 0.1595 | 0.2012 | 3.8330 | 0.0254 | 0.2031 |
| Step 2,900,000 | 14,875 | 0.1150 | 0.0360 | 0.0152 | 0.1296 | 0.1502 | 3.5138 | 0.0197 | 0.2216 |
| Step 2,950,000 | 15,115 | 0.0454 | 0.1531 | 0.1238 | 0.1676 | 0.1851 | 3.9964 | 0.0327 | 0.3489 |
| Step 3,000,000 | 15,342 | 0.2338 | 0.1238 | 0.0347 | 0.1649 | 0.1484 | 3.9306 | 0.0274 | 0.0923 |
| Final | 15,343 | 0.0288 | 0.0523 | 0.0178 | 0.1353 | 0.1570 | 3.9587 | 0.0277 | 0.2031 |
