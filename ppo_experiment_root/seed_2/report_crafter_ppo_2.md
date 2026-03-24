# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 2  
**Total episodes:** 14,869  
**Experiment root:** `experiment_root\seed_2`  
**Generated:** 2026-03-21 22:34

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.3160 | 0.6958 | 0.0496 | 0.0426 | 0.0131 | 0.1137 | 0.0024 | -0.6547 |
| place_table @ ep1 | 1 | -0.1096 | 0.6262 | 0.1057 | 0.0422 | 0.0169 | 0.0932 | 0.0016 | -0.3928 |
| wake_up @ ep2 | 2 | 0.0401 | 0.9020 | 0.0637 | 0.0650 | 0.0217 | 0.1736 | 0.0024 | 0.1309 |
| collect_sapling @ ep3 | 3 | 0.2505 | 0.8920 | 0.0892 | 0.0633 | 0.0245 | 0.1422 | 0.0016 | nan |
| place_plant @ ep3 | 3 | 0.0558 | 0.8699 | 0.0325 | 0.0579 | 0.0240 | 0.1597 | 0.0020 | nan |
| collect_drink @ ep5 | 5 | 0.1524 | 0.8854 | -0.0334 | 0.0637 | 0.0166 | 0.1610 | 0.0020 | -0.6547 |
| eat_cow @ ep70 | 70 | 0.8386 | 0.8958 | 0.6628 | 0.1948 | 0.1332 | 0.3748 | 0.0011 | nan |
| make_wood_pickaxe @ ep72 | 72 | 0.6663 | 0.9376 | 0.2143 | 0.2143 | 0.1009 | 0.3476 | 0.0009 | -0.6547 |
| make_wood_sword @ ep137 | 137 | 0.8363 | 0.8507 | 0.1744 | 0.2030 | 0.1774 | 0.7333 | 0.0004 | -0.5222 |
| defeat_skeleton @ ep205 | 205 | 0.3268 | 0.6106 | 0.3098 | 0.2305 | 0.2030 | 0.7545 | 0.0003 | -0.1741 |
| defeat_zombie @ ep226 | 226 | 0.2455 | 0.4133 | 0.2624 | 0.1805 | 0.1731 | 0.8038 | 0.0003 | -0.2094 |
| periodic_step50000_ep282 | 282 | -0.6970 | 0.0366 | 0.3191 | 0.0965 | 0.1395 | 1.0327 | 0.0005 | -0.2094 |
| collect_stone @ ep427 | 427 | 0.2407 | 0.0987 | 0.1716 | 0.1039 | 0.1808 | 1.2253 | 0.0006 | -0.2443 |
| periodic_step100000_ep565 | 565 | 0.0936 | 0.0657 | -0.0218 | 0.1260 | 0.1079 | 1.2973 | 0.0003 | 0.0000 |
| periodic_step150000_ep849 | 849 | 0.3059 | 0.1412 | 0.2970 | 0.1376 | 0.2572 | 0.3048 | 0.0002 | -0.1745 |
| place_stone @ ep1109 | 1,109 | -0.5123 | 0.1200 | 0.0111 | 0.1395 | 0.1144 | 1.0177 | 0.0009 | 0.1741 |
| periodic_step200000_ep1132 | 1,132 | -0.6253 | 0.1281 | 0.0551 | 0.1267 | 0.0944 | 0.3994 | 0.0005 | -0.1741 |
| make_stone_pickaxe @ ep1183 | 1,183 | 0.2238 | 0.0564 | 0.2479 | 0.0908 | 0.2007 | 0.7262 | 0.0005 | 0.0000 |
| collect_coal @ ep1231 | 1,231 | 0.5333 | 0.6664 | 0.3583 | 0.2242 | 0.2532 | 0.3661 | 0.0008 | -0.3140 |
| make_stone_sword @ ep1231 | 1,231 | 0.7306 | 0.5473 | 0.4439 | 0.1994 | 0.2618 | 0.3984 | 0.0010 | -0.2094 |
| periodic_step250000_ep1421 | 1,421 | 0.6836 | 0.3698 | 0.4379 | 0.1655 | 0.1967 | 0.5300 | 0.0008 | 0.6547 |
| periodic_step300000_ep1712 | 1,712 | -0.6069 | 0.1585 | 0.2836 | 0.1088 | 0.2586 | 0.8650 | 0.0028 | -0.2443 |
| periodic_step350000_ep1992 | 1,992 | -0.2352 | 0.3447 | 0.0928 | 0.1667 | 0.1086 | 1.3179 | 0.0051 | -0.3838 |
| periodic_step400000_ep2269 | 2,269 | 0.2092 | 0.1206 | 0.1001 | 0.1441 | 0.1805 | 1.5606 | 0.0067 | -0.3838 |
| place_furnace @ ep2277 | 2,277 | 0.7587 | 0.1215 | 0.2131 | 0.1325 | 0.2365 | 1.4662 | 0.0058 | -0.3482 |
| periodic_step450000_ep2555 | 2,555 | 0.3330 | 0.1708 | 0.0343 | 0.1342 | 0.0995 | 1.2784 | 0.0043 | -0.3489 |
| periodic_step500000_ep2830 | 2,830 | 0.4920 | 0.1928 | 0.0897 | 0.1568 | 0.2417 | 1.5901 | 0.0048 | -0.2443 |
| eat_plant @ ep3022 | 3,022 | -0.2583 | 0.1700 | 0.0943 | 0.1399 | 0.1878 | 2.0432 | 0.0096 | -0.1047 |
| periodic_step550000_ep3100 | 3,100 | -0.2541 | 0.1073 | 0.2457 | 0.1160 | 0.2106 | 1.9124 | 0.0102 | -0.1745 |
| periodic_step600000_ep3360 | 3,360 | 0.6352 | 0.3062 | 0.2183 | 0.1610 | 0.2430 | 2.3048 | 0.0142 | 0.1108 |
| periodic_step650000_ep3636 | 3,636 | 0.6270 | 0.1838 | 0.2249 | 0.1544 | 0.3500 | 2.3062 | 0.0109 | -0.0739 |
| periodic_step700000_ep3910 | 3,910 | 0.8274 | 0.1649 | 0.0895 | 0.1879 | 0.2032 | 2.0474 | 0.0109 | -0.1396 |
| periodic_step750000_ep4177 | 4,177 | 0.3227 | 0.0905 | 0.2229 | 0.1303 | 0.2855 | 2.6554 | 0.0133 | -0.1396 |
| periodic_step800000_ep4445 | 4,445 | 0.8568 | 0.5159 | 0.3145 | 0.3389 | 0.3697 | 2.9103 | 0.0121 | -0.1292 |
| periodic_step850000_ep4710 | 4,710 | 0.5386 | 0.1800 | 0.0001 | 0.1560 | 0.1318 | 2.6389 | 0.0122 | -0.1108 |
| periodic_step900000_ep4979 | 4,979 | 0.5537 | 0.0566 | 0.0715 | 0.1283 | 0.1647 | 2.4009 | 0.0101 | -0.1047 |
| periodic_step950000_ep5250 | 5,250 | 0.2754 | 0.1285 | 0.0357 | 0.1487 | 0.1653 | 2.8129 | 0.0137 | -0.1477 |
| periodic_step1000000_ep5511 | 5,511 | -0.1192 | 0.0417 | 0.0569 | 0.1264 | 0.1914 | 2.7841 | 0.0142 | -0.0185 |
| periodic_step1050000_ep5779 | 5,779 | 0.3082 | 0.1812 | 0.0715 | 0.1749 | 0.1682 | 3.3460 | 0.0166 | -0.1047 |
| periodic_step1100000_ep6046 | 6,046 | 0.5727 | 0.1324 | 0.1429 | 0.1545 | 0.2062 | 3.6576 | 0.0179 | 0.0000 |
| periodic_step1150000_ep6309 | 6,309 | 0.8519 | 0.2044 | 0.2499 | 0.2295 | 0.3465 | 3.0579 | 0.0124 | -0.0349 |
| periodic_step1200000_ep6573 | 6,573 | 0.7529 | 0.3179 | 0.1855 | 0.3161 | 0.3274 | 3.0088 | 0.0099 | -0.0349 |
| periodic_step1250000_ep6841 | 6,841 | 0.4616 | 0.0622 | -0.0050 | 0.1338 | 0.1302 | 3.8879 | 0.0126 | -0.0349 |
| periodic_step1300000_ep7092 | 7,092 | 0.1164 | 0.2350 | 0.0695 | 0.1721 | 0.2057 | 4.6512 | 0.0238 | -0.0349 |
| periodic_step1350000_ep7348 | 7,348 | -0.2090 | -0.0012 | 0.0702 | 0.1114 | 0.1934 | 3.8207 | 0.0166 | 0.0698 |
| periodic_step1400000_ep7583 | 7,583 | 0.3682 | 0.1224 | 0.0572 | 0.1579 | 0.2144 | 4.3710 | 0.0191 | 0.1745 |
| periodic_step1450000_ep7842 | 7,842 | 0.5509 | 0.1540 | 0.1007 | 0.1905 | 0.2814 | 4.3243 | 0.0196 | 0.0000 |
| periodic_step1500000_ep8093 | 8,093 | 0.7458 | 0.2470 | 0.2707 | 0.2152 | 0.3958 | 3.8701 | 0.0177 | 0.1292 |
| periodic_step1550000_ep8346 | 8,346 | 0.2775 | 0.1331 | 0.0135 | 0.1532 | 0.1678 | 3.7288 | 0.0146 | 0.1396 |
| periodic_step1600000_ep8588 | 8,588 | 0.3904 | 0.3103 | 0.0685 | 0.2405 | 0.1904 | 3.7089 | 0.0163 | 0.2094 |
| periodic_step1650000_ep8831 | 8,831 | 0.5173 | 0.1821 | 0.0491 | 0.2379 | 0.2252 | 4.5067 | 0.0188 | 0.2094 |
| periodic_step1700000_ep9060 | 9,060 | 0.2167 | 0.0831 | 0.0574 | 0.1511 | 0.1885 | 3.7985 | 0.0139 | 0.2443 |
| periodic_step1750000_ep9301 | 9,301 | 0.4519 | 0.0952 | 0.1140 | 0.1643 | 0.2615 | 3.8382 | 0.0153 | 0.3489 |
| periodic_step1800000_ep9541 | 9,541 | 0.6137 | 0.1994 | 0.1235 | 0.1925 | 0.2853 | 3.8576 | 0.0226 | 0.3140 |
| periodic_step1850000_ep9778 | 9,778 | 0.2327 | 0.2726 | -0.0108 | 0.2084 | 0.1395 | 3.5791 | 0.0170 | 0.2094 |
| periodic_step1900000_ep10013 | 10,013 | 0.1180 | 0.0425 | 0.1099 | 0.1340 | 0.2057 | 3.7062 | 0.0179 | 0.2216 |
| collect_iron @ ep10100 | 10,100 | -0.0347 | 0.0827 | 0.0659 | 0.1459 | 0.2077 | 3.8153 | 0.0183 | 0.3139 |
| periodic_step1950000_ep10243 | 10,243 | 0.4499 | 0.1045 | 0.1261 | 0.1627 | 0.2606 | 4.0157 | 0.0182 | 0.2216 |
| periodic_step2000000_ep10466 | 10,466 | 0.1376 | 0.1015 | 0.0436 | 0.1671 | 0.1857 | 3.4282 | 0.0146 | 0.2216 |
| periodic_step2050000_ep10692 | 10,692 | 0.3837 | 0.1568 | 0.1311 | 0.1767 | 0.2865 | 4.5588 | 0.0266 | 0.1662 |
| periodic_step2100000_ep10912 | 10,912 | 0.4597 | 0.4211 | 0.0123 | 0.3347 | 0.1790 | 3.6391 | 0.0162 | 0.2770 |
| periodic_step2150000_ep11130 | 11,130 | 0.5050 | 0.3897 | 0.0238 | 0.2838 | 0.1865 | 3.7898 | 0.0213 | 0.0554 |
| periodic_step2200000_ep11348 | 11,348 | 0.5721 | 0.2985 | -0.0216 | 0.2602 | 0.1419 | 3.0583 | 0.0146 | 0.2216 |
| periodic_step2250000_ep11567 | 11,567 | 0.2058 | 0.1716 | 0.0592 | 0.1808 | 0.1855 | 3.7519 | 0.0184 | 0.2216 |
| periodic_step2300000_ep11772 | 11,772 | 0.4628 | 0.2134 | 0.0183 | 0.2206 | 0.1841 | 3.7909 | 0.0219 | 0.1477 |
| periodic_step2350000_ep11991 | 11,991 | 0.4799 | 0.2248 | 0.1221 | 0.1785 | 0.2244 | 3.2568 | 0.0182 | 0.1662 |
| periodic_step2400000_ep12213 | 12,213 | 0.5054 | 0.1452 | 0.0255 | 0.1848 | 0.1576 | 3.5546 | 0.0170 | 0.0783 |
| periodic_step2450000_ep12441 | 12,441 | 0.5056 | 0.2051 | 0.0486 | 0.1928 | 0.1697 | 4.1834 | 0.0271 | 0.1477 |
| periodic_step2500000_ep12667 | 12,667 | 0.5964 | 0.1801 | 0.0897 | 0.2037 | 0.1945 | 4.1909 | 0.0295 | 0.2585 |
| periodic_step2550000_ep12889 | 12,889 | 0.1660 | 0.1851 | -0.0072 | 0.2069 | 0.1243 | 3.9295 | 0.0233 | 0.3139 |
| periodic_step2600000_ep13104 | 13,104 | 0.6246 | 0.2978 | 0.0692 | 0.2485 | 0.1825 | 4.0554 | 0.0246 | 0.2400 |
| periodic_step2650000_ep13319 | 13,319 | 0.5027 | 0.5268 | 0.0096 | 0.4252 | 0.1917 | 3.9269 | 0.0194 | 0.1846 |
| periodic_step2700000_ep13532 | 13,532 | 0.4356 | 0.3842 | -0.0322 | 0.3061 | 0.1490 | 3.9119 | 0.0208 | 0.1846 |
| periodic_step2750000_ep13764 | 13,764 | 0.2440 | 0.2111 | 0.0085 | 0.2037 | 0.1384 | 3.1398 | 0.0193 | 0.1846 |
| periodic_step2800000_ep13983 | 13,983 | 0.4572 | 0.1820 | 0.0309 | 0.2139 | 0.1846 | 3.3702 | 0.0194 | 0.1477 |
| periodic_step2850000_ep14202 | 14,202 | 0.6666 | 0.1830 | 0.0726 | 0.2135 | 0.2349 | 4.1911 | 0.0307 | 0.0923 |
| periodic_step2900000_ep14417 | 14,417 | 0.6258 | 0.2430 | 0.0281 | 0.2499 | 0.2292 | 4.3558 | 0.0263 | 0.1662 |
| periodic_step2950000_ep14644 | 14,644 | 0.1748 | 0.1156 | 0.0277 | 0.1740 | 0.1768 | 4.4179 | 0.0355 | 0.1292 |
| periodic_step3000000_ep14867 | 14,867 | 0.5542 | 0.3163 | 0.1074 | 0.2498 | 0.2327 | 3.3373 | 0.0206 | 0.1108 |
| final_step3000320_ep14869 | 14,869 | 0.4951 | 0.2505 | 0.0539 | 0.2682 | 0.1953 | 3.9545 | 0.0277 | 0.1664 |

---

## collect_wood_ep1_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3160 |
| Coherence (Success) | 0.6958 |
| Coherence (Failure) | 0.0496 |
| Gradient Magnitude (Success) | 0.0426 |
| Gradient Magnitude (Failure) | 0.0131 |
| Activation Separation | 0.1137 |
| Cosine Distance | 0.0024 |
| Clusters | 1,483 |
| Noise Fraction | 0.2537 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep1_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1096 |
| Coherence (Success) | 0.6262 |
| Coherence (Failure) | 0.1057 |
| Gradient Magnitude (Success) | 0.0422 |
| Gradient Magnitude (Failure) | 0.0169 |
| Activation Separation | 0.0932 |
| Cosine Distance | 0.0016 |
| Clusters | 1,412 |
| Noise Fraction | 0.2644 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## wake_up_ep2_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0401 |
| Coherence (Success) | 0.9020 |
| Coherence (Failure) | 0.0637 |
| Gradient Magnitude (Success) | 0.0650 |
| Gradient Magnitude (Failure) | 0.0217 |
| Activation Separation | 0.1736 |
| Cosine Distance | 0.0024 |
| Clusters | 1,511 |
| Noise Fraction | 0.2539 |
| RSA Alignment (ρ) | 0.1309 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep3_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2505 |
| Coherence (Success) | 0.8920 |
| Coherence (Failure) | 0.0892 |
| Gradient Magnitude (Success) | 0.0633 |
| Gradient Magnitude (Failure) | 0.0245 |
| Activation Separation | 0.1422 |
| Cosine Distance | 0.0016 |
| Clusters | 1,481 |
| Noise Fraction | 0.2452 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep3_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0558 |
| Coherence (Success) | 0.8699 |
| Coherence (Failure) | 0.0325 |
| Gradient Magnitude (Success) | 0.0579 |
| Gradient Magnitude (Failure) | 0.0240 |
| Activation Separation | 0.1597 |
| Cosine Distance | 0.0020 |
| Clusters | 1,459 |
| Noise Fraction | 0.2273 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep5_lower0.100_upper2.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1524 |
| Coherence (Success) | 0.8854 |
| Coherence (Failure) | -0.0334 |
| Gradient Magnitude (Success) | 0.0637 |
| Gradient Magnitude (Failure) | 0.0166 |
| Activation Separation | 0.1610 |
| Cosine Distance | 0.0020 |
| Clusters | 1,483 |
| Noise Fraction | 0.2349 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep70_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8386 |
| Coherence (Success) | 0.8958 |
| Coherence (Failure) | 0.6628 |
| Gradient Magnitude (Success) | 0.1948 |
| Gradient Magnitude (Failure) | 0.1332 |
| Activation Separation | 0.3748 |
| Cosine Distance | 0.0011 |
| Clusters | 1,549 |
| Noise Fraction | 0.2377 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep72_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6663 |
| Coherence (Success) | 0.9376 |
| Coherence (Failure) | 0.2143 |
| Gradient Magnitude (Success) | 0.2143 |
| Gradient Magnitude (Failure) | 0.1009 |
| Activation Separation | 0.3476 |
| Cosine Distance | 0.0009 |
| Clusters | 1,513 |
| Noise Fraction | 0.2482 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Coal, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep137_lower2.100_upper3.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8363 |
| Coherence (Success) | 0.8507 |
| Coherence (Failure) | 0.1744 |
| Gradient Magnitude (Success) | 0.2030 |
| Gradient Magnitude (Failure) | 0.1774 |
| Activation Separation | 0.7333 |
| Cosine Distance | 0.0004 |
| Clusters | 1,569 |
| Noise Fraction | 0.2771 |
| RSA Alignment (ρ) | -0.5222 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep205_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3268 |
| Coherence (Success) | 0.6106 |
| Coherence (Failure) | 0.3098 |
| Gradient Magnitude (Success) | 0.2305 |
| Gradient Magnitude (Failure) | 0.2030 |
| Activation Separation | 0.7545 |
| Cosine Distance | 0.0003 |
| Clusters | 1,677 |
| Noise Fraction | 0.2397 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep226_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2455 |
| Coherence (Success) | 0.4133 |
| Coherence (Failure) | 0.2624 |
| Gradient Magnitude (Success) | 0.1805 |
| Gradient Magnitude (Failure) | 0.1731 |
| Activation Separation | 0.8038 |
| Cosine Distance | 0.0003 |
| Clusters | 1,606 |
| Noise Fraction | 0.2411 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep282_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.6970 |
| Coherence (Success) | 0.0366 |
| Coherence (Failure) | 0.3191 |
| Gradient Magnitude (Success) | 0.0965 |
| Gradient Magnitude (Failure) | 0.1395 |
| Activation Separation | 1.0327 |
| Cosine Distance | 0.0005 |
| Clusters | 1,554 |
| Noise Fraction | 0.2690 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep427_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2407 |
| Coherence (Success) | 0.0987 |
| Coherence (Failure) | 0.1716 |
| Gradient Magnitude (Success) | 0.1039 |
| Gradient Magnitude (Failure) | 0.1808 |
| Activation Separation | 1.2253 |
| Cosine Distance | 0.0006 |
| Clusters | 1,340 |
| Noise Fraction | 0.3268 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep565_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0936 |
| Coherence (Success) | 0.0657 |
| Coherence (Failure) | -0.0218 |
| Gradient Magnitude (Success) | 0.1260 |
| Gradient Magnitude (Failure) | 0.1079 |
| Activation Separation | 1.2973 |
| Cosine Distance | 0.0003 |
| Clusters | 1,275 |
| Noise Fraction | 0.3006 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep849_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3059 |
| Coherence (Success) | 0.1412 |
| Coherence (Failure) | 0.2970 |
| Gradient Magnitude (Success) | 0.1376 |
| Gradient Magnitude (Failure) | 0.2572 |
| Activation Separation | 0.3048 |
| Cosine Distance | 0.0002 |
| Clusters | 1,599 |
| Noise Fraction | 0.2617 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep1109_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.5123 |
| Coherence (Success) | 0.1200 |
| Coherence (Failure) | 0.0111 |
| Gradient Magnitude (Success) | 0.1395 |
| Gradient Magnitude (Failure) | 0.1144 |
| Activation Separation | 1.0177 |
| Cosine Distance | 0.0009 |
| Clusters | 1,550 |
| Noise Fraction | 0.2505 |
| RSA Alignment (ρ) | 0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1132_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.6253 |
| Coherence (Success) | 0.1281 |
| Coherence (Failure) | 0.0551 |
| Gradient Magnitude (Success) | 0.1267 |
| Gradient Magnitude (Failure) | 0.0944 |
| Activation Separation | 0.3994 |
| Cosine Distance | 0.0005 |
| Clusters | 1,573 |
| Noise Fraction | 0.2555 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep1183_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2238 |
| Coherence (Success) | 0.0564 |
| Coherence (Failure) | 0.2479 |
| Gradient Magnitude (Success) | 0.0908 |
| Gradient Magnitude (Failure) | 0.2007 |
| Activation Separation | 0.7262 |
| Cosine Distance | 0.0005 |
| Clusters | 1,449 |
| Noise Fraction | 0.2940 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep1231_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5333 |
| Coherence (Success) | 0.6664 |
| Coherence (Failure) | 0.3583 |
| Gradient Magnitude (Success) | 0.2242 |
| Gradient Magnitude (Failure) | 0.2532 |
| Activation Separation | 0.3661 |
| Cosine Distance | 0.0008 |
| Clusters | 1,632 |
| Noise Fraction | 0.2655 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep1231_lower2.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7306 |
| Coherence (Success) | 0.5473 |
| Coherence (Failure) | 0.4439 |
| Gradient Magnitude (Success) | 0.1994 |
| Gradient Magnitude (Failure) | 0.2618 |
| Activation Separation | 0.3984 |
| Cosine Distance | 0.0010 |
| Clusters | 1,566 |
| Noise Fraction | 0.2733 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1421_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6836 |
| Coherence (Success) | 0.3698 |
| Coherence (Failure) | 0.4379 |
| Gradient Magnitude (Success) | 0.1655 |
| Gradient Magnitude (Failure) | 0.1967 |
| Activation Separation | 0.5300 |
| Cosine Distance | 0.0008 |
| Clusters | 1,460 |
| Noise Fraction | 0.2730 |
| RSA Alignment (ρ) | 0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## ep1712_lower3.100_upper4.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.6069 |
| Coherence (Success) | 0.1585 |
| Coherence (Failure) | 0.2836 |
| Gradient Magnitude (Success) | 0.1088 |
| Gradient Magnitude (Failure) | 0.2586 |
| Activation Separation | 0.8650 |
| Cosine Distance | 0.0028 |
| Clusters | 1,438 |
| Noise Fraction | 0.2623 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1992_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2352 |
| Coherence (Success) | 0.3447 |
| Coherence (Failure) | 0.0928 |
| Gradient Magnitude (Success) | 0.1667 |
| Gradient Magnitude (Failure) | 0.1086 |
| Activation Separation | 1.3179 |
| Cosine Distance | 0.0051 |
| Clusters | 1,323 |
| Noise Fraction | 0.2769 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2269_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2092 |
| Coherence (Success) | 0.1206 |
| Coherence (Failure) | 0.1001 |
| Gradient Magnitude (Success) | 0.1441 |
| Gradient Magnitude (Failure) | 0.1805 |
| Activation Separation | 1.5606 |
| Cosine Distance | 0.0067 |
| Clusters | 1,143 |
| Noise Fraction | 0.3145 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2277_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7587 |
| Coherence (Success) | 0.1215 |
| Coherence (Failure) | 0.2131 |
| Gradient Magnitude (Success) | 0.1325 |
| Gradient Magnitude (Failure) | 0.2365 |
| Activation Separation | 1.4662 |
| Cosine Distance | 0.0058 |
| Clusters | 1,150 |
| Noise Fraction | 0.3208 |
| RSA Alignment (ρ) | -0.3482 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2555_lower3.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3330 |
| Coherence (Success) | 0.1708 |
| Coherence (Failure) | 0.0343 |
| Gradient Magnitude (Success) | 0.1342 |
| Gradient Magnitude (Failure) | 0.0995 |
| Activation Separation | 1.2784 |
| Cosine Distance | 0.0043 |
| Clusters | 1,260 |
| Noise Fraction | 0.2991 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2830_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4920 |
| Coherence (Success) | 0.1928 |
| Coherence (Failure) | 0.0897 |
| Gradient Magnitude (Success) | 0.1568 |
| Gradient Magnitude (Failure) | 0.2417 |
| Activation Separation | 1.5901 |
| Cosine Distance | 0.0048 |
| Clusters | 1,601 |
| Noise Fraction | 0.2901 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep3022_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2583 |
| Coherence (Success) | 0.1700 |
| Coherence (Failure) | 0.0943 |
| Gradient Magnitude (Success) | 0.1399 |
| Gradient Magnitude (Failure) | 0.1878 |
| Activation Separation | 2.0432 |
| Cosine Distance | 0.0096 |
| Clusters | 1,547 |
| Noise Fraction | 0.3013 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3100_lower4.100_upper5.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2541 |
| Coherence (Success) | 0.1073 |
| Coherence (Failure) | 0.2457 |
| Gradient Magnitude (Success) | 0.1160 |
| Gradient Magnitude (Failure) | 0.2106 |
| Activation Separation | 1.9124 |
| Cosine Distance | 0.0102 |
| Clusters | 1,447 |
| Noise Fraction | 0.2991 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3360_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6352 |
| Coherence (Success) | 0.3062 |
| Coherence (Failure) | 0.2183 |
| Gradient Magnitude (Success) | 0.1610 |
| Gradient Magnitude (Failure) | 0.2430 |
| Activation Separation | 2.3048 |
| Cosine Distance | 0.0142 |
| Clusters | 1,303 |
| Noise Fraction | 0.3066 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3636_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6270 |
| Coherence (Success) | 0.1838 |
| Coherence (Failure) | 0.2249 |
| Gradient Magnitude (Success) | 0.1544 |
| Gradient Magnitude (Failure) | 0.3500 |
| Activation Separation | 2.3062 |
| Cosine Distance | 0.0109 |
| Clusters | 1,392 |
| Noise Fraction | 0.3201 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3910_lower4.100_upper6.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8274 |
| Coherence (Success) | 0.1649 |
| Coherence (Failure) | 0.0895 |
| Gradient Magnitude (Success) | 0.1879 |
| Gradient Magnitude (Failure) | 0.2032 |
| Activation Separation | 2.0474 |
| Cosine Distance | 0.0109 |
| Clusters | 1,320 |
| Noise Fraction | 0.3168 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4177_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3227 |
| Coherence (Success) | 0.0905 |
| Coherence (Failure) | 0.2229 |
| Gradient Magnitude (Success) | 0.1303 |
| Gradient Magnitude (Failure) | 0.2855 |
| Activation Separation | 2.6554 |
| Cosine Distance | 0.0133 |
| Clusters | 1,067 |
| Noise Fraction | 0.2993 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4445_lower4.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8568 |
| Coherence (Success) | 0.5159 |
| Coherence (Failure) | 0.3145 |
| Gradient Magnitude (Success) | 0.3389 |
| Gradient Magnitude (Failure) | 0.3697 |
| Activation Separation | 2.9103 |
| Cosine Distance | 0.0121 |
| Clusters | 1,441 |
| Noise Fraction | 0.3209 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4710_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5386 |
| Coherence (Success) | 0.1800 |
| Coherence (Failure) | 0.0001 |
| Gradient Magnitude (Success) | 0.1560 |
| Gradient Magnitude (Failure) | 0.1318 |
| Activation Separation | 2.6389 |
| Cosine Distance | 0.0122 |
| Clusters | 1,510 |
| Noise Fraction | 0.3064 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4979_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5537 |
| Coherence (Success) | 0.0566 |
| Coherence (Failure) | 0.0715 |
| Gradient Magnitude (Success) | 0.1283 |
| Gradient Magnitude (Failure) | 0.1647 |
| Activation Separation | 2.4009 |
| Cosine Distance | 0.0101 |
| Clusters | 1,258 |
| Noise Fraction | 0.3199 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5250_lower5.100_upper7.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2754 |
| Coherence (Success) | 0.1285 |
| Coherence (Failure) | 0.0357 |
| Gradient Magnitude (Success) | 0.1487 |
| Gradient Magnitude (Failure) | 0.1653 |
| Activation Separation | 2.8129 |
| Cosine Distance | 0.0137 |
| Clusters | 1,389 |
| Noise Fraction | 0.3108 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5511_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1192 |
| Coherence (Success) | 0.0417 |
| Coherence (Failure) | 0.0569 |
| Gradient Magnitude (Success) | 0.1264 |
| Gradient Magnitude (Failure) | 0.1914 |
| Activation Separation | 2.7841 |
| Cosine Distance | 0.0142 |
| Clusters | 1,354 |
| Noise Fraction | 0.3357 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5779_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3082 |
| Coherence (Success) | 0.1812 |
| Coherence (Failure) | 0.0715 |
| Gradient Magnitude (Success) | 0.1749 |
| Gradient Magnitude (Failure) | 0.1682 |
| Activation Separation | 3.3460 |
| Cosine Distance | 0.0166 |
| Clusters | 1,469 |
| Noise Fraction | 0.3510 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6046_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5727 |
| Coherence (Success) | 0.1324 |
| Coherence (Failure) | 0.1429 |
| Gradient Magnitude (Success) | 0.1545 |
| Gradient Magnitude (Failure) | 0.2062 |
| Activation Separation | 3.6576 |
| Cosine Distance | 0.0179 |
| Clusters | 1,367 |
| Noise Fraction | 0.3329 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6309_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8519 |
| Coherence (Success) | 0.2044 |
| Coherence (Failure) | 0.2499 |
| Gradient Magnitude (Success) | 0.2295 |
| Gradient Magnitude (Failure) | 0.3465 |
| Activation Separation | 3.0579 |
| Cosine Distance | 0.0124 |
| Clusters | 1,246 |
| Noise Fraction | 0.3313 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6573_lower5.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7529 |
| Coherence (Success) | 0.3179 |
| Coherence (Failure) | 0.1855 |
| Gradient Magnitude (Success) | 0.3161 |
| Gradient Magnitude (Failure) | 0.3274 |
| Activation Separation | 3.0088 |
| Cosine Distance | 0.0099 |
| Clusters | 1,209 |
| Noise Fraction | 0.3737 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6841_lower6.100_upper8.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4616 |
| Coherence (Success) | 0.0622 |
| Coherence (Failure) | -0.0050 |
| Gradient Magnitude (Success) | 0.1338 |
| Gradient Magnitude (Failure) | 0.1302 |
| Activation Separation | 3.8879 |
| Cosine Distance | 0.0126 |
| Clusters | 1,111 |
| Noise Fraction | 0.3639 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7092_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1164 |
| Coherence (Success) | 0.2350 |
| Coherence (Failure) | 0.0695 |
| Gradient Magnitude (Success) | 0.1721 |
| Gradient Magnitude (Failure) | 0.2057 |
| Activation Separation | 4.6512 |
| Cosine Distance | 0.0238 |
| Clusters | 1,365 |
| Noise Fraction | 0.3796 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7348_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2090 |
| Coherence (Success) | -0.0012 |
| Coherence (Failure) | 0.0702 |
| Gradient Magnitude (Success) | 0.1114 |
| Gradient Magnitude (Failure) | 0.1934 |
| Activation Separation | 3.8207 |
| Cosine Distance | 0.0166 |
| Clusters | 1,408 |
| Noise Fraction | 0.3612 |
| RSA Alignment (ρ) | 0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7583_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3682 |
| Coherence (Success) | 0.1224 |
| Coherence (Failure) | 0.0572 |
| Gradient Magnitude (Success) | 0.1579 |
| Gradient Magnitude (Failure) | 0.2144 |
| Activation Separation | 4.3710 |
| Cosine Distance | 0.0191 |
| Clusters | 1,579 |
| Noise Fraction | 0.3224 |
| RSA Alignment (ρ) | 0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7842_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5509 |
| Coherence (Success) | 0.1540 |
| Coherence (Failure) | 0.1007 |
| Gradient Magnitude (Success) | 0.1905 |
| Gradient Magnitude (Failure) | 0.2814 |
| Activation Separation | 4.3243 |
| Cosine Distance | 0.0196 |
| Clusters | 1,367 |
| Noise Fraction | 0.3463 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8093_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7458 |
| Coherence (Success) | 0.2470 |
| Coherence (Failure) | 0.2707 |
| Gradient Magnitude (Success) | 0.2152 |
| Gradient Magnitude (Failure) | 0.3958 |
| Activation Separation | 3.8701 |
| Cosine Distance | 0.0177 |
| Clusters | 1,323 |
| Noise Fraction | 0.3480 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8346_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2775 |
| Coherence (Success) | 0.1331 |
| Coherence (Failure) | 0.0135 |
| Gradient Magnitude (Success) | 0.1532 |
| Gradient Magnitude (Failure) | 0.1678 |
| Activation Separation | 3.7288 |
| Cosine Distance | 0.0146 |
| Clusters | 1,327 |
| Noise Fraction | 0.3495 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8588_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3904 |
| Coherence (Success) | 0.3103 |
| Coherence (Failure) | 0.0685 |
| Gradient Magnitude (Success) | 0.2405 |
| Gradient Magnitude (Failure) | 0.1904 |
| Activation Separation | 3.7089 |
| Cosine Distance | 0.0163 |
| Clusters | 1,316 |
| Noise Fraction | 0.3448 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep8831_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5173 |
| Coherence (Success) | 0.1821 |
| Coherence (Failure) | 0.0491 |
| Gradient Magnitude (Success) | 0.2379 |
| Gradient Magnitude (Failure) | 0.2252 |
| Activation Separation | 4.5067 |
| Cosine Distance | 0.0188 |
| Clusters | 1,391 |
| Noise Fraction | 0.3618 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9060_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2167 |
| Coherence (Success) | 0.0831 |
| Coherence (Failure) | 0.0574 |
| Gradient Magnitude (Success) | 0.1511 |
| Gradient Magnitude (Failure) | 0.1885 |
| Activation Separation | 3.7985 |
| Cosine Distance | 0.0139 |
| Clusters | 1,345 |
| Noise Fraction | 0.3280 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9301_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4519 |
| Coherence (Success) | 0.0952 |
| Coherence (Failure) | 0.1140 |
| Gradient Magnitude (Success) | 0.1643 |
| Gradient Magnitude (Failure) | 0.2615 |
| Activation Separation | 3.8382 |
| Cosine Distance | 0.0153 |
| Clusters | 1,376 |
| Noise Fraction | 0.3502 |
| RSA Alignment (ρ) | 0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9541_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6137 |
| Coherence (Success) | 0.1994 |
| Coherence (Failure) | 0.1235 |
| Gradient Magnitude (Success) | 0.1925 |
| Gradient Magnitude (Failure) | 0.2853 |
| Activation Separation | 3.8576 |
| Cosine Distance | 0.0226 |
| Clusters | 1,316 |
| Noise Fraction | 0.3334 |
| RSA Alignment (ρ) | 0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9778_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2327 |
| Coherence (Success) | 0.2726 |
| Coherence (Failure) | -0.0108 |
| Gradient Magnitude (Success) | 0.2084 |
| Gradient Magnitude (Failure) | 0.1395 |
| Activation Separation | 3.5791 |
| Cosine Distance | 0.0170 |
| Clusters | 1,258 |
| Noise Fraction | 0.3756 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep10013_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1180 |
| Coherence (Success) | 0.0425 |
| Coherence (Failure) | 0.1099 |
| Gradient Magnitude (Success) | 0.1340 |
| Gradient Magnitude (Failure) | 0.2057 |
| Activation Separation | 3.7062 |
| Cosine Distance | 0.0179 |
| Clusters | 1,434 |
| Noise Fraction | 0.3724 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep10100_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0347 |
| Coherence (Success) | 0.0827 |
| Coherence (Failure) | 0.0659 |
| Gradient Magnitude (Success) | 0.1459 |
| Gradient Magnitude (Failure) | 0.2077 |
| Activation Separation | 3.8153 |
| Cosine Distance | 0.0183 |
| Clusters | 1,377 |
| Noise Fraction | 0.3205 |
| RSA Alignment (ρ) | 0.3139 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10243_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4499 |
| Coherence (Success) | 0.1045 |
| Coherence (Failure) | 0.1261 |
| Gradient Magnitude (Success) | 0.1627 |
| Gradient Magnitude (Failure) | 0.2606 |
| Activation Separation | 4.0157 |
| Cosine Distance | 0.0182 |
| Clusters | 1,379 |
| Noise Fraction | 0.3686 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10466_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1376 |
| Coherence (Success) | 0.1015 |
| Coherence (Failure) | 0.0436 |
| Gradient Magnitude (Success) | 0.1671 |
| Gradient Magnitude (Failure) | 0.1857 |
| Activation Separation | 3.4282 |
| Cosine Distance | 0.0146 |
| Clusters | 1,136 |
| Noise Fraction | 0.3591 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10692_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3837 |
| Coherence (Success) | 0.1568 |
| Coherence (Failure) | 0.1311 |
| Gradient Magnitude (Success) | 0.1767 |
| Gradient Magnitude (Failure) | 0.2865 |
| Activation Separation | 4.5588 |
| Cosine Distance | 0.0266 |
| Clusters | 1,315 |
| Noise Fraction | 0.3465 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10912_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4597 |
| Coherence (Success) | 0.4211 |
| Coherence (Failure) | 0.0123 |
| Gradient Magnitude (Success) | 0.3347 |
| Gradient Magnitude (Failure) | 0.1790 |
| Activation Separation | 3.6391 |
| Cosine Distance | 0.0162 |
| Clusters | 1,364 |
| Noise Fraction | 0.3526 |
| RSA Alignment (ρ) | 0.2770 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11130_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5050 |
| Coherence (Success) | 0.3897 |
| Coherence (Failure) | 0.0238 |
| Gradient Magnitude (Success) | 0.2838 |
| Gradient Magnitude (Failure) | 0.1865 |
| Activation Separation | 3.7898 |
| Cosine Distance | 0.0213 |
| Clusters | 1,284 |
| Noise Fraction | 0.3418 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11348_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5721 |
| Coherence (Success) | 0.2985 |
| Coherence (Failure) | -0.0216 |
| Gradient Magnitude (Success) | 0.2602 |
| Gradient Magnitude (Failure) | 0.1419 |
| Activation Separation | 3.0583 |
| Cosine Distance | 0.0146 |
| Clusters | 1,302 |
| Noise Fraction | 0.3241 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11567_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2058 |
| Coherence (Success) | 0.1716 |
| Coherence (Failure) | 0.0592 |
| Gradient Magnitude (Success) | 0.1808 |
| Gradient Magnitude (Failure) | 0.1855 |
| Activation Separation | 3.7519 |
| Cosine Distance | 0.0184 |
| Clusters | 1,255 |
| Noise Fraction | 0.3543 |
| RSA Alignment (ρ) | 0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11772_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4628 |
| Coherence (Success) | 0.2134 |
| Coherence (Failure) | 0.0183 |
| Gradient Magnitude (Success) | 0.2206 |
| Gradient Magnitude (Failure) | 0.1841 |
| Activation Separation | 3.7909 |
| Cosine Distance | 0.0219 |
| Clusters | 1,304 |
| Noise Fraction | 0.3771 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11991_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4799 |
| Coherence (Success) | 0.2248 |
| Coherence (Failure) | 0.1221 |
| Gradient Magnitude (Success) | 0.1785 |
| Gradient Magnitude (Failure) | 0.2244 |
| Activation Separation | 3.2568 |
| Cosine Distance | 0.0182 |
| Clusters | 1,297 |
| Noise Fraction | 0.3663 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12213_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5054 |
| Coherence (Success) | 0.1452 |
| Coherence (Failure) | 0.0255 |
| Gradient Magnitude (Success) | 0.1848 |
| Gradient Magnitude (Failure) | 0.1576 |
| Activation Separation | 3.5546 |
| Cosine Distance | 0.0170 |
| Clusters | 1,213 |
| Noise Fraction | 0.3396 |
| RSA Alignment (ρ) | 0.0783 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12441_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5056 |
| Coherence (Success) | 0.2051 |
| Coherence (Failure) | 0.0486 |
| Gradient Magnitude (Success) | 0.1928 |
| Gradient Magnitude (Failure) | 0.1697 |
| Activation Separation | 4.1834 |
| Cosine Distance | 0.0271 |
| Clusters | 1,403 |
| Noise Fraction | 0.3566 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12667_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5964 |
| Coherence (Success) | 0.1801 |
| Coherence (Failure) | 0.0897 |
| Gradient Magnitude (Success) | 0.2037 |
| Gradient Magnitude (Failure) | 0.1945 |
| Activation Separation | 4.1909 |
| Cosine Distance | 0.0295 |
| Clusters | 1,350 |
| Noise Fraction | 0.3437 |
| RSA Alignment (ρ) | 0.2585 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12889_lower8.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1660 |
| Coherence (Success) | 0.1851 |
| Coherence (Failure) | -0.0072 |
| Gradient Magnitude (Success) | 0.2069 |
| Gradient Magnitude (Failure) | 0.1243 |
| Activation Separation | 3.9295 |
| Cosine Distance | 0.0233 |
| Clusters | 1,304 |
| Noise Fraction | 0.3374 |
| RSA Alignment (ρ) | 0.3139 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13104_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6246 |
| Coherence (Success) | 0.2978 |
| Coherence (Failure) | 0.0692 |
| Gradient Magnitude (Success) | 0.2485 |
| Gradient Magnitude (Failure) | 0.1825 |
| Activation Separation | 4.0554 |
| Cosine Distance | 0.0246 |
| Clusters | 1,367 |
| Noise Fraction | 0.3266 |
| RSA Alignment (ρ) | 0.2400 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13319_lower6.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5027 |
| Coherence (Success) | 0.5268 |
| Coherence (Failure) | 0.0096 |
| Gradient Magnitude (Success) | 0.4252 |
| Gradient Magnitude (Failure) | 0.1917 |
| Activation Separation | 3.9269 |
| Cosine Distance | 0.0194 |
| Clusters | 1,177 |
| Noise Fraction | 0.3318 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13532_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4356 |
| Coherence (Success) | 0.3842 |
| Coherence (Failure) | -0.0322 |
| Gradient Magnitude (Success) | 0.3061 |
| Gradient Magnitude (Failure) | 0.1490 |
| Activation Separation | 3.9119 |
| Cosine Distance | 0.0208 |
| Clusters | 1,115 |
| Noise Fraction | 0.3107 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13764_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2440 |
| Coherence (Success) | 0.2111 |
| Coherence (Failure) | 0.0085 |
| Gradient Magnitude (Success) | 0.2037 |
| Gradient Magnitude (Failure) | 0.1384 |
| Activation Separation | 3.1398 |
| Cosine Distance | 0.0193 |
| Clusters | 1,211 |
| Noise Fraction | 0.3235 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13983_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4572 |
| Coherence (Success) | 0.1820 |
| Coherence (Failure) | 0.0309 |
| Gradient Magnitude (Success) | 0.2139 |
| Gradient Magnitude (Failure) | 0.1846 |
| Activation Separation | 3.3702 |
| Cosine Distance | 0.0194 |
| Clusters | 1,198 |
| Noise Fraction | 0.3383 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14202_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6666 |
| Coherence (Success) | 0.1830 |
| Coherence (Failure) | 0.0726 |
| Gradient Magnitude (Success) | 0.2135 |
| Gradient Magnitude (Failure) | 0.2349 |
| Activation Separation | 4.1911 |
| Cosine Distance | 0.0307 |
| Clusters | 1,374 |
| Noise Fraction | 0.3525 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14417_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6258 |
| Coherence (Success) | 0.2430 |
| Coherence (Failure) | 0.0281 |
| Gradient Magnitude (Success) | 0.2499 |
| Gradient Magnitude (Failure) | 0.2292 |
| Activation Separation | 4.3558 |
| Cosine Distance | 0.0263 |
| Clusters | 1,357 |
| Noise Fraction | 0.3627 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14644_lower7.100_upper10.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1748 |
| Coherence (Success) | 0.1156 |
| Coherence (Failure) | 0.0277 |
| Gradient Magnitude (Success) | 0.1740 |
| Gradient Magnitude (Failure) | 0.1768 |
| Activation Separation | 4.4179 |
| Cosine Distance | 0.0355 |
| Clusters | 1,250 |
| Noise Fraction | 0.3337 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14867_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5542 |
| Coherence (Success) | 0.3163 |
| Coherence (Failure) | 0.1074 |
| Gradient Magnitude (Success) | 0.2498 |
| Gradient Magnitude (Failure) | 0.2327 |
| Activation Separation | 3.3373 |
| Cosine Distance | 0.0206 |
| Clusters | 1,174 |
| Noise Fraction | 0.3095 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14869_lower7.100_upper9.100

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4951 |
| Coherence (Success) | 0.2505 |
| Coherence (Failure) | 0.0539 |
| Gradient Magnitude (Success) | 0.2682 |
| Gradient Magnitude (Failure) | 0.1953 |
| Activation Separation | 3.9545 |
| Cosine Distance | 0.0277 |
| Clusters | 1,226 |
| Noise Fraction | 0.3330 |
| RSA Alignment (ρ) | 0.1664 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.3160 | 0.6958 | 0.0496 | 0.0426 | 0.0131 | 0.1137 | 0.0024 | -0.6547 |
| place_table @ ep1 | 1 | -0.1096 | 0.6262 | 0.1057 | 0.0422 | 0.0169 | 0.0932 | 0.0016 | -0.3928 |
| wake_up @ ep2 | 2 | 0.0401 | 0.9020 | 0.0637 | 0.0650 | 0.0217 | 0.1736 | 0.0024 | 0.1309 |
| collect_sapling @ ep3 | 3 | 0.2505 | 0.8920 | 0.0892 | 0.0633 | 0.0245 | 0.1422 | 0.0016 | nan |
| place_plant @ ep3 | 3 | 0.0558 | 0.8699 | 0.0325 | 0.0579 | 0.0240 | 0.1597 | 0.0020 | nan |
| collect_drink @ ep5 | 5 | 0.1524 | 0.8854 | -0.0334 | 0.0637 | 0.0166 | 0.1610 | 0.0020 | -0.6547 |
| eat_cow @ ep70 | 70 | 0.8386 | 0.8958 | 0.6628 | 0.1948 | 0.1332 | 0.3748 | 0.0011 | nan |
| make_wood_pickaxe @ ep72 | 72 | 0.6663 | 0.9376 | 0.2143 | 0.2143 | 0.1009 | 0.3476 | 0.0009 | -0.6547 |
| make_wood_sword @ ep137 | 137 | 0.8363 | 0.8507 | 0.1744 | 0.2030 | 0.1774 | 0.7333 | 0.0004 | -0.5222 |
| defeat_skeleton @ ep205 | 205 | 0.3268 | 0.6106 | 0.3098 | 0.2305 | 0.2030 | 0.7545 | 0.0003 | -0.1741 |
| defeat_zombie @ ep226 | 226 | 0.2455 | 0.4133 | 0.2624 | 0.1805 | 0.1731 | 0.8038 | 0.0003 | -0.2094 |
| collect_stone @ ep427 | 427 | 0.2407 | 0.0987 | 0.1716 | 0.1039 | 0.1808 | 1.2253 | 0.0006 | -0.2443 |
| place_stone @ ep1109 | 1,109 | -0.5123 | 0.1200 | 0.0111 | 0.1395 | 0.1144 | 1.0177 | 0.0009 | 0.1741 |
| make_stone_pickaxe @ ep1183 | 1,183 | 0.2238 | 0.0564 | 0.2479 | 0.0908 | 0.2007 | 0.7262 | 0.0005 | 0.0000 |
| collect_coal @ ep1231 | 1,231 | 0.5333 | 0.6664 | 0.3583 | 0.2242 | 0.2532 | 0.3661 | 0.0008 | -0.3140 |
| make_stone_sword @ ep1231 | 1,231 | 0.7306 | 0.5473 | 0.4439 | 0.1994 | 0.2618 | 0.3984 | 0.0010 | -0.2094 |
| place_furnace @ ep2277 | 2,277 | 0.7587 | 0.1215 | 0.2131 | 0.1325 | 0.2365 | 1.4662 | 0.0058 | -0.3482 |
| eat_plant @ ep3022 | 3,022 | -0.2583 | 0.1700 | 0.0943 | 0.1399 | 0.1878 | 2.0432 | 0.0096 | -0.1047 |
| collect_iron @ ep10100 | 10,100 | -0.0347 | 0.0827 | 0.0659 | 0.1459 | 0.2077 | 3.8153 | 0.0183 | 0.3139 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic_step50000_ep282 | 282 | -0.6970 | 0.0366 | 0.3191 | 0.0965 | 0.1395 | 1.0327 | 0.0005 | -0.2094 |
| periodic_step100000_ep565 | 565 | 0.0936 | 0.0657 | -0.0218 | 0.1260 | 0.1079 | 1.2973 | 0.0003 | 0.0000 |
| periodic_step150000_ep849 | 849 | 0.3059 | 0.1412 | 0.2970 | 0.1376 | 0.2572 | 0.3048 | 0.0002 | -0.1745 |
| periodic_step200000_ep1132 | 1,132 | -0.6253 | 0.1281 | 0.0551 | 0.1267 | 0.0944 | 0.3994 | 0.0005 | -0.1741 |
| periodic_step250000_ep1421 | 1,421 | 0.6836 | 0.3698 | 0.4379 | 0.1655 | 0.1967 | 0.5300 | 0.0008 | 0.6547 |
| periodic_step300000_ep1712 | 1,712 | -0.6069 | 0.1585 | 0.2836 | 0.1088 | 0.2586 | 0.8650 | 0.0028 | -0.2443 |
| periodic_step350000_ep1992 | 1,992 | -0.2352 | 0.3447 | 0.0928 | 0.1667 | 0.1086 | 1.3179 | 0.0051 | -0.3838 |
| periodic_step400000_ep2269 | 2,269 | 0.2092 | 0.1206 | 0.1001 | 0.1441 | 0.1805 | 1.5606 | 0.0067 | -0.3838 |
| periodic_step450000_ep2555 | 2,555 | 0.3330 | 0.1708 | 0.0343 | 0.1342 | 0.0995 | 1.2784 | 0.0043 | -0.3489 |
| periodic_step500000_ep2830 | 2,830 | 0.4920 | 0.1928 | 0.0897 | 0.1568 | 0.2417 | 1.5901 | 0.0048 | -0.2443 |
| periodic_step550000_ep3100 | 3,100 | -0.2541 | 0.1073 | 0.2457 | 0.1160 | 0.2106 | 1.9124 | 0.0102 | -0.1745 |
| periodic_step600000_ep3360 | 3,360 | 0.6352 | 0.3062 | 0.2183 | 0.1610 | 0.2430 | 2.3048 | 0.0142 | 0.1108 |
| periodic_step650000_ep3636 | 3,636 | 0.6270 | 0.1838 | 0.2249 | 0.1544 | 0.3500 | 2.3062 | 0.0109 | -0.0739 |
| periodic_step700000_ep3910 | 3,910 | 0.8274 | 0.1649 | 0.0895 | 0.1879 | 0.2032 | 2.0474 | 0.0109 | -0.1396 |
| periodic_step750000_ep4177 | 4,177 | 0.3227 | 0.0905 | 0.2229 | 0.1303 | 0.2855 | 2.6554 | 0.0133 | -0.1396 |
| periodic_step800000_ep4445 | 4,445 | 0.8568 | 0.5159 | 0.3145 | 0.3389 | 0.3697 | 2.9103 | 0.0121 | -0.1292 |
| periodic_step850000_ep4710 | 4,710 | 0.5386 | 0.1800 | 0.0001 | 0.1560 | 0.1318 | 2.6389 | 0.0122 | -0.1108 |
| periodic_step900000_ep4979 | 4,979 | 0.5537 | 0.0566 | 0.0715 | 0.1283 | 0.1647 | 2.4009 | 0.0101 | -0.1047 |
| periodic_step950000_ep5250 | 5,250 | 0.2754 | 0.1285 | 0.0357 | 0.1487 | 0.1653 | 2.8129 | 0.0137 | -0.1477 |
| periodic_step1000000_ep5511 | 5,511 | -0.1192 | 0.0417 | 0.0569 | 0.1264 | 0.1914 | 2.7841 | 0.0142 | -0.0185 |
| periodic_step1050000_ep5779 | 5,779 | 0.3082 | 0.1812 | 0.0715 | 0.1749 | 0.1682 | 3.3460 | 0.0166 | -0.1047 |
| periodic_step1100000_ep6046 | 6,046 | 0.5727 | 0.1324 | 0.1429 | 0.1545 | 0.2062 | 3.6576 | 0.0179 | 0.0000 |
| periodic_step1150000_ep6309 | 6,309 | 0.8519 | 0.2044 | 0.2499 | 0.2295 | 0.3465 | 3.0579 | 0.0124 | -0.0349 |
| periodic_step1200000_ep6573 | 6,573 | 0.7529 | 0.3179 | 0.1855 | 0.3161 | 0.3274 | 3.0088 | 0.0099 | -0.0349 |
| periodic_step1250000_ep6841 | 6,841 | 0.4616 | 0.0622 | -0.0050 | 0.1338 | 0.1302 | 3.8879 | 0.0126 | -0.0349 |
| periodic_step1300000_ep7092 | 7,092 | 0.1164 | 0.2350 | 0.0695 | 0.1721 | 0.2057 | 4.6512 | 0.0238 | -0.0349 |
| periodic_step1350000_ep7348 | 7,348 | -0.2090 | -0.0012 | 0.0702 | 0.1114 | 0.1934 | 3.8207 | 0.0166 | 0.0698 |
| periodic_step1400000_ep7583 | 7,583 | 0.3682 | 0.1224 | 0.0572 | 0.1579 | 0.2144 | 4.3710 | 0.0191 | 0.1745 |
| periodic_step1450000_ep7842 | 7,842 | 0.5509 | 0.1540 | 0.1007 | 0.1905 | 0.2814 | 4.3243 | 0.0196 | 0.0000 |
| periodic_step1500000_ep8093 | 8,093 | 0.7458 | 0.2470 | 0.2707 | 0.2152 | 0.3958 | 3.8701 | 0.0177 | 0.1292 |
| periodic_step1550000_ep8346 | 8,346 | 0.2775 | 0.1331 | 0.0135 | 0.1532 | 0.1678 | 3.7288 | 0.0146 | 0.1396 |
| periodic_step1600000_ep8588 | 8,588 | 0.3904 | 0.3103 | 0.0685 | 0.2405 | 0.1904 | 3.7089 | 0.0163 | 0.2094 |
| periodic_step1650000_ep8831 | 8,831 | 0.5173 | 0.1821 | 0.0491 | 0.2379 | 0.2252 | 4.5067 | 0.0188 | 0.2094 |
| periodic_step1700000_ep9060 | 9,060 | 0.2167 | 0.0831 | 0.0574 | 0.1511 | 0.1885 | 3.7985 | 0.0139 | 0.2443 |
| periodic_step1750000_ep9301 | 9,301 | 0.4519 | 0.0952 | 0.1140 | 0.1643 | 0.2615 | 3.8382 | 0.0153 | 0.3489 |
| periodic_step1800000_ep9541 | 9,541 | 0.6137 | 0.1994 | 0.1235 | 0.1925 | 0.2853 | 3.8576 | 0.0226 | 0.3140 |
| periodic_step1850000_ep9778 | 9,778 | 0.2327 | 0.2726 | -0.0108 | 0.2084 | 0.1395 | 3.5791 | 0.0170 | 0.2094 |
| periodic_step1900000_ep10013 | 10,013 | 0.1180 | 0.0425 | 0.1099 | 0.1340 | 0.2057 | 3.7062 | 0.0179 | 0.2216 |
| periodic_step1950000_ep10243 | 10,243 | 0.4499 | 0.1045 | 0.1261 | 0.1627 | 0.2606 | 4.0157 | 0.0182 | 0.2216 |
| periodic_step2000000_ep10466 | 10,466 | 0.1376 | 0.1015 | 0.0436 | 0.1671 | 0.1857 | 3.4282 | 0.0146 | 0.2216 |
| periodic_step2050000_ep10692 | 10,692 | 0.3837 | 0.1568 | 0.1311 | 0.1767 | 0.2865 | 4.5588 | 0.0266 | 0.1662 |
| periodic_step2100000_ep10912 | 10,912 | 0.4597 | 0.4211 | 0.0123 | 0.3347 | 0.1790 | 3.6391 | 0.0162 | 0.2770 |
| periodic_step2150000_ep11130 | 11,130 | 0.5050 | 0.3897 | 0.0238 | 0.2838 | 0.1865 | 3.7898 | 0.0213 | 0.0554 |
| periodic_step2200000_ep11348 | 11,348 | 0.5721 | 0.2985 | -0.0216 | 0.2602 | 0.1419 | 3.0583 | 0.0146 | 0.2216 |
| periodic_step2250000_ep11567 | 11,567 | 0.2058 | 0.1716 | 0.0592 | 0.1808 | 0.1855 | 3.7519 | 0.0184 | 0.2216 |
| periodic_step2300000_ep11772 | 11,772 | 0.4628 | 0.2134 | 0.0183 | 0.2206 | 0.1841 | 3.7909 | 0.0219 | 0.1477 |
| periodic_step2350000_ep11991 | 11,991 | 0.4799 | 0.2248 | 0.1221 | 0.1785 | 0.2244 | 3.2568 | 0.0182 | 0.1662 |
| periodic_step2400000_ep12213 | 12,213 | 0.5054 | 0.1452 | 0.0255 | 0.1848 | 0.1576 | 3.5546 | 0.0170 | 0.0783 |
| periodic_step2450000_ep12441 | 12,441 | 0.5056 | 0.2051 | 0.0486 | 0.1928 | 0.1697 | 4.1834 | 0.0271 | 0.1477 |
| periodic_step2500000_ep12667 | 12,667 | 0.5964 | 0.1801 | 0.0897 | 0.2037 | 0.1945 | 4.1909 | 0.0295 | 0.2585 |
| periodic_step2550000_ep12889 | 12,889 | 0.1660 | 0.1851 | -0.0072 | 0.2069 | 0.1243 | 3.9295 | 0.0233 | 0.3139 |
| periodic_step2600000_ep13104 | 13,104 | 0.6246 | 0.2978 | 0.0692 | 0.2485 | 0.1825 | 4.0554 | 0.0246 | 0.2400 |
| periodic_step2650000_ep13319 | 13,319 | 0.5027 | 0.5268 | 0.0096 | 0.4252 | 0.1917 | 3.9269 | 0.0194 | 0.1846 |
| periodic_step2700000_ep13532 | 13,532 | 0.4356 | 0.3842 | -0.0322 | 0.3061 | 0.1490 | 3.9119 | 0.0208 | 0.1846 |
| periodic_step2750000_ep13764 | 13,764 | 0.2440 | 0.2111 | 0.0085 | 0.2037 | 0.1384 | 3.1398 | 0.0193 | 0.1846 |
| periodic_step2800000_ep13983 | 13,983 | 0.4572 | 0.1820 | 0.0309 | 0.2139 | 0.1846 | 3.3702 | 0.0194 | 0.1477 |
| periodic_step2850000_ep14202 | 14,202 | 0.6666 | 0.1830 | 0.0726 | 0.2135 | 0.2349 | 4.1911 | 0.0307 | 0.0923 |
| periodic_step2900000_ep14417 | 14,417 | 0.6258 | 0.2430 | 0.0281 | 0.2499 | 0.2292 | 4.3558 | 0.0263 | 0.1662 |
| periodic_step2950000_ep14644 | 14,644 | 0.1748 | 0.1156 | 0.0277 | 0.1740 | 0.1768 | 4.4179 | 0.0355 | 0.1292 |
| periodic_step3000000_ep14867 | 14,867 | 0.5542 | 0.3163 | 0.1074 | 0.2498 | 0.2327 | 3.3373 | 0.0206 | 0.1108 |
| final_step3000320_ep14869 | 14,869 | 0.4951 | 0.2505 | 0.0539 | 0.2682 | 0.1953 | 3.9545 | 0.0277 | 0.1664 |
