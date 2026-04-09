# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 5  
**Total episodes:** 15,231  
**Experiment root:** `ppo_experiment_root\seed_5`  
**Generated:** 2026-04-01 00:33

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_sapling @ ep1 | 1 | 0.2285 | 0.7095 | 0.0236 | 0.0606 | 0.0188 | 0.1420 | 0.0011 | nan |
| wake_up @ ep2 | 2 | -0.2861 | 0.8346 | 0.0250 | 0.0706 | 0.0228 | 0.1930 | 0.0018 | nan |
| place_plant @ ep3 | 3 | -0.0498 | 0.8716 | 0.0384 | 0.0748 | 0.0332 | 0.1568 | 0.0012 | nan |
| collect_wood @ ep8 | 8 | -0.0309 | 0.8385 | -0.0012 | 0.0694 | 0.0267 | 0.1908 | 0.0017 | nan |
| collect_drink @ ep23 | 23 | 0.0880 | 0.7844 | -0.0067 | 0.1580 | 0.0628 | 0.3121 | 0.0012 | -0.3928 |
| place_table @ ep57 | 57 | 0.4846 | 0.9138 | 0.0497 | 0.2283 | 0.1204 | 0.4980 | 0.0008 | -0.6547 |
| eat_cow @ ep133 | 133 | 0.7626 | 0.8838 | 0.3137 | 0.2437 | 0.1796 | 1.0110 | 0.0003 | -0.6547 |
| make_wood_pickaxe @ ep147 | 147 | 0.7666 | 0.7879 | 0.5306 | 0.2325 | 0.2059 | 1.0028 | 0.0004 | -0.4352 |
| defeat_zombie @ ep161 | 161 | 0.7840 | 0.7155 | 0.8183 | 0.2081 | 0.2359 | 0.7284 | 0.0002 | -0.6547 |
| collect_stone @ ep283 | 283 | 0.4509 | 0.1545 | 0.6227 | 0.1639 | 0.2277 | 0.9447 | 0.0008 | -0.0923 |
| make_stone_sword @ ep283 | 283 | 0.6875 | 0.5747 | 0.5702 | 0.1807 | 0.2677 | 0.7808 | 0.0007 | -0.3419 |
| make_wood_sword @ ep283 | 283 | 0.4690 | 0.4487 | 0.3400 | 0.1638 | 0.2073 | 1.4129 | 0.0021 | -0.1396 |
| Step 50,000 | 285 | 0.6430 | 0.7487 | 0.6791 | 0.2379 | 0.2231 | 1.1513 | 0.0012 | -0.1396 |
| place_furnace @ ep380 | 380 | 0.1329 | 0.2398 | 0.4418 | 0.1285 | 0.3105 | 1.0658 | 0.0010 | -0.3838 |
| place_stone @ ep380 | 380 | 0.0623 | 0.2914 | 0.2823 | 0.1478 | 0.2053 | 0.8133 | 0.0007 | -0.2443 |
| defeat_skeleton @ ep385 | 385 | 0.4707 | 0.5568 | 0.3199 | 0.1625 | 0.1742 | 0.7807 | 0.0008 | -0.0870 |
| make_stone_pickaxe @ ep544 | 544 | 0.8718 | 0.5520 | 0.4920 | 0.1814 | 0.3412 | 0.4698 | 0.0006 | -0.5222 |
| Step 100,000 | 564 | 0.7020 | 0.2640 | 0.6595 | 0.1756 | 0.3821 | 0.4471 | 0.0003 | -0.2791 |
| collect_coal @ ep639 | 639 | 0.8250 | 0.2361 | 0.3704 | 0.1761 | 0.2420 | 0.7305 | 0.0008 | -0.3489 |
| Step 150,000 | 817 | 0.0588 | 0.1890 | 0.5741 | 0.1385 | 0.3366 | 1.1315 | 0.0025 | -0.2585 |
| Step 200,000 | 1,086 | -0.0576 | 0.0230 | 0.1458 | 0.0863 | 0.1476 | 1.0043 | 0.0016 | -0.1477 |
| Step 250,000 | 1,363 | -0.0436 | 0.0740 | 0.2503 | 0.0856 | 0.2155 | 1.1205 | 0.0030 | -0.1047 |
| Step 300,000 | 1,643 | -0.3088 | 0.0643 | 0.4504 | 0.1010 | 0.3347 | 1.3665 | 0.0045 | -0.0489 |
| Step 350,000 | 1,893 | -0.5539 | 0.0505 | 0.1317 | 0.0992 | 0.1556 | 1.9456 | 0.0093 | -0.1662 |
| Step 400,000 | 2,148 | 0.4455 | 0.0027 | 0.3122 | 0.1113 | 0.3153 | 2.0377 | 0.0082 | -0.1396 |
| Step 450,000 | 2,414 | 0.7672 | 0.3021 | 0.4587 | 0.2550 | 0.4345 | 2.3782 | 0.0092 | -0.1396 |
| Step 500,000 | 2,678 | 0.8062 | 0.4061 | 0.5938 | 0.2687 | 0.4329 | 2.0459 | 0.0099 | -0.1108 |
| Step 550,000 | 2,947 | 0.4398 | 0.0704 | 0.3101 | 0.0988 | 0.3047 | 2.6197 | 0.0169 | -0.1047 |
| eat_plant @ ep2952 | 2,952 | 0.5536 | 0.1992 | 0.1855 | 0.1671 | 0.2254 | 2.6510 | 0.0152 | -0.0698 |
| Step 600,000 | 3,206 | 0.8416 | 0.1860 | 0.3365 | 0.1855 | 0.3951 | 2.1607 | 0.0088 | -0.0698 |
| Step 650,000 | 3,466 | 0.8012 | 0.4258 | 0.2601 | 0.3375 | 0.3035 | 2.6724 | 0.0128 | -0.1292 |
| Step 700,000 | 3,719 | 0.8110 | 0.3474 | 0.2892 | 0.2679 | 0.3801 | 3.1496 | 0.0135 | 0.0000 |
| Step 750,000 | 3,985 | 0.3733 | 0.1015 | 0.1103 | 0.1561 | 0.1851 | 3.5728 | 0.0201 | -0.0923 |
| Step 800,000 | 4,249 | 0.5192 | 0.1492 | 0.0593 | 0.1860 | 0.1601 | 3.8540 | 0.0211 | -0.1047 |
| Step 850,000 | 4,508 | 0.7085 | 0.2113 | 0.2100 | 0.2251 | 0.3189 | 3.9934 | 0.0207 | -0.1047 |
| Step 900,000 | 4,766 | 0.3150 | 0.1365 | 0.1231 | 0.1574 | 0.1718 | 3.7347 | 0.0212 | -0.1477 |
| Step 950,000 | 5,023 | 0.2805 | 0.0603 | 0.1997 | 0.1275 | 0.2465 | 3.9566 | 0.0239 | -0.1108 |
| Step 1,000,000 | 5,292 | -0.3021 | 0.1617 | 0.0382 | 0.1433 | 0.1635 | 3.6544 | 0.0188 | -0.0554 |
| Step 1,050,000 | 5,552 | 0.3350 | 0.0204 | 0.0033 | 0.1279 | 0.1327 | 4.1473 | 0.0203 | -0.0923 |
| Step 1,100,000 | 5,810 | -0.2222 | 0.0888 | 0.0423 | 0.1551 | 0.1574 | 4.1853 | 0.0239 | -0.0979 |
| Step 1,150,000 | 6,068 | 0.6748 | 0.1951 | 0.1335 | 0.2194 | 0.2141 | 3.9912 | 0.0195 | -0.1292 |
| Step 1,200,000 | 6,322 | 0.6045 | 0.3903 | 0.2201 | 0.3170 | 0.2774 | 4.9523 | 0.0325 | -0.0881 |
| Step 1,250,000 | 6,588 | 0.6276 | 0.2592 | 0.1137 | 0.2416 | 0.2223 | 4.2575 | 0.0289 | -0.0185 |
| Step 1,300,000 | 6,844 | 0.2816 | 0.1821 | 0.0433 | 0.2410 | 0.1976 | 5.1227 | 0.0279 | -0.0554 |
| collect_iron @ ep7035 | 7,035 | 0.5413 | 0.1825 | 0.0339 | 0.2330 | 0.1795 | 4.9955 | 0.0229 | -0.1108 |
| Step 1,350,000 | 7,102 | 0.7617 | 0.2395 | 0.1279 | 0.2598 | 0.3459 | 4.6139 | 0.0237 | -0.0979 |
| Step 1,400,000 | 7,352 | 0.6632 | 0.3976 | 0.1059 | 0.3252 | 0.2339 | 4.6025 | 0.0254 | -0.0369 |
| Step 1,450,000 | 7,608 | 0.4790 | 0.1482 | 0.0763 | 0.1724 | 0.2125 | 4.7943 | 0.0310 | 0.0000 |
| Step 1,500,000 | 7,863 | 0.3144 | 0.2789 | 0.0204 | 0.2799 | 0.1882 | 5.1464 | 0.0360 | 0.0185 |
| Step 1,550,000 | 8,124 | 0.3199 | 0.0807 | -0.0121 | 0.1594 | 0.1414 | 5.1828 | 0.0277 | -0.0489 |
| Step 1,600,000 | 8,383 | 0.5840 | 0.1735 | 0.0719 | 0.2376 | 0.2220 | 6.4477 | 0.0396 | 0.0739 |
| Step 1,650,000 | 8,649 | 0.7092 | 0.1973 | 0.0916 | 0.2171 | 0.2399 | 5.6640 | 0.0351 | 0.0098 |
| Step 1,700,000 | 8,909 | 0.3486 | 0.1417 | 0.0696 | 0.1656 | 0.1772 | 4.8808 | 0.0265 | 0.0000 |
| Step 1,750,000 | 9,170 | 0.3113 | 0.1400 | 0.0928 | 0.1946 | 0.2348 | 5.0194 | 0.0308 | 0.1468 |
| Step 1,800,000 | 9,436 | -0.0411 | 0.1005 | 0.0154 | 0.1610 | 0.1588 | 5.1849 | 0.0324 | 0.0923 |
| Step 1,850,000 | 9,692 | 0.1124 | -0.0020 | 0.0449 | 0.1132 | 0.1741 | 5.1972 | 0.0333 | 0.1468 |
| Step 1,900,000 | 9,951 | 0.6571 | 0.2665 | 0.1193 | 0.2363 | 0.2234 | 4.6832 | 0.0324 | 0.2153 |
| Step 1,950,000 | 10,213 | 0.3781 | 0.2936 | 0.0270 | 0.2920 | 0.2106 | 5.3713 | 0.0355 | 0.1664 |
| Step 2,000,000 | 10,485 | 0.6159 | 0.4124 | 0.1053 | 0.3701 | 0.2851 | 4.3908 | 0.0230 | 0.1664 |
| Step 2,050,000 | 10,751 | 0.3449 | 0.2729 | 0.0816 | 0.2794 | 0.2342 | 4.6554 | 0.0237 | 0.0489 |
| Step 2,100,000 | 11,014 | 0.4902 | 0.2685 | 0.0659 | 0.2712 | 0.2604 | 4.8120 | 0.0243 | 0.1108 |
| Step 2,150,000 | 11,276 | 0.4136 | 0.1874 | 0.0371 | 0.2317 | 0.2187 | 4.4584 | 0.0206 | 0.0098 |
| Step 2,200,000 | 11,535 | 0.0274 | 0.1456 | 0.0175 | 0.2087 | 0.1711 | 4.0036 | 0.0211 | 0.0587 |
| Step 2,250,000 | 11,784 | 0.4499 | 0.1708 | 0.0302 | 0.2133 | 0.1759 | 3.7930 | 0.0215 | 0.1108 |
| Step 2,300,000 | 12,028 | 0.4270 | 0.1706 | 0.0087 | 0.1893 | 0.1918 | 4.2440 | 0.0282 | 0.0185 |
| Step 2,350,000 | 12,278 | -0.0926 | 0.0422 | 0.0094 | 0.1292 | 0.1533 | 3.5725 | 0.0219 | 0.0739 |
| Step 2,400,000 | 12,516 | 0.3993 | 0.1120 | 0.0530 | 0.1511 | 0.1812 | 3.9931 | 0.0240 | 0.1292 |
| Step 2,450,000 | 12,752 | -0.0803 | 0.1385 | 0.0927 | 0.1836 | 0.2017 | 4.8248 | 0.0450 | 0.0739 |
| Step 2,500,000 | 13,000 | 0.1091 | 0.1060 | 0.0220 | 0.1621 | 0.1694 | 4.2569 | 0.0315 | 0.1662 |
| Step 2,550,000 | 13,243 | 0.6466 | 0.1785 | 0.0705 | 0.2091 | 0.2181 | 3.2905 | 0.0197 | 0.0739 |
| Step 2,600,000 | 13,474 | 0.4653 | 0.1465 | 0.0246 | 0.2015 | 0.1960 | 3.5224 | 0.0164 | 0.1662 |
| Step 2,650,000 | 13,696 | 0.4105 | 0.1997 | 0.0847 | 0.2092 | 0.2502 | 4.2310 | 0.0229 | 0.1396 |
| Step 2,700,000 | 13,909 | 0.1610 | 0.1143 | 0.0170 | 0.1543 | 0.1630 | 3.9097 | 0.0232 | 0.2443 |
| Step 2,750,000 | 14,123 | 0.2467 | 0.1271 | 0.0441 | 0.1557 | 0.1981 | 3.2311 | 0.0171 | 0.2443 |
| Step 2,800,000 | 14,351 | 0.4715 | 0.2429 | 0.0158 | 0.2005 | 0.1820 | 3.9061 | 0.0208 | 0.2443 |
| Step 2,850,000 | 14,566 | 0.4724 | 0.1640 | 0.0830 | 0.1732 | 0.2168 | 2.8758 | 0.0133 | 0.2443 |
| Step 2,900,000 | 14,782 | 0.6988 | 0.1172 | 0.0565 | 0.1993 | 0.2217 | 4.3761 | 0.0241 | 0.2791 |
| Step 2,950,000 | 15,005 | 0.7108 | 0.3235 | 0.0595 | 0.3050 | 0.2094 | 3.0038 | 0.0131 | 0.2443 |
| Step 3,000,000 | 15,229 | 0.4370 | 0.0710 | 0.0505 | 0.1692 | 0.2101 | 3.3461 | 0.0174 | 0.2443 |
| Final | 15,231 | 0.3122 | 0.1374 | 0.0616 | 0.1674 | 0.1821 | 3.0837 | 0.0144 | 0.2791 |

---

## collect_sapling_ep1_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2285 |
| Coherence (Success) | 0.7095 |
| Coherence (Failure) | 0.0236 |
| Gradient Magnitude (Success) | 0.0606 |
| Gradient Magnitude (Failure) | 0.0188 |
| Activation Separation | 0.1420 |
| Cosine Distance | 0.0011 |
| Clusters | 1,464 |
| Noise Fraction | 0.2606 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Skeleton, Wood, Wood Pickaxe |

---

## wake_up_ep2_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2861 |
| Coherence (Success) | 0.8346 |
| Coherence (Failure) | 0.0250 |
| Gradient Magnitude (Success) | 0.0706 |
| Gradient Magnitude (Failure) | 0.0228 |
| Activation Separation | 0.1930 |
| Cosine Distance | 0.0018 |
| Clusters | 1,583 |
| Noise Fraction | 0.2660 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0498 |
| Coherence (Success) | 0.8716 |
| Coherence (Failure) | 0.0384 |
| Gradient Magnitude (Success) | 0.0748 |
| Gradient Magnitude (Failure) | 0.0332 |
| Activation Separation | 0.1568 |
| Cosine Distance | 0.0012 |
| Clusters | 1,546 |
| Noise Fraction | 0.2559 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_wood_ep8_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0309 |
| Coherence (Success) | 0.8385 |
| Coherence (Failure) | -0.0012 |
| Gradient Magnitude (Success) | 0.0694 |
| Gradient Magnitude (Failure) | 0.0267 |
| Activation Separation | 0.1908 |
| Cosine Distance | 0.0017 |
| Clusters | 1,513 |
| Noise Fraction | 0.2539 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep23_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0880 |
| Coherence (Success) | 0.7844 |
| Coherence (Failure) | -0.0067 |
| Gradient Magnitude (Success) | 0.1580 |
| Gradient Magnitude (Failure) | 0.0628 |
| Activation Separation | 0.3121 |
| Cosine Distance | 0.0012 |
| Clusters | 1,549 |
| Noise Fraction | 0.2419 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## place_table_ep57_lower2.900_upper4.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4846 |
| Coherence (Success) | 0.9138 |
| Coherence (Failure) | 0.0497 |
| Gradient Magnitude (Success) | 0.2283 |
| Gradient Magnitude (Failure) | 0.1204 |
| Activation Separation | 0.4980 |
| Cosine Distance | 0.0008 |
| Clusters | 1,529 |
| Noise Fraction | 0.2459 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep133_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7626 |
| Coherence (Success) | 0.8838 |
| Coherence (Failure) | 0.3137 |
| Gradient Magnitude (Success) | 0.2437 |
| Gradient Magnitude (Failure) | 0.1796 |
| Activation Separation | 1.0110 |
| Cosine Distance | 0.0003 |
| Clusters | 1,290 |
| Noise Fraction | 0.2830 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep147_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7666 |
| Coherence (Success) | 0.7879 |
| Coherence (Failure) | 0.5306 |
| Gradient Magnitude (Success) | 0.2325 |
| Gradient Magnitude (Failure) | 0.2059 |
| Activation Separation | 1.0028 |
| Cosine Distance | 0.0004 |
| Clusters | 1,334 |
| Noise Fraction | 0.2964 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep161_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7840 |
| Coherence (Success) | 0.7155 |
| Coherence (Failure) | 0.8183 |
| Gradient Magnitude (Success) | 0.2081 |
| Gradient Magnitude (Failure) | 0.2359 |
| Activation Separation | 0.7284 |
| Cosine Distance | 0.0002 |
| Clusters | 1,291 |
| Noise Fraction | 0.3006 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep283_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4509 |
| Coherence (Success) | 0.1545 |
| Coherence (Failure) | 0.6227 |
| Gradient Magnitude (Success) | 0.1639 |
| Gradient Magnitude (Failure) | 0.2277 |
| Activation Separation | 0.9447 |
| Cosine Distance | 0.0008 |
| Clusters | 1,655 |
| Noise Fraction | 0.2658 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep283_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6875 |
| Coherence (Success) | 0.5747 |
| Coherence (Failure) | 0.5702 |
| Gradient Magnitude (Success) | 0.1807 |
| Gradient Magnitude (Failure) | 0.2677 |
| Activation Separation | 0.7808 |
| Cosine Distance | 0.0007 |
| Clusters | 1,612 |
| Noise Fraction | 0.2567 |
| RSA Alignment (ρ) | -0.3419 |
| RSA Stimuli (5) | Coal, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep283_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4690 |
| Coherence (Success) | 0.4487 |
| Coherence (Failure) | 0.3400 |
| Gradient Magnitude (Success) | 0.1638 |
| Gradient Magnitude (Failure) | 0.2073 |
| Activation Separation | 1.4129 |
| Cosine Distance | 0.0021 |
| Clusters | 1,684 |
| Noise Fraction | 0.2545 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep285_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6430 |
| Coherence (Success) | 0.7487 |
| Coherence (Failure) | 0.6791 |
| Gradient Magnitude (Success) | 0.2379 |
| Gradient Magnitude (Failure) | 0.2231 |
| Activation Separation | 1.1513 |
| Cosine Distance | 0.0012 |
| Clusters | 1,617 |
| Noise Fraction | 0.2747 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep380_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1329 |
| Coherence (Success) | 0.2398 |
| Coherence (Failure) | 0.4418 |
| Gradient Magnitude (Success) | 0.1285 |
| Gradient Magnitude (Failure) | 0.3105 |
| Activation Separation | 1.0658 |
| Cosine Distance | 0.0010 |
| Clusters | 1,701 |
| Noise Fraction | 0.2658 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep380_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0623 |
| Coherence (Success) | 0.2914 |
| Coherence (Failure) | 0.2823 |
| Gradient Magnitude (Success) | 0.1478 |
| Gradient Magnitude (Failure) | 0.2053 |
| Activation Separation | 0.8133 |
| Cosine Distance | 0.0007 |
| Clusters | 1,660 |
| Noise Fraction | 0.2587 |
| RSA Alignment (ρ) | -0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep385_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4707 |
| Coherence (Success) | 0.5568 |
| Coherence (Failure) | 0.3199 |
| Gradient Magnitude (Success) | 0.1625 |
| Gradient Magnitude (Failure) | 0.1742 |
| Activation Separation | 0.7807 |
| Cosine Distance | 0.0008 |
| Clusters | 1,691 |
| Noise Fraction | 0.2679 |
| RSA Alignment (ρ) | -0.0870 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep544_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8718 |
| Coherence (Success) | 0.5520 |
| Coherence (Failure) | 0.4920 |
| Gradient Magnitude (Success) | 0.1814 |
| Gradient Magnitude (Failure) | 0.3412 |
| Activation Separation | 0.4698 |
| Cosine Distance | 0.0006 |
| Clusters | 1,818 |
| Noise Fraction | 0.2417 |
| RSA Alignment (ρ) | -0.5222 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep564_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7020 |
| Coherence (Success) | 0.2640 |
| Coherence (Failure) | 0.6595 |
| Gradient Magnitude (Success) | 0.1756 |
| Gradient Magnitude (Failure) | 0.3821 |
| Activation Separation | 0.4471 |
| Cosine Distance | 0.0003 |
| Clusters | 1,800 |
| Noise Fraction | 0.2484 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep639_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8250 |
| Coherence (Success) | 0.2361 |
| Coherence (Failure) | 0.3704 |
| Gradient Magnitude (Success) | 0.1761 |
| Gradient Magnitude (Failure) | 0.2420 |
| Activation Separation | 0.7305 |
| Cosine Distance | 0.0008 |
| Clusters | 1,957 |
| Noise Fraction | 0.2504 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep817_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0588 |
| Coherence (Success) | 0.1890 |
| Coherence (Failure) | 0.5741 |
| Gradient Magnitude (Success) | 0.1385 |
| Gradient Magnitude (Failure) | 0.3366 |
| Activation Separation | 1.1315 |
| Cosine Distance | 0.0025 |
| Clusters | 1,856 |
| Noise Fraction | 0.2433 |
| RSA Alignment (ρ) | -0.2585 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1086_lower3.900_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0576 |
| Coherence (Success) | 0.0230 |
| Coherence (Failure) | 0.1458 |
| Gradient Magnitude (Success) | 0.0863 |
| Gradient Magnitude (Failure) | 0.1476 |
| Activation Separation | 1.0043 |
| Cosine Distance | 0.0016 |
| Clusters | 1,746 |
| Noise Fraction | 0.2438 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1363_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0436 |
| Coherence (Success) | 0.0740 |
| Coherence (Failure) | 0.2503 |
| Gradient Magnitude (Success) | 0.0856 |
| Gradient Magnitude (Failure) | 0.2155 |
| Activation Separation | 1.1205 |
| Cosine Distance | 0.0030 |
| Clusters | 1,856 |
| Noise Fraction | 0.2384 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1643_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3088 |
| Coherence (Success) | 0.0643 |
| Coherence (Failure) | 0.4504 |
| Gradient Magnitude (Success) | 0.1010 |
| Gradient Magnitude (Failure) | 0.3347 |
| Activation Separation | 1.3665 |
| Cosine Distance | 0.0045 |
| Clusters | 1,945 |
| Noise Fraction | 0.2416 |
| RSA Alignment (ρ) | -0.0489 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1893_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.5539 |
| Coherence (Success) | 0.0505 |
| Coherence (Failure) | 0.1317 |
| Gradient Magnitude (Success) | 0.0992 |
| Gradient Magnitude (Failure) | 0.1556 |
| Activation Separation | 1.9456 |
| Cosine Distance | 0.0093 |
| Clusters | 1,695 |
| Noise Fraction | 0.2570 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2148_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4455 |
| Coherence (Success) | 0.0027 |
| Coherence (Failure) | 0.3122 |
| Gradient Magnitude (Success) | 0.1113 |
| Gradient Magnitude (Failure) | 0.3153 |
| Activation Separation | 2.0377 |
| Cosine Distance | 0.0082 |
| Clusters | 1,824 |
| Noise Fraction | 0.2446 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2414_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7672 |
| Coherence (Success) | 0.3021 |
| Coherence (Failure) | 0.4587 |
| Gradient Magnitude (Success) | 0.2550 |
| Gradient Magnitude (Failure) | 0.4345 |
| Activation Separation | 2.3782 |
| Cosine Distance | 0.0092 |
| Clusters | 1,309 |
| Noise Fraction | 0.3269 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2678_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8062 |
| Coherence (Success) | 0.4061 |
| Coherence (Failure) | 0.5938 |
| Gradient Magnitude (Success) | 0.2687 |
| Gradient Magnitude (Failure) | 0.4329 |
| Activation Separation | 2.0459 |
| Cosine Distance | 0.0099 |
| Clusters | 1,646 |
| Noise Fraction | 0.2834 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2947_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4398 |
| Coherence (Success) | 0.0704 |
| Coherence (Failure) | 0.3101 |
| Gradient Magnitude (Success) | 0.0988 |
| Gradient Magnitude (Failure) | 0.3047 |
| Activation Separation | 2.6197 |
| Cosine Distance | 0.0169 |
| Clusters | 1,581 |
| Noise Fraction | 0.2904 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep2952_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5536 |
| Coherence (Success) | 0.1992 |
| Coherence (Failure) | 0.1855 |
| Gradient Magnitude (Success) | 0.1671 |
| Gradient Magnitude (Failure) | 0.2254 |
| Activation Separation | 2.6510 |
| Cosine Distance | 0.0152 |
| Clusters | 1,508 |
| Noise Fraction | 0.2756 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3206_lower5.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8416 |
| Coherence (Success) | 0.1860 |
| Coherence (Failure) | 0.3365 |
| Gradient Magnitude (Success) | 0.1855 |
| Gradient Magnitude (Failure) | 0.3951 |
| Activation Separation | 2.1607 |
| Cosine Distance | 0.0088 |
| Clusters | 1,477 |
| Noise Fraction | 0.3074 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3466_lower5.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8012 |
| Coherence (Success) | 0.4258 |
| Coherence (Failure) | 0.2601 |
| Gradient Magnitude (Success) | 0.3375 |
| Gradient Magnitude (Failure) | 0.3035 |
| Activation Separation | 2.6724 |
| Cosine Distance | 0.0128 |
| Clusters | 1,603 |
| Noise Fraction | 0.2922 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3719_lower5.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8110 |
| Coherence (Success) | 0.3474 |
| Coherence (Failure) | 0.2892 |
| Gradient Magnitude (Success) | 0.2679 |
| Gradient Magnitude (Failure) | 0.3801 |
| Activation Separation | 3.1496 |
| Cosine Distance | 0.0135 |
| Clusters | 1,400 |
| Noise Fraction | 0.3124 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3985_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3733 |
| Coherence (Success) | 0.1015 |
| Coherence (Failure) | 0.1103 |
| Gradient Magnitude (Success) | 0.1561 |
| Gradient Magnitude (Failure) | 0.1851 |
| Activation Separation | 3.5728 |
| Cosine Distance | 0.0201 |
| Clusters | 1,417 |
| Noise Fraction | 0.3276 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4249_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5192 |
| Coherence (Success) | 0.1492 |
| Coherence (Failure) | 0.0593 |
| Gradient Magnitude (Success) | 0.1860 |
| Gradient Magnitude (Failure) | 0.1601 |
| Activation Separation | 3.8540 |
| Cosine Distance | 0.0211 |
| Clusters | 1,398 |
| Noise Fraction | 0.3039 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4508_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7085 |
| Coherence (Success) | 0.2113 |
| Coherence (Failure) | 0.2100 |
| Gradient Magnitude (Success) | 0.2251 |
| Gradient Magnitude (Failure) | 0.3189 |
| Activation Separation | 3.9934 |
| Cosine Distance | 0.0207 |
| Clusters | 1,306 |
| Noise Fraction | 0.3665 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4766_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3150 |
| Coherence (Success) | 0.1365 |
| Coherence (Failure) | 0.1231 |
| Gradient Magnitude (Success) | 0.1574 |
| Gradient Magnitude (Failure) | 0.1718 |
| Activation Separation | 3.7347 |
| Cosine Distance | 0.0212 |
| Clusters | 1,419 |
| Noise Fraction | 0.3184 |
| RSA Alignment (ρ) | -0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5023_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2805 |
| Coherence (Success) | 0.0603 |
| Coherence (Failure) | 0.1997 |
| Gradient Magnitude (Success) | 0.1275 |
| Gradient Magnitude (Failure) | 0.2465 |
| Activation Separation | 3.9566 |
| Cosine Distance | 0.0239 |
| Clusters | 1,467 |
| Noise Fraction | 0.3239 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5292_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3021 |
| Coherence (Success) | 0.1617 |
| Coherence (Failure) | 0.0382 |
| Gradient Magnitude (Success) | 0.1433 |
| Gradient Magnitude (Failure) | 0.1635 |
| Activation Separation | 3.6544 |
| Cosine Distance | 0.0188 |
| Clusters | 1,426 |
| Noise Fraction | 0.3186 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5552_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3350 |
| Coherence (Success) | 0.0204 |
| Coherence (Failure) | 0.0033 |
| Gradient Magnitude (Success) | 0.1279 |
| Gradient Magnitude (Failure) | 0.1327 |
| Activation Separation | 4.1473 |
| Cosine Distance | 0.0203 |
| Clusters | 1,552 |
| Noise Fraction | 0.3243 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5810_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2222 |
| Coherence (Success) | 0.0888 |
| Coherence (Failure) | 0.0423 |
| Gradient Magnitude (Success) | 0.1551 |
| Gradient Magnitude (Failure) | 0.1574 |
| Activation Separation | 4.1853 |
| Cosine Distance | 0.0239 |
| Clusters | 1,463 |
| Noise Fraction | 0.3215 |
| RSA Alignment (ρ) | -0.0979 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6068_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6748 |
| Coherence (Success) | 0.1951 |
| Coherence (Failure) | 0.1335 |
| Gradient Magnitude (Success) | 0.2194 |
| Gradient Magnitude (Failure) | 0.2141 |
| Activation Separation | 3.9912 |
| Cosine Distance | 0.0195 |
| Clusters | 1,331 |
| Noise Fraction | 0.3402 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6322_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6045 |
| Coherence (Success) | 0.3903 |
| Coherence (Failure) | 0.2201 |
| Gradient Magnitude (Success) | 0.3170 |
| Gradient Magnitude (Failure) | 0.2774 |
| Activation Separation | 4.9523 |
| Cosine Distance | 0.0325 |
| Clusters | 1,424 |
| Noise Fraction | 0.3433 |
| RSA Alignment (ρ) | -0.0881 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6588_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6276 |
| Coherence (Success) | 0.2592 |
| Coherence (Failure) | 0.1137 |
| Gradient Magnitude (Success) | 0.2416 |
| Gradient Magnitude (Failure) | 0.2223 |
| Activation Separation | 4.2575 |
| Cosine Distance | 0.0289 |
| Clusters | 1,535 |
| Noise Fraction | 0.3242 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6844_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2816 |
| Coherence (Success) | 0.1821 |
| Coherence (Failure) | 0.0433 |
| Gradient Magnitude (Success) | 0.2410 |
| Gradient Magnitude (Failure) | 0.1976 |
| Activation Separation | 5.1227 |
| Cosine Distance | 0.0279 |
| Clusters | 1,450 |
| Noise Fraction | 0.3267 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep7035_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5413 |
| Coherence (Success) | 0.1825 |
| Coherence (Failure) | 0.0339 |
| Gradient Magnitude (Success) | 0.2330 |
| Gradient Magnitude (Failure) | 0.1795 |
| Activation Separation | 4.9955 |
| Cosine Distance | 0.0229 |
| Clusters | 1,350 |
| Noise Fraction | 0.3307 |
| RSA Alignment (ρ) | -0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7102_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7617 |
| Coherence (Success) | 0.2395 |
| Coherence (Failure) | 0.1279 |
| Gradient Magnitude (Success) | 0.2598 |
| Gradient Magnitude (Failure) | 0.3459 |
| Activation Separation | 4.6139 |
| Cosine Distance | 0.0237 |
| Clusters | 1,290 |
| Noise Fraction | 0.3039 |
| RSA Alignment (ρ) | -0.0979 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7352_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6632 |
| Coherence (Success) | 0.3976 |
| Coherence (Failure) | 0.1059 |
| Gradient Magnitude (Success) | 0.3252 |
| Gradient Magnitude (Failure) | 0.2339 |
| Activation Separation | 4.6025 |
| Cosine Distance | 0.0254 |
| Clusters | 1,214 |
| Noise Fraction | 0.2564 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7608_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4790 |
| Coherence (Success) | 0.1482 |
| Coherence (Failure) | 0.0763 |
| Gradient Magnitude (Success) | 0.1724 |
| Gradient Magnitude (Failure) | 0.2125 |
| Activation Separation | 4.7943 |
| Cosine Distance | 0.0310 |
| Clusters | 1,510 |
| Noise Fraction | 0.3023 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7863_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3144 |
| Coherence (Success) | 0.2789 |
| Coherence (Failure) | 0.0204 |
| Gradient Magnitude (Success) | 0.2799 |
| Gradient Magnitude (Failure) | 0.1882 |
| Activation Separation | 5.1464 |
| Cosine Distance | 0.0360 |
| Clusters | 1,448 |
| Noise Fraction | 0.3156 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8124_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3199 |
| Coherence (Success) | 0.0807 |
| Coherence (Failure) | -0.0121 |
| Gradient Magnitude (Success) | 0.1594 |
| Gradient Magnitude (Failure) | 0.1414 |
| Activation Separation | 5.1828 |
| Cosine Distance | 0.0277 |
| Clusters | 1,408 |
| Noise Fraction | 0.3135 |
| RSA Alignment (ρ) | -0.0489 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8383_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5840 |
| Coherence (Success) | 0.1735 |
| Coherence (Failure) | 0.0719 |
| Gradient Magnitude (Success) | 0.2376 |
| Gradient Magnitude (Failure) | 0.2220 |
| Activation Separation | 6.4477 |
| Cosine Distance | 0.0396 |
| Clusters | 1,419 |
| Noise Fraction | 0.3185 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8649_lower7.000_upper10.300

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7092 |
| Coherence (Success) | 0.1973 |
| Coherence (Failure) | 0.0916 |
| Gradient Magnitude (Success) | 0.2171 |
| Gradient Magnitude (Failure) | 0.2399 |
| Activation Separation | 5.6640 |
| Cosine Distance | 0.0351 |
| Clusters | 1,435 |
| Noise Fraction | 0.3111 |
| RSA Alignment (ρ) | 0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8909_lower7.900_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3486 |
| Coherence (Success) | 0.1417 |
| Coherence (Failure) | 0.0696 |
| Gradient Magnitude (Success) | 0.1656 |
| Gradient Magnitude (Failure) | 0.1772 |
| Activation Separation | 4.8808 |
| Cosine Distance | 0.0265 |
| Clusters | 1,398 |
| Noise Fraction | 0.3348 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9170_lower7.450_upper10.300

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3113 |
| Coherence (Success) | 0.1400 |
| Coherence (Failure) | 0.0928 |
| Gradient Magnitude (Success) | 0.1946 |
| Gradient Magnitude (Failure) | 0.2348 |
| Activation Separation | 5.0194 |
| Cosine Distance | 0.0308 |
| Clusters | 1,381 |
| Noise Fraction | 0.3524 |
| RSA Alignment (ρ) | 0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9436_lower8.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0411 |
| Coherence (Success) | 0.1005 |
| Coherence (Failure) | 0.0154 |
| Gradient Magnitude (Success) | 0.1610 |
| Gradient Magnitude (Failure) | 0.1588 |
| Activation Separation | 5.1849 |
| Cosine Distance | 0.0324 |
| Clusters | 1,263 |
| Noise Fraction | 0.3240 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9692_lower7.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1124 |
| Coherence (Success) | -0.0020 |
| Coherence (Failure) | 0.0449 |
| Gradient Magnitude (Success) | 0.1132 |
| Gradient Magnitude (Failure) | 0.1741 |
| Activation Separation | 5.1972 |
| Cosine Distance | 0.0333 |
| Clusters | 1,405 |
| Noise Fraction | 0.3309 |
| RSA Alignment (ρ) | 0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9951_lower7.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6571 |
| Coherence (Success) | 0.2665 |
| Coherence (Failure) | 0.1193 |
| Gradient Magnitude (Success) | 0.2363 |
| Gradient Magnitude (Failure) | 0.2234 |
| Activation Separation | 4.6832 |
| Cosine Distance | 0.0324 |
| Clusters | 1,254 |
| Noise Fraction | 0.3463 |
| RSA Alignment (ρ) | 0.2153 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10213_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3781 |
| Coherence (Success) | 0.2936 |
| Coherence (Failure) | 0.0270 |
| Gradient Magnitude (Success) | 0.2920 |
| Gradient Magnitude (Failure) | 0.2106 |
| Activation Separation | 5.3713 |
| Cosine Distance | 0.0355 |
| Clusters | 1,231 |
| Noise Fraction | 0.3400 |
| RSA Alignment (ρ) | 0.1664 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10485_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6159 |
| Coherence (Success) | 0.4124 |
| Coherence (Failure) | 0.1053 |
| Gradient Magnitude (Success) | 0.3701 |
| Gradient Magnitude (Failure) | 0.2851 |
| Activation Separation | 4.3908 |
| Cosine Distance | 0.0230 |
| Clusters | 1,230 |
| Noise Fraction | 0.3258 |
| RSA Alignment (ρ) | 0.1664 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10751_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3449 |
| Coherence (Success) | 0.2729 |
| Coherence (Failure) | 0.0816 |
| Gradient Magnitude (Success) | 0.2794 |
| Gradient Magnitude (Failure) | 0.2342 |
| Activation Separation | 4.6554 |
| Cosine Distance | 0.0237 |
| Clusters | 1,275 |
| Noise Fraction | 0.3317 |
| RSA Alignment (ρ) | 0.0489 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11014_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4902 |
| Coherence (Success) | 0.2685 |
| Coherence (Failure) | 0.0659 |
| Gradient Magnitude (Success) | 0.2712 |
| Gradient Magnitude (Failure) | 0.2604 |
| Activation Separation | 4.8120 |
| Cosine Distance | 0.0243 |
| Clusters | 1,282 |
| Noise Fraction | 0.3368 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11276_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4136 |
| Coherence (Success) | 0.1874 |
| Coherence (Failure) | 0.0371 |
| Gradient Magnitude (Success) | 0.2317 |
| Gradient Magnitude (Failure) | 0.2187 |
| Activation Separation | 4.4584 |
| Cosine Distance | 0.0206 |
| Clusters | 1,143 |
| Noise Fraction | 0.3435 |
| RSA Alignment (ρ) | 0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11535_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0274 |
| Coherence (Success) | 0.1456 |
| Coherence (Failure) | 0.0175 |
| Gradient Magnitude (Success) | 0.2087 |
| Gradient Magnitude (Failure) | 0.1711 |
| Activation Separation | 4.0036 |
| Cosine Distance | 0.0211 |
| Clusters | 1,268 |
| Noise Fraction | 0.3090 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11784_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4499 |
| Coherence (Success) | 0.1708 |
| Coherence (Failure) | 0.0302 |
| Gradient Magnitude (Success) | 0.2133 |
| Gradient Magnitude (Failure) | 0.1759 |
| Activation Separation | 3.7930 |
| Cosine Distance | 0.0215 |
| Clusters | 1,264 |
| Noise Fraction | 0.3615 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12028_lower7.450_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4270 |
| Coherence (Success) | 0.1706 |
| Coherence (Failure) | 0.0087 |
| Gradient Magnitude (Success) | 0.1893 |
| Gradient Magnitude (Failure) | 0.1918 |
| Activation Separation | 4.2440 |
| Cosine Distance | 0.0282 |
| Clusters | 1,286 |
| Noise Fraction | 0.3226 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12278_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0926 |
| Coherence (Success) | 0.0422 |
| Coherence (Failure) | 0.0094 |
| Gradient Magnitude (Success) | 0.1292 |
| Gradient Magnitude (Failure) | 0.1533 |
| Activation Separation | 3.5725 |
| Cosine Distance | 0.0219 |
| Clusters | 1,230 |
| Noise Fraction | 0.3506 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12516_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3993 |
| Coherence (Success) | 0.1120 |
| Coherence (Failure) | 0.0530 |
| Gradient Magnitude (Success) | 0.1511 |
| Gradient Magnitude (Failure) | 0.1812 |
| Activation Separation | 3.9931 |
| Cosine Distance | 0.0240 |
| Clusters | 1,203 |
| Noise Fraction | 0.3362 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12752_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0803 |
| Coherence (Success) | 0.1385 |
| Coherence (Failure) | 0.0927 |
| Gradient Magnitude (Success) | 0.1836 |
| Gradient Magnitude (Failure) | 0.2017 |
| Activation Separation | 4.8248 |
| Cosine Distance | 0.0450 |
| Clusters | 1,286 |
| Noise Fraction | 0.3429 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13000_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1091 |
| Coherence (Success) | 0.1060 |
| Coherence (Failure) | 0.0220 |
| Gradient Magnitude (Success) | 0.1621 |
| Gradient Magnitude (Failure) | 0.1694 |
| Activation Separation | 4.2569 |
| Cosine Distance | 0.0315 |
| Clusters | 1,139 |
| Noise Fraction | 0.3321 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13243_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6466 |
| Coherence (Success) | 0.1785 |
| Coherence (Failure) | 0.0705 |
| Gradient Magnitude (Success) | 0.2091 |
| Gradient Magnitude (Failure) | 0.2181 |
| Activation Separation | 3.2905 |
| Cosine Distance | 0.0197 |
| Clusters | 1,162 |
| Noise Fraction | 0.3453 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13474_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4653 |
| Coherence (Success) | 0.1465 |
| Coherence (Failure) | 0.0246 |
| Gradient Magnitude (Success) | 0.2015 |
| Gradient Magnitude (Failure) | 0.1960 |
| Activation Separation | 3.5224 |
| Cosine Distance | 0.0164 |
| Clusters | 1,221 |
| Noise Fraction | 0.3490 |
| RSA Alignment (ρ) | 0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13696_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4105 |
| Coherence (Success) | 0.1997 |
| Coherence (Failure) | 0.0847 |
| Gradient Magnitude (Success) | 0.2092 |
| Gradient Magnitude (Failure) | 0.2502 |
| Activation Separation | 4.2310 |
| Cosine Distance | 0.0229 |
| Clusters | 1,260 |
| Noise Fraction | 0.3498 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep13909_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1610 |
| Coherence (Success) | 0.1143 |
| Coherence (Failure) | 0.0170 |
| Gradient Magnitude (Success) | 0.1543 |
| Gradient Magnitude (Failure) | 0.1630 |
| Activation Separation | 3.9097 |
| Cosine Distance | 0.0232 |
| Clusters | 1,210 |
| Noise Fraction | 0.3131 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14123_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2467 |
| Coherence (Success) | 0.1271 |
| Coherence (Failure) | 0.0441 |
| Gradient Magnitude (Success) | 0.1557 |
| Gradient Magnitude (Failure) | 0.1981 |
| Activation Separation | 3.2311 |
| Cosine Distance | 0.0171 |
| Clusters | 1,328 |
| Noise Fraction | 0.3532 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14351_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4715 |
| Coherence (Success) | 0.2429 |
| Coherence (Failure) | 0.0158 |
| Gradient Magnitude (Success) | 0.2005 |
| Gradient Magnitude (Failure) | 0.1820 |
| Activation Separation | 3.9061 |
| Cosine Distance | 0.0208 |
| Clusters | 1,235 |
| Noise Fraction | 0.2988 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14566_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4724 |
| Coherence (Success) | 0.1640 |
| Coherence (Failure) | 0.0830 |
| Gradient Magnitude (Success) | 0.1732 |
| Gradient Magnitude (Failure) | 0.2168 |
| Activation Separation | 2.8758 |
| Cosine Distance | 0.0133 |
| Clusters | 1,155 |
| Noise Fraction | 0.3058 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep14782_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6988 |
| Coherence (Success) | 0.1172 |
| Coherence (Failure) | 0.0565 |
| Gradient Magnitude (Success) | 0.1993 |
| Gradient Magnitude (Failure) | 0.2217 |
| Activation Separation | 4.3761 |
| Cosine Distance | 0.0241 |
| Clusters | 1,445 |
| Noise Fraction | 0.3063 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep15005_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7108 |
| Coherence (Success) | 0.3235 |
| Coherence (Failure) | 0.0595 |
| Gradient Magnitude (Success) | 0.3050 |
| Gradient Magnitude (Failure) | 0.2094 |
| Activation Separation | 3.0038 |
| Cosine Distance | 0.0131 |
| Clusters | 1,127 |
| Noise Fraction | 0.3116 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep15229_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4370 |
| Coherence (Success) | 0.0710 |
| Coherence (Failure) | 0.0505 |
| Gradient Magnitude (Success) | 0.1692 |
| Gradient Magnitude (Failure) | 0.2101 |
| Activation Separation | 3.3461 |
| Cosine Distance | 0.0174 |
| Clusters | 1,201 |
| Noise Fraction | 0.3624 |
| RSA Alignment (ρ) | 0.2443 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep15231_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3122 |
| Coherence (Success) | 0.1374 |
| Coherence (Failure) | 0.0616 |
| Gradient Magnitude (Success) | 0.1674 |
| Gradient Magnitude (Failure) | 0.1821 |
| Activation Separation | 3.0837 |
| Cosine Distance | 0.0144 |
| Clusters | 1,061 |
| Noise Fraction | 0.2662 |
| RSA Alignment (ρ) | 0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_sapling @ ep1 | 1 | 0.2285 | 0.7095 | 0.0236 | 0.0606 | 0.0188 | 0.1420 | 0.0011 | nan |
| wake_up @ ep2 | 2 | -0.2861 | 0.8346 | 0.0250 | 0.0706 | 0.0228 | 0.1930 | 0.0018 | nan |
| place_plant @ ep3 | 3 | -0.0498 | 0.8716 | 0.0384 | 0.0748 | 0.0332 | 0.1568 | 0.0012 | nan |
| collect_wood @ ep8 | 8 | -0.0309 | 0.8385 | -0.0012 | 0.0694 | 0.0267 | 0.1908 | 0.0017 | nan |
| collect_drink @ ep23 | 23 | 0.0880 | 0.7844 | -0.0067 | 0.1580 | 0.0628 | 0.3121 | 0.0012 | -0.3928 |
| place_table @ ep57 | 57 | 0.4846 | 0.9138 | 0.0497 | 0.2283 | 0.1204 | 0.4980 | 0.0008 | -0.6547 |
| eat_cow @ ep133 | 133 | 0.7626 | 0.8838 | 0.3137 | 0.2437 | 0.1796 | 1.0110 | 0.0003 | -0.6547 |
| make_wood_pickaxe @ ep147 | 147 | 0.7666 | 0.7879 | 0.5306 | 0.2325 | 0.2059 | 1.0028 | 0.0004 | -0.4352 |
| defeat_zombie @ ep161 | 161 | 0.7840 | 0.7155 | 0.8183 | 0.2081 | 0.2359 | 0.7284 | 0.0002 | -0.6547 |
| collect_stone @ ep283 | 283 | 0.4509 | 0.1545 | 0.6227 | 0.1639 | 0.2277 | 0.9447 | 0.0008 | -0.0923 |
| make_stone_sword @ ep283 | 283 | 0.6875 | 0.5747 | 0.5702 | 0.1807 | 0.2677 | 0.7808 | 0.0007 | -0.3419 |
| make_wood_sword @ ep283 | 283 | 0.4690 | 0.4487 | 0.3400 | 0.1638 | 0.2073 | 1.4129 | 0.0021 | -0.1396 |
| place_furnace @ ep380 | 380 | 0.1329 | 0.2398 | 0.4418 | 0.1285 | 0.3105 | 1.0658 | 0.0010 | -0.3838 |
| place_stone @ ep380 | 380 | 0.0623 | 0.2914 | 0.2823 | 0.1478 | 0.2053 | 0.8133 | 0.0007 | -0.2443 |
| defeat_skeleton @ ep385 | 385 | 0.4707 | 0.5568 | 0.3199 | 0.1625 | 0.1742 | 0.7807 | 0.0008 | -0.0870 |
| make_stone_pickaxe @ ep544 | 544 | 0.8718 | 0.5520 | 0.4920 | 0.1814 | 0.3412 | 0.4698 | 0.0006 | -0.5222 |
| collect_coal @ ep639 | 639 | 0.8250 | 0.2361 | 0.3704 | 0.1761 | 0.2420 | 0.7305 | 0.0008 | -0.3489 |
| eat_plant @ ep2952 | 2,952 | 0.5536 | 0.1992 | 0.1855 | 0.1671 | 0.2254 | 2.6510 | 0.0152 | -0.0698 |
| collect_iron @ ep7035 | 7,035 | 0.5413 | 0.1825 | 0.0339 | 0.2330 | 0.1795 | 4.9955 | 0.0229 | -0.1108 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 285 | 0.6430 | 0.7487 | 0.6791 | 0.2379 | 0.2231 | 1.1513 | 0.0012 | -0.1396 |
| Step 100,000 | 564 | 0.7020 | 0.2640 | 0.6595 | 0.1756 | 0.3821 | 0.4471 | 0.0003 | -0.2791 |
| Step 150,000 | 817 | 0.0588 | 0.1890 | 0.5741 | 0.1385 | 0.3366 | 1.1315 | 0.0025 | -0.2585 |
| Step 200,000 | 1,086 | -0.0576 | 0.0230 | 0.1458 | 0.0863 | 0.1476 | 1.0043 | 0.0016 | -0.1477 |
| Step 250,000 | 1,363 | -0.0436 | 0.0740 | 0.2503 | 0.0856 | 0.2155 | 1.1205 | 0.0030 | -0.1047 |
| Step 300,000 | 1,643 | -0.3088 | 0.0643 | 0.4504 | 0.1010 | 0.3347 | 1.3665 | 0.0045 | -0.0489 |
| Step 350,000 | 1,893 | -0.5539 | 0.0505 | 0.1317 | 0.0992 | 0.1556 | 1.9456 | 0.0093 | -0.1662 |
| Step 400,000 | 2,148 | 0.4455 | 0.0027 | 0.3122 | 0.1113 | 0.3153 | 2.0377 | 0.0082 | -0.1396 |
| Step 450,000 | 2,414 | 0.7672 | 0.3021 | 0.4587 | 0.2550 | 0.4345 | 2.3782 | 0.0092 | -0.1396 |
| Step 500,000 | 2,678 | 0.8062 | 0.4061 | 0.5938 | 0.2687 | 0.4329 | 2.0459 | 0.0099 | -0.1108 |
| Step 550,000 | 2,947 | 0.4398 | 0.0704 | 0.3101 | 0.0988 | 0.3047 | 2.6197 | 0.0169 | -0.1047 |
| Step 600,000 | 3,206 | 0.8416 | 0.1860 | 0.3365 | 0.1855 | 0.3951 | 2.1607 | 0.0088 | -0.0698 |
| Step 650,000 | 3,466 | 0.8012 | 0.4258 | 0.2601 | 0.3375 | 0.3035 | 2.6724 | 0.0128 | -0.1292 |
| Step 700,000 | 3,719 | 0.8110 | 0.3474 | 0.2892 | 0.2679 | 0.3801 | 3.1496 | 0.0135 | 0.0000 |
| Step 750,000 | 3,985 | 0.3733 | 0.1015 | 0.1103 | 0.1561 | 0.1851 | 3.5728 | 0.0201 | -0.0923 |
| Step 800,000 | 4,249 | 0.5192 | 0.1492 | 0.0593 | 0.1860 | 0.1601 | 3.8540 | 0.0211 | -0.1047 |
| Step 850,000 | 4,508 | 0.7085 | 0.2113 | 0.2100 | 0.2251 | 0.3189 | 3.9934 | 0.0207 | -0.1047 |
| Step 900,000 | 4,766 | 0.3150 | 0.1365 | 0.1231 | 0.1574 | 0.1718 | 3.7347 | 0.0212 | -0.1477 |
| Step 950,000 | 5,023 | 0.2805 | 0.0603 | 0.1997 | 0.1275 | 0.2465 | 3.9566 | 0.0239 | -0.1108 |
| Step 1,000,000 | 5,292 | -0.3021 | 0.1617 | 0.0382 | 0.1433 | 0.1635 | 3.6544 | 0.0188 | -0.0554 |
| Step 1,050,000 | 5,552 | 0.3350 | 0.0204 | 0.0033 | 0.1279 | 0.1327 | 4.1473 | 0.0203 | -0.0923 |
| Step 1,100,000 | 5,810 | -0.2222 | 0.0888 | 0.0423 | 0.1551 | 0.1574 | 4.1853 | 0.0239 | -0.0979 |
| Step 1,150,000 | 6,068 | 0.6748 | 0.1951 | 0.1335 | 0.2194 | 0.2141 | 3.9912 | 0.0195 | -0.1292 |
| Step 1,200,000 | 6,322 | 0.6045 | 0.3903 | 0.2201 | 0.3170 | 0.2774 | 4.9523 | 0.0325 | -0.0881 |
| Step 1,250,000 | 6,588 | 0.6276 | 0.2592 | 0.1137 | 0.2416 | 0.2223 | 4.2575 | 0.0289 | -0.0185 |
| Step 1,300,000 | 6,844 | 0.2816 | 0.1821 | 0.0433 | 0.2410 | 0.1976 | 5.1227 | 0.0279 | -0.0554 |
| Step 1,350,000 | 7,102 | 0.7617 | 0.2395 | 0.1279 | 0.2598 | 0.3459 | 4.6139 | 0.0237 | -0.0979 |
| Step 1,400,000 | 7,352 | 0.6632 | 0.3976 | 0.1059 | 0.3252 | 0.2339 | 4.6025 | 0.0254 | -0.0369 |
| Step 1,450,000 | 7,608 | 0.4790 | 0.1482 | 0.0763 | 0.1724 | 0.2125 | 4.7943 | 0.0310 | 0.0000 |
| Step 1,500,000 | 7,863 | 0.3144 | 0.2789 | 0.0204 | 0.2799 | 0.1882 | 5.1464 | 0.0360 | 0.0185 |
| Step 1,550,000 | 8,124 | 0.3199 | 0.0807 | -0.0121 | 0.1594 | 0.1414 | 5.1828 | 0.0277 | -0.0489 |
| Step 1,600,000 | 8,383 | 0.5840 | 0.1735 | 0.0719 | 0.2376 | 0.2220 | 6.4477 | 0.0396 | 0.0739 |
| Step 1,650,000 | 8,649 | 0.7092 | 0.1973 | 0.0916 | 0.2171 | 0.2399 | 5.6640 | 0.0351 | 0.0098 |
| Step 1,700,000 | 8,909 | 0.3486 | 0.1417 | 0.0696 | 0.1656 | 0.1772 | 4.8808 | 0.0265 | 0.0000 |
| Step 1,750,000 | 9,170 | 0.3113 | 0.1400 | 0.0928 | 0.1946 | 0.2348 | 5.0194 | 0.0308 | 0.1468 |
| Step 1,800,000 | 9,436 | -0.0411 | 0.1005 | 0.0154 | 0.1610 | 0.1588 | 5.1849 | 0.0324 | 0.0923 |
| Step 1,850,000 | 9,692 | 0.1124 | -0.0020 | 0.0449 | 0.1132 | 0.1741 | 5.1972 | 0.0333 | 0.1468 |
| Step 1,900,000 | 9,951 | 0.6571 | 0.2665 | 0.1193 | 0.2363 | 0.2234 | 4.6832 | 0.0324 | 0.2153 |
| Step 1,950,000 | 10,213 | 0.3781 | 0.2936 | 0.0270 | 0.2920 | 0.2106 | 5.3713 | 0.0355 | 0.1664 |
| Step 2,000,000 | 10,485 | 0.6159 | 0.4124 | 0.1053 | 0.3701 | 0.2851 | 4.3908 | 0.0230 | 0.1664 |
| Step 2,050,000 | 10,751 | 0.3449 | 0.2729 | 0.0816 | 0.2794 | 0.2342 | 4.6554 | 0.0237 | 0.0489 |
| Step 2,100,000 | 11,014 | 0.4902 | 0.2685 | 0.0659 | 0.2712 | 0.2604 | 4.8120 | 0.0243 | 0.1108 |
| Step 2,150,000 | 11,276 | 0.4136 | 0.1874 | 0.0371 | 0.2317 | 0.2187 | 4.4584 | 0.0206 | 0.0098 |
| Step 2,200,000 | 11,535 | 0.0274 | 0.1456 | 0.0175 | 0.2087 | 0.1711 | 4.0036 | 0.0211 | 0.0587 |
| Step 2,250,000 | 11,784 | 0.4499 | 0.1708 | 0.0302 | 0.2133 | 0.1759 | 3.7930 | 0.0215 | 0.1108 |
| Step 2,300,000 | 12,028 | 0.4270 | 0.1706 | 0.0087 | 0.1893 | 0.1918 | 4.2440 | 0.0282 | 0.0185 |
| Step 2,350,000 | 12,278 | -0.0926 | 0.0422 | 0.0094 | 0.1292 | 0.1533 | 3.5725 | 0.0219 | 0.0739 |
| Step 2,400,000 | 12,516 | 0.3993 | 0.1120 | 0.0530 | 0.1511 | 0.1812 | 3.9931 | 0.0240 | 0.1292 |
| Step 2,450,000 | 12,752 | -0.0803 | 0.1385 | 0.0927 | 0.1836 | 0.2017 | 4.8248 | 0.0450 | 0.0739 |
| Step 2,500,000 | 13,000 | 0.1091 | 0.1060 | 0.0220 | 0.1621 | 0.1694 | 4.2569 | 0.0315 | 0.1662 |
| Step 2,550,000 | 13,243 | 0.6466 | 0.1785 | 0.0705 | 0.2091 | 0.2181 | 3.2905 | 0.0197 | 0.0739 |
| Step 2,600,000 | 13,474 | 0.4653 | 0.1465 | 0.0246 | 0.2015 | 0.1960 | 3.5224 | 0.0164 | 0.1662 |
| Step 2,650,000 | 13,696 | 0.4105 | 0.1997 | 0.0847 | 0.2092 | 0.2502 | 4.2310 | 0.0229 | 0.1396 |
| Step 2,700,000 | 13,909 | 0.1610 | 0.1143 | 0.0170 | 0.1543 | 0.1630 | 3.9097 | 0.0232 | 0.2443 |
| Step 2,750,000 | 14,123 | 0.2467 | 0.1271 | 0.0441 | 0.1557 | 0.1981 | 3.2311 | 0.0171 | 0.2443 |
| Step 2,800,000 | 14,351 | 0.4715 | 0.2429 | 0.0158 | 0.2005 | 0.1820 | 3.9061 | 0.0208 | 0.2443 |
| Step 2,850,000 | 14,566 | 0.4724 | 0.1640 | 0.0830 | 0.1732 | 0.2168 | 2.8758 | 0.0133 | 0.2443 |
| Step 2,900,000 | 14,782 | 0.6988 | 0.1172 | 0.0565 | 0.1993 | 0.2217 | 4.3761 | 0.0241 | 0.2791 |
| Step 2,950,000 | 15,005 | 0.7108 | 0.3235 | 0.0595 | 0.3050 | 0.2094 | 3.0038 | 0.0131 | 0.2443 |
| Step 3,000,000 | 15,229 | 0.4370 | 0.0710 | 0.0505 | 0.1692 | 0.2101 | 3.3461 | 0.0174 | 0.2443 |
| Final | 15,231 | 0.3122 | 0.1374 | 0.0616 | 0.1674 | 0.1821 | 3.0837 | 0.0144 | 0.2791 |
