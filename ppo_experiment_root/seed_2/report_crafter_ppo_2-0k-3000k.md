# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 2  
**Total episodes:** 15,435  
**Experiment root:** `ppo_experiment_root\seed_2`  
**Generated:** 2026-04-01 00:33

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.1354 | 0.6703 | 0.0874 | 0.0429 | 0.0146 | 0.1004 | 0.0019 | nan |
| place_table @ ep1 | 1 | 0.0573 | 0.6918 | 0.0013 | 0.0415 | 0.0112 | 0.1032 | 0.0020 | nan |
| wake_up @ ep2 | 2 | -0.0420 | 0.9035 | -0.0248 | 0.0805 | 0.0214 | 0.2128 | 0.0022 | -0.6547 |
| collect_drink @ ep3 | 3 | 0.0583 | 0.8607 | 0.1578 | 0.0788 | 0.0322 | 0.2114 | 0.0022 | nan |
| collect_sapling @ ep3 | 3 | 0.1363 | 0.9027 | 0.0555 | 0.0748 | 0.0260 | 0.2216 | 0.0022 | — |
| place_plant @ ep3 | 3 | -0.0251 | 0.9085 | 0.0643 | 0.0749 | 0.0245 | 0.2216 | 0.0022 | nan |
| make_wood_sword @ ep82 | 82 | 0.8331 | 0.8970 | 0.7713 | 0.3155 | 0.1702 | 0.4470 | 0.0002 | -0.3838 |
| defeat_skeleton @ ep113 | 113 | 0.8779 | 0.8537 | 0.5084 | 0.2233 | 0.1975 | 0.3591 | 0.0004 | -0.5222 |
| eat_cow @ ep131 | 131 | 0.8449 | 0.8388 | 0.4426 | 0.1958 | 0.2258 | 0.6093 | 0.0004 | -0.6547 |
| defeat_zombie @ ep166 | 166 | 0.4162 | 0.3773 | 0.2442 | 0.1830 | 0.1938 | 0.9454 | 0.0004 | -0.1309 |
| collect_stone @ ep214 | 214 | 0.2215 | 0.5441 | 0.0033 | 0.1412 | 0.1167 | 0.5428 | 0.0005 | -0.1741 |
| make_stone_sword @ ep214 | 214 | -0.0480 | 0.5047 | 0.0235 | 0.1626 | 0.1098 | 0.4651 | 0.0004 | 0.3928 |
| make_wood_pickaxe @ ep214 | 214 | -0.0224 | 0.4280 | -0.0189 | 0.1690 | 0.1004 | 0.5475 | 0.0005 | -0.5234 |
| place_furnace @ ep214 | 214 | -0.0412 | 0.4276 | 0.1287 | 0.1536 | 0.1545 | 0.6626 | 0.0006 | -0.3140 |
| place_stone @ ep214 | 214 | -0.2129 | 0.4652 | -0.0453 | 0.1633 | 0.1028 | 0.4682 | 0.0006 | -0.2611 |
| collect_coal @ ep223 | 223 | 0.1713 | 0.0393 | 0.0268 | 0.1102 | 0.0863 | 0.5068 | 0.0005 | -0.1396 |
| Step 50,000 | 288 | 0.0744 | 0.5716 | 0.3679 | 0.1664 | 0.1250 | 0.7952 | 0.0006 | -0.2611 |
| Step 100,000 | 586 | 0.0390 | 0.3164 | 0.3427 | 0.1432 | 0.1606 | 0.6503 | 0.0001 | -0.3489 |
| Step 150,000 | 871 | -0.3512 | -0.0250 | 0.2153 | 0.0801 | 0.1659 | 0.8134 | 0.0002 | -0.0698 |
| Step 200,000 | 1,159 | -0.1161 | 0.1027 | 0.0464 | 0.1200 | 0.1222 | 0.8777 | 0.0003 | -0.1741 |
| Step 250,000 | 1,432 | 0.2908 | 0.0644 | 0.2662 | 0.1024 | 0.1873 | 0.4979 | 0.0010 | -0.1396 |
| eat_plant @ ep1710 | 1,710 | -0.1625 | 0.2843 | 0.2885 | 0.1550 | 0.2476 | 0.6295 | 0.0016 | -0.6547 |
| Step 300,000 | 1,726 | 0.3893 | 0.2585 | 0.3493 | 0.1698 | 0.2313 | 0.6772 | 0.0017 | -0.1047 |
| Step 350,000 | 2,004 | 0.3353 | 0.1360 | 0.2169 | 0.1003 | 0.1830 | 1.0771 | 0.0042 | -0.0349 |
| Step 400,000 | 2,287 | 0.2634 | 0.1404 | 0.1616 | 0.1219 | 0.1744 | 1.1686 | 0.0057 | -0.1662 |
| Step 450,000 | 2,570 | 0.2481 | 0.4032 | 0.1656 | 0.1651 | 0.1646 | 1.1369 | 0.0065 | -0.0783 |
| Step 500,000 | 2,829 | 0.5174 | 0.2343 | 0.6138 | 0.1505 | 0.4291 | 1.6598 | 0.0119 | -0.0369 |
| Step 550,000 | 3,091 | 0.1511 | 0.0664 | 0.1573 | 0.1116 | 0.1963 | 1.9442 | 0.0101 | 0.0000 |
| Step 600,000 | 3,367 | 0.5752 | 0.2687 | 0.0520 | 0.1777 | 0.1226 | 1.7017 | 0.0099 | 0.0000 |
| Step 650,000 | 3,647 | 0.6673 | 0.1159 | 0.2090 | 0.1284 | 0.2598 | 2.2091 | 0.0172 | -0.0185 |
| Step 700,000 | 3,916 | 0.3592 | 0.1421 | 0.0828 | 0.1398 | 0.1674 | 2.4073 | 0.0167 | -0.0554 |
| Step 750,000 | 4,174 | 0.5622 | 0.0451 | 0.1600 | 0.1006 | 0.2324 | 2.7539 | 0.0210 | -0.0698 |
| Step 800,000 | 4,442 | 0.6426 | 0.1117 | 0.2494 | 0.1454 | 0.2468 | 2.3417 | 0.0162 | -0.1846 |
| Step 850,000 | 4,709 | 0.8086 | 0.3898 | 0.3789 | 0.2304 | 0.3582 | 2.8867 | 0.0241 | -0.1662 |
| make_stone_pickaxe @ ep4843 | 4,843 | 0.5727 | 0.2082 | 0.0757 | 0.1674 | 0.2138 | 2.7515 | 0.0169 | -0.1292 |
| Step 900,000 | 4,978 | 0.6622 | 0.1090 | 0.0172 | 0.1424 | 0.1682 | 2.7962 | 0.0180 | -0.1846 |
| Step 950,000 | 5,237 | 0.5840 | 0.1756 | 0.2349 | 0.1892 | 0.3164 | 3.6160 | 0.0247 | -0.1047 |
| Step 1,000,000 | 5,494 | 0.3767 | 0.0602 | 0.0610 | 0.1223 | 0.1552 | 3.1516 | 0.0208 | -0.0349 |
| Step 1,050,000 | 5,768 | 0.7063 | 0.0586 | 0.1169 | 0.1290 | 0.2130 | 3.6072 | 0.0259 | -0.0739 |
| Step 1,100,000 | 6,039 | 0.2819 | 0.0465 | 0.0278 | 0.1237 | 0.1574 | 4.1180 | 0.0311 | 0.0000 |
| Step 1,150,000 | 6,314 | 0.3931 | 0.1418 | 0.0202 | 0.1225 | 0.1313 | 3.8260 | 0.0362 | -0.0554 |
| Step 1,200,000 | 6,576 | 0.3852 | 0.1988 | 0.0358 | 0.1723 | 0.1450 | 4.1264 | 0.0374 | -0.1846 |
| Step 1,250,000 | 6,841 | -0.1472 | 0.1819 | 0.0660 | 0.1724 | 0.1877 | 5.0009 | 0.0452 | -0.0698 |
| Step 1,300,000 | 7,103 | 0.6091 | 0.1986 | 0.1383 | 0.1868 | 0.2550 | 4.5823 | 0.0420 | 0.1396 |
| Step 1,350,000 | 7,356 | 0.5284 | 0.4224 | 0.0802 | 0.2958 | 0.1897 | 4.3889 | 0.0304 | 0.2094 |
| Step 1,400,000 | 7,621 | 0.4680 | 0.0306 | 0.0771 | 0.1234 | 0.1997 | 5.4216 | 0.0505 | 0.0698 |
| Step 1,450,000 | 7,880 | 0.2306 | 0.0110 | 0.0152 | 0.1039 | 0.1575 | 5.0246 | 0.0442 | 0.0000 |
| Step 1,500,000 | 8,130 | 0.2804 | 0.1732 | 0.0383 | 0.1588 | 0.1683 | 4.6376 | 0.0420 | -0.0739 |
| Step 1,550,000 | 8,374 | 0.4930 | 0.2526 | 0.1257 | 0.2598 | 0.2452 | 4.9873 | 0.0429 | -0.0739 |
| Step 1,600,000 | 8,614 | 0.8029 | 0.2545 | 0.3777 | 0.2024 | 0.3457 | 5.2450 | 0.0616 | 0.0000 |
| collect_iron @ ep8840 | 8,840 | 0.5079 | 0.0797 | 0.0136 | 0.1376 | 0.1799 | 5.4611 | 0.0488 | -0.0923 |
| Step 1,650,000 | 8,861 | 0.5746 | 0.0046 | 0.1192 | 0.1193 | 0.2250 | 5.8185 | 0.0533 | -0.0923 |
| Step 1,700,000 | 9,117 | 0.6647 | 0.1605 | 0.1855 | 0.1654 | 0.2946 | 5.8835 | 0.0632 | -0.0369 |
| Step 1,750,000 | 9,369 | 0.1355 | 0.0490 | 0.0204 | 0.1429 | 0.1813 | 5.5083 | 0.0418 | 0.1396 |
| Step 1,800,000 | 9,619 | 0.5308 | 0.0781 | 0.0632 | 0.1547 | 0.2010 | 5.1160 | 0.0432 | 0.1396 |
| Step 1,850,000 | 9,866 | 0.2085 | 0.1296 | -0.0043 | 0.1453 | 0.1492 | 5.2607 | 0.0490 | 0.0554 |
| Step 1,900,000 | 10,120 | 0.5144 | 0.1767 | 0.2032 | 0.1884 | 0.2653 | 4.2630 | 0.0285 | 0.0000 |
| Step 1,950,000 | 10,377 | 0.3404 | 0.0238 | 0.0732 | 0.1168 | 0.1982 | 4.6528 | 0.0303 | 0.0185 |
| Step 2,000,000 | 10,632 | 0.0974 | 0.0430 | 0.0387 | 0.1319 | 0.1544 | 4.2074 | 0.0394 | 0.1108 |
| Step 2,050,000 | 10,885 | 0.4909 | 0.2543 | 0.0121 | 0.2174 | 0.1498 | 4.4637 | 0.0331 | 0.0923 |
| Step 2,100,000 | 11,123 | 0.4651 | 0.0193 | 0.0857 | 0.1222 | 0.1969 | 3.7071 | 0.0277 | 0.0923 |
| Step 2,150,000 | 11,361 | 0.1082 | 0.0540 | -0.0016 | 0.1409 | 0.1739 | 4.5083 | 0.0309 | 0.0369 |
| Step 2,200,000 | 11,601 | 0.7425 | 0.3441 | 0.1477 | 0.2920 | 0.2634 | 3.5879 | 0.0224 | 0.1272 |
| Step 2,250,000 | 11,850 | 0.6905 | 0.2002 | 0.1487 | 0.2124 | 0.2473 | 3.6616 | 0.0293 | -0.0098 |
| Step 2,300,000 | 12,093 | 0.5796 | 0.2380 | 0.1449 | 0.2410 | 0.2539 | 2.6947 | 0.0141 | 0.0881 |
| Step 2,350,000 | 12,346 | 0.1987 | 0.0745 | -0.0088 | 0.1476 | 0.1471 | 3.5467 | 0.0256 | -0.0196 |
| Step 2,400,000 | 12,586 | 0.3545 | 0.1820 | 0.0879 | 0.2029 | 0.1827 | 3.4655 | 0.0196 | 0.0196 |
| Step 2,450,000 | 12,830 | 0.4677 | 0.1744 | 0.0591 | 0.1914 | 0.2024 | 3.3638 | 0.0178 | 0.0000 |
| Step 2,500,000 | 13,060 | 0.3255 | 0.0434 | 0.0078 | 0.1337 | 0.1740 | 4.1560 | 0.0274 | -0.0294 |
| Step 2,550,000 | 13,308 | 0.1971 | 0.1298 | 0.0368 | 0.1923 | 0.1960 | 4.0357 | 0.0216 | 0.0196 |
| Step 2,600,000 | 13,543 | 0.1053 | 0.1506 | 0.0232 | 0.1852 | 0.1670 | 3.8262 | 0.0207 | 0.0196 |
| Step 2,650,000 | 13,782 | 0.5938 | 0.3029 | 0.0520 | 0.2737 | 0.2264 | 4.0063 | 0.0203 | 0.0294 |
| Step 2,700,000 | 14,015 | 0.3833 | 0.2817 | 0.0224 | 0.3110 | 0.1719 | 2.9746 | 0.0107 | -0.0294 |
| Step 2,750,000 | 14,256 | 0.6639 | 0.3229 | 0.1502 | 0.2842 | 0.2483 | 3.1671 | 0.0164 | 0.0000 |
| Step 2,800,000 | 14,493 | 0.2606 | 0.0904 | 0.1299 | 0.1539 | 0.2629 | 4.8873 | 0.0288 | 0.0098 |
| Step 2,850,000 | 14,727 | 0.2867 | 0.0952 | -0.0064 | 0.1532 | 0.1457 | 4.3136 | 0.0257 | 0.0000 |
| Step 2,900,000 | 14,958 | 0.4923 | 0.2885 | 0.1301 | 0.2818 | 0.2912 | 4.4301 | 0.0253 | 0.0098 |
| Step 2,950,000 | 15,196 | 0.6773 | 0.2416 | 0.0584 | 0.2362 | 0.2090 | 2.9728 | 0.0179 | 0.0000 |
| Step 3,000,000 | 15,435 | 0.0097 | 0.1116 | 0.0130 | 0.1784 | 0.1750 | 3.3856 | 0.0201 | -0.0881 |
| Final | 15,435 | 0.0970 | 0.0219 | 0.0444 | 0.1204 | 0.1822 | 3.6031 | 0.0216 | 0.0783 |

---

## collect_wood_ep1_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1354 |
| Coherence (Success) | 0.6703 |
| Coherence (Failure) | 0.0874 |
| Gradient Magnitude (Success) | 0.0429 |
| Gradient Magnitude (Failure) | 0.0146 |
| Activation Separation | 0.1004 |
| Cosine Distance | 0.0019 |
| Clusters | 1,467 |
| Noise Fraction | 0.2603 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## place_table_ep1_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0573 |
| Coherence (Success) | 0.6918 |
| Coherence (Failure) | 0.0013 |
| Gradient Magnitude (Success) | 0.0415 |
| Gradient Magnitude (Failure) | 0.0112 |
| Activation Separation | 0.1032 |
| Cosine Distance | 0.0020 |
| Clusters | 1,484 |
| Noise Fraction | 0.2521 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## wake_up_ep2_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0420 |
| Coherence (Success) | 0.9035 |
| Coherence (Failure) | -0.0248 |
| Gradient Magnitude (Success) | 0.0805 |
| Gradient Magnitude (Failure) | 0.0214 |
| Activation Separation | 0.2128 |
| Cosine Distance | 0.0022 |
| Clusters | 1,490 |
| Noise Fraction | 0.2340 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0583 |
| Coherence (Success) | 0.8607 |
| Coherence (Failure) | 0.1578 |
| Gradient Magnitude (Success) | 0.0788 |
| Gradient Magnitude (Failure) | 0.0322 |
| Activation Separation | 0.2114 |
| Cosine Distance | 0.0022 |
| Clusters | 1,535 |
| Noise Fraction | 0.2353 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1363 |
| Coherence (Success) | 0.9027 |
| Coherence (Failure) | 0.0555 |
| Gradient Magnitude (Success) | 0.0748 |
| Gradient Magnitude (Failure) | 0.0260 |
| Activation Separation | 0.2216 |
| Cosine Distance | 0.0022 |
| Clusters | 1,472 |
| Noise Fraction | 0.2179 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## place_plant_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0251 |
| Coherence (Success) | 0.9085 |
| Coherence (Failure) | 0.0643 |
| Gradient Magnitude (Success) | 0.0749 |
| Gradient Magnitude (Failure) | 0.0245 |
| Activation Separation | 0.2216 |
| Cosine Distance | 0.0022 |
| Clusters | 1,534 |
| Noise Fraction | 0.2345 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep82_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8331 |
| Coherence (Success) | 0.8970 |
| Coherence (Failure) | 0.7713 |
| Gradient Magnitude (Success) | 0.3155 |
| Gradient Magnitude (Failure) | 0.1702 |
| Activation Separation | 0.4470 |
| Cosine Distance | 0.0002 |
| Clusters | 1,495 |
| Noise Fraction | 0.2371 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep113_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8779 |
| Coherence (Success) | 0.8537 |
| Coherence (Failure) | 0.5084 |
| Gradient Magnitude (Success) | 0.2233 |
| Gradient Magnitude (Failure) | 0.1975 |
| Activation Separation | 0.3591 |
| Cosine Distance | 0.0004 |
| Clusters | 1,479 |
| Noise Fraction | 0.2513 |
| RSA Alignment (ρ) | -0.5222 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep131_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8449 |
| Coherence (Success) | 0.8388 |
| Coherence (Failure) | 0.4426 |
| Gradient Magnitude (Success) | 0.1958 |
| Gradient Magnitude (Failure) | 0.2258 |
| Activation Separation | 0.6093 |
| Cosine Distance | 0.0004 |
| Clusters | 1,299 |
| Noise Fraction | 0.2506 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep166_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4162 |
| Coherence (Success) | 0.3773 |
| Coherence (Failure) | 0.2442 |
| Gradient Magnitude (Success) | 0.1830 |
| Gradient Magnitude (Failure) | 0.1938 |
| Activation Separation | 0.9454 |
| Cosine Distance | 0.0004 |
| Clusters | 1,363 |
| Noise Fraction | 0.2471 |
| RSA Alignment (ρ) | -0.1309 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep214_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2215 |
| Coherence (Success) | 0.5441 |
| Coherence (Failure) | 0.0033 |
| Gradient Magnitude (Success) | 0.1412 |
| Gradient Magnitude (Failure) | 0.1167 |
| Activation Separation | 0.5428 |
| Cosine Distance | 0.0005 |
| Clusters | 1,497 |
| Noise Fraction | 0.2463 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep214_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0480 |
| Coherence (Success) | 0.5047 |
| Coherence (Failure) | 0.0235 |
| Gradient Magnitude (Success) | 0.1626 |
| Gradient Magnitude (Failure) | 0.1098 |
| Activation Separation | 0.4651 |
| Cosine Distance | 0.0004 |
| Clusters | 1,584 |
| Noise Fraction | 0.2594 |
| RSA Alignment (ρ) | 0.3928 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep214_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0224 |
| Coherence (Success) | 0.4280 |
| Coherence (Failure) | -0.0189 |
| Gradient Magnitude (Success) | 0.1690 |
| Gradient Magnitude (Failure) | 0.1004 |
| Activation Separation | 0.5475 |
| Cosine Distance | 0.0005 |
| Clusters | 1,596 |
| Noise Fraction | 0.2497 |
| RSA Alignment (ρ) | -0.5234 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep214_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0412 |
| Coherence (Success) | 0.4276 |
| Coherence (Failure) | 0.1287 |
| Gradient Magnitude (Success) | 0.1536 |
| Gradient Magnitude (Failure) | 0.1545 |
| Activation Separation | 0.6626 |
| Cosine Distance | 0.0006 |
| Clusters | 1,572 |
| Noise Fraction | 0.2725 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep214_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2129 |
| Coherence (Success) | 0.4652 |
| Coherence (Failure) | -0.0453 |
| Gradient Magnitude (Success) | 0.1633 |
| Gradient Magnitude (Failure) | 0.1028 |
| Activation Separation | 0.4682 |
| Cosine Distance | 0.0006 |
| Clusters | 1,527 |
| Noise Fraction | 0.2629 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep223_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1713 |
| Coherence (Success) | 0.0393 |
| Coherence (Failure) | 0.0268 |
| Gradient Magnitude (Success) | 0.1102 |
| Gradient Magnitude (Failure) | 0.0863 |
| Activation Separation | 0.5068 |
| Cosine Distance | 0.0005 |
| Clusters | 1,585 |
| Noise Fraction | 0.2654 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep288_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0744 |
| Coherence (Success) | 0.5716 |
| Coherence (Failure) | 0.3679 |
| Gradient Magnitude (Success) | 0.1664 |
| Gradient Magnitude (Failure) | 0.1250 |
| Activation Separation | 0.7952 |
| Cosine Distance | 0.0006 |
| Clusters | 1,499 |
| Noise Fraction | 0.2685 |
| RSA Alignment (ρ) | -0.2611 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep586_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0390 |
| Coherence (Success) | 0.3164 |
| Coherence (Failure) | 0.3427 |
| Gradient Magnitude (Success) | 0.1432 |
| Gradient Magnitude (Failure) | 0.1606 |
| Activation Separation | 0.6503 |
| Cosine Distance | 0.0001 |
| Clusters | 1,395 |
| Noise Fraction | 0.2962 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep871_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3512 |
| Coherence (Success) | -0.0250 |
| Coherence (Failure) | 0.2153 |
| Gradient Magnitude (Success) | 0.0801 |
| Gradient Magnitude (Failure) | 0.1659 |
| Activation Separation | 0.8134 |
| Cosine Distance | 0.0002 |
| Clusters | 1,440 |
| Noise Fraction | 0.3177 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1159_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1161 |
| Coherence (Success) | 0.1027 |
| Coherence (Failure) | 0.0464 |
| Gradient Magnitude (Success) | 0.1200 |
| Gradient Magnitude (Failure) | 0.1222 |
| Activation Separation | 0.8777 |
| Cosine Distance | 0.0003 |
| Clusters | 1,393 |
| Noise Fraction | 0.3376 |
| RSA Alignment (ρ) | -0.1741 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1432_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2908 |
| Coherence (Success) | 0.0644 |
| Coherence (Failure) | 0.2662 |
| Gradient Magnitude (Success) | 0.1024 |
| Gradient Magnitude (Failure) | 0.1873 |
| Activation Separation | 0.4979 |
| Cosine Distance | 0.0010 |
| Clusters | 1,525 |
| Noise Fraction | 0.2669 |
| RSA Alignment (ρ) | -0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep1710_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1625 |
| Coherence (Success) | 0.2843 |
| Coherence (Failure) | 0.2885 |
| Gradient Magnitude (Success) | 0.1550 |
| Gradient Magnitude (Failure) | 0.2476 |
| Activation Separation | 0.6295 |
| Cosine Distance | 0.0016 |
| Clusters | 1,245 |
| Noise Fraction | 0.2950 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1726_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3893 |
| Coherence (Success) | 0.2585 |
| Coherence (Failure) | 0.3493 |
| Gradient Magnitude (Success) | 0.1698 |
| Gradient Magnitude (Failure) | 0.2313 |
| Activation Separation | 0.6772 |
| Cosine Distance | 0.0017 |
| Clusters | 1,256 |
| Noise Fraction | 0.2818 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2004_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3353 |
| Coherence (Success) | 0.1360 |
| Coherence (Failure) | 0.2169 |
| Gradient Magnitude (Success) | 0.1003 |
| Gradient Magnitude (Failure) | 0.1830 |
| Activation Separation | 1.0771 |
| Cosine Distance | 0.0042 |
| Clusters | 1,330 |
| Noise Fraction | 0.2849 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2287_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2634 |
| Coherence (Success) | 0.1404 |
| Coherence (Failure) | 0.1616 |
| Gradient Magnitude (Success) | 0.1219 |
| Gradient Magnitude (Failure) | 0.1744 |
| Activation Separation | 1.1686 |
| Cosine Distance | 0.0057 |
| Clusters | 1,511 |
| Noise Fraction | 0.2555 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2570_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2481 |
| Coherence (Success) | 0.4032 |
| Coherence (Failure) | 0.1656 |
| Gradient Magnitude (Success) | 0.1651 |
| Gradient Magnitude (Failure) | 0.1646 |
| Activation Separation | 1.1369 |
| Cosine Distance | 0.0065 |
| Clusters | 1,287 |
| Noise Fraction | 0.2728 |
| RSA Alignment (ρ) | -0.0783 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2829_lower4.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5174 |
| Coherence (Success) | 0.2343 |
| Coherence (Failure) | 0.6138 |
| Gradient Magnitude (Success) | 0.1505 |
| Gradient Magnitude (Failure) | 0.4291 |
| Activation Separation | 1.6598 |
| Cosine Distance | 0.0119 |
| Clusters | 1,577 |
| Noise Fraction | 0.2856 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3091_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1511 |
| Coherence (Success) | 0.0664 |
| Coherence (Failure) | 0.1573 |
| Gradient Magnitude (Success) | 0.1116 |
| Gradient Magnitude (Failure) | 0.1963 |
| Activation Separation | 1.9442 |
| Cosine Distance | 0.0101 |
| Clusters | 1,660 |
| Noise Fraction | 0.2636 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3367_lower5.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5752 |
| Coherence (Success) | 0.2687 |
| Coherence (Failure) | 0.0520 |
| Gradient Magnitude (Success) | 0.1777 |
| Gradient Magnitude (Failure) | 0.1226 |
| Activation Separation | 1.7017 |
| Cosine Distance | 0.0099 |
| Clusters | 1,575 |
| Noise Fraction | 0.2780 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3647_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6673 |
| Coherence (Success) | 0.1159 |
| Coherence (Failure) | 0.2090 |
| Gradient Magnitude (Success) | 0.1284 |
| Gradient Magnitude (Failure) | 0.2598 |
| Activation Separation | 2.2091 |
| Cosine Distance | 0.0172 |
| Clusters | 1,530 |
| Noise Fraction | 0.2724 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3916_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3592 |
| Coherence (Success) | 0.1421 |
| Coherence (Failure) | 0.0828 |
| Gradient Magnitude (Success) | 0.1398 |
| Gradient Magnitude (Failure) | 0.1674 |
| Activation Separation | 2.4073 |
| Cosine Distance | 0.0167 |
| Clusters | 1,545 |
| Noise Fraction | 0.2604 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4174_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5622 |
| Coherence (Success) | 0.0451 |
| Coherence (Failure) | 0.1600 |
| Gradient Magnitude (Success) | 0.1006 |
| Gradient Magnitude (Failure) | 0.2324 |
| Activation Separation | 2.7539 |
| Cosine Distance | 0.0210 |
| Clusters | 1,599 |
| Noise Fraction | 0.2795 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4442_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6426 |
| Coherence (Success) | 0.1117 |
| Coherence (Failure) | 0.2494 |
| Gradient Magnitude (Success) | 0.1454 |
| Gradient Magnitude (Failure) | 0.2468 |
| Activation Separation | 2.3417 |
| Cosine Distance | 0.0162 |
| Clusters | 1,397 |
| Noise Fraction | 0.3278 |
| RSA Alignment (ρ) | -0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4709_lower5.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8086 |
| Coherence (Success) | 0.3898 |
| Coherence (Failure) | 0.3789 |
| Gradient Magnitude (Success) | 0.2304 |
| Gradient Magnitude (Failure) | 0.3582 |
| Activation Separation | 2.8867 |
| Cosine Distance | 0.0241 |
| Clusters | 1,270 |
| Noise Fraction | 0.3411 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep4843_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5727 |
| Coherence (Success) | 0.2082 |
| Coherence (Failure) | 0.0757 |
| Gradient Magnitude (Success) | 0.1674 |
| Gradient Magnitude (Failure) | 0.2138 |
| Activation Separation | 2.7515 |
| Cosine Distance | 0.0169 |
| Clusters | 1,317 |
| Noise Fraction | 0.3426 |
| RSA Alignment (ρ) | -0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep4978_lower5.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6622 |
| Coherence (Success) | 0.1090 |
| Coherence (Failure) | 0.0172 |
| Gradient Magnitude (Success) | 0.1424 |
| Gradient Magnitude (Failure) | 0.1682 |
| Activation Separation | 2.7962 |
| Cosine Distance | 0.0180 |
| Clusters | 1,331 |
| Noise Fraction | 0.2832 |
| RSA Alignment (ρ) | -0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep5237_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5840 |
| Coherence (Success) | 0.1756 |
| Coherence (Failure) | 0.2349 |
| Gradient Magnitude (Success) | 0.1892 |
| Gradient Magnitude (Failure) | 0.3164 |
| Activation Separation | 3.6160 |
| Cosine Distance | 0.0247 |
| Clusters | 1,545 |
| Noise Fraction | 0.2940 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5494_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3767 |
| Coherence (Success) | 0.0602 |
| Coherence (Failure) | 0.0610 |
| Gradient Magnitude (Success) | 0.1223 |
| Gradient Magnitude (Failure) | 0.1552 |
| Activation Separation | 3.1516 |
| Cosine Distance | 0.0208 |
| Clusters | 1,530 |
| Noise Fraction | 0.3103 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5768_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7063 |
| Coherence (Success) | 0.0586 |
| Coherence (Failure) | 0.1169 |
| Gradient Magnitude (Success) | 0.1290 |
| Gradient Magnitude (Failure) | 0.2130 |
| Activation Separation | 3.6072 |
| Cosine Distance | 0.0259 |
| Clusters | 1,594 |
| Noise Fraction | 0.3123 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6039_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2819 |
| Coherence (Success) | 0.0465 |
| Coherence (Failure) | 0.0278 |
| Gradient Magnitude (Success) | 0.1237 |
| Gradient Magnitude (Failure) | 0.1574 |
| Activation Separation | 4.1180 |
| Cosine Distance | 0.0311 |
| Clusters | 1,499 |
| Noise Fraction | 0.2965 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6314_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3931 |
| Coherence (Success) | 0.1418 |
| Coherence (Failure) | 0.0202 |
| Gradient Magnitude (Success) | 0.1225 |
| Gradient Magnitude (Failure) | 0.1313 |
| Activation Separation | 3.8260 |
| Cosine Distance | 0.0362 |
| Clusters | 1,368 |
| Noise Fraction | 0.3045 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6576_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3852 |
| Coherence (Success) | 0.1988 |
| Coherence (Failure) | 0.0358 |
| Gradient Magnitude (Success) | 0.1723 |
| Gradient Magnitude (Failure) | 0.1450 |
| Activation Separation | 4.1264 |
| Cosine Distance | 0.0374 |
| Clusters | 1,460 |
| Noise Fraction | 0.3128 |
| RSA Alignment (ρ) | -0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6841_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1472 |
| Coherence (Success) | 0.1819 |
| Coherence (Failure) | 0.0660 |
| Gradient Magnitude (Success) | 0.1724 |
| Gradient Magnitude (Failure) | 0.1877 |
| Activation Separation | 5.0009 |
| Cosine Distance | 0.0452 |
| Clusters | 1,583 |
| Noise Fraction | 0.3087 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7103_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6091 |
| Coherence (Success) | 0.1986 |
| Coherence (Failure) | 0.1383 |
| Gradient Magnitude (Success) | 0.1868 |
| Gradient Magnitude (Failure) | 0.2550 |
| Activation Separation | 4.5823 |
| Cosine Distance | 0.0420 |
| Clusters | 1,507 |
| Noise Fraction | 0.3178 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7356_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5284 |
| Coherence (Success) | 0.4224 |
| Coherence (Failure) | 0.0802 |
| Gradient Magnitude (Success) | 0.2958 |
| Gradient Magnitude (Failure) | 0.1897 |
| Activation Separation | 4.3889 |
| Cosine Distance | 0.0304 |
| Clusters | 1,524 |
| Noise Fraction | 0.3066 |
| RSA Alignment (ρ) | 0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7621_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4680 |
| Coherence (Success) | 0.0306 |
| Coherence (Failure) | 0.0771 |
| Gradient Magnitude (Success) | 0.1234 |
| Gradient Magnitude (Failure) | 0.1997 |
| Activation Separation | 5.4216 |
| Cosine Distance | 0.0505 |
| Clusters | 1,515 |
| Noise Fraction | 0.2703 |
| RSA Alignment (ρ) | 0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep7880_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2306 |
| Coherence (Success) | 0.0110 |
| Coherence (Failure) | 0.0152 |
| Gradient Magnitude (Success) | 0.1039 |
| Gradient Magnitude (Failure) | 0.1575 |
| Activation Separation | 5.0246 |
| Cosine Distance | 0.0442 |
| Clusters | 1,577 |
| Noise Fraction | 0.3108 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8130_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2804 |
| Coherence (Success) | 0.1732 |
| Coherence (Failure) | 0.0383 |
| Gradient Magnitude (Success) | 0.1588 |
| Gradient Magnitude (Failure) | 0.1683 |
| Activation Separation | 4.6376 |
| Cosine Distance | 0.0420 |
| Clusters | 1,656 |
| Noise Fraction | 0.3081 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8374_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4930 |
| Coherence (Success) | 0.2526 |
| Coherence (Failure) | 0.1257 |
| Gradient Magnitude (Success) | 0.2598 |
| Gradient Magnitude (Failure) | 0.2452 |
| Activation Separation | 4.9873 |
| Cosine Distance | 0.0429 |
| Clusters | 1,553 |
| Noise Fraction | 0.3261 |
| RSA Alignment (ρ) | -0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8614_lower7.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8029 |
| Coherence (Success) | 0.2545 |
| Coherence (Failure) | 0.3777 |
| Gradient Magnitude (Success) | 0.2024 |
| Gradient Magnitude (Failure) | 0.3457 |
| Activation Separation | 5.2450 |
| Cosine Distance | 0.0616 |
| Clusters | 1,612 |
| Noise Fraction | 0.2964 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep8840_lower7.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5079 |
| Coherence (Success) | 0.0797 |
| Coherence (Failure) | 0.0136 |
| Gradient Magnitude (Success) | 0.1376 |
| Gradient Magnitude (Failure) | 0.1799 |
| Activation Separation | 5.4611 |
| Cosine Distance | 0.0488 |
| Clusters | 1,473 |
| Noise Fraction | 0.2945 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8861_lower7.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5746 |
| Coherence (Success) | 0.0046 |
| Coherence (Failure) | 0.1192 |
| Gradient Magnitude (Success) | 0.1193 |
| Gradient Magnitude (Failure) | 0.2250 |
| Activation Separation | 5.8185 |
| Cosine Distance | 0.0533 |
| Clusters | 1,431 |
| Noise Fraction | 0.2990 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9117_lower7.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6647 |
| Coherence (Success) | 0.1605 |
| Coherence (Failure) | 0.1855 |
| Gradient Magnitude (Success) | 0.1654 |
| Gradient Magnitude (Failure) | 0.2946 |
| Activation Separation | 5.8835 |
| Cosine Distance | 0.0632 |
| Clusters | 1,508 |
| Noise Fraction | 0.3227 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9369_lower7.900_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1355 |
| Coherence (Success) | 0.0490 |
| Coherence (Failure) | 0.0204 |
| Gradient Magnitude (Success) | 0.1429 |
| Gradient Magnitude (Failure) | 0.1813 |
| Activation Separation | 5.5083 |
| Cosine Distance | 0.0418 |
| Clusters | 1,593 |
| Noise Fraction | 0.3486 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9619_lower7.900_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5308 |
| Coherence (Success) | 0.0781 |
| Coherence (Failure) | 0.0632 |
| Gradient Magnitude (Success) | 0.1547 |
| Gradient Magnitude (Failure) | 0.2010 |
| Activation Separation | 5.1160 |
| Cosine Distance | 0.0432 |
| Clusters | 1,354 |
| Noise Fraction | 0.3173 |
| RSA Alignment (ρ) | 0.1396 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep9866_lower7.900_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2085 |
| Coherence (Success) | 0.1296 |
| Coherence (Failure) | -0.0043 |
| Gradient Magnitude (Success) | 0.1453 |
| Gradient Magnitude (Failure) | 0.1492 |
| Activation Separation | 5.2607 |
| Cosine Distance | 0.0490 |
| Clusters | 1,461 |
| Noise Fraction | 0.3122 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10120_lower7.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5144 |
| Coherence (Success) | 0.1767 |
| Coherence (Failure) | 0.2032 |
| Gradient Magnitude (Success) | 0.1884 |
| Gradient Magnitude (Failure) | 0.2653 |
| Activation Separation | 4.2630 |
| Cosine Distance | 0.0285 |
| Clusters | 1,358 |
| Noise Fraction | 0.3218 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10377_lower8.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3404 |
| Coherence (Success) | 0.0238 |
| Coherence (Failure) | 0.0732 |
| Gradient Magnitude (Success) | 0.1168 |
| Gradient Magnitude (Failure) | 0.1982 |
| Activation Separation | 4.6528 |
| Cosine Distance | 0.0303 |
| Clusters | 1,344 |
| Noise Fraction | 0.3614 |
| RSA Alignment (ρ) | 0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10632_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0974 |
| Coherence (Success) | 0.0430 |
| Coherence (Failure) | 0.0387 |
| Gradient Magnitude (Success) | 0.1319 |
| Gradient Magnitude (Failure) | 0.1544 |
| Activation Separation | 4.2074 |
| Cosine Distance | 0.0394 |
| Clusters | 1,351 |
| Noise Fraction | 0.2965 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10885_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4909 |
| Coherence (Success) | 0.2543 |
| Coherence (Failure) | 0.0121 |
| Gradient Magnitude (Success) | 0.2174 |
| Gradient Magnitude (Failure) | 0.1498 |
| Activation Separation | 4.4637 |
| Cosine Distance | 0.0331 |
| Clusters | 1,427 |
| Noise Fraction | 0.3137 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11123_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4651 |
| Coherence (Success) | 0.0193 |
| Coherence (Failure) | 0.0857 |
| Gradient Magnitude (Success) | 0.1222 |
| Gradient Magnitude (Failure) | 0.1969 |
| Activation Separation | 3.7071 |
| Cosine Distance | 0.0277 |
| Clusters | 1,335 |
| Noise Fraction | 0.3357 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11361_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1082 |
| Coherence (Success) | 0.0540 |
| Coherence (Failure) | -0.0016 |
| Gradient Magnitude (Success) | 0.1409 |
| Gradient Magnitude (Failure) | 0.1739 |
| Activation Separation | 4.5083 |
| Cosine Distance | 0.0309 |
| Clusters | 1,419 |
| Noise Fraction | 0.3343 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11601_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7425 |
| Coherence (Success) | 0.3441 |
| Coherence (Failure) | 0.1477 |
| Gradient Magnitude (Success) | 0.2920 |
| Gradient Magnitude (Failure) | 0.2634 |
| Activation Separation | 3.5879 |
| Cosine Distance | 0.0224 |
| Clusters | 1,431 |
| Noise Fraction | 0.3372 |
| RSA Alignment (ρ) | 0.1272 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11850_lower7.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6905 |
| Coherence (Success) | 0.2002 |
| Coherence (Failure) | 0.1487 |
| Gradient Magnitude (Success) | 0.2124 |
| Gradient Magnitude (Failure) | 0.2473 |
| Activation Separation | 3.6616 |
| Cosine Distance | 0.0293 |
| Clusters | 1,405 |
| Noise Fraction | 0.3361 |
| RSA Alignment (ρ) | -0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12093_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5796 |
| Coherence (Success) | 0.2380 |
| Coherence (Failure) | 0.1449 |
| Gradient Magnitude (Success) | 0.2410 |
| Gradient Magnitude (Failure) | 0.2539 |
| Activation Separation | 2.6947 |
| Cosine Distance | 0.0141 |
| Clusters | 1,177 |
| Noise Fraction | 0.3625 |
| RSA Alignment (ρ) | 0.0881 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12346_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1987 |
| Coherence (Success) | 0.0745 |
| Coherence (Failure) | -0.0088 |
| Gradient Magnitude (Success) | 0.1476 |
| Gradient Magnitude (Failure) | 0.1471 |
| Activation Separation | 3.5467 |
| Cosine Distance | 0.0256 |
| Clusters | 1,371 |
| Noise Fraction | 0.3342 |
| RSA Alignment (ρ) | -0.0196 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12586_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3545 |
| Coherence (Success) | 0.1820 |
| Coherence (Failure) | 0.0879 |
| Gradient Magnitude (Success) | 0.2029 |
| Gradient Magnitude (Failure) | 0.1827 |
| Activation Separation | 3.4655 |
| Cosine Distance | 0.0196 |
| Clusters | 1,337 |
| Noise Fraction | 0.2972 |
| RSA Alignment (ρ) | 0.0196 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12830_lower8.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4677 |
| Coherence (Success) | 0.1744 |
| Coherence (Failure) | 0.0591 |
| Gradient Magnitude (Success) | 0.1914 |
| Gradient Magnitude (Failure) | 0.2024 |
| Activation Separation | 3.3638 |
| Cosine Distance | 0.0178 |
| Clusters | 1,322 |
| Noise Fraction | 0.3249 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13060_lower8.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3255 |
| Coherence (Success) | 0.0434 |
| Coherence (Failure) | 0.0078 |
| Gradient Magnitude (Success) | 0.1337 |
| Gradient Magnitude (Failure) | 0.1740 |
| Activation Separation | 4.1560 |
| Cosine Distance | 0.0274 |
| Clusters | 1,426 |
| Noise Fraction | 0.3621 |
| RSA Alignment (ρ) | -0.0294 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13308_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1971 |
| Coherence (Success) | 0.1298 |
| Coherence (Failure) | 0.0368 |
| Gradient Magnitude (Success) | 0.1923 |
| Gradient Magnitude (Failure) | 0.1960 |
| Activation Separation | 4.0357 |
| Cosine Distance | 0.0216 |
| Clusters | 1,324 |
| Noise Fraction | 0.3114 |
| RSA Alignment (ρ) | 0.0196 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13543_lower8.900_upper11.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1053 |
| Coherence (Success) | 0.1506 |
| Coherence (Failure) | 0.0232 |
| Gradient Magnitude (Success) | 0.1852 |
| Gradient Magnitude (Failure) | 0.1670 |
| Activation Separation | 3.8262 |
| Cosine Distance | 0.0207 |
| Clusters | 1,411 |
| Noise Fraction | 0.3257 |
| RSA Alignment (ρ) | 0.0196 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13782_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5938 |
| Coherence (Success) | 0.3029 |
| Coherence (Failure) | 0.0520 |
| Gradient Magnitude (Success) | 0.2737 |
| Gradient Magnitude (Failure) | 0.2264 |
| Activation Separation | 4.0063 |
| Cosine Distance | 0.0203 |
| Clusters | 1,285 |
| Noise Fraction | 0.3417 |
| RSA Alignment (ρ) | 0.0294 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14015_lower8.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3833 |
| Coherence (Success) | 0.2817 |
| Coherence (Failure) | 0.0224 |
| Gradient Magnitude (Success) | 0.3110 |
| Gradient Magnitude (Failure) | 0.1719 |
| Activation Separation | 2.9746 |
| Cosine Distance | 0.0107 |
| Clusters | 1,325 |
| Noise Fraction | 0.3358 |
| RSA Alignment (ρ) | -0.0294 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14256_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6639 |
| Coherence (Success) | 0.3229 |
| Coherence (Failure) | 0.1502 |
| Gradient Magnitude (Success) | 0.2842 |
| Gradient Magnitude (Failure) | 0.2483 |
| Activation Separation | 3.1671 |
| Cosine Distance | 0.0164 |
| Clusters | 1,301 |
| Noise Fraction | 0.3560 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14493_lower8.900_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2606 |
| Coherence (Success) | 0.0904 |
| Coherence (Failure) | 0.1299 |
| Gradient Magnitude (Success) | 0.1539 |
| Gradient Magnitude (Failure) | 0.2629 |
| Activation Separation | 4.8873 |
| Cosine Distance | 0.0288 |
| Clusters | 1,213 |
| Noise Fraction | 0.3175 |
| RSA Alignment (ρ) | 0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14727_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2867 |
| Coherence (Success) | 0.0952 |
| Coherence (Failure) | -0.0064 |
| Gradient Magnitude (Success) | 0.1532 |
| Gradient Magnitude (Failure) | 0.1457 |
| Activation Separation | 4.3136 |
| Cosine Distance | 0.0257 |
| Clusters | 1,196 |
| Noise Fraction | 0.3063 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14958_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4923 |
| Coherence (Success) | 0.2885 |
| Coherence (Failure) | 0.1301 |
| Gradient Magnitude (Success) | 0.2818 |
| Gradient Magnitude (Failure) | 0.2912 |
| Activation Separation | 4.4301 |
| Cosine Distance | 0.0253 |
| Clusters | 1,457 |
| Noise Fraction | 0.3117 |
| RSA Alignment (ρ) | 0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15196_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6773 |
| Coherence (Success) | 0.2416 |
| Coherence (Failure) | 0.0584 |
| Gradient Magnitude (Success) | 0.2362 |
| Gradient Magnitude (Failure) | 0.2090 |
| Activation Separation | 2.9728 |
| Cosine Distance | 0.0179 |
| Clusters | 1,334 |
| Noise Fraction | 0.3537 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15435_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0097 |
| Coherence (Success) | 0.1116 |
| Coherence (Failure) | 0.0130 |
| Gradient Magnitude (Success) | 0.1784 |
| Gradient Magnitude (Failure) | 0.1750 |
| Activation Separation | 3.3856 |
| Cosine Distance | 0.0201 |
| Clusters | 1,472 |
| Noise Fraction | 0.3202 |
| RSA Alignment (ρ) | -0.0881 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep15435_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0970 |
| Coherence (Success) | 0.0219 |
| Coherence (Failure) | 0.0444 |
| Gradient Magnitude (Success) | 0.1204 |
| Gradient Magnitude (Failure) | 0.1822 |
| Activation Separation | 3.6031 |
| Cosine Distance | 0.0216 |
| Clusters | 1,427 |
| Noise Fraction | 0.2912 |
| RSA Alignment (ρ) | 0.0783 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collect_wood @ ep1 | 1 | 0.1354 | 0.6703 | 0.0874 | 0.0429 | 0.0146 | 0.1004 | 0.0019 | nan |
| place_table @ ep1 | 1 | 0.0573 | 0.6918 | 0.0013 | 0.0415 | 0.0112 | 0.1032 | 0.0020 | nan |
| wake_up @ ep2 | 2 | -0.0420 | 0.9035 | -0.0248 | 0.0805 | 0.0214 | 0.2128 | 0.0022 | -0.6547 |
| collect_drink @ ep3 | 3 | 0.0583 | 0.8607 | 0.1578 | 0.0788 | 0.0322 | 0.2114 | 0.0022 | nan |
| collect_sapling @ ep3 | 3 | 0.1363 | 0.9027 | 0.0555 | 0.0748 | 0.0260 | 0.2216 | 0.0022 | — |
| place_plant @ ep3 | 3 | -0.0251 | 0.9085 | 0.0643 | 0.0749 | 0.0245 | 0.2216 | 0.0022 | nan |
| make_wood_sword @ ep82 | 82 | 0.8331 | 0.8970 | 0.7713 | 0.3155 | 0.1702 | 0.4470 | 0.0002 | -0.3838 |
| defeat_skeleton @ ep113 | 113 | 0.8779 | 0.8537 | 0.5084 | 0.2233 | 0.1975 | 0.3591 | 0.0004 | -0.5222 |
| eat_cow @ ep131 | 131 | 0.8449 | 0.8388 | 0.4426 | 0.1958 | 0.2258 | 0.6093 | 0.0004 | -0.6547 |
| defeat_zombie @ ep166 | 166 | 0.4162 | 0.3773 | 0.2442 | 0.1830 | 0.1938 | 0.9454 | 0.0004 | -0.1309 |
| collect_stone @ ep214 | 214 | 0.2215 | 0.5441 | 0.0033 | 0.1412 | 0.1167 | 0.5428 | 0.0005 | -0.1741 |
| make_stone_sword @ ep214 | 214 | -0.0480 | 0.5047 | 0.0235 | 0.1626 | 0.1098 | 0.4651 | 0.0004 | 0.3928 |
| make_wood_pickaxe @ ep214 | 214 | -0.0224 | 0.4280 | -0.0189 | 0.1690 | 0.1004 | 0.5475 | 0.0005 | -0.5234 |
| place_furnace @ ep214 | 214 | -0.0412 | 0.4276 | 0.1287 | 0.1536 | 0.1545 | 0.6626 | 0.0006 | -0.3140 |
| place_stone @ ep214 | 214 | -0.2129 | 0.4652 | -0.0453 | 0.1633 | 0.1028 | 0.4682 | 0.0006 | -0.2611 |
| collect_coal @ ep223 | 223 | 0.1713 | 0.0393 | 0.0268 | 0.1102 | 0.0863 | 0.5068 | 0.0005 | -0.1396 |
| eat_plant @ ep1710 | 1,710 | -0.1625 | 0.2843 | 0.2885 | 0.1550 | 0.2476 | 0.6295 | 0.0016 | -0.6547 |
| make_stone_pickaxe @ ep4843 | 4,843 | 0.5727 | 0.2082 | 0.0757 | 0.1674 | 0.2138 | 2.7515 | 0.0169 | -0.1292 |
| collect_iron @ ep8840 | 8,840 | 0.5079 | 0.0797 | 0.0136 | 0.1376 | 0.1799 | 5.4611 | 0.0488 | -0.0923 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 288 | 0.0744 | 0.5716 | 0.3679 | 0.1664 | 0.1250 | 0.7952 | 0.0006 | -0.2611 |
| Step 100,000 | 586 | 0.0390 | 0.3164 | 0.3427 | 0.1432 | 0.1606 | 0.6503 | 0.0001 | -0.3489 |
| Step 150,000 | 871 | -0.3512 | -0.0250 | 0.2153 | 0.0801 | 0.1659 | 0.8134 | 0.0002 | -0.0698 |
| Step 200,000 | 1,159 | -0.1161 | 0.1027 | 0.0464 | 0.1200 | 0.1222 | 0.8777 | 0.0003 | -0.1741 |
| Step 250,000 | 1,432 | 0.2908 | 0.0644 | 0.2662 | 0.1024 | 0.1873 | 0.4979 | 0.0010 | -0.1396 |
| Step 300,000 | 1,726 | 0.3893 | 0.2585 | 0.3493 | 0.1698 | 0.2313 | 0.6772 | 0.0017 | -0.1047 |
| Step 350,000 | 2,004 | 0.3353 | 0.1360 | 0.2169 | 0.1003 | 0.1830 | 1.0771 | 0.0042 | -0.0349 |
| Step 400,000 | 2,287 | 0.2634 | 0.1404 | 0.1616 | 0.1219 | 0.1744 | 1.1686 | 0.0057 | -0.1662 |
| Step 450,000 | 2,570 | 0.2481 | 0.4032 | 0.1656 | 0.1651 | 0.1646 | 1.1369 | 0.0065 | -0.0783 |
| Step 500,000 | 2,829 | 0.5174 | 0.2343 | 0.6138 | 0.1505 | 0.4291 | 1.6598 | 0.0119 | -0.0369 |
| Step 550,000 | 3,091 | 0.1511 | 0.0664 | 0.1573 | 0.1116 | 0.1963 | 1.9442 | 0.0101 | 0.0000 |
| Step 600,000 | 3,367 | 0.5752 | 0.2687 | 0.0520 | 0.1777 | 0.1226 | 1.7017 | 0.0099 | 0.0000 |
| Step 650,000 | 3,647 | 0.6673 | 0.1159 | 0.2090 | 0.1284 | 0.2598 | 2.2091 | 0.0172 | -0.0185 |
| Step 700,000 | 3,916 | 0.3592 | 0.1421 | 0.0828 | 0.1398 | 0.1674 | 2.4073 | 0.0167 | -0.0554 |
| Step 750,000 | 4,174 | 0.5622 | 0.0451 | 0.1600 | 0.1006 | 0.2324 | 2.7539 | 0.0210 | -0.0698 |
| Step 800,000 | 4,442 | 0.6426 | 0.1117 | 0.2494 | 0.1454 | 0.2468 | 2.3417 | 0.0162 | -0.1846 |
| Step 850,000 | 4,709 | 0.8086 | 0.3898 | 0.3789 | 0.2304 | 0.3582 | 2.8867 | 0.0241 | -0.1662 |
| Step 900,000 | 4,978 | 0.6622 | 0.1090 | 0.0172 | 0.1424 | 0.1682 | 2.7962 | 0.0180 | -0.1846 |
| Step 950,000 | 5,237 | 0.5840 | 0.1756 | 0.2349 | 0.1892 | 0.3164 | 3.6160 | 0.0247 | -0.1047 |
| Step 1,000,000 | 5,494 | 0.3767 | 0.0602 | 0.0610 | 0.1223 | 0.1552 | 3.1516 | 0.0208 | -0.0349 |
| Step 1,050,000 | 5,768 | 0.7063 | 0.0586 | 0.1169 | 0.1290 | 0.2130 | 3.6072 | 0.0259 | -0.0739 |
| Step 1,100,000 | 6,039 | 0.2819 | 0.0465 | 0.0278 | 0.1237 | 0.1574 | 4.1180 | 0.0311 | 0.0000 |
| Step 1,150,000 | 6,314 | 0.3931 | 0.1418 | 0.0202 | 0.1225 | 0.1313 | 3.8260 | 0.0362 | -0.0554 |
| Step 1,200,000 | 6,576 | 0.3852 | 0.1988 | 0.0358 | 0.1723 | 0.1450 | 4.1264 | 0.0374 | -0.1846 |
| Step 1,250,000 | 6,841 | -0.1472 | 0.1819 | 0.0660 | 0.1724 | 0.1877 | 5.0009 | 0.0452 | -0.0698 |
| Step 1,300,000 | 7,103 | 0.6091 | 0.1986 | 0.1383 | 0.1868 | 0.2550 | 4.5823 | 0.0420 | 0.1396 |
| Step 1,350,000 | 7,356 | 0.5284 | 0.4224 | 0.0802 | 0.2958 | 0.1897 | 4.3889 | 0.0304 | 0.2094 |
| Step 1,400,000 | 7,621 | 0.4680 | 0.0306 | 0.0771 | 0.1234 | 0.1997 | 5.4216 | 0.0505 | 0.0698 |
| Step 1,450,000 | 7,880 | 0.2306 | 0.0110 | 0.0152 | 0.1039 | 0.1575 | 5.0246 | 0.0442 | 0.0000 |
| Step 1,500,000 | 8,130 | 0.2804 | 0.1732 | 0.0383 | 0.1588 | 0.1683 | 4.6376 | 0.0420 | -0.0739 |
| Step 1,550,000 | 8,374 | 0.4930 | 0.2526 | 0.1257 | 0.2598 | 0.2452 | 4.9873 | 0.0429 | -0.0739 |
| Step 1,600,000 | 8,614 | 0.8029 | 0.2545 | 0.3777 | 0.2024 | 0.3457 | 5.2450 | 0.0616 | 0.0000 |
| Step 1,650,000 | 8,861 | 0.5746 | 0.0046 | 0.1192 | 0.1193 | 0.2250 | 5.8185 | 0.0533 | -0.0923 |
| Step 1,700,000 | 9,117 | 0.6647 | 0.1605 | 0.1855 | 0.1654 | 0.2946 | 5.8835 | 0.0632 | -0.0369 |
| Step 1,750,000 | 9,369 | 0.1355 | 0.0490 | 0.0204 | 0.1429 | 0.1813 | 5.5083 | 0.0418 | 0.1396 |
| Step 1,800,000 | 9,619 | 0.5308 | 0.0781 | 0.0632 | 0.1547 | 0.2010 | 5.1160 | 0.0432 | 0.1396 |
| Step 1,850,000 | 9,866 | 0.2085 | 0.1296 | -0.0043 | 0.1453 | 0.1492 | 5.2607 | 0.0490 | 0.0554 |
| Step 1,900,000 | 10,120 | 0.5144 | 0.1767 | 0.2032 | 0.1884 | 0.2653 | 4.2630 | 0.0285 | 0.0000 |
| Step 1,950,000 | 10,377 | 0.3404 | 0.0238 | 0.0732 | 0.1168 | 0.1982 | 4.6528 | 0.0303 | 0.0185 |
| Step 2,000,000 | 10,632 | 0.0974 | 0.0430 | 0.0387 | 0.1319 | 0.1544 | 4.2074 | 0.0394 | 0.1108 |
| Step 2,050,000 | 10,885 | 0.4909 | 0.2543 | 0.0121 | 0.2174 | 0.1498 | 4.4637 | 0.0331 | 0.0923 |
| Step 2,100,000 | 11,123 | 0.4651 | 0.0193 | 0.0857 | 0.1222 | 0.1969 | 3.7071 | 0.0277 | 0.0923 |
| Step 2,150,000 | 11,361 | 0.1082 | 0.0540 | -0.0016 | 0.1409 | 0.1739 | 4.5083 | 0.0309 | 0.0369 |
| Step 2,200,000 | 11,601 | 0.7425 | 0.3441 | 0.1477 | 0.2920 | 0.2634 | 3.5879 | 0.0224 | 0.1272 |
| Step 2,250,000 | 11,850 | 0.6905 | 0.2002 | 0.1487 | 0.2124 | 0.2473 | 3.6616 | 0.0293 | -0.0098 |
| Step 2,300,000 | 12,093 | 0.5796 | 0.2380 | 0.1449 | 0.2410 | 0.2539 | 2.6947 | 0.0141 | 0.0881 |
| Step 2,350,000 | 12,346 | 0.1987 | 0.0745 | -0.0088 | 0.1476 | 0.1471 | 3.5467 | 0.0256 | -0.0196 |
| Step 2,400,000 | 12,586 | 0.3545 | 0.1820 | 0.0879 | 0.2029 | 0.1827 | 3.4655 | 0.0196 | 0.0196 |
| Step 2,450,000 | 12,830 | 0.4677 | 0.1744 | 0.0591 | 0.1914 | 0.2024 | 3.3638 | 0.0178 | 0.0000 |
| Step 2,500,000 | 13,060 | 0.3255 | 0.0434 | 0.0078 | 0.1337 | 0.1740 | 4.1560 | 0.0274 | -0.0294 |
| Step 2,550,000 | 13,308 | 0.1971 | 0.1298 | 0.0368 | 0.1923 | 0.1960 | 4.0357 | 0.0216 | 0.0196 |
| Step 2,600,000 | 13,543 | 0.1053 | 0.1506 | 0.0232 | 0.1852 | 0.1670 | 3.8262 | 0.0207 | 0.0196 |
| Step 2,650,000 | 13,782 | 0.5938 | 0.3029 | 0.0520 | 0.2737 | 0.2264 | 4.0063 | 0.0203 | 0.0294 |
| Step 2,700,000 | 14,015 | 0.3833 | 0.2817 | 0.0224 | 0.3110 | 0.1719 | 2.9746 | 0.0107 | -0.0294 |
| Step 2,750,000 | 14,256 | 0.6639 | 0.3229 | 0.1502 | 0.2842 | 0.2483 | 3.1671 | 0.0164 | 0.0000 |
| Step 2,800,000 | 14,493 | 0.2606 | 0.0904 | 0.1299 | 0.1539 | 0.2629 | 4.8873 | 0.0288 | 0.0098 |
| Step 2,850,000 | 14,727 | 0.2867 | 0.0952 | -0.0064 | 0.1532 | 0.1457 | 4.3136 | 0.0257 | 0.0000 |
| Step 2,900,000 | 14,958 | 0.4923 | 0.2885 | 0.1301 | 0.2818 | 0.2912 | 4.4301 | 0.0253 | 0.0098 |
| Step 2,950,000 | 15,196 | 0.6773 | 0.2416 | 0.0584 | 0.2362 | 0.2090 | 2.9728 | 0.0179 | 0.0000 |
| Step 3,000,000 | 15,435 | 0.0097 | 0.1116 | 0.0130 | 0.1784 | 0.1750 | 3.3856 | 0.0201 | -0.0881 |
| Final | 15,435 | 0.0970 | 0.0219 | 0.0444 | 0.1204 | 0.1822 | 3.6031 | 0.0216 | 0.0783 |
