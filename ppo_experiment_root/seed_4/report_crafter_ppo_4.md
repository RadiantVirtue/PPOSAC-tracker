# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 4  
**Total episodes:** 14,763  
**Experiment root:** `ppo_experiment_root\seed_4`  
**Generated:** 2026-03-31 02:37

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep2 | 2 | 0.0039 | 0.7484 | 0.1829 | 0.0923 | 0.0438 | 0.1263 | 0.0006 | nan |
| collect_sapling @ ep3 | 3 | 0.0876 | 0.7057 | 0.2673 | 0.0792 | 0.0510 | 0.1611 | 0.0010 | -0.6547 |
| place_plant @ ep3 | 3 | 0.0328 | 0.7081 | 0.1990 | 0.0857 | 0.0413 | 0.1788 | 0.0011 | nan |
| collect_drink @ ep6 | 6 | 0.0100 | 0.8961 | 0.5458 | 0.0830 | 0.0518 | 0.1647 | 0.0012 | 0.6547 |
| collect_wood @ ep9 | 9 | -0.1238 | 0.8690 | 0.3032 | 0.0823 | 0.0393 | 0.2057 | 0.0020 | — |
| place_table @ ep34 | 34 | 0.5596 | 0.9236 | 0.0852 | 0.1583 | 0.0456 | 0.3280 | 0.0015 | -0.6547 |
| make_wood_sword @ ep61 | 61 | 0.7532 | 0.9340 | 0.5751 | 0.2296 | 0.0959 | 0.3186 | 0.0004 | -0.6547 |
| make_wood_pickaxe @ ep99 | 99 | 0.8109 | 0.8818 | 0.4431 | 0.2456 | 0.1509 | 0.5364 | 0.0002 | -0.3928 |
| defeat_zombie @ ep140 | 140 | 0.9342 | 0.9028 | 0.5890 | 0.3094 | 0.2172 | 0.7556 | 0.0005 | -0.3489 |
| eat_cow @ ep140 | 140 | 0.8722 | 0.7148 | 0.5257 | 0.2950 | 0.2303 | 0.6983 | 0.0005 | -0.3838 |
| Step 50,000 | 285 | 0.5399 | 0.1650 | 0.5385 | 0.1412 | 0.3011 | 0.3985 | 0.0001 | -0.4352 |
| defeat_skeleton @ ep527 | 527 | -0.8135 | 0.0585 | 0.0825 | 0.1280 | 0.1955 | 1.7782 | 0.0004 | 0.0772 |
| Step 100,000 | 576 | -0.5252 | 0.0496 | 0.1974 | 0.1055 | 0.3035 | 0.3939 | 0.0004 | 0.0386 |
| collect_stone @ ep580 | 580 | 0.1051 | 0.0980 | 0.2050 | 0.0863 | 0.2474 | 0.7460 | 0.0004 | -0.1745 |
| eat_plant @ ep849 | 849 | 0.2445 | 0.0475 | 0.2297 | 0.1100 | 0.2641 | 0.6171 | 0.0008 | -0.1047 |
| Step 150,000 | 873 | -0.8501 | -0.0137 | 0.4088 | 0.0983 | 0.3200 | 0.5276 | 0.0006 | 0.0739 |
| place_stone @ ep912 | 912 | -0.4237 | 0.0839 | 0.3376 | 0.1233 | 0.4050 | 0.6559 | 0.0008 | -0.1047 |
| Step 200,000 | 1,150 | 0.3802 | 0.3424 | 0.4657 | 0.1884 | 0.4322 | 0.6240 | 0.0009 | -0.2791 |
| Step 250,000 | 1,438 | 0.4604 | 0.0411 | 0.4206 | 0.1051 | 0.2935 | 0.8493 | 0.0017 | -0.3140 |
| collect_coal @ ep1685 | 1,685 | 0.0340 | 0.0670 | 0.1726 | 0.1157 | 0.1730 | 0.9561 | 0.0013 | -0.4187 |
| Step 300,000 | 1,722 | 0.7545 | 0.0853 | 0.4126 | 0.0927 | 0.3353 | 0.9074 | 0.0020 | -0.0554 |
| Step 350,000 | 1,995 | 0.1197 | 0.0843 | 0.0736 | 0.1364 | 0.1732 | 1.0230 | 0.0014 | -0.3489 |
| place_furnace @ ep2087 | 2,087 | 0.5509 | 0.1071 | 0.0318 | 0.1204 | 0.1778 | 1.5622 | 0.0019 | -0.0185 |
| make_stone_pickaxe @ ep2199 | 2,199 | 0.6467 | 0.2286 | 0.4858 | 0.1201 | 0.4316 | 1.0246 | 0.0020 | -0.1662 |
| Step 400,000 | 2,265 | 0.5460 | 0.4530 | 0.1083 | 0.2145 | 0.2164 | 1.1392 | 0.0021 | -0.0554 |
| Step 450,000 | 2,532 | 0.9196 | 0.3241 | 0.3852 | 0.2317 | 0.3747 | 1.1926 | 0.0033 | -0.4536 |
| Step 500,000 | 2,809 | 0.5991 | 0.4000 | 0.0541 | 0.2241 | 0.1399 | 1.5579 | 0.0061 | -0.2791 |
| Step 550,000 | 3,084 | 0.8883 | 0.5931 | 0.1036 | 0.3717 | 0.2442 | 1.7694 | 0.0080 | -0.2216 |
| Step 600,000 | 3,360 | 0.4325 | 0.3241 | 0.0731 | 0.1994 | 0.1437 | 1.3827 | 0.0035 | 0.0000 |
| Step 650,000 | 3,624 | 0.8324 | 0.4905 | 0.2688 | 0.2732 | 0.2663 | 1.2068 | 0.0023 | -0.3489 |
| Step 700,000 | 3,903 | 0.8075 | 0.2883 | 0.3083 | 0.1966 | 0.3041 | 1.5467 | 0.0042 | -0.2094 |
| Step 750,000 | 4,174 | 0.3454 | 0.0995 | -0.0072 | 0.1259 | 0.1501 | 1.8967 | 0.0057 | -0.1745 |
| Step 800,000 | 4,448 | 0.5776 | 0.0698 | 0.0955 | 0.1246 | 0.1949 | 2.0321 | 0.0062 | -0.3140 |
| Step 850,000 | 4,720 | 0.5475 | 0.2748 | 0.0904 | 0.1965 | 0.1951 | 2.7125 | 0.0090 | -0.0698 |
| Step 900,000 | 4,989 | 0.8316 | 0.2848 | 0.1905 | 0.2571 | 0.2835 | 2.4345 | 0.0069 | -0.1047 |
| Step 950,000 | 5,251 | 0.6536 | 0.2928 | 0.3225 | 0.2399 | 0.3969 | 2.9606 | 0.0118 | -0.0349 |
| Step 1,000,000 | 5,510 | 0.4950 | 0.1104 | 0.0809 | 0.1727 | 0.2364 | 3.3511 | 0.0104 | 0.0000 |
| Step 1,050,000 | 5,771 | 0.7803 | 0.1887 | 0.1036 | 0.1754 | 0.2291 | 2.9614 | 0.0138 | 0.0000 |
| Step 1,100,000 | 6,020 | 0.0680 | 0.2532 | 0.0957 | 0.1994 | 0.2002 | 2.9296 | 0.0101 | 0.0000 |
| Step 1,150,000 | 6,260 | 0.6722 | 0.3575 | 0.1351 | 0.2613 | 0.2510 | 2.8672 | 0.0105 | 0.0000 |
| Step 1,200,000 | 6,502 | 0.8197 | 0.5006 | 0.1590 | 0.2955 | 0.2828 | 2.7290 | 0.0132 | -0.0185 |
| make_stone_sword @ ep6592 | 6,592 | 0.7677 | 0.3099 | 0.2039 | 0.2433 | 0.3148 | 2.6974 | 0.0129 | -0.0923 |
| Step 1,250,000 | 6,748 | 0.5984 | 0.4395 | 0.3558 | 0.3106 | 0.3980 | 2.3697 | 0.0090 | -0.0369 |
| Step 1,300,000 | 6,991 | 0.4200 | 0.2451 | 0.0627 | 0.2042 | 0.2042 | 2.6489 | 0.0109 | -0.0554 |
| Step 1,350,000 | 7,233 | 0.0838 | 0.0335 | 0.1108 | 0.1112 | 0.1922 | 2.7621 | 0.0139 | 0.0554 |
| Step 1,400,000 | 7,473 | 0.3410 | 0.0850 | -0.0099 | 0.1424 | 0.1397 | 3.0155 | 0.0163 | -0.0185 |
| Step 1,450,000 | 7,720 | -0.3552 | 0.0850 | 0.0583 | 0.1528 | 0.1860 | 3.3938 | 0.0147 | 0.0587 |
| Step 1,500,000 | 7,964 | 0.1916 | 0.0934 | 0.0555 | 0.1509 | 0.1890 | 3.4761 | 0.0180 | -0.0369 |
| Step 1,550,000 | 8,209 | 0.7490 | 0.3656 | 0.1509 | 0.3067 | 0.3286 | 3.6146 | 0.0175 | 0.0369 |
| Step 1,600,000 | 8,437 | 0.4789 | 0.1213 | 0.0301 | 0.1709 | 0.1791 | 3.5569 | 0.0213 | 0.1477 |
| Step 1,650,000 | 8,667 | 0.4429 | 0.2433 | 0.1021 | 0.2108 | 0.2103 | 3.8852 | 0.0217 | 0.0923 |
| Step 1,700,000 | 8,911 | 0.5640 | 0.1629 | 0.0851 | 0.1807 | 0.2421 | 3.4308 | 0.0194 | 0.1292 |
| Step 1,750,000 | 9,141 | -0.0432 | 0.1278 | 0.0033 | 0.1737 | 0.2083 | 3.4526 | 0.0147 | 0.1477 |
| Step 1,800,000 | 9,371 | 0.3324 | 0.1216 | 0.0525 | 0.1855 | 0.1717 | 3.9135 | 0.0229 | 0.0369 |
| Step 1,850,000 | 9,597 | 0.6674 | 0.2132 | 0.1390 | 0.2217 | 0.2421 | 2.9509 | 0.0146 | 0.2251 |
| collect_iron @ ep9655 | 9,655 | 0.6446 | 0.4267 | 0.1424 | 0.2992 | 0.2840 | 2.7193 | 0.0122 | 0.1477 |
| Step 1,900,000 | 9,838 | 0.3088 | 0.1807 | 0.0326 | 0.1732 | 0.1685 | 3.7401 | 0.0249 | 0.0739 |
| Step 1,950,000 | 10,076 | 0.2744 | 0.1306 | 0.0723 | 0.1578 | 0.2081 | 4.0910 | 0.0226 | 0.1477 |
| Step 2,000,000 | 10,295 | 0.6471 | 0.3941 | 0.0313 | 0.3303 | 0.2261 | 3.1265 | 0.0109 | 0.0587 |
| Step 2,050,000 | 10,534 | 0.7824 | 0.4953 | 0.2425 | 0.4235 | 0.4155 | 3.8679 | 0.0148 | -0.0185 |
| Step 2,100,000 | 10,773 | 0.5048 | 0.2114 | 0.0267 | 0.2063 | 0.2053 | 3.7310 | 0.0182 | 0.0554 |
| Step 2,150,000 | 10,996 | 0.5607 | 0.3098 | 0.1626 | 0.2753 | 0.2893 | 4.2128 | 0.0199 | 0.0369 |
| Step 2,200,000 | 11,220 | 0.1893 | -0.0083 | -0.0202 | 0.1048 | 0.1386 | 4.1590 | 0.0225 | 0.0369 |
| Step 2,250,000 | 11,446 | 0.6513 | 0.3698 | 0.1530 | 0.2674 | 0.3327 | 3.5736 | 0.0169 | 0.2400 |
| Step 2,300,000 | 11,676 | 0.5254 | 0.0545 | 0.0160 | 0.1351 | 0.1563 | 3.4652 | 0.0203 | 0.1477 |
| Step 2,350,000 | 11,895 | 0.6569 | 0.1100 | 0.0926 | 0.1740 | 0.2234 | 3.4414 | 0.0177 | -0.0098 |
| Step 2,400,000 | 12,105 | 0.2389 | 0.1143 | 0.0045 | 0.1657 | 0.1745 | 3.1483 | 0.0129 | 0.1174 |
| Step 2,450,000 | 12,323 | 0.0770 | 0.0484 | 0.0773 | 0.1395 | 0.2228 | 4.4282 | 0.0298 | 0.1468 |
| Step 2,500,000 | 12,540 | 0.4976 | 0.1190 | 0.0810 | 0.1742 | 0.2274 | 3.7018 | 0.0207 | 0.1077 |
| Step 2,550,000 | 12,770 | 0.5057 | 0.0957 | 0.0732 | 0.1616 | 0.2798 | 4.0294 | 0.0248 | 0.1477 |
| Step 2,600,000 | 12,970 | 0.1646 | 0.0611 | 0.0199 | 0.1386 | 0.1694 | 4.3310 | 0.0258 | 0.0923 |
| Step 2,650,000 | 13,191 | 0.5557 | 0.2299 | 0.0353 | 0.2268 | 0.2417 | 3.4298 | 0.0143 | 0.2031 |
| Step 2,700,000 | 13,411 | 0.3681 | 0.0800 | 0.0973 | 0.1689 | 0.2242 | 2.8976 | 0.0139 | 0.1566 |
| Step 2,750,000 | 13,638 | 0.6234 | 0.2633 | 0.0888 | 0.2416 | 0.2427 | 3.4064 | 0.0199 | 0.1292 |
| Step 2,800,000 | 13,867 | 0.4309 | 0.0826 | 0.1406 | 0.1720 | 0.2407 | 3.8957 | 0.0342 | 0.1846 |
| Step 2,850,000 | 14,093 | 0.4649 | 0.1452 | 0.0934 | 0.1860 | 0.2125 | 2.9916 | 0.0191 | 0.0979 |
| Step 2,900,000 | 14,322 | 0.4617 | 0.0371 | 0.0168 | 0.1271 | 0.2080 | 3.1111 | 0.0176 | 0.1846 |
| Step 2,950,000 | 14,541 | 0.4367 | 0.1395 | 0.1076 | 0.1730 | 0.2332 | 3.1068 | 0.0129 | 0.1108 |
| Step 3,000,000 | 14,762 | 0.7168 | 0.1946 | 0.0779 | 0.2268 | 0.2619 | 3.3222 | 0.0136 | 0.1477 |
| Final | 14,763 | 0.2379 | 0.1151 | 0.0018 | 0.1633 | 0.1673 | 3.4067 | 0.0153 | 0.1477 |

---

## wake_up_ep2_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0039 |
| Coherence (Success) | 0.7484 |
| Coherence (Failure) | 0.1829 |
| Gradient Magnitude (Success) | 0.0923 |
| Gradient Magnitude (Failure) | 0.0438 |
| Activation Separation | 0.1263 |
| Cosine Distance | 0.0006 |
| Clusters | 1,437 |
| Noise Fraction | 0.2582 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_sapling_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0876 |
| Coherence (Success) | 0.7057 |
| Coherence (Failure) | 0.2673 |
| Gradient Magnitude (Success) | 0.0792 |
| Gradient Magnitude (Failure) | 0.0510 |
| Activation Separation | 0.1611 |
| Cosine Distance | 0.0010 |
| Clusters | 1,520 |
| Noise Fraction | 0.2841 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## place_plant_ep3_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0328 |
| Coherence (Success) | 0.7081 |
| Coherence (Failure) | 0.1990 |
| Gradient Magnitude (Success) | 0.0857 |
| Gradient Magnitude (Failure) | 0.0413 |
| Activation Separation | 0.1788 |
| Cosine Distance | 0.0011 |
| Clusters | 1,507 |
| Noise Fraction | 0.2713 |
| RSA Alignment (ρ) | nan |
| RSA Stimuli (3) | Wood, Wood Pickaxe, Zombie |

---

## collect_drink_ep6_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0100 |
| Coherence (Success) | 0.8961 |
| Coherence (Failure) | 0.5458 |
| Gradient Magnitude (Success) | 0.0830 |
| Gradient Magnitude (Failure) | 0.0518 |
| Activation Separation | 0.1647 |
| Cosine Distance | 0.0012 |
| Clusters | 1,364 |
| Noise Fraction | 0.2636 |
| RSA Alignment (ρ) | 0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## collect_wood_ep9_lower1.000_upper3.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1238 |
| Coherence (Success) | 0.8690 |
| Coherence (Failure) | 0.3032 |
| Gradient Magnitude (Success) | 0.0823 |
| Gradient Magnitude (Failure) | 0.0393 |
| Activation Separation | 0.2057 |
| Cosine Distance | 0.0020 |
| Clusters | 1,439 |
| Noise Fraction | 0.2718 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wood, Wood Pickaxe |

---

## place_table_ep34_lower2.000_upper3.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5596 |
| Coherence (Success) | 0.9236 |
| Coherence (Failure) | 0.0852 |
| Gradient Magnitude (Success) | 0.1583 |
| Gradient Magnitude (Failure) | 0.0456 |
| Activation Separation | 0.3280 |
| Cosine Distance | 0.0015 |
| Clusters | 1,421 |
| Noise Fraction | 0.2704 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## make_wood_sword_ep61_lower2.900_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7532 |
| Coherence (Success) | 0.9340 |
| Coherence (Failure) | 0.5751 |
| Gradient Magnitude (Success) | 0.2296 |
| Gradient Magnitude (Failure) | 0.0959 |
| Activation Separation | 0.3186 |
| Cosine Distance | 0.0004 |
| Clusters | 1,514 |
| Noise Fraction | 0.2386 |
| RSA Alignment (ρ) | -0.6547 |
| RSA Stimuli (4) | Skeleton, Wood, Wood Pickaxe, Zombie |

---

## make_wood_pickaxe_ep99_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8109 |
| Coherence (Success) | 0.8818 |
| Coherence (Failure) | 0.4431 |
| Gradient Magnitude (Success) | 0.2456 |
| Gradient Magnitude (Failure) | 0.1509 |
| Activation Separation | 0.5364 |
| Cosine Distance | 0.0002 |
| Clusters | 1,592 |
| Noise Fraction | 0.2197 |
| RSA Alignment (ρ) | -0.3928 |
| RSA Stimuli (4) | Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_zombie_ep140_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9342 |
| Coherence (Success) | 0.9028 |
| Coherence (Failure) | 0.5890 |
| Gradient Magnitude (Success) | 0.3094 |
| Gradient Magnitude (Failure) | 0.2172 |
| Activation Separation | 0.7556 |
| Cosine Distance | 0.0005 |
| Clusters | 1,625 |
| Noise Fraction | 0.2501 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_cow_ep140_lower3.000_upper5.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8722 |
| Coherence (Success) | 0.7148 |
| Coherence (Failure) | 0.5257 |
| Gradient Magnitude (Success) | 0.2950 |
| Gradient Magnitude (Failure) | 0.2303 |
| Activation Separation | 0.6983 |
| Cosine Distance | 0.0005 |
| Clusters | 1,622 |
| Noise Fraction | 0.2539 |
| RSA Alignment (ρ) | -0.3838 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep285_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5399 |
| Coherence (Success) | 0.1650 |
| Coherence (Failure) | 0.5385 |
| Gradient Magnitude (Success) | 0.1412 |
| Gradient Magnitude (Failure) | 0.3011 |
| Activation Separation | 0.3985 |
| Cosine Distance | 0.0001 |
| Clusters | 1,456 |
| Noise Fraction | 0.2697 |
| RSA Alignment (ρ) | -0.4352 |
| RSA Stimuli (5) | Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## defeat_skeleton_ep527_lower3.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.8135 |
| Coherence (Success) | 0.0585 |
| Coherence (Failure) | 0.0825 |
| Gradient Magnitude (Success) | 0.1280 |
| Gradient Magnitude (Failure) | 0.1955 |
| Activation Separation | 1.7782 |
| Cosine Distance | 0.0004 |
| Clusters | 1,528 |
| Noise Fraction | 0.2781 |
| RSA Alignment (ρ) | 0.0772 |
| RSA Stimuli (6) | Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep576_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.5252 |
| Coherence (Success) | 0.0496 |
| Coherence (Failure) | 0.1974 |
| Gradient Magnitude (Success) | 0.1055 |
| Gradient Magnitude (Failure) | 0.3035 |
| Activation Separation | 0.3939 |
| Cosine Distance | 0.0004 |
| Clusters | 1,611 |
| Noise Fraction | 0.2663 |
| RSA Alignment (ρ) | 0.0386 |
| RSA Stimuli (6) | Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_stone_ep580_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1051 |
| Coherence (Success) | 0.0980 |
| Coherence (Failure) | 0.2050 |
| Gradient Magnitude (Success) | 0.0863 |
| Gradient Magnitude (Failure) | 0.2474 |
| Activation Separation | 0.7460 |
| Cosine Distance | 0.0004 |
| Clusters | 1,568 |
| Noise Fraction | 0.2639 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## eat_plant_ep849_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2445 |
| Coherence (Success) | 0.0475 |
| Coherence (Failure) | 0.2297 |
| Gradient Magnitude (Success) | 0.1100 |
| Gradient Magnitude (Failure) | 0.2641 |
| Activation Separation | 0.6171 |
| Cosine Distance | 0.0008 |
| Clusters | 1,625 |
| Noise Fraction | 0.2833 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep873_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.8501 |
| Coherence (Success) | -0.0137 |
| Coherence (Failure) | 0.4088 |
| Gradient Magnitude (Success) | 0.0983 |
| Gradient Magnitude (Failure) | 0.3200 |
| Activation Separation | 0.5276 |
| Cosine Distance | 0.0006 |
| Clusters | 1,643 |
| Noise Fraction | 0.2770 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## place_stone_ep912_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.4237 |
| Coherence (Success) | 0.0839 |
| Coherence (Failure) | 0.3376 |
| Gradient Magnitude (Success) | 0.1233 |
| Gradient Magnitude (Failure) | 0.4050 |
| Activation Separation | 0.6559 |
| Cosine Distance | 0.0008 |
| Clusters | 1,513 |
| Noise Fraction | 0.2809 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1150_lower3.900_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3802 |
| Coherence (Success) | 0.3424 |
| Coherence (Failure) | 0.4657 |
| Gradient Magnitude (Success) | 0.1884 |
| Gradient Magnitude (Failure) | 0.4322 |
| Activation Separation | 0.6240 |
| Cosine Distance | 0.0009 |
| Clusters | 1,625 |
| Noise Fraction | 0.2715 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1438_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4604 |
| Coherence (Success) | 0.0411 |
| Coherence (Failure) | 0.4206 |
| Gradient Magnitude (Success) | 0.1051 |
| Gradient Magnitude (Failure) | 0.2935 |
| Activation Separation | 0.8493 |
| Cosine Distance | 0.0017 |
| Clusters | 1,505 |
| Noise Fraction | 0.2801 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## collect_coal_ep1685_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0340 |
| Coherence (Success) | 0.0670 |
| Coherence (Failure) | 0.1726 |
| Gradient Magnitude (Success) | 0.1157 |
| Gradient Magnitude (Failure) | 0.1730 |
| Activation Separation | 0.9561 |
| Cosine Distance | 0.0013 |
| Clusters | 1,561 |
| Noise Fraction | 0.2447 |
| RSA Alignment (ρ) | -0.4187 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep1722_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7545 |
| Coherence (Success) | 0.0853 |
| Coherence (Failure) | 0.4126 |
| Gradient Magnitude (Success) | 0.0927 |
| Gradient Magnitude (Failure) | 0.3353 |
| Activation Separation | 0.9074 |
| Cosine Distance | 0.0020 |
| Clusters | 1,639 |
| Noise Fraction | 0.2566 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep1995_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1197 |
| Coherence (Success) | 0.0843 |
| Coherence (Failure) | 0.0736 |
| Gradient Magnitude (Success) | 0.1364 |
| Gradient Magnitude (Failure) | 0.1732 |
| Activation Separation | 1.0230 |
| Cosine Distance | 0.0014 |
| Clusters | 1,537 |
| Noise Fraction | 0.2847 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## place_furnace_ep2087_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5509 |
| Coherence (Success) | 0.1071 |
| Coherence (Failure) | 0.0318 |
| Gradient Magnitude (Success) | 0.1204 |
| Gradient Magnitude (Failure) | 0.1778 |
| Activation Separation | 1.5622 |
| Cosine Distance | 0.0019 |
| Clusters | 1,509 |
| Noise Fraction | 0.2893 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_pickaxe_ep2199_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6467 |
| Coherence (Success) | 0.2286 |
| Coherence (Failure) | 0.4858 |
| Gradient Magnitude (Success) | 0.1201 |
| Gradient Magnitude (Failure) | 0.4316 |
| Activation Separation | 1.0246 |
| Cosine Distance | 0.0020 |
| Clusters | 1,535 |
| Noise Fraction | 0.2816 |
| RSA Alignment (ρ) | -0.1662 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2265_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5460 |
| Coherence (Success) | 0.4530 |
| Coherence (Failure) | 0.1083 |
| Gradient Magnitude (Success) | 0.2145 |
| Gradient Magnitude (Failure) | 0.2164 |
| Activation Separation | 1.1392 |
| Cosine Distance | 0.0021 |
| Clusters | 1,500 |
| Noise Fraction | 0.2773 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep2532_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9196 |
| Coherence (Success) | 0.3241 |
| Coherence (Failure) | 0.3852 |
| Gradient Magnitude (Success) | 0.2317 |
| Gradient Magnitude (Failure) | 0.3747 |
| Activation Separation | 1.1926 |
| Cosine Distance | 0.0033 |
| Clusters | 1,545 |
| Noise Fraction | 0.2773 |
| RSA Alignment (ρ) | -0.4536 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep2809_lower4.000_upper6.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5991 |
| Coherence (Success) | 0.4000 |
| Coherence (Failure) | 0.0541 |
| Gradient Magnitude (Success) | 0.2241 |
| Gradient Magnitude (Failure) | 0.1399 |
| Activation Separation | 1.5579 |
| Cosine Distance | 0.0061 |
| Clusters | 1,622 |
| Noise Fraction | 0.2679 |
| RSA Alignment (ρ) | -0.2791 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3084_lower4.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8883 |
| Coherence (Success) | 0.5931 |
| Coherence (Failure) | 0.1036 |
| Gradient Magnitude (Success) | 0.3717 |
| Gradient Magnitude (Failure) | 0.2442 |
| Activation Separation | 1.7694 |
| Cosine Distance | 0.0080 |
| Clusters | 1,627 |
| Noise Fraction | 0.2684 |
| RSA Alignment (ρ) | -0.2216 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep3360_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4325 |
| Coherence (Success) | 0.3241 |
| Coherence (Failure) | 0.0731 |
| Gradient Magnitude (Success) | 0.1994 |
| Gradient Magnitude (Failure) | 0.1437 |
| Activation Separation | 1.3827 |
| Cosine Distance | 0.0035 |
| Clusters | 1,566 |
| Noise Fraction | 0.2680 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3624_lower5.000_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8324 |
| Coherence (Success) | 0.4905 |
| Coherence (Failure) | 0.2688 |
| Gradient Magnitude (Success) | 0.2732 |
| Gradient Magnitude (Failure) | 0.2663 |
| Activation Separation | 1.2068 |
| Cosine Distance | 0.0023 |
| Clusters | 1,414 |
| Noise Fraction | 0.3128 |
| RSA Alignment (ρ) | -0.3489 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep3903_lower5.900_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8075 |
| Coherence (Success) | 0.2883 |
| Coherence (Failure) | 0.3083 |
| Gradient Magnitude (Success) | 0.1966 |
| Gradient Magnitude (Failure) | 0.3041 |
| Activation Separation | 1.5467 |
| Cosine Distance | 0.0042 |
| Clusters | 1,299 |
| Noise Fraction | 0.3283 |
| RSA Alignment (ρ) | -0.2094 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4174_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3454 |
| Coherence (Success) | 0.0995 |
| Coherence (Failure) | -0.0072 |
| Gradient Magnitude (Success) | 0.1259 |
| Gradient Magnitude (Failure) | 0.1501 |
| Activation Separation | 1.8967 |
| Cosine Distance | 0.0057 |
| Clusters | 1,405 |
| Noise Fraction | 0.3080 |
| RSA Alignment (ρ) | -0.1745 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4448_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5776 |
| Coherence (Success) | 0.0698 |
| Coherence (Failure) | 0.0955 |
| Gradient Magnitude (Success) | 0.1246 |
| Gradient Magnitude (Failure) | 0.1949 |
| Activation Separation | 2.0321 |
| Cosine Distance | 0.0062 |
| Clusters | 1,361 |
| Noise Fraction | 0.3270 |
| RSA Alignment (ρ) | -0.3140 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4720_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5475 |
| Coherence (Success) | 0.2748 |
| Coherence (Failure) | 0.0904 |
| Gradient Magnitude (Success) | 0.1965 |
| Gradient Magnitude (Failure) | 0.1951 |
| Activation Separation | 2.7125 |
| Cosine Distance | 0.0090 |
| Clusters | 1,401 |
| Noise Fraction | 0.2988 |
| RSA Alignment (ρ) | -0.0698 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep4989_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8316 |
| Coherence (Success) | 0.2848 |
| Coherence (Failure) | 0.1905 |
| Gradient Magnitude (Success) | 0.2571 |
| Gradient Magnitude (Failure) | 0.2835 |
| Activation Separation | 2.4345 |
| Cosine Distance | 0.0069 |
| Clusters | 1,249 |
| Noise Fraction | 0.3025 |
| RSA Alignment (ρ) | -0.1047 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5251_lower6.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6536 |
| Coherence (Success) | 0.2928 |
| Coherence (Failure) | 0.3225 |
| Gradient Magnitude (Success) | 0.2399 |
| Gradient Magnitude (Failure) | 0.3969 |
| Activation Separation | 2.9606 |
| Cosine Distance | 0.0118 |
| Clusters | 1,460 |
| Noise Fraction | 0.3247 |
| RSA Alignment (ρ) | -0.0349 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5510_lower6.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4950 |
| Coherence (Success) | 0.1104 |
| Coherence (Failure) | 0.0809 |
| Gradient Magnitude (Success) | 0.1727 |
| Gradient Magnitude (Failure) | 0.2364 |
| Activation Separation | 3.3511 |
| Cosine Distance | 0.0104 |
| Clusters | 1,295 |
| Noise Fraction | 0.3607 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep5771_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7803 |
| Coherence (Success) | 0.1887 |
| Coherence (Failure) | 0.1036 |
| Gradient Magnitude (Success) | 0.1754 |
| Gradient Magnitude (Failure) | 0.2291 |
| Activation Separation | 2.9614 |
| Cosine Distance | 0.0138 |
| Clusters | 1,245 |
| Noise Fraction | 0.3393 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6020_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0680 |
| Coherence (Success) | 0.2532 |
| Coherence (Failure) | 0.0957 |
| Gradient Magnitude (Success) | 0.1994 |
| Gradient Magnitude (Failure) | 0.2002 |
| Activation Separation | 2.9296 |
| Cosine Distance | 0.0101 |
| Clusters | 1,329 |
| Noise Fraction | 0.3214 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6260_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6722 |
| Coherence (Success) | 0.3575 |
| Coherence (Failure) | 0.1351 |
| Gradient Magnitude (Success) | 0.2613 |
| Gradient Magnitude (Failure) | 0.2510 |
| Activation Separation | 2.8672 |
| Cosine Distance | 0.0105 |
| Clusters | 1,437 |
| Noise Fraction | 0.3376 |
| RSA Alignment (ρ) | 0.0000 |
| RSA Stimuli (6) | Coal, Skeleton, Stone, Wood, Wood Pickaxe, Zombie |

---

## ep6502_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8197 |
| Coherence (Success) | 0.5006 |
| Coherence (Failure) | 0.1590 |
| Gradient Magnitude (Success) | 0.2955 |
| Gradient Magnitude (Failure) | 0.2828 |
| Activation Separation | 2.7290 |
| Cosine Distance | 0.0132 |
| Clusters | 1,267 |
| Noise Fraction | 0.3875 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## make_stone_sword_ep6592_lower6.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7677 |
| Coherence (Success) | 0.3099 |
| Coherence (Failure) | 0.2039 |
| Gradient Magnitude (Success) | 0.2433 |
| Gradient Magnitude (Failure) | 0.3148 |
| Activation Separation | 2.6974 |
| Cosine Distance | 0.0129 |
| Clusters | 1,248 |
| Noise Fraction | 0.3647 |
| RSA Alignment (ρ) | -0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6748_lower6.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5984 |
| Coherence (Success) | 0.4395 |
| Coherence (Failure) | 0.3558 |
| Gradient Magnitude (Success) | 0.3106 |
| Gradient Magnitude (Failure) | 0.3980 |
| Activation Separation | 2.3697 |
| Cosine Distance | 0.0090 |
| Clusters | 1,266 |
| Noise Fraction | 0.3401 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep6991_lower6.900_upper9.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4200 |
| Coherence (Success) | 0.2451 |
| Coherence (Failure) | 0.0627 |
| Gradient Magnitude (Success) | 0.2042 |
| Gradient Magnitude (Failure) | 0.2042 |
| Activation Separation | 2.6489 |
| Cosine Distance | 0.0109 |
| Clusters | 1,502 |
| Noise Fraction | 0.3522 |
| RSA Alignment (ρ) | -0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7233_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0838 |
| Coherence (Success) | 0.0335 |
| Coherence (Failure) | 0.1108 |
| Gradient Magnitude (Success) | 0.1112 |
| Gradient Magnitude (Failure) | 0.1922 |
| Activation Separation | 2.7621 |
| Cosine Distance | 0.0139 |
| Clusters | 1,491 |
| Noise Fraction | 0.3320 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7473_lower6.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3410 |
| Coherence (Success) | 0.0850 |
| Coherence (Failure) | -0.0099 |
| Gradient Magnitude (Success) | 0.1424 |
| Gradient Magnitude (Failure) | 0.1397 |
| Activation Separation | 3.0155 |
| Cosine Distance | 0.0163 |
| Clusters | 1,380 |
| Noise Fraction | 0.3639 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7720_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.3552 |
| Coherence (Success) | 0.0850 |
| Coherence (Failure) | 0.0583 |
| Gradient Magnitude (Success) | 0.1528 |
| Gradient Magnitude (Failure) | 0.1860 |
| Activation Separation | 3.3938 |
| Cosine Distance | 0.0147 |
| Clusters | 1,583 |
| Noise Fraction | 0.3416 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep7964_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1916 |
| Coherence (Success) | 0.0934 |
| Coherence (Failure) | 0.0555 |
| Gradient Magnitude (Success) | 0.1509 |
| Gradient Magnitude (Failure) | 0.1890 |
| Activation Separation | 3.4761 |
| Cosine Distance | 0.0180 |
| Clusters | 1,359 |
| Noise Fraction | 0.3402 |
| RSA Alignment (ρ) | -0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8209_lower6.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7490 |
| Coherence (Success) | 0.3656 |
| Coherence (Failure) | 0.1509 |
| Gradient Magnitude (Success) | 0.3067 |
| Gradient Magnitude (Failure) | 0.3286 |
| Activation Separation | 3.6146 |
| Cosine Distance | 0.0175 |
| Clusters | 1,651 |
| Noise Fraction | 0.3268 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8437_lower6.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4789 |
| Coherence (Success) | 0.1213 |
| Coherence (Failure) | 0.0301 |
| Gradient Magnitude (Success) | 0.1709 |
| Gradient Magnitude (Failure) | 0.1791 |
| Activation Separation | 3.5569 |
| Cosine Distance | 0.0213 |
| Clusters | 1,581 |
| Noise Fraction | 0.3253 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8667_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4429 |
| Coherence (Success) | 0.2433 |
| Coherence (Failure) | 0.1021 |
| Gradient Magnitude (Success) | 0.2108 |
| Gradient Magnitude (Failure) | 0.2103 |
| Activation Separation | 3.8852 |
| Cosine Distance | 0.0217 |
| Clusters | 1,574 |
| Noise Fraction | 0.3043 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep8911_lower7.000_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5640 |
| Coherence (Success) | 0.1629 |
| Coherence (Failure) | 0.0851 |
| Gradient Magnitude (Success) | 0.1807 |
| Gradient Magnitude (Failure) | 0.2421 |
| Activation Separation | 3.4308 |
| Cosine Distance | 0.0194 |
| Clusters | 1,417 |
| Noise Fraction | 0.3565 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9141_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0432 |
| Coherence (Success) | 0.1278 |
| Coherence (Failure) | 0.0033 |
| Gradient Magnitude (Success) | 0.1737 |
| Gradient Magnitude (Failure) | 0.2083 |
| Activation Separation | 3.4526 |
| Cosine Distance | 0.0147 |
| Clusters | 1,269 |
| Noise Fraction | 0.3279 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9371_lower7.900_upper10.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3324 |
| Coherence (Success) | 0.1216 |
| Coherence (Failure) | 0.0525 |
| Gradient Magnitude (Success) | 0.1855 |
| Gradient Magnitude (Failure) | 0.1717 |
| Activation Separation | 3.9135 |
| Cosine Distance | 0.0229 |
| Clusters | 1,278 |
| Noise Fraction | 0.3129 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9597_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6674 |
| Coherence (Success) | 0.2132 |
| Coherence (Failure) | 0.1390 |
| Gradient Magnitude (Success) | 0.2217 |
| Gradient Magnitude (Failure) | 0.2421 |
| Activation Separation | 2.9509 |
| Cosine Distance | 0.0146 |
| Clusters | 1,193 |
| Noise Fraction | 0.3542 |
| RSA Alignment (ρ) | 0.2251 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## collect_iron_ep9655_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6446 |
| Coherence (Success) | 0.4267 |
| Coherence (Failure) | 0.1424 |
| Gradient Magnitude (Success) | 0.2992 |
| Gradient Magnitude (Failure) | 0.2840 |
| Activation Separation | 2.7193 |
| Cosine Distance | 0.0122 |
| Clusters | 1,232 |
| Noise Fraction | 0.3517 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep9838_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3088 |
| Coherence (Success) | 0.1807 |
| Coherence (Failure) | 0.0326 |
| Gradient Magnitude (Success) | 0.1732 |
| Gradient Magnitude (Failure) | 0.1685 |
| Activation Separation | 3.7401 |
| Cosine Distance | 0.0249 |
| Clusters | 1,115 |
| Noise Fraction | 0.3282 |
| RSA Alignment (ρ) | 0.0739 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10076_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2744 |
| Coherence (Success) | 0.1306 |
| Coherence (Failure) | 0.0723 |
| Gradient Magnitude (Success) | 0.1578 |
| Gradient Magnitude (Failure) | 0.2081 |
| Activation Separation | 4.0910 |
| Cosine Distance | 0.0226 |
| Clusters | 1,287 |
| Noise Fraction | 0.3389 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10295_lower8.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6471 |
| Coherence (Success) | 0.3941 |
| Coherence (Failure) | 0.0313 |
| Gradient Magnitude (Success) | 0.3303 |
| Gradient Magnitude (Failure) | 0.2261 |
| Activation Separation | 3.1265 |
| Cosine Distance | 0.0109 |
| Clusters | 1,104 |
| Noise Fraction | 0.3335 |
| RSA Alignment (ρ) | 0.0587 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10534_lower7.000_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7824 |
| Coherence (Success) | 0.4953 |
| Coherence (Failure) | 0.2425 |
| Gradient Magnitude (Success) | 0.4235 |
| Gradient Magnitude (Failure) | 0.4155 |
| Activation Separation | 3.8679 |
| Cosine Distance | 0.0148 |
| Clusters | 1,127 |
| Noise Fraction | 0.3633 |
| RSA Alignment (ρ) | -0.0185 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10773_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5048 |
| Coherence (Success) | 0.2114 |
| Coherence (Failure) | 0.0267 |
| Gradient Magnitude (Success) | 0.2063 |
| Gradient Magnitude (Failure) | 0.2053 |
| Activation Separation | 3.7310 |
| Cosine Distance | 0.0182 |
| Clusters | 1,169 |
| Noise Fraction | 0.3369 |
| RSA Alignment (ρ) | 0.0554 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep10996_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5607 |
| Coherence (Success) | 0.3098 |
| Coherence (Failure) | 0.1626 |
| Gradient Magnitude (Success) | 0.2753 |
| Gradient Magnitude (Failure) | 0.2893 |
| Activation Separation | 4.2128 |
| Cosine Distance | 0.0199 |
| Clusters | 1,282 |
| Noise Fraction | 0.3372 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11220_lower8.900_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1893 |
| Coherence (Success) | -0.0083 |
| Coherence (Failure) | -0.0202 |
| Gradient Magnitude (Success) | 0.1048 |
| Gradient Magnitude (Failure) | 0.1386 |
| Activation Separation | 4.1590 |
| Cosine Distance | 0.0225 |
| Clusters | 1,367 |
| Noise Fraction | 0.3442 |
| RSA Alignment (ρ) | 0.0369 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11446_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6513 |
| Coherence (Success) | 0.3698 |
| Coherence (Failure) | 0.1530 |
| Gradient Magnitude (Success) | 0.2674 |
| Gradient Magnitude (Failure) | 0.3327 |
| Activation Separation | 3.5736 |
| Cosine Distance | 0.0169 |
| Clusters | 1,192 |
| Noise Fraction | 0.3504 |
| RSA Alignment (ρ) | 0.2400 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11676_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5254 |
| Coherence (Success) | 0.0545 |
| Coherence (Failure) | 0.0160 |
| Gradient Magnitude (Success) | 0.1351 |
| Gradient Magnitude (Failure) | 0.1563 |
| Activation Separation | 3.4652 |
| Cosine Distance | 0.0203 |
| Clusters | 1,444 |
| Noise Fraction | 0.3515 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep11895_lower8.450_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6569 |
| Coherence (Success) | 0.1100 |
| Coherence (Failure) | 0.0926 |
| Gradient Magnitude (Success) | 0.1740 |
| Gradient Magnitude (Failure) | 0.2234 |
| Activation Separation | 3.4414 |
| Cosine Distance | 0.0177 |
| Clusters | 1,310 |
| Noise Fraction | 0.3622 |
| RSA Alignment (ρ) | -0.0098 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12105_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2389 |
| Coherence (Success) | 0.1143 |
| Coherence (Failure) | 0.0045 |
| Gradient Magnitude (Success) | 0.1657 |
| Gradient Magnitude (Failure) | 0.1745 |
| Activation Separation | 3.1483 |
| Cosine Distance | 0.0129 |
| Clusters | 1,351 |
| Noise Fraction | 0.3489 |
| RSA Alignment (ρ) | 0.1174 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12323_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0770 |
| Coherence (Success) | 0.0484 |
| Coherence (Failure) | 0.0773 |
| Gradient Magnitude (Success) | 0.1395 |
| Gradient Magnitude (Failure) | 0.2228 |
| Activation Separation | 4.4282 |
| Cosine Distance | 0.0298 |
| Clusters | 1,469 |
| Noise Fraction | 0.3288 |
| RSA Alignment (ρ) | 0.1468 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12540_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4976 |
| Coherence (Success) | 0.1190 |
| Coherence (Failure) | 0.0810 |
| Gradient Magnitude (Success) | 0.1742 |
| Gradient Magnitude (Failure) | 0.2274 |
| Activation Separation | 3.7018 |
| Cosine Distance | 0.0207 |
| Clusters | 1,351 |
| Noise Fraction | 0.3360 |
| RSA Alignment (ρ) | 0.1077 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12770_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5057 |
| Coherence (Success) | 0.0957 |
| Coherence (Failure) | 0.0732 |
| Gradient Magnitude (Success) | 0.1616 |
| Gradient Magnitude (Failure) | 0.2798 |
| Activation Separation | 4.0294 |
| Cosine Distance | 0.0248 |
| Clusters | 1,508 |
| Noise Fraction | 0.3058 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep12970_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1646 |
| Coherence (Success) | 0.0611 |
| Coherence (Failure) | 0.0199 |
| Gradient Magnitude (Success) | 0.1386 |
| Gradient Magnitude (Failure) | 0.1694 |
| Activation Separation | 4.3310 |
| Cosine Distance | 0.0258 |
| Clusters | 1,372 |
| Noise Fraction | 0.3350 |
| RSA Alignment (ρ) | 0.0923 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13191_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5557 |
| Coherence (Success) | 0.2299 |
| Coherence (Failure) | 0.0353 |
| Gradient Magnitude (Success) | 0.2268 |
| Gradient Magnitude (Failure) | 0.2417 |
| Activation Separation | 3.4298 |
| Cosine Distance | 0.0143 |
| Clusters | 1,304 |
| Noise Fraction | 0.3515 |
| RSA Alignment (ρ) | 0.2031 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13411_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3681 |
| Coherence (Success) | 0.0800 |
| Coherence (Failure) | 0.0973 |
| Gradient Magnitude (Success) | 0.1689 |
| Gradient Magnitude (Failure) | 0.2242 |
| Activation Separation | 2.8976 |
| Cosine Distance | 0.0139 |
| Clusters | 1,301 |
| Noise Fraction | 0.3509 |
| RSA Alignment (ρ) | 0.1566 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13638_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6234 |
| Coherence (Success) | 0.2633 |
| Coherence (Failure) | 0.0888 |
| Gradient Magnitude (Success) | 0.2416 |
| Gradient Magnitude (Failure) | 0.2427 |
| Activation Separation | 3.4064 |
| Cosine Distance | 0.0199 |
| Clusters | 1,098 |
| Noise Fraction | 0.3534 |
| RSA Alignment (ρ) | 0.1292 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep13867_lower8.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4309 |
| Coherence (Success) | 0.0826 |
| Coherence (Failure) | 0.1406 |
| Gradient Magnitude (Success) | 0.1720 |
| Gradient Magnitude (Failure) | 0.2407 |
| Activation Separation | 3.8957 |
| Cosine Distance | 0.0342 |
| Clusters | 1,353 |
| Noise Fraction | 0.3496 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14093_lower8.900_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4649 |
| Coherence (Success) | 0.1452 |
| Coherence (Failure) | 0.0934 |
| Gradient Magnitude (Success) | 0.1860 |
| Gradient Magnitude (Failure) | 0.2125 |
| Activation Separation | 2.9916 |
| Cosine Distance | 0.0191 |
| Clusters | 1,365 |
| Noise Fraction | 0.3708 |
| RSA Alignment (ρ) | 0.0979 |
| RSA Stimuli (8) | Coal, Iron, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14322_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4617 |
| Coherence (Success) | 0.0371 |
| Coherence (Failure) | 0.0168 |
| Gradient Magnitude (Success) | 0.1271 |
| Gradient Magnitude (Failure) | 0.2080 |
| Activation Separation | 3.1111 |
| Cosine Distance | 0.0176 |
| Clusters | 1,178 |
| Noise Fraction | 0.3047 |
| RSA Alignment (ρ) | 0.1846 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14541_lower9.000_upper11.450

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4367 |
| Coherence (Success) | 0.1395 |
| Coherence (Failure) | 0.1076 |
| Gradient Magnitude (Success) | 0.1730 |
| Gradient Magnitude (Failure) | 0.2332 |
| Activation Separation | 3.1068 |
| Cosine Distance | 0.0129 |
| Clusters | 1,251 |
| Noise Fraction | 0.3336 |
| RSA Alignment (ρ) | 0.1108 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14762_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7168 |
| Coherence (Success) | 0.1946 |
| Coherence (Failure) | 0.0779 |
| Gradient Magnitude (Success) | 0.2268 |
| Gradient Magnitude (Failure) | 0.2619 |
| Activation Separation | 3.3222 |
| Cosine Distance | 0.0136 |
| Clusters | 1,158 |
| Noise Fraction | 0.3288 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## ep14763_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2379 |
| Coherence (Success) | 0.1151 |
| Coherence (Failure) | 0.0018 |
| Gradient Magnitude (Success) | 0.1633 |
| Gradient Magnitude (Failure) | 0.1673 |
| Activation Separation | 3.4067 |
| Cosine Distance | 0.0153 |
| Clusters | 1,268 |
| Noise Fraction | 0.3123 |
| RSA Alignment (ρ) | 0.1477 |
| RSA Stimuli (7) | Coal, Skeleton, Stone, Stone Pickaxe, Wood, Wood Pickaxe, Zombie |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep2 | 2 | 0.0039 | 0.7484 | 0.1829 | 0.0923 | 0.0438 | 0.1263 | 0.0006 | nan |
| collect_sapling @ ep3 | 3 | 0.0876 | 0.7057 | 0.2673 | 0.0792 | 0.0510 | 0.1611 | 0.0010 | -0.6547 |
| place_plant @ ep3 | 3 | 0.0328 | 0.7081 | 0.1990 | 0.0857 | 0.0413 | 0.1788 | 0.0011 | nan |
| collect_drink @ ep6 | 6 | 0.0100 | 0.8961 | 0.5458 | 0.0830 | 0.0518 | 0.1647 | 0.0012 | 0.6547 |
| collect_wood @ ep9 | 9 | -0.1238 | 0.8690 | 0.3032 | 0.0823 | 0.0393 | 0.2057 | 0.0020 | — |
| place_table @ ep34 | 34 | 0.5596 | 0.9236 | 0.0852 | 0.1583 | 0.0456 | 0.3280 | 0.0015 | -0.6547 |
| make_wood_sword @ ep61 | 61 | 0.7532 | 0.9340 | 0.5751 | 0.2296 | 0.0959 | 0.3186 | 0.0004 | -0.6547 |
| make_wood_pickaxe @ ep99 | 99 | 0.8109 | 0.8818 | 0.4431 | 0.2456 | 0.1509 | 0.5364 | 0.0002 | -0.3928 |
| defeat_zombie @ ep140 | 140 | 0.9342 | 0.9028 | 0.5890 | 0.3094 | 0.2172 | 0.7556 | 0.0005 | -0.3489 |
| eat_cow @ ep140 | 140 | 0.8722 | 0.7148 | 0.5257 | 0.2950 | 0.2303 | 0.6983 | 0.0005 | -0.3838 |
| defeat_skeleton @ ep527 | 527 | -0.8135 | 0.0585 | 0.0825 | 0.1280 | 0.1955 | 1.7782 | 0.0004 | 0.0772 |
| collect_stone @ ep580 | 580 | 0.1051 | 0.0980 | 0.2050 | 0.0863 | 0.2474 | 0.7460 | 0.0004 | -0.1745 |
| eat_plant @ ep849 | 849 | 0.2445 | 0.0475 | 0.2297 | 0.1100 | 0.2641 | 0.6171 | 0.0008 | -0.1047 |
| place_stone @ ep912 | 912 | -0.4237 | 0.0839 | 0.3376 | 0.1233 | 0.4050 | 0.6559 | 0.0008 | -0.1047 |
| collect_coal @ ep1685 | 1,685 | 0.0340 | 0.0670 | 0.1726 | 0.1157 | 0.1730 | 0.9561 | 0.0013 | -0.4187 |
| place_furnace @ ep2087 | 2,087 | 0.5509 | 0.1071 | 0.0318 | 0.1204 | 0.1778 | 1.5622 | 0.0019 | -0.0185 |
| make_stone_pickaxe @ ep2199 | 2,199 | 0.6467 | 0.2286 | 0.4858 | 0.1201 | 0.4316 | 1.0246 | 0.0020 | -0.1662 |
| make_stone_sword @ ep6592 | 6,592 | 0.7677 | 0.3099 | 0.2039 | 0.2433 | 0.3148 | 2.6974 | 0.0129 | -0.0923 |
| collect_iron @ ep9655 | 9,655 | 0.6446 | 0.4267 | 0.1424 | 0.2992 | 0.2840 | 2.7193 | 0.0122 | 0.1477 |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 285 | 0.5399 | 0.1650 | 0.5385 | 0.1412 | 0.3011 | 0.3985 | 0.0001 | -0.4352 |
| Step 100,000 | 576 | -0.5252 | 0.0496 | 0.1974 | 0.1055 | 0.3035 | 0.3939 | 0.0004 | 0.0386 |
| Step 150,000 | 873 | -0.8501 | -0.0137 | 0.4088 | 0.0983 | 0.3200 | 0.5276 | 0.0006 | 0.0739 |
| Step 200,000 | 1,150 | 0.3802 | 0.3424 | 0.4657 | 0.1884 | 0.4322 | 0.6240 | 0.0009 | -0.2791 |
| Step 250,000 | 1,438 | 0.4604 | 0.0411 | 0.4206 | 0.1051 | 0.2935 | 0.8493 | 0.0017 | -0.3140 |
| Step 300,000 | 1,722 | 0.7545 | 0.0853 | 0.4126 | 0.0927 | 0.3353 | 0.9074 | 0.0020 | -0.0554 |
| Step 350,000 | 1,995 | 0.1197 | 0.0843 | 0.0736 | 0.1364 | 0.1732 | 1.0230 | 0.0014 | -0.3489 |
| Step 400,000 | 2,265 | 0.5460 | 0.4530 | 0.1083 | 0.2145 | 0.2164 | 1.1392 | 0.0021 | -0.0554 |
| Step 450,000 | 2,532 | 0.9196 | 0.3241 | 0.3852 | 0.2317 | 0.3747 | 1.1926 | 0.0033 | -0.4536 |
| Step 500,000 | 2,809 | 0.5991 | 0.4000 | 0.0541 | 0.2241 | 0.1399 | 1.5579 | 0.0061 | -0.2791 |
| Step 550,000 | 3,084 | 0.8883 | 0.5931 | 0.1036 | 0.3717 | 0.2442 | 1.7694 | 0.0080 | -0.2216 |
| Step 600,000 | 3,360 | 0.4325 | 0.3241 | 0.0731 | 0.1994 | 0.1437 | 1.3827 | 0.0035 | 0.0000 |
| Step 650,000 | 3,624 | 0.8324 | 0.4905 | 0.2688 | 0.2732 | 0.2663 | 1.2068 | 0.0023 | -0.3489 |
| Step 700,000 | 3,903 | 0.8075 | 0.2883 | 0.3083 | 0.1966 | 0.3041 | 1.5467 | 0.0042 | -0.2094 |
| Step 750,000 | 4,174 | 0.3454 | 0.0995 | -0.0072 | 0.1259 | 0.1501 | 1.8967 | 0.0057 | -0.1745 |
| Step 800,000 | 4,448 | 0.5776 | 0.0698 | 0.0955 | 0.1246 | 0.1949 | 2.0321 | 0.0062 | -0.3140 |
| Step 850,000 | 4,720 | 0.5475 | 0.2748 | 0.0904 | 0.1965 | 0.1951 | 2.7125 | 0.0090 | -0.0698 |
| Step 900,000 | 4,989 | 0.8316 | 0.2848 | 0.1905 | 0.2571 | 0.2835 | 2.4345 | 0.0069 | -0.1047 |
| Step 950,000 | 5,251 | 0.6536 | 0.2928 | 0.3225 | 0.2399 | 0.3969 | 2.9606 | 0.0118 | -0.0349 |
| Step 1,000,000 | 5,510 | 0.4950 | 0.1104 | 0.0809 | 0.1727 | 0.2364 | 3.3511 | 0.0104 | 0.0000 |
| Step 1,050,000 | 5,771 | 0.7803 | 0.1887 | 0.1036 | 0.1754 | 0.2291 | 2.9614 | 0.0138 | 0.0000 |
| Step 1,100,000 | 6,020 | 0.0680 | 0.2532 | 0.0957 | 0.1994 | 0.2002 | 2.9296 | 0.0101 | 0.0000 |
| Step 1,150,000 | 6,260 | 0.6722 | 0.3575 | 0.1351 | 0.2613 | 0.2510 | 2.8672 | 0.0105 | 0.0000 |
| Step 1,200,000 | 6,502 | 0.8197 | 0.5006 | 0.1590 | 0.2955 | 0.2828 | 2.7290 | 0.0132 | -0.0185 |
| Step 1,250,000 | 6,748 | 0.5984 | 0.4395 | 0.3558 | 0.3106 | 0.3980 | 2.3697 | 0.0090 | -0.0369 |
| Step 1,300,000 | 6,991 | 0.4200 | 0.2451 | 0.0627 | 0.2042 | 0.2042 | 2.6489 | 0.0109 | -0.0554 |
| Step 1,350,000 | 7,233 | 0.0838 | 0.0335 | 0.1108 | 0.1112 | 0.1922 | 2.7621 | 0.0139 | 0.0554 |
| Step 1,400,000 | 7,473 | 0.3410 | 0.0850 | -0.0099 | 0.1424 | 0.1397 | 3.0155 | 0.0163 | -0.0185 |
| Step 1,450,000 | 7,720 | -0.3552 | 0.0850 | 0.0583 | 0.1528 | 0.1860 | 3.3938 | 0.0147 | 0.0587 |
| Step 1,500,000 | 7,964 | 0.1916 | 0.0934 | 0.0555 | 0.1509 | 0.1890 | 3.4761 | 0.0180 | -0.0369 |
| Step 1,550,000 | 8,209 | 0.7490 | 0.3656 | 0.1509 | 0.3067 | 0.3286 | 3.6146 | 0.0175 | 0.0369 |
| Step 1,600,000 | 8,437 | 0.4789 | 0.1213 | 0.0301 | 0.1709 | 0.1791 | 3.5569 | 0.0213 | 0.1477 |
| Step 1,650,000 | 8,667 | 0.4429 | 0.2433 | 0.1021 | 0.2108 | 0.2103 | 3.8852 | 0.0217 | 0.0923 |
| Step 1,700,000 | 8,911 | 0.5640 | 0.1629 | 0.0851 | 0.1807 | 0.2421 | 3.4308 | 0.0194 | 0.1292 |
| Step 1,750,000 | 9,141 | -0.0432 | 0.1278 | 0.0033 | 0.1737 | 0.2083 | 3.4526 | 0.0147 | 0.1477 |
| Step 1,800,000 | 9,371 | 0.3324 | 0.1216 | 0.0525 | 0.1855 | 0.1717 | 3.9135 | 0.0229 | 0.0369 |
| Step 1,850,000 | 9,597 | 0.6674 | 0.2132 | 0.1390 | 0.2217 | 0.2421 | 2.9509 | 0.0146 | 0.2251 |
| Step 1,900,000 | 9,838 | 0.3088 | 0.1807 | 0.0326 | 0.1732 | 0.1685 | 3.7401 | 0.0249 | 0.0739 |
| Step 1,950,000 | 10,076 | 0.2744 | 0.1306 | 0.0723 | 0.1578 | 0.2081 | 4.0910 | 0.0226 | 0.1477 |
| Step 2,000,000 | 10,295 | 0.6471 | 0.3941 | 0.0313 | 0.3303 | 0.2261 | 3.1265 | 0.0109 | 0.0587 |
| Step 2,050,000 | 10,534 | 0.7824 | 0.4953 | 0.2425 | 0.4235 | 0.4155 | 3.8679 | 0.0148 | -0.0185 |
| Step 2,100,000 | 10,773 | 0.5048 | 0.2114 | 0.0267 | 0.2063 | 0.2053 | 3.7310 | 0.0182 | 0.0554 |
| Step 2,150,000 | 10,996 | 0.5607 | 0.3098 | 0.1626 | 0.2753 | 0.2893 | 4.2128 | 0.0199 | 0.0369 |
| Step 2,200,000 | 11,220 | 0.1893 | -0.0083 | -0.0202 | 0.1048 | 0.1386 | 4.1590 | 0.0225 | 0.0369 |
| Step 2,250,000 | 11,446 | 0.6513 | 0.3698 | 0.1530 | 0.2674 | 0.3327 | 3.5736 | 0.0169 | 0.2400 |
| Step 2,300,000 | 11,676 | 0.5254 | 0.0545 | 0.0160 | 0.1351 | 0.1563 | 3.4652 | 0.0203 | 0.1477 |
| Step 2,350,000 | 11,895 | 0.6569 | 0.1100 | 0.0926 | 0.1740 | 0.2234 | 3.4414 | 0.0177 | -0.0098 |
| Step 2,400,000 | 12,105 | 0.2389 | 0.1143 | 0.0045 | 0.1657 | 0.1745 | 3.1483 | 0.0129 | 0.1174 |
| Step 2,450,000 | 12,323 | 0.0770 | 0.0484 | 0.0773 | 0.1395 | 0.2228 | 4.4282 | 0.0298 | 0.1468 |
| Step 2,500,000 | 12,540 | 0.4976 | 0.1190 | 0.0810 | 0.1742 | 0.2274 | 3.7018 | 0.0207 | 0.1077 |
| Step 2,550,000 | 12,770 | 0.5057 | 0.0957 | 0.0732 | 0.1616 | 0.2798 | 4.0294 | 0.0248 | 0.1477 |
| Step 2,600,000 | 12,970 | 0.1646 | 0.0611 | 0.0199 | 0.1386 | 0.1694 | 4.3310 | 0.0258 | 0.0923 |
| Step 2,650,000 | 13,191 | 0.5557 | 0.2299 | 0.0353 | 0.2268 | 0.2417 | 3.4298 | 0.0143 | 0.2031 |
| Step 2,700,000 | 13,411 | 0.3681 | 0.0800 | 0.0973 | 0.1689 | 0.2242 | 2.8976 | 0.0139 | 0.1566 |
| Step 2,750,000 | 13,638 | 0.6234 | 0.2633 | 0.0888 | 0.2416 | 0.2427 | 3.4064 | 0.0199 | 0.1292 |
| Step 2,800,000 | 13,867 | 0.4309 | 0.0826 | 0.1406 | 0.1720 | 0.2407 | 3.8957 | 0.0342 | 0.1846 |
| Step 2,850,000 | 14,093 | 0.4649 | 0.1452 | 0.0934 | 0.1860 | 0.2125 | 2.9916 | 0.0191 | 0.0979 |
| Step 2,900,000 | 14,322 | 0.4617 | 0.0371 | 0.0168 | 0.1271 | 0.2080 | 3.1111 | 0.0176 | 0.1846 |
| Step 2,950,000 | 14,541 | 0.4367 | 0.1395 | 0.1076 | 0.1730 | 0.2332 | 3.1068 | 0.0129 | 0.1108 |
| Step 3,000,000 | 14,762 | 0.7168 | 0.1946 | 0.0779 | 0.2268 | 0.2619 | 3.3222 | 0.0136 | 0.1477 |
| Final | 14,763 | 0.2379 | 0.1151 | 0.0018 | 0.1633 | 0.1673 | 3.4067 | 0.0153 | 0.1477 |
