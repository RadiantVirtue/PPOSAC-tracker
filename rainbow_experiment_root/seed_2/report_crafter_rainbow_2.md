# Training & Analysis Report

**Environment:** `Crafter`  
**Seed:** 2  
**Total episodes:** 13,532  
**Experiment root:** `rainbow_experiment_root\seed_2`  
**Generated:** 2026-04-09 11:16

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep183 | 183 | 0.9976 | 0.9217 | 0.9184 | 0.1038 | 0.0973 | 0.2546 | 0.0009 | — |
| collect_sapling @ ep282 | 282 | 0.9955 | 0.9004 | 0.9234 | 0.1621 | 0.1554 | 0.0427 | 0.0000 | — |
| collect_wood @ ep283 | 283 | 0.9978 | 0.8603 | 0.9313 | 0.0935 | 0.0890 | 0.1634 | 0.0002 | — |
| place_plant @ ep283 | 283 | 0.9961 | 0.8724 | 0.9069 | 0.0884 | 0.0875 | 0.1517 | 0.0002 | — |
| Step 50,000 | 294 | 0.9789 | 0.9228 | 0.9034 | 0.1167 | 0.1013 | 0.4580 | 0.0008 | — |
| collect_drink @ ep313 | 313 | 0.8487 | 0.9517 | 0.7747 | 0.2659 | 0.0933 | 0.3345 | 0.0007 | — |
| eat_cow @ ep315 | 315 | 0.9453 | 0.9656 | 0.9638 | 0.4641 | 0.2120 | 0.2816 | 0.0006 | — |
| defeat_zombie @ ep450 | 450 | 0.9604 | 0.9186 | 0.9651 | 0.2168 | 0.1510 | 0.2506 | 0.0006 | — |
| Step 100,000 | 572 | 0.7975 | 0.9103 | 0.9739 | 0.2912 | 0.2268 | 0.7545 | 0.0049 | — |
| defeat_skeleton @ ep803 | 803 | 0.7485 | 0.8904 | 0.9369 | 0.1028 | 0.1469 | 0.8166 | 0.0075 | — |
| Step 150,000 | 836 | 0.9322 | 0.9763 | 0.9223 | 0.5124 | 0.3195 | 0.6753 | 0.0052 | — |
| Step 200,000 | 1,110 | 0.5738 | 0.9384 | 0.8257 | 0.1637 | 0.1506 | 0.9311 | 0.0110 | — |
| Step 250,000 | 1,389 | 0.8974 | 0.9843 | 0.9546 | 0.3346 | 0.2884 | 0.6422 | 0.0059 | — |
| Step 300,000 | 1,656 | 0.8086 | 0.9460 | 0.9157 | 0.2525 | 0.2260 | 0.5994 | 0.0057 | — |
| Step 350,000 | 1,923 | 0.7787 | 0.9701 | 0.9366 | 0.3319 | 0.2983 | 0.6977 | 0.0077 | — |
| place_table @ ep2115 | 2,115 | 0.8100 | 0.9497 | 0.8877 | 0.2543 | 0.2467 | 0.8974 | 0.0137 | — |
| Step 400,000 | 2,184 | 0.7864 | 0.9401 | 0.8864 | 0.2225 | 0.1883 | 0.9123 | 0.0152 | — |
| Step 450,000 | 2,439 | 0.9111 | 0.9574 | 0.9080 | 0.4155 | 0.3844 | 0.8557 | 0.0155 | — |
| Step 500,000 | 2,697 | 0.9301 | 0.9581 | 0.8967 | 0.5736 | 0.4478 | 1.0688 | 0.0218 | — |
| Step 550,000 | 2,957 | 0.8959 | 0.9799 | 0.9074 | 0.5368 | 0.4379 | 0.7151 | 0.0122 | — |
| Step 600,000 | 3,201 | 0.7583 | 0.9584 | 0.8305 | 0.3938 | 0.2539 | 1.0752 | 0.0321 | — |
| Step 650,000 | 3,449 | 0.8003 | 0.9627 | 0.8281 | 0.4625 | 0.4439 | 0.7500 | 0.0164 | — |
| make_wood_pickaxe @ ep3557 | 3,557 | 0.8622 | 0.9716 | 0.8191 | 0.4862 | 0.4020 | 0.8029 | 0.0184 | — |
| Step 700,000 | 3,676 | 0.6782 | 0.9417 | 0.8147 | 0.3433 | 0.3609 | 1.1596 | 0.0381 | — |
| Step 750,000 | 3,908 | 0.6709 | 0.8810 | 0.7762 | 0.2683 | 0.3147 | 0.5178 | 0.0094 | — |
| Step 800,000 | 4,125 | 0.8522 | 0.9172 | 0.7738 | 0.2578 | 0.2646 | 0.5122 | 0.0104 | — |
| Step 850,000 | 4,341 | 0.9375 | 0.9272 | 0.8466 | 0.3090 | 0.4175 | 0.4745 | 0.0095 | — |
| make_wood_sword @ ep4521 | 4,521 | 0.9023 | 0.9009 | 0.8773 | 0.2595 | 0.3794 | 0.4389 | 0.0083 | — |
| Step 900,000 | 4,564 | 0.5656 | 0.8845 | 0.7378 | 0.1882 | 0.2722 | 0.4220 | 0.0078 | — |
| Step 950,000 | 4,785 | 0.6837 | 0.8811 | 0.7146 | 0.1748 | 0.2346 | 0.4022 | 0.0064 | — |
| Step 1,000,000 | 5,002 | 0.7919 | 0.9266 | 0.7364 | 0.2570 | 0.3089 | 0.3100 | 0.0041 | — |
| Step 1,050,000 | 5,222 | 0.9025 | 0.9352 | 0.8096 | 0.3230 | 0.3095 | 0.3024 | 0.0035 | — |
| Step 1,100,000 | 5,445 | 0.8949 | 0.9338 | 0.7624 | 0.3160 | 0.3385 | 0.3728 | 0.0055 | — |
| Step 1,150,000 | 5,652 | 0.8786 | 0.9268 | 0.8145 | 0.3104 | 0.2923 | 0.3529 | 0.0051 | — |
| Step 1,200,000 | 5,858 | 0.7548 | 0.9012 | 0.7886 | 0.2489 | 0.3668 | 0.3867 | 0.0063 | — |
| Step 1,250,000 | 6,066 | 0.0114 | 0.8643 | 0.6556 | 0.2228 | 0.2723 | 0.3856 | 0.0065 | — |
| Step 1,300,000 | 6,275 | 0.6525 | 0.9251 | 0.7370 | 0.3034 | 0.2775 | 0.4090 | 0.0073 | — |
| collect_stone @ ep6399 | 6,399 | 0.9927 | 0.9470 | 0.8309 | 0.7798 | 0.6678 | 0.4253 | 0.0072 | — |
| Step 1,350,000 | 6,492 | 0.8552 | 0.8975 | 0.6578 | 0.2340 | 0.2308 | 0.4161 | 0.0075 | — |
| Step 1,400,000 | 6,704 | 0.8301 | 0.8913 | 0.7632 | 0.2117 | 0.2783 | 0.4661 | 0.0094 | — |
| Step 1,450,000 | 6,913 | 0.7543 | 0.8507 | 0.7261 | 0.2117 | 0.3143 | 0.4348 | 0.0086 | — |
| Step 1,500,000 | 7,121 | 0.9802 | 0.9211 | 0.7341 | 0.6192 | 0.5292 | 0.4052 | 0.0076 | — |
| Step 1,550,000 | 7,331 | 0.8576 | 0.8612 | 0.6775 | 0.2365 | 0.2253 | 0.3276 | 0.0050 | — |
| Step 1,600,000 | 7,545 | 0.8013 | 0.8559 | 0.7094 | 0.2073 | 0.2331 | 0.3979 | 0.0072 | — |
| collect_coal @ ep7687 | 7,687 | 0.9432 | 0.8945 | 0.7324 | 0.3416 | 0.2527 | 0.3004 | 0.0042 | — |
| Step 1,650,000 | 7,750 | 0.8114 | 0.7544 | 0.6959 | 0.1637 | 0.2273 | 0.3145 | 0.0049 | — |
| Step 1,700,000 | 7,956 | 0.6475 | 0.8508 | 0.6593 | 0.1800 | 0.1891 | 0.3226 | 0.0044 | — |
| Step 1,750,000 | 8,172 | 0.9678 | 0.8805 | 0.7805 | 0.2644 | 0.2658 | 0.3795 | 0.0067 | — |
| Step 1,800,000 | 8,374 | 0.8515 | 0.9027 | 0.7440 | 0.2547 | 0.3111 | 0.2717 | 0.0033 | — |
| Step 1,850,000 | 8,578 | 0.8901 | 0.8347 | 0.6537 | 0.3133 | 0.2426 | 0.3866 | 0.0060 | — |
| Step 1,900,000 | 8,792 | 0.9529 | 0.9209 | 0.7688 | 0.5428 | 0.6947 | 0.3565 | 0.0056 | — |
| Step 1,950,000 | 9,001 | 0.4775 | 0.7962 | 0.5503 | 0.2692 | 0.1737 | 0.3745 | 0.0061 | — |
| collect_drink @ ep9002 | 9,002 | 0.5616 | 0.8031 | 0.5811 | 0.2972 | 0.1746 | 0.3520 | 0.0054 | — |
| collect_sapling @ ep9002 | 9,002 | 0.6134 | 0.7666 | 0.5847 | 0.2757 | 0.1684 | 0.3793 | 0.0064 | — |
| collect_wood @ ep9002 | 9,002 | 0.4604 | 0.8041 | 0.5961 | 0.2909 | 0.1667 | 0.3639 | 0.0058 | — |
| defeat_zombie @ ep9002 | 9,002 | 0.7927 | 0.7808 | 0.6019 | 0.2782 | 0.1803 | 0.3587 | 0.0056 | — |
| eat_cow @ ep9002 | 9,002 | 0.6677 | 0.7975 | 0.5718 | 0.2868 | 0.1744 | 0.3609 | 0.0057 | — |
| place_plant @ ep9002 | 9,002 | 0.5093 | 0.7924 | 0.5217 | 0.2818 | 0.1671 | 0.3792 | 0.0063 | — |
| place_table @ ep9002 | 9,002 | 0.6077 | 0.7588 | 0.5673 | 0.2826 | 0.1668 | 0.3927 | 0.0068 | — |
| wake_up @ ep9002 | 9,002 | 0.5170 | 0.7783 | 0.5668 | 0.3249 | 0.1661 | 0.3561 | 0.0056 | — |
| make_wood_pickaxe @ ep9004 | 9,004 | 0.1929 | 0.7603 | 0.5595 | 0.2754 | 0.1733 | 0.3824 | 0.0065 | — |
| defeat_skeleton @ ep9017 | 9,017 | 0.5164 | 0.7789 | 0.5595 | 0.2834 | 0.1715 | 0.3538 | 0.0055 | — |
| collect_stone @ ep9058 | 9,058 | 0.8392 | 0.7781 | 0.6002 | 0.2690 | 0.1942 | 0.3391 | 0.0051 | — |
| make_wood_sword @ ep9214 | 9,214 | 0.8084 | 0.8691 | 0.7765 | 0.2805 | 0.3849 | 0.4303 | 0.0075 | — |
| Step 2,000,000 | 9,215 | 0.8953 | 0.9095 | 0.6900 | 0.3674 | 0.2865 | 0.4386 | 0.0082 | — |
| collect_coal @ ep9224 | 9,224 | 0.9600 | 0.9116 | 0.8571 | 0.4597 | 0.6248 | 0.4472 | 0.0080 | — |
| Step 2,050,000 | 9,420 | 0.9189 | 0.8830 | 0.7477 | 0.4927 | 0.4338 | 0.3601 | 0.0057 | — |
| Step 2,100,000 | 9,642 | 0.9319 | 0.9143 | 0.7303 | 0.6065 | 0.4632 | 0.3498 | 0.0050 | — |
| Step 2,150,000 | 9,870 | 0.6685 | 0.8860 | 0.7045 | 0.2713 | 0.2765 | 0.3652 | 0.0054 | — |
| make_stone_sword @ ep9875 | 9,875 | 0.9227 | 0.9251 | 0.7778 | 0.4301 | 0.3005 | 0.3979 | 0.0064 | — |
| Step 2,200,000 | 10,087 | 0.7583 | 0.8755 | 0.7589 | 0.2309 | 0.2270 | 0.3957 | 0.0068 | — |
| Step 2,250,000 | 10,316 | 0.8094 | 0.8800 | 0.6944 | 0.3006 | 0.3151 | 0.4223 | 0.0075 | — |
| Step 2,300,000 | 10,541 | 0.9590 | 0.8752 | 0.7783 | 0.4732 | 0.4509 | 0.3864 | 0.0065 | — |
| Step 2,350,000 | 10,763 | 0.9681 | 0.9256 | 0.8244 | 0.4988 | 0.4017 | 0.4754 | 0.0095 | — |
| Step 2,400,000 | 10,998 | 0.7569 | 0.8135 | 0.6993 | 0.1974 | 0.2109 | 0.4698 | 0.0103 | — |
| place_furnace @ ep11209 | 11,209 | 0.9264 | 0.9143 | 0.6859 | 0.4015 | 0.3027 | 0.4332 | 0.0084 | — |
| Step 2,450,000 | 11,220 | 0.7687 | 0.8755 | 0.7130 | 0.2275 | 0.3138 | 0.4751 | 0.0104 | — |
| Step 2,500,000 | 11,443 | 0.9166 | 0.8953 | 0.6908 | 0.4298 | 0.3637 | 0.5487 | 0.0144 | — |
| Step 2,550,000 | 11,669 | 0.4175 | 0.8613 | 0.5963 | 0.2464 | 0.1870 | 0.6915 | 0.0205 | — |
| Step 2,600,000 | 11,881 | 0.8471 | 0.8891 | 0.6405 | 0.3049 | 0.2171 | 0.6426 | 0.0192 | — |
| Step 2,650,000 | 12,087 | 0.7548 | 0.8593 | 0.7476 | 0.2588 | 0.3188 | 0.8807 | 0.0345 | — |
| place_stone @ ep12088 | 12,088 | 0.6899 | 0.9047 | 0.7211 | 0.2819 | 0.2557 | 0.7875 | 0.0286 | — |
| Step 2,700,000 | 12,284 | 0.9255 | 0.8995 | 0.8402 | 0.5379 | 0.6274 | 0.8897 | 0.0362 | — |
| make_stone_pickaxe @ ep12409 | 12,409 | 0.9846 | 0.9351 | 0.8274 | 0.9435 | 0.8200 | 0.7891 | 0.0272 | — |
| Step 2,750,000 | 12,503 | 0.5770 | 0.8602 | 0.6185 | 0.2183 | 0.1860 | 0.7976 | 0.0273 | — |
| Step 2,800,000 | 12,701 | 0.6319 | 0.8789 | 0.6332 | 0.2282 | 0.2056 | 0.9543 | 0.0372 | — |
| Step 2,850,000 | 12,908 | 0.9332 | 0.8967 | 0.7187 | 0.4312 | 0.4509 | 1.1081 | 0.0547 | — |
| Step 2,900,000 | 13,118 | 0.9520 | 0.8754 | 0.7618 | 0.4309 | 0.4474 | 1.1821 | 0.0556 | — |
| Step 2,950,000 | 13,326 | 0.8130 | 0.8810 | 0.7464 | 0.3115 | 0.4263 | 1.3181 | 0.0682 | — |
| Step 3,000,000 | 13,532 | 0.7021 | 0.9048 | 0.6644 | 0.2734 | 0.2073 | 1.3695 | 0.0729 | — |

---

## wake_up_ep183_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9976 |
| Coherence (Success) | 0.9217 |
| Coherence (Failure) | 0.9184 |
| Gradient Magnitude (Success) | 0.1038 |
| Gradient Magnitude (Failure) | 0.0973 |
| Activation Separation | 0.2546 |
| Cosine Distance | 0.0009 |
| Clusters | 1,700 |
| Noise Fraction | 0.2212 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (4) | Place Plant, Sapling, Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9976 | 0.9217 | 0.9184 | 0.1038 | 0.0973 |
| G_IS (β=0.402) | 0.9978 | 0.9280 | 0.9234 | 0.1008 | 0.0946 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9998 | 0.9998 |
| cos(G_IS, G_reward)  | 0.9880 | 0.9837 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 245 | 32.9680 | 0.8052 |
| Neutral  (r = 0) | 39,903 | 0.0972 | 0.9636 |
| Negative (r < 0) | 1,307 | 3.1897 | 0.9950 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.8212 |
| Pos vs Negative  | -0.7752 |
| Neutral vs Neg.  | 0.5393 |
| Pos vs Failure   | -0.8921 |
| Neutral vs Fail. | 0.7786 |
| Neg. vs Failure  | 0.9130 |

---

## collect_sapling_ep282_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9955 |
| Coherence (Success) | 0.9004 |
| Coherence (Failure) | 0.9234 |
| Gradient Magnitude (Success) | 0.1621 |
| Gradient Magnitude (Failure) | 0.1554 |
| Activation Separation | 0.0427 |
| Cosine Distance | 0.0000 |
| Clusters | 1,886 |
| Noise Fraction | 0.1253 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9955 | 0.9004 | 0.9234 | 0.1621 | 0.1554 |
| G_IS (β=0.406) | 0.9939 | 0.8616 | 0.9021 | 0.0942 | 0.0914 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9966 | 0.9953 |
| cos(G_IS, G_reward)  | -0.7446 | -0.6959 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 194 | 252.4019 | 0.8882 |
| Neutral  (r = 0) | 41,260 | 0.1734 | 0.9643 |
| Negative (r < 0) | 1,291 | 3.2924 | 0.5895 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.7338 |
| Pos vs Negative  | -0.4944 |
| Neutral vs Neg.  | -0.8627 |
| Pos vs Failure   | 0.9293 |
| Neutral vs Fail. | 0.7369 |
| Neg. vs Failure  | -0.5557 |

---

## collect_wood_ep283_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9978 |
| Coherence (Success) | 0.8603 |
| Coherence (Failure) | 0.9313 |
| Gradient Magnitude (Success) | 0.0935 |
| Gradient Magnitude (Failure) | 0.0890 |
| Activation Separation | 0.1634 |
| Cosine Distance | 0.0002 |
| Clusters | 1,944 |
| Noise Fraction | 0.1294 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (2) | Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9978 | 0.8603 | 0.9313 | 0.0935 | 0.0890 |
| G_IS (β=0.406) | 0.9966 | 0.7309 | 0.8452 | 0.0538 | 0.0506 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9658 | 0.9626 |
| cos(G_IS, G_reward)  | -0.2516 | -0.2636 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 196 | 222.6345 | 0.8928 |
| Neutral  (r = 0) | 42,680 | 0.0955 | 0.8946 |
| Negative (r < 0) | 1,298 | 3.4755 | 0.7890 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.4970 |
| Pos vs Negative  | -0.4907 |
| Neutral vs Neg.  | -0.8631 |
| Pos vs Failure   | 0.7986 |
| Neutral vs Fail. | 0.1317 |
| Neg. vs Failure  | -0.2206 |

---

## place_plant_ep283_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9961 |
| Coherence (Success) | 0.8724 |
| Coherence (Failure) | 0.9069 |
| Gradient Magnitude (Success) | 0.0884 |
| Gradient Magnitude (Failure) | 0.0875 |
| Activation Separation | 0.1517 |
| Cosine Distance | 0.0002 |
| Clusters | 1,914 |
| Noise Fraction | 0.1272 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (4) | Place Plant, Sapling, Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9961 | 0.8724 | 0.9069 | 0.0884 | 0.0875 |
| G_IS (β=0.406) | 0.9942 | 0.7199 | 0.7649 | 0.0498 | 0.0507 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9600 | 0.9565 |
| cos(G_IS, G_reward)  | -0.2512 | -0.1902 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 194 | 219.8587 | 0.8648 |
| Neutral  (r = 0) | 42,432 | 0.0954 | 0.9585 |
| Negative (r < 0) | 1,305 | 3.5123 | 0.8079 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.4845 |
| Pos vs Negative  | -0.5020 |
| Neutral vs Neg.  | -0.8529 |
| Pos vs Failure   | 0.7712 |
| Neutral vs Fail. | 0.0761 |
| Neg. vs Failure  | -0.2018 |

---

## ep294_lower0.000_upper0.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9789 |
| Coherence (Success) | 0.9228 |
| Coherence (Failure) | 0.9034 |
| Gradient Magnitude (Success) | 0.1167 |
| Gradient Magnitude (Failure) | 0.1013 |
| Activation Separation | 0.4580 |
| Cosine Distance | 0.0008 |
| Clusters | 2,208 |
| Noise Fraction | 0.1343 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (6) | Drink, Eat Cow, Place Plant, Sapling, Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9789 | 0.9228 | 0.9034 | 0.1167 | 0.1013 |
| G_IS (β=0.406) | 0.9606 | 0.8948 | 0.8472 | 0.0685 | 0.0569 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9955 | 0.9851 |
| cos(G_IS, G_reward)  | -0.3918 | -0.5768 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 417 | 92.8296 | 0.8153 |
| Neutral  (r = 0) | 44,914 | 0.1376 | 0.9499 |
| Negative (r < 0) | 1,302 | 3.6561 | 0.8068 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.2924 |
| Pos vs Negative  | -0.3760 |
| Neutral vs Neg.  | -0.8323 |
| Pos vs Failure   | 0.8044 |
| Neutral vs Fail. | 0.1774 |
| Neg. vs Failure  | -0.4393 |

---

## collect_drink_ep313_lower0.000_upper1.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8487 |
| Coherence (Success) | 0.9517 |
| Coherence (Failure) | 0.7747 |
| Gradient Magnitude (Success) | 0.2659 |
| Gradient Magnitude (Failure) | 0.0933 |
| Activation Separation | 0.3345 |
| Cosine Distance | 0.0007 |
| Clusters | 1,745 |
| Noise Fraction | 0.1342 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (5) | Eat Cow, Place Plant, Sapling, Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8487 | 0.9517 | 0.7747 | 0.2659 | 0.0933 |
| G_IS (β=0.407) | 0.6248 | 0.9397 | 0.6571 | 0.1686 | 0.0552 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9992 | 0.9186 |
| cos(G_IS, G_reward)  | 0.9597 | -0.1420 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 664 | 10.8222 | 0.8041 |
| Neutral  (r = 0) | 42,613 | 0.1409 | 0.9489 |
| Negative (r < 0) | 1,179 | 4.8294 | 0.9647 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.2341 |
| Pos vs Negative  | -0.6143 |
| Neutral vs Neg.  | -0.7907 |
| Pos vs Failure   | 0.5211 |
| Neutral vs Fail. | 0.1195 |
| Neg. vs Failure  | -0.5578 |

---

## eat_cow_ep315_lower1.900_upper2.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9453 |
| Coherence (Success) | 0.9656 |
| Coherence (Failure) | 0.9638 |
| Gradient Magnitude (Success) | 0.4641 |
| Gradient Magnitude (Failure) | 0.2120 |
| Activation Separation | 0.2816 |
| Cosine Distance | 0.0006 |
| Clusters | 1,764 |
| Noise Fraction | 0.1477 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (5) | Eat Cow, Place Plant, Sapling, Wake Up, Wood |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9453 | 0.9656 | 0.9638 | 0.4641 | 0.2120 |
| G_IS (β=0.407) | 0.9365 | 0.9576 | 0.9611 | 0.3077 | 0.1430 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9994 | 0.9966 |
| cos(G_IS, G_reward)  | 0.9446 | 0.0367 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 813 | 3.9620 | 0.9866 |
| Neutral  (r = 0) | 39,340 | 0.2747 | 0.9488 |
| Negative (r < 0) | 1,262 | 3.8567 | 0.8249 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.6323 |
| Pos vs Negative  | -0.5092 |
| Neutral vs Neg.  | -0.8547 |
| Pos vs Failure   | 0.5097 |
| Neutral vs Fail. | 0.7789 |
| Neg. vs Failure  | -0.4266 |

---

## defeat_zombie_ep450_lower1.900_upper2.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9604 |
| Coherence (Success) | 0.9186 |
| Coherence (Failure) | 0.9651 |
| Gradient Magnitude (Success) | 0.2168 |
| Gradient Magnitude (Failure) | 0.1510 |
| Activation Separation | 0.2506 |
| Cosine Distance | 0.0006 |
| Clusters | 1,822 |
| Noise Fraction | 0.1341 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (7) | Drink, Eat Cow, Place Plant, Sapling, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9604 | 0.9186 | 0.9651 | 0.2168 | 0.1510 |
| G_IS (β=0.412) | 0.9506 | 0.9066 | 0.9576 | 0.1379 | 0.0978 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9979 | 0.9941 |
| cos(G_IS, G_reward)  | 0.6845 | -0.6683 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0345 | 0.0541 |
| cos(G_IS, Δθ)      | 0.0365 | 0.0564 |
| cos(G_reward, Δθ)  | -0.0306 | -0.0293 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 769 | 3.4932 | 0.8494 |
| Neutral  (r = 0) | 44,413 | 0.2105 | 0.9465 |
| Negative (r < 0) | 1,351 | 4.2214 | 0.9544 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.2984 |
| Pos vs Negative  | -0.4045 |
| Neutral vs Neg.  | -0.7765 |
| Pos vs Failure   | 0.1206 |
| Neutral vs Fail. | 0.3521 |
| Neg. vs Failure  | -0.7116 |

---

## ep572_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7975 |
| Coherence (Success) | 0.9103 |
| Coherence (Failure) | 0.9739 |
| Gradient Magnitude (Success) | 0.2912 |
| Gradient Magnitude (Failure) | 0.2268 |
| Activation Separation | 0.7545 |
| Cosine Distance | 0.0049 |
| Clusters | 2,131 |
| Noise Fraction | 0.1343 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (8) | Drink, Eat Cow, Place Plant, Sapling, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7975 | 0.9103 | 0.9739 | 0.2912 | 0.2268 |
| G_IS (β=0.416) | 0.7330 | 0.8888 | 0.9725 | 0.1895 | 0.1539 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9975 | 0.9848 |
| cos(G_IS, G_reward)  | 0.7347 | -0.1664 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0297 | 0.0758 |
| cos(G_IS, Δθ)      | 0.0335 | 0.0812 |
| cos(G_reward, Δθ)  | -0.0382 | 0.0049 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 836 | 5.2878 | 0.9046 |
| Neutral  (r = 0) | 49,296 | 0.2325 | 0.9765 |
| Negative (r < 0) | 1,315 | 3.3184 | 0.6645 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.3978 |
| Pos vs Negative  | -0.3452 |
| Neutral vs Neg.  | -0.6191 |
| Pos vs Failure   | 0.2067 |
| Neutral vs Fail. | 0.3288 |
| Neg. vs Failure  | 0.2943 |

---

## defeat_skeleton_ep803_lower1.900_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7485 |
| Coherence (Success) | 0.8904 |
| Coherence (Failure) | 0.9369 |
| Gradient Magnitude (Success) | 0.1028 |
| Gradient Magnitude (Failure) | 0.1469 |
| Activation Separation | 0.8166 |
| Cosine Distance | 0.0075 |
| Clusters | 1,963 |
| Noise Fraction | 0.1866 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (8) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7485 | 0.8904 | 0.9369 | 0.1028 | 0.1469 |
| G_IS (β=0.425) | 0.7544 | 0.8859 | 0.9347 | 0.0672 | 0.1088 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9686 | 0.9760 |
| cos(G_IS, G_reward)  | 0.1231 | 0.4325 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0123 | -0.0102 |
| cos(G_IS, Δθ)      | -0.0130 | -0.0110 |
| cos(G_reward, Δθ)  | -0.0081 | -0.0183 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 953 | 4.7430 | 0.9746 |
| Neutral  (r = 0) | 49,808 | 0.2041 | 0.9920 |
| Negative (r < 0) | 1,365 | 3.7478 | 0.9917 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0711 |
| Pos vs Negative  | -0.7192 |
| Neutral vs Neg.  | -0.3008 |
| Pos vs Failure   | -0.2985 |
| Neutral vs Fail. | 0.0929 |
| Neg. vs Failure  | 0.2597 |

---

## ep836_lower2.000_upper2.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9322 |
| Coherence (Success) | 0.9763 |
| Coherence (Failure) | 0.9223 |
| Gradient Magnitude (Success) | 0.5124 |
| Gradient Magnitude (Failure) | 0.3195 |
| Activation Separation | 0.6753 |
| Cosine Distance | 0.0052 |
| Clusters | 1,890 |
| Noise Fraction | 0.1816 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (8) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9322 | 0.9763 | 0.9223 | 0.5124 | 0.3195 |
| G_IS (β=0.426) | 0.9074 | 0.9724 | 0.9206 | 0.3476 | 0.2063 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9996 | 0.9960 |
| cos(G_IS, G_reward)  | 0.7960 | -0.4054 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0154 | 0.0187 |
| cos(G_IS, Δθ)      | 0.0157 | 0.0187 |
| cos(G_reward, Δθ)  | -0.0036 | -0.0196 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 995 | 6.3610 | 0.9791 |
| Neutral  (r = 0) | 47,270 | 0.4147 | 0.9963 |
| Negative (r < 0) | 1,336 | 3.3221 | 0.9946 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.6244 |
| Pos vs Negative  | -0.6205 |
| Neutral vs Neg.  | -0.7919 |
| Pos vs Failure   | 0.6348 |
| Neutral vs Fail. | 0.7928 |
| Neg. vs Failure  | -0.4205 |

---

## ep1110_lower2.000_upper3.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5738 |
| Coherence (Success) | 0.9384 |
| Coherence (Failure) | 0.8257 |
| Gradient Magnitude (Success) | 0.1637 |
| Gradient Magnitude (Failure) | 0.1506 |
| Activation Separation | 0.9311 |
| Cosine Distance | 0.0110 |
| Clusters | 2,097 |
| Noise Fraction | 0.1489 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5738 | 0.9384 | 0.8257 | 0.1637 | 0.1506 |
| G_IS (β=0.436) | 0.5216 | 0.9209 | 0.8668 | 0.1054 | 0.1160 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9842 | 0.9708 |
| cos(G_IS, G_reward)  | 0.6138 | 0.0793 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0456 | -0.0374 |
| cos(G_IS, Δθ)      | -0.0475 | -0.0324 |
| cos(G_reward, Δθ)  | -0.0231 | -0.0387 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,110 | 6.4167 | 0.9919 |
| Neutral  (r = 0) | 45,822 | 0.2459 | 0.9892 |
| Negative (r < 0) | 1,315 | 4.2108 | 0.9933 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3515 |
| Pos vs Negative  | -0.5552 |
| Neutral vs Neg.  | -0.3290 |
| Pos vs Failure   | -0.1324 |
| Neutral vs Fail. | 0.0387 |
| Neg. vs Failure  | 0.3776 |

---

## ep1389_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8974 |
| Coherence (Success) | 0.9843 |
| Coherence (Failure) | 0.9546 |
| Gradient Magnitude (Success) | 0.3346 |
| Gradient Magnitude (Failure) | 0.2884 |
| Activation Separation | 0.6422 |
| Cosine Distance | 0.0059 |
| Clusters | 1,918 |
| Noise Fraction | 0.1479 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (8) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8974 | 0.9843 | 0.9546 | 0.3346 | 0.2884 |
| G_IS (β=0.446) | 0.8678 | 0.9817 | 0.9477 | 0.2197 | 0.1930 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9975 | 0.9906 |
| cos(G_IS, G_reward)  | 0.7779 | 0.3067 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0157 | 0.0157 |
| cos(G_IS, Δθ)      | 0.0153 | 0.0145 |
| cos(G_reward, Δθ)  | 0.0094 | 0.0038 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,337 | 5.9167 | 0.9905 |
| Neutral  (r = 0) | 47,649 | 0.2419 | 0.9910 |
| Negative (r < 0) | 1,295 | 3.7881 | 0.9867 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.3850 |
| Pos vs Negative  | -0.4562 |
| Neutral vs Neg.  | -0.7699 |
| Pos vs Failure   | 0.5464 |
| Neutral vs Fail. | 0.2664 |
| Neg. vs Failure  | 0.0480 |

---

## ep1656_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8086 |
| Coherence (Success) | 0.9460 |
| Coherence (Failure) | 0.9157 |
| Gradient Magnitude (Success) | 0.2525 |
| Gradient Magnitude (Failure) | 0.2260 |
| Activation Separation | 0.5994 |
| Cosine Distance | 0.0057 |
| Clusters | 2,080 |
| Noise Fraction | 0.1449 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (8) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8086 | 0.9460 | 0.9157 | 0.2525 | 0.2260 |
| G_IS (β=0.456) | 0.7648 | 0.9380 | 0.9082 | 0.1638 | 0.1528 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9929 | 0.9821 |
| cos(G_IS, G_reward)  | 0.4341 | 0.1596 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0105 | 0.0128 |
| cos(G_IS, Δθ)      | 0.0098 | 0.0117 |
| cos(G_reward, Δθ)  | -0.0048 | -0.0041 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,408 | 6.0488 | 0.9878 |
| Neutral  (r = 0) | 47,423 | 0.2485 | 0.9897 |
| Negative (r < 0) | 1,334 | 3.4888 | 0.9856 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1606 |
| Pos vs Negative  | -0.3621 |
| Neutral vs Neg.  | -0.5000 |
| Pos vs Failure   | 0.1877 |
| Neutral vs Fail. | 0.0266 |
| Neg. vs Failure  | 0.3519 |

---

## ep1923_lower3.000_upper4.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7787 |
| Coherence (Success) | 0.9701 |
| Coherence (Failure) | 0.9366 |
| Gradient Magnitude (Success) | 0.3319 |
| Gradient Magnitude (Failure) | 0.2983 |
| Activation Separation | 0.6977 |
| Cosine Distance | 0.0077 |
| Clusters | 2,181 |
| Noise Fraction | 0.1567 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7787 | 0.9701 | 0.9366 | 0.3319 | 0.2983 |
| G_IS (β=0.466) | 0.7234 | 0.9675 | 0.9293 | 0.2110 | 0.2011 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9943 | 0.9840 |
| cos(G_IS, G_reward)  | 0.7718 | 0.6290 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0141 | 0.0134 |
| cos(G_IS, Δθ)      | 0.0142 | 0.0124 |
| cos(G_reward, Δθ)  | 0.0009 | -0.0011 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,535 | 8.7002 | 0.9787 |
| Neutral  (r = 0) | 48,288 | 0.2466 | 0.9835 |
| Negative (r < 0) | 1,356 | 9.9466 | 0.5990 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0438 |
| Pos vs Negative  | 0.2007 |
| Neutral vs Neg.  | 0.0170 |
| Pos vs Failure   | 0.6063 |
| Neutral vs Fail. | -0.0602 |
| Neg. vs Failure  | 0.5391 |

---

## place_table_ep2115_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8100 |
| Coherence (Success) | 0.9497 |
| Coherence (Failure) | 0.8877 |
| Gradient Magnitude (Success) | 0.2543 |
| Gradient Magnitude (Failure) | 0.2467 |
| Activation Separation | 0.8974 |
| Cosine Distance | 0.0137 |
| Clusters | 2,394 |
| Noise Fraction | 0.1534 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8100 | 0.9497 | 0.8877 | 0.2543 | 0.2467 |
| G_IS (β=0.474) | 0.7486 | 0.9375 | 0.8747 | 0.1517 | 0.1585 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9894 | 0.9757 |
| cos(G_IS, G_reward)  | 0.7511 | 0.6485 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0007 | 0.0048 |
| cos(G_IS, Δθ)      | 0.0023 | 0.0061 |
| cos(G_reward, Δθ)  | -0.0107 | -0.0066 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,617 | 8.8094 | 0.9904 |
| Neutral  (r = 0) | 50,712 | 0.2976 | 0.9831 |
| Negative (r < 0) | 1,327 | 38.9752 | 0.3086 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3089 |
| Pos vs Negative  | 0.3235 |
| Neutral vs Neg.  | -0.0664 |
| Pos vs Failure   | 0.6625 |
| Neutral vs Fail. | 0.0385 |
| Neg. vs Failure  | 0.3997 |

---

## ep2184_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7864 |
| Coherence (Success) | 0.9401 |
| Coherence (Failure) | 0.8864 |
| Gradient Magnitude (Success) | 0.2225 |
| Gradient Magnitude (Failure) | 0.1883 |
| Activation Separation | 0.9123 |
| Cosine Distance | 0.0152 |
| Clusters | 2,371 |
| Noise Fraction | 0.1692 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7864 | 0.9401 | 0.8864 | 0.2225 | 0.1883 |
| G_IS (β=0.477) | 0.6801 | 0.9210 | 0.8724 | 0.1261 | 0.1136 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9899 | 0.9630 |
| cos(G_IS, G_reward)  | 0.8097 | 0.5899 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0172 | -0.0083 |
| cos(G_IS, Δθ)      | -0.0194 | -0.0079 |
| cos(G_reward, Δθ)  | -0.0147 | -0.0073 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,627 | 8.1823 | 0.9918 |
| Neutral  (r = 0) | 50,198 | 0.2570 | 0.9796 |
| Negative (r < 0) | 1,298 | 9.0948 | 0.5675 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4780 |
| Pos vs Negative  | 0.2530 |
| Neutral vs Neg.  | -0.2819 |
| Pos vs Failure   | 0.6026 |
| Neutral vs Fail. | -0.3507 |
| Neg. vs Failure  | 0.5039 |

---

## ep2439_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9111 |
| Coherence (Success) | 0.9574 |
| Coherence (Failure) | 0.9080 |
| Gradient Magnitude (Success) | 0.4155 |
| Gradient Magnitude (Failure) | 0.3844 |
| Activation Separation | 0.8557 |
| Cosine Distance | 0.0155 |
| Clusters | 2,269 |
| Noise Fraction | 0.1837 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9111 | 0.9574 | 0.9080 | 0.4155 | 0.3844 |
| G_IS (β=0.487) | 0.8812 | 0.9461 | 0.8920 | 0.2554 | 0.2417 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9944 | 0.9860 |
| cos(G_IS, G_reward)  | 0.9276 | 0.8741 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0097 | 0.0128 |
| cos(G_IS, Δθ)      | 0.0091 | 0.0124 |
| cos(G_reward, Δθ)  | 0.0051 | 0.0042 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,735 | 9.7783 | 0.9825 |
| Neutral  (r = 0) | 50,402 | 0.2140 | 0.9428 |
| Negative (r < 0) | 1,367 | 8.6723 | 0.6287 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0032 |
| Pos vs Negative  | 0.2088 |
| Neutral vs Neg.  | -0.1236 |
| Pos vs Failure   | 0.8445 |
| Neutral vs Fail. | 0.0348 |
| Neg. vs Failure  | 0.3679 |

---

## ep2697_lower4.000_upper5.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9301 |
| Coherence (Success) | 0.9581 |
| Coherence (Failure) | 0.8967 |
| Gradient Magnitude (Success) | 0.5736 |
| Gradient Magnitude (Failure) | 0.4478 |
| Activation Separation | 1.0688 |
| Cosine Distance | 0.0218 |
| Clusters | 2,366 |
| Noise Fraction | 0.1411 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9301 | 0.9581 | 0.8967 | 0.5736 | 0.4478 |
| G_IS (β=0.497) | 0.9073 | 0.9504 | 0.8908 | 0.3575 | 0.2809 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9952 | 0.9838 |
| cos(G_IS, G_reward)  | 0.9003 | 0.7610 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0041 | 0.0108 |
| cos(G_IS, Δθ)      | 0.0042 | 0.0109 |
| cos(G_reward, Δθ)  | -0.0045 | -0.0021 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,648 | 11.9947 | 0.9841 |
| Neutral  (r = 0) | 49,484 | 0.2102 | 0.9645 |
| Negative (r < 0) | 1,358 | 4.0093 | 0.9753 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0035 |
| Pos vs Negative  | -0.1332 |
| Neutral vs Neg.  | -0.3965 |
| Pos vs Failure   | 0.8065 |
| Neutral vs Fail. | 0.1313 |
| Neg. vs Failure  | 0.2230 |

---

## ep2957_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8959 |
| Coherence (Success) | 0.9799 |
| Coherence (Failure) | 0.9074 |
| Gradient Magnitude (Success) | 0.5368 |
| Gradient Magnitude (Failure) | 0.4379 |
| Activation Separation | 0.7151 |
| Cosine Distance | 0.0122 |
| Clusters | 2,414 |
| Noise Fraction | 0.1680 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8959 | 0.9799 | 0.9074 | 0.5368 | 0.4379 |
| G_IS (β=0.507) | 0.8616 | 0.9767 | 0.8904 | 0.3356 | 0.2630 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9987 | 0.9926 |
| cos(G_IS, G_reward)  | 0.8897 | 0.6802 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0070 | 0.0171 |
| cos(G_IS, Δθ)      | 0.0075 | 0.0189 |
| cos(G_reward, Δθ)  | 0.0007 | 0.0093 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,910 | 11.4557 | 0.9934 |
| Neutral  (r = 0) | 53,203 | 0.3313 | 0.9828 |
| Negative (r < 0) | 1,399 | 3.5276 | 0.9829 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1203 |
| Pos vs Negative  | -0.2374 |
| Neutral vs Neg.  | -0.5978 |
| Pos vs Failure   | 0.7757 |
| Neutral vs Fail. | 0.2354 |
| Neg. vs Failure  | -0.1880 |

---

## ep3201_lower4.900_upper6.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7583 |
| Coherence (Success) | 0.9584 |
| Coherence (Failure) | 0.8305 |
| Gradient Magnitude (Success) | 0.3938 |
| Gradient Magnitude (Failure) | 0.2539 |
| Activation Separation | 1.0752 |
| Cosine Distance | 0.0321 |
| Clusters | 2,140 |
| Noise Fraction | 0.2012 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7583 | 0.9584 | 0.8305 | 0.3938 | 0.2539 |
| G_IS (β=0.517) | 0.6581 | 0.9517 | 0.7971 | 0.2316 | 0.1420 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9973 | 0.9754 |
| cos(G_IS, G_reward)  | 0.8832 | 0.4715 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0076 | -0.0062 |
| cos(G_IS, Δθ)      | -0.0076 | -0.0053 |
| cos(G_reward, Δθ)  | -0.0032 | 0.0003 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 1,946 | 12.5950 | 0.9902 |
| Neutral  (r = 0) | 51,930 | 0.3155 | 0.9880 |
| Negative (r < 0) | 1,405 | 3.5989 | 0.9752 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6432 |
| Pos vs Negative  | -0.2016 |
| Neutral vs Neg.  | -0.3520 |
| Pos vs Failure   | 0.5991 |
| Neutral vs Fail. | -0.2614 |
| Neg. vs Failure  | -0.0833 |

---

## ep3449_lower5.900_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8003 |
| Coherence (Success) | 0.9627 |
| Coherence (Failure) | 0.8281 |
| Gradient Magnitude (Success) | 0.4625 |
| Gradient Magnitude (Failure) | 0.4439 |
| Activation Separation | 0.7500 |
| Cosine Distance | 0.0164 |
| Clusters | 2,094 |
| Noise Fraction | 0.2318 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8003 | 0.9627 | 0.8281 | 0.4625 | 0.4439 |
| G_IS (β=0.527) | 0.7683 | 0.9564 | 0.7978 | 0.2840 | 0.2579 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9983 | 0.9925 |
| cos(G_IS, G_reward)  | 0.9111 | 0.6804 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0065 | 0.0034 |
| cos(G_IS, Δθ)      | -0.0065 | 0.0046 |
| cos(G_reward, Δθ)  | -0.0117 | -0.0010 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,206 | 12.5833 | 0.9927 |
| Neutral  (r = 0) | 56,606 | 0.2848 | 0.9842 |
| Negative (r < 0) | 1,479 | 3.3598 | 0.9673 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3551 |
| Pos vs Negative  | -0.1700 |
| Neutral vs Neg.  | -0.3658 |
| Pos vs Failure   | 0.6970 |
| Neutral vs Fail. | 0.0623 |
| Neg. vs Failure  | -0.1814 |

---

## make_wood_pickaxe_ep3557_lower5.000_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8622 |
| Coherence (Success) | 0.9716 |
| Coherence (Failure) | 0.8191 |
| Gradient Magnitude (Success) | 0.4862 |
| Gradient Magnitude (Failure) | 0.4020 |
| Activation Separation | 0.8029 |
| Cosine Distance | 0.0184 |
| Clusters | 2,522 |
| Noise Fraction | 0.1966 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8622 | 0.9716 | 0.8191 | 0.4862 | 0.4020 |
| G_IS (β=0.532) | 0.8372 | 0.9692 | 0.7791 | 0.3030 | 0.2288 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9991 | 0.9963 |
| cos(G_IS, G_reward)  | 0.8652 | 0.5800 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0131 | 0.0043 |
| cos(G_IS, Δθ)      | 0.0136 | 0.0046 |
| cos(G_reward, Δθ)  | 0.0105 | -0.0104 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,277 | 11.2480 | 0.9911 |
| Neutral  (r = 0) | 60,228 | 0.3211 | 0.9801 |
| Negative (r < 0) | 1,565 | 3.0569 | 0.9630 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3116 |
| Pos vs Negative  | -0.1277 |
| Neutral vs Neg.  | -0.6637 |
| Pos vs Failure   | 0.7640 |
| Neutral vs Fail. | 0.0641 |
| Neg. vs Failure  | -0.2055 |

---

## ep3676_lower5.900_upper7.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6782 |
| Coherence (Success) | 0.9417 |
| Coherence (Failure) | 0.8147 |
| Gradient Magnitude (Success) | 0.3433 |
| Gradient Magnitude (Failure) | 0.3609 |
| Activation Separation | 1.1596 |
| Cosine Distance | 0.0381 |
| Clusters | 2,382 |
| Noise Fraction | 0.1983 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6782 | 0.9417 | 0.8147 | 0.3433 | 0.3609 |
| G_IS (β=0.537) | 0.6083 | 0.9337 | 0.7942 | 0.2055 | 0.2194 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9941 | 0.9802 |
| cos(G_IS, G_reward)  | 0.8495 | 0.7397 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0018 | -0.0044 |
| cos(G_IS, Δθ)      | -0.0007 | -0.0031 |
| cos(G_reward, Δθ)  | -0.0015 | -0.0010 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,252 | 10.7905 | 0.9891 |
| Neutral  (r = 0) | 57,519 | 0.2530 | 0.9812 |
| Negative (r < 0) | 1,516 | 3.5564 | 0.9655 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6754 |
| Pos vs Negative  | -0.1425 |
| Neutral vs Neg.  | -0.0445 |
| Pos vs Failure   | 0.5624 |
| Neutral vs Fail. | -0.1757 |
| Neg. vs Failure  | 0.2588 |

---

## ep3908_lower5.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6709 |
| Coherence (Success) | 0.8810 |
| Coherence (Failure) | 0.7762 |
| Gradient Magnitude (Success) | 0.2683 |
| Gradient Magnitude (Failure) | 0.3147 |
| Activation Separation | 0.5178 |
| Cosine Distance | 0.0094 |
| Clusters | 2,285 |
| Noise Fraction | 0.2135 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6709 | 0.8810 | 0.7762 | 0.2683 | 0.3147 |
| G_IS (β=0.547) | 0.6431 | 0.8675 | 0.7731 | 0.1652 | 0.2097 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9842 | 0.9727 |
| cos(G_IS, G_reward)  | 0.6334 | 0.6432 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0083 | -0.0042 |
| cos(G_IS, Δθ)      | -0.0064 | -0.0016 |
| cos(G_reward, Δθ)  | -0.0156 | -0.0133 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,355 | 8.4714 | 0.9863 |
| Neutral  (r = 0) | 56,866 | 0.3303 | 0.9852 |
| Negative (r < 0) | 1,496 | 5.2674 | 0.7082 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4018 |
| Pos vs Negative  | 0.2743 |
| Neutral vs Neg.  | -0.1524 |
| Pos vs Failure   | 0.5170 |
| Neutral vs Fail. | 0.2128 |
| Neg. vs Failure  | 0.4448 |

---

## ep4125_lower6.000_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8522 |
| Coherence (Success) | 0.9172 |
| Coherence (Failure) | 0.7738 |
| Gradient Magnitude (Success) | 0.2578 |
| Gradient Magnitude (Failure) | 0.2646 |
| Activation Separation | 0.5122 |
| Cosine Distance | 0.0104 |
| Clusters | 2,023 |
| Noise Fraction | 0.2299 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8522 | 0.9172 | 0.7738 | 0.2578 | 0.2646 |
| G_IS (β=0.557) | 0.7892 | 0.8951 | 0.7352 | 0.1453 | 0.1496 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9952 | 0.9752 |
| cos(G_IS, G_reward)  | 0.8830 | 0.7566 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0085 | -0.0111 |
| cos(G_IS, Δθ)      | -0.0095 | -0.0118 |
| cos(G_reward, Δθ)  | -0.0060 | -0.0109 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,495 | 9.5346 | 0.9884 |
| Neutral  (r = 0) | 59,595 | 0.2302 | 0.9857 |
| Negative (r < 0) | 1,580 | 6.4380 | 0.6807 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5031 |
| Pos vs Negative  | 0.2932 |
| Neutral vs Neg.  | -0.2664 |
| Pos vs Failure   | 0.8045 |
| Neutral vs Fail. | -0.3898 |
| Neg. vs Failure  | 0.4064 |

---

## ep4341_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9375 |
| Coherence (Success) | 0.9272 |
| Coherence (Failure) | 0.8466 |
| Gradient Magnitude (Success) | 0.3090 |
| Gradient Magnitude (Failure) | 0.4175 |
| Activation Separation | 0.4745 |
| Cosine Distance | 0.0095 |
| Clusters | 1,929 |
| Noise Fraction | 0.2640 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9375 | 0.9272 | 0.8466 | 0.3090 | 0.4175 |
| G_IS (β=0.567) | 0.9350 | 0.9229 | 0.8435 | 0.1854 | 0.2572 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9915 | 0.9910 |
| cos(G_IS, G_reward)  | 0.7287 | 0.8117 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0064 | -0.0025 |
| cos(G_IS, Δθ)      | -0.0042 | -0.0003 |
| cos(G_reward, Δθ)  | -0.0155 | -0.0038 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,726 | 8.1433 | 0.9888 |
| Neutral  (r = 0) | 61,569 | 0.2856 | 0.9872 |
| Negative (r < 0) | 1,627 | 3.5982 | 0.9764 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1779 |
| Pos vs Negative  | -0.0379 |
| Neutral vs Neg.  | 0.1773 |
| Pos vs Failure   | 0.8048 |
| Neutral vs Fail. | 0.2213 |
| Neg. vs Failure  | 0.3538 |

---

## make_wood_sword_ep4521_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9023 |
| Coherence (Success) | 0.9009 |
| Coherence (Failure) | 0.8773 |
| Gradient Magnitude (Success) | 0.2595 |
| Gradient Magnitude (Failure) | 0.3794 |
| Activation Separation | 0.4389 |
| Cosine Distance | 0.0083 |
| Clusters | 2,109 |
| Noise Fraction | 0.2300 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9023 | 0.9009 | 0.8773 | 0.2595 | 0.3794 |
| G_IS (β=0.575) | 0.9000 | 0.8863 | 0.8738 | 0.1484 | 0.2285 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9884 | 0.9887 |
| cos(G_IS, G_reward)  | 0.6225 | 0.8446 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0006 | 0.0028 |
| cos(G_IS, Δθ)      | 0.0009 | 0.0033 |
| cos(G_reward, Δθ)  | -0.0032 | -0.0030 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,541 | 7.6702 | 0.9868 |
| Neutral  (r = 0) | 59,815 | 0.2897 | 0.9816 |
| Negative (r < 0) | 1,555 | 12.6768 | 0.4525 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.2191 |
| Pos vs Negative  | 0.4013 |
| Neutral vs Neg.  | -0.2447 |
| Pos vs Failure   | 0.7625 |
| Neutral vs Fail. | 0.1752 |
| Neg. vs Failure  | 0.2496 |

---

## ep4564_lower6.900_upper7.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5656 |
| Coherence (Success) | 0.8845 |
| Coherence (Failure) | 0.7378 |
| Gradient Magnitude (Success) | 0.1882 |
| Gradient Magnitude (Failure) | 0.2722 |
| Activation Separation | 0.4220 |
| Cosine Distance | 0.0078 |
| Clusters | 2,089 |
| Noise Fraction | 0.2504 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5656 | 0.8845 | 0.7378 | 0.1882 | 0.2722 |
| G_IS (β=0.577) | 0.4908 | 0.8529 | 0.7227 | 0.1015 | 0.1608 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9863 | 0.9764 |
| cos(G_IS, G_reward)  | 0.6552 | 0.7835 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0012 | -0.0021 |
| cos(G_IS, Δθ)      | -0.0009 | -0.0016 |
| cos(G_reward, Δθ)  | -0.0037 | -0.0054 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,762 | 6.8929 | 0.9900 |
| Neutral  (r = 0) | 64,809 | 0.2632 | 0.9835 |
| Negative (r < 0) | 1,643 | 9.3875 | 0.4698 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5570 |
| Pos vs Negative  | 0.3657 |
| Neutral vs Neg.  | -0.3451 |
| Pos vs Failure   | 0.5330 |
| Neutral vs Fail. | 0.0097 |
| Neg. vs Failure  | 0.1732 |

---

## ep4785_lower6.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6837 |
| Coherence (Success) | 0.8811 |
| Coherence (Failure) | 0.7146 |
| Gradient Magnitude (Success) | 0.1748 |
| Gradient Magnitude (Failure) | 0.2346 |
| Activation Separation | 0.4022 |
| Cosine Distance | 0.0064 |
| Clusters | 2,101 |
| Noise Fraction | 0.2646 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6837 | 0.8811 | 0.7146 | 0.1748 | 0.2346 |
| G_IS (β=0.587) | 0.5678 | 0.8487 | 0.6842 | 0.0931 | 0.1346 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9828 | 0.9642 |
| cos(G_IS, G_reward)  | 0.5760 | 0.5697 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0016 | 0.0003 |
| cos(G_IS, Δθ)      | -0.0014 | 0.0012 |
| cos(G_reward, Δθ)  | -0.0060 | -0.0062 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,745 | 6.0472 | 0.9886 |
| Neutral  (r = 0) | 62,987 | 0.3056 | 0.9877 |
| Negative (r < 0) | 1,635 | 3.5423 | 0.9770 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5889 |
| Pos vs Negative  | -0.1010 |
| Neutral vs Neg.  | -0.1114 |
| Pos vs Failure   | 0.1218 |
| Neutral vs Fail. | 0.4287 |
| Neg. vs Failure  | 0.1791 |

---

## ep5002_lower6.900_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7919 |
| Coherence (Success) | 0.9266 |
| Coherence (Failure) | 0.7364 |
| Gradient Magnitude (Success) | 0.2570 |
| Gradient Magnitude (Failure) | 0.3089 |
| Activation Separation | 0.3100 |
| Cosine Distance | 0.0041 |
| Clusters | 2,075 |
| Noise Fraction | 0.2283 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (9) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7919 | 0.9266 | 0.7364 | 0.2570 | 0.3089 |
| G_IS (β=0.597) | 0.7669 | 0.9051 | 0.7153 | 0.1345 | 0.1726 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9910 | 0.9808 |
| cos(G_IS, G_reward)  | 0.8607 | 0.8550 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0024 | 0.0005 |
| cos(G_IS, Δθ)      | 0.0020 | 0.0002 |
| cos(G_reward, Δθ)  | -0.0015 | -0.0048 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,713 | 8.3880 | 0.9896 |
| Neutral  (r = 0) | 61,093 | 0.2311 | 0.9817 |
| Negative (r < 0) | 1,591 | 3.5775 | 0.9711 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5330 |
| Pos vs Negative  | 0.0049 |
| Neutral vs Neg.  | -0.2058 |
| Pos vs Failure   | 0.7805 |
| Neutral vs Fail. | -0.3290 |
| Neg. vs Failure  | 0.2863 |

---

## ep5222_lower7.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9025 |
| Coherence (Success) | 0.9352 |
| Coherence (Failure) | 0.8096 |
| Gradient Magnitude (Success) | 0.3230 |
| Gradient Magnitude (Failure) | 0.3095 |
| Activation Separation | 0.3024 |
| Cosine Distance | 0.0035 |
| Clusters | 1,998 |
| Noise Fraction | 0.2383 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9025 | 0.9352 | 0.8096 | 0.3230 | 0.3095 |
| G_IS (β=0.607) | 0.8783 | 0.9310 | 0.7958 | 0.1742 | 0.1697 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9923 | 0.9825 |
| cos(G_IS, G_reward)  | 0.8600 | 0.8256 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0061 | 0.0078 |
| cos(G_IS, Δθ)      | 0.0070 | 0.0086 |
| cos(G_reward, Δθ)  | 0.0000 | 0.0036 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,712 | 8.5436 | 0.9881 |
| Neutral  (r = 0) | 57,074 | 0.2236 | 0.9713 |
| Negative (r < 0) | 1,550 | 3.9396 | 0.9702 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5665 |
| Pos vs Negative  | -0.0641 |
| Neutral vs Neg.  | -0.0149 |
| Pos vs Failure   | 0.8172 |
| Neutral vs Fail. | -0.3647 |
| Neg. vs Failure  | 0.2557 |

---

## ep5445_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8949 |
| Coherence (Success) | 0.9338 |
| Coherence (Failure) | 0.7624 |
| Gradient Magnitude (Success) | 0.3160 |
| Gradient Magnitude (Failure) | 0.3385 |
| Activation Separation | 0.3728 |
| Cosine Distance | 0.0055 |
| Clusters | 1,970 |
| Noise Fraction | 0.2688 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8949 | 0.9338 | 0.7624 | 0.3160 | 0.3385 |
| G_IS (β=0.617) | 0.8711 | 0.9159 | 0.7240 | 0.1725 | 0.1827 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9934 | 0.9878 |
| cos(G_IS, G_reward)  | 0.5350 | 0.1635 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0003 | 0.0045 |
| cos(G_IS, Δθ)      | 0.0009 | 0.0063 |
| cos(G_reward, Δθ)  | 0.0022 | 0.0033 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,773 | 7.0049 | 0.9883 |
| Neutral  (r = 0) | 60,475 | 0.4231 | 0.9818 |
| Negative (r < 0) | 1,622 | 3.8120 | 0.9732 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7224 |
| Pos vs Negative  | -0.0485 |
| Neutral vs Neg.  | -0.1652 |
| Pos vs Failure   | -0.1929 |
| Neutral vs Fail. | 0.6705 |
| Neg. vs Failure  | 0.0754 |

---

## ep5652_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8786 |
| Coherence (Success) | 0.9268 |
| Coherence (Failure) | 0.8145 |
| Gradient Magnitude (Success) | 0.3104 |
| Gradient Magnitude (Failure) | 0.2923 |
| Activation Separation | 0.3529 |
| Cosine Distance | 0.0051 |
| Clusters | 2,134 |
| Noise Fraction | 0.2549 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8786 | 0.9268 | 0.8145 | 0.3104 | 0.2923 |
| G_IS (β=0.628) | 0.8342 | 0.9104 | 0.7776 | 0.1605 | 0.1499 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9960 | 0.9869 |
| cos(G_IS, G_reward)  | 0.8325 | 0.7785 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0096 | 0.0051 |
| cos(G_IS, Δθ)      | 0.0103 | 0.0048 |
| cos(G_reward, Δθ)  | 0.0024 | -0.0026 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,822 | 7.0012 | 0.9866 |
| Neutral  (r = 0) | 61,646 | 0.2473 | 0.9783 |
| Negative (r < 0) | 1,646 | 7.1184 | 0.6840 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5173 |
| Pos vs Negative  | 0.2161 |
| Neutral vs Neg.  | -0.0965 |
| Pos vs Failure   | 0.7480 |
| Neutral vs Fail. | -0.2136 |
| Neg. vs Failure  | 0.3260 |

---

## ep5858_lower6.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7548 |
| Coherence (Success) | 0.9012 |
| Coherence (Failure) | 0.7886 |
| Gradient Magnitude (Success) | 0.2489 |
| Gradient Magnitude (Failure) | 0.3668 |
| Activation Separation | 0.3867 |
| Cosine Distance | 0.0063 |
| Clusters | 2,135 |
| Noise Fraction | 0.2453 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7548 | 0.9012 | 0.7886 | 0.2489 | 0.3668 |
| G_IS (β=0.638) | 0.7097 | 0.8775 | 0.7692 | 0.1333 | 0.2231 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9866 | 0.9824 |
| cos(G_IS, G_reward)  | 0.8696 | 0.8244 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0026 | 0.0020 |
| cos(G_IS, Δθ)      | -0.0023 | 0.0027 |
| cos(G_reward, Δθ)  | -0.0003 | 0.0047 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,645 | 7.5706 | 0.9848 |
| Neutral  (r = 0) | 56,577 | 0.2387 | 0.9811 |
| Negative (r < 0) | 1,515 | 3.6334 | 0.9721 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5927 |
| Pos vs Negative  | -0.0430 |
| Neutral vs Neg.  | -0.1115 |
| Pos vs Failure   | 0.7618 |
| Neutral vs Fail. | -0.3149 |
| Neg. vs Failure  | 0.1947 |

---

## ep6066_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.0114 |
| Coherence (Success) | 0.8643 |
| Coherence (Failure) | 0.6556 |
| Gradient Magnitude (Success) | 0.2228 |
| Gradient Magnitude (Failure) | 0.2723 |
| Activation Separation | 0.3856 |
| Cosine Distance | 0.0065 |
| Clusters | 2,015 |
| Noise Fraction | 0.2556 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.0114 | 0.8643 | 0.6556 | 0.2228 | 0.2723 |
| G_IS (β=0.648) | -0.1327 | 0.8272 | 0.6621 | 0.1199 | 0.1669 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9827 | 0.9726 |
| cos(G_IS, G_reward)  | 0.5122 | 0.7123 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0012 | 0.0019 |
| cos(G_IS, Δθ)      | -0.0005 | 0.0026 |
| cos(G_reward, Δθ)  | -0.0031 | -0.0019 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,799 | 5.5759 | 0.9747 |
| Neutral  (r = 0) | 59,136 | 0.3037 | 0.9807 |
| Negative (r < 0) | 1,596 | 3.4638 | 0.9704 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6342 |
| Pos vs Negative  | -0.0152 |
| Neutral vs Neg.  | -0.1774 |
| Pos vs Failure   | 0.4685 |
| Neutral vs Fail. | -0.2634 |
| Neg. vs Failure  | 0.4088 |

---

## ep6275_lower7.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6525 |
| Coherence (Success) | 0.9251 |
| Coherence (Failure) | 0.7370 |
| Gradient Magnitude (Success) | 0.3034 |
| Gradient Magnitude (Failure) | 0.2775 |
| Activation Separation | 0.4090 |
| Cosine Distance | 0.0073 |
| Clusters | 2,337 |
| Noise Fraction | 0.2358 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6525 | 0.9251 | 0.7370 | 0.3034 | 0.2775 |
| G_IS (β=0.658) | 0.5567 | 0.9132 | 0.7161 | 0.1537 | 0.1489 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9915 | 0.9742 |
| cos(G_IS, G_reward)  | 0.7908 | 0.5855 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0151 | 0.0127 |
| cos(G_IS, Δθ)      | 0.0161 | 0.0125 |
| cos(G_reward, Δθ)  | 0.0086 | 0.0037 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,779 | 5.3960 | 0.9775 |
| Neutral  (r = 0) | 62,038 | 0.2398 | 0.9728 |
| Negative (r < 0) | 1,611 | 7.7109 | 0.6282 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4601 |
| Pos vs Negative  | 0.2330 |
| Neutral vs Neg.  | -0.0416 |
| Pos vs Failure   | 0.6180 |
| Neutral vs Fail. | -0.0214 |
| Neg. vs Failure  | 0.3704 |

---

## collect_stone_ep6399_lower7.000_upper8.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9927 |
| Coherence (Success) | 0.9470 |
| Coherence (Failure) | 0.8309 |
| Gradient Magnitude (Success) | 0.7798 |
| Gradient Magnitude (Failure) | 0.6678 |
| Activation Separation | 0.4253 |
| Cosine Distance | 0.0072 |
| Clusters | 2,357 |
| Noise Fraction | 0.2241 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (10) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9927 | 0.9470 | 0.8309 | 0.7798 | 0.6678 |
| G_IS (β=0.664) | 0.9921 | 0.9364 | 0.7762 | 0.4306 | 0.3575 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9987 | 0.9970 |
| cos(G_IS, G_reward)  | 0.8683 | 0.5376 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0062 | 0.0041 |
| cos(G_IS, Δθ)      | 0.0055 | 0.0027 |
| cos(G_reward, Δθ)  | 0.0116 | 0.0088 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,713 | 5.9850 | 0.9833 |
| Neutral  (r = 0) | 58,491 | 0.6263 | 0.9806 |
| Negative (r < 0) | 1,572 | 3.4064 | 0.9626 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0495 |
| Pos vs Negative  | -0.1036 |
| Neutral vs Neg.  | -0.3422 |
| Pos vs Failure   | 0.3483 |
| Neutral vs Fail. | 0.8543 |
| Neg. vs Failure  | -0.2355 |

---

## ep6492_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8552 |
| Coherence (Success) | 0.8975 |
| Coherence (Failure) | 0.6578 |
| Gradient Magnitude (Success) | 0.2340 |
| Gradient Magnitude (Failure) | 0.2308 |
| Activation Separation | 0.4161 |
| Cosine Distance | 0.0075 |
| Clusters | 2,299 |
| Noise Fraction | 0.2420 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8552 | 0.8975 | 0.6578 | 0.2340 | 0.2308 |
| G_IS (β=0.668) | 0.8100 | 0.8752 | 0.6176 | 0.1213 | 0.1207 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9818 | 0.9629 |
| cos(G_IS, G_reward)  | 0.7377 | 0.5546 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0143 | 0.0069 |
| cos(G_IS, Δθ)      | 0.0152 | 0.0069 |
| cos(G_reward, Δθ)  | 0.0108 | 0.0075 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,798 | 6.4002 | 0.9824 |
| Neutral  (r = 0) | 63,425 | 0.2680 | 0.9816 |
| Negative (r < 0) | 1,635 | 3.5580 | 0.9642 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7121 |
| Pos vs Negative  | 0.0086 |
| Neutral vs Neg.  | -0.0839 |
| Pos vs Failure   | 0.4735 |
| Neutral vs Fail. | -0.0918 |
| Neg. vs Failure  | 0.2735 |

---

## ep6704_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8301 |
| Coherence (Success) | 0.8913 |
| Coherence (Failure) | 0.7632 |
| Gradient Magnitude (Success) | 0.2117 |
| Gradient Magnitude (Failure) | 0.2783 |
| Activation Separation | 0.4661 |
| Cosine Distance | 0.0094 |
| Clusters | 2,137 |
| Noise Fraction | 0.2440 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8301 | 0.8913 | 0.7632 | 0.2117 | 0.2783 |
| G_IS (β=0.678) | 0.8005 | 0.8716 | 0.7404 | 0.1041 | 0.1536 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9856 | 0.9742 |
| cos(G_IS, G_reward)  | 0.8198 | 0.8489 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0078 | 0.0061 |
| cos(G_IS, Δθ)      | 0.0078 | 0.0055 |
| cos(G_reward, Δθ)  | 0.0117 | 0.0105 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,794 | 6.4866 | 0.9839 |
| Neutral  (r = 0) | 61,447 | 0.2334 | 0.9781 |
| Negative (r < 0) | 1,589 | 3.6554 | 0.9735 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4799 |
| Pos vs Negative  | -0.0256 |
| Neutral vs Neg.  | -0.0940 |
| Pos vs Failure   | 0.7527 |
| Neutral vs Fail. | -0.2663 |
| Neg. vs Failure  | 0.3267 |

---

## ep6913_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7543 |
| Coherence (Success) | 0.8507 |
| Coherence (Failure) | 0.7261 |
| Gradient Magnitude (Success) | 0.2117 |
| Gradient Magnitude (Failure) | 0.3143 |
| Activation Separation | 0.4348 |
| Cosine Distance | 0.0086 |
| Clusters | 2,479 |
| Noise Fraction | 0.2316 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7543 | 0.8507 | 0.7261 | 0.2117 | 0.3143 |
| G_IS (β=0.688) | 0.7192 | 0.8104 | 0.7110 | 0.1023 | 0.1693 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9847 | 0.9818 |
| cos(G_IS, G_reward)  | 0.8234 | 0.8342 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0010 | 0.0009 |
| cos(G_IS, Δθ)      | 0.0011 | 0.0008 |
| cos(G_reward, Δθ)  | -0.0063 | -0.0066 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,821 | 6.8131 | 0.9797 |
| Neutral  (r = 0) | 67,269 | 0.2149 | 0.9677 |
| Negative (r < 0) | 1,711 | 3.2435 | 0.9627 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5262 |
| Pos vs Negative  | -0.0238 |
| Neutral vs Neg.  | -0.2511 |
| Pos vs Failure   | 0.7858 |
| Neutral vs Fail. | -0.3194 |
| Neg. vs Failure  | 0.2515 |

---

## ep7121_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9802 |
| Coherence (Success) | 0.9211 |
| Coherence (Failure) | 0.7341 |
| Gradient Magnitude (Success) | 0.6192 |
| Gradient Magnitude (Failure) | 0.5292 |
| Activation Separation | 0.4052 |
| Cosine Distance | 0.0076 |
| Clusters | 2,421 |
| Noise Fraction | 0.2546 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9802 | 0.9211 | 0.7341 | 0.6192 | 0.5292 |
| G_IS (β=0.698) | 0.9754 | 0.9026 | 0.6916 | 0.3286 | 0.2771 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9977 | 0.9938 |
| cos(G_IS, G_reward)  | 0.8271 | 0.5306 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0022 | 0.0036 |
| cos(G_IS, Δθ)      | 0.0025 | 0.0042 |
| cos(G_reward, Δθ)  | 0.0003 | 0.0013 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,861 | 5.6748 | 0.9844 |
| Neutral  (r = 0) | 64,021 | 0.4438 | 0.9781 |
| Negative (r < 0) | 1,669 | 3.5738 | 0.9647 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.2655 |
| Pos vs Negative  | -0.1168 |
| Neutral vs Neg.  | -0.3382 |
| Pos vs Failure   | 0.3010 |
| Neutral vs Fail. | 0.7381 |
| Neg. vs Failure  | -0.2331 |

---

## ep7331_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8576 |
| Coherence (Success) | 0.8612 |
| Coherence (Failure) | 0.6775 |
| Gradient Magnitude (Success) | 0.2365 |
| Gradient Magnitude (Failure) | 0.2253 |
| Activation Separation | 0.3276 |
| Cosine Distance | 0.0050 |
| Clusters | 1,887 |
| Noise Fraction | 0.2610 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8576 | 0.8612 | 0.6775 | 0.2365 | 0.2253 |
| G_IS (β=0.708) | 0.8154 | 0.8123 | 0.6262 | 0.1267 | 0.1212 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9793 | 0.9621 |
| cos(G_IS, G_reward)  | 0.3453 | 0.1424 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0045 | 0.0058 |
| cos(G_IS, Δθ)      | 0.0050 | 0.0064 |
| cos(G_reward, Δθ)  | 0.0045 | 0.0000 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,835 | 4.8374 | 0.9659 |
| Neutral  (r = 0) | 58,633 | 0.3745 | 0.9852 |
| Negative (r < 0) | 1,594 | 3.5274 | 0.9644 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7121 |
| Pos vs Negative  | -0.1016 |
| Neutral vs Neg.  | -0.1295 |
| Pos vs Failure   | -0.2920 |
| Neutral vs Fail. | 0.5989 |
| Neg. vs Failure  | -0.0639 |

---

## ep7545_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8013 |
| Coherence (Success) | 0.8559 |
| Coherence (Failure) | 0.7094 |
| Gradient Magnitude (Success) | 0.2073 |
| Gradient Magnitude (Failure) | 0.2331 |
| Activation Separation | 0.3979 |
| Cosine Distance | 0.0072 |
| Clusters | 2,153 |
| Noise Fraction | 0.2606 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Drink, Eat Cow, Eat Plant, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8013 | 0.8559 | 0.7094 | 0.2073 | 0.2331 |
| G_IS (β=0.718) | 0.6980 | 0.8127 | 0.6382 | 0.0933 | 0.1169 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9802 | 0.9562 |
| cos(G_IS, G_reward)  | 0.7817 | 0.7394 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0144 | 0.0124 |
| cos(G_IS, Δθ)      | 0.0144 | 0.0101 |
| cos(G_reward, Δθ)  | 0.0113 | 0.0114 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,792 | 5.5588 | 0.9695 |
| Neutral  (r = 0) | 59,604 | 0.2516 | 0.9811 |
| Negative (r < 0) | 1,609 | 3.6965 | 0.9687 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4458 |
| Pos vs Negative  | -0.1190 |
| Neutral vs Neg.  | -0.0829 |
| Pos vs Failure   | 0.7431 |
| Neutral vs Fail. | -0.0700 |
| Neg. vs Failure  | 0.1564 |

---

## collect_coal_ep7687_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9432 |
| Coherence (Success) | 0.8945 |
| Coherence (Failure) | 0.7324 |
| Gradient Magnitude (Success) | 0.3416 |
| Gradient Magnitude (Failure) | 0.2527 |
| Activation Separation | 0.3004 |
| Cosine Distance | 0.0042 |
| Clusters | 1,969 |
| Noise Fraction | 0.2611 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Drink, Eat Cow, Eat Plant, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9432 | 0.8945 | 0.7324 | 0.3416 | 0.2527 |
| G_IS (β=0.725) | 0.9424 | 0.8744 | 0.6844 | 0.1844 | 0.1395 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9900 | 0.9718 |
| cos(G_IS, G_reward)  | 0.5638 | 0.6782 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0069 | 0.0096 |
| cos(G_IS, Δθ)      | 0.0066 | 0.0089 |
| cos(G_reward, Δθ)  | 0.0130 | 0.0110 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,747 | 5.8023 | 0.9819 |
| Neutral  (r = 0) | 57,642 | 0.3877 | 0.9730 |
| Negative (r < 0) | 1,585 | 3.3306 | 0.9614 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0967 |
| Pos vs Negative  | -0.1175 |
| Neutral vs Neg.  | -0.0475 |
| Pos vs Failure   | 0.6638 |
| Neutral vs Fail. | 0.5362 |
| Neg. vs Failure  | 0.0587 |

---

## ep7750_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8114 |
| Coherence (Success) | 0.7544 |
| Coherence (Failure) | 0.6959 |
| Gradient Magnitude (Success) | 0.1637 |
| Gradient Magnitude (Failure) | 0.2273 |
| Activation Separation | 0.3145 |
| Cosine Distance | 0.0049 |
| Clusters | 1,991 |
| Noise Fraction | 0.2529 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8114 | 0.7544 | 0.6959 | 0.1637 | 0.2273 |
| G_IS (β=0.728) | 0.8027 | 0.7265 | 0.7091 | 0.0837 | 0.1331 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9532 | 0.9601 |
| cos(G_IS, G_reward)  | 0.2455 | 0.5157 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0018 | 0.0083 |
| cos(G_IS, Δθ)      | 0.0036 | 0.0096 |
| cos(G_reward, Δθ)  | 0.0013 | 0.0037 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,743 | 5.6019 | 0.9696 |
| Neutral  (r = 0) | 57,117 | 0.3175 | 0.9781 |
| Negative (r < 0) | 1,546 | 3.5467 | 0.9682 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6192 |
| Pos vs Negative  | -0.0631 |
| Neutral vs Neg.  | -0.0309 |
| Pos vs Failure   | 0.2744 |
| Neutral vs Fail. | 0.2903 |
| Neg. vs Failure  | 0.3729 |

---

## ep7956_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6475 |
| Coherence (Success) | 0.8508 |
| Coherence (Failure) | 0.6593 |
| Gradient Magnitude (Success) | 0.1800 |
| Gradient Magnitude (Failure) | 0.1891 |
| Activation Separation | 0.3226 |
| Cosine Distance | 0.0044 |
| Clusters | 2,059 |
| Noise Fraction | 0.2655 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6475 | 0.8508 | 0.6593 | 0.1800 | 0.1891 |
| G_IS (β=0.738) | 0.5236 | 0.8084 | 0.6293 | 0.0777 | 0.0962 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9609 | 0.9332 |
| cos(G_IS, G_reward)  | 0.6475 | 0.4744 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0141 | 0.0069 |
| cos(G_IS, Δθ)      | 0.0132 | 0.0024 |
| cos(G_reward, Δθ)  | 0.0119 | 0.0091 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,828 | 5.0741 | 0.9790 |
| Neutral  (r = 0) | 60,070 | 0.3016 | 0.9769 |
| Negative (r < 0) | 1,631 | 3.3202 | 0.9647 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5916 |
| Pos vs Negative  | -0.2055 |
| Neutral vs Neg.  | 0.0562 |
| Pos vs Failure   | 0.3978 |
| Neutral vs Fail. | 0.1958 |
| Neg. vs Failure  | 0.2771 |

---

## ep8172_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9678 |
| Coherence (Success) | 0.8805 |
| Coherence (Failure) | 0.7805 |
| Gradient Magnitude (Success) | 0.2644 |
| Gradient Magnitude (Failure) | 0.2658 |
| Activation Separation | 0.3795 |
| Cosine Distance | 0.0067 |
| Clusters | 2,264 |
| Noise Fraction | 0.2530 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9678 | 0.8805 | 0.7805 | 0.2644 | 0.2658 |
| G_IS (β=0.748) | 0.9595 | 0.8645 | 0.7534 | 0.1228 | 0.1196 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9664 | 0.9597 |
| cos(G_IS, G_reward)  | 0.6226 | 0.5957 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0232 | 0.0217 |
| cos(G_IS, Δθ)      | 0.0237 | 0.0220 |
| cos(G_reward, Δθ)  | 0.0114 | 0.0131 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,801 | 5.3555 | 0.9781 |
| Neutral  (r = 0) | 61,605 | 0.2619 | 0.9596 |
| Negative (r < 0) | 1,648 | 3.6755 | 0.9605 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5049 |
| Pos vs Negative  | -0.1248 |
| Neutral vs Neg.  | 0.1167 |
| Pos vs Failure   | 0.5746 |
| Neutral vs Fail. | -0.1313 |
| Neg. vs Failure  | 0.2848 |

---

## ep8374_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8515 |
| Coherence (Success) | 0.9027 |
| Coherence (Failure) | 0.7440 |
| Gradient Magnitude (Success) | 0.2547 |
| Gradient Magnitude (Failure) | 0.3111 |
| Activation Separation | 0.2717 |
| Cosine Distance | 0.0033 |
| Clusters | 2,146 |
| Noise Fraction | 0.2611 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (11) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8515 | 0.9027 | 0.7440 | 0.2547 | 0.3111 |
| G_IS (β=0.758) | 0.8229 | 0.8762 | 0.6857 | 0.1104 | 0.1475 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9838 | 0.9782 |
| cos(G_IS, G_reward)  | 0.8465 | 0.7733 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0187 | 0.0166 |
| cos(G_IS, Δθ)      | 0.0182 | 0.0146 |
| cos(G_reward, Δθ)  | 0.0223 | 0.0194 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,772 | 6.9574 | 0.9752 |
| Neutral  (r = 0) | 60,301 | 0.2310 | 0.9718 |
| Negative (r < 0) | 1,604 | 3.3942 | 0.9565 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5155 |
| Pos vs Negative  | -0.0844 |
| Neutral vs Neg.  | -0.2067 |
| Pos vs Failure   | 0.7888 |
| Neutral vs Fail. | -0.2085 |
| Neg. vs Failure  | 0.2078 |

---

## ep8578_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8901 |
| Coherence (Success) | 0.8347 |
| Coherence (Failure) | 0.6537 |
| Gradient Magnitude (Success) | 0.3133 |
| Gradient Magnitude (Failure) | 0.2426 |
| Activation Separation | 0.3866 |
| Cosine Distance | 0.0060 |
| Clusters | 2,129 |
| Noise Fraction | 0.2702 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8901 | 0.8347 | 0.6537 | 0.3133 | 0.2426 |
| G_IS (β=0.768) | 0.8804 | 0.8076 | 0.6362 | 0.1523 | 0.1170 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9829 | 0.9570 |
| cos(G_IS, G_reward)  | 0.3955 | 0.3852 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0002 | 0.0061 |
| cos(G_IS, Δθ)      | 0.0011 | 0.0085 |
| cos(G_reward, Δθ)  | -0.0020 | 0.0023 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,895 | 6.2698 | 0.9766 |
| Neutral  (r = 0) | 59,767 | 0.4507 | 0.9754 |
| Negative (r < 0) | 1,646 | 3.6876 | 0.9546 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5473 |
| Pos vs Negative  | -0.1425 |
| Neutral vs Neg.  | -0.0448 |
| Pos vs Failure   | 0.0678 |
| Neutral vs Fail. | 0.4798 |
| Neg. vs Failure  | -0.0077 |

---

## ep8792_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9529 |
| Coherence (Success) | 0.9209 |
| Coherence (Failure) | 0.7688 |
| Gradient Magnitude (Success) | 0.5428 |
| Gradient Magnitude (Failure) | 0.6947 |
| Activation Separation | 0.3565 |
| Cosine Distance | 0.0056 |
| Clusters | 2,022 |
| Noise Fraction | 0.2575 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Drink, Eat Cow, Eat Plant, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9529 | 0.9209 | 0.7688 | 0.5428 | 0.6947 |
| G_IS (β=0.779) | 0.9498 | 0.8974 | 0.7340 | 0.2590 | 0.3418 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9941 | 0.9934 |
| cos(G_IS, G_reward)  | 0.7535 | 0.8528 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0047 | 0.0044 |
| cos(G_IS, Δθ)      | 0.0048 | 0.0045 |
| cos(G_reward, Δθ)  | 0.0028 | 0.0027 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,958 | 6.9243 | 0.9691 |
| Neutral  (r = 0) | 60,020 | 0.4646 | 0.9674 |
| Negative (r < 0) | 1,626 | 3.2571 | 0.9537 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.1590 |
| Pos vs Negative  | -0.1244 |
| Neutral vs Neg.  | -0.0254 |
| Pos vs Failure   | 0.6796 |
| Neutral vs Fail. | 0.7529 |
| Neg. vs Failure  | 0.1338 |

---

## ep9001_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4775 |
| Coherence (Success) | 0.7962 |
| Coherence (Failure) | 0.5503 |
| Gradient Magnitude (Success) | 0.2692 |
| Gradient Magnitude (Failure) | 0.1737 |
| Activation Separation | 0.3745 |
| Cosine Distance | 0.0061 |
| Clusters | 1,967 |
| Noise Fraction | 0.2679 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.4775 | 0.7962 | 0.5503 | 0.2692 | 0.1737 |
| G_IS (β=0.789) | 0.3243 | 0.7538 | 0.5268 | 0.1221 | 0.0826 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9729 | 0.9047 |
| cos(G_IS, G_reward)  | 0.4451 | 0.1724 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0039 | 0.0099 |
| cos(G_IS, Δθ)      | 0.0054 | 0.0121 |
| cos(G_reward, Δθ)  | 0.0012 | 0.0035 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,878 | 5.4299 | 0.9581 |
| Neutral  (r = 0) | 57,546 | 0.3820 | 0.9714 |
| Negative (r < 0) | 1,560 | 3.5441 | 0.9590 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6771 |
| Pos vs Negative  | -0.1035 |
| Neutral vs Neg.  | -0.1151 |
| Pos vs Failure   | 0.1708 |
| Neutral vs Fail. | 0.2942 |
| Neg. vs Failure  | 0.2552 |

---

## collect_drink_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5616 |
| Coherence (Success) | 0.8031 |
| Coherence (Failure) | 0.5811 |
| Gradient Magnitude (Success) | 0.2972 |
| Gradient Magnitude (Failure) | 0.1746 |
| Activation Separation | 0.3520 |
| Cosine Distance | 0.0054 |
| Clusters | 2,022 |
| Noise Fraction | 0.2754 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5616 | 0.8031 | 0.5811 | 0.2972 | 0.1746 |
| G_IS (β=0.789) | 0.4082 | 0.7369 | 0.5419 | 0.1337 | 0.0755 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9814 | 0.8969 |
| cos(G_IS, G_reward)  | 0.5263 | 0.1506 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,900 | 5.4558 | 0.9599 |
| Neutral  (r = 0) | 58,767 | 0.3850 | 0.9821 |
| Negative (r < 0) | 1,543 | 3.4387 | 0.9547 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6479 |
| Pos vs Negative  | -0.1235 |
| Neutral vs Neg.  | -0.1719 |
| Pos vs Failure   | 0.2372 |
| Neutral vs Fail. | 0.2806 |
| Neg. vs Failure  | 0.1802 |

---

## collect_sapling_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6134 |
| Coherence (Success) | 0.7666 |
| Coherence (Failure) | 0.5847 |
| Gradient Magnitude (Success) | 0.2757 |
| Gradient Magnitude (Failure) | 0.1684 |
| Activation Separation | 0.3793 |
| Cosine Distance | 0.0064 |
| Clusters | 1,936 |
| Noise Fraction | 0.2452 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6134 | 0.7666 | 0.5847 | 0.2757 | 0.1684 |
| G_IS (β=0.789) | 0.5139 | 0.7002 | 0.5537 | 0.1242 | 0.0751 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9755 | 0.9017 |
| cos(G_IS, G_reward)  | 0.3926 | 0.0231 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,851 | 5.5208 | 0.9529 |
| Neutral  (r = 0) | 56,567 | 0.3950 | 0.9767 |
| Negative (r < 0) | 1,517 | 8.3379 | 0.6318 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6889 |
| Pos vs Negative  | 0.2165 |
| Neutral vs Neg.  | -0.1713 |
| Pos vs Failure   | 0.1391 |
| Neutral vs Fail. | 0.3628 |
| Neg. vs Failure  | 0.3227 |

---

## collect_wood_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4604 |
| Coherence (Success) | 0.8041 |
| Coherence (Failure) | 0.5961 |
| Gradient Magnitude (Success) | 0.2909 |
| Gradient Magnitude (Failure) | 0.1667 |
| Activation Separation | 0.3639 |
| Cosine Distance | 0.0058 |
| Clusters | 1,999 |
| Noise Fraction | 0.2613 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.4604 | 0.8041 | 0.5961 | 0.2909 | 0.1667 |
| G_IS (β=0.789) | 0.3375 | 0.7653 | 0.5691 | 0.1337 | 0.0727 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9789 | 0.8992 |
| cos(G_IS, G_reward)  | 0.4534 | 0.0846 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,907 | 5.4122 | 0.9707 |
| Neutral  (r = 0) | 59,649 | 0.3860 | 0.9795 |
| Negative (r < 0) | 1,571 | 3.4264 | 0.9549 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6530 |
| Pos vs Negative  | -0.1105 |
| Neutral vs Neg.  | -0.1554 |
| Pos vs Failure   | 0.2129 |
| Neutral vs Fail. | 0.2404 |
| Neg. vs Failure  | 0.1904 |

---

## defeat_zombie_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7927 |
| Coherence (Success) | 0.7808 |
| Coherence (Failure) | 0.6019 |
| Gradient Magnitude (Success) | 0.2782 |
| Gradient Magnitude (Failure) | 0.1803 |
| Activation Separation | 0.3587 |
| Cosine Distance | 0.0056 |
| Clusters | 2,020 |
| Noise Fraction | 0.2593 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7927 | 0.7808 | 0.6019 | 0.2782 | 0.1803 |
| G_IS (β=0.789) | 0.7273 | 0.7209 | 0.5614 | 0.1277 | 0.0787 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9762 | 0.9170 |
| cos(G_IS, G_reward)  | 0.4658 | 0.0139 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,891 | 5.5407 | 0.9624 |
| Neutral  (r = 0) | 57,281 | 0.3799 | 0.9780 |
| Negative (r < 0) | 1,529 | 3.5319 | 0.9586 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6776 |
| Pos vs Negative  | -0.1022 |
| Neutral vs Neg.  | -0.1245 |
| Pos vs Failure   | 0.0603 |
| Neutral vs Fail. | 0.4561 |
| Neg. vs Failure  | 0.0771 |

---

## eat_cow_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6677 |
| Coherence (Success) | 0.7975 |
| Coherence (Failure) | 0.5718 |
| Gradient Magnitude (Success) | 0.2868 |
| Gradient Magnitude (Failure) | 0.1744 |
| Activation Separation | 0.3609 |
| Cosine Distance | 0.0057 |
| Clusters | 2,061 |
| Noise Fraction | 0.2573 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Drink, Eat Cow, Place Plant, Place Stone, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6677 | 0.7975 | 0.5718 | 0.2868 | 0.1744 |
| G_IS (β=0.789) | 0.5764 | 0.7418 | 0.5313 | 0.1289 | 0.0762 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9796 | 0.9080 |
| cos(G_IS, G_reward)  | 0.4584 | 0.0973 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,908 | 5.5586 | 0.9704 |
| Neutral  (r = 0) | 59,096 | 0.3899 | 0.9792 |
| Negative (r < 0) | 1,578 | 3.5461 | 0.9564 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6757 |
| Pos vs Negative  | -0.0985 |
| Neutral vs Neg.  | -0.1668 |
| Pos vs Failure   | 0.1642 |
| Neutral vs Fail. | 0.3612 |
| Neg. vs Failure  | 0.1431 |

---

## place_plant_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5093 |
| Coherence (Success) | 0.7924 |
| Coherence (Failure) | 0.5217 |
| Gradient Magnitude (Success) | 0.2818 |
| Gradient Magnitude (Failure) | 0.1671 |
| Activation Separation | 0.3792 |
| Cosine Distance | 0.0063 |
| Clusters | 2,000 |
| Noise Fraction | 0.2684 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5093 | 0.7924 | 0.5217 | 0.2818 | 0.1671 |
| G_IS (β=0.789) | 0.3819 | 0.7467 | 0.4911 | 0.1291 | 0.0719 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9779 | 0.9075 |
| cos(G_IS, G_reward)  | 0.3876 | 0.1720 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,889 | 5.4999 | 0.9628 |
| Neutral  (r = 0) | 58,073 | 0.3880 | 0.9797 |
| Negative (r < 0) | 1,545 | 8.3248 | 0.6229 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6773 |
| Pos vs Negative  | 0.2178 |
| Neutral vs Neg.  | -0.1997 |
| Pos vs Failure   | 0.2817 |
| Neutral vs Fail. | 0.1917 |
| Neg. vs Failure  | 0.3693 |

---

## place_table_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6077 |
| Coherence (Success) | 0.7588 |
| Coherence (Failure) | 0.5673 |
| Gradient Magnitude (Success) | 0.2826 |
| Gradient Magnitude (Failure) | 0.1668 |
| Activation Separation | 0.3927 |
| Cosine Distance | 0.0068 |
| Clusters | 1,981 |
| Noise Fraction | 0.2484 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6077 | 0.7588 | 0.5673 | 0.2826 | 0.1668 |
| G_IS (β=0.789) | 0.5149 | 0.6931 | 0.5333 | 0.1276 | 0.0737 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9774 | 0.9055 |
| cos(G_IS, G_reward)  | 0.4597 | 0.0736 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,896 | 5.4718 | 0.9610 |
| Neutral  (r = 0) | 57,875 | 0.3896 | 0.9774 |
| Negative (r < 0) | 1,535 | 3.6088 | 0.9539 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6848 |
| Pos vs Negative  | -0.0889 |
| Neutral vs Neg.  | -0.1563 |
| Pos vs Failure   | 0.1650 |
| Neutral vs Fail. | 0.3267 |
| Neg. vs Failure  | 0.1791 |

---

## wake_up_ep9002_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5170 |
| Coherence (Success) | 0.7783 |
| Coherence (Failure) | 0.5668 |
| Gradient Magnitude (Success) | 0.3249 |
| Gradient Magnitude (Failure) | 0.1661 |
| Activation Separation | 0.3561 |
| Cosine Distance | 0.0056 |
| Clusters | 1,947 |
| Noise Fraction | 0.2668 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5170 | 0.7783 | 0.5668 | 0.3249 | 0.1661 |
| G_IS (β=0.789) | 0.4234 | 0.7165 | 0.5498 | 0.1491 | 0.0743 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9833 | 0.8984 |
| cos(G_IS, G_reward)  | 0.5034 | 0.0972 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,875 | 5.3645 | 0.9572 |
| Neutral  (r = 0) | 57,401 | 0.3992 | 0.9794 |
| Negative (r < 0) | 1,531 | 3.4572 | 0.9529 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6442 |
| Pos vs Negative  | -0.1360 |
| Neutral vs Neg.  | -0.1390 |
| Pos vs Failure   | 0.2196 |
| Neutral vs Fail. | 0.2989 |
| Neg. vs Failure  | 0.1719 |

---

## make_wood_pickaxe_ep9004_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1929 |
| Coherence (Success) | 0.7603 |
| Coherence (Failure) | 0.5595 |
| Gradient Magnitude (Success) | 0.2754 |
| Gradient Magnitude (Failure) | 0.1733 |
| Activation Separation | 0.3824 |
| Cosine Distance | 0.0065 |
| Clusters | 2,006 |
| Noise Fraction | 0.2611 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.1929 | 0.7603 | 0.5595 | 0.2754 | 0.1733 |
| G_IS (β=0.789) | 0.0491 | 0.6896 | 0.5365 | 0.1226 | 0.0760 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9778 | 0.9151 |
| cos(G_IS, G_reward)  | 0.4709 | 0.2736 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,891 | 5.5636 | 0.9575 |
| Neutral  (r = 0) | 59,258 | 0.3813 | 0.9760 |
| Negative (r < 0) | 1,546 | 3.4809 | 0.9574 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6732 |
| Pos vs Negative  | -0.0897 |
| Neutral vs Neg.  | -0.1726 |
| Pos vs Failure   | 0.2965 |
| Neutral vs Fail. | 0.0625 |
| Neg. vs Failure  | 0.2888 |

---

## defeat_skeleton_ep9017_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5164 |
| Coherence (Success) | 0.7789 |
| Coherence (Failure) | 0.5595 |
| Gradient Magnitude (Success) | 0.2834 |
| Gradient Magnitude (Failure) | 0.1715 |
| Activation Separation | 0.3538 |
| Cosine Distance | 0.0055 |
| Clusters | 1,976 |
| Noise Fraction | 0.2782 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5164 | 0.7789 | 0.5595 | 0.2834 | 0.1715 |
| G_IS (β=0.789) | 0.3309 | 0.7112 | 0.5368 | 0.1305 | 0.0810 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9775 | 0.8982 |
| cos(G_IS, G_reward)  | 0.4544 | 0.1966 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,900 | 5.4359 | 0.9580 |
| Neutral  (r = 0) | 58,333 | 0.3850 | 0.9744 |
| Negative (r < 0) | 1,560 | 3.5640 | 0.9556 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6563 |
| Pos vs Negative  | -0.1199 |
| Neutral vs Neg.  | -0.1432 |
| Pos vs Failure   | 0.2489 |
| Neutral vs Fail. | 0.2535 |
| Neg. vs Failure  | 0.2225 |

---

## collect_stone_ep9058_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8392 |
| Coherence (Success) | 0.7781 |
| Coherence (Failure) | 0.6002 |
| Gradient Magnitude (Success) | 0.2690 |
| Gradient Magnitude (Failure) | 0.1942 |
| Activation Separation | 0.3391 |
| Cosine Distance | 0.0051 |
| Clusters | 2,042 |
| Noise Fraction | 0.2677 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8392 | 0.7781 | 0.6002 | 0.2690 | 0.1942 |
| G_IS (β=0.791) | 0.7813 | 0.7310 | 0.5672 | 0.1211 | 0.0850 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9745 | 0.9290 |
| cos(G_IS, G_reward)  | 0.4415 | 0.0219 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,925 | 5.3584 | 0.9603 |
| Neutral  (r = 0) | 59,514 | 0.3849 | 0.9770 |
| Negative (r < 0) | 1,557 | 3.6249 | 0.9595 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6693 |
| Pos vs Negative  | -0.0955 |
| Neutral vs Neg.  | -0.1561 |
| Pos vs Failure   | 0.0711 |
| Neutral vs Fail. | 0.4629 |
| Neg. vs Failure  | 0.0671 |

---

## make_wood_sword_ep9214_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8084 |
| Coherence (Success) | 0.8691 |
| Coherence (Failure) | 0.7765 |
| Gradient Magnitude (Success) | 0.2805 |
| Gradient Magnitude (Failure) | 0.3849 |
| Activation Separation | 0.4303 |
| Cosine Distance | 0.0075 |
| Clusters | 1,972 |
| Noise Fraction | 0.2651 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8084 | 0.8691 | 0.7765 | 0.2805 | 0.3849 |
| G_IS (β=0.799) | 0.8074 | 0.8332 | 0.7598 | 0.1286 | 0.1909 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9714 | 0.9796 |
| cos(G_IS, G_reward)  | 0.4622 | 0.5850 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,840 | 6.4608 | 0.9769 |
| Neutral  (r = 0) | 55,253 | 0.4196 | 0.9633 |
| Negative (r < 0) | 1,545 | 3.4317 | 0.9397 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3830 |
| Pos vs Negative  | -0.0823 |
| Neutral vs Neg.  | 0.0560 |
| Pos vs Failure   | 0.4951 |
| Neutral vs Fail. | 0.4348 |
| Neg. vs Failure  | 0.2933 |

---

## ep9215_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8953 |
| Coherence (Success) | 0.9095 |
| Coherence (Failure) | 0.6900 |
| Gradient Magnitude (Success) | 0.3674 |
| Gradient Magnitude (Failure) | 0.2865 |
| Activation Separation | 0.4386 |
| Cosine Distance | 0.0082 |
| Clusters | 2,230 |
| Noise Fraction | 0.2517 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (12) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8953 | 0.9095 | 0.6900 | 0.3674 | 0.2865 |
| G_IS (β=0.799) | 0.8287 | 0.8776 | 0.6334 | 0.1521 | 0.1167 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9933 | 0.9767 |
| cos(G_IS, G_reward)  | 0.9314 | 0.8592 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0114 | 0.0149 |
| cos(G_IS, Δθ)      | 0.0118 | 0.0157 |
| cos(G_reward, Δθ)  | 0.0104 | 0.0115 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,759 | 7.6654 | 0.9773 |
| Neutral  (r = 0) | 56,241 | 0.2594 | 0.9678 |
| Negative (r < 0) | 1,492 | 3.5114 | 0.9365 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6667 |
| Pos vs Negative  | -0.1849 |
| Neutral vs Neg.  | -0.0585 |
| Pos vs Failure   | 0.8878 |
| Neutral vs Fail. | -0.5399 |
| Neg. vs Failure  | -0.0024 |

---

## collect_coal_ep9224_lower7.000_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9600 |
| Coherence (Success) | 0.9116 |
| Coherence (Failure) | 0.8571 |
| Gradient Magnitude (Success) | 0.4597 |
| Gradient Magnitude (Failure) | 0.6248 |
| Activation Separation | 0.4472 |
| Cosine Distance | 0.0080 |
| Clusters | 2,036 |
| Noise Fraction | 0.2577 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9600 | 0.9116 | 0.8571 | 0.4597 | 0.6248 |
| G_IS (β=0.799) | 0.9555 | 0.8786 | 0.8248 | 0.2101 | 0.2893 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9907 | 0.9953 |
| cos(G_IS, G_reward)  | 0.8081 | 0.7203 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,744 | 6.1438 | 0.9705 |
| Neutral  (r = 0) | 55,486 | 0.3465 | 0.9668 |
| Negative (r < 0) | 1,483 | 3.4396 | 0.9377 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4513 |
| Pos vs Negative  | -0.1745 |
| Neutral vs Neg.  | -0.3454 |
| Pos vs Failure   | 0.3683 |
| Neutral vs Fail. | 0.4383 |
| Neg. vs Failure  | -0.3253 |

---

## ep9420_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9189 |
| Coherence (Success) | 0.8830 |
| Coherence (Failure) | 0.7477 |
| Gradient Magnitude (Success) | 0.4927 |
| Gradient Magnitude (Failure) | 0.4338 |
| Activation Separation | 0.3601 |
| Cosine Distance | 0.0057 |
| Clusters | 2,195 |
| Noise Fraction | 0.2400 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9189 | 0.8830 | 0.7477 | 0.4927 | 0.4338 |
| G_IS (β=0.809) | 0.8962 | 0.8678 | 0.7347 | 0.2206 | 0.2008 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9920 | 0.9812 |
| cos(G_IS, G_reward)  | 0.7311 | 0.6006 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0035 | 0.0020 |
| cos(G_IS, Δθ)      | 0.0044 | 0.0027 |
| cos(G_reward, Δθ)  | 0.0028 | 0.0026 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,927 | 6.3899 | 0.9684 |
| Neutral  (r = 0) | 60,325 | 0.3338 | 0.9735 |
| Negative (r < 0) | 1,599 | 3.7705 | 0.9518 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3657 |
| Pos vs Negative  | -0.1748 |
| Neutral vs Neg.  | 0.1390 |
| Pos vs Failure   | 0.4215 |
| Neutral vs Fail. | 0.4246 |
| Neg. vs Failure  | 0.2003 |

---

## ep9642_lower7.900_upper8.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9319 |
| Coherence (Success) | 0.9143 |
| Coherence (Failure) | 0.7303 |
| Gradient Magnitude (Success) | 0.6065 |
| Gradient Magnitude (Failure) | 0.4632 |
| Activation Separation | 0.3498 |
| Cosine Distance | 0.0050 |
| Clusters | 1,841 |
| Noise Fraction | 0.2736 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9319 | 0.9143 | 0.7303 | 0.6065 | 0.4632 |
| G_IS (β=0.819) | 0.9116 | 0.8898 | 0.6753 | 0.2741 | 0.2054 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9958 | 0.9854 |
| cos(G_IS, G_reward)  | 0.7016 | 0.6261 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0026 | 0.0038 |
| cos(G_IS, Δθ)      | 0.0031 | 0.0040 |
| cos(G_reward, Δθ)  | -0.0001 | 0.0034 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,935 | 6.4486 | 0.9772 |
| Neutral  (r = 0) | 55,669 | 0.5133 | 0.9708 |
| Negative (r < 0) | 1,469 | 3.7219 | 0.9469 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.1156 |
| Pos vs Negative  | -0.1895 |
| Neutral vs Neg.  | -0.0361 |
| Pos vs Failure   | 0.6807 |
| Neutral vs Fail. | 0.6858 |
| Neg. vs Failure  | 0.1215 |

---

## ep9870_lower7.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6685 |
| Coherence (Success) | 0.8860 |
| Coherence (Failure) | 0.7045 |
| Gradient Magnitude (Success) | 0.2713 |
| Gradient Magnitude (Failure) | 0.2765 |
| Activation Separation | 0.3652 |
| Cosine Distance | 0.0054 |
| Clusters | 1,819 |
| Noise Fraction | 0.2706 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6685 | 0.8860 | 0.7045 | 0.2713 | 0.2765 |
| G_IS (β=0.829) | 0.5660 | 0.8505 | 0.6706 | 0.1092 | 0.1237 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9812 | 0.9590 |
| cos(G_IS, G_reward)  | 0.7208 | 0.6921 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0012 | 0.0031 |
| cos(G_IS, Δθ)      | 0.0000 | 0.0051 |
| cos(G_reward, Δθ)  | -0.0015 | 0.0038 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,962 | 5.9277 | 0.9780 |
| Neutral  (r = 0) | 57,022 | 0.2784 | 0.9530 |
| Negative (r < 0) | 1,530 | 3.7108 | 0.9437 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3236 |
| Pos vs Negative  | -0.1471 |
| Neutral vs Neg.  | 0.0662 |
| Pos vs Failure   | 0.5943 |
| Neutral vs Fail. | 0.2095 |
| Neg. vs Failure  | 0.3136 |

---

## make_stone_sword_ep9875_lower7.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9227 |
| Coherence (Success) | 0.9251 |
| Coherence (Failure) | 0.7778 |
| Gradient Magnitude (Success) | 0.4301 |
| Gradient Magnitude (Failure) | 0.3005 |
| Activation Separation | 0.3979 |
| Cosine Distance | 0.0064 |
| Clusters | 1,966 |
| Noise Fraction | 0.2744 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9227 | 0.9251 | 0.7778 | 0.4301 | 0.3005 |
| G_IS (β=0.829) | 0.8627 | 0.9044 | 0.7200 | 0.1859 | 0.1267 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9964 | 0.9766 |
| cos(G_IS, G_reward)  | 0.6841 | 0.5535 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,999 | 5.7570 | 0.9736 |
| Neutral  (r = 0) | 57,709 | 0.3244 | 0.9600 |
| Negative (r < 0) | 1,503 | 6.8152 | 0.6981 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.0808 |
| Pos vs Negative  | 0.2064 |
| Neutral vs Neg.  | 0.0336 |
| Pos vs Failure   | 0.6132 |
| Neutral vs Fail. | 0.5542 |
| Neg. vs Failure  | 0.3315 |

---

## ep10087_lower7.900_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7583 |
| Coherence (Success) | 0.8755 |
| Coherence (Failure) | 0.7589 |
| Gradient Magnitude (Success) | 0.2309 |
| Gradient Magnitude (Failure) | 0.2270 |
| Activation Separation | 0.3957 |
| Cosine Distance | 0.0068 |
| Clusters | 1,913 |
| Noise Fraction | 0.2862 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7583 | 0.8755 | 0.7589 | 0.2309 | 0.2270 |
| G_IS (β=0.839) | 0.6575 | 0.8418 | 0.7031 | 0.0889 | 0.0955 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9762 | 0.9477 |
| cos(G_IS, G_reward)  | 0.7659 | 0.4591 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0007 | -0.0012 |
| cos(G_IS, Δθ)      | -0.0005 | -0.0005 |
| cos(G_reward, Δθ)  | -0.0023 | -0.0046 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,038 | 5.6849 | 0.9778 |
| Neutral  (r = 0) | 57,620 | 0.2500 | 0.9644 |
| Negative (r < 0) | 1,545 | 3.4971 | 0.9479 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4995 |
| Pos vs Negative  | -0.1071 |
| Neutral vs Neg.  | -0.0514 |
| Pos vs Failure   | 0.7427 |
| Neutral vs Fail. | -0.1493 |
| Neg. vs Failure  | 0.2994 |

---

## ep10316_lower8.000_upper9.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8094 |
| Coherence (Success) | 0.8800 |
| Coherence (Failure) | 0.6944 |
| Gradient Magnitude (Success) | 0.3006 |
| Gradient Magnitude (Failure) | 0.3151 |
| Activation Separation | 0.4223 |
| Cosine Distance | 0.0075 |
| Clusters | 1,953 |
| Noise Fraction | 0.2691 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8094 | 0.8800 | 0.6944 | 0.3006 | 0.3151 |
| G_IS (β=0.849) | 0.7405 | 0.8480 | 0.6450 | 0.1251 | 0.1340 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9913 | 0.9760 |
| cos(G_IS, G_reward)  | 0.7608 | 0.5612 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0028 | 0.0057 |
| cos(G_IS, Δθ)      | 0.0041 | 0.0066 |
| cos(G_reward, Δθ)  | 0.0040 | 0.0106 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,024 | 5.5488 | 0.9705 |
| Neutral  (r = 0) | 57,776 | 0.2812 | 0.9608 |
| Negative (r < 0) | 1,527 | 3.5322 | 0.9474 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1869 |
| Pos vs Negative  | -0.1684 |
| Neutral vs Neg.  | -0.0786 |
| Pos vs Failure   | 0.6775 |
| Neutral vs Fail. | 0.3247 |
| Neg. vs Failure  | 0.1778 |

---

## ep10541_lower8.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9590 |
| Coherence (Success) | 0.8752 |
| Coherence (Failure) | 0.7783 |
| Gradient Magnitude (Success) | 0.4732 |
| Gradient Magnitude (Failure) | 0.4509 |
| Activation Separation | 0.3864 |
| Cosine Distance | 0.0065 |
| Clusters | 1,819 |
| Noise Fraction | 0.2492 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Eat Plant, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9590 | 0.8752 | 0.7783 | 0.4732 | 0.4509 |
| G_IS (β=0.859) | 0.9482 | 0.8340 | 0.7510 | 0.2060 | 0.1961 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9923 | 0.9889 |
| cos(G_IS, G_reward)  | 0.8112 | 0.7296 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0024 | 0.0077 |
| cos(G_IS, Δθ)      | 0.0043 | 0.0099 |
| cos(G_reward, Δθ)  | -0.0049 | -0.0005 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 2,961 | 5.2812 | 0.9742 |
| Neutral  (r = 0) | 55,945 | 0.2768 | 0.9603 |
| Negative (r < 0) | 1,490 | 3.8083 | 0.9395 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.4124 |
| Pos vs Negative  | -0.2480 |
| Neutral vs Neg.  | -0.0248 |
| Pos vs Failure   | 0.5462 |
| Neutral vs Fail. | 0.3110 |
| Neg. vs Failure  | -0.0222 |

---

## ep10763_lower8.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9681 |
| Coherence (Success) | 0.9256 |
| Coherence (Failure) | 0.8244 |
| Gradient Magnitude (Success) | 0.4988 |
| Gradient Magnitude (Failure) | 0.4017 |
| Activation Separation | 0.4754 |
| Cosine Distance | 0.0095 |
| Clusters | 1,975 |
| Noise Fraction | 0.2536 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9681 | 0.9256 | 0.8244 | 0.4988 | 0.4017 |
| G_IS (β=0.869) | 0.9587 | 0.9008 | 0.7689 | 0.2045 | 0.1593 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9968 | 0.9917 |
| cos(G_IS, G_reward)  | 0.9273 | 0.7994 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0069 | 0.0062 |
| cos(G_IS, Δθ)      | 0.0065 | 0.0055 |
| cos(G_reward, Δθ)  | 0.0064 | 0.0037 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,059 | 6.3401 | 0.9736 |
| Neutral  (r = 0) | 57,897 | 0.2280 | 0.9521 |
| Negative (r < 0) | 1,424 | 3.8022 | 0.9338 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.0994 |
| Pos vs Negative  | -0.2447 |
| Neutral vs Neg.  | -0.3640 |
| Pos vs Failure   | 0.8413 |
| Neutral vs Fail. | 0.3493 |
| Neg. vs Failure  | -0.1960 |

---

## ep10998_lower8.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7569 |
| Coherence (Success) | 0.8135 |
| Coherence (Failure) | 0.6993 |
| Gradient Magnitude (Success) | 0.1974 |
| Gradient Magnitude (Failure) | 0.2109 |
| Activation Separation | 0.4698 |
| Cosine Distance | 0.0103 |
| Clusters | 1,806 |
| Noise Fraction | 0.2709 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (13) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7569 | 0.8135 | 0.6993 | 0.1974 | 0.2109 |
| G_IS (β=0.879) | 0.6615 | 0.7521 | 0.6460 | 0.0749 | 0.0863 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9712 | 0.9309 |
| cos(G_IS, G_reward)  | 0.7876 | 0.6513 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0072 | -0.0018 |
| cos(G_IS, Δθ)      | -0.0044 | 0.0025 |
| cos(G_reward, Δθ)  | -0.0066 | -0.0016 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,145 | 5.0612 | 0.9706 |
| Neutral  (r = 0) | 57,679 | 0.2517 | 0.9629 |
| Negative (r < 0) | 1,499 | 3.5629 | 0.9295 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6006 |
| Pos vs Negative  | -0.1878 |
| Neutral vs Neg.  | 0.0064 |
| Pos vs Failure   | 0.6156 |
| Neutral vs Fail. | -0.1288 |
| Neg. vs Failure  | 0.2414 |

---

## place_furnace_ep11209_lower8.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9264 |
| Coherence (Success) | 0.9143 |
| Coherence (Failure) | 0.6859 |
| Gradient Magnitude (Success) | 0.4015 |
| Gradient Magnitude (Failure) | 0.3027 |
| Activation Separation | 0.4332 |
| Cosine Distance | 0.0084 |
| Clusters | 1,820 |
| Noise Fraction | 0.2703 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Furnace, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9264 | 0.9143 | 0.6859 | 0.4015 | 0.3027 |
| G_IS (β=0.889) | 0.8967 | 0.8924 | 0.5944 | 0.1677 | 0.1180 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9924 | 0.9787 |
| cos(G_IS, G_reward)  | 0.8181 | 0.7415 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0113 | 0.0140 |
| cos(G_IS, Δθ)      | 0.0108 | 0.0139 |
| cos(G_reward, Δθ)  | 0.0122 | 0.0137 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,163 | 5.6142 | 0.9692 |
| Neutral  (r = 0) | 57,778 | 0.2590 | 0.9544 |
| Negative (r < 0) | 1,488 | 3.6265 | 0.9398 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5074 |
| Pos vs Negative  | -0.2129 |
| Neutral vs Neg.  | -0.2027 |
| Pos vs Failure   | 0.6337 |
| Neutral vs Fail. | 0.0973 |
| Neg. vs Failure  | -0.0371 |

---

## ep11220_lower8.000_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7687 |
| Coherence (Success) | 0.8755 |
| Coherence (Failure) | 0.7130 |
| Gradient Magnitude (Success) | 0.2275 |
| Gradient Magnitude (Failure) | 0.3138 |
| Activation Separation | 0.4751 |
| Cosine Distance | 0.0104 |
| Clusters | 1,742 |
| Noise Fraction | 0.2678 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Furnace, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7687 | 0.8755 | 0.7130 | 0.2275 | 0.3138 |
| G_IS (β=0.889) | 0.7600 | 0.8429 | 0.7067 | 0.0920 | 0.1403 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9680 | 0.9742 |
| cos(G_IS, G_reward)  | 0.5682 | 0.4773 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0089 | 0.0067 |
| cos(G_IS, Δθ)      | 0.0089 | 0.0058 |
| cos(G_reward, Δθ)  | 0.0084 | 0.0069 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,156 | 5.1904 | 0.9659 |
| Neutral  (r = 0) | 57,235 | 0.3292 | 0.9681 |
| Negative (r < 0) | 1,472 | 3.6322 | 0.9392 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5587 |
| Pos vs Negative  | -0.1685 |
| Neutral vs Neg.  | 0.0805 |
| Pos vs Failure   | 0.3226 |
| Neutral vs Fail. | 0.3973 |
| Neg. vs Failure  | 0.3447 |

---

## ep11443_lower8.900_upper9.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9166 |
| Coherence (Success) | 0.8953 |
| Coherence (Failure) | 0.6908 |
| Gradient Magnitude (Success) | 0.4298 |
| Gradient Magnitude (Failure) | 0.3637 |
| Activation Separation | 0.5487 |
| Cosine Distance | 0.0144 |
| Clusters | 2,005 |
| Noise Fraction | 0.2649 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9166 | 0.8953 | 0.6908 | 0.4298 | 0.3637 |
| G_IS (β=0.899) | 0.8900 | 0.8586 | 0.6207 | 0.1786 | 0.1490 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9924 | 0.9824 |
| cos(G_IS, G_reward)  | 0.8414 | 0.6373 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0050 | 0.0054 |
| cos(G_IS, Δθ)      | 0.0059 | 0.0060 |
| cos(G_reward, Δθ)  | 0.0052 | 0.0072 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,285 | 5.2961 | 0.9566 |
| Neutral  (r = 0) | 60,711 | 0.2391 | 0.9533 |
| Negative (r < 0) | 1,573 | 3.1128 | 0.9299 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1798 |
| Pos vs Negative  | -0.2715 |
| Neutral vs Neg.  | -0.3898 |
| Pos vs Failure   | 0.6460 |
| Neutral vs Fail. | 0.2884 |
| Neg. vs Failure  | -0.1664 |

---

## ep11669_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.4175 |
| Coherence (Success) | 0.8613 |
| Coherence (Failure) | 0.5963 |
| Gradient Magnitude (Success) | 0.2464 |
| Gradient Magnitude (Failure) | 0.1870 |
| Activation Separation | 0.6915 |
| Cosine Distance | 0.0205 |
| Clusters | 2,265 |
| Noise Fraction | 0.2611 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Place Plant, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.4175 | 0.8613 | 0.5963 | 0.2464 | 0.1870 |
| G_IS (β=0.909) | 0.2101 | 0.8171 | 0.5719 | 0.0935 | 0.0828 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9756 | 0.9350 |
| cos(G_IS, G_reward)  | 0.7966 | 0.5004 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0007 | 0.0091 |
| cos(G_IS, Δθ)      | 0.0021 | 0.0100 |
| cos(G_reward, Δθ)  | -0.0059 | -0.0014 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,382 | 4.9935 | 0.9779 |
| Neutral  (r = 0) | 60,630 | 0.2103 | 0.9516 |
| Negative (r < 0) | 1,587 | 3.1887 | 0.9410 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6306 |
| Pos vs Negative  | -0.2899 |
| Neutral vs Neg.  | -0.1089 |
| Pos vs Failure   | 0.4215 |
| Neutral vs Fail. | -0.0947 |
| Neg. vs Failure  | 0.2134 |

---

## ep11881_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8471 |
| Coherence (Success) | 0.8891 |
| Coherence (Failure) | 0.6405 |
| Gradient Magnitude (Success) | 0.3049 |
| Gradient Magnitude (Failure) | 0.2171 |
| Activation Separation | 0.6426 |
| Cosine Distance | 0.0192 |
| Clusters | 2,009 |
| Noise Fraction | 0.2619 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (14) | Coal, Drink, Eat Cow, Furnace, Place Plant, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8471 | 0.8891 | 0.6405 | 0.3049 | 0.2171 |
| G_IS (β=0.919) | 0.7773 | 0.8579 | 0.5924 | 0.1182 | 0.0871 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9899 | 0.9592 |
| cos(G_IS, G_reward)  | 0.8634 | 0.7232 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0018 | 0.0047 |
| cos(G_IS, Δθ)      | 0.0016 | 0.0046 |
| cos(G_reward, Δθ)  | 0.0065 | 0.0079 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,403 | 4.7009 | 0.9637 |
| Neutral  (r = 0) | 60,749 | 0.2474 | 0.9672 |
| Negative (r < 0) | 1,607 | 3.0557 | 0.9356 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5415 |
| Pos vs Negative  | -0.3206 |
| Neutral vs Neg.  | -0.0611 |
| Pos vs Failure   | 0.5343 |
| Neutral vs Fail. | -0.0028 |
| Neg. vs Failure  | -0.0117 |

---

## ep12087_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7548 |
| Coherence (Success) | 0.8593 |
| Coherence (Failure) | 0.7476 |
| Gradient Magnitude (Success) | 0.2588 |
| Gradient Magnitude (Failure) | 0.3188 |
| Activation Separation | 0.8807 |
| Cosine Distance | 0.0345 |
| Clusters | 2,314 |
| Noise Fraction | 0.2494 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (15) | Coal, Drink, Eat Cow, Furnace, Place Plant, Sapling, Skeleton, Stone, Stone Pickaxe, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7548 | 0.8593 | 0.7476 | 0.2588 | 0.3188 |
| G_IS (β=0.930) | 0.7248 | 0.8245 | 0.7449 | 0.0994 | 0.1378 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9745 | 0.9734 |
| cos(G_IS, G_reward)  | 0.5610 | 0.2577 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0033 | -0.0036 |
| cos(G_IS, Δθ)      | -0.0038 | -0.0040 |
| cos(G_reward, Δθ)  | -0.0038 | -0.0026 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,415 | 5.1613 | 0.9692 |
| Neutral  (r = 0) | 61,799 | 0.3933 | 0.9710 |
| Negative (r < 0) | 1,631 | 7.1416 | 0.5852 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6088 |
| Pos vs Negative  | 0.2357 |
| Neutral vs Neg.  | -0.1666 |
| Pos vs Failure   | 0.1011 |
| Neutral vs Fail. | 0.5630 |
| Neg. vs Failure  | 0.1015 |

---

## place_stone_ep12088_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6899 |
| Coherence (Success) | 0.9047 |
| Coherence (Failure) | 0.7211 |
| Gradient Magnitude (Success) | 0.2819 |
| Gradient Magnitude (Failure) | 0.2557 |
| Activation Separation | 0.7875 |
| Cosine Distance | 0.0286 |
| Clusters | 2,197 |
| Noise Fraction | 0.2465 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Pickaxe, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6899 | 0.9047 | 0.7211 | 0.2819 | 0.2557 |
| G_IS (β=0.930) | 0.5303 | 0.8746 | 0.6666 | 0.0976 | 0.0951 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9886 | 0.9547 |
| cos(G_IS, G_reward)  | 0.9274 | 0.7269 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,411 | 5.5981 | 0.9742 |
| Neutral  (r = 0) | 60,704 | 0.2669 | 0.9633 |
| Negative (r < 0) | 1,615 | 2.8408 | 0.9325 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5696 |
| Pos vs Negative  | -0.2545 |
| Neutral vs Neg.  | -0.1638 |
| Pos vs Failure   | 0.6735 |
| Neutral vs Fail. | -0.1252 |
| Neg. vs Failure  | 0.0673 |

---

## ep12284_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9255 |
| Coherence (Success) | 0.8995 |
| Coherence (Failure) | 0.8402 |
| Gradient Magnitude (Success) | 0.5379 |
| Gradient Magnitude (Failure) | 0.6274 |
| Activation Separation | 0.8897 |
| Cosine Distance | 0.0362 |
| Clusters | 2,158 |
| Noise Fraction | 0.2423 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (15) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9255 | 0.8995 | 0.8402 | 0.5379 | 0.6274 |
| G_IS (β=0.940) | 0.9127 | 0.8700 | 0.8205 | 0.2015 | 0.2353 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9939 | 0.9925 |
| cos(G_IS, G_reward)  | 0.7014 | 0.6871 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0042 | 0.0083 |
| cos(G_IS, Δθ)      | 0.0051 | 0.0095 |
| cos(G_reward, Δθ)  | 0.0018 | 0.0100 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,493 | 5.5184 | 0.9741 |
| Neutral  (r = 0) | 60,266 | 0.3783 | 0.9601 |
| Negative (r < 0) | 1,603 | 9.4130 | 0.4816 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.1707 |
| Pos vs Negative  | 0.2076 |
| Neutral vs Neg.  | 0.1683 |
| Pos vs Failure   | 0.6851 |
| Neutral vs Fail. | 0.7268 |
| Neg. vs Failure  | 0.2400 |

---

## make_stone_pickaxe_ep12409_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9846 |
| Coherence (Success) | 0.9351 |
| Coherence (Failure) | 0.8274 |
| Gradient Magnitude (Success) | 0.9435 |
| Gradient Magnitude (Failure) | 0.8200 |
| Activation Separation | 0.7891 |
| Cosine Distance | 0.0272 |
| Clusters | 2,215 |
| Noise Fraction | 0.2700 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9846 | 0.9351 | 0.8274 | 0.9435 | 0.8200 |
| G_IS (β=0.945) | 0.9827 | 0.9223 | 0.7934 | 0.3670 | 0.3142 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9985 | 0.9969 |
| cos(G_IS, G_reward)  | 0.9352 | 0.8997 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0059 | 0.0059 |
| cos(G_IS, Δθ)      | 0.0055 | 0.0053 |
| cos(G_reward, Δθ)  | 0.0069 | 0.0062 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,415 | 7.6677 | 0.9745 |
| Neutral  (r = 0) | 59,491 | 0.5573 | 0.9769 |
| Negative (r < 0) | 1,593 | 3.1527 | 0.9225 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | 0.5529 |
| Pos vs Negative  | -0.2132 |
| Neutral vs Neg.  | -0.1630 |
| Pos vs Failure   | 0.8456 |
| Neutral vs Fail. | 0.8549 |
| Neg. vs Failure  | -0.1116 |

---

## ep12503_lower8.900_upper10.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.5770 |
| Coherence (Success) | 0.8602 |
| Coherence (Failure) | 0.6185 |
| Gradient Magnitude (Success) | 0.2183 |
| Gradient Magnitude (Failure) | 0.1860 |
| Activation Separation | 0.7976 |
| Cosine Distance | 0.0273 |
| Clusters | 2,312 |
| Noise Fraction | 0.2523 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Eat Plant, Furnace, Place Plant, Sapling, Skeleton, Stone, Stone Pickaxe, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.5770 | 0.8602 | 0.6185 | 0.2183 | 0.1860 |
| G_IS (β=0.950) | 0.3149 | 0.8096 | 0.5734 | 0.0732 | 0.0698 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9877 | 0.9212 |
| cos(G_IS, G_reward)  | 0.7586 | 0.2376 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0039 | 0.0094 |
| cos(G_IS, Δθ)      | 0.0033 | 0.0081 |
| cos(G_reward, Δθ)  | 0.0032 | 0.0073 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,445 | 4.8677 | 0.9682 |
| Neutral  (r = 0) | 62,087 | 0.2749 | 0.9593 |
| Negative (r < 0) | 1,621 | 14.8907 | 0.3383 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.5742 |
| Pos vs Negative  | 0.3210 |
| Neutral vs Neg.  | 0.0255 |
| Pos vs Failure   | 0.4780 |
| Neutral vs Fail. | 0.1382 |
| Neg. vs Failure  | 0.4291 |

---

## ep12701_lower8.900_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6319 |
| Coherence (Success) | 0.8789 |
| Coherence (Failure) | 0.6332 |
| Gradient Magnitude (Success) | 0.2282 |
| Gradient Magnitude (Failure) | 0.2056 |
| Activation Separation | 0.9543 |
| Cosine Distance | 0.0372 |
| Clusters | 2,254 |
| Noise Fraction | 0.2333 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.6319 | 0.8789 | 0.6332 | 0.2282 | 0.2056 |
| G_IS (β=0.960) | 0.4474 | 0.8412 | 0.5925 | 0.0822 | 0.0840 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9837 | 0.9365 |
| cos(G_IS, G_reward)  | 0.8893 | 0.6393 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0005 | 0.0028 |
| cos(G_IS, Δθ)      | -0.0007 | 0.0030 |
| cos(G_reward, Δθ)  | -0.0015 | 0.0000 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,529 | 4.9982 | 0.9765 |
| Neutral  (r = 0) | 62,271 | 0.2497 | 0.9612 |
| Negative (r < 0) | 1,625 | 3.0973 | 0.9009 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.6111 |
| Pos vs Negative  | -0.2244 |
| Neutral vs Neg.  | -0.1416 |
| Pos vs Failure   | 0.5865 |
| Neutral vs Fail. | -0.1547 |
| Neg. vs Failure  | 0.1022 |

---

## ep12908_lower9.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9332 |
| Coherence (Success) | 0.8967 |
| Coherence (Failure) | 0.7187 |
| Gradient Magnitude (Success) | 0.4312 |
| Gradient Magnitude (Failure) | 0.4509 |
| Activation Separation | 1.1081 |
| Cosine Distance | 0.0547 |
| Clusters | 2,265 |
| Noise Fraction | 0.2486 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (17) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Pickaxe, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9332 | 0.8967 | 0.7187 | 0.4312 | 0.4509 |
| G_IS (β=0.970) | 0.9090 | 0.8515 | 0.6487 | 0.1707 | 0.1821 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9941 | 0.9857 |
| cos(G_IS, G_reward)  | 0.8848 | 0.7717 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | -0.0018 | 0.0043 |
| cos(G_IS, Δθ)      | -0.0011 | 0.0052 |
| cos(G_reward, Δθ)  | -0.0058 | 0.0035 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,477 | 5.5523 | 0.9751 |
| Neutral  (r = 0) | 62,359 | 0.2740 | 0.9608 |
| Negative (r < 0) | 1,578 | 2.9791 | 0.9091 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.3223 |
| Pos vs Negative  | -0.2639 |
| Neutral vs Neg.  | -0.2757 |
| Pos vs Failure   | 0.6401 |
| Neutral vs Fail. | 0.2363 |
| Neg. vs Failure  | -0.0975 |

---

## ep13118_lower9.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9520 |
| Coherence (Success) | 0.8754 |
| Coherence (Failure) | 0.7618 |
| Gradient Magnitude (Success) | 0.4309 |
| Gradient Magnitude (Failure) | 0.4474 |
| Activation Separation | 1.1821 |
| Cosine Distance | 0.0556 |
| Clusters | 2,325 |
| Noise Fraction | 0.2395 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.9520 | 0.8754 | 0.7618 | 0.4309 | 0.4474 |
| G_IS (β=0.980) | 0.9396 | 0.8404 | 0.7259 | 0.1640 | 0.1680 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9931 | 0.9878 |
| cos(G_IS, G_reward)  | 0.7770 | 0.7462 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0001 | 0.0042 |
| cos(G_IS, Δθ)      | 0.0003 | 0.0041 |
| cos(G_reward, Δθ)  | -0.0072 | 0.0012 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,602 | 4.8593 | 0.9739 |
| Neutral  (r = 0) | 61,276 | 0.4000 | 0.9798 |
| Negative (r < 0) | 1,637 | 3.1054 | 0.9227 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.2843 |
| Pos vs Negative  | -0.1687 |
| Neutral vs Neg.  | -0.1135 |
| Pos vs Failure   | 0.3924 |
| Neutral vs Fail. | 0.6344 |
| Neg. vs Failure  | 0.0491 |

---

## ep13326_lower9.000_upper10.900

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8130 |
| Coherence (Success) | 0.8810 |
| Coherence (Failure) | 0.7464 |
| Gradient Magnitude (Success) | 0.3115 |
| Gradient Magnitude (Failure) | 0.4263 |
| Activation Separation | 1.3181 |
| Cosine Distance | 0.0682 |
| Clusters | 2,327 |
| Noise Fraction | 0.2550 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (17) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Pickaxe, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.8130 | 0.8810 | 0.7464 | 0.3115 | 0.4263 |
| G_IS (β=0.990) | 0.7792 | 0.8506 | 0.7394 | 0.1212 | 0.1798 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9877 | 0.9817 |
| cos(G_IS, G_reward)  | 0.7184 | 0.6564 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0010 | 0.0072 |
| cos(G_IS, Δθ)      | 0.0022 | 0.0087 |
| cos(G_reward, Δθ)  | -0.0030 | 0.0028 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,644 | 5.0998 | 0.9802 |
| Neutral  (r = 0) | 65,790 | 0.2664 | 0.9587 |
| Negative (r < 0) | 1,734 | 7.3569 | 0.5667 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.1255 |
| Pos vs Negative  | 0.1324 |
| Neutral vs Neg.  | -0.0630 |
| Pos vs Failure   | 0.5609 |
| Neutral vs Fail. | 0.5044 |
| Neg. vs Failure  | 0.0903 |

---

## ep13532_lower9.000_upper11.000

### Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7021 |
| Coherence (Success) | 0.9048 |
| Coherence (Failure) | 0.6644 |
| Gradient Magnitude (Success) | 0.2734 |
| Gradient Magnitude (Failure) | 0.2073 |
| Activation Separation | 1.3695 |
| Cosine Distance | 0.0729 |
| Clusters | 2,305 |
| Noise Fraction | 0.2346 |
| RSA Alignment (ρ) | — |
| RSA Stimuli (16) | Coal, Drink, Eat Cow, Furnace, Place Plant, Place Stone, Sapling, Skeleton, Stone, Stone Sword, Table, Wake Up, Wood, Wood Pickaxe, Wood Sword, Zombie |

### Gradient Variant Analysis (RQ1 / RQ2)

| Variant | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) |
|---|---:|---:|---:|---:|---:|
| G_uniform | 0.7021 | 0.9048 | 0.6644 | 0.2734 | 0.2073 |
| G_IS (β=1.000) | 0.5197 | 0.8754 | 0.6274 | 0.0945 | 0.0759 |

| Directional Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, G_IS) | 0.9857 | 0.9341 |
| cos(G_IS, G_reward)  | 0.9318 | 0.7045 |

| Weight-Delta Alignment | Success | Failure |
|---|---:|---:|
| cos(G_uniform, Δθ) | 0.0001 | 0.0008 |
| cos(G_IS, Δθ)      | 0.0000 | 0.0006 |
| cos(G_reward, Δθ)  | -0.0023 | 0.0015 |

### Moment of Reward (Rainbow)

| Subgroup | N transitions | Grad Mag | Coherence |
|---|---:|---:|---:|
| Positive (r > 0) | 3,638 | 6.1729 | 0.9822 |
| Neutral  (r = 0) | 64,289 | 0.2455 | 0.9703 |
| Negative (r < 0) | 1,713 | 3.2200 | 0.9350 |

| Comparison | Opp. Score |
|---|---:|
| Pos vs Neutral   | -0.7036 |
| Pos vs Negative  | -0.2179 |
| Neutral vs Neg.  | -0.0929 |
| Pos vs Failure   | 0.7104 |
| Neutral vs Fail. | -0.3322 |
| Neg. vs Failure  | 0.0387 |

---

## Achievement Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wake_up @ ep183 | 183 | 0.9976 | 0.9217 | 0.9184 | 0.1038 | 0.0973 | 0.2546 | 0.0009 | — |
| collect_sapling @ ep282 | 282 | 0.9955 | 0.9004 | 0.9234 | 0.1621 | 0.1554 | 0.0427 | 0.0000 | — |
| collect_wood @ ep283 | 283 | 0.9978 | 0.8603 | 0.9313 | 0.0935 | 0.0890 | 0.1634 | 0.0002 | — |
| place_plant @ ep283 | 283 | 0.9961 | 0.8724 | 0.9069 | 0.0884 | 0.0875 | 0.1517 | 0.0002 | — |
| collect_drink @ ep313 | 313 | 0.8487 | 0.9517 | 0.7747 | 0.2659 | 0.0933 | 0.3345 | 0.0007 | — |
| eat_cow @ ep315 | 315 | 0.9453 | 0.9656 | 0.9638 | 0.4641 | 0.2120 | 0.2816 | 0.0006 | — |
| defeat_zombie @ ep450 | 450 | 0.9604 | 0.9186 | 0.9651 | 0.2168 | 0.1510 | 0.2506 | 0.0006 | — |
| defeat_skeleton @ ep803 | 803 | 0.7485 | 0.8904 | 0.9369 | 0.1028 | 0.1469 | 0.8166 | 0.0075 | — |
| place_table @ ep2115 | 2,115 | 0.8100 | 0.9497 | 0.8877 | 0.2543 | 0.2467 | 0.8974 | 0.0137 | — |
| make_wood_pickaxe @ ep3557 | 3,557 | 0.8622 | 0.9716 | 0.8191 | 0.4862 | 0.4020 | 0.8029 | 0.0184 | — |
| make_wood_sword @ ep4521 | 4,521 | 0.9023 | 0.9009 | 0.8773 | 0.2595 | 0.3794 | 0.4389 | 0.0083 | — |
| collect_stone @ ep6399 | 6,399 | 0.9927 | 0.9470 | 0.8309 | 0.7798 | 0.6678 | 0.4253 | 0.0072 | — |
| collect_coal @ ep7687 | 7,687 | 0.9432 | 0.8945 | 0.7324 | 0.3416 | 0.2527 | 0.3004 | 0.0042 | — |
| collect_drink @ ep9002 | 9,002 | 0.5616 | 0.8031 | 0.5811 | 0.2972 | 0.1746 | 0.3520 | 0.0054 | — |
| collect_sapling @ ep9002 | 9,002 | 0.6134 | 0.7666 | 0.5847 | 0.2757 | 0.1684 | 0.3793 | 0.0064 | — |
| collect_wood @ ep9002 | 9,002 | 0.4604 | 0.8041 | 0.5961 | 0.2909 | 0.1667 | 0.3639 | 0.0058 | — |
| defeat_zombie @ ep9002 | 9,002 | 0.7927 | 0.7808 | 0.6019 | 0.2782 | 0.1803 | 0.3587 | 0.0056 | — |
| eat_cow @ ep9002 | 9,002 | 0.6677 | 0.7975 | 0.5718 | 0.2868 | 0.1744 | 0.3609 | 0.0057 | — |
| place_plant @ ep9002 | 9,002 | 0.5093 | 0.7924 | 0.5217 | 0.2818 | 0.1671 | 0.3792 | 0.0063 | — |
| place_table @ ep9002 | 9,002 | 0.6077 | 0.7588 | 0.5673 | 0.2826 | 0.1668 | 0.3927 | 0.0068 | — |
| wake_up @ ep9002 | 9,002 | 0.5170 | 0.7783 | 0.5668 | 0.3249 | 0.1661 | 0.3561 | 0.0056 | — |
| make_wood_pickaxe @ ep9004 | 9,004 | 0.1929 | 0.7603 | 0.5595 | 0.2754 | 0.1733 | 0.3824 | 0.0065 | — |
| defeat_skeleton @ ep9017 | 9,017 | 0.5164 | 0.7789 | 0.5595 | 0.2834 | 0.1715 | 0.3538 | 0.0055 | — |
| collect_stone @ ep9058 | 9,058 | 0.8392 | 0.7781 | 0.6002 | 0.2690 | 0.1942 | 0.3391 | 0.0051 | — |
| make_wood_sword @ ep9214 | 9,214 | 0.8084 | 0.8691 | 0.7765 | 0.2805 | 0.3849 | 0.4303 | 0.0075 | — |
| collect_coal @ ep9224 | 9,224 | 0.9600 | 0.9116 | 0.8571 | 0.4597 | 0.6248 | 0.4472 | 0.0080 | — |
| make_stone_sword @ ep9875 | 9,875 | 0.9227 | 0.9251 | 0.7778 | 0.4301 | 0.3005 | 0.3979 | 0.0064 | — |
| place_furnace @ ep11209 | 11,209 | 0.9264 | 0.9143 | 0.6859 | 0.4015 | 0.3027 | 0.4332 | 0.0084 | — |
| place_stone @ ep12088 | 12,088 | 0.6899 | 0.9047 | 0.7211 | 0.2819 | 0.2557 | 0.7875 | 0.0286 | — |
| make_stone_pickaxe @ ep12409 | 12,409 | 0.9846 | 0.9351 | 0.8274 | 0.9435 | 0.8200 | 0.7891 | 0.0272 | — |

---

## Periodic Checkpoints

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA (ρ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 50,000 | 294 | 0.9789 | 0.9228 | 0.9034 | 0.1167 | 0.1013 | 0.4580 | 0.0008 | — |
| Step 100,000 | 572 | 0.7975 | 0.9103 | 0.9739 | 0.2912 | 0.2268 | 0.7545 | 0.0049 | — |
| Step 150,000 | 836 | 0.9322 | 0.9763 | 0.9223 | 0.5124 | 0.3195 | 0.6753 | 0.0052 | — |
| Step 200,000 | 1,110 | 0.5738 | 0.9384 | 0.8257 | 0.1637 | 0.1506 | 0.9311 | 0.0110 | — |
| Step 250,000 | 1,389 | 0.8974 | 0.9843 | 0.9546 | 0.3346 | 0.2884 | 0.6422 | 0.0059 | — |
| Step 300,000 | 1,656 | 0.8086 | 0.9460 | 0.9157 | 0.2525 | 0.2260 | 0.5994 | 0.0057 | — |
| Step 350,000 | 1,923 | 0.7787 | 0.9701 | 0.9366 | 0.3319 | 0.2983 | 0.6977 | 0.0077 | — |
| Step 400,000 | 2,184 | 0.7864 | 0.9401 | 0.8864 | 0.2225 | 0.1883 | 0.9123 | 0.0152 | — |
| Step 450,000 | 2,439 | 0.9111 | 0.9574 | 0.9080 | 0.4155 | 0.3844 | 0.8557 | 0.0155 | — |
| Step 500,000 | 2,697 | 0.9301 | 0.9581 | 0.8967 | 0.5736 | 0.4478 | 1.0688 | 0.0218 | — |
| Step 550,000 | 2,957 | 0.8959 | 0.9799 | 0.9074 | 0.5368 | 0.4379 | 0.7151 | 0.0122 | — |
| Step 600,000 | 3,201 | 0.7583 | 0.9584 | 0.8305 | 0.3938 | 0.2539 | 1.0752 | 0.0321 | — |
| Step 650,000 | 3,449 | 0.8003 | 0.9627 | 0.8281 | 0.4625 | 0.4439 | 0.7500 | 0.0164 | — |
| Step 700,000 | 3,676 | 0.6782 | 0.9417 | 0.8147 | 0.3433 | 0.3609 | 1.1596 | 0.0381 | — |
| Step 750,000 | 3,908 | 0.6709 | 0.8810 | 0.7762 | 0.2683 | 0.3147 | 0.5178 | 0.0094 | — |
| Step 800,000 | 4,125 | 0.8522 | 0.9172 | 0.7738 | 0.2578 | 0.2646 | 0.5122 | 0.0104 | — |
| Step 850,000 | 4,341 | 0.9375 | 0.9272 | 0.8466 | 0.3090 | 0.4175 | 0.4745 | 0.0095 | — |
| Step 900,000 | 4,564 | 0.5656 | 0.8845 | 0.7378 | 0.1882 | 0.2722 | 0.4220 | 0.0078 | — |
| Step 950,000 | 4,785 | 0.6837 | 0.8811 | 0.7146 | 0.1748 | 0.2346 | 0.4022 | 0.0064 | — |
| Step 1,000,000 | 5,002 | 0.7919 | 0.9266 | 0.7364 | 0.2570 | 0.3089 | 0.3100 | 0.0041 | — |
| Step 1,050,000 | 5,222 | 0.9025 | 0.9352 | 0.8096 | 0.3230 | 0.3095 | 0.3024 | 0.0035 | — |
| Step 1,100,000 | 5,445 | 0.8949 | 0.9338 | 0.7624 | 0.3160 | 0.3385 | 0.3728 | 0.0055 | — |
| Step 1,150,000 | 5,652 | 0.8786 | 0.9268 | 0.8145 | 0.3104 | 0.2923 | 0.3529 | 0.0051 | — |
| Step 1,200,000 | 5,858 | 0.7548 | 0.9012 | 0.7886 | 0.2489 | 0.3668 | 0.3867 | 0.0063 | — |
| Step 1,250,000 | 6,066 | 0.0114 | 0.8643 | 0.6556 | 0.2228 | 0.2723 | 0.3856 | 0.0065 | — |
| Step 1,300,000 | 6,275 | 0.6525 | 0.9251 | 0.7370 | 0.3034 | 0.2775 | 0.4090 | 0.0073 | — |
| Step 1,350,000 | 6,492 | 0.8552 | 0.8975 | 0.6578 | 0.2340 | 0.2308 | 0.4161 | 0.0075 | — |
| Step 1,400,000 | 6,704 | 0.8301 | 0.8913 | 0.7632 | 0.2117 | 0.2783 | 0.4661 | 0.0094 | — |
| Step 1,450,000 | 6,913 | 0.7543 | 0.8507 | 0.7261 | 0.2117 | 0.3143 | 0.4348 | 0.0086 | — |
| Step 1,500,000 | 7,121 | 0.9802 | 0.9211 | 0.7341 | 0.6192 | 0.5292 | 0.4052 | 0.0076 | — |
| Step 1,550,000 | 7,331 | 0.8576 | 0.8612 | 0.6775 | 0.2365 | 0.2253 | 0.3276 | 0.0050 | — |
| Step 1,600,000 | 7,545 | 0.8013 | 0.8559 | 0.7094 | 0.2073 | 0.2331 | 0.3979 | 0.0072 | — |
| Step 1,650,000 | 7,750 | 0.8114 | 0.7544 | 0.6959 | 0.1637 | 0.2273 | 0.3145 | 0.0049 | — |
| Step 1,700,000 | 7,956 | 0.6475 | 0.8508 | 0.6593 | 0.1800 | 0.1891 | 0.3226 | 0.0044 | — |
| Step 1,750,000 | 8,172 | 0.9678 | 0.8805 | 0.7805 | 0.2644 | 0.2658 | 0.3795 | 0.0067 | — |
| Step 1,800,000 | 8,374 | 0.8515 | 0.9027 | 0.7440 | 0.2547 | 0.3111 | 0.2717 | 0.0033 | — |
| Step 1,850,000 | 8,578 | 0.8901 | 0.8347 | 0.6537 | 0.3133 | 0.2426 | 0.3866 | 0.0060 | — |
| Step 1,900,000 | 8,792 | 0.9529 | 0.9209 | 0.7688 | 0.5428 | 0.6947 | 0.3565 | 0.0056 | — |
| Step 1,950,000 | 9,001 | 0.4775 | 0.7962 | 0.5503 | 0.2692 | 0.1737 | 0.3745 | 0.0061 | — |
| Step 2,000,000 | 9,215 | 0.8953 | 0.9095 | 0.6900 | 0.3674 | 0.2865 | 0.4386 | 0.0082 | — |
| Step 2,050,000 | 9,420 | 0.9189 | 0.8830 | 0.7477 | 0.4927 | 0.4338 | 0.3601 | 0.0057 | — |
| Step 2,100,000 | 9,642 | 0.9319 | 0.9143 | 0.7303 | 0.6065 | 0.4632 | 0.3498 | 0.0050 | — |
| Step 2,150,000 | 9,870 | 0.6685 | 0.8860 | 0.7045 | 0.2713 | 0.2765 | 0.3652 | 0.0054 | — |
| Step 2,200,000 | 10,087 | 0.7583 | 0.8755 | 0.7589 | 0.2309 | 0.2270 | 0.3957 | 0.0068 | — |
| Step 2,250,000 | 10,316 | 0.8094 | 0.8800 | 0.6944 | 0.3006 | 0.3151 | 0.4223 | 0.0075 | — |
| Step 2,300,000 | 10,541 | 0.9590 | 0.8752 | 0.7783 | 0.4732 | 0.4509 | 0.3864 | 0.0065 | — |
| Step 2,350,000 | 10,763 | 0.9681 | 0.9256 | 0.8244 | 0.4988 | 0.4017 | 0.4754 | 0.0095 | — |
| Step 2,400,000 | 10,998 | 0.7569 | 0.8135 | 0.6993 | 0.1974 | 0.2109 | 0.4698 | 0.0103 | — |
| Step 2,450,000 | 11,220 | 0.7687 | 0.8755 | 0.7130 | 0.2275 | 0.3138 | 0.4751 | 0.0104 | — |
| Step 2,500,000 | 11,443 | 0.9166 | 0.8953 | 0.6908 | 0.4298 | 0.3637 | 0.5487 | 0.0144 | — |
| Step 2,550,000 | 11,669 | 0.4175 | 0.8613 | 0.5963 | 0.2464 | 0.1870 | 0.6915 | 0.0205 | — |
| Step 2,600,000 | 11,881 | 0.8471 | 0.8891 | 0.6405 | 0.3049 | 0.2171 | 0.6426 | 0.0192 | — |
| Step 2,650,000 | 12,087 | 0.7548 | 0.8593 | 0.7476 | 0.2588 | 0.3188 | 0.8807 | 0.0345 | — |
| Step 2,700,000 | 12,284 | 0.9255 | 0.8995 | 0.8402 | 0.5379 | 0.6274 | 0.8897 | 0.0362 | — |
| Step 2,750,000 | 12,503 | 0.5770 | 0.8602 | 0.6185 | 0.2183 | 0.1860 | 0.7976 | 0.0273 | — |
| Step 2,800,000 | 12,701 | 0.6319 | 0.8789 | 0.6332 | 0.2282 | 0.2056 | 0.9543 | 0.0372 | — |
| Step 2,850,000 | 12,908 | 0.9332 | 0.8967 | 0.7187 | 0.4312 | 0.4509 | 1.1081 | 0.0547 | — |
| Step 2,900,000 | 13,118 | 0.9520 | 0.8754 | 0.7618 | 0.4309 | 0.4474 | 1.1821 | 0.0556 | — |
| Step 2,950,000 | 13,326 | 0.8130 | 0.8810 | 0.7464 | 0.3115 | 0.4263 | 1.3181 | 0.0682 | — |
| Step 3,000,000 | 13,532 | 0.7021 | 0.9048 | 0.6644 | 0.2734 | 0.2073 | 1.3695 | 0.0729 | — |
