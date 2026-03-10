# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 3  
**Total episodes:** 18,512  
**Experiment root:** `Rainbow-Proof-of-Concept-Runs\seed_3`  
**Generated:** 2026-03-09 12:20

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 16 episodes — step 5,008 | 16 | 0.1211 | — | — | 2.9296 | 1.9665 | 0.0697 | 0.0061 | — |
| Checkpoint 96 episodes — step 30,048 | 96 | 0.9257 | — | — | 1.1668 | 1.2860 | 0.0782 | 0.0036 | — |
| Checkpoint 192 episodes — step 55,088 | 192 | 0.9247 | — | — | 1.1350 | 1.2284 | 0.0673 | 0.0021 | — |
| Checkpoint 288 episodes — step 80,128 | 288 | 0.9085 | — | — | 1.1461 | 1.1700 | 0.0898 | 0.0042 | — |
| Checkpoint 384 episodes — step 105,168 | 384 | 0.8657 | — | — | 1.1242 | 1.1526 | 0.0965 | 0.0053 | — |
| Checkpoint 480 episodes — step 130,208 | 480 | 0.8834 | — | — | 1.1002 | 1.1216 | 0.1174 | 0.0079 | — |
| Checkpoint 560 episodes — step 155,248 | 560 | 0.8784 | — | — | 1.1670 | 1.1059 | 0.0980 | 0.0056 | — |
| Checkpoint 656 episodes — step 180,288 | 656 | 0.8833 | — | — | 1.0921 | 1.0926 | 0.0596 | 0.0022 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 752 episodes — step 205,328 | 752 | 0.8938 | — | — | 1.2302 | 1.0914 | 0.0699 | 0.0017 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 848 episodes — step 230,368 | 848 | 0.9696 | — | — | 1.1785 | 1.1444 | 0.1441 | 0.0049 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 944 episodes — step 255,408 | 944 | 0.7639 | — | — | 1.2197 | 1.1673 | 0.2146 | 0.0143 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,024 episodes — step 280,448 | 1,024 | 0.8334 | — | — | 1.2901 | 1.2511 | 0.1765 | 0.0099 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,120 episodes — step 305,488 | 1,120 | 0.8732 | — | — | 1.2610 | 1.3479 | 0.1551 | 0.0067 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 1,216 episodes — step 330,528 | 1,216 | 0.8701 | — | — | 1.2765 | 1.5078 | 0.1722 | 0.0076 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 1,312 episodes — step 355,568 | 1,312 | 0.7341 | — | — | 1.2905 | 1.6956 | 0.1833 | 0.0100 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,408 episodes — step 380,608 | 1,408 | 0.7551 | — | — | 1.2518 | 1.9186 | 0.2137 | 0.0140 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 1,488 episodes — step 405,648 | 1,488 | 0.7024 | — | — | 1.2538 | 2.1371 | 0.2515 | 0.0183 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 1,584 episodes — step 430,688 | 1,584 | 0.6332 | — | — | 1.2764 | 2.6235 | 0.2906 | 0.0280 | — |
| Checkpoint 1,680 episodes — step 455,728 | 1,680 | 0.6266 | — | — | 1.3304 | 2.6949 | 0.2944 | 0.0290 | — |
| Checkpoint 1,776 episodes — step 480,768 | 1,776 | 0.7006 | — | — | 1.3222 | 2.5291 | 0.2943 | 0.0318 | — |
| Checkpoint 1,872 episodes — step 505,808 | 1,872 | 0.7526 | — | — | 1.5523 | 2.3632 | 0.2522 | 0.0238 | — |
| Checkpoint 1,952 episodes — step 530,848 | 1,952 | 0.7971 | — | — | 1.7469 | 2.1530 | 0.2108 | 0.0160 | — |
| Checkpoint 2,048 episodes — step 555,888 | 2,048 | 0.8458 | — | — | 1.8530 | 1.8992 | 0.1392 | 0.0073 | — |
| Checkpoint 2,144 episodes — step 580,928 | 2,144 | 0.8814 | — | — | 1.8718 | 1.8793 | 0.1663 | 0.0101 | — |
| Checkpoint 2,240 episodes — step 605,968 | 2,240 | 0.8411 | — | — | 1.4858 | 1.8954 | 0.1633 | 0.0094 | — |
| Checkpoint 2,336 episodes — step 631,008 | 2,336 | 0.8506 | — | — | 1.4936 | 1.6156 | 0.1034 | 0.0040 | — |
| Checkpoint 2,416 episodes — step 656,048 | 2,416 | 0.8576 | — | — | 1.4694 | 1.3383 | 0.1150 | 0.0042 | — |
| Checkpoint 2,512 episodes — step 681,088 | 2,512 | 0.9110 | — | — | 1.3707 | 1.4835 | 0.1064 | 0.0036 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,608 episodes — step 706,128 | 2,608 | 0.8891 | — | — | 1.3234 | 1.5002 | 0.1173 | 0.0045 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,704 episodes — step 731,168 | 2,704 | 0.8758 | — | — | 1.3519 | 1.5630 | 0.0853 | 0.0034 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,800 episodes — step 756,208 | 2,800 | 0.8777 | — | — | 1.3700 | 1.6497 | 0.0910 | 0.0042 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,880 episodes — step 781,248 | 2,880 | 0.8294 | — | — | 1.5049 | 1.6288 | 0.0891 | 0.0040 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,976 episodes — step 806,288 | 2,976 | 0.9641 | — | — | 1.5146 | 1.6419 | 0.0693 | 0.0020 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 3,072 episodes — step 831,328 | 3,072 | 0.9389 | — | — | 1.4960 | 1.6134 | 0.0480 | 0.0010 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,168 episodes — step 856,368 | 3,168 | 0.9263 | — | — | 1.4078 | 1.6304 | 0.0749 | 0.0015 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,264 episodes — step 881,408 | 3,264 | 0.9152 | — | — | 1.3963 | 1.6775 | 0.0948 | 0.0023 | — |
| Checkpoint 3,344 episodes — step 906,448 | 3,344 | 0.9143 | — | — | 1.4245 | 1.6903 | 0.1246 | 0.0034 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 3,440 episodes — step 931,488 | 3,440 | 0.9347 | — | — | 1.4371 | 1.8228 | 0.1145 | 0.0031 | — |
| Checkpoint 3,536 episodes — step 956,528 | 3,536 | 0.9268 | — | — | 1.4239 | 1.8548 | 0.1609 | 0.0053 | — |
| Checkpoint 3,632 episodes — step 981,568 | 3,632 | 0.8553 | — | — | 1.3505 | 1.8737 | 0.1891 | 0.0072 | — |
| Checkpoint 3,728 episodes — step 1,006,608 | 3,728 | 0.8468 | — | — | 1.2867 | 1.7479 | 0.1410 | 0.0041 | — |
| Checkpoint 3,808 episodes — step 1,031,648 | 3,808 | 0.8116 | — | — | 1.3632 | 1.7406 | 0.1131 | 0.0025 | — |
| Checkpoint 3,904 episodes — step 1,056,688 | 3,904 | 0.8982 | — | — | 1.3344 | 1.6860 | 0.0682 | 0.0013 | — |
| Checkpoint 4,000 episodes — step 1,081,728 | 4,000 | 0.9134 | — | — | 1.4144 | 1.7580 | 0.0925 | 0.0022 | — |
| Checkpoint 4,096 episodes — step 1,106,768 | 4,096 | 0.7720 | — | — | 1.3530 | 1.8040 | 0.0938 | 0.0027 | — |
| Checkpoint 4,176 episodes — step 1,131,808 | 4,176 | 0.8112 | — | — | 1.3592 | 1.6566 | 0.1002 | 0.0030 | — |
| Checkpoint 4,272 episodes — step 1,156,848 | 4,272 | 0.8277 | — | — | 1.4466 | 1.6258 | 0.0853 | 0.0028 | — |
| Checkpoint 4,368 episodes — step 1,181,888 | 4,368 | 0.8457 | — | — | 1.3473 | 1.5979 | 0.0786 | 0.0024 | — |
| Checkpoint 4,464 episodes — step 1,206,928 | 4,464 | 0.8042 | — | — | 1.2751 | 1.5215 | 0.1094 | 0.0031 | — |
| Checkpoint 4,560 episodes — step 1,231,968 | 4,560 | 0.9078 | — | — | 1.1516 | 1.4292 | 0.1860 | 0.0064 | — |
| Checkpoint 4,640 episodes — step 1,257,008 | 4,640 | 0.9186 | — | — | 1.1728 | 1.3937 | 0.1999 | 0.0065 | — |
| Checkpoint 4,736 episodes — step 1,282,048 | 4,736 | 0.9304 | — | — | 1.2429 | 1.4288 | 0.1609 | 0.0046 | — |
| Checkpoint 4,832 episodes — step 1,307,088 | 4,832 | 0.9622 | — | — | 1.2137 | 1.4087 | 0.1207 | 0.0029 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4,928 episodes — step 1,332,128 | 4,928 | 0.9619 | — | — | 1.2321 | 1.4632 | 0.1072 | 0.0027 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,024 episodes — step 1,357,168 | 5,024 | 0.9257 | — | — | 1.1786 | 1.4813 | 0.1501 | 0.0051 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,104 episodes — step 1,382,208 | 5,104 | 0.8669 | — | — | 1.1395 | 1.5986 | 0.2335 | 0.0109 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,200 episodes — step 1,407,248 | 5,200 | 0.8758 | — | — | 1.1802 | 1.6205 | 0.2886 | 0.0154 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,296 episodes — step 1,432,288 | 5,296 | 0.8826 | — | — | 1.2260 | 1.6778 | 0.2971 | 0.0157 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,392 episodes — step 1,457,328 | 5,392 | 0.8781 | — | — | 1.2096 | 1.6991 | 0.2742 | 0.0143 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 5,488 episodes — step 1,482,368 | 5,488 | 0.8846 | — | — | 1.1764 | 1.6161 | 0.2746 | 0.0128 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,568 episodes — step 1,507,408 | 5,568 | 0.8550 | — | — | 1.1511 | 1.5664 | 0.2707 | 0.0128 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,664 episodes — step 1,532,448 | 5,664 | 0.8101 | — | — | 1.1593 | 1.5885 | 0.2631 | 0.0100 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,760 episodes — step 1,557,488 | 5,760 | 0.9169 | — | — | 1.1686 | 1.4249 | 0.2004 | 0.0055 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,856 episodes — step 1,582,528 | 5,856 | 0.8797 | — | — | 1.2152 | 1.4869 | 0.1972 | 0.0054 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,952 episodes — step 1,607,568 | 5,952 | 0.9138 | — | — | 1.1849 | 1.4696 | 0.1750 | 0.0047 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,032 episodes — step 1,632,608 | 6,032 | 0.9176 | — | — | 1.1999 | 1.4403 | 0.1931 | 0.0047 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,128 episodes — step 1,657,648 | 6,128 | 0.8898 | — | — | 1.1739 | 1.4798 | 0.2049 | 0.0055 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6,224 episodes — step 1,682,688 | 6,224 | 0.9055 | — | — | 1.1671 | 1.5062 | 0.2347 | 0.0055 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6,320 episodes — step 1,707,728 | 6,320 | 0.9028 | — | — | 1.1726 | 1.5268 | 0.2225 | 0.0051 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6,416 episodes — step 1,732,768 | 6,416 | 0.9107 | — | — | 1.2038 | 1.5494 | 0.2537 | 0.0059 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6,496 episodes — step 1,757,808 | 6,496 | 0.9018 | — | — | 1.1346 | 1.4302 | 0.2296 | 0.0053 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,592 episodes — step 1,782,848 | 6,592 | 0.8555 | — | — | 1.1545 | 1.5156 | 0.2124 | 0.0053 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,688 episodes — step 1,807,888 | 6,688 | 0.8642 | — | — | 1.1450 | 1.5454 | 0.2087 | 0.0050 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,784 episodes — step 1,832,928 | 6,784 | 0.8907 | — | — | 1.1489 | 1.4973 | 0.1866 | 0.0043 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 6,880 episodes — step 1,857,968 | 6,880 | 0.9284 | — | — | 1.1370 | 1.2952 | 0.1478 | 0.0034 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,960 episodes — step 1,883,008 | 6,960 | 0.9244 | — | — | 1.1370 | 1.3749 | 0.1012 | 0.0028 | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7,056 episodes — step 1,908,048 | 7,056 | 0.8941 | — | — | 1.1578 | 1.4077 | 0.1029 | 0.0031 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,152 episodes — step 1,933,088 | 7,152 | 0.9490 | — | — | 1.1645 | 1.2992 | 0.1658 | 0.0044 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,248 episodes — step 1,958,128 | 7,248 | 0.9006 | — | — | 1.1328 | 1.3732 | 0.1816 | 0.0054 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,344 episodes — step 1,983,168 | 7,344 | 0.9268 | — | — | 1.1178 | 1.3245 | 0.1749 | 0.0051 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,424 episodes — step 2,008,208 | 7,424 | 0.9168 | — | — | 1.0963 | 1.2820 | 0.1915 | 0.0066 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,520 episodes — step 2,033,248 | 7,520 | 0.8828 | — | — | 1.1081 | 1.3844 | 0.1701 | 0.0069 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,616 episodes — step 2,058,288 | 7,616 | 0.8579 | — | — | 1.0929 | 1.4182 | 0.1755 | 0.0079 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,712 episodes — step 2,083,328 | 7,712 | 0.8424 | — | — | 1.1184 | 1.3194 | 0.1791 | 0.0075 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,808 episodes — step 2,108,368 | 7,808 | 0.9079 | — | — | 1.0994 | 1.3475 | 0.1688 | 0.0066 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,888 episodes — step 2,133,408 | 7,888 | 0.9421 | — | — | 1.1103 | 1.2604 | 0.1344 | 0.0039 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,984 episodes — step 2,158,448 | 7,984 | 0.8863 | — | — | 1.1245 | 1.3102 | 0.1535 | 0.0033 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,080 episodes — step 2,183,488 | 8,080 | 0.9430 | — | — | 1.1489 | 1.1921 | 0.1440 | 0.0026 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,176 episodes — step 2,208,528 | 8,176 | 0.9087 | — | — | 1.1668 | 1.2370 | 0.1480 | 0.0027 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,272 episodes — step 2,233,568 | 8,272 | 0.8484 | — | — | 1.1705 | 1.4451 | 0.2475 | 0.0053 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,352 episodes — step 2,258,608 | 8,352 | 0.8497 | — | — | 1.1477 | 1.5386 | 0.2824 | 0.0080 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,448 episodes — step 2,283,648 | 8,448 | 0.8389 | — | — | 1.2058 | 1.5411 | 0.1847 | 0.0054 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,544 episodes — step 2,308,688 | 8,544 | 0.9060 | — | — | 1.1405 | 1.4522 | 0.1437 | 0.0039 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,640 episodes — step 2,333,728 | 8,640 | 0.9481 | — | — | 1.1221 | 1.2879 | 0.0774 | 0.0019 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,736 episodes — step 2,358,768 | 8,736 | 0.8632 | — | — | 1.2137 | 1.2626 | 0.0806 | 0.0024 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,816 episodes — step 2,383,808 | 8,816 | 0.9445 | — | — | 1.1091 | 1.2847 | 0.0941 | 0.0030 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,912 episodes — step 2,408,848 | 8,912 | 0.7360 | — | — | 1.1944 | 1.3282 | 0.1153 | 0.0036 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,008 episodes — step 2,433,888 | 9,008 | 0.9186 | — | — | 1.1206 | 1.2800 | 0.1730 | 0.0050 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,104 episodes — step 2,458,928 | 9,104 | 0.8559 | — | — | 1.1367 | 1.3351 | 0.2666 | 0.0080 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,184 episodes — step 2,483,968 | 9,184 | 0.8547 | — | — | 1.1791 | 1.3644 | 0.3114 | 0.0094 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,280 episodes — step 2,509,008 | 9,280 | 0.8517 | — | — | 1.1762 | 1.3293 | 0.3363 | 0.0095 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,376 episodes — step 2,534,048 | 9,376 | 0.8413 | — | — | 1.1956 | 1.3735 | 0.3233 | 0.0087 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 9,472 episodes — step 2,559,088 | 9,472 | 0.7931 | — | — | 1.2322 | 1.4514 | 0.3396 | 0.0099 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9,568 episodes — step 2,584,128 | 9,568 | 0.7932 | — | — | 1.2679 | 1.4976 | 0.3679 | 0.0106 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9,648 episodes — step 2,609,168 | 9,648 | 0.8027 | — | — | 1.2167 | 1.4926 | 0.3415 | 0.0098 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 9,744 episodes — step 2,634,208 | 9,744 | 0.7257 | — | — | 1.2753 | 1.4925 | 0.3555 | 0.0096 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9,840 episodes — step 2,659,248 | 9,840 | 0.7455 | — | — | 1.2117 | 1.5832 | 0.3825 | 0.0110 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 9,936 episodes — step 2,684,288 | 9,936 | 0.7791 | — | — | 1.2034 | 1.5627 | 0.3952 | 0.0128 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 10,032 episodes — step 2,709,328 | 10,032 | 0.7041 | — | — | 1.2748 | 1.5666 | 0.3855 | 0.0129 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 10,112 episodes — step 2,734,368 | 10,112 | 0.8505 | — | — | 1.2242 | 1.5870 | 0.3429 | 0.0122 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 10,208 episodes — step 2,759,408 | 10,208 | 0.8625 | — | — | 1.2223 | 1.5600 | 0.3141 | 0.0103 | — |
| Checkpoint 10,304 episodes — step 2,784,448 | 10,304 | 0.8489 | — | — | 1.2071 | 1.5697 | 0.3017 | 0.0097 | — |
| Checkpoint 10,400 episodes — step 2,809,488 | 10,400 | 0.8474 | — | — | 1.1742 | 1.5632 | 0.3184 | 0.0109 | — |
| Checkpoint 10,496 episodes — step 2,834,528 | 10,496 | 0.8565 | — | — | 1.2057 | 1.5616 | 0.3334 | 0.0110 | — |
| Checkpoint 10,576 episodes — step 2,859,568 | 10,576 | 0.8241 | — | — | 1.1716 | 1.5357 | 0.3468 | 0.0127 | — |
| Checkpoint 10,672 episodes — step 2,884,608 | 10,672 | 0.8389 | — | — | 1.1808 | 1.4862 | 0.3353 | 0.0119 | — |
| Checkpoint 10,768 episodes — step 2,909,648 | 10,768 | 0.7945 | — | — | 1.2481 | 1.4605 | 0.3752 | 0.0127 | — |
| Checkpoint 10,864 episodes — step 2,934,688 | 10,864 | 0.8024 | — | — | 1.1873 | 1.5031 | 0.3511 | 0.0115 | — |
| Checkpoint 10,960 episodes — step 2,959,728 | 10,960 | 0.7794 | — | — | 1.1913 | 1.5068 | 0.3595 | 0.0108 | — |
| Checkpoint 11,040 episodes — step 2,984,768 | 11,040 | 0.7512 | — | — | 1.3698 | 1.4334 | 0.3475 | 0.0115 | — |
| Checkpoint 11,136 episodes — step 3,009,808 | 11,136 | 0.8117 | — | — | 1.2239 | 1.4525 | 0.2757 | 0.0104 | — |
| Checkpoint 11,232 episodes — step 3,034,848 | 11,232 | 0.7855 | — | — | 1.2626 | 1.4257 | 0.2778 | 0.0093 | — |
| Checkpoint 11,328 episodes — step 3,059,888 | 11,328 | 0.8406 | — | — | 1.2246 | 1.4334 | 0.2530 | 0.0089 | — |
| Checkpoint 11,424 episodes — step 3,084,928 | 11,424 | 0.8121 | — | — | 1.2127 | 1.4638 | 0.2967 | 0.0115 | — |
| Checkpoint 11,504 episodes — step 3,109,968 | 11,504 | 0.8615 | — | — | 1.1778 | 1.4710 | 0.3412 | 0.0123 | — |
| Checkpoint 11,600 episodes — step 3,135,008 | 11,600 | 0.8327 | — | — | 1.1647 | 1.4525 | 0.3473 | 0.0120 | — |
| Checkpoint 11,696 episodes — step 3,160,048 | 11,696 | 0.8405 | — | — | 1.1608 | 1.4374 | 0.3745 | 0.0129 | — |
| Checkpoint 11,792 episodes — step 3,185,088 | 11,792 | 0.8705 | — | — | 1.1533 | 1.4513 | 0.3503 | 0.0136 | — |
| Checkpoint 11,888 episodes — step 3,210,128 | 11,888 | 0.8564 | — | — | 1.2355 | 1.4669 | 0.2983 | 0.0133 | — |
| Checkpoint 11,968 episodes — step 3,235,168 | 11,968 | 0.8141 | — | — | 1.2641 | 1.4957 | 0.3150 | 0.0143 | — |
| Checkpoint 12,064 episodes — step 3,260,208 | 12,064 | 0.8401 | — | — | 1.2569 | 1.4885 | 0.3531 | 0.0143 | — |
| Checkpoint 12,160 episodes — step 3,285,248 | 12,160 | 0.8900 | — | — | 1.1565 | 1.4901 | 0.3616 | 0.0135 | — |
| Checkpoint 12,256 episodes — step 3,310,288 | 12,256 | 0.8847 | — | — | 1.2373 | 1.4714 | 0.2638 | 0.0123 | — |
| Checkpoint 12,352 episodes — step 3,335,328 | 12,352 | 0.8293 | — | — | 1.2833 | 1.5280 | 0.2449 | 0.0136 | — |
| Checkpoint 12,432 episodes — step 3,360,368 | 12,432 | 0.8389 | — | — | 1.2090 | 1.4920 | 0.3237 | 0.0150 | — |
| Checkpoint 12,528 episodes — step 3,385,408 | 12,528 | 0.8302 | — | — | 1.3277 | 1.4603 | 0.2537 | 0.0122 | — |
| Checkpoint 12,624 episodes — step 3,410,448 | 12,624 | 0.8933 | — | — | 1.2125 | 1.4694 | 0.2840 | 0.0120 | — |
| Checkpoint 12,720 episodes — step 3,435,488 | 12,720 | 0.8885 | — | — | 1.3384 | 1.5463 | 0.2687 | 0.0125 | — |
| Checkpoint 12,816 episodes — step 3,460,528 | 12,816 | 0.9331 | — | — | 1.1915 | 1.4544 | 0.2807 | 0.0108 | — |
| Checkpoint 12,896 episodes — step 3,485,568 | 12,896 | 0.9154 | — | — | 1.1958 | 1.4779 | 0.2619 | 0.0098 | — |
| Checkpoint 12,992 episodes — step 3,510,608 | 12,992 | 0.9080 | — | — | 1.2347 | 1.4726 | 0.2456 | 0.0095 | — |
| Checkpoint 13,088 episodes — step 3,535,648 | 13,088 | 0.8380 | — | — | 1.3644 | 1.4423 | 0.1970 | 0.0095 | — |
| Checkpoint 13,184 episodes — step 3,560,688 | 13,184 | 0.8673 | — | — | 1.3420 | 1.4394 | 0.1640 | 0.0076 | — |
| Checkpoint 13,280 episodes — step 3,585,728 | 13,280 | 0.8817 | — | — | 1.2544 | 1.3650 | 0.1907 | 0.0088 | — |
| Checkpoint 13,360 episodes — step 3,610,768 | 13,360 | 0.9184 | — | — | 1.2305 | 1.4266 | 0.1867 | 0.0083 | — |
| Checkpoint 13,456 episodes — step 3,635,808 | 13,456 | 0.8388 | — | — | 1.3330 | 1.4181 | 0.1898 | 0.0102 | — |
| Checkpoint 13,552 episodes — step 3,660,848 | 13,552 | 0.8229 | — | — | 1.3216 | 1.4247 | 0.1878 | 0.0108 | — |
| Checkpoint 13,648 episodes — step 3,685,888 | 13,648 | 0.8502 | — | — | 1.2740 | 1.4282 | 0.2451 | 0.0107 | — |
| Checkpoint 13,744 episodes — step 3,710,928 | 13,744 | 0.8787 | — | — | 1.1663 | 1.4493 | 0.2939 | 0.0120 | — |
| Checkpoint 13,824 episodes — step 3,735,968 | 13,824 | 0.8406 | — | — | 1.3243 | 1.4335 | 0.2195 | 0.0100 | — |
| Checkpoint 13,920 episodes — step 3,761,008 | 13,920 | 0.9018 | — | — | 1.2466 | 1.4458 | 0.1846 | 0.0088 | — |
| Checkpoint 14,016 episodes — step 3,786,048 | 14,016 | 0.9158 | — | — | 1.2096 | 1.4666 | 0.2212 | 0.0091 | — |
| Checkpoint 14,112 episodes — step 3,811,088 | 14,112 | 0.8749 | — | — | 1.2104 | 1.5251 | 0.2603 | 0.0122 | — |
| Checkpoint 14,192 episodes — step 3,836,128 | 14,192 | 0.8494 | — | — | 1.2338 | 1.5176 | 0.2455 | 0.0114 | — |
| Checkpoint 14,288 episodes — step 3,861,168 | 14,288 | 0.8670 | — | — | 1.2947 | 1.5447 | 0.1968 | 0.0100 | — |
| Checkpoint 14,384 episodes — step 3,886,208 | 14,384 | 0.9214 | — | — | 1.2992 | 1.4323 | 0.1771 | 0.0076 | — |
| Checkpoint 14,480 episodes — step 3,911,248 | 14,480 | 0.9082 | — | — | 1.2122 | 1.4822 | 0.2980 | 0.0102 | — |
| Checkpoint 14,576 episodes — step 3,936,288 | 14,576 | 0.9232 | — | — | 1.2138 | 1.4850 | 0.2694 | 0.0104 | — |
| Checkpoint 14,656 episodes — step 3,961,328 | 14,656 | 0.8764 | — | — | 1.2686 | 1.5068 | 0.2104 | 0.0107 | — |
| Checkpoint 14,752 episodes — step 3,986,368 | 14,752 | 0.9238 | — | — | 1.2557 | 1.4752 | 0.2190 | 0.0109 | — |
| Checkpoint 14,848 episodes — step 4,011,408 | 14,848 | 0.9316 | — | — | 1.2453 | 1.4582 | 0.2257 | 0.0098 | — |
| Checkpoint 14,944 episodes — step 4,036,448 | 14,944 | 0.9100 | — | — | 1.2339 | 1.4988 | 0.2356 | 0.0094 | — |
| Checkpoint 15,040 episodes — step 4,061,488 | 15,040 | 0.9222 | — | — | 1.2167 | 1.4456 | 0.2970 | 0.0125 | — |
| Checkpoint 15,120 episodes — step 4,086,528 | 15,120 | 0.8848 | — | — | 1.2370 | 1.4085 | 0.2710 | 0.0127 | — |
| Checkpoint 15,216 episodes — step 4,111,568 | 15,216 | 0.8729 | — | — | 1.2229 | 1.4274 | 0.2682 | 0.0122 | — |
| Checkpoint 15,312 episodes — step 4,136,608 | 15,312 | 0.8961 | — | — | 1.1295 | 1.4284 | 0.4231 | 0.0182 | — |
| Checkpoint 15,408 episodes — step 4,161,648 | 15,408 | 0.8006 | — | — | 1.2519 | 1.4043 | 0.3311 | 0.0152 | — |
| Checkpoint 15,504 episodes — step 4,186,688 | 15,504 | 0.8508 | — | — | 1.2574 | 1.4011 | 0.2914 | 0.0159 | — |
| Checkpoint 15,584 episodes — step 4,211,728 | 15,584 | 0.8982 | — | — | 1.1552 | 1.4015 | 0.3061 | 0.0147 | — |
| Checkpoint 15,680 episodes — step 4,236,768 | 15,680 | 0.9153 | — | — | 1.1570 | 1.3909 | 0.2698 | 0.0129 | — |
| Checkpoint 15,776 episodes — step 4,261,808 | 15,776 | 0.9166 | — | — | 1.1782 | 1.3817 | 0.2604 | 0.0121 | — |
| Checkpoint 15,872 episodes — step 4,286,848 | 15,872 | 0.9230 | — | — | 1.1916 | 1.4040 | 0.2494 | 0.0109 | — |
| Checkpoint 15,968 episodes — step 4,311,888 | 15,968 | 0.9035 | — | — | 1.2513 | 1.4469 | 0.2310 | 0.0103 | — |
| Checkpoint 16,048 episodes — step 4,336,928 | 16,048 | 0.8944 | — | — | 1.2716 | 1.4449 | 0.2746 | 0.0138 | — |
| Checkpoint 16,144 episodes — step 4,361,968 | 16,144 | 0.8764 | — | — | 1.1556 | 1.4861 | 0.4224 | 0.0186 | — |
| Checkpoint 16,240 episodes — step 4,387,008 | 16,240 | 0.9039 | — | — | 1.2326 | 1.4921 | 0.3109 | 0.0158 | — |
| Checkpoint 16,336 episodes — step 4,412,048 | 16,336 | 0.8794 | — | — | 1.3678 | 1.4955 | 0.2452 | 0.0145 | — |
| Checkpoint 16,432 episodes — step 4,437,088 | 16,432 | 0.9249 | — | — | 1.3984 | 1.5704 | 0.1950 | 0.0111 | — |
| Checkpoint 16,512 episodes — step 4,462,128 | 16,512 | 0.9370 | — | — | 1.3933 | 1.5686 | 0.1784 | 0.0094 | — |
| Checkpoint 16,608 episodes — step 4,487,168 | 16,608 | 0.9143 | — | — | 1.3580 | 1.5493 | 0.2334 | 0.0117 | — |
| Checkpoint 16,704 episodes — step 4,512,208 | 16,704 | 0.9048 | — | — | 1.2945 | 1.5838 | 0.3063 | 0.0131 | — |
| Checkpoint 16,800 episodes — step 4,537,248 | 16,800 | 0.9055 | — | — | 1.3436 | 1.5268 | 0.3063 | 0.0119 | — |
| Checkpoint 16,896 episodes — step 4,562,288 | 16,896 | 0.9395 | — | — | 1.4228 | 1.5258 | 0.2221 | 0.0078 | — |
| Checkpoint 16,976 episodes — step 4,587,328 | 16,976 | 0.9211 | — | — | 1.5038 | 1.5894 | 0.1694 | 0.0054 | — |
| Checkpoint 17,072 episodes — step 4,612,368 | 17,072 | 0.9371 | — | — | 1.5867 | 1.4899 | 0.1135 | 0.0038 | — |
| Checkpoint 17,168 episodes — step 4,637,408 | 17,168 | 0.9106 | — | — | 1.6220 | 1.5141 | 0.0798 | 0.0027 | — |
| Checkpoint 17,264 episodes — step 4,662,448 | 17,264 | 0.8971 | — | — | 1.6560 | 1.5655 | 0.0688 | 0.0021 | — |
| Checkpoint 17,360 episodes — step 4,687,488 | 17,360 | 0.8654 | — | — | 1.7492 | 1.6280 | 0.0728 | 0.0024 | — |
| Checkpoint 17,440 episodes — step 4,712,528 | 17,440 | 0.9019 | — | — | 1.7681 | 1.6415 | 0.0620 | 0.0017 | — |
| Checkpoint 17,536 episodes — step 4,737,568 | 17,536 | 0.9334 | — | — | 1.7745 | 1.7009 | 0.1081 | 0.0024 | — |
| Checkpoint 17,632 episodes — step 4,762,608 | 17,632 | 0.9478 | — | — | 1.7844 | 1.7202 | 0.0653 | 0.0012 | — |
| Checkpoint 17,728 episodes — step 4,787,648 | 17,728 | 0.9294 | — | — | 1.7952 | 1.7470 | 0.0596 | 0.0013 | — |
| Checkpoint 17,824 episodes — step 4,812,688 | 17,824 | 0.9103 | — | — | 1.8714 | 1.6619 | 0.0578 | 0.0016 | — |
| Checkpoint 17,904 episodes — step 4,837,728 | 17,904 | 0.9060 | — | — | 1.8721 | 1.7071 | 0.0833 | 0.0022 | — |
| Checkpoint 18,000 episodes — step 4,862,768 | 18,000 | 0.9484 | — | — | 1.8693 | 1.7053 | 0.0934 | 0.0028 | — |
| Checkpoint 18,096 episodes — step 4,887,808 | 18,096 | 0.9426 | — | — | 1.8267 | 1.6291 | 0.0772 | 0.0019 | — |
| Checkpoint 18,192 episodes — step 4,912,848 | 18,192 | 0.9372 | — | — | 1.8354 | 1.6543 | 0.0754 | 0.0017 | — |
| Checkpoint 18,288 episodes — step 4,937,888 | 18,288 | 0.9295 | — | — | 1.8156 | 1.6364 | 0.0671 | 0.0015 | — |
| Checkpoint 18,368 episodes — step 4,962,928 | 18,368 | 0.9023 | — | — | 1.9001 | 1.6325 | 0.0608 | 0.0016 | — |
| Checkpoint 18,464 episodes — step 4,987,968 | 18,464 | 0.9501 | — | — | 1.8165 | 1.6001 | 0.0586 | 0.0017 | — |
| Checkpoint 18,512 episodes — step 5,000,000 | 18,512 | 0.9457 | — | — | 1.8247 | 1.5698 | 0.0663 | 0.0020 | — |

---

## Checkpoint 16 episodes — step 5,008

**Episodes:** 16  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.1211 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 2.9296 |
| Gradient Magnitude (Failure) | 1.9665 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9400 | 0.0496 |
| Neutral | 2.2385 | 0.1675 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0697 |
| Cosine Distance | 0.0061 |
| Clusters | 65 |
| Noise Fraction | 0.0069 |

---

## Checkpoint 96 episodes — step 30,048

**Episodes:** 96  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.000) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9257 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1668 |
| Gradient Magnitude (Failure) | 1.2860 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3883 | 0.6758 |
| Neutral | 1.1431 | 0.9690 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0782 |
| Cosine Distance | 0.0036 |
| Clusters | 294 |
| Noise Fraction | 0.0164 |

---

## Checkpoint 192 episodes — step 55,088

**Episodes:** 192  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.000) | top 25% (≥ 0.200)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9247 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1350 |
| Gradient Magnitude (Failure) | 1.2284 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2006 | 0.8088 |
| Neutral | 1.1267 | 0.9955 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0673 |
| Cosine Distance | 0.0021 |
| Clusters | 333 |
| Noise Fraction | 0.0200 |

---

## Checkpoint 288 episodes — step 80,128

**Episodes:** 288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.000) | top 25% (≥ 0.200)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9085 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1461 |
| Gradient Magnitude (Failure) | 1.1700 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4840 | 0.8448 |
| Neutral | 1.0990 | 0.9572 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0898 |
| Cosine Distance | 0.0042 |
| Clusters | 316 |
| Noise Fraction | 0.0220 |

---

## Checkpoint 384 episodes — step 105,168

**Episodes:** 384  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.000) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8657 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1242 |
| Gradient Magnitude (Failure) | 1.1526 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3369 | 0.7267 |
| Neutral | 1.1288 | 0.9946 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0965 |
| Cosine Distance | 0.0053 |
| Clusters | 317 |
| Noise Fraction | 0.0327 |

---

## Checkpoint 480 episodes — step 130,208

**Episodes:** 480  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.000) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8834 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1002 |
| Gradient Magnitude (Failure) | 1.1216 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5087 | 0.6677 |
| Neutral | 1.0969 | 0.9914 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1174 |
| Cosine Distance | 0.0079 |
| Clusters | 317 |
| Noise Fraction | 0.0218 |

---

## Checkpoint 560 episodes — step 155,248

**Episodes:** 560  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.100) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8784 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1670 |
| Gradient Magnitude (Failure) | 1.1059 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5738 | 0.4344 |
| Neutral | 1.0999 | 0.9044 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0980 |
| Cosine Distance | 0.0056 |
| Clusters | 306 |
| Noise Fraction | 0.0254 |

---

## Checkpoint 656 episodes — step 180,288

**Episodes:** 656  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.100) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8833 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0921 |
| Gradient Magnitude (Failure) | 1.0926 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1010 | 0.5567 |
| Neutral | 1.1298 | 0.9388 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0596 |
| Cosine Distance | 0.0022 |
| Clusters | 310 |
| Noise Fraction | 0.0456 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.004 | 0.009 |
| locked_door | 0.005 | -0.000 | 0.006 | 0.014 |
| open_door | 0.004 | 0.006 | 0.000 | 0.015 |
| target_ball | 0.009 | 0.014 | 0.015 | -0.000 |

---

## Checkpoint 752 episodes — step 205,328

**Episodes:** 752  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8938 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2302 |
| Gradient Magnitude (Failure) | 1.0914 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5467 | 0.8029 |
| Neutral | 1.0876 | 0.9234 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0699 |
| Cosine Distance | 0.0017 |
| Clusters | 315 |
| Noise Fraction | 0.0435 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.003 | 0.014 |
| locked_door | 0.005 | 0.000 | 0.006 | 0.028 |
| open_door | 0.003 | 0.006 | -0.000 | 0.019 |
| target_ball | 0.014 | 0.028 | 0.019 | -0.000 |

---

## Checkpoint 848 episodes — step 230,368

**Episodes:** 848  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9696 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1785 |
| Gradient Magnitude (Failure) | 1.1444 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8787 | 0.7282 |
| Neutral | 1.1774 | 0.9960 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1441 |
| Cosine Distance | 0.0049 |
| Clusters | 301 |
| Noise Fraction | 0.0633 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.007 |
| locked_door | 0.004 | 0.000 | 0.008 | 0.016 |
| open_door | 0.002 | 0.008 | 0.000 | 0.010 |
| target_ball | 0.007 | 0.016 | 0.010 | 0.000 |

---

## Checkpoint 944 episodes — step 255,408

**Episodes:** 944  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7639 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2197 |
| Gradient Magnitude (Failure) | 1.1673 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2215 | 0.5744 |
| Neutral | 1.2043 | 0.9839 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2146 |
| Cosine Distance | 0.0143 |
| Clusters | 317 |
| Noise Fraction | 0.0455 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.006 | 0.003 | 0.006 |
| locked_door | 0.006 | 0.000 | 0.010 | 0.021 |
| open_door | 0.003 | 0.010 | -0.000 | 0.010 |
| target_ball | 0.006 | 0.021 | 0.010 | 0.000 |

---

## Checkpoint 1,024 episodes — step 280,448

**Episodes:** 1,024  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8334 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2901 |
| Gradient Magnitude (Failure) | 1.2511 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3867 | 0.5544 |
| Neutral | 1.2321 | 0.9596 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1765 |
| Cosine Distance | 0.0099 |
| Clusters | 328 |
| Noise Fraction | 0.0485 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.005 | 0.002 | 0.005 |
| locked_door | 0.005 | -0.000 | 0.007 | 0.017 |
| open_door | 0.002 | 0.007 | 0.000 | 0.006 |
| target_ball | 0.005 | 0.017 | 0.006 | 0.000 |

---

## Checkpoint 1,120 episodes — step 305,488

**Episodes:** 1,120  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8732 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2610 |
| Gradient Magnitude (Failure) | 1.3479 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5848 | 0.6603 |
| Neutral | 1.2510 | 0.9393 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1551 |
| Cosine Distance | 0.0067 |
| Clusters | 311 |
| Noise Fraction | 0.0554 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.004 |
| locked_door | 0.003 | 0.000 | 0.004 | 0.012 |
| open_door | 0.002 | 0.004 | -0.000 | 0.006 |
| target_ball | 0.004 | 0.012 | 0.006 | -0.000 |

---

## Checkpoint 1,216 episodes — step 330,528

**Episodes:** 1,216  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8701 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2765 |
| Gradient Magnitude (Failure) | 1.5078 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2916 | 0.7044 |
| Neutral | 1.2716 | 0.9811 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1722 |
| Cosine Distance | 0.0076 |
| Clusters | 317 |
| Noise Fraction | 0.0603 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.007 |
| locked_door | 0.003 | 0.000 | 0.004 | 0.015 |
| open_door | 0.002 | 0.004 | 0.000 | 0.009 |
| target_ball | 0.007 | 0.015 | 0.009 | -0.000 |

---

## Checkpoint 1,312 episodes — step 355,568

**Episodes:** 1,312  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7341 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2905 |
| Gradient Magnitude (Failure) | 1.6956 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4039 | 0.4765 |
| Neutral | 1.2874 | 0.8867 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1833 |
| Cosine Distance | 0.0100 |
| Clusters | 323 |
| Noise Fraction | 0.0618 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.003 | 0.003 | 0.003 |
| locked_door | 0.003 | 0.000 | 0.004 | 0.006 |
| open_door | 0.003 | 0.004 | 0.000 | 0.006 |
| target_ball | 0.003 | 0.006 | 0.006 | 0.000 |

---

## Checkpoint 1,408 episodes — step 380,608

**Episodes:** 1,408  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7551 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2518 |
| Gradient Magnitude (Failure) | 1.9186 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7474 | 0.6594 |
| Neutral | 1.2711 | 0.9931 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2137 |
| Cosine Distance | 0.0140 |
| Clusters | 321 |
| Noise Fraction | 0.0368 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.007 |
| locked_door | 0.003 | 0.000 | 0.004 | 0.015 |
| open_door | 0.003 | 0.004 | -0.000 | 0.009 |
| target_ball | 0.007 | 0.015 | 0.009 | -0.000 |

---

## Checkpoint 1,488 episodes — step 405,648

**Episodes:** 1,488  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7024 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2538 |
| Gradient Magnitude (Failure) | 2.1371 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6775 | 0.6235 |
| Neutral | 1.2692 | 0.9889 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2515 |
| Cosine Distance | 0.0183 |
| Clusters | 330 |
| Noise Fraction | 0.0286 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.004 | 0.003 | 0.005 |
| locked_door | 0.004 | 0.000 | 0.006 | 0.013 |
| open_door | 0.003 | 0.006 | 0.000 | 0.008 |
| target_ball | 0.005 | 0.013 | 0.008 | 0.000 |

---

## Checkpoint 1,584 episodes — step 430,688

**Episodes:** 1,584  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6332 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2764 |
| Gradient Magnitude (Failure) | 2.6235 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6384 | 0.6992 |
| Neutral | 1.2749 | 0.9973 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2906 |
| Cosine Distance | 0.0280 |
| Clusters | 336 |
| Noise Fraction | 0.0296 |

---

## Checkpoint 1,680 episodes — step 455,728

**Episodes:** 1,680  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.100) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6266 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3304 |
| Gradient Magnitude (Failure) | 2.6949 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6271 | 0.6449 |
| Neutral | 1.3239 | 0.9937 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2944 |
| Cosine Distance | 0.0290 |
| Clusters | 345 |
| Noise Fraction | 0.0230 |

---

## Checkpoint 1,776 episodes — step 480,768

**Episodes:** 1,776  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.100) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7006 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3222 |
| Gradient Magnitude (Failure) | 2.5291 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5841 | 0.7052 |
| Neutral | 1.3331 | 0.9944 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2943 |
| Cosine Distance | 0.0318 |
| Clusters | 341 |
| Noise Fraction | 0.0178 |

---

## Checkpoint 1,872 episodes — step 505,808

**Episodes:** 1,872  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.100) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7526 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.5523 |
| Gradient Magnitude (Failure) | 2.3632 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7143 | 0.7649 |
| Neutral | 1.5472 | 0.9911 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2522 |
| Cosine Distance | 0.0238 |
| Clusters | 346 |
| Noise Fraction | 0.0186 |

---

## Checkpoint 1,952 episodes — step 530,848

**Episodes:** 1,952  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7971 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7469 |
| Gradient Magnitude (Failure) | 2.1530 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7609 | 0.7985 |
| Neutral | 1.7653 | 0.9929 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2108 |
| Cosine Distance | 0.0160 |
| Clusters | 336 |
| Noise Fraction | 0.0160 |

---

## Checkpoint 2,048 episodes — step 555,888

**Episodes:** 2,048  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8458 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8530 |
| Gradient Magnitude (Failure) | 1.8992 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.0729 | 0.6360 |
| Neutral | 1.8383 | 0.9957 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1392 |
| Cosine Distance | 0.0073 |
| Clusters | 338 |
| Noise Fraction | 0.0132 |

---

## Checkpoint 2,144 episodes — step 580,928

**Episodes:** 2,144  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8814 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8718 |
| Gradient Magnitude (Failure) | 1.8793 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.0170 | 0.6695 |
| Neutral | 1.8708 | 0.9707 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1663 |
| Cosine Distance | 0.0101 |
| Clusters | 341 |
| Noise Fraction | 0.0241 |

---

## Checkpoint 2,240 episodes — step 605,968

**Episodes:** 2,240  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8411 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4858 |
| Gradient Magnitude (Failure) | 1.8954 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.0378 | 0.6954 |
| Neutral | 1.4804 | 0.9975 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1633 |
| Cosine Distance | 0.0094 |
| Clusters | 323 |
| Noise Fraction | 0.0279 |

---

## Checkpoint 2,336 episodes — step 631,008

**Episodes:** 2,336  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8506 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4936 |
| Gradient Magnitude (Failure) | 1.6156 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6936 | 0.8337 |
| Neutral | 1.4744 | 0.9960 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1034 |
| Cosine Distance | 0.0040 |
| Clusters | 334 |
| Noise Fraction | 0.0341 |

---

## Checkpoint 2,416 episodes — step 656,048

**Episodes:** 2,416  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8576 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4694 |
| Gradient Magnitude (Failure) | 1.3383 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6512 | 0.8038 |
| Neutral | 1.4296 | 0.9872 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1150 |
| Cosine Distance | 0.0042 |
| Clusters | 322 |
| Noise Fraction | 0.0481 |

---

## Checkpoint 2,512 episodes — step 681,088

**Episodes:** 2,512  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9110 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3707 |
| Gradient Magnitude (Failure) | 1.4835 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4630 | 0.8136 |
| Neutral | 1.3709 | 0.9963 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1064 |
| Cosine Distance | 0.0036 |
| Clusters | 325 |
| Noise Fraction | 0.0439 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.010 | 0.009 | 0.017 |
| locked_door | 0.010 | 0.000 | 0.003 | 0.045 |
| open_door | 0.009 | 0.003 | -0.000 | 0.043 |
| target_ball | 0.017 | 0.045 | 0.043 | -0.000 |

---

## Checkpoint 2,608 episodes — step 706,128

**Episodes:** 2,608  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8891 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3234 |
| Gradient Magnitude (Failure) | 1.5002 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7415 | 0.6379 |
| Neutral | 1.3096 | 0.9921 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1173 |
| Cosine Distance | 0.0045 |
| Clusters | 302 |
| Noise Fraction | 0.0561 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.008 | 0.007 | 0.015 |
| locked_door | 0.008 | 0.000 | 0.003 | 0.039 |
| open_door | 0.007 | 0.003 | 0.000 | 0.036 |
| target_ball | 0.015 | 0.039 | 0.036 | 0.000 |

---

## Checkpoint 2,704 episodes — step 731,168

**Episodes:** 2,704  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8758 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3519 |
| Gradient Magnitude (Failure) | 1.5630 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7016 | 0.6204 |
| Neutral | 1.4002 | 0.9349 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0853 |
| Cosine Distance | 0.0034 |
| Clusters | 319 |
| Noise Fraction | 0.0377 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.005 | 0.005 | 0.017 |
| locked_door | 0.005 | -0.000 | 0.003 | 0.035 |
| open_door | 0.005 | 0.003 | 0.000 | 0.032 |
| target_ball | 0.017 | 0.035 | 0.032 | 0.000 |

---

## Checkpoint 2,800 episodes — step 756,208

**Episodes:** 2,800  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8777 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3700 |
| Gradient Magnitude (Failure) | 1.6497 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8899 | 0.5622 |
| Neutral | 1.3627 | 0.9929 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0910 |
| Cosine Distance | 0.0042 |
| Clusters | 321 |
| Noise Fraction | 0.0289 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.005 | 0.006 |
| locked_door | 0.005 | 0.000 | 0.003 | 0.016 |
| open_door | 0.005 | 0.003 | 0.000 | 0.013 |
| target_ball | 0.006 | 0.016 | 0.013 | 0.000 |

---

## Checkpoint 2,880 episodes — step 781,248

**Episodes:** 2,880  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8294 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.5049 |
| Gradient Magnitude (Failure) | 1.6288 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7241 | 0.7338 |
| Neutral | 1.4085 | 0.8360 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0891 |
| Cosine Distance | 0.0040 |
| Clusters | 330 |
| Noise Fraction | 0.0473 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.004 | 0.008 |
| locked_door | 0.005 | -0.000 | 0.003 | 0.017 |
| open_door | 0.004 | 0.003 | 0.000 | 0.012 |
| target_ball | 0.008 | 0.017 | 0.012 | 0.000 |

---

## Checkpoint 2,976 episodes — step 806,288

**Episodes:** 2,976  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9641 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.5146 |
| Gradient Magnitude (Failure) | 1.6419 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5021 | 0.7829 |
| Neutral | 1.5083 | 0.9949 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0693 |
| Cosine Distance | 0.0020 |
| Clusters | 321 |
| Noise Fraction | 0.0437 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.006 | 0.005 | 0.010 |
| locked_door | 0.006 | 0.000 | 0.005 | 0.019 |
| open_door | 0.005 | 0.005 | 0.000 | 0.016 |
| target_ball | 0.010 | 0.019 | 0.016 | 0.000 |

---

## Checkpoint 3,072 episodes — step 831,328

**Episodes:** 3,072  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9389 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4960 |
| Gradient Magnitude (Failure) | 1.6134 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6107 | 0.8103 |
| Neutral | 1.5131 | 0.9767 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0480 |
| Cosine Distance | 0.0010 |
| Clusters | 322 |
| Noise Fraction | 0.0434 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.004 | 0.011 |
| locked_door | 0.005 | 0.000 | 0.006 | 0.022 |
| open_door | 0.004 | 0.006 | 0.000 | 0.018 |
| target_ball | 0.011 | 0.022 | 0.018 | 0.000 |

---

## Checkpoint 3,168 episodes — step 856,368

**Episodes:** 3,168  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9263 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4078 |
| Gradient Magnitude (Failure) | 1.6304 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4380 | 0.7745 |
| Neutral | 1.4038 | 0.9935 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0749 |
| Cosine Distance | 0.0015 |
| Clusters | 322 |
| Noise Fraction | 0.0481 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.018 |
| locked_door | 0.004 | 0.000 | 0.004 | 0.024 |
| open_door | 0.002 | 0.004 | -0.000 | 0.021 |
| target_ball | 0.018 | 0.024 | 0.021 | 0.000 |

---

## Checkpoint 3,264 episodes — step 881,408

**Episodes:** 3,264  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9152 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3963 |
| Gradient Magnitude (Failure) | 1.6775 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5971 | 0.7411 |
| Neutral | 1.3846 | 0.9804 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0948 |
| Cosine Distance | 0.0023 |
| Clusters | 339 |
| Noise Fraction | 0.0533 |

---

## Checkpoint 3,344 episodes — step 906,448

**Episodes:** 3,344  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9143 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4245 |
| Gradient Magnitude (Failure) | 1.6903 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.0615 | 0.7201 |
| Neutral | 1.4103 | 0.9792 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1246 |
| Cosine Distance | 0.0034 |
| Clusters | 323 |
| Noise Fraction | 0.0605 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.006 | 0.002 | 0.013 |
| locked_door | 0.006 | 0.000 | 0.004 | 0.026 |
| open_door | 0.002 | 0.004 | 0.000 | 0.016 |
| target_ball | 0.013 | 0.026 | 0.016 | 0.000 |

---

## Checkpoint 3,440 episodes — step 931,488

**Episodes:** 3,440  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9347 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4371 |
| Gradient Magnitude (Failure) | 1.8228 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8011 | 0.7158 |
| Neutral | 1.4788 | 0.9606 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1145 |
| Cosine Distance | 0.0031 |
| Clusters | 327 |
| Noise Fraction | 0.0359 |

---

## Checkpoint 3,536 episodes — step 956,528

**Episodes:** 3,536  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9268 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4239 |
| Gradient Magnitude (Failure) | 1.8548 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8619 | 0.7294 |
| Neutral | 1.4077 | 0.9918 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1609 |
| Cosine Distance | 0.0053 |
| Clusters | 317 |
| Noise Fraction | 0.0413 |

---

## Checkpoint 3,632 episodes — step 981,568

**Episodes:** 3,632  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8553 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3505 |
| Gradient Magnitude (Failure) | 1.8737 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6745 | 0.6842 |
| Neutral | 1.3634 | 0.9964 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1891 |
| Cosine Distance | 0.0072 |
| Clusters | 314 |
| Noise Fraction | 0.0725 |

---

## Checkpoint 3,728 episodes — step 1,006,608

**Episodes:** 3,728  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8468 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2867 |
| Gradient Magnitude (Failure) | 1.7479 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5347 | 0.6784 |
| Neutral | 1.2684 | 0.9474 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1410 |
| Cosine Distance | 0.0041 |
| Clusters | 303 |
| Noise Fraction | 0.0515 |

---

## Checkpoint 3,808 episodes — step 1,031,648

**Episodes:** 3,808  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8116 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3632 |
| Gradient Magnitude (Failure) | 1.7406 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7111 | 0.4953 |
| Neutral | 1.3024 | 0.9581 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1131 |
| Cosine Distance | 0.0025 |
| Clusters | 310 |
| Noise Fraction | 0.0381 |

---

## Checkpoint 3,904 episodes — step 1,056,688

**Episodes:** 3,904  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8982 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3344 |
| Gradient Magnitude (Failure) | 1.6860 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7056 | 0.6527 |
| Neutral | 1.3286 | 0.9981 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0682 |
| Cosine Distance | 0.0013 |
| Clusters | 311 |
| Noise Fraction | 0.0449 |

---

## Checkpoint 4,000 episodes — step 1,081,728

**Episodes:** 4,000  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9134 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4144 |
| Gradient Magnitude (Failure) | 1.7580 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6253 | 0.7159 |
| Neutral | 1.4401 | 0.9807 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0925 |
| Cosine Distance | 0.0022 |
| Clusters | 314 |
| Noise Fraction | 0.0217 |

---

## Checkpoint 4,096 episodes — step 1,106,768

**Episodes:** 4,096  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7720 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3530 |
| Gradient Magnitude (Failure) | 1.8040 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8740 | 0.5622 |
| Neutral | 1.3529 | 0.9374 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0938 |
| Cosine Distance | 0.0027 |
| Clusters | 316 |
| Noise Fraction | 0.0230 |

---

## Checkpoint 4,176 episodes — step 1,131,808

**Episodes:** 4,176  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8112 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3592 |
| Gradient Magnitude (Failure) | 1.6566 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6221 | 0.6941 |
| Neutral | 1.3361 | 0.9936 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1002 |
| Cosine Distance | 0.0030 |
| Clusters | 313 |
| Noise Fraction | 0.0432 |

---

## Checkpoint 4,272 episodes — step 1,156,848

**Episodes:** 4,272  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8277 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4466 |
| Gradient Magnitude (Failure) | 1.6258 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3591 | 0.7344 |
| Neutral | 1.4205 | 0.9688 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0853 |
| Cosine Distance | 0.0028 |
| Clusters | 326 |
| Noise Fraction | 0.0351 |

---

## Checkpoint 4,368 episodes — step 1,181,888

**Episodes:** 4,368  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8457 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3473 |
| Gradient Magnitude (Failure) | 1.5979 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5381 | 0.6643 |
| Neutral | 1.3445 | 0.9931 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0786 |
| Cosine Distance | 0.0024 |
| Clusters | 292 |
| Noise Fraction | 0.0283 |

---

## Checkpoint 4,464 episodes — step 1,206,928

**Episodes:** 4,464  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8042 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2751 |
| Gradient Magnitude (Failure) | 1.5215 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4072 | 0.7199 |
| Neutral | 1.2027 | 0.9203 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1094 |
| Cosine Distance | 0.0031 |
| Clusters | 323 |
| Noise Fraction | 0.0331 |

---

## Checkpoint 4,560 episodes — step 1,231,968

**Episodes:** 4,560  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9078 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1516 |
| Gradient Magnitude (Failure) | 1.4292 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5328 | 0.6997 |
| Neutral | 1.1625 | 0.9809 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1860 |
| Cosine Distance | 0.0064 |
| Clusters | 315 |
| Noise Fraction | 0.0380 |

---

## Checkpoint 4,640 episodes — step 1,257,008

**Episodes:** 4,640  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9186 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1728 |
| Gradient Magnitude (Failure) | 1.3937 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7322 | 0.6698 |
| Neutral | 1.1672 | 0.9918 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1999 |
| Cosine Distance | 0.0065 |
| Clusters | 307 |
| Noise Fraction | 0.0331 |

---

## Checkpoint 4,736 episodes — step 1,282,048

**Episodes:** 4,736  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9304 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2429 |
| Gradient Magnitude (Failure) | 1.4288 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6849 | 0.7341 |
| Neutral | 1.1927 | 0.9659 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1609 |
| Cosine Distance | 0.0046 |
| Clusters | 299 |
| Noise Fraction | 0.0356 |

---

## Checkpoint 4,832 episodes — step 1,307,088

**Episodes:** 4,832  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9622 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2137 |
| Gradient Magnitude (Failure) | 1.4087 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3037 | 0.7126 |
| Neutral | 1.1960 | 0.9945 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1207 |
| Cosine Distance | 0.0029 |
| Clusters | 308 |
| Noise Fraction | 0.0321 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.037 |
| locked_door | 0.003 | -0.000 | 0.002 | 0.030 |
| open_door | 0.003 | 0.002 | 0.000 | 0.038 |
| target_ball | 0.037 | 0.030 | 0.038 | 0.000 |

---

## Checkpoint 4,928 episodes — step 1,332,128

**Episodes:** 4,928  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9619 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2321 |
| Gradient Magnitude (Failure) | 1.4632 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1277 | 0.7557 |
| Neutral | 1.2266 | 0.9977 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1072 |
| Cosine Distance | 0.0027 |
| Clusters | 304 |
| Noise Fraction | 0.0327 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.006 |
| locked_door | 0.003 | -0.000 | 0.003 | 0.009 |
| open_door | 0.003 | 0.003 | 0.000 | 0.008 |
| target_ball | 0.006 | 0.009 | 0.008 | -0.000 |

---

## Checkpoint 5,024 episodes — step 1,357,168

**Episodes:** 5,024  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9257 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1786 |
| Gradient Magnitude (Failure) | 1.4813 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8355 | 0.5625 |
| Neutral | 1.1723 | 0.9969 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1501 |
| Cosine Distance | 0.0051 |
| Clusters | 310 |
| Noise Fraction | 0.0389 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.006 |
| locked_door | 0.003 | 0.000 | 0.003 | 0.009 |
| open_door | 0.002 | 0.003 | 0.000 | 0.007 |
| target_ball | 0.006 | 0.009 | 0.007 | -0.000 |

---

## Checkpoint 5,104 episodes — step 1,382,208

**Episodes:** 5,104  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8669 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1395 |
| Gradient Magnitude (Failure) | 1.5986 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3797 | 0.6875 |
| Neutral | 1.1537 | 0.9685 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2335 |
| Cosine Distance | 0.0109 |
| Clusters | 306 |
| Noise Fraction | 0.0555 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.005 |
| locked_door | 0.003 | 0.000 | 0.003 | 0.009 |
| open_door | 0.002 | 0.003 | 0.000 | 0.005 |
| target_ball | 0.005 | 0.009 | 0.005 | 0.000 |

---

## Checkpoint 5,200 episodes — step 1,407,248

**Episodes:** 5,200  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8758 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1802 |
| Gradient Magnitude (Failure) | 1.6205 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2648 | 0.7536 |
| Neutral | 1.1721 | 0.9801 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2886 |
| Cosine Distance | 0.0154 |
| Clusters | 301 |
| Noise Fraction | 0.0602 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.007 |
| locked_door | 0.004 | -0.000 | 0.004 | 0.012 |
| open_door | 0.002 | 0.004 | -0.000 | 0.005 |
| target_ball | 0.007 | 0.012 | 0.005 | 0.000 |

---

## Checkpoint 5,296 episodes — step 1,432,288

**Episodes:** 5,296  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8826 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2260 |
| Gradient Magnitude (Failure) | 1.6778 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2292 | 0.7655 |
| Neutral | 1.2019 | 0.9455 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2971 |
| Cosine Distance | 0.0157 |
| Clusters | 301 |
| Noise Fraction | 0.0448 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.003 | 0.009 |
| locked_door | 0.005 | 0.000 | 0.005 | 0.018 |
| open_door | 0.003 | 0.005 | -0.000 | 0.007 |
| target_ball | 0.009 | 0.018 | 0.007 | 0.000 |

---

## Checkpoint 5,392 episodes — step 1,457,328

**Episodes:** 5,392  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8781 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2096 |
| Gradient Magnitude (Failure) | 1.6991 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3224 | 0.7401 |
| Neutral | 1.1924 | 0.9923 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2742 |
| Cosine Distance | 0.0143 |
| Clusters | 293 |
| Noise Fraction | 0.0582 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.005 | 0.002 | 0.007 |
| locked_door | 0.005 | 0.000 | 0.006 | 0.016 |
| open_door | 0.002 | 0.006 | -0.000 | 0.005 |
| target_ball | 0.007 | 0.016 | 0.005 | 0.000 |

---

## Checkpoint 5,488 episodes — step 1,482,368

**Episodes:** 5,488  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8846 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1764 |
| Gradient Magnitude (Failure) | 1.6161 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4883 | 0.7363 |
| Neutral | 1.1697 | 0.9983 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2746 |
| Cosine Distance | 0.0128 |
| Clusters | 311 |
| Noise Fraction | 0.0541 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.006 |
| locked_door | 0.004 | 0.000 | 0.005 | 0.014 |
| open_door | 0.002 | 0.005 | 0.000 | 0.005 |
| target_ball | 0.006 | 0.014 | 0.005 | 0.000 |

---

## Checkpoint 5,568 episodes — step 1,507,408

**Episodes:** 5,568  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8550 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1511 |
| Gradient Magnitude (Failure) | 1.5664 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6599 | 0.5993 |
| Neutral | 1.1892 | 0.8766 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2707 |
| Cosine Distance | 0.0128 |
| Clusters | 305 |
| Noise Fraction | 0.0559 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.010 |
| locked_door | 0.004 | 0.000 | 0.005 | 0.019 |
| open_door | 0.002 | 0.005 | 0.000 | 0.011 |
| target_ball | 0.010 | 0.019 | 0.011 | 0.000 |

---

## Checkpoint 5,664 episodes — step 1,532,448

**Episodes:** 5,664  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8101 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1593 |
| Gradient Magnitude (Failure) | 1.5885 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2483 | 0.5795 |
| Neutral | 1.1541 | 0.9995 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2631 |
| Cosine Distance | 0.0100 |
| Clusters | 293 |
| Noise Fraction | 0.0432 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.006 |
| locked_door | 0.004 | 0.000 | 0.004 | 0.015 |
| open_door | 0.002 | 0.004 | 0.000 | 0.008 |
| target_ball | 0.006 | 0.015 | 0.008 | 0.000 |

---

## Checkpoint 5,760 episodes — step 1,557,488

**Episodes:** 5,760  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9169 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1686 |
| Gradient Magnitude (Failure) | 1.4249 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4690 | 0.6355 |
| Neutral | 1.1824 | 0.9588 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2004 |
| Cosine Distance | 0.0055 |
| Clusters | 304 |
| Noise Fraction | 0.0452 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.009 |
| locked_door | 0.004 | 0.000 | 0.003 | 0.017 |
| open_door | 0.002 | 0.003 | 0.000 | 0.011 |
| target_ball | 0.009 | 0.017 | 0.011 | 0.000 |

---

## Checkpoint 5,856 episodes — step 1,582,528

**Episodes:** 5,856  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8797 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2152 |
| Gradient Magnitude (Failure) | 1.4869 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2698 | 0.7088 |
| Neutral | 1.2079 | 0.9935 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1972 |
| Cosine Distance | 0.0054 |
| Clusters | 312 |
| Noise Fraction | 0.0536 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.009 |
| locked_door | 0.004 | 0.000 | 0.002 | 0.016 |
| open_door | 0.002 | 0.002 | 0.000 | 0.011 |
| target_ball | 0.009 | 0.016 | 0.011 | 0.000 |

---

## Checkpoint 5,952 episodes — step 1,607,568

**Episodes:** 5,952  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9138 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1849 |
| Gradient Magnitude (Failure) | 1.4696 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2435 | 0.6661 |
| Neutral | 1.1828 | 0.9986 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1750 |
| Cosine Distance | 0.0047 |
| Clusters | 309 |
| Noise Fraction | 0.0591 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.004 | 0.002 | 0.017 |
| locked_door | 0.004 | 0.000 | 0.003 | 0.027 |
| open_door | 0.002 | 0.003 | -0.000 | 0.023 |
| target_ball | 0.017 | 0.027 | 0.023 | 0.000 |

---

## Checkpoint 6,032 episodes — step 1,632,608

**Episodes:** 6,032  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9176 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1999 |
| Gradient Magnitude (Failure) | 1.4403 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3039 | 0.6820 |
| Neutral | 1.1862 | 0.9562 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1931 |
| Cosine Distance | 0.0047 |
| Clusters | 314 |
| Noise Fraction | 0.0488 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | -0.000 | 0.003 | 0.019 |
| open_door | 0.002 | 0.003 | 0.000 | 0.015 |
| target_ball | 0.010 | 0.019 | 0.015 | -0.000 |

---

## Checkpoint 6,128 episodes — step 1,657,648

**Episodes:** 6,128  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8898 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1739 |
| Gradient Magnitude (Failure) | 1.4798 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4174 | 0.6726 |
| Neutral | 1.1525 | 0.9460 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2049 |
| Cosine Distance | 0.0055 |
| Clusters | 310 |
| Noise Fraction | 0.0558 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.017 |
| open_door | 0.002 | 0.002 | 0.000 | 0.014 |
| target_ball | 0.010 | 0.017 | 0.014 | 0.000 |

---

## Checkpoint 6,224 episodes — step 1,682,688

**Episodes:** 6,224  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9055 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1671 |
| Gradient Magnitude (Failure) | 1.5062 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1905 | 0.6281 |
| Neutral | 1.1786 | 0.9924 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2347 |
| Cosine Distance | 0.0055 |
| Clusters | 317 |
| Noise Fraction | 0.0666 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.003 | 0.002 | 0.009 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.017 |
| open_door | 0.002 | 0.002 | -0.000 | 0.013 |
| target_ball | 0.009 | 0.017 | 0.013 | 0.000 |

---

## Checkpoint 6,320 episodes — step 1,707,728

**Episodes:** 6,320  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9028 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1726 |
| Gradient Magnitude (Failure) | 1.5268 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4052 | 0.5849 |
| Neutral | 1.1717 | 0.9555 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2225 |
| Cosine Distance | 0.0051 |
| Clusters | 325 |
| Noise Fraction | 0.0696 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | -0.000 | 0.002 | 0.017 |
| open_door | 0.002 | 0.002 | 0.000 | 0.014 |
| target_ball | 0.010 | 0.017 | 0.014 | 0.000 |

---

## Checkpoint 6,416 episodes — step 1,732,768

**Episodes:** 6,416  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9107 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2038 |
| Gradient Magnitude (Failure) | 1.5494 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7283 | 0.7051 |
| Neutral | 1.1764 | 0.9906 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2537 |
| Cosine Distance | 0.0059 |
| Clusters | 319 |
| Noise Fraction | 0.0599 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.012 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.022 |
| open_door | 0.002 | 0.002 | 0.000 | 0.018 |
| target_ball | 0.012 | 0.022 | 0.018 | 0.000 |

---

## Checkpoint 6,496 episodes — step 1,757,808

**Episodes:** 6,496  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9018 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1346 |
| Gradient Magnitude (Failure) | 1.4302 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6822 | 0.5885 |
| Neutral | 1.1312 | 0.9883 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2296 |
| Cosine Distance | 0.0053 |
| Clusters | 314 |
| Noise Fraction | 0.0557 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | -0.000 | 0.002 | 0.017 |
| open_door | 0.002 | 0.002 | 0.000 | 0.014 |
| target_ball | 0.010 | 0.017 | 0.014 | 0.000 |

---

## Checkpoint 6,592 episodes — step 1,782,848

**Episodes:** 6,592  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8555 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1545 |
| Gradient Magnitude (Failure) | 1.5156 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3088 | 0.6103 |
| Neutral | 1.1274 | 0.9808 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2124 |
| Cosine Distance | 0.0053 |
| Clusters | 302 |
| Noise Fraction | 0.0678 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | -0.000 | 0.002 | 0.017 |
| open_door | 0.002 | 0.002 | 0.000 | 0.013 |
| target_ball | 0.010 | 0.017 | 0.013 | 0.000 |

---

## Checkpoint 6,688 episodes — step 1,807,888

**Episodes:** 6,688  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8642 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1450 |
| Gradient Magnitude (Failure) | 1.5454 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3039 | 0.6588 |
| Neutral | 1.1419 | 0.9983 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2087 |
| Cosine Distance | 0.0050 |
| Clusters | 293 |
| Noise Fraction | 0.0646 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.009 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.015 |
| open_door | 0.002 | 0.002 | 0.000 | 0.011 |
| target_ball | 0.009 | 0.015 | 0.011 | 0.000 |

---

## Checkpoint 6,784 episodes — step 1,832,928

**Episodes:** 6,784  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8907 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1489 |
| Gradient Magnitude (Failure) | 1.4973 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3493 | 0.6622 |
| Neutral | 1.1350 | 0.9942 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1866 |
| Cosine Distance | 0.0043 |
| Clusters | 299 |
| Noise Fraction | 0.0623 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.008 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.011 |
| open_door | 0.002 | 0.002 | 0.000 | 0.008 |
| target_ball | 0.008 | 0.011 | 0.008 | 0.000 |

---

## Checkpoint 6,880 episodes — step 1,857,968

**Episodes:** 6,880  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9284 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1370 |
| Gradient Magnitude (Failure) | 1.2952 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5127 | 0.6519 |
| Neutral | 1.1277 | 0.9975 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1478 |
| Cosine Distance | 0.0034 |
| Clusters | 296 |
| Noise Fraction | 0.0753 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | 0.000 | 0.003 | 0.015 |
| open_door | 0.002 | 0.003 | 0.000 | 0.010 |
| target_ball | 0.010 | 0.015 | 0.010 | 0.000 |

---

## Checkpoint 6,960 episodes — step 1,883,008

**Episodes:** 6,960  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9244 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1370 |
| Gradient Magnitude (Failure) | 1.3749 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3529 | 0.6168 |
| Neutral | 1.1736 | 0.9777 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1012 |
| Cosine Distance | 0.0028 |
| Clusters | 305 |
| Noise Fraction | 0.0545 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.009 |
| locked_door | 0.004 | -0.000 | 0.002 | 0.014 |
| open_door | 0.002 | 0.002 | 0.000 | 0.010 |
| target_ball | 0.009 | 0.014 | 0.010 | 0.000 |

---

## Checkpoint 7,056 episodes — step 1,908,048

**Episodes:** 7,056  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8941 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1578 |
| Gradient Magnitude (Failure) | 1.4077 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5148 | 0.6763 |
| Neutral | 1.1514 | 0.9968 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1029 |
| Cosine Distance | 0.0031 |
| Clusters | 307 |
| Noise Fraction | 0.0793 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.004 | 0.002 | 0.010 |
| locked_door | 0.004 | 0.000 | 0.003 | 0.012 |
| open_door | 0.002 | 0.003 | 0.000 | 0.008 |
| target_ball | 0.010 | 0.012 | 0.008 | 0.000 |

---

## Checkpoint 7,152 episodes — step 1,933,088

**Episodes:** 7,152  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9490 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1645 |
| Gradient Magnitude (Failure) | 1.2992 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6812 | 0.6994 |
| Neutral | 1.1319 | 0.9582 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1658 |
| Cosine Distance | 0.0044 |
| Clusters | 306 |
| Noise Fraction | 0.0812 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.007 |
| locked_door | 0.004 | 0.000 | 0.003 | 0.011 |
| open_door | 0.002 | 0.003 | -0.000 | 0.006 |
| target_ball | 0.007 | 0.011 | 0.006 | 0.000 |

---

## Checkpoint 7,248 episodes — step 1,958,128

**Episodes:** 7,248  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9006 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1328 |
| Gradient Magnitude (Failure) | 1.3732 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6318 | 0.5531 |
| Neutral | 1.1304 | 0.9950 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1816 |
| Cosine Distance | 0.0054 |
| Clusters | 305 |
| Noise Fraction | 0.0697 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.006 |
| locked_door | 0.004 | 0.000 | 0.003 | 0.008 |
| open_door | 0.002 | 0.003 | 0.000 | 0.004 |
| target_ball | 0.006 | 0.008 | 0.004 | 0.000 |

---

## Checkpoint 7,344 episodes — step 1,983,168

**Episodes:** 7,344  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9268 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1178 |
| Gradient Magnitude (Failure) | 1.3245 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4487 | 0.6597 |
| Neutral | 1.1496 | 0.9802 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1749 |
| Cosine Distance | 0.0051 |
| Clusters | 303 |
| Noise Fraction | 0.0686 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.007 |
| locked_door | 0.004 | 0.000 | 0.002 | 0.007 |
| open_door | 0.002 | 0.002 | 0.000 | 0.004 |
| target_ball | 0.007 | 0.007 | 0.004 | -0.000 |

---

## Checkpoint 7,424 episodes — step 2,008,208

**Episodes:** 7,424  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9168 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0963 |
| Gradient Magnitude (Failure) | 1.2820 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6321 | 0.5842 |
| Neutral | 1.0928 | 0.9965 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1915 |
| Cosine Distance | 0.0066 |
| Clusters | 286 |
| Noise Fraction | 0.0530 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.006 |
| locked_door | 0.004 | 0.000 | 0.002 | 0.006 |
| open_door | 0.002 | 0.002 | -0.000 | 0.003 |
| target_ball | 0.006 | 0.006 | 0.003 | -0.000 |

---

## Checkpoint 7,520 episodes — step 2,033,248

**Episodes:** 7,520  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8828 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1081 |
| Gradient Magnitude (Failure) | 1.3844 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6857 | 0.5544 |
| Neutral | 1.1432 | 0.9401 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1701 |
| Cosine Distance | 0.0069 |
| Clusters | 319 |
| Noise Fraction | 0.0629 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.005 | 0.002 | 0.006 |
| locked_door | 0.005 | 0.000 | 0.003 | 0.007 |
| open_door | 0.002 | 0.003 | 0.000 | 0.003 |
| target_ball | 0.006 | 0.007 | 0.003 | 0.000 |

---

## Checkpoint 7,616 episodes — step 2,058,288

**Episodes:** 7,616  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8579 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0929 |
| Gradient Magnitude (Failure) | 1.4182 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7585 | 0.5655 |
| Neutral | 1.0952 | 0.9744 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1755 |
| Cosine Distance | 0.0079 |
| Clusters | 303 |
| Noise Fraction | 0.0730 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.007 | 0.002 | 0.009 |
| locked_door | 0.007 | -0.000 | 0.004 | 0.008 |
| open_door | 0.002 | 0.004 | -0.000 | 0.004 |
| target_ball | 0.009 | 0.008 | 0.004 | 0.000 |

---

## Checkpoint 7,712 episodes — step 2,083,328

**Episodes:** 7,712  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8424 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1184 |
| Gradient Magnitude (Failure) | 1.3194 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3887 | 0.6677 |
| Neutral | 1.0800 | 0.9807 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1791 |
| Cosine Distance | 0.0075 |
| Clusters | 310 |
| Noise Fraction | 0.0638 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.007 | 0.002 | 0.009 |
| locked_door | 0.007 | 0.000 | 0.005 | 0.009 |
| open_door | 0.002 | 0.005 | 0.000 | 0.005 |
| target_ball | 0.009 | 0.009 | 0.005 | 0.000 |

---

## Checkpoint 7,808 episodes — step 2,108,368

**Episodes:** 7,808  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9079 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0994 |
| Gradient Magnitude (Failure) | 1.3475 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4656 | 0.6590 |
| Neutral | 1.0864 | 0.9868 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1688 |
| Cosine Distance | 0.0066 |
| Clusters | 300 |
| Noise Fraction | 0.0612 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.005 | 0.002 | 0.007 |
| locked_door | 0.005 | 0.000 | 0.003 | 0.008 |
| open_door | 0.002 | 0.003 | -0.000 | 0.005 |
| target_ball | 0.007 | 0.008 | 0.005 | 0.000 |

---

## Checkpoint 7,888 episodes — step 2,133,408

**Episodes:** 7,888  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9421 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1103 |
| Gradient Magnitude (Failure) | 1.2604 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8511 | 0.5585 |
| Neutral | 1.1038 | 0.9964 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1344 |
| Cosine Distance | 0.0039 |
| Clusters | 319 |
| Noise Fraction | 0.0448 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.003 | 0.009 |
| locked_door | 0.004 | -0.000 | 0.002 | 0.008 |
| open_door | 0.003 | 0.002 | 0.000 | 0.007 |
| target_ball | 0.009 | 0.008 | 0.007 | 0.000 |

---

## Checkpoint 7,984 episodes — step 2,158,448

**Episodes:** 7,984  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8863 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1245 |
| Gradient Magnitude (Failure) | 1.3102 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6384 | 0.5461 |
| Neutral | 1.1156 | 0.9952 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1535 |
| Cosine Distance | 0.0033 |
| Clusters | 310 |
| Noise Fraction | 0.0581 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.008 |
| locked_door | 0.003 | -0.000 | 0.002 | 0.008 |
| open_door | 0.002 | 0.002 | 0.000 | 0.005 |
| target_ball | 0.008 | 0.008 | 0.005 | 0.000 |

---

## Checkpoint 8,080 episodes — step 2,183,488

**Episodes:** 8,080  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9430 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1489 |
| Gradient Magnitude (Failure) | 1.1921 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6050 | 0.5747 |
| Neutral | 1.1990 | 0.9515 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1440 |
| Cosine Distance | 0.0026 |
| Clusters | 314 |
| Noise Fraction | 0.0618 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.004 | 0.002 | 0.008 |
| locked_door | 0.004 | 0.000 | 0.002 | 0.009 |
| open_door | 0.002 | 0.002 | 0.000 | 0.006 |
| target_ball | 0.008 | 0.009 | 0.006 | 0.000 |

---

## Checkpoint 8,176 episodes — step 2,208,528

**Episodes:** 8,176  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9087 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1668 |
| Gradient Magnitude (Failure) | 1.2370 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5735 | 0.6415 |
| Neutral | 1.1517 | 0.9940 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1480 |
| Cosine Distance | 0.0027 |
| Clusters | 303 |
| Noise Fraction | 0.0696 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.010 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.013 |
| open_door | 0.002 | 0.002 | -0.000 | 0.008 |
| target_ball | 0.010 | 0.013 | 0.008 | 0.000 |

---

## Checkpoint 8,272 episodes — step 2,233,568

**Episodes:** 8,272  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8484 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1705 |
| Gradient Magnitude (Failure) | 1.4451 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8050 | 0.5477 |
| Neutral | 1.2085 | 0.9589 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2475 |
| Cosine Distance | 0.0053 |
| Clusters | 316 |
| Noise Fraction | 0.0495 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.001 | 0.005 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.009 |
| open_door | 0.001 | 0.002 | 0.000 | 0.005 |
| target_ball | 0.005 | 0.009 | 0.005 | 0.000 |

---

## Checkpoint 8,352 episodes — step 2,258,608

**Episodes:** 8,352  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8497 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1477 |
| Gradient Magnitude (Failure) | 1.5386 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5160 | 0.6175 |
| Neutral | 1.1498 | 0.9950 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2824 |
| Cosine Distance | 0.0080 |
| Clusters | 316 |
| Noise Fraction | 0.0482 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.002 | 0.006 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.009 |
| open_door | 0.002 | 0.002 | -0.000 | 0.004 |
| target_ball | 0.006 | 0.009 | 0.004 | 0.000 |

---

## Checkpoint 8,448 episodes — step 2,283,648

**Episodes:** 8,448  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8389 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2058 |
| Gradient Magnitude (Failure) | 1.5411 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1322 | 0.6202 |
| Neutral | 1.1311 | 0.8828 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1847 |
| Cosine Distance | 0.0054 |
| Clusters | 306 |
| Noise Fraction | 0.0673 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.007 |
| locked_door | 0.002 | 0.000 | 0.003 | 0.007 |
| open_door | 0.003 | 0.003 | 0.000 | 0.004 |
| target_ball | 0.007 | 0.007 | 0.004 | -0.000 |

---

## Checkpoint 8,544 episodes — step 2,308,688

**Episodes:** 8,544  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9060 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1405 |
| Gradient Magnitude (Failure) | 1.4522 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8199 | 0.6325 |
| Neutral | 1.1425 | 0.9587 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1437 |
| Cosine Distance | 0.0039 |
| Clusters | 303 |
| Noise Fraction | 0.0673 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.007 |
| locked_door | 0.003 | 0.000 | 0.003 | 0.010 |
| open_door | 0.003 | 0.003 | 0.000 | 0.005 |
| target_ball | 0.007 | 0.010 | 0.005 | 0.000 |

---

## Checkpoint 8,640 episodes — step 2,333,728

**Episodes:** 8,640  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9481 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1221 |
| Gradient Magnitude (Failure) | 1.2879 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3803 | 0.6495 |
| Neutral | 1.1100 | 0.9955 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0774 |
| Cosine Distance | 0.0019 |
| Clusters | 306 |
| Noise Fraction | 0.0620 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.007 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.006 |
| open_door | 0.003 | 0.002 | 0.000 | 0.003 |
| target_ball | 0.007 | 0.006 | 0.003 | 0.000 |

---

## Checkpoint 8,736 episodes — step 2,358,768

**Episodes:** 8,736  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8632 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2137 |
| Gradient Magnitude (Failure) | 1.2626 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3349 | 0.6775 |
| Neutral | 1.2183 | 0.6498 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0806 |
| Cosine Distance | 0.0024 |
| Clusters | 304 |
| Noise Fraction | 0.0655 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.004 | 0.003 | 0.010 |
| locked_door | 0.004 | 0.000 | 0.001 | 0.008 |
| open_door | 0.003 | 0.001 | 0.000 | 0.006 |
| target_ball | 0.010 | 0.008 | 0.006 | 0.000 |

---

## Checkpoint 8,816 episodes — step 2,383,808

**Episodes:** 8,816  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9445 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1091 |
| Gradient Magnitude (Failure) | 1.2847 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1591 | 0.6604 |
| Neutral | 1.1064 | 0.9876 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0941 |
| Cosine Distance | 0.0030 |
| Clusters | 297 |
| Noise Fraction | 0.0538 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.011 |
| locked_door | 0.003 | 0.000 | 0.002 | 0.008 |
| open_door | 0.003 | 0.002 | 0.000 | 0.006 |
| target_ball | 0.011 | 0.008 | 0.006 | 0.000 |

---

## Checkpoint 8,912 episodes — step 2,408,848

**Episodes:** 8,912  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7360 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1944 |
| Gradient Magnitude (Failure) | 1.3282 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3421 | 0.5497 |
| Neutral | 1.1157 | 0.9692 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1153 |
| Cosine Distance | 0.0036 |
| Clusters | 288 |
| Noise Fraction | 0.0677 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.008 |
| locked_door | 0.003 | 0.000 | 0.001 | 0.006 |
| open_door | 0.003 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.008 | 0.006 | 0.004 | 0.000 |

---

## Checkpoint 9,008 episodes — step 2,433,888

**Episodes:** 9,008  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9186 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1206 |
| Gradient Magnitude (Failure) | 1.2800 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4922 | 0.6205 |
| Neutral | 1.1632 | 0.9556 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1730 |
| Cosine Distance | 0.0050 |
| Clusters | 304 |
| Noise Fraction | 0.0536 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.003 | 0.010 |
| locked_door | 0.003 | 0.000 | 0.001 | 0.007 |
| open_door | 0.003 | 0.001 | 0.000 | 0.005 |
| target_ball | 0.010 | 0.007 | 0.005 | -0.000 |

---

## Checkpoint 9,104 episodes — step 2,458,928

**Episodes:** 9,104  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8559 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1367 |
| Gradient Magnitude (Failure) | 1.3351 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4655 | 0.5772 |
| Neutral | 1.1875 | 0.9250 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2666 |
| Cosine Distance | 0.0080 |
| Clusters | 313 |
| Noise Fraction | 0.0683 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.002 | 0.010 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.006 |
| open_door | 0.002 | 0.002 | 0.000 | 0.005 |
| target_ball | 0.010 | 0.006 | 0.005 | 0.000 |

---

## Checkpoint 9,184 episodes — step 2,483,968

**Episodes:** 9,184  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8547 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1791 |
| Gradient Magnitude (Failure) | 1.3644 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5366 | 0.4947 |
| Neutral | 1.2068 | 0.8612 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3114 |
| Cosine Distance | 0.0094 |
| Clusters | 310 |
| Noise Fraction | 0.0684 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.002 | 0.010 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.006 |
| open_door | 0.002 | 0.002 | 0.000 | 0.005 |
| target_ball | 0.010 | 0.006 | 0.005 | 0.000 |

---

## Checkpoint 9,280 episodes — step 2,509,008

**Episodes:** 9,280  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8517 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1762 |
| Gradient Magnitude (Failure) | 1.3293 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5188 | 0.5293 |
| Neutral | 1.2430 | 0.9695 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3363 |
| Cosine Distance | 0.0095 |
| Clusters | 306 |
| Noise Fraction | 0.0496 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.007 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.006 |
| open_door | 0.003 | 0.002 | -0.000 | 0.004 |
| target_ball | 0.007 | 0.006 | 0.004 | 0.000 |

---

## Checkpoint 9,376 episodes — step 2,534,048

**Episodes:** 9,376  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8413 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1956 |
| Gradient Magnitude (Failure) | 1.3735 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6185 | 0.4183 |
| Neutral | 1.1938 | 0.9609 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3233 |
| Cosine Distance | 0.0087 |
| Clusters | 308 |
| Noise Fraction | 0.0643 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.005 |
| locked_door | 0.002 | 0.000 | 0.001 | 0.003 |
| open_door | 0.003 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.005 | 0.003 | 0.002 | 0.000 |

---

## Checkpoint 9,472 episodes — step 2,559,088

**Episodes:** 9,472  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7931 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2322 |
| Gradient Magnitude (Failure) | 1.4514 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4899 | 0.4126 |
| Neutral | 1.1855 | 0.9602 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3396 |
| Cosine Distance | 0.0099 |
| Clusters | 316 |
| Noise Fraction | 0.0572 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.006 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.006 |
| open_door | 0.003 | 0.002 | 0.000 | 0.002 |
| target_ball | 0.006 | 0.006 | 0.002 | -0.000 |

---

## Checkpoint 9,568 episodes — step 2,584,128

**Episodes:** 9,568  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7932 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2679 |
| Gradient Magnitude (Failure) | 1.4976 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6616 | 0.3698 |
| Neutral | 1.2451 | 0.8519 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3679 |
| Cosine Distance | 0.0106 |
| Clusters | 331 |
| Noise Fraction | 0.0510 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.006 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.005 |
| open_door | 0.003 | 0.002 | -0.000 | 0.002 |
| target_ball | 0.006 | 0.005 | 0.002 | 0.000 |

---

## Checkpoint 9,648 episodes — step 2,609,168

**Episodes:** 9,648  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8027 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2167 |
| Gradient Magnitude (Failure) | 1.4926 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7313 | 0.4660 |
| Neutral | 1.2113 | 0.9886 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3415 |
| Cosine Distance | 0.0098 |
| Clusters | 329 |
| Noise Fraction | 0.0400 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.002 | 0.003 | 0.005 |
| locked_door | 0.002 | -0.000 | 0.002 | 0.005 |
| open_door | 0.003 | 0.002 | 0.000 | 0.001 |
| target_ball | 0.005 | 0.005 | 0.001 | -0.000 |

---

## Checkpoint 9,744 episodes — step 2,634,208

**Episodes:** 9,744  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7257 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2753 |
| Gradient Magnitude (Failure) | 1.4925 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7325 | 0.5033 |
| Neutral | 1.2223 | 0.9300 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3555 |
| Cosine Distance | 0.0096 |
| Clusters | 328 |
| Noise Fraction | 0.0560 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.003 | 0.006 |
| locked_door | 0.001 | 0.000 | 0.002 | 0.005 |
| open_door | 0.003 | 0.002 | 0.000 | 0.002 |
| target_ball | 0.006 | 0.005 | 0.002 | 0.000 |

---

## Checkpoint 9,840 episodes — step 2,659,248

**Episodes:** 9,840  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7455 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2117 |
| Gradient Magnitude (Failure) | 1.5832 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8182 | 0.3856 |
| Neutral | 1.2150 | 0.9856 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3825 |
| Cosine Distance | 0.0110 |
| Clusters | 339 |
| Noise Fraction | 0.0403 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.002 | 0.003 | 0.004 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.004 |
| open_door | 0.003 | 0.002 | -0.000 | 0.001 |
| target_ball | 0.004 | 0.004 | 0.001 | 0.000 |

---

## Checkpoint 9,936 episodes — step 2,684,288

**Episodes:** 9,936  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7791 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2034 |
| Gradient Magnitude (Failure) | 1.5627 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5503 | 0.4833 |
| Neutral | 1.2180 | 0.9849 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3952 |
| Cosine Distance | 0.0128 |
| Clusters | 328 |
| Noise Fraction | 0.0483 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.005 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.004 |
| open_door | 0.003 | 0.002 | 0.000 | 0.002 |
| target_ball | 0.005 | 0.004 | 0.002 | 0.000 |

---

## Checkpoint 10,032 episodes — step 2,709,328

**Episodes:** 10,032  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7041 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2748 |
| Gradient Magnitude (Failure) | 1.5666 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.9566 | 0.2938 |
| Neutral | 1.2184 | 0.9177 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3855 |
| Cosine Distance | 0.0129 |
| Clusters | 321 |
| Noise Fraction | 0.0447 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.003 | 0.008 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.009 |
| open_door | 0.003 | 0.002 | 0.000 | 0.004 |
| target_ball | 0.008 | 0.009 | 0.004 | 0.000 |

---

## Checkpoint 10,112 episodes — step 2,734,368

**Episodes:** 10,112  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8505 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2242 |
| Gradient Magnitude (Failure) | 1.5870 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3007 | 0.6362 |
| Neutral | 1.2126 | 0.9868 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3429 |
| Cosine Distance | 0.0122 |
| Clusters | 313 |
| Noise Fraction | 0.0592 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.003 | 0.004 | 0.011 |
| locked_door | 0.003 | 0.000 | 0.003 | 0.012 |
| open_door | 0.004 | 0.003 | 0.000 | 0.004 |
| target_ball | 0.011 | 0.012 | 0.004 | 0.000 |

---

## Checkpoint 10,208 episodes — step 2,759,408

**Episodes:** 10,208  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8625 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2223 |
| Gradient Magnitude (Failure) | 1.5600 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4471 | 0.5978 |
| Neutral | 1.2192 | 0.9930 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3141 |
| Cosine Distance | 0.0103 |
| Clusters | 310 |
| Noise Fraction | 0.0495 |

---

## Checkpoint 10,304 episodes — step 2,784,448

**Episodes:** 10,304  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8489 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2071 |
| Gradient Magnitude (Failure) | 1.5697 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4847 | 0.5089 |
| Neutral | 1.2326 | 0.9954 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3017 |
| Cosine Distance | 0.0097 |
| Clusters | 301 |
| Noise Fraction | 0.0634 |

---

## Checkpoint 10,400 episodes — step 2,809,488

**Episodes:** 10,400  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8474 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1742 |
| Gradient Magnitude (Failure) | 1.5632 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2077 | 0.6001 |
| Neutral | 1.1688 | 0.9953 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3184 |
| Cosine Distance | 0.0109 |
| Clusters | 295 |
| Noise Fraction | 0.0645 |

---

## Checkpoint 10,496 episodes — step 2,834,528

**Episodes:** 10,496  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8565 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2057 |
| Gradient Magnitude (Failure) | 1.5616 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4817 | 0.6641 |
| Neutral | 1.1738 | 0.9574 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3334 |
| Cosine Distance | 0.0110 |
| Clusters | 318 |
| Noise Fraction | 0.0543 |

---

## Checkpoint 10,576 episodes — step 2,859,568

**Episodes:** 10,576  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8241 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1716 |
| Gradient Magnitude (Failure) | 1.5357 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2554 | 0.6312 |
| Neutral | 1.1820 | 0.9921 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3468 |
| Cosine Distance | 0.0127 |
| Clusters | 309 |
| Noise Fraction | 0.0460 |

---

## Checkpoint 10,672 episodes — step 2,884,608

**Episodes:** 10,672  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8389 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1808 |
| Gradient Magnitude (Failure) | 1.4862 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4303 | 0.5039 |
| Neutral | 1.1984 | 0.9592 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3353 |
| Cosine Distance | 0.0119 |
| Clusters | 320 |
| Noise Fraction | 0.0509 |

---

## Checkpoint 10,768 episodes — step 2,909,648

**Episodes:** 10,768  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7945 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2481 |
| Gradient Magnitude (Failure) | 1.4605 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2852 | 0.6249 |
| Neutral | 1.2069 | 0.9749 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3752 |
| Cosine Distance | 0.0127 |
| Clusters | 319 |
| Noise Fraction | 0.0493 |

---

## Checkpoint 10,864 episodes — step 2,934,688

**Episodes:** 10,864  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8024 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1873 |
| Gradient Magnitude (Failure) | 1.5031 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8115 | 0.4837 |
| Neutral | 1.2180 | 0.9175 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3511 |
| Cosine Distance | 0.0115 |
| Clusters | 324 |
| Noise Fraction | 0.0630 |

---

## Checkpoint 10,960 episodes — step 2,959,728

**Episodes:** 10,960  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7794 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1913 |
| Gradient Magnitude (Failure) | 1.5068 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4053 | 0.5083 |
| Neutral | 1.1914 | 0.9972 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3595 |
| Cosine Distance | 0.0108 |
| Clusters | 319 |
| Noise Fraction | 0.0594 |

---

## Checkpoint 11,040 episodes — step 2,984,768

**Episodes:** 11,040  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.400)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7512 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3698 |
| Gradient Magnitude (Failure) | 1.4334 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3635 | 0.5095 |
| Neutral | 1.2088 | 0.9175 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3475 |
| Cosine Distance | 0.0115 |
| Clusters | 327 |
| Noise Fraction | 0.0670 |

---

## Checkpoint 11,136 episodes — step 3,009,808

**Episodes:** 11,136  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8117 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2239 |
| Gradient Magnitude (Failure) | 1.4525 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3879 | 0.4994 |
| Neutral | 1.2273 | 0.9922 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2757 |
| Cosine Distance | 0.0104 |
| Clusters | 316 |
| Noise Fraction | 0.0610 |

---

## Checkpoint 11,232 episodes — step 3,034,848

**Episodes:** 11,232  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7855 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2626 |
| Gradient Magnitude (Failure) | 1.4257 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2381 | 0.5412 |
| Neutral | 1.2314 | 0.9606 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2778 |
| Cosine Distance | 0.0093 |
| Clusters | 316 |
| Noise Fraction | 0.0590 |

---

## Checkpoint 11,328 episodes — step 3,059,888

**Episodes:** 11,328  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8406 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2246 |
| Gradient Magnitude (Failure) | 1.4334 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8030 | 0.3416 |
| Neutral | 1.2191 | 0.9616 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2530 |
| Cosine Distance | 0.0089 |
| Clusters | 326 |
| Noise Fraction | 0.0589 |

---

## Checkpoint 11,424 episodes — step 3,084,928

**Episodes:** 11,424  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8121 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2127 |
| Gradient Magnitude (Failure) | 1.4638 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3509 | 0.4491 |
| Neutral | 1.1908 | 0.9887 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2967 |
| Cosine Distance | 0.0115 |
| Clusters | 307 |
| Noise Fraction | 0.0471 |

---

## Checkpoint 11,504 episodes — step 3,109,968

**Episodes:** 11,504  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8615 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1778 |
| Gradient Magnitude (Failure) | 1.4710 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1315 | 0.5202 |
| Neutral | 1.1630 | 0.9572 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3412 |
| Cosine Distance | 0.0123 |
| Clusters | 334 |
| Noise Fraction | 0.0639 |

---

## Checkpoint 11,600 episodes — step 3,135,008

**Episodes:** 11,600  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8327 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1647 |
| Gradient Magnitude (Failure) | 1.4525 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4373 | 0.6210 |
| Neutral | 1.1423 | 0.9652 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3473 |
| Cosine Distance | 0.0120 |
| Clusters | 320 |
| Noise Fraction | 0.0575 |

---

## Checkpoint 11,696 episodes — step 3,160,048

**Episodes:** 11,696  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8405 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1608 |
| Gradient Magnitude (Failure) | 1.4374 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3118 | 0.4589 |
| Neutral | 1.1563 | 0.8833 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3745 |
| Cosine Distance | 0.0129 |
| Clusters | 324 |
| Noise Fraction | 0.0627 |

---

## Checkpoint 11,792 episodes — step 3,185,088

**Episodes:** 11,792  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8705 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1533 |
| Gradient Magnitude (Failure) | 1.4513 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0923 | 0.6313 |
| Neutral | 1.1396 | 0.9768 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3503 |
| Cosine Distance | 0.0136 |
| Clusters | 310 |
| Noise Fraction | 0.0514 |

---

## Checkpoint 11,888 episodes — step 3,210,128

**Episodes:** 11,888  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8564 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2355 |
| Gradient Magnitude (Failure) | 1.4669 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3787 | 0.6746 |
| Neutral | 1.2411 | 0.9921 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2983 |
| Cosine Distance | 0.0133 |
| Clusters | 332 |
| Noise Fraction | 0.0625 |

---

## Checkpoint 11,968 episodes — step 3,235,168

**Episodes:** 11,968  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8141 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2641 |
| Gradient Magnitude (Failure) | 1.4957 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1938 | 0.6838 |
| Neutral | 1.2536 | 0.9745 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3150 |
| Cosine Distance | 0.0143 |
| Clusters | 314 |
| Noise Fraction | 0.0465 |

---

## Checkpoint 12,064 episodes — step 3,260,208

**Episodes:** 12,064  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8401 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2569 |
| Gradient Magnitude (Failure) | 1.4885 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5222 | 0.7597 |
| Neutral | 1.1621 | 0.8870 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3531 |
| Cosine Distance | 0.0143 |
| Clusters | 328 |
| Noise Fraction | 0.0553 |

---

## Checkpoint 12,160 episodes — step 3,285,248

**Episodes:** 12,160  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8900 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1565 |
| Gradient Magnitude (Failure) | 1.4901 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2052 | 0.6499 |
| Neutral | 1.1535 | 0.9921 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3616 |
| Cosine Distance | 0.0135 |
| Clusters | 315 |
| Noise Fraction | 0.0531 |

---

## Checkpoint 12,256 episodes — step 3,310,288

**Episodes:** 12,256  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8847 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2373 |
| Gradient Magnitude (Failure) | 1.4714 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3055 | 0.6228 |
| Neutral | 1.2377 | 0.9971 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2638 |
| Cosine Distance | 0.0123 |
| Clusters | 302 |
| Noise Fraction | 0.0544 |

---

## Checkpoint 12,352 episodes — step 3,335,328

**Episodes:** 12,352  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8293 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2833 |
| Gradient Magnitude (Failure) | 1.5280 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0001 | 0.7864 |
| Neutral | 1.2761 | 0.9940 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2449 |
| Cosine Distance | 0.0136 |
| Clusters | 320 |
| Noise Fraction | 0.0503 |

---

## Checkpoint 12,432 episodes — step 3,360,368

**Episodes:** 12,432  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8389 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2090 |
| Gradient Magnitude (Failure) | 1.4920 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2412 | 0.6688 |
| Neutral | 1.2132 | 0.9916 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3237 |
| Cosine Distance | 0.0150 |
| Clusters | 308 |
| Noise Fraction | 0.0412 |

---

## Checkpoint 12,528 episodes — step 3,385,408

**Episodes:** 12,528  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8302 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3277 |
| Gradient Magnitude (Failure) | 1.4603 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0515 | 0.6869 |
| Neutral | 1.2920 | 0.9717 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2537 |
| Cosine Distance | 0.0122 |
| Clusters | 309 |
| Noise Fraction | 0.0566 |

---

## Checkpoint 12,624 episodes — step 3,410,448

**Episodes:** 12,624  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8933 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2125 |
| Gradient Magnitude (Failure) | 1.4694 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3942 | 0.6115 |
| Neutral | 1.1977 | 0.9937 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2840 |
| Cosine Distance | 0.0120 |
| Clusters | 323 |
| Noise Fraction | 0.0558 |

---

## Checkpoint 12,720 episodes — step 3,435,488

**Episodes:** 12,720  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8885 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3384 |
| Gradient Magnitude (Failure) | 1.5463 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3747 | 0.7181 |
| Neutral | 1.3396 | 0.9997 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2687 |
| Cosine Distance | 0.0125 |
| Clusters | 327 |
| Noise Fraction | 0.0420 |

---

## Checkpoint 12,816 episodes — step 3,460,528

**Episodes:** 12,816  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9331 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1915 |
| Gradient Magnitude (Failure) | 1.4544 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1853 | 0.6392 |
| Neutral | 1.1865 | 0.9966 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2807 |
| Cosine Distance | 0.0108 |
| Clusters | 308 |
| Noise Fraction | 0.0490 |

---

## Checkpoint 12,896 episodes — step 3,485,568

**Episodes:** 12,896  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9154 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1958 |
| Gradient Magnitude (Failure) | 1.4779 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1346 | 0.7078 |
| Neutral | 1.1864 | 0.9886 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2619 |
| Cosine Distance | 0.0098 |
| Clusters | 322 |
| Noise Fraction | 0.0435 |

---

## Checkpoint 12,992 episodes — step 3,510,608

**Episodes:** 12,992  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9080 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2347 |
| Gradient Magnitude (Failure) | 1.4726 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3659 | 0.7665 |
| Neutral | 1.2175 | 0.8905 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2456 |
| Cosine Distance | 0.0095 |
| Clusters | 324 |
| Noise Fraction | 0.0608 |

---

## Checkpoint 13,088 episodes — step 3,535,648

**Episodes:** 13,088  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8380 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3644 |
| Gradient Magnitude (Failure) | 1.4423 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4112 | 0.5863 |
| Neutral | 1.2757 | 0.9359 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1970 |
| Cosine Distance | 0.0095 |
| Clusters | 313 |
| Noise Fraction | 0.0448 |

---

## Checkpoint 13,184 episodes — step 3,560,688

**Episodes:** 13,184  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8673 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3420 |
| Gradient Magnitude (Failure) | 1.4394 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2740 | 0.7958 |
| Neutral | 1.2922 | 0.9572 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1640 |
| Cosine Distance | 0.0076 |
| Clusters | 310 |
| Noise Fraction | 0.0347 |

---

## Checkpoint 13,280 episodes — step 3,585,728

**Episodes:** 13,280  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8817 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2544 |
| Gradient Magnitude (Failure) | 1.3650 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4867 | 0.4905 |
| Neutral | 1.2269 | 0.9918 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1907 |
| Cosine Distance | 0.0088 |
| Clusters | 309 |
| Noise Fraction | 0.0702 |

---

## Checkpoint 13,360 episodes — step 3,610,768

**Episodes:** 13,360  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9184 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2305 |
| Gradient Magnitude (Failure) | 1.4266 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5103 | 0.6915 |
| Neutral | 1.2634 | 0.9711 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1867 |
| Cosine Distance | 0.0083 |
| Clusters | 316 |
| Noise Fraction | 0.0381 |

---

## Checkpoint 13,456 episodes — step 3,635,808

**Episodes:** 13,456  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8388 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3330 |
| Gradient Magnitude (Failure) | 1.4181 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1468 | 0.7330 |
| Neutral | 1.3000 | 0.9748 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1898 |
| Cosine Distance | 0.0102 |
| Clusters | 327 |
| Noise Fraction | 0.0442 |

---

## Checkpoint 13,552 episodes — step 3,660,848

**Episodes:** 13,552  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8229 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3216 |
| Gradient Magnitude (Failure) | 1.4247 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8371 | 0.6106 |
| Neutral | 1.3238 | 0.9952 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1878 |
| Cosine Distance | 0.0108 |
| Clusters | 332 |
| Noise Fraction | 0.0445 |

---

## Checkpoint 13,648 episodes — step 3,685,888

**Episodes:** 13,648  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8502 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2740 |
| Gradient Magnitude (Failure) | 1.4282 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3782 | 0.7669 |
| Neutral | 1.1906 | 0.8636 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2451 |
| Cosine Distance | 0.0107 |
| Clusters | 310 |
| Noise Fraction | 0.0660 |

---

## Checkpoint 13,744 episodes — step 3,710,928

**Episodes:** 13,744  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8787 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1663 |
| Gradient Magnitude (Failure) | 1.4493 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3047 | 0.6025 |
| Neutral | 1.1685 | 0.9969 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2939 |
| Cosine Distance | 0.0120 |
| Clusters | 323 |
| Noise Fraction | 0.0470 |

---

## Checkpoint 13,824 episodes — step 3,735,968

**Episodes:** 13,824  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8406 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3243 |
| Gradient Magnitude (Failure) | 1.4335 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6133 | 0.7353 |
| Neutral | 1.2530 | 0.9572 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2195 |
| Cosine Distance | 0.0100 |
| Clusters | 326 |
| Noise Fraction | 0.0552 |

---

## Checkpoint 13,920 episodes — step 3,761,008

**Episodes:** 13,920  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9018 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2466 |
| Gradient Magnitude (Failure) | 1.4458 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4814 | 0.6644 |
| Neutral | 1.2425 | 0.9633 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1846 |
| Cosine Distance | 0.0088 |
| Clusters | 309 |
| Noise Fraction | 0.0445 |

---

## Checkpoint 14,016 episodes — step 3,786,048

**Episodes:** 14,016  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9158 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2096 |
| Gradient Magnitude (Failure) | 1.4666 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5040 | 0.6120 |
| Neutral | 1.2165 | 0.9935 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2212 |
| Cosine Distance | 0.0091 |
| Clusters | 306 |
| Noise Fraction | 0.0513 |

---

## Checkpoint 14,112 episodes — step 3,811,088

**Episodes:** 14,112  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8749 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2104 |
| Gradient Magnitude (Failure) | 1.5251 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2369 | 0.7099 |
| Neutral | 1.2222 | 0.9682 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2603 |
| Cosine Distance | 0.0122 |
| Clusters | 309 |
| Noise Fraction | 0.0330 |

---

## Checkpoint 14,192 episodes — step 3,836,128

**Episodes:** 14,192  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8494 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2338 |
| Gradient Magnitude (Failure) | 1.5176 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2153 | 0.6514 |
| Neutral | 1.3770 | 0.9528 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2455 |
| Cosine Distance | 0.0114 |
| Clusters | 319 |
| Noise Fraction | 0.0472 |

---

## Checkpoint 14,288 episodes — step 3,861,168

**Episodes:** 14,288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8670 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2947 |
| Gradient Magnitude (Failure) | 1.5447 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4467 | 0.6913 |
| Neutral | 1.2937 | 0.9848 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1968 |
| Cosine Distance | 0.0100 |
| Clusters | 318 |
| Noise Fraction | 0.0377 |

---

## Checkpoint 14,384 episodes — step 3,886,208

**Episodes:** 14,384  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9214 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2992 |
| Gradient Magnitude (Failure) | 1.4323 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.1288 | 0.8065 |
| Neutral | 1.2925 | 0.9965 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1771 |
| Cosine Distance | 0.0076 |
| Clusters | 322 |
| Noise Fraction | 0.0443 |

---

## Checkpoint 14,480 episodes — step 3,911,248

**Episodes:** 14,480  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9082 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2122 |
| Gradient Magnitude (Failure) | 1.4822 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4837 | 0.7386 |
| Neutral | 1.1736 | 0.9895 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2980 |
| Cosine Distance | 0.0102 |
| Clusters | 310 |
| Noise Fraction | 0.0548 |

---

## Checkpoint 14,576 episodes — step 3,936,288

**Episodes:** 14,576  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9232 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2138 |
| Gradient Magnitude (Failure) | 1.4850 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5678 | 0.7434 |
| Neutral | 1.1858 | 0.9941 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2694 |
| Cosine Distance | 0.0104 |
| Clusters | 313 |
| Noise Fraction | 0.0447 |

---

## Checkpoint 14,656 episodes — step 3,961,328

**Episodes:** 14,656  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8764 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2686 |
| Gradient Magnitude (Failure) | 1.5068 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4563 | 0.6767 |
| Neutral | 1.2651 | 0.9953 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2104 |
| Cosine Distance | 0.0107 |
| Clusters | 320 |
| Noise Fraction | 0.0401 |

---

## Checkpoint 14,752 episodes — step 3,986,368

**Episodes:** 14,752  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9238 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2557 |
| Gradient Magnitude (Failure) | 1.4752 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.6775 | 0.6690 |
| Neutral | 1.2748 | 0.9877 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2190 |
| Cosine Distance | 0.0109 |
| Clusters | 315 |
| Noise Fraction | 0.0502 |

---

## Checkpoint 14,848 episodes — step 4,011,408

**Episodes:** 14,848  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9316 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2453 |
| Gradient Magnitude (Failure) | 1.4582 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4605 | 0.6987 |
| Neutral | 1.2321 | 0.9881 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2257 |
| Cosine Distance | 0.0098 |
| Clusters | 311 |
| Noise Fraction | 0.0471 |

---

## Checkpoint 14,944 episodes — step 4,036,448

**Episodes:** 14,944  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9100 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2339 |
| Gradient Magnitude (Failure) | 1.4988 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5727 | 0.6226 |
| Neutral | 1.2320 | 0.9990 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2356 |
| Cosine Distance | 0.0094 |
| Clusters | 311 |
| Noise Fraction | 0.0441 |

---

## Checkpoint 15,040 episodes — step 4,061,488

**Episodes:** 15,040  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9222 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2167 |
| Gradient Magnitude (Failure) | 1.4456 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7393 | 0.6091 |
| Neutral | 1.2340 | 0.9794 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2970 |
| Cosine Distance | 0.0125 |
| Clusters | 318 |
| Noise Fraction | 0.0361 |

---

## Checkpoint 15,120 episodes — step 4,086,528

**Episodes:** 15,120  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8848 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2370 |
| Gradient Magnitude (Failure) | 1.4085 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5348 | 0.6840 |
| Neutral | 1.2737 | 0.9011 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2710 |
| Cosine Distance | 0.0127 |
| Clusters | 307 |
| Noise Fraction | 0.0506 |

---

## Checkpoint 15,216 episodes — step 4,111,568

**Episodes:** 15,216  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8729 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2229 |
| Gradient Magnitude (Failure) | 1.4274 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4024 | 0.7066 |
| Neutral | 1.2175 | 0.9923 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2682 |
| Cosine Distance | 0.0122 |
| Clusters | 312 |
| Noise Fraction | 0.0571 |

---

## Checkpoint 15,312 episodes — step 4,136,608

**Episodes:** 15,312  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8961 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1295 |
| Gradient Magnitude (Failure) | 1.4284 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3706 | 0.6327 |
| Neutral | 1.1317 | 0.9962 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.4231 |
| Cosine Distance | 0.0182 |
| Clusters | 323 |
| Noise Fraction | 0.0567 |

---

## Checkpoint 15,408 episodes — step 4,161,648

**Episodes:** 15,408  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8006 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2519 |
| Gradient Magnitude (Failure) | 1.4043 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3565 | 0.6913 |
| Neutral | 1.1967 | 0.9780 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3311 |
| Cosine Distance | 0.0152 |
| Clusters | 315 |
| Noise Fraction | 0.0426 |

---

## Checkpoint 15,504 episodes — step 4,186,688

**Episodes:** 15,504  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8508 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2574 |
| Gradient Magnitude (Failure) | 1.4011 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3860 | 0.7212 |
| Neutral | 1.2531 | 0.9987 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2914 |
| Cosine Distance | 0.0159 |
| Clusters | 330 |
| Noise Fraction | 0.0221 |

---

## Checkpoint 15,584 episodes — step 4,211,728

**Episodes:** 15,584  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8982 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1552 |
| Gradient Magnitude (Failure) | 1.4015 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4823 | 0.6291 |
| Neutral | 1.2122 | 0.9613 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3061 |
| Cosine Distance | 0.0147 |
| Clusters | 310 |
| Noise Fraction | 0.0427 |

---

## Checkpoint 15,680 episodes — step 4,236,768

**Episodes:** 15,680  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9153 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1570 |
| Gradient Magnitude (Failure) | 1.3909 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3985 | 0.6379 |
| Neutral | 1.1648 | 0.9921 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2698 |
| Cosine Distance | 0.0129 |
| Clusters | 318 |
| Noise Fraction | 0.0442 |

---

## Checkpoint 15,776 episodes — step 4,261,808

**Episodes:** 15,776  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9166 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1782 |
| Gradient Magnitude (Failure) | 1.3817 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4429 | 0.6475 |
| Neutral | 1.1737 | 0.9988 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2604 |
| Cosine Distance | 0.0121 |
| Clusters | 311 |
| Noise Fraction | 0.0347 |

---

## Checkpoint 15,872 episodes — step 4,286,848

**Episodes:** 15,872  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9230 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1916 |
| Gradient Magnitude (Failure) | 1.4040 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2257 | 0.5929 |
| Neutral | 1.2176 | 0.9085 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2494 |
| Cosine Distance | 0.0109 |
| Clusters | 318 |
| Noise Fraction | 0.0359 |

---

## Checkpoint 15,968 episodes — step 4,311,888

**Episodes:** 15,968  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9035 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2513 |
| Gradient Magnitude (Failure) | 1.4469 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5354 | 0.6399 |
| Neutral | 1.2900 | 0.9617 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2310 |
| Cosine Distance | 0.0103 |
| Clusters | 323 |
| Noise Fraction | 0.0430 |

---

## Checkpoint 16,048 episodes — step 4,336,928

**Episodes:** 16,048  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8944 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2716 |
| Gradient Magnitude (Failure) | 1.4449 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2665 | 0.6817 |
| Neutral | 1.2622 | 0.9797 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2746 |
| Cosine Distance | 0.0138 |
| Clusters | 319 |
| Noise Fraction | 0.0344 |

---

## Checkpoint 16,144 episodes — step 4,361,968

**Episodes:** 16,144  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8764 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1556 |
| Gradient Magnitude (Failure) | 1.4861 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4156 | 0.5871 |
| Neutral | 1.1486 | 0.9982 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.4224 |
| Cosine Distance | 0.0186 |
| Clusters | 313 |
| Noise Fraction | 0.0468 |

---

## Checkpoint 16,240 episodes — step 4,387,008

**Episodes:** 16,240  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9039 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2326 |
| Gradient Magnitude (Failure) | 1.4921 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4311 | 0.6685 |
| Neutral | 1.2444 | 0.9903 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3109 |
| Cosine Distance | 0.0158 |
| Clusters | 309 |
| Noise Fraction | 0.0257 |

---

## Checkpoint 16,336 episodes — step 4,412,048

**Episodes:** 16,336  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8794 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3678 |
| Gradient Magnitude (Failure) | 1.4955 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3990 | 0.7528 |
| Neutral | 1.3700 | 0.9958 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2452 |
| Cosine Distance | 0.0145 |
| Clusters | 313 |
| Noise Fraction | 0.0333 |

---

## Checkpoint 16,432 episodes — step 4,437,088

**Episodes:** 16,432  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9249 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3984 |
| Gradient Magnitude (Failure) | 1.5704 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4645 | 0.5537 |
| Neutral | 1.3786 | 0.9678 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1950 |
| Cosine Distance | 0.0111 |
| Clusters | 307 |
| Noise Fraction | 0.0317 |

---

## Checkpoint 16,512 episodes — step 4,462,128

**Episodes:** 16,512  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9370 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3933 |
| Gradient Magnitude (Failure) | 1.5686 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5713 | 0.6424 |
| Neutral | 1.4126 | 0.9788 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1784 |
| Cosine Distance | 0.0094 |
| Clusters | 303 |
| Noise Fraction | 0.0287 |

---

## Checkpoint 16,608 episodes — step 4,487,168

**Episodes:** 16,608  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9143 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3580 |
| Gradient Magnitude (Failure) | 1.5493 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5603 | 0.5304 |
| Neutral | 1.3132 | 0.9782 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2334 |
| Cosine Distance | 0.0117 |
| Clusters | 314 |
| Noise Fraction | 0.0248 |

---

## Checkpoint 16,704 episodes — step 4,512,208

**Episodes:** 16,704  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9048 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2945 |
| Gradient Magnitude (Failure) | 1.5838 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4803 | 0.6346 |
| Neutral | 1.2985 | 0.9355 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3063 |
| Cosine Distance | 0.0131 |
| Clusters | 310 |
| Noise Fraction | 0.0464 |

---

## Checkpoint 16,800 episodes — step 4,537,248

**Episodes:** 16,800  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9055 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3436 |
| Gradient Magnitude (Failure) | 1.5268 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.2593 | 0.6664 |
| Neutral | 1.3370 | 0.9847 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3063 |
| Cosine Distance | 0.0119 |
| Clusters | 320 |
| Noise Fraction | 0.0362 |

---

## Checkpoint 16,896 episodes — step 4,562,288

**Episodes:** 16,896  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9395 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.4228 |
| Gradient Magnitude (Failure) | 1.5258 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4616 | 0.7267 |
| Neutral | 1.4165 | 0.9906 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2221 |
| Cosine Distance | 0.0078 |
| Clusters | 315 |
| Noise Fraction | 0.0377 |

---

## Checkpoint 16,976 episodes — step 4,587,328

**Episodes:** 16,976  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9211 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.5038 |
| Gradient Magnitude (Failure) | 1.5894 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5556 | 0.6968 |
| Neutral | 1.5435 | 0.9773 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1694 |
| Cosine Distance | 0.0054 |
| Clusters | 303 |
| Noise Fraction | 0.0317 |

---

## Checkpoint 17,072 episodes — step 4,612,368

**Episodes:** 17,072  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9371 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.5867 |
| Gradient Magnitude (Failure) | 1.4899 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4910 | 0.7111 |
| Neutral | 1.5813 | 0.9608 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1135 |
| Cosine Distance | 0.0038 |
| Clusters | 310 |
| Noise Fraction | 0.0349 |

---

## Checkpoint 17,168 episodes — step 4,637,408

**Episodes:** 17,168  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9106 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.6220 |
| Gradient Magnitude (Failure) | 1.5141 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7635 | 0.6813 |
| Neutral | 1.6566 | 0.9709 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0798 |
| Cosine Distance | 0.0027 |
| Clusters | 326 |
| Noise Fraction | 0.0327 |

---

## Checkpoint 17,264 episodes — step 4,662,448

**Episodes:** 17,264  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8971 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.6560 |
| Gradient Magnitude (Failure) | 1.5655 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7945 | 0.7094 |
| Neutral | 1.6475 | 0.9891 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0688 |
| Cosine Distance | 0.0021 |
| Clusters | 314 |
| Noise Fraction | 0.0359 |

---

## Checkpoint 17,360 episodes — step 4,687,488

**Episodes:** 17,360  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8654 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7492 |
| Gradient Magnitude (Failure) | 1.6280 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8838 | 0.6528 |
| Neutral | 1.7413 | 0.9949 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0728 |
| Cosine Distance | 0.0024 |
| Clusters | 321 |
| Noise Fraction | 0.0260 |

---

## Checkpoint 17,440 episodes — step 4,712,528

**Episodes:** 17,440  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9019 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7681 |
| Gradient Magnitude (Failure) | 1.6415 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5606 | 0.7385 |
| Neutral | 1.7650 | 0.9991 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0620 |
| Cosine Distance | 0.0017 |
| Clusters | 329 |
| Noise Fraction | 0.0261 |

---

## Checkpoint 17,536 episodes — step 4,737,568

**Episodes:** 17,536  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9334 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7745 |
| Gradient Magnitude (Failure) | 1.7009 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.0251 | 0.6554 |
| Neutral | 1.7525 | 0.9712 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1081 |
| Cosine Distance | 0.0024 |
| Clusters | 318 |
| Noise Fraction | 0.0179 |

---

## Checkpoint 17,632 episodes — step 4,762,608

**Episodes:** 17,632  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9478 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7844 |
| Gradient Magnitude (Failure) | 1.7202 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.5564 | 0.8222 |
| Neutral | 1.7989 | 0.9818 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0653 |
| Cosine Distance | 0.0012 |
| Clusters | 327 |
| Noise Fraction | 0.0146 |

---

## Checkpoint 17,728 episodes — step 4,787,648

**Episodes:** 17,728  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9294 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.7952 |
| Gradient Magnitude (Failure) | 1.7470 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7437 | 0.6666 |
| Neutral | 1.7844 | 0.9848 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0596 |
| Cosine Distance | 0.0013 |
| Clusters | 316 |
| Noise Fraction | 0.0244 |

---

## Checkpoint 17,824 episodes — step 4,812,688

**Episodes:** 17,824  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9103 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8714 |
| Gradient Magnitude (Failure) | 1.6619 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.9714 | 0.5867 |
| Neutral | 1.8675 | 0.9976 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0578 |
| Cosine Distance | 0.0016 |
| Clusters | 309 |
| Noise Fraction | 0.0238 |

---

## Checkpoint 17,904 episodes — step 4,837,728

**Episodes:** 17,904  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9060 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8721 |
| Gradient Magnitude (Failure) | 1.7071 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4658 | 0.7975 |
| Neutral | 1.8628 | 0.9924 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0833 |
| Cosine Distance | 0.0022 |
| Clusters | 316 |
| Noise Fraction | 0.0207 |

---

## Checkpoint 18,000 episodes — step 4,862,768

**Episodes:** 18,000  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9484 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8693 |
| Gradient Magnitude (Failure) | 1.7053 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.8275 | 0.7447 |
| Neutral | 1.8775 | 0.9988 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0934 |
| Cosine Distance | 0.0028 |
| Clusters | 289 |
| Noise Fraction | 0.0164 |

---

## Checkpoint 18,096 episodes — step 4,887,808

**Episodes:** 18,096  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9426 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8267 |
| Gradient Magnitude (Failure) | 1.6291 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.9169 | 0.6605 |
| Neutral | 1.8305 | 0.9874 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0772 |
| Cosine Distance | 0.0019 |
| Clusters | 305 |
| Noise Fraction | 0.0214 |

---

## Checkpoint 18,192 episodes — step 4,912,848

**Episodes:** 18,192  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9372 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8354 |
| Gradient Magnitude (Failure) | 1.6543 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7990 | 0.6830 |
| Neutral | 1.8694 | 0.9819 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0754 |
| Cosine Distance | 0.0017 |
| Clusters | 310 |
| Noise Fraction | 0.0201 |

---

## Checkpoint 18,288 episodes — step 4,937,888

**Episodes:** 18,288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9295 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8156 |
| Gradient Magnitude (Failure) | 1.6364 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7209 | 0.7357 |
| Neutral | 1.8246 | 0.9879 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0671 |
| Cosine Distance | 0.0015 |
| Clusters | 314 |
| Noise Fraction | 0.0176 |

---

## Checkpoint 18,368 episodes — step 4,962,928

**Episodes:** 18,368  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9023 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.9001 |
| Gradient Magnitude (Failure) | 1.6325 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.4570 | 0.7729 |
| Neutral | 1.8509 | 0.9695 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0608 |
| Cosine Distance | 0.0016 |
| Clusters | 297 |
| Noise Fraction | 0.0175 |

---

## Checkpoint 18,464 episodes — step 4,987,968

**Episodes:** 18,464  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9501 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8165 |
| Gradient Magnitude (Failure) | 1.6001 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.7643 | 0.6809 |
| Neutral | 1.8214 | 0.9779 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0586 |
| Cosine Distance | 0.0017 |
| Clusters | 310 |
| Noise Fraction | 0.0221 |

---

## Checkpoint 18,512 episodes — step 5,000,000

**Episodes:** 18,512  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9457 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.8247 |
| Gradient Magnitude (Failure) | 1.5698 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.9316 | 0.5909 |
| Neutral | 1.8131 | 0.9965 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0663 |
| Cosine Distance | 0.0020 |
| Clusters | 327 |
| Noise Fraction | 0.0316 |
