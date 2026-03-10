# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 4  
**Total episodes:** 18,512  
**Experiment root:** `Rainbow-Proof-of-Concept-Runs\seed_4`  
**Generated:** 2026-03-09 16:23

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | Act. Cos. Dist. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 16 episodes — step 5,008 | 16 | -0.0062 | — | — | 2.8135 | 1.9058 | 0.0903 | 0.0045 | — |
| Checkpoint 96 episodes — step 30,048 | 96 | 0.6764 | — | — | 1.3137 | 1.4028 | 0.0747 | 0.0018 | — |
| Checkpoint 192 episodes — step 55,088 | 192 | 0.7429 | — | — | 1.2405 | 1.6587 | 0.0551 | 0.0005 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 288 episodes — step 80,128 | 288 | 0.8485 | — | — | 1.1092 | 1.1685 | 0.0840 | 0.0005 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 384 episodes — step 105,168 | 384 | 0.9606 | — | — | 1.0479 | 1.0810 | 0.0804 | 0.0004 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 480 episodes — step 130,208 | 480 | 0.9891 | — | — | 1.0602 | 1.0490 | 0.0668 | 0.0003 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 561 episodes — step 155,248 | 561 | 0.9965 | — | — | 1.0478 | 1.0416 | 0.0650 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 656 episodes — step 180,288 | 656 | 0.9809 | — | — | 1.0848 | 1.0413 | 0.0886 | 0.0003 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 752 episodes — step 205,328 | 752 | 0.9868 | — | — | 1.0491 | 1.0412 | 0.1021 | 0.0004 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 848 episodes — step 230,368 | 848 | 0.9077 | — | — | 1.0678 | 1.0809 | 0.1021 | 0.0004 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 944 episodes — step 255,408 | 944 | 0.9227 | — | — | 1.1493 | 1.0649 | 0.0974 | 0.0004 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 1,025 episodes — step 280,448 | 1,025 | 0.9385 | — | — | 1.0800 | 1.0950 | 0.1741 | 0.0012 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,120 episodes — step 305,488 | 1,120 | 0.9609 | — | — | 1.0690 | 1.0941 | 0.1350 | 0.0009 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 1,216 episodes — step 330,528 | 1,216 | 0.9633 | — | — | 1.0751 | 1.1010 | 0.0856 | 0.0005 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,312 episodes — step 355,568 | 1,312 | 0.9584 | — | — | 1.0411 | 1.1152 | 0.0924 | 0.0004 | — |
| Checkpoint 1,408 episodes — step 380,608 | 1,408 | 0.9235 | — | — | 1.0298 | 1.1288 | 0.1914 | 0.0014 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 1,489 episodes — step 405,648 | 1,489 | 0.8927 | — | — | 1.0235 | 1.1608 | 0.2549 | 0.0023 | — |
| Checkpoint 1,584 episodes — step 430,688 | 1,584 | 0.8805 | — | — | 1.0443 | 1.1568 | 0.3628 | 0.0048 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,680 episodes — step 455,728 | 1,680 | 0.9039 | — | — | 1.0579 | 1.1424 | 0.3806 | 0.0057 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,776 episodes — step 480,768 | 1,776 | 0.9762 | — | — | 1.0693 | 1.0448 | 0.1305 | 0.0005 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 1,872 episodes — step 505,808 | 1,872 | 0.9526 | — | — | 1.0833 | 1.0736 | 0.0472 | 0.0001 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 1,953 episodes — step 530,848 | 1,953 | 0.9430 | — | — | 1.1011 | 1.0654 | 0.0250 | 0.0001 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 2,048 episodes — step 555,888 | 2,048 | 0.9813 | — | — | 1.1072 | 1.0571 | 0.0338 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 2,144 episodes — step 580,928 | 2,144 | 0.9907 | — | — | 1.0817 | 1.0720 | 0.0997 | 0.0004 | — |
| Checkpoint 2,240 episodes — step 605,968 | 2,240 | 0.9803 | — | — | 1.0795 | 1.1014 | 0.1922 | 0.0013 | — |
| Checkpoint 2,336 episodes — step 631,008 | 2,336 | 0.9164 | — | — | 1.0654 | 1.1356 | 0.1790 | 0.0011 | — |
| Checkpoint 2,417 episodes — step 656,048 | 2,417 | 0.9703 | — | — | 1.0612 | 1.0920 | 0.1732 | 0.0011 | — |
| Checkpoint 2,512 episodes — step 681,088 | 2,512 | 0.9722 | — | — | 1.0221 | 1.0802 | 0.1906 | 0.0013 | — |
| Checkpoint 2,608 episodes — step 706,128 | 2,608 | 0.9771 | — | — | 1.0320 | 1.0607 | 0.1213 | 0.0005 | — |
| Checkpoint 2,704 episodes — step 731,168 | 2,704 | 0.9843 | — | — | 1.0570 | 1.0745 | 0.0720 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2,800 episodes — step 756,208 | 2,800 | 0.9602 | — | — | 1.0897 | 1.0386 | 0.0152 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2,881 episodes — step 781,248 | 2,881 | 0.9628 | — | — | 1.0813 | 1.0495 | 0.0405 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 2,976 episodes — step 806,288 | 2,976 | 0.9181 | — | — | 1.0869 | 1.0557 | 0.1007 | 0.0004 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,072 episodes — step 831,328 | 3,072 | 0.9779 | — | — | 1.1013 | 1.0829 | 0.1254 | 0.0005 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,168 episodes — step 856,368 | 3,168 | 0.9809 | — | — | 1.0891 | 1.0571 | 0.1344 | 0.0006 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,264 episodes — step 881,408 | 3,264 | 0.6663 | — | — | 1.1296 | 1.2285 | 0.1132 | 0.0004 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,345 episodes — step 906,448 | 3,345 | 0.9637 | — | — | 1.0715 | 1.0806 | 0.1082 | 0.0004 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,440 episodes — step 931,488 | 3,440 | 0.9513 | — | — | 1.0806 | 1.0777 | 0.1190 | 0.0005 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,536 episodes — step 956,528 | 3,536 | 0.9385 | — | — | 1.0802 | 1.1384 | 0.1174 | 0.0004 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,632 episodes — step 981,568 | 3,632 | 0.9686 | — | — | 1.0880 | 1.0787 | 0.0427 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,728 episodes — step 1,006,608 | 3,728 | 0.9779 | — | — | 1.0823 | 1.0712 | 0.0235 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,809 episodes — step 1,031,648 | 3,809 | 0.9322 | — | — | 1.1147 | 1.0776 | 0.0905 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3,904 episodes — step 1,056,688 | 3,904 | 0.9891 | — | — | 1.0872 | 1.0736 | 0.0484 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 4,000 episodes — step 1,081,728 | 4,000 | 0.9268 | — | — | 1.0830 | 1.1003 | 0.0727 | 0.0001 | — |
| Checkpoint 4,096 episodes — step 1,106,768 | 4,096 | 0.9911 | — | — | 1.1138 | 1.1158 | 0.0602 | 0.0001 | — |
| Checkpoint 4,177 episodes — step 1,131,808 | 4,177 | 0.9931 | — | — | 1.0608 | 1.0793 | 0.0521 | 0.0001 | — |
| Checkpoint 4,272 episodes — step 1,156,848 | 4,272 | 0.9763 | — | — | 1.0814 | 1.0716 | 0.0536 | 0.0001 | — |
| Checkpoint 4,368 episodes — step 1,181,888 | 4,368 | 0.9736 | — | — | 1.0715 | 1.0804 | 0.0138 | 0.0000 | — |
| Checkpoint 4,464 episodes — step 1,206,928 | 4,464 | 0.8840 | — | — | 1.0991 | 1.1112 | 0.0286 | 0.0000 | — |
| Checkpoint 4,560 episodes — step 1,231,968 | 4,560 | 0.9798 | — | — | 1.1203 | 1.0736 | 0.0785 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 4,641 episodes — step 1,257,008 | 4,641 | 0.9826 | — | — | 1.1263 | 1.0701 | 0.0419 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 4,736 episodes — step 1,282,048 | 4,736 | 0.9840 | — | — | 1.0735 | 1.0831 | 0.0708 | 0.0001 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4,832 episodes — step 1,307,088 | 4,832 | 0.9571 | — | — | 1.1525 | 1.0621 | 0.0975 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 4,928 episodes — step 1,332,128 | 4,928 | 0.9772 | — | — | 1.1240 | 1.0856 | 0.1140 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,024 episodes — step 1,357,168 | 5,024 | 0.9197 | — | — | 1.0813 | 1.1028 | 0.1219 | 0.0003 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,105 episodes — step 1,382,208 | 5,105 | 0.9045 | — | — | 1.1115 | 1.0961 | 0.1226 | 0.0003 | — |
| Checkpoint 5,200 episodes — step 1,407,248 | 5,200 | 0.9837 | — | — | 1.0818 | 1.0893 | 0.1009 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,296 episodes — step 1,432,288 | 5,296 | 0.9918 | — | — | 1.0967 | 1.0818 | 0.0889 | 0.0001 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,392 episodes — step 1,457,328 | 5,392 | 0.9920 | — | — | 1.0907 | 1.0883 | 0.1049 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 5,488 episodes — step 1,482,368 | 5,488 | 0.9638 | — | — | 1.0797 | 1.0667 | 0.0702 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,569 episodes — step 1,507,408 | 5,569 | 0.9260 | — | — | 1.0846 | 1.0787 | 0.0549 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,664 episodes — step 1,532,448 | 5,664 | 0.9623 | — | — | 1.0504 | 1.1146 | 0.0545 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,760 episodes — step 1,557,488 | 5,760 | 0.9955 | — | — | 1.0476 | 1.0581 | 0.1137 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,856 episodes — step 1,582,528 | 5,856 | 0.9923 | — | — | 1.0776 | 1.0831 | 0.1302 | 0.0003 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 5,952 episodes — step 1,607,568 | 5,952 | 0.9932 | — | — | 1.0650 | 1.0614 | 0.0781 | 0.0001 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,033 episodes — step 1,632,608 | 6,033 | 0.9108 | — | — | 1.1401 | 1.0619 | 0.0294 | 0.0000 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,128 episodes — step 1,657,648 | 6,128 | 0.9795 | — | — | 1.0869 | 1.0415 | 0.0928 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,224 episodes — step 1,682,688 | 6,224 | 0.9815 | — | — | 1.0653 | 1.0917 | 0.1074 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,320 episodes — step 1,707,728 | 6,320 | 0.9519 | — | — | 1.0619 | 1.0817 | 0.1037 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 6,416 episodes — step 1,732,768 | 6,416 | 0.9591 | — | — | 1.0733 | 1.1648 | 0.0304 | 0.0000 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,497 episodes — step 1,757,808 | 6,497 | 0.9925 | — | — | 1.0656 | 1.0751 | 0.0961 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 6,592 episodes — step 1,782,848 | 6,592 | 0.9871 | — | — | 1.0764 | 1.1084 | 0.0781 | 0.0001 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 6,688 episodes — step 1,807,888 | 6,688 | 0.9522 | — | — | 1.0642 | 1.0808 | 0.1096 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 6,784 episodes — step 1,832,928 | 6,784 | 0.9820 | — | — | 1.0613 | 1.0674 | 0.1331 | 0.0003 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 6,880 episodes — step 1,857,968 | 6,880 | 0.9690 | — | — | 1.0733 | 1.0825 | 0.1051 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 6,961 episodes — step 1,883,008 | 6,961 | 0.9739 | — | — | 1.0855 | 1.0828 | 0.1168 | 0.0002 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7,056 episodes — step 1,908,048 | 7,056 | 0.9929 | — | — | 1.0993 | 1.0846 | 0.0862 | 0.0001 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 7,152 episodes — step 1,933,088 | 7,152 | 0.9482 | — | — | 1.1018 | 1.0691 | 0.0604 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,248 episodes — step 1,958,128 | 7,248 | 0.9664 | — | — | 1.1256 | 1.0679 | 0.1227 | 0.0003 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,344 episodes — step 1,983,168 | 7,344 | 0.9592 | — | — | 1.0823 | 1.0892 | 0.0506 | 0.0001 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,425 episodes — step 2,008,208 | 7,425 | 0.9833 | — | — | 1.0889 | 1.0763 | 0.0324 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,520 episodes — step 2,033,248 | 7,520 | 0.9693 | — | — | 1.0994 | 1.0811 | 0.0923 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 7,616 episodes — step 2,058,288 | 7,616 | 0.9587 | — | — | 1.1019 | 1.0831 | 0.0912 | 0.0001 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,712 episodes — step 2,083,328 | 7,712 | 0.9382 | — | — | 1.0896 | 1.1206 | 0.1068 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,808 episodes — step 2,108,368 | 7,808 | 0.9776 | — | — | 1.1026 | 1.1074 | 0.0310 | 0.0000 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,889 episodes — step 2,133,408 | 7,889 | 0.9731 | — | — | 1.0969 | 1.0786 | 0.0398 | 0.0000 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7,984 episodes — step 2,158,448 | 7,984 | 0.9767 | — | — | 1.0814 | 1.0825 | 0.1013 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,080 episodes — step 2,183,488 | 8,080 | 0.9428 | — | — | 1.1199 | 1.0857 | 0.0089 | 0.0000 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,176 episodes — step 2,208,528 | 8,176 | 0.9686 | — | — | 1.0886 | 1.1425 | 0.0428 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,272 episodes — step 2,233,568 | 8,272 | 0.8261 | — | — | 1.1263 | 1.1118 | 0.0143 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,353 episodes — step 2,258,608 | 8,353 | 0.8971 | — | — | 1.1213 | 1.0812 | 0.0518 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,448 episodes — step 2,283,648 | 8,448 | 0.9894 | — | — | 1.1117 | 1.0914 | 0.0966 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,544 episodes — step 2,308,688 | 8,544 | 0.9875 | — | — | 1.1031 | 1.0955 | 0.0529 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,640 episodes — step 2,333,728 | 8,640 | 0.9559 | — | — | 1.0723 | 1.1124 | 0.0727 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,736 episodes — step 2,358,768 | 8,736 | 0.9679 | — | — | 1.1256 | 1.0877 | 0.0965 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8,817 episodes — step 2,383,808 | 8,817 | 0.9471 | — | — | 1.0771 | 1.1456 | 0.1151 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 8,912 episodes — step 2,408,848 | 8,912 | 0.9746 | — | — | 1.1032 | 1.0681 | 0.0383 | 0.0000 | — |
| Checkpoint 9,008 episodes — step 2,433,888 | 9,008 | 0.9849 | — | — | 1.0873 | 1.0848 | 0.0674 | 0.0001 | — |
| Checkpoint 9,104 episodes — step 2,458,928 | 9,104 | 0.9184 | — | — | 1.0805 | 1.1724 | 0.0631 | 0.0001 | — |
| Checkpoint 9,185 episodes — step 2,483,968 | 9,185 | 0.9816 | — | — | 1.1101 | 1.0876 | 0.1179 | 0.0002 | — |
| Checkpoint 9,280 episodes — step 2,509,008 | 9,280 | 0.9784 | — | — | 1.0878 | 1.0821 | 0.1278 | 0.0002 | — |
| Checkpoint 9,376 episodes — step 2,534,048 | 9,376 | 0.9602 | — | — | 1.0937 | 1.0960 | 0.1191 | 0.0002 | — |
| Checkpoint 9,472 episodes — step 2,559,088 | 9,472 | 0.9887 | — | — | 1.1283 | 1.1250 | 0.0782 | 0.0001 | — |
| Checkpoint 9,568 episodes — step 2,584,128 | 9,568 | 0.9661 | — | — | 1.1082 | 1.1478 | 0.0278 | 0.0000 | — |
| Checkpoint 9,649 episodes — step 2,609,168 | 9,649 | 0.9890 | — | — | 1.1293 | 1.1134 | 0.0180 | 0.0000 | — |
| Checkpoint 9,744 episodes — step 2,634,208 | 9,744 | 0.9771 | — | — | 1.1197 | 1.1083 | 0.0215 | 0.0000 | — |
| Checkpoint 9,840 episodes — step 2,659,248 | 9,840 | 0.9859 | — | — | 1.1095 | 1.0986 | 0.0481 | 0.0000 | — |
| Checkpoint 9,936 episodes — step 2,684,288 | 9,936 | 0.9570 | — | — | 1.1039 | 1.0884 | 0.0766 | 0.0001 | — |
| Checkpoint 10,032 episodes — step 2,709,328 | 10,032 | 0.9754 | — | — | 1.0936 | 1.0940 | 0.1232 | 0.0003 | — |
| Checkpoint 10,113 episodes — step 2,734,368 | 10,113 | 0.9384 | — | — | 1.1464 | 1.0863 | 0.1160 | 0.0002 | — |
| Checkpoint 10,208 episodes — step 2,759,408 | 10,208 | 0.9675 | — | — | 1.1206 | 1.0857 | 0.1273 | 0.0003 | — |
| Checkpoint 10,304 episodes — step 2,784,448 | 10,304 | 0.9925 | — | — | 1.0961 | 1.0858 | 0.1146 | 0.0002 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 10,400 episodes — step 2,809,488 | 10,400 | 0.9072 | — | — | 1.1730 | 1.0736 | 0.0882 | 0.0001 | — |
| Checkpoint 10,496 episodes — step 2,834,528 | 10,496 | 0.9711 | — | — | 1.1228 | 1.1033 | 0.0764 | 0.0001 | — |
| Checkpoint 10,577 episodes — step 2,859,568 | 10,577 | 0.9857 | — | — | 1.1046 | 1.0653 | 0.0302 | 0.0000 | — |
| Checkpoint 10,672 episodes — step 2,884,608 | 10,672 | 0.9148 | — | — | 1.0819 | 1.1741 | 0.0856 | 0.0001 | — |
| Checkpoint 10,768 episodes — step 2,909,648 | 10,768 | 0.9934 | — | — | 1.0626 | 1.0716 | 0.1078 | 0.0002 | — |
| Checkpoint 10,864 episodes — step 2,934,688 | 10,864 | 0.9747 | — | — | 1.0542 | 1.1039 | 0.1048 | 0.0002 | — |
| Checkpoint 10,960 episodes — step 2,959,728 | 10,960 | 0.9764 | — | — | 1.0875 | 1.0864 | 0.0362 | 0.0000 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 11,041 episodes — step 2,984,768 | 11,041 | 0.9709 | — | — | 1.1009 | 1.0807 | 0.0324 | 0.0000 | — |
| Checkpoint 11,136 episodes — step 3,009,808 | 11,136 | 0.9837 | — | — | 1.0745 | 1.0719 | 0.0458 | 0.0001 | — |
| Checkpoint 11,232 episodes — step 3,034,848 | 11,232 | 0.8571 | — | — | 1.1703 | 1.0721 | 0.0664 | 0.0001 | — |
| Checkpoint 11,328 episodes — step 3,059,888 | 11,328 | 0.9712 | — | — | 1.0916 | 1.0632 | 0.0233 | 0.0000 | — |
| Checkpoint 11,424 episodes — step 3,084,928 | 11,424 | 0.9881 | — | — | 1.0804 | 1.0527 | 0.0223 | 0.0000 | — |
| Checkpoint 11,505 episodes — step 3,109,968 | 11,505 | 0.9714 | — | — | 1.1658 | 1.0882 | 0.0851 | 0.0001 | — |
| Checkpoint 11,600 episodes — step 3,135,008 | 11,600 | 0.9518 | — | — | 1.1292 | 1.0743 | 0.0334 | 0.0000 | — |
| Checkpoint 11,696 episodes — step 3,160,048 | 11,696 | 0.9433 | — | — | 1.1275 | 1.0785 | 0.0980 | 0.0001 | — |
| Checkpoint 11,792 episodes — step 3,185,088 | 11,792 | 0.9940 | — | — | 1.1085 | 1.0944 | 0.1070 | 0.0002 | — |
| Checkpoint 11,888 episodes — step 3,210,128 | 11,888 | 0.9297 | — | — | 1.1173 | 1.0839 | 0.1065 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 11,969 episodes — step 3,235,168 | 11,969 | 0.9872 | — | — | 1.1009 | 1.0826 | 0.0811 | 0.0001 | — |
| Checkpoint 12,064 episodes — step 3,260,208 | 12,064 | 0.9943 | — | — | 1.0797 | 1.0817 | 0.1441 | 0.0004 | — |
| Checkpoint 12,160 episodes — step 3,285,248 | 12,160 | 0.8925 | — | — | 1.0667 | 1.1955 | 0.1096 | 0.0002 | — |
| Checkpoint 12,256 episodes — step 3,310,288 | 12,256 | 0.9830 | — | — | 1.0833 | 1.0720 | 0.0504 | 0.0001 | — |
| Checkpoint 12,352 episodes — step 3,335,328 | 12,352 | 0.9771 | — | — | 1.0813 | 1.0695 | 0.1031 | 0.0002 | — |
| Checkpoint 12,433 episodes — step 3,360,368 | 12,433 | 0.9721 | — | — | 1.0891 | 1.0721 | 0.1241 | 0.0003 | — |
| Checkpoint 12,528 episodes — step 3,385,408 | 12,528 | 0.9789 | — | — | 1.0838 | 1.0824 | 0.0789 | 0.0001 | — |
| Checkpoint 12,624 episodes — step 3,410,448 | 12,624 | 0.9050 | — | — | 1.1209 | 1.0817 | 0.0230 | 0.0000 | — |
| Checkpoint 12,720 episodes — step 3,435,488 | 12,720 | 0.9640 | — | — | 1.1060 | 1.0847 | 0.0483 | 0.0000 | — |
| Checkpoint 12,816 episodes — step 3,460,528 | 12,816 | 0.9459 | — | — | 1.1120 | 1.1165 | 0.0429 | 0.0000 | — |
| Checkpoint 12,897 episodes — step 3,485,568 | 12,897 | 0.9864 | — | — | 1.0901 | 1.0858 | 0.1119 | 0.0002 | — |
| Checkpoint 12,992 episodes — step 3,510,608 | 12,992 | 0.9549 | — | — | 1.0888 | 1.1158 | 0.1529 | 0.0004 | — |
| Checkpoint 13,088 episodes — step 3,535,648 | 13,088 | 0.8558 | — | — | 1.1268 | 1.1296 | 0.1147 | 0.0002 | — |
| Checkpoint 13,184 episodes — step 3,560,688 | 13,184 | 0.9788 | — | — | 1.1010 | 1.1002 | 0.1334 | 0.0003 | — |
| Checkpoint 13,280 episodes — step 3,585,728 | 13,280 | 0.9574 | — | — | 1.0954 | 1.0885 | 0.2152 | 0.0008 | — |
| Checkpoint 13,361 episodes — step 3,610,768 | 13,361 | 0.8058 | — | — | 1.0858 | 1.1555 | 0.1741 | 0.0006 | — |
| Checkpoint 13,456 episodes — step 3,635,808 | 13,456 | 0.9590 | — | — | 1.0725 | 1.0855 | 0.0594 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 13,552 episodes — step 3,660,848 | 13,552 | 0.9725 | — | — | 1.0943 | 1.1003 | 0.0439 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 13,648 episodes — step 3,685,888 | 13,648 | 0.9827 | — | — | 1.0987 | 1.0677 | 0.0719 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 13,744 episodes — step 3,710,928 | 13,744 | 0.9036 | — | — | 1.0902 | 1.1112 | 0.0818 | 0.0001 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 13,825 episodes — step 3,735,968 | 13,825 | 0.9264 | — | — | 1.0838 | 1.1864 | 0.0990 | 0.0002 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 13,920 episodes — step 3,761,008 | 13,920 | 0.9546 | — | — | 1.0733 | 1.0721 | 0.1298 | 0.0003 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 14,016 episodes — step 3,786,048 | 14,016 | 0.9049 | — | — | 1.0828 | 1.0988 | 0.1436 | 0.0003 | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 14,112 episodes — step 3,811,088 | 14,112 | 0.9924 | — | — | 1.0698 | 1.0679 | 0.1235 | 0.0003 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 14,193 episodes — step 3,836,128 | 14,193 | 0.9952 | — | — | 1.0644 | 1.0663 | 0.0948 | 0.0001 | — |
| Checkpoint 14,288 episodes — step 3,861,168 | 14,288 | 0.9767 | — | — | 1.0833 | 1.0768 | 0.0337 | 0.0000 | — |
| Checkpoint 14,384 episodes — step 3,886,208 | 14,384 | 0.9673 | — | — | 1.1242 | 1.0766 | 0.0167 | 0.0000 | — |
| Checkpoint 14,480 episodes — step 3,911,248 | 14,480 | 0.9684 | — | — | 1.1095 | 1.1130 | 0.0784 | 0.0001 | — |
| Checkpoint 14,576 episodes — step 3,936,288 | 14,576 | 0.9930 | — | — | 1.0934 | 1.1045 | 0.1551 | 0.0004 | — |
| Checkpoint 14,657 episodes — step 3,961,328 | 14,657 | 0.9679 | — | — | 1.1271 | 1.1025 | 0.1179 | 0.0002 | — |
| Checkpoint 14,752 episodes — step 3,986,368 | 14,752 | 0.9887 | — | — | 1.0827 | 1.0873 | 0.1334 | 0.0003 | — |
| Checkpoint 14,848 episodes — step 4,011,408 | 14,848 | 0.9762 | — | — | 1.0794 | 1.1049 | 0.1572 | 0.0004 | — |
| Checkpoint 14,944 episodes — step 4,036,448 | 14,944 | 0.9348 | — | — | 1.0958 | 1.0938 | 0.1652 | 0.0005 | — |
| Checkpoint 15,040 episodes — step 4,061,488 | 15,040 | 0.9850 | — | — | 1.1067 | 1.1265 | 0.1564 | 0.0004 | — |
| Checkpoint 15,121 episodes — step 4,086,528 | 15,121 | 0.9416 | — | — | 1.1188 | 1.0966 | 0.0755 | 0.0001 | — |
| Checkpoint 15,216 episodes — step 4,111,568 | 15,216 | 0.8940 | — | — | 1.1497 | 1.1017 | 0.0539 | 0.0001 | — |
| Checkpoint 15,312 episodes — step 4,136,608 | 15,312 | 0.9871 | — | — | 1.0942 | 1.0903 | 0.1359 | 0.0003 | — |
| Checkpoint 15,408 episodes — step 4,161,648 | 15,408 | 0.9676 | — | — | 1.0890 | 1.0770 | 0.1316 | 0.0003 | — |
| Checkpoint 15,504 episodes — step 4,186,688 | 15,504 | 0.9902 | — | — | 1.0844 | 1.1075 | 0.1313 | 0.0003 | — |
| Checkpoint 15,585 episodes — step 4,211,728 | 15,585 | 0.8836 | — | — | 1.1155 | 1.0787 | 0.1764 | 0.0005 | — |
| Checkpoint 15,680 episodes — step 4,236,768 | 15,680 | 0.9708 | — | — | 1.0869 | 1.0675 | 0.1245 | 0.0002 | — |
| Checkpoint 15,776 episodes — step 4,261,808 | 15,776 | 0.9791 | — | — | 1.0754 | 1.0773 | 0.1269 | 0.0002 | — |
| Checkpoint 15,872 episodes — step 4,286,848 | 15,872 | 0.9731 | — | — | 1.0684 | 1.0728 | 0.0964 | 0.0001 | — |
| Checkpoint 15,968 episodes — step 4,311,888 | 15,968 | 0.9566 | — | — | 1.1167 | 1.0620 | 0.0853 | 0.0001 | — |
| Checkpoint 16,049 episodes — step 4,336,928 | 16,049 | 0.9544 | — | — | 1.1286 | 1.0626 | 0.0791 | 0.0002 | — |
| Checkpoint 16,144 episodes — step 4,361,968 | 16,144 | 0.9721 | — | — | 1.1070 | 1.0729 | 0.0134 | 0.0000 | — |
| Checkpoint 16,240 episodes — step 4,387,008 | 16,240 | 0.9750 | — | — | 1.0943 | 1.0887 | 0.0791 | 0.0001 | — |
| Checkpoint 16,336 episodes — step 4,412,048 | 16,336 | 0.9519 | — | — | 1.0851 | 1.1450 | 0.1099 | 0.0002 | — |
| Checkpoint 16,432 episodes — step 4,437,088 | 16,432 | 0.9435 | — | — | 1.1450 | 1.0787 | 0.1988 | 0.0007 | — |
| Checkpoint 16,513 episodes — step 4,462,128 | 16,513 | 0.9333 | — | — | 1.0995 | 1.1244 | 0.1959 | 0.0007 | — |
| Checkpoint 16,608 episodes — step 4,487,168 | 16,608 | 0.9042 | — | — | 1.0926 | 1.1591 | 0.2097 | 0.0007 | — |
| Checkpoint 16,704 episodes — step 4,512,208 | 16,704 | 0.9284 | — | — | 1.1094 | 1.0861 | 0.1375 | 0.0003 | — |
| Checkpoint 16,800 episodes — step 4,537,248 | 16,800 | 0.9811 | — | — | 1.1111 | 1.1374 | 0.1480 | 0.0004 | — |
| Checkpoint 16,896 episodes — step 4,562,288 | 16,896 | 0.9862 | — | — | 1.0774 | 1.0746 | 0.1072 | 0.0002 | — |
| Checkpoint 16,977 episodes — step 4,587,328 | 16,977 | 0.8862 | — | — | 1.1448 | 1.0848 | 0.0922 | 0.0001 | — |
| Checkpoint 17,072 episodes — step 4,612,368 | 17,072 | 0.9657 | — | — | 1.1135 | 1.1006 | 0.1270 | 0.0003 | — |
| Checkpoint 17,168 episodes — step 4,637,408 | 17,168 | 0.9839 | — | — | 1.1002 | 1.0979 | 0.1370 | 0.0003 | — |
| Checkpoint 17,264 episodes — step 4,662,448 | 17,264 | 0.9768 | — | — | 1.0651 | 1.0956 | 0.1615 | 0.0004 | — |
| Checkpoint 17,360 episodes — step 4,687,488 | 17,360 | 0.9614 | — | — | 1.0716 | 1.0908 | 0.1389 | 0.0003 | — |
| Checkpoint 17,441 episodes — step 4,712,528 | 17,441 | 0.9736 | — | — | 1.0671 | 1.0888 | 0.1075 | 0.0002 | — |
| Checkpoint 17,536 episodes — step 4,737,568 | 17,536 | 0.9826 | — | — | 1.0725 | 1.0912 | 0.0839 | 0.0001 | — |
| Checkpoint 17,632 episodes — step 4,762,608 | 17,632 | 0.9757 | — | — | 1.0703 | 1.0967 | 0.1529 | 0.0004 | — |
| Checkpoint 17,728 episodes — step 4,787,648 | 17,728 | 0.9795 | — | — | 1.0739 | 1.0920 | 0.1471 | 0.0004 | — |
| Checkpoint 17,824 episodes — step 4,812,688 | 17,824 | 0.9691 | — | — | 1.1047 | 1.0714 | 0.1698 | 0.0005 | — |
| Checkpoint 17,905 episodes — step 4,837,728 | 17,905 | 0.8754 | — | — | 1.1215 | 1.1021 | 0.1628 | 0.0005 | — |
| Checkpoint 18,000 episodes — step 4,862,768 | 18,000 | 0.9850 | — | — | 1.0919 | 1.1032 | 0.1870 | 0.0007 | — |
| Checkpoint 18,096 episodes — step 4,887,808 | 18,096 | 0.9803 | — | — | 1.1004 | 1.0906 | 0.2318 | 0.0010 | — |
| Checkpoint 18,192 episodes — step 4,912,848 | 18,192 | 0.9774 | — | — | 1.0937 | 1.1639 | 0.2261 | 0.0010 | — |
| Checkpoint 18,288 episodes — step 4,937,888 | 18,288 | 0.9692 | — | — | 1.0775 | 1.1122 | 0.2419 | 0.0010 | — |
| Checkpoint 18,369 episodes — step 4,962,928 | 18,369 | 0.8789 | — | — | 1.0808 | 1.2426 | 0.1809 | 0.0006 | — |
| Checkpoint 18,464 episodes — step 4,987,968 | 18,464 | 0.9640 | — | — | 1.1046 | 1.1031 | 0.1661 | 0.0005 | — |
| Checkpoint 18,512 episodes — step 5,000,000 | 18,512 | 0.9909 | — | — | 1.1212 | 1.1230 | 0.1114 | 0.0002 | — |

---

## Checkpoint 16 episodes — step 5,008

**Episodes:** 16  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0062 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 2.8135 |
| Gradient Magnitude (Failure) | 1.9058 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 4.6499 | 0.1724 |
| Neutral | 2.2137 | 0.6255 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0903 |
| Cosine Distance | 0.0045 |
| Clusters | 76 |
| Noise Fraction | 0.0431 |

---

## Checkpoint 96 episodes — step 30,048

**Episodes:** 96  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6764 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.3137 |
| Gradient Magnitude (Failure) | 1.4028 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 3.5705 | 0.1538 |
| Neutral | 1.3678 | 0.8061 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0747 |
| Cosine Distance | 0.0018 |
| Clusters | 337 |
| Noise Fraction | 0.0301 |

---

## Checkpoint 192 episodes — step 55,088

**Episodes:** 192  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7429 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.2405 |
| Gradient Magnitude (Failure) | 1.6587 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.3045 | 0.4935 |
| Neutral | 1.2134 | 0.8260 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0551 |
| Cosine Distance | 0.0005 |
| Clusters | 311 |
| Noise Fraction | 0.0578 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.002 | 0.001 | 0.003 |
| locked_door | 0.002 | 0.000 | 0.002 | 0.004 |
| open_door | 0.001 | 0.002 | 0.000 | 0.003 |
| target_ball | 0.003 | 0.004 | 0.003 | 0.000 |

---

## Checkpoint 288 episodes — step 80,128

**Episodes:** 288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8485 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1092 |
| Gradient Magnitude (Failure) | 1.1685 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.9470 | 0.3622 |
| Neutral | 1.1591 | 0.9634 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0840 |
| Cosine Distance | 0.0005 |
| Clusters | 298 |
| Noise Fraction | 0.1016 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.002 | 0.003 |
| open_door | 0.000 | 0.002 | -0.000 | 0.001 |
| target_ball | 0.001 | 0.003 | 0.001 | 0.000 |

---

## Checkpoint 384 episodes — step 105,168

**Episodes:** 384  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9606 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0479 |
| Gradient Magnitude (Failure) | 1.0810 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4133 | 0.6679 |
| Neutral | 1.0643 | 0.9706 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0804 |
| Cosine Distance | 0.0004 |
| Clusters | 291 |
| Noise Fraction | 0.0957 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.002 | 0.002 |
| open_door | 0.000 | 0.002 | -0.000 | 0.002 |
| target_ball | 0.002 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 480 episodes — step 130,208

**Episodes:** 480  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9891 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0602 |
| Gradient Magnitude (Failure) | 1.0490 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.1571 | 0.7959 |
| Neutral | 1.0411 | 0.9806 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0668 |
| Cosine Distance | 0.0003 |
| Clusters | 284 |
| Noise Fraction | 0.1452 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.002 | 0.001 | 0.000 |

---

## Checkpoint 561 episodes — step 155,248

**Episodes:** 561  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9965 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0478 |
| Gradient Magnitude (Failure) | 1.0416 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3792 | 0.6630 |
| Neutral | 1.0480 | 0.9985 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0650 |
| Cosine Distance | 0.0002 |
| Clusters | 262 |
| Noise Fraction | 0.1274 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.002 | 0.004 | 0.001 | -0.000 |

---

## Checkpoint 656 episodes — step 180,288

**Episodes:** 656  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9809 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0848 |
| Gradient Magnitude (Failure) | 1.0413 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5373 | 0.7618 |
| Neutral | 1.0453 | 0.9599 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0886 |
| Cosine Distance | 0.0003 |
| Clusters | 251 |
| Noise Fraction | 0.1172 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.002 | 0.002 | 0.001 | 0.000 |

---

## Checkpoint 752 episodes — step 205,328

**Episodes:** 752  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9868 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0491 |
| Gradient Magnitude (Failure) | 1.0412 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2956 | 0.7713 |
| Neutral | 1.0480 | 0.9899 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1021 |
| Cosine Distance | 0.0004 |
| Clusters | 247 |
| Noise Fraction | 0.1367 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.000 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.000 |
| target_ball | 0.000 | 0.001 | 0.000 | -0.000 |

---

## Checkpoint 848 episodes — step 230,368

**Episodes:** 848  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9077 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0678 |
| Gradient Magnitude (Failure) | 1.0809 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5536 | 0.6967 |
| Neutral | 1.0831 | 0.9290 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1021 |
| Cosine Distance | 0.0004 |
| Clusters | 248 |
| Noise Fraction | 0.0737 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | -0.000 | 0.001 |
| target_ball | 0.002 | 0.002 | 0.001 | 0.000 |

---

## Checkpoint 944 episodes — step 255,408

**Episodes:** 944  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9227 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1493 |
| Gradient Magnitude (Failure) | 1.0649 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3265 | 0.8545 |
| Neutral | 1.0772 | 0.8928 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0974 |
| Cosine Distance | 0.0004 |
| Clusters | 236 |
| Noise Fraction | 0.0921 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.001 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.001 |
| open_door | 0.001 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 1,025 episodes — step 280,448

**Episodes:** 1,025  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9385 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0800 |
| Gradient Magnitude (Failure) | 1.0950 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.1986 | 0.8525 |
| Neutral | 1.1657 | 0.8349 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1741 |
| Cosine Distance | 0.0012 |
| Clusters | 252 |
| Noise Fraction | 0.0728 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.005 |
| open_door | 0.001 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.004 | 0.005 | 0.002 | 0.000 |

---

## Checkpoint 1,120 episodes — step 305,488

**Episodes:** 1,120  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9609 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0690 |
| Gradient Magnitude (Failure) | 1.0941 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5449 | 0.6029 |
| Neutral | 1.0626 | 0.9919 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1350 |
| Cosine Distance | 0.0009 |
| Clusters | 264 |
| Noise Fraction | 0.0802 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.002 | 0.003 |
| open_door | 0.001 | 0.002 | 0.000 | 0.001 |
| target_ball | 0.003 | 0.003 | 0.001 | 0.000 |

---

## Checkpoint 1,216 episodes — step 330,528

**Episodes:** 1,216  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9633 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0751 |
| Gradient Magnitude (Failure) | 1.1010 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5872 | 0.6006 |
| Neutral | 1.1650 | 0.9598 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0856 |
| Cosine Distance | 0.0005 |
| Clusters | 269 |
| Noise Fraction | 0.0665 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.002 | 0.006 |
| open_door | 0.001 | 0.002 | -0.000 | 0.004 |
| target_ball | 0.005 | 0.006 | 0.004 | -0.000 |

---

## Checkpoint 1,312 episodes — step 355,568

**Episodes:** 1,312  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9584 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0411 |
| Gradient Magnitude (Failure) | 1.1152 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4415 | 0.7579 |
| Neutral | 1.0552 | 0.9838 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0924 |
| Cosine Distance | 0.0004 |
| Clusters | 285 |
| Noise Fraction | 0.0758 |

---

## Checkpoint 1,408 episodes — step 380,608

**Episodes:** 1,408  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9235 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0298 |
| Gradient Magnitude (Failure) | 1.1288 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6174 | 0.7359 |
| Neutral | 1.0510 | 0.9896 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1914 |
| Cosine Distance | 0.0014 |
| Clusters | 278 |
| Noise Fraction | 0.0964 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.005 |
| locked_door | 0.001 | -0.000 | 0.002 | 0.005 |
| open_door | 0.001 | 0.002 | 0.000 | 0.004 |
| target_ball | 0.005 | 0.005 | 0.004 | 0.000 |

---

## Checkpoint 1,489 episodes — step 405,648

**Episodes:** 1,489  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8927 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0235 |
| Gradient Magnitude (Failure) | 1.1608 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6006 | 0.6641 |
| Neutral | 1.0234 | 0.9969 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2549 |
| Cosine Distance | 0.0023 |
| Clusters | 266 |
| Noise Fraction | 0.0818 |

---

## Checkpoint 1,584 episodes — step 430,688

**Episodes:** 1,584  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8805 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0443 |
| Gradient Magnitude (Failure) | 1.1568 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5532 | 0.7322 |
| Neutral | 1.0568 | 0.9415 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3628 |
| Cosine Distance | 0.0048 |
| Clusters | 278 |
| Noise Fraction | 0.0609 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.012 |
| locked_door | 0.001 | 0.000 | 0.003 | 0.015 |
| open_door | 0.001 | 0.003 | -0.000 | 0.010 |
| target_ball | 0.012 | 0.015 | 0.010 | 0.000 |

---

## Checkpoint 1,680 episodes — step 455,728

**Episodes:** 1,680  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9039 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0579 |
| Gradient Magnitude (Failure) | 1.1424 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4438 | 0.6208 |
| Neutral | 1.0904 | 0.9099 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.3806 |
| Cosine Distance | 0.0057 |
| Clusters | 287 |
| Noise Fraction | 0.0552 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.013 |
| locked_door | 0.001 | 0.000 | 0.003 | 0.016 |
| open_door | 0.001 | 0.003 | 0.000 | 0.011 |
| target_ball | 0.013 | 0.016 | 0.011 | 0.000 |

---

## Checkpoint 1,776 episodes — step 480,768

**Episodes:** 1,776  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9762 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0693 |
| Gradient Magnitude (Failure) | 1.0448 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3292 | 0.8103 |
| Neutral | 1.1179 | 0.9709 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1305 |
| Cosine Distance | 0.0005 |
| Clusters | 285 |
| Noise Fraction | 0.0887 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.010 |
| locked_door | 0.001 | -0.000 | 0.002 | 0.010 |
| open_door | 0.001 | 0.002 | 0.000 | 0.008 |
| target_ball | 0.010 | 0.010 | 0.008 | 0.000 |

---

## Checkpoint 1,872 episodes — step 505,808

**Episodes:** 1,872  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9526 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0833 |
| Gradient Magnitude (Failure) | 1.0736 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4313 | 0.7689 |
| Neutral | 1.0888 | 0.9872 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0472 |
| Cosine Distance | 0.0001 |
| Clusters | 279 |
| Noise Fraction | 0.0863 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.011 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.010 |
| open_door | 0.000 | 0.001 | 0.000 | 0.010 |
| target_ball | 0.011 | 0.010 | 0.010 | 0.000 |

---

## Checkpoint 1,953 episodes — step 530,848

**Episodes:** 1,953  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9430 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1011 |
| Gradient Magnitude (Failure) | 1.0654 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6695 | 0.7237 |
| Neutral | 1.1053 | 0.9976 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0250 |
| Cosine Distance | 0.0001 |
| Clusters | 285 |
| Noise Fraction | 0.0632 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.012 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.011 |
| open_door | 0.001 | 0.001 | 0.000 | 0.009 |
| target_ball | 0.012 | 0.011 | 0.009 | -0.000 |

---

## Checkpoint 2,048 episodes — step 555,888

**Episodes:** 2,048  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9813 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1072 |
| Gradient Magnitude (Failure) | 1.0571 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3622 | 0.8007 |
| Neutral | 1.1643 | 0.9434 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0338 |
| Cosine Distance | 0.0002 |
| Clusters | 305 |
| Noise Fraction | 0.0698 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.005 |
| open_door | 0.001 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.003 | 0.005 | 0.003 | 0.000 |

---

## Checkpoint 2,144 episodes — step 580,928

**Episodes:** 2,144  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9907 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0817 |
| Gradient Magnitude (Failure) | 1.0720 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5584 | 0.7205 |
| Neutral | 1.1778 | 0.9143 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0997 |
| Cosine Distance | 0.0004 |
| Clusters | 284 |
| Noise Fraction | 0.0907 |

---

## Checkpoint 2,240 episodes — step 605,968

**Episodes:** 2,240  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9803 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0795 |
| Gradient Magnitude (Failure) | 1.1014 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8002 | 0.6655 |
| Neutral | 1.1052 | 0.9497 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1922 |
| Cosine Distance | 0.0013 |
| Clusters | 279 |
| Noise Fraction | 0.0787 |

---

## Checkpoint 2,336 episodes — step 631,008

**Episodes:** 2,336  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9164 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0654 |
| Gradient Magnitude (Failure) | 1.1356 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6304 | 0.7312 |
| Neutral | 1.0670 | 0.9969 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1790 |
| Cosine Distance | 0.0011 |
| Clusters | 287 |
| Noise Fraction | 0.0847 |

---

## Checkpoint 2,417 episodes — step 656,048

**Episodes:** 2,417  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9703 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0612 |
| Gradient Magnitude (Failure) | 1.0920 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4771 | 0.7862 |
| Neutral | 1.0623 | 0.9454 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1732 |
| Cosine Distance | 0.0011 |
| Clusters | 284 |
| Noise Fraction | 0.0812 |

---

## Checkpoint 2,512 episodes — step 681,088

**Episodes:** 2,512  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9722 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0221 |
| Gradient Magnitude (Failure) | 1.0802 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2767 | 0.8417 |
| Neutral | 1.0344 | 0.9730 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1906 |
| Cosine Distance | 0.0013 |
| Clusters | 286 |
| Noise Fraction | 0.0870 |

---

## Checkpoint 2,608 episodes — step 706,128

**Episodes:** 2,608  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9771 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0320 |
| Gradient Magnitude (Failure) | 1.0607 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4550 | 0.6580 |
| Neutral | 1.0386 | 0.9976 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1213 |
| Cosine Distance | 0.0005 |
| Clusters | 279 |
| Noise Fraction | 0.0997 |

---

## Checkpoint 2,704 episodes — step 731,168

**Episodes:** 2,704  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9843 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0570 |
| Gradient Magnitude (Failure) | 1.0745 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2334 | 0.9156 |
| Neutral | 1.0521 | 0.9761 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0720 |
| Cosine Distance | 0.0002 |
| Clusters | 276 |
| Noise Fraction | 0.0768 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.007 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.009 |
| open_door | 0.001 | 0.001 | 0.000 | 0.010 |
| target_ball | 0.007 | 0.009 | 0.010 | 0.000 |

---

## Checkpoint 2,800 episodes — step 756,208

**Episodes:** 2,800  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9602 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0897 |
| Gradient Magnitude (Failure) | 1.0386 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6182 | 0.7871 |
| Neutral | 1.0838 | 0.9431 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0152 |
| Cosine Distance | 0.0000 |
| Clusters | 290 |
| Noise Fraction | 0.0722 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.008 |
| open_door | 0.000 | 0.001 | -0.000 | 0.008 |
| target_ball | 0.005 | 0.008 | 0.008 | 0.000 |

---

## Checkpoint 2,881 episodes — step 781,248

**Episodes:** 2,881  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9628 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0813 |
| Gradient Magnitude (Failure) | 1.0495 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8131 | 0.6388 |
| Neutral | 1.0773 | 0.9975 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0405 |
| Cosine Distance | 0.0001 |
| Clusters | 284 |
| Noise Fraction | 0.0655 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.008 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.009 |
| open_door | 0.000 | 0.001 | 0.000 | 0.010 |
| target_ball | 0.008 | 0.009 | 0.010 | 0.000 |

---

## Checkpoint 2,976 episodes — step 806,288

**Episodes:** 2,976  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9181 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0869 |
| Gradient Magnitude (Failure) | 1.0557 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5120 | 0.6083 |
| Neutral | 1.1099 | 0.9816 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1007 |
| Cosine Distance | 0.0004 |
| Clusters | 284 |
| Noise Fraction | 0.0879 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.001 | 0.001 | 0.000 | 0.007 |
| target_ball | 0.005 | 0.007 | 0.007 | 0.000 |

---

## Checkpoint 3,072 episodes — step 831,328

**Episodes:** 3,072  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9779 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1013 |
| Gradient Magnitude (Failure) | 1.0829 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2374 | 0.7898 |
| Neutral | 1.0877 | 0.9975 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1254 |
| Cosine Distance | 0.0005 |
| Clusters | 278 |
| Noise Fraction | 0.0892 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.000 | 0.001 | 0.000 | 0.006 |
| target_ball | 0.005 | 0.007 | 0.006 | -0.000 |

---

## Checkpoint 3,168 episodes — step 856,368

**Episodes:** 3,168  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9809 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0891 |
| Gradient Magnitude (Failure) | 1.0571 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4012 | 0.6708 |
| Neutral | 1.0853 | 0.9468 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1344 |
| Cosine Distance | 0.0006 |
| Clusters | 290 |
| Noise Fraction | 0.0851 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.001 | 0.001 | 0.000 | 0.006 |
| target_ball | 0.004 | 0.007 | 0.006 | 0.000 |

---

## Checkpoint 3,264 episodes — step 881,408

**Episodes:** 3,264  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.6663 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1296 |
| Gradient Magnitude (Failure) | 1.2285 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5710 | 0.5322 |
| Neutral | 1.1341 | 0.8178 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1132 |
| Cosine Distance | 0.0004 |
| Clusters | 276 |
| Noise Fraction | 0.0774 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.001 | 0.001 | -0.000 | 0.006 |
| target_ball | 0.004 | 0.007 | 0.006 | -0.000 |

---

## Checkpoint 3,345 episodes — step 906,448

**Episodes:** 3,345  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9637 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0715 |
| Gradient Magnitude (Failure) | 1.0806 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5518 | 0.7742 |
| Neutral | 1.0860 | 0.9570 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1082 |
| Cosine Distance | 0.0004 |
| Clusters | 273 |
| Noise Fraction | 0.0823 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.004 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.006 |
| open_door | 0.001 | 0.001 | 0.000 | 0.007 |
| target_ball | 0.004 | 0.006 | 0.007 | -0.000 |

---

## Checkpoint 3,440 episodes — step 931,488

**Episodes:** 3,440  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9513 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0806 |
| Gradient Magnitude (Failure) | 1.0777 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4533 | 0.6715 |
| Neutral | 1.1876 | 0.9536 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1190 |
| Cosine Distance | 0.0005 |
| Clusters | 273 |
| Noise Fraction | 0.0787 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | 0.000 |

---

## Checkpoint 3,536 episodes — step 956,528

**Episodes:** 3,536  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9385 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0802 |
| Gradient Magnitude (Failure) | 1.1384 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3009 | 0.8310 |
| Neutral | 1.0887 | 0.9862 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1174 |
| Cosine Distance | 0.0004 |
| Clusters | 280 |
| Noise Fraction | 0.0912 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.002 | 0.004 | 0.002 | 0.000 |

---

## Checkpoint 3,632 episodes — step 981,568

**Episodes:** 3,632  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9686 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0880 |
| Gradient Magnitude (Failure) | 1.0787 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5814 | 0.7144 |
| Neutral | 1.0834 | 0.9882 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0427 |
| Cosine Distance | 0.0000 |
| Clusters | 272 |
| Noise Fraction | 0.1016 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.003 | 0.003 | 0.000 |

---

## Checkpoint 3,728 episodes — step 1,006,608

**Episodes:** 3,728  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9779 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0823 |
| Gradient Magnitude (Failure) | 1.0712 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7822 | 0.6181 |
| Neutral | 1.1914 | 0.9031 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0235 |
| Cosine Distance | 0.0000 |
| Clusters | 279 |
| Noise Fraction | 0.0968 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.002 | 0.003 | 0.003 | 0.000 |

---

## Checkpoint 3,809 episodes — step 1,031,648

**Episodes:** 3,809  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9322 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1147 |
| Gradient Magnitude (Failure) | 1.0776 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6756 | 0.4563 |
| Neutral | 1.1064 | 0.9948 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0905 |
| Cosine Distance | 0.0002 |
| Clusters | 275 |
| Noise Fraction | 0.0856 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.003 |
| open_door | 0.001 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.002 | 0.003 | 0.003 | 0.000 |

---

## Checkpoint 3,904 episodes — step 1,056,688

**Episodes:** 3,904  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9891 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0872 |
| Gradient Magnitude (Failure) | 1.0736 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3591 | 0.7892 |
| Neutral | 1.1030 | 0.9910 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0484 |
| Cosine Distance | 0.0001 |
| Clusters | 273 |
| Noise Fraction | 0.0734 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 4,000 episodes — step 1,081,728

**Episodes:** 4,000  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9268 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0830 |
| Gradient Magnitude (Failure) | 1.1003 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6490 | 0.7461 |
| Neutral | 1.0847 | 0.9940 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0727 |
| Cosine Distance | 0.0001 |
| Clusters | 271 |
| Noise Fraction | 0.0737 |

---

## Checkpoint 4,096 episodes — step 1,106,768

**Episodes:** 4,096  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9911 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1138 |
| Gradient Magnitude (Failure) | 1.1158 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4380 | 0.8543 |
| Neutral | 1.0716 | 0.9639 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0602 |
| Cosine Distance | 0.0001 |
| Clusters | 272 |
| Noise Fraction | 0.0777 |

---

## Checkpoint 4,177 episodes — step 1,131,808

**Episodes:** 4,177  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9931 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0608 |
| Gradient Magnitude (Failure) | 1.0793 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5322 | 0.7242 |
| Neutral | 1.0770 | 0.9870 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0521 |
| Cosine Distance | 0.0001 |
| Clusters | 273 |
| Noise Fraction | 0.0701 |

---

## Checkpoint 4,272 episodes — step 1,156,848

**Episodes:** 4,272  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9763 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0814 |
| Gradient Magnitude (Failure) | 1.0716 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.1858 | 0.8101 |
| Neutral | 1.0588 | 0.9623 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0536 |
| Cosine Distance | 0.0001 |
| Clusters | 260 |
| Noise Fraction | 0.0515 |

---

## Checkpoint 4,368 episodes — step 1,181,888

**Episodes:** 4,368  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9736 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0715 |
| Gradient Magnitude (Failure) | 1.0804 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9609 | 0.5692 |
| Neutral | 1.0740 | 0.9923 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0138 |
| Cosine Distance | 0.0000 |
| Clusters | 269 |
| Noise Fraction | 0.0933 |

---

## Checkpoint 4,464 episodes — step 1,206,928

**Episodes:** 4,464  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8840 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0991 |
| Gradient Magnitude (Failure) | 1.1112 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3780 | 0.7172 |
| Neutral | 1.0839 | 0.9787 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0286 |
| Cosine Distance | 0.0000 |
| Clusters | 251 |
| Noise Fraction | 0.0674 |

---

## Checkpoint 4,560 episodes — step 1,231,968

**Episodes:** 4,560  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9798 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1203 |
| Gradient Magnitude (Failure) | 1.0736 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3522 | 0.8817 |
| Neutral | 1.1134 | 0.9989 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0785 |
| Cosine Distance | 0.0001 |
| Clusters | 289 |
| Noise Fraction | 0.0799 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.003 | 0.004 | 0.004 | -0.000 |

---

## Checkpoint 4,641 episodes — step 1,257,008

**Episodes:** 4,641  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9826 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1263 |
| Gradient Magnitude (Failure) | 1.0701 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2031 | 0.7571 |
| Neutral | 1.0743 | 0.9554 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0419 |
| Cosine Distance | 0.0000 |
| Clusters | 269 |
| Noise Fraction | 0.0824 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.003 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.005 |
| target_ball | 0.003 | 0.004 | 0.005 | 0.000 |

---

## Checkpoint 4,736 episodes — step 1,282,048

**Episodes:** 4,736  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9840 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0735 |
| Gradient Magnitude (Failure) | 1.0831 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8389 | 0.5983 |
| Neutral | 1.0759 | 0.9801 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0708 |
| Cosine Distance | 0.0001 |
| Clusters | 278 |
| Noise Fraction | 0.0690 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 4,832 episodes — step 1,307,088

**Episodes:** 4,832  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9571 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1525 |
| Gradient Magnitude (Failure) | 1.0621 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2879 | 0.9149 |
| Neutral | 1.0679 | 0.9224 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0975 |
| Cosine Distance | 0.0002 |
| Clusters | 281 |
| Noise Fraction | 0.0632 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.002 | 0.003 | 0.002 | 0.000 |

---

## Checkpoint 4,928 episodes — step 1,332,128

**Episodes:** 4,928  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9772 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1240 |
| Gradient Magnitude (Failure) | 1.0856 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2978 | 0.6504 |
| Neutral | 1.0922 | 0.8366 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1140 |
| Cosine Distance | 0.0002 |
| Clusters | 246 |
| Noise Fraction | 0.0640 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.002 | 0.003 | 0.000 |

---

## Checkpoint 5,024 episodes — step 1,357,168

**Episodes:** 5,024  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9197 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0813 |
| Gradient Magnitude (Failure) | 1.1028 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5843 | 0.5490 |
| Neutral | 1.0752 | 0.9465 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1219 |
| Cosine Distance | 0.0003 |
| Clusters | 278 |
| Noise Fraction | 0.0944 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | 0.000 |

---

## Checkpoint 5,105 episodes — step 1,382,208

**Episodes:** 5,105  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9045 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1115 |
| Gradient Magnitude (Failure) | 1.0961 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3366 | 0.8767 |
| Neutral | 1.0733 | 0.9668 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1226 |
| Cosine Distance | 0.0003 |
| Clusters | 269 |
| Noise Fraction | 0.0895 |

---

## Checkpoint 5,200 episodes — step 1,407,248

**Episodes:** 5,200  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9837 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0818 |
| Gradient Magnitude (Failure) | 1.0893 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4037 | 0.7542 |
| Neutral | 1.1343 | 0.9485 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1009 |
| Cosine Distance | 0.0002 |
| Clusters | 271 |
| Noise Fraction | 0.0698 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.002 | 0.004 | 0.002 | 0.000 |

---

## Checkpoint 5,296 episodes — step 1,432,288

**Episodes:** 5,296  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9918 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0967 |
| Gradient Magnitude (Failure) | 1.0818 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4725 | 0.8489 |
| Neutral | 1.0967 | 0.9944 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0889 |
| Cosine Distance | 0.0001 |
| Clusters | 259 |
| Noise Fraction | 0.0801 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.002 | 0.002 | 0.002 | -0.000 |

---

## Checkpoint 5,392 episodes — step 1,457,328

**Episodes:** 5,392  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9920 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0907 |
| Gradient Magnitude (Failure) | 1.0883 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5152 | 0.6684 |
| Neutral | 1.0883 | 0.9929 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1049 |
| Cosine Distance | 0.0002 |
| Clusters | 273 |
| Noise Fraction | 0.0795 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.004 | 0.004 | 0.003 | -0.000 |

---

## Checkpoint 5,488 episodes — step 1,482,368

**Episodes:** 5,488  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9638 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0797 |
| Gradient Magnitude (Failure) | 1.0667 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2424 | 0.8904 |
| Neutral | 1.1043 | 0.9178 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0702 |
| Cosine Distance | 0.0001 |
| Clusters | 276 |
| Noise Fraction | 0.1064 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.001 | 0.001 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.002 |
| open_door | 0.001 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.001 | 0.002 | 0.003 | 0.000 |

---

## Checkpoint 5,569 episodes — step 1,507,408

**Episodes:** 5,569  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9260 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0846 |
| Gradient Magnitude (Failure) | 1.0787 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5800 | 0.7858 |
| Neutral | 1.0711 | 0.9690 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0549 |
| Cosine Distance | 0.0001 |
| Clusters | 291 |
| Noise Fraction | 0.0590 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.001 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.003 |
| open_door | 0.001 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.002 | 0.003 | 0.004 | 0.000 |

---

## Checkpoint 5,664 episodes — step 1,532,448

**Episodes:** 5,664  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9623 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0504 |
| Gradient Magnitude (Failure) | 1.1146 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4661 | 0.7071 |
| Neutral | 1.0515 | 0.9918 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0545 |
| Cosine Distance | 0.0001 |
| Clusters | 280 |
| Noise Fraction | 0.0868 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.002 | 0.002 | 0.004 | 0.000 |

---

## Checkpoint 5,760 episodes — step 1,557,488

**Episodes:** 5,760  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9955 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0476 |
| Gradient Magnitude (Failure) | 1.0581 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6229 | 0.6819 |
| Neutral | 1.0555 | 0.9943 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1137 |
| Cosine Distance | 0.0002 |
| Clusters | 270 |
| Noise Fraction | 0.0744 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | 0.000 |

---

## Checkpoint 5,856 episodes — step 1,582,528

**Episodes:** 5,856  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9923 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0776 |
| Gradient Magnitude (Failure) | 1.0831 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3711 | 0.8281 |
| Neutral | 1.0477 | 0.9795 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1302 |
| Cosine Distance | 0.0003 |
| Clusters | 259 |
| Noise Fraction | 0.0617 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | -0.000 |

---

## Checkpoint 5,952 episodes — step 1,607,568

**Episodes:** 5,952  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9932 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0650 |
| Gradient Magnitude (Failure) | 1.0614 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5680 | 0.6388 |
| Neutral | 1.0743 | 0.9688 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0781 |
| Cosine Distance | 0.0001 |
| Clusters | 274 |
| Noise Fraction | 0.0829 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 6,033 episodes — step 1,632,608

**Episodes:** 6,033  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9108 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1401 |
| Gradient Magnitude (Failure) | 1.0619 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6673 | 0.8418 |
| Neutral | 1.1757 | 0.7670 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0294 |
| Cosine Distance | 0.0000 |
| Clusters | 295 |
| Noise Fraction | 0.0836 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 6,128 episodes — step 1,657,648

**Episodes:** 6,128  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9795 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0869 |
| Gradient Magnitude (Failure) | 1.0415 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4509 | 0.8209 |
| Neutral | 1.0930 | 0.9958 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0928 |
| Cosine Distance | 0.0002 |
| Clusters | 271 |
| Noise Fraction | 0.0838 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | 0.000 |

---

## Checkpoint 6,224 episodes — step 1,682,688

**Episodes:** 6,224  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9815 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0653 |
| Gradient Magnitude (Failure) | 1.0917 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6057 | 0.6192 |
| Neutral | 1.0891 | 0.9463 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1074 |
| Cosine Distance | 0.0002 |
| Clusters | 293 |
| Noise Fraction | 0.0833 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.003 | 0.003 | 0.003 | 0.000 |

---

## Checkpoint 6,320 episodes — step 1,707,728

**Episodes:** 6,320  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9519 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0619 |
| Gradient Magnitude (Failure) | 1.0817 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6145 | 0.7385 |
| Neutral | 1.1147 | 0.9684 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1037 |
| Cosine Distance | 0.0002 |
| Clusters | 289 |
| Noise Fraction | 0.1111 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | -0.000 |

---

## Checkpoint 6,416 episodes — step 1,732,768

**Episodes:** 6,416  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9591 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0733 |
| Gradient Magnitude (Failure) | 1.1648 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4958 | 0.8163 |
| Neutral | 1.0831 | 0.9946 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0304 |
| Cosine Distance | 0.0000 |
| Clusters | 267 |
| Noise Fraction | 0.0797 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.000 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.001 |
| target_ball | 0.000 | 0.001 | 0.001 | -0.000 |

---

## Checkpoint 6,497 episodes — step 1,757,808

**Episodes:** 6,497  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9925 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0656 |
| Gradient Magnitude (Failure) | 1.0751 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2764 | 0.8200 |
| Neutral | 1.0625 | 0.9971 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0961 |
| Cosine Distance | 0.0001 |
| Clusters | 290 |
| Noise Fraction | 0.0873 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.000 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.000 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 6,592 episodes — step 1,782,848

**Episodes:** 6,592  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9871 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0764 |
| Gradient Magnitude (Failure) | 1.1084 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4746 | 0.8551 |
| Neutral | 1.0593 | 0.9731 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0781 |
| Cosine Distance | 0.0001 |
| Clusters | 261 |
| Noise Fraction | 0.0822 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | -0.000 |

---

## Checkpoint 6,688 episodes — step 1,807,888

**Episodes:** 6,688  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9522 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0642 |
| Gradient Magnitude (Failure) | 1.0808 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2445 | 0.8255 |
| Neutral | 1.0533 | 0.9843 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1096 |
| Cosine Distance | 0.0002 |
| Clusters | 279 |
| Noise Fraction | 0.0845 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.000 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.000 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 6,784 episodes — step 1,832,928

**Episodes:** 6,784  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9820 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0613 |
| Gradient Magnitude (Failure) | 1.0674 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5413 | 0.6974 |
| Neutral | 1.0834 | 0.9800 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1331 |
| Cosine Distance | 0.0003 |
| Clusters | 284 |
| Noise Fraction | 0.1117 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.002 | 0.001 | 0.000 |

---

## Checkpoint 6,880 episodes — step 1,857,968

**Episodes:** 6,880  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9690 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0733 |
| Gradient Magnitude (Failure) | 1.0825 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8465 | 0.5629 |
| Neutral | 1.0737 | 0.9904 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1051 |
| Cosine Distance | 0.0002 |
| Clusters | 288 |
| Noise Fraction | 0.0822 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | -0.000 |

---

## Checkpoint 6,961 episodes — step 1,883,008

**Episodes:** 6,961  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9739 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0855 |
| Gradient Magnitude (Failure) | 1.0828 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3512 | 0.7861 |
| Neutral | 1.0979 | 0.9749 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1168 |
| Cosine Distance | 0.0002 |
| Clusters | 284 |
| Noise Fraction | 0.0855 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.002 | 0.001 | 0.000 |

---

## Checkpoint 7,056 episodes — step 1,908,048

**Episodes:** 7,056  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9929 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0993 |
| Gradient Magnitude (Failure) | 1.0846 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5576 | 0.8266 |
| Neutral | 1.0802 | 0.9775 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0862 |
| Cosine Distance | 0.0001 |
| Clusters | 256 |
| Noise Fraction | 0.0666 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.000 |
| target_ball | 0.001 | 0.001 | 0.000 | 0.000 |

---

## Checkpoint 7,152 episodes — step 1,933,088

**Episodes:** 7,152  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9482 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1018 |
| Gradient Magnitude (Failure) | 1.0691 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5727 | 0.5855 |
| Neutral | 1.1280 | 0.8893 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0604 |
| Cosine Distance | 0.0001 |
| Clusters | 278 |
| Noise Fraction | 0.1039 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 7,248 episodes — step 1,958,128

**Episodes:** 7,248  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9664 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1256 |
| Gradient Magnitude (Failure) | 1.0679 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7408 | 0.4251 |
| Neutral | 1.0828 | 0.9671 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1227 |
| Cosine Distance | 0.0003 |
| Clusters | 277 |
| Noise Fraction | 0.0986 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.004 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.004 | 0.004 | 0.004 | 0.000 |

---

## Checkpoint 7,344 episodes — step 1,983,168

**Episodes:** 7,344  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9592 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0823 |
| Gradient Magnitude (Failure) | 1.0892 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3486 | 0.8318 |
| Neutral | 1.0938 | 0.9730 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0506 |
| Cosine Distance | 0.0001 |
| Clusters | 277 |
| Noise Fraction | 0.0836 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.005 |
| target_ball | 0.004 | 0.004 | 0.005 | 0.000 |

---

## Checkpoint 7,425 episodes — step 2,008,208

**Episodes:** 7,425  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9833 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0889 |
| Gradient Magnitude (Failure) | 1.0763 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2994 | 0.8479 |
| Neutral | 1.0962 | 0.9817 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0324 |
| Cosine Distance | 0.0000 |
| Clusters | 265 |
| Noise Fraction | 0.0860 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.002 | 0.003 | 0.000 |

---

## Checkpoint 7,520 episodes — step 2,033,248

**Episodes:** 7,520  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9693 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0994 |
| Gradient Magnitude (Failure) | 1.0811 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6873 | 0.7538 |
| Neutral | 1.1014 | 0.9968 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0923 |
| Cosine Distance | 0.0001 |
| Clusters | 266 |
| Noise Fraction | 0.0834 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.003 | 0.003 | 0.003 | 0.000 |

---

## Checkpoint 7,616 episodes — step 2,058,288

**Episodes:** 7,616  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9587 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1019 |
| Gradient Magnitude (Failure) | 1.0831 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4724 | 0.8000 |
| Neutral | 1.0852 | 0.9847 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0912 |
| Cosine Distance | 0.0001 |
| Clusters | 269 |
| Noise Fraction | 0.0849 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.002 | 0.001 | 0.003 | 0.000 |

---

## Checkpoint 7,712 episodes — step 2,083,328

**Episodes:** 7,712  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9382 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0896 |
| Gradient Magnitude (Failure) | 1.1206 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6335 | 0.7075 |
| Neutral | 1.0909 | 0.9869 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1068 |
| Cosine Distance | 0.0002 |
| Clusters | 272 |
| Noise Fraction | 0.0856 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | -0.000 | 0.003 |
| target_ball | 0.002 | 0.001 | 0.003 | 0.000 |

---

## Checkpoint 7,808 episodes — step 2,108,368

**Episodes:** 7,808  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9776 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1026 |
| Gradient Magnitude (Failure) | 1.1074 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4806 | 0.7917 |
| Neutral | 1.0901 | 0.9793 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0310 |
| Cosine Distance | 0.0000 |
| Clusters | 251 |
| Noise Fraction | 0.0780 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.001 | 0.003 | 0.000 |

---

## Checkpoint 7,889 episodes — step 2,133,408

**Episodes:** 7,889  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9731 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0969 |
| Gradient Magnitude (Failure) | 1.0786 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5737 | 0.7264 |
| Neutral | 1.0967 | 0.9932 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0398 |
| Cosine Distance | 0.0000 |
| Clusters | 281 |
| Noise Fraction | 0.0683 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.001 | 0.003 | 0.000 |

---

## Checkpoint 7,984 episodes — step 2,158,448

**Episodes:** 7,984  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9767 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0814 |
| Gradient Magnitude (Failure) | 1.0825 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6714 | 0.6437 |
| Neutral | 1.0906 | 0.9800 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1013 |
| Cosine Distance | 0.0002 |
| Clusters | 278 |
| Noise Fraction | 0.0788 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.001 | 0.002 | 0.000 |

---

## Checkpoint 8,080 episodes — step 2,183,488

**Episodes:** 8,080  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9428 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1199 |
| Gradient Magnitude (Failure) | 1.0857 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3364 | 0.8238 |
| Neutral | 1.1021 | 0.9759 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0089 |
| Cosine Distance | 0.0000 |
| Clusters | 284 |
| Noise Fraction | 0.0773 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | -0.000 |

---

## Checkpoint 8,176 episodes — step 2,208,528

**Episodes:** 8,176  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9686 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0886 |
| Gradient Magnitude (Failure) | 1.1425 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5719 | 0.7926 |
| Neutral | 1.0883 | 0.9566 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0428 |
| Cosine Distance | 0.0000 |
| Clusters | 266 |
| Noise Fraction | 0.0728 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.002 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.002 | 0.003 | 0.002 | -0.000 |

---

## Checkpoint 8,272 episodes — step 2,233,568

**Episodes:** 8,272  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8261 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1263 |
| Gradient Magnitude (Failure) | 1.1118 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6586 | 0.8436 |
| Neutral | 1.0933 | 0.8457 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0143 |
| Cosine Distance | 0.0000 |
| Clusters | 281 |
| Noise Fraction | 0.0826 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | 0.000 | 0.001 | 0.001 |
| open_door | 0.000 | 0.001 | 0.000 | 0.001 |
| target_ball | 0.001 | 0.001 | 0.001 | 0.000 |

---

## Checkpoint 8,353 episodes — step 2,258,608

**Episodes:** 8,353  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8971 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1213 |
| Gradient Magnitude (Failure) | 1.0812 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4129 | 0.6292 |
| Neutral | 1.0845 | 0.9873 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0518 |
| Cosine Distance | 0.0000 |
| Clusters | 274 |
| Noise Fraction | 0.0762 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.000 | 0.000 | 0.003 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | -0.000 | 0.005 |
| target_ball | 0.003 | 0.004 | 0.005 | 0.000 |

---

## Checkpoint 8,448 episodes — step 2,283,648

**Episodes:** 8,448  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9894 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1117 |
| Gradient Magnitude (Failure) | 1.0914 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4651 | 0.8412 |
| Neutral | 1.1452 | 0.8399 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0966 |
| Cosine Distance | 0.0002 |
| Clusters | 258 |
| Noise Fraction | 0.0531 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.002 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 8,544 episodes — step 2,308,688

**Episodes:** 8,544  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9875 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1031 |
| Gradient Magnitude (Failure) | 1.0955 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3561 | 0.6993 |
| Neutral | 1.0859 | 0.9270 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0529 |
| Cosine Distance | 0.0000 |
| Clusters | 275 |
| Noise Fraction | 0.0865 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 8,640 episodes — step 2,333,728

**Episodes:** 8,640  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9559 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0723 |
| Gradient Magnitude (Failure) | 1.1124 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5597 | 0.7028 |
| Neutral | 1.1070 | 0.9619 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0727 |
| Cosine Distance | 0.0001 |
| Clusters | 264 |
| Noise Fraction | 0.0885 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.000 | 0.000 | 0.001 |
| locked_door | 0.000 | -0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 8,736 episodes — step 2,358,768

**Episodes:** 8,736  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9679 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1256 |
| Gradient Magnitude (Failure) | 1.0877 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8940 | 0.3261 |
| Neutral | 1.1006 | 0.8653 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0965 |
| Cosine Distance | 0.0002 |
| Clusters | 287 |
| Noise Fraction | 0.0771 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.003 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.003 | 0.003 | 0.004 | 0.000 |

---

## Checkpoint 8,817 episodes — step 2,383,808

**Episodes:** 8,817  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9471 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0771 |
| Gradient Magnitude (Failure) | 1.1456 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7829 | 0.5649 |
| Neutral | 1.0811 | 0.9673 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1151 |
| Cosine Distance | 0.0002 |
| Clusters | 270 |
| Noise Fraction | 0.0916 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.004 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.003 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.004 | 0.003 | 0.004 | 0.000 |

---

## Checkpoint 8,912 episodes — step 2,408,848

**Episodes:** 8,912  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9746 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1032 |
| Gradient Magnitude (Failure) | 1.0681 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5195 | 0.8280 |
| Neutral | 1.0886 | 0.9566 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0383 |
| Cosine Distance | 0.0000 |
| Clusters | 270 |
| Noise Fraction | 0.0815 |

---

## Checkpoint 9,008 episodes — step 2,433,888

**Episodes:** 9,008  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9849 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0873 |
| Gradient Magnitude (Failure) | 1.0848 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4158 | 0.8119 |
| Neutral | 1.0885 | 0.9700 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0674 |
| Cosine Distance | 0.0001 |
| Clusters | 276 |
| Noise Fraction | 0.0831 |

---

## Checkpoint 9,104 episodes — step 2,458,928

**Episodes:** 9,104  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9184 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0805 |
| Gradient Magnitude (Failure) | 1.1724 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3022 | 0.8093 |
| Neutral | 1.0794 | 0.9951 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0631 |
| Cosine Distance | 0.0001 |
| Clusters | 292 |
| Noise Fraction | 0.0938 |

---

## Checkpoint 9,185 episodes — step 2,483,968

**Episodes:** 9,185  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9816 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1101 |
| Gradient Magnitude (Failure) | 1.0876 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6731 | 0.7694 |
| Neutral | 1.0939 | 0.9451 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1179 |
| Cosine Distance | 0.0002 |
| Clusters | 272 |
| Noise Fraction | 0.0944 |

---

## Checkpoint 9,280 episodes — step 2,509,008

**Episodes:** 9,280  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9784 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0878 |
| Gradient Magnitude (Failure) | 1.0821 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5211 | 0.7171 |
| Neutral | 1.1239 | 0.9654 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1278 |
| Cosine Distance | 0.0002 |
| Clusters | 286 |
| Noise Fraction | 0.0784 |

---

## Checkpoint 9,376 episodes — step 2,534,048

**Episodes:** 9,376  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9602 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0937 |
| Gradient Magnitude (Failure) | 1.0960 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8109 | 0.5474 |
| Neutral | 1.1011 | 0.9756 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1191 |
| Cosine Distance | 0.0002 |
| Clusters | 277 |
| Noise Fraction | 0.0804 |

---

## Checkpoint 9,472 episodes — step 2,559,088

**Episodes:** 9,472  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9887 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1283 |
| Gradient Magnitude (Failure) | 1.1250 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7273 | 0.7821 |
| Neutral | 1.1241 | 0.9939 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0782 |
| Cosine Distance | 0.0001 |
| Clusters | 266 |
| Noise Fraction | 0.0945 |

---

## Checkpoint 9,568 episodes — step 2,584,128

**Episodes:** 9,568  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9661 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1082 |
| Gradient Magnitude (Failure) | 1.1478 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7130 | 0.6891 |
| Neutral | 1.1015 | 0.9937 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0278 |
| Cosine Distance | 0.0000 |
| Clusters | 273 |
| Noise Fraction | 0.0945 |

---

## Checkpoint 9,649 episodes — step 2,609,168

**Episodes:** 9,649  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9890 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1293 |
| Gradient Magnitude (Failure) | 1.1134 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6130 | 0.7775 |
| Neutral | 1.1162 | 0.9832 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0180 |
| Cosine Distance | 0.0000 |
| Clusters | 274 |
| Noise Fraction | 0.0950 |

---

## Checkpoint 9,744 episodes — step 2,634,208

**Episodes:** 9,744  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9771 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1197 |
| Gradient Magnitude (Failure) | 1.1083 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3315 | 0.8414 |
| Neutral | 1.1208 | 0.9949 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0215 |
| Cosine Distance | 0.0000 |
| Clusters | 282 |
| Noise Fraction | 0.0667 |

---

## Checkpoint 9,840 episodes — step 2,659,248

**Episodes:** 9,840  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9859 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1095 |
| Gradient Magnitude (Failure) | 1.0986 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4790 | 0.8023 |
| Neutral | 1.1095 | 0.9819 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0481 |
| Cosine Distance | 0.0000 |
| Clusters | 273 |
| Noise Fraction | 0.0745 |

---

## Checkpoint 9,936 episodes — step 2,684,288

**Episodes:** 9,936  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9570 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1039 |
| Gradient Magnitude (Failure) | 1.0884 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8097 | 0.4434 |
| Neutral | 1.0723 | 0.9293 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0766 |
| Cosine Distance | 0.0001 |
| Clusters | 260 |
| Noise Fraction | 0.0691 |

---

## Checkpoint 10,032 episodes — step 2,709,328

**Episodes:** 10,032  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9754 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0936 |
| Gradient Magnitude (Failure) | 1.0940 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5482 | 0.5353 |
| Neutral | 1.1056 | 0.8990 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1232 |
| Cosine Distance | 0.0003 |
| Clusters | 267 |
| Noise Fraction | 0.0891 |

---

## Checkpoint 10,113 episodes — step 2,734,368

**Episodes:** 10,113  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9384 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1464 |
| Gradient Magnitude (Failure) | 1.0863 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2389 | 0.7867 |
| Neutral | 1.1890 | 0.9850 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1160 |
| Cosine Distance | 0.0002 |
| Clusters | 276 |
| Noise Fraction | 0.1001 |

---

## Checkpoint 10,208 episodes — step 2,759,408

**Episodes:** 10,208  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9675 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1206 |
| Gradient Magnitude (Failure) | 1.0857 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4829 | 0.6414 |
| Neutral | 1.1224 | 0.9154 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1273 |
| Cosine Distance | 0.0003 |
| Clusters | 284 |
| Noise Fraction | 0.0749 |

---

## Checkpoint 10,304 episodes — step 2,784,448

**Episodes:** 10,304  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9925 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0961 |
| Gradient Magnitude (Failure) | 1.0858 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6825 | 0.6956 |
| Neutral | 1.0986 | 0.9947 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1146 |
| Cosine Distance | 0.0002 |
| Clusters | 278 |
| Noise Fraction | 0.0721 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.006 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.005 | 0.006 | 0.004 | 0.000 |

---

## Checkpoint 10,400 episodes — step 2,809,488

**Episodes:** 10,400  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9072 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1730 |
| Gradient Magnitude (Failure) | 1.0736 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4424 | 0.8517 |
| Neutral | 1.0886 | 0.9203 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0882 |
| Cosine Distance | 0.0001 |
| Clusters | 280 |
| Noise Fraction | 0.0761 |

---

## Checkpoint 10,496 episodes — step 2,834,528

**Episodes:** 10,496  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9711 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1228 |
| Gradient Magnitude (Failure) | 1.1033 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3723 | 0.7288 |
| Neutral | 1.1024 | 0.9882 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0764 |
| Cosine Distance | 0.0001 |
| Clusters | 288 |
| Noise Fraction | 0.0860 |

---

## Checkpoint 10,577 episodes — step 2,859,568

**Episodes:** 10,577  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9857 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1046 |
| Gradient Magnitude (Failure) | 1.0653 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0083 | 0.5913 |
| Neutral | 1.2514 | 0.8525 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0302 |
| Cosine Distance | 0.0000 |
| Clusters | 283 |
| Noise Fraction | 0.0987 |

---

## Checkpoint 10,672 episodes — step 2,884,608

**Episodes:** 10,672  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9148 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0819 |
| Gradient Magnitude (Failure) | 1.1741 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5164 | 0.7619 |
| Neutral | 1.0811 | 0.9951 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0856 |
| Cosine Distance | 0.0001 |
| Clusters | 284 |
| Noise Fraction | 0.0797 |

---

## Checkpoint 10,768 episodes — step 2,909,648

**Episodes:** 10,768  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9934 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0626 |
| Gradient Magnitude (Failure) | 1.0716 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5622 | 0.6843 |
| Neutral | 1.0755 | 0.9679 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1078 |
| Cosine Distance | 0.0002 |
| Clusters | 275 |
| Noise Fraction | 0.0824 |

---

## Checkpoint 10,864 episodes — step 2,934,688

**Episodes:** 10,864  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9747 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0542 |
| Gradient Magnitude (Failure) | 1.1039 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4013 | 0.8008 |
| Neutral | 1.0628 | 0.9959 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1048 |
| Cosine Distance | 0.0002 |
| Clusters | 271 |
| Noise Fraction | 0.0721 |

---

## Checkpoint 10,960 episodes — step 2,959,728

**Episodes:** 10,960  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9764 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0875 |
| Gradient Magnitude (Failure) | 1.0864 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2254 | 0.8485 |
| Neutral | 1.1359 | 0.9372 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0362 |
| Cosine Distance | 0.0000 |
| Clusters | 272 |
| Noise Fraction | 0.1083 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.004 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.005 |
| open_door | 0.000 | 0.001 | 0.000 | 0.004 |
| target_ball | 0.004 | 0.005 | 0.004 | 0.000 |

---

## Checkpoint 11,041 episodes — step 2,984,768

**Episodes:** 11,041  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9709 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1009 |
| Gradient Magnitude (Failure) | 1.0807 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7132 | 0.7229 |
| Neutral | 1.0911 | 0.9914 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0324 |
| Cosine Distance | 0.0000 |
| Clusters | 275 |
| Noise Fraction | 0.0782 |

---

## Checkpoint 11,136 episodes — step 3,009,808

**Episodes:** 11,136  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9837 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0745 |
| Gradient Magnitude (Failure) | 1.0719 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4371 | 0.7896 |
| Neutral | 1.1188 | 0.9347 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0458 |
| Cosine Distance | 0.0001 |
| Clusters | 284 |
| Noise Fraction | 0.0784 |

---

## Checkpoint 11,232 episodes — step 3,034,848

**Episodes:** 11,232  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8571 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1703 |
| Gradient Magnitude (Failure) | 1.0721 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4791 | 0.9185 |
| Neutral | 1.0652 | 0.8814 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0664 |
| Cosine Distance | 0.0001 |
| Clusters | 282 |
| Noise Fraction | 0.1127 |

---

## Checkpoint 11,328 episodes — step 3,059,888

**Episodes:** 11,328  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9712 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0916 |
| Gradient Magnitude (Failure) | 1.0632 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7211 | 0.5917 |
| Neutral | 1.1049 | 0.9919 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0233 |
| Cosine Distance | 0.0000 |
| Clusters | 287 |
| Noise Fraction | 0.0651 |

---

## Checkpoint 11,424 episodes — step 3,084,928

**Episodes:** 11,424  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9881 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0804 |
| Gradient Magnitude (Failure) | 1.0527 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4958 | 0.7812 |
| Neutral | 1.0801 | 0.9924 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0223 |
| Cosine Distance | 0.0000 |
| Clusters | 301 |
| Noise Fraction | 0.0951 |

---

## Checkpoint 11,505 episodes — step 3,109,968

**Episodes:** 11,505  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9714 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1658 |
| Gradient Magnitude (Failure) | 1.0882 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6091 | 0.4039 |
| Neutral | 1.0906 | 0.9656 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0851 |
| Cosine Distance | 0.0001 |
| Clusters | 275 |
| Noise Fraction | 0.0974 |

---

## Checkpoint 11,600 episodes — step 3,135,008

**Episodes:** 11,600  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9518 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1292 |
| Gradient Magnitude (Failure) | 1.0743 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5349 | 0.8421 |
| Neutral | 1.1074 | 0.9814 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0334 |
| Cosine Distance | 0.0000 |
| Clusters | 298 |
| Noise Fraction | 0.0917 |

---

## Checkpoint 11,696 episodes — step 3,160,048

**Episodes:** 11,696  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9433 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1275 |
| Gradient Magnitude (Failure) | 1.0785 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5869 | 0.8435 |
| Neutral | 1.0795 | 0.9259 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0980 |
| Cosine Distance | 0.0001 |
| Clusters | 283 |
| Noise Fraction | 0.0881 |

---

## Checkpoint 11,792 episodes — step 3,185,088

**Episodes:** 11,792  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9940 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1085 |
| Gradient Magnitude (Failure) | 1.0944 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7136 | 0.4951 |
| Neutral | 1.0862 | 0.9632 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1070 |
| Cosine Distance | 0.0002 |
| Clusters | 284 |
| Noise Fraction | 0.0757 |

---

## Checkpoint 11,888 episodes — step 3,210,128

**Episodes:** 11,888  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9297 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1173 |
| Gradient Magnitude (Failure) | 1.0839 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6589 | 0.4974 |
| Neutral | 1.0951 | 0.9803 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1065 |
| Cosine Distance | 0.0002 |
| Clusters | 277 |
| Noise Fraction | 0.0765 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.000 | 0.001 | 0.000 | 0.006 |
| target_ball | 0.005 | 0.007 | 0.006 | 0.000 |

---

## Checkpoint 11,969 episodes — step 3,235,168

**Episodes:** 11,969  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9872 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1009 |
| Gradient Magnitude (Failure) | 1.0826 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7529 | 0.7571 |
| Neutral | 1.1012 | 0.9079 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0811 |
| Cosine Distance | 0.0001 |
| Clusters | 289 |
| Noise Fraction | 0.0735 |

---

## Checkpoint 12,064 episodes — step 3,260,208

**Episodes:** 12,064  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9943 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0797 |
| Gradient Magnitude (Failure) | 1.0817 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6323 | 0.7787 |
| Neutral | 1.0614 | 0.9804 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1441 |
| Cosine Distance | 0.0004 |
| Clusters | 302 |
| Noise Fraction | 0.0680 |

---

## Checkpoint 12,160 episodes — step 3,285,248

**Episodes:** 12,160  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8925 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0667 |
| Gradient Magnitude (Failure) | 1.1955 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2604 | 0.9074 |
| Neutral | 1.0706 | 0.9864 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1096 |
| Cosine Distance | 0.0002 |
| Clusters | 286 |
| Noise Fraction | 0.0957 |

---

## Checkpoint 12,256 episodes — step 3,310,288

**Episodes:** 12,256  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9830 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0833 |
| Gradient Magnitude (Failure) | 1.0720 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3393 | 0.8434 |
| Neutral | 1.1236 | 0.9518 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0504 |
| Cosine Distance | 0.0001 |
| Clusters | 282 |
| Noise Fraction | 0.0825 |

---

## Checkpoint 12,352 episodes — step 3,335,328

**Episodes:** 12,352  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9771 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0813 |
| Gradient Magnitude (Failure) | 1.0695 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6885 | 0.6620 |
| Neutral | 1.0878 | 0.9935 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1031 |
| Cosine Distance | 0.0002 |
| Clusters | 273 |
| Noise Fraction | 0.0690 |

---

## Checkpoint 12,433 episodes — step 3,360,368

**Episodes:** 12,433  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9721 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0891 |
| Gradient Magnitude (Failure) | 1.0721 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9260 | 0.5335 |
| Neutral | 1.1191 | 0.9612 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1241 |
| Cosine Distance | 0.0003 |
| Clusters | 281 |
| Noise Fraction | 0.0942 |

---

## Checkpoint 12,528 episodes — step 3,385,408

**Episodes:** 12,528  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9789 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0838 |
| Gradient Magnitude (Failure) | 1.0824 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8292 | 0.5059 |
| Neutral | 1.0880 | 0.9942 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0789 |
| Cosine Distance | 0.0001 |
| Clusters | 283 |
| Noise Fraction | 0.0918 |

---

## Checkpoint 12,624 episodes — step 3,410,448

**Episodes:** 12,624  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9050 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1209 |
| Gradient Magnitude (Failure) | 1.0817 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8841 | 0.7572 |
| Neutral | 1.1109 | 0.9914 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0230 |
| Cosine Distance | 0.0000 |
| Clusters | 285 |
| Noise Fraction | 0.0742 |

---

## Checkpoint 12,720 episodes — step 3,435,488

**Episodes:** 12,720  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9640 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1060 |
| Gradient Magnitude (Failure) | 1.0847 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3077 | 0.8458 |
| Neutral | 1.1063 | 0.9948 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0483 |
| Cosine Distance | 0.0000 |
| Clusters | 293 |
| Noise Fraction | 0.0740 |

---

## Checkpoint 12,816 episodes — step 3,460,528

**Episodes:** 12,816  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9459 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1120 |
| Gradient Magnitude (Failure) | 1.1165 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6710 | 0.6652 |
| Neutral | 1.1143 | 0.9816 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0429 |
| Cosine Distance | 0.0000 |
| Clusters | 281 |
| Noise Fraction | 0.0892 |

---

## Checkpoint 12,897 episodes — step 3,485,568

**Episodes:** 12,897  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9864 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0901 |
| Gradient Magnitude (Failure) | 1.0858 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5248 | 0.7473 |
| Neutral | 1.1094 | 0.9818 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1119 |
| Cosine Distance | 0.0002 |
| Clusters | 300 |
| Noise Fraction | 0.0872 |

---

## Checkpoint 12,992 episodes — step 3,510,608

**Episodes:** 12,992  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9549 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0888 |
| Gradient Magnitude (Failure) | 1.1158 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6823 | 0.6827 |
| Neutral | 1.0902 | 0.9932 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1529 |
| Cosine Distance | 0.0004 |
| Clusters | 278 |
| Noise Fraction | 0.0716 |

---

## Checkpoint 13,088 episodes — step 3,535,648

**Episodes:** 13,088  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8558 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1268 |
| Gradient Magnitude (Failure) | 1.1296 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6298 | 0.5228 |
| Neutral | 1.1144 | 0.9059 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1147 |
| Cosine Distance | 0.0002 |
| Clusters | 283 |
| Noise Fraction | 0.0716 |

---

## Checkpoint 13,184 episodes — step 3,560,688

**Episodes:** 13,184  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9788 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1010 |
| Gradient Magnitude (Failure) | 1.1002 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4016 | 0.7675 |
| Neutral | 1.0878 | 0.9799 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1334 |
| Cosine Distance | 0.0003 |
| Clusters | 264 |
| Noise Fraction | 0.0538 |

---

## Checkpoint 13,280 episodes — step 3,585,728

**Episodes:** 13,280  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9574 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0954 |
| Gradient Magnitude (Failure) | 1.0885 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4363 | 0.6167 |
| Neutral | 1.0951 | 0.8392 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2152 |
| Cosine Distance | 0.0008 |
| Clusters | 291 |
| Noise Fraction | 0.0898 |

---

## Checkpoint 13,361 episodes — step 3,610,768

**Episodes:** 13,361  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8058 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0858 |
| Gradient Magnitude (Failure) | 1.1555 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4823 | 0.8343 |
| Neutral | 1.0575 | 0.9574 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1741 |
| Cosine Distance | 0.0006 |
| Clusters | 289 |
| Noise Fraction | 0.0900 |

---

## Checkpoint 13,456 episodes — step 3,635,808

**Episodes:** 13,456  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9590 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0725 |
| Gradient Magnitude (Failure) | 1.0855 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8030 | 0.6515 |
| Neutral | 1.0931 | 0.9887 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0594 |
| Cosine Distance | 0.0001 |
| Clusters | 275 |
| Noise Fraction | 0.0799 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.001 | 0.006 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.007 |
| open_door | 0.001 | 0.001 | 0.000 | 0.010 |
| target_ball | 0.006 | 0.007 | 0.010 | -0.000 |

---

## Checkpoint 13,552 episodes — step 3,660,848

**Episodes:** 13,552  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9725 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0943 |
| Gradient Magnitude (Failure) | 1.1003 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2991 | 0.8181 |
| Neutral | 1.1023 | 0.9629 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0439 |
| Cosine Distance | 0.0001 |
| Clusters | 277 |
| Noise Fraction | 0.0825 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.001 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.001 | 0.001 | -0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 13,648 episodes — step 3,685,888

**Episodes:** 13,648  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9827 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0987 |
| Gradient Magnitude (Failure) | 1.0677 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4916 | 0.7689 |
| Neutral | 1.0905 | 0.9832 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0719 |
| Cosine Distance | 0.0001 |
| Clusters | 271 |
| Noise Fraction | 0.0879 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.001 | 0.002 | 0.003 | -0.000 |

---

## Checkpoint 13,744 episodes — step 3,710,928

**Episodes:** 13,744  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9036 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0902 |
| Gradient Magnitude (Failure) | 1.1112 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4068 | 0.8422 |
| Neutral | 1.1221 | 0.8984 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0818 |
| Cosine Distance | 0.0001 |
| Clusters | 271 |
| Noise Fraction | 0.0724 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 13,825 episodes — step 3,735,968

**Episodes:** 13,825  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9264 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0838 |
| Gradient Magnitude (Failure) | 1.1864 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4752 | 0.8398 |
| Neutral | 1.0958 | 0.9797 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0990 |
| Cosine Distance | 0.0002 |
| Clusters | 274 |
| Noise Fraction | 0.0804 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 13,920 episodes — step 3,761,008

**Episodes:** 13,920  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9546 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0733 |
| Gradient Magnitude (Failure) | 1.0721 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7197 | 0.6610 |
| Neutral | 1.0708 | 0.9894 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1298 |
| Cosine Distance | 0.0003 |
| Clusters | 284 |
| Noise Fraction | 0.0836 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.001 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.002 |
| target_ball | 0.001 | 0.002 | 0.002 | 0.000 |

---

## Checkpoint 14,016 episodes — step 3,786,048

**Episodes:** 14,016  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9049 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0828 |
| Gradient Magnitude (Failure) | 1.0988 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3144 | 0.8172 |
| Neutral | 1.0780 | 0.9841 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1436 |
| Cosine Distance | 0.0003 |
| Clusters | 282 |
| Noise Fraction | 0.0894 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.001 | 0.000 | 0.005 |
| locked_door | 0.001 | 0.000 | 0.001 | 0.004 |
| open_door | 0.000 | 0.001 | 0.000 | 0.006 |
| target_ball | 0.005 | 0.004 | 0.006 | 0.000 |

---

## Checkpoint 14,112 episodes — step 3,811,088

**Episodes:** 14,112  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9924 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0698 |
| Gradient Magnitude (Failure) | 1.0679 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7116 | 0.6551 |
| Neutral | 1.0790 | 0.9746 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1235 |
| Cosine Distance | 0.0003 |
| Clusters | 278 |
| Noise Fraction | 0.0925 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.001 | 0.000 | 0.002 |
| locked_door | 0.001 | -0.000 | 0.001 | 0.002 |
| open_door | 0.000 | 0.001 | 0.000 | 0.003 |
| target_ball | 0.002 | 0.002 | 0.003 | 0.000 |

---

## Checkpoint 14,193 episodes — step 3,836,128

**Episodes:** 14,193  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9952 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0644 |
| Gradient Magnitude (Failure) | 1.0663 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7779 | 0.5927 |
| Neutral | 1.0637 | 0.9983 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0948 |
| Cosine Distance | 0.0001 |
| Clusters | 294 |
| Noise Fraction | 0.0887 |

---

## Checkpoint 14,288 episodes — step 3,861,168

**Episodes:** 14,288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9767 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0833 |
| Gradient Magnitude (Failure) | 1.0768 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5157 | 0.7550 |
| Neutral | 1.0856 | 0.9979 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0337 |
| Cosine Distance | 0.0000 |
| Clusters | 281 |
| Noise Fraction | 0.0653 |

---

## Checkpoint 14,384 episodes — step 3,886,208

**Episodes:** 14,384  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9673 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1242 |
| Gradient Magnitude (Failure) | 1.0766 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4470 | 0.7956 |
| Neutral | 1.1115 | 0.9766 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0167 |
| Cosine Distance | 0.0000 |
| Clusters | 288 |
| Noise Fraction | 0.0672 |

---

## Checkpoint 14,480 episodes — step 3,911,248

**Episodes:** 14,480  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9684 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1095 |
| Gradient Magnitude (Failure) | 1.1130 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5906 | 0.7840 |
| Neutral | 1.1026 | 0.9732 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0784 |
| Cosine Distance | 0.0001 |
| Clusters | 281 |
| Noise Fraction | 0.0521 |

---

## Checkpoint 14,576 episodes — step 3,936,288

**Episodes:** 14,576  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9930 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0934 |
| Gradient Magnitude (Failure) | 1.1045 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0292 | 0.6545 |
| Neutral | 1.0868 | 0.9703 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1551 |
| Cosine Distance | 0.0004 |
| Clusters | 291 |
| Noise Fraction | 0.0502 |

---

## Checkpoint 14,657 episodes — step 3,961,328

**Episodes:** 14,657  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9679 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1271 |
| Gradient Magnitude (Failure) | 1.1025 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6042 | 0.4646 |
| Neutral | 1.1539 | 0.7759 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1179 |
| Cosine Distance | 0.0002 |
| Clusters | 294 |
| Noise Fraction | 0.0740 |

---

## Checkpoint 14,752 episodes — step 3,986,368

**Episodes:** 14,752  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9887 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0827 |
| Gradient Magnitude (Failure) | 1.0873 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5804 | 0.6364 |
| Neutral | 1.1093 | 0.9829 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1334 |
| Cosine Distance | 0.0003 |
| Clusters | 284 |
| Noise Fraction | 0.0824 |

---

## Checkpoint 14,848 episodes — step 4,011,408

**Episodes:** 14,848  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9762 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0794 |
| Gradient Magnitude (Failure) | 1.1049 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3938 | 0.8043 |
| Neutral | 1.0885 | 0.9750 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1572 |
| Cosine Distance | 0.0004 |
| Clusters | 281 |
| Noise Fraction | 0.1072 |

---

## Checkpoint 14,944 episodes — step 4,036,448

**Episodes:** 14,944  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9348 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0958 |
| Gradient Magnitude (Failure) | 1.0938 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3164 | 0.7514 |
| Neutral | 1.1009 | 0.8810 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1652 |
| Cosine Distance | 0.0005 |
| Clusters | 274 |
| Noise Fraction | 0.0779 |

---

## Checkpoint 15,040 episodes — step 4,061,488

**Episodes:** 15,040  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9850 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1067 |
| Gradient Magnitude (Failure) | 1.1265 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5170 | 0.5683 |
| Neutral | 1.0779 | 0.9339 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1564 |
| Cosine Distance | 0.0004 |
| Clusters | 288 |
| Noise Fraction | 0.1092 |

---

## Checkpoint 15,121 episodes — step 4,086,528

**Episodes:** 15,121  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9416 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1188 |
| Gradient Magnitude (Failure) | 1.0966 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4358 | 0.8204 |
| Neutral | 1.0912 | 0.9575 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0755 |
| Cosine Distance | 0.0001 |
| Clusters | 280 |
| Noise Fraction | 0.0797 |

---

## Checkpoint 15,216 episodes — step 4,111,568

**Episodes:** 15,216  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8940 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1497 |
| Gradient Magnitude (Failure) | 1.1017 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.7515 | 0.7683 |
| Neutral | 1.1113 | 0.9593 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0539 |
| Cosine Distance | 0.0001 |
| Clusters | 281 |
| Noise Fraction | 0.0774 |

---

## Checkpoint 15,312 episodes — step 4,136,608

**Episodes:** 15,312  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9871 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0942 |
| Gradient Magnitude (Failure) | 1.0903 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8356 | 0.6357 |
| Neutral | 1.0930 | 0.9880 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1359 |
| Cosine Distance | 0.0003 |
| Clusters | 289 |
| Noise Fraction | 0.0836 |

---

## Checkpoint 15,408 episodes — step 4,161,648

**Episodes:** 15,408  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9676 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0890 |
| Gradient Magnitude (Failure) | 1.0770 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4466 | 0.8507 |
| Neutral | 1.0727 | 0.9892 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1316 |
| Cosine Distance | 0.0003 |
| Clusters | 273 |
| Noise Fraction | 0.1071 |

---

## Checkpoint 15,504 episodes — step 4,186,688

**Episodes:** 15,504  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9902 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0844 |
| Gradient Magnitude (Failure) | 1.1075 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3566 | 0.7461 |
| Neutral | 1.0846 | 0.9964 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1313 |
| Cosine Distance | 0.0003 |
| Clusters | 288 |
| Noise Fraction | 0.0765 |

---

## Checkpoint 15,585 episodes — step 4,211,728

**Episodes:** 15,585  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8836 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1155 |
| Gradient Magnitude (Failure) | 1.0787 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9971 | 0.2752 |
| Neutral | 1.0598 | 0.9575 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1764 |
| Cosine Distance | 0.0005 |
| Clusters | 276 |
| Noise Fraction | 0.1078 |

---

## Checkpoint 15,680 episodes — step 4,236,768

**Episodes:** 15,680  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9708 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0869 |
| Gradient Magnitude (Failure) | 1.0675 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4159 | 0.6509 |
| Neutral | 1.0892 | 0.9303 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1245 |
| Cosine Distance | 0.0002 |
| Clusters | 269 |
| Noise Fraction | 0.0897 |

---

## Checkpoint 15,776 episodes — step 4,261,808

**Episodes:** 15,776  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9791 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0754 |
| Gradient Magnitude (Failure) | 1.0773 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2661 | 0.8618 |
| Neutral | 1.1075 | 0.9682 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1269 |
| Cosine Distance | 0.0002 |
| Clusters | 278 |
| Noise Fraction | 0.1011 |

---

## Checkpoint 15,872 episodes — step 4,286,848

**Episodes:** 15,872  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.300) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9731 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0684 |
| Gradient Magnitude (Failure) | 1.0728 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3381 | 0.8567 |
| Neutral | 1.1334 | 0.9018 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0964 |
| Cosine Distance | 0.0001 |
| Clusters | 293 |
| Noise Fraction | 0.1342 |

---

## Checkpoint 15,968 episodes — step 4,311,888

**Episodes:** 15,968  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9566 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1167 |
| Gradient Magnitude (Failure) | 1.0620 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5339 | 0.8286 |
| Neutral | 1.0815 | 0.9676 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0853 |
| Cosine Distance | 0.0001 |
| Clusters | 274 |
| Noise Fraction | 0.0760 |

---

## Checkpoint 16,049 episodes — step 4,336,928

**Episodes:** 16,049  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9544 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1286 |
| Gradient Magnitude (Failure) | 1.0626 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5803 | 0.5041 |
| Neutral | 1.0965 | 0.9817 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0791 |
| Cosine Distance | 0.0002 |
| Clusters | 270 |
| Noise Fraction | 0.0671 |

---

## Checkpoint 16,144 episodes — step 4,361,968

**Episodes:** 16,144  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9721 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1070 |
| Gradient Magnitude (Failure) | 1.0729 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3812 | 0.7400 |
| Neutral | 1.1131 | 0.9649 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0134 |
| Cosine Distance | 0.0000 |
| Clusters | 288 |
| Noise Fraction | 0.0831 |

---

## Checkpoint 16,240 episodes — step 4,387,008

**Episodes:** 16,240  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9750 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0943 |
| Gradient Magnitude (Failure) | 1.0887 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6635 | 0.7511 |
| Neutral | 1.1614 | 0.8242 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0791 |
| Cosine Distance | 0.0001 |
| Clusters | 280 |
| Noise Fraction | 0.0666 |

---

## Checkpoint 16,336 episodes — step 4,412,048

**Episodes:** 16,336  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9519 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0851 |
| Gradient Magnitude (Failure) | 1.1450 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4502 | 0.7769 |
| Neutral | 1.0866 | 0.9871 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1099 |
| Cosine Distance | 0.0002 |
| Clusters | 298 |
| Noise Fraction | 0.0742 |

---

## Checkpoint 16,432 episodes — step 4,437,088

**Episodes:** 16,432  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9435 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1450 |
| Gradient Magnitude (Failure) | 1.0787 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2522 | 0.8490 |
| Neutral | 1.1764 | 0.7579 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1988 |
| Cosine Distance | 0.0007 |
| Clusters | 275 |
| Noise Fraction | 0.0685 |

---

## Checkpoint 16,513 episodes — step 4,462,128

**Episodes:** 16,513  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9333 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0995 |
| Gradient Magnitude (Failure) | 1.1244 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6382 | 0.7228 |
| Neutral | 1.0981 | 0.9866 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1959 |
| Cosine Distance | 0.0007 |
| Clusters | 284 |
| Noise Fraction | 0.0674 |

---

## Checkpoint 16,608 episodes — step 4,487,168

**Episodes:** 16,608  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9042 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0926 |
| Gradient Magnitude (Failure) | 1.1591 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9980 | 0.5488 |
| Neutral | 1.1047 | 0.9852 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2097 |
| Cosine Distance | 0.0007 |
| Clusters | 277 |
| Noise Fraction | 0.0713 |

---

## Checkpoint 16,704 episodes — step 4,512,208

**Episodes:** 16,704  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9284 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1094 |
| Gradient Magnitude (Failure) | 1.0861 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3399 | 0.8689 |
| Neutral | 1.1176 | 0.9909 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1375 |
| Cosine Distance | 0.0003 |
| Clusters | 304 |
| Noise Fraction | 0.0721 |

---

## Checkpoint 16,800 episodes — step 4,537,248

**Episodes:** 16,800  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9811 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1111 |
| Gradient Magnitude (Failure) | 1.1374 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6540 | 0.7688 |
| Neutral | 1.0970 | 0.9915 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1480 |
| Cosine Distance | 0.0004 |
| Clusters | 281 |
| Noise Fraction | 0.0805 |

---

## Checkpoint 16,896 episodes — step 4,562,288

**Episodes:** 16,896  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9862 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0774 |
| Gradient Magnitude (Failure) | 1.0746 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 2.0829 | 0.6162 |
| Neutral | 1.0994 | 0.9875 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1072 |
| Cosine Distance | 0.0002 |
| Clusters | 269 |
| Noise Fraction | 0.0818 |

---

## Checkpoint 16,977 episodes — step 4,587,328

**Episodes:** 16,977  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8862 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1448 |
| Gradient Magnitude (Failure) | 1.0848 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3930 | 0.8584 |
| Neutral | 1.0806 | 0.8652 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0922 |
| Cosine Distance | 0.0001 |
| Clusters | 280 |
| Noise Fraction | 0.0716 |

---

## Checkpoint 17,072 episodes — step 4,612,368

**Episodes:** 17,072  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9657 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1135 |
| Gradient Magnitude (Failure) | 1.1006 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6675 | 0.8101 |
| Neutral | 1.0791 | 0.9861 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1270 |
| Cosine Distance | 0.0003 |
| Clusters | 275 |
| Noise Fraction | 0.0793 |

---

## Checkpoint 17,168 episodes — step 4,637,408

**Episodes:** 17,168  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9839 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1002 |
| Gradient Magnitude (Failure) | 1.0979 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3572 | 0.7777 |
| Neutral | 1.0847 | 0.9727 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1370 |
| Cosine Distance | 0.0003 |
| Clusters | 273 |
| Noise Fraction | 0.0707 |

---

## Checkpoint 17,264 episodes — step 4,662,448

**Episodes:** 17,264  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9768 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0651 |
| Gradient Magnitude (Failure) | 1.0956 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5839 | 0.7417 |
| Neutral | 1.0566 | 0.9913 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1615 |
| Cosine Distance | 0.0004 |
| Clusters | 282 |
| Noise Fraction | 0.0769 |

---

## Checkpoint 17,360 episodes — step 4,687,488

**Episodes:** 17,360  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9614 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0716 |
| Gradient Magnitude (Failure) | 1.0908 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3661 | 0.7107 |
| Neutral | 1.0770 | 0.9986 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1389 |
| Cosine Distance | 0.0003 |
| Clusters | 262 |
| Noise Fraction | 0.0676 |

---

## Checkpoint 17,441 episodes — step 4,712,528

**Episodes:** 17,441  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9736 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0671 |
| Gradient Magnitude (Failure) | 1.0888 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.9023 | 0.4828 |
| Neutral | 1.1994 | 0.9186 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1075 |
| Cosine Distance | 0.0002 |
| Clusters | 286 |
| Noise Fraction | 0.0953 |

---

## Checkpoint 17,536 episodes — step 4,737,568

**Episodes:** 17,536  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9826 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0725 |
| Gradient Magnitude (Failure) | 1.0912 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4412 | 0.7835 |
| Neutral | 1.0887 | 0.9849 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0839 |
| Cosine Distance | 0.0001 |
| Clusters | 273 |
| Noise Fraction | 0.0780 |

---

## Checkpoint 17,632 episodes — step 4,762,608

**Episodes:** 17,632  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9757 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0703 |
| Gradient Magnitude (Failure) | 1.0967 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.8062 | 0.6351 |
| Neutral | 1.0908 | 0.9857 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1529 |
| Cosine Distance | 0.0004 |
| Clusters | 280 |
| Noise Fraction | 0.0604 |

---

## Checkpoint 17,728 episodes — step 4,787,648

**Episodes:** 17,728  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9795 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0739 |
| Gradient Magnitude (Failure) | 1.0920 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.4400 | 0.8177 |
| Neutral | 1.0839 | 0.9960 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1471 |
| Cosine Distance | 0.0004 |
| Clusters | 275 |
| Noise Fraction | 0.0477 |

---

## Checkpoint 17,824 episodes — step 4,812,688

**Episodes:** 17,824  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9691 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1047 |
| Gradient Magnitude (Failure) | 1.0714 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3100 | 0.7525 |
| Neutral | 1.1257 | 0.8705 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1698 |
| Cosine Distance | 0.0005 |
| Clusters | 302 |
| Noise Fraction | 0.0626 |

---

## Checkpoint 17,905 episodes — step 4,837,728

**Episodes:** 17,905  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8754 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1215 |
| Gradient Magnitude (Failure) | 1.1021 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2813 | 0.9258 |
| Neutral | 1.1043 | 0.8853 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1628 |
| Cosine Distance | 0.0005 |
| Clusters | 269 |
| Noise Fraction | 0.0649 |

---

## Checkpoint 18,000 episodes — step 4,862,768

**Episodes:** 18,000  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9850 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0919 |
| Gradient Magnitude (Failure) | 1.1032 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.2833 | 0.8658 |
| Neutral | 1.0967 | 0.9959 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1870 |
| Cosine Distance | 0.0007 |
| Clusters | 282 |
| Noise Fraction | 0.0732 |

---

## Checkpoint 18,096 episodes — step 4,887,808

**Episodes:** 18,096  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9803 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1004 |
| Gradient Magnitude (Failure) | 1.0906 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3570 | 0.8580 |
| Neutral | 1.0967 | 0.9850 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2318 |
| Cosine Distance | 0.0010 |
| Clusters | 283 |
| Noise Fraction | 0.0550 |

---

## Checkpoint 18,192 episodes — step 4,912,848

**Episodes:** 18,192  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9774 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0937 |
| Gradient Magnitude (Failure) | 1.1639 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.6081 | 0.5171 |
| Neutral | 1.0767 | 0.9908 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2261 |
| Cosine Distance | 0.0010 |
| Clusters | 300 |
| Noise Fraction | 0.0665 |

---

## Checkpoint 18,288 episodes — step 4,937,888

**Episodes:** 18,288  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9692 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0775 |
| Gradient Magnitude (Failure) | 1.1122 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5535 | 0.7009 |
| Neutral | 1.0904 | 0.9827 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.2419 |
| Cosine Distance | 0.0010 |
| Clusters | 277 |
| Noise Fraction | 0.0451 |

---

## Checkpoint 18,369 episodes — step 4,962,928

**Episodes:** 18,369  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8789 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.0808 |
| Gradient Magnitude (Failure) | 1.2426 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5376 | 0.7370 |
| Neutral | 1.0890 | 0.9872 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1809 |
| Cosine Distance | 0.0006 |
| Clusters | 272 |
| Noise Fraction | 0.0798 |

---

## Checkpoint 18,464 episodes — step 4,987,968

**Episodes:** 18,464  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9640 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1046 |
| Gradient Magnitude (Failure) | 1.1031 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.5972 | 0.7556 |
| Neutral | 1.0994 | 0.9822 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1661 |
| Cosine Distance | 0.0005 |
| Clusters | 280 |
| Noise Fraction | 0.0733 |

---

## Checkpoint 18,512 episodes — step 5,000,000

**Episodes:** 18,512  
**Success:** —  
**Failure:** —  
**Threshold:** bottom 25% (≤ 0.200) | top 25% (≥ 0.300)

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.9909 |
| Coherence (Success) | — |
| Coherence (Failure) | — |
| Gradient Magnitude (Success) | 1.1212 |
| Gradient Magnitude (Failure) | 1.1230 |

### Reward Moments

| Moment | Grad Magnitude | Cosine vs. Success |
|---|---:|---:|
| Positive | 1.3822 | 0.7361 |
| Neutral | 1.1271 | 0.9418 |
| Negative | — | — |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.1114 |
| Cosine Distance | 0.0002 |
| Clusters | 298 |
| Noise Fraction | 0.1035 |
