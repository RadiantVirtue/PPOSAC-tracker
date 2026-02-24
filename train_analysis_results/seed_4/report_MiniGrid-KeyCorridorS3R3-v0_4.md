# Training & Analysis Report

**Environment:** `MiniGrid-KeyCorridorS3R3-v0`  
**Seed:** 4  
**Total episodes:** 30,017  
**Experiment root:** `train_analysis_results\seed_4`  
**Generated:** 2026-02-24 06:52

## Summary

| Checkpoint | Episodes | Opp. Score | Coh. (S) | Coh. (F) | Grad Mag (S) | Grad Mag (F) | Act. Sep. | RSA Align. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 — 3k (3005 episodes) | 3,005 | 0.8248 | 0.2335 | 0.2880 | 0.0907 | 0.0633 | 0.0575 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 2 — 6k (6007 episodes) | 6,007 | 0.2865 | 0.2782 | 0.1285 | 0.0866 | 0.0382 | 0.0542 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 3 — 9k (9001 episodes) | 9,001 | 0.8187 | 0.2555 | 0.3741 | 0.0673 | 0.0810 | 0.0636 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 4 — 12k (12007 episodes) | 12,007 | 0.7284 | 0.3261 | 0.3484 | 0.1068 | 0.0678 | 0.0433 | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |
| Checkpoint 5 — 15k (15002 episodes) | 15,002 | -0.1805 | 0.4200 | 0.0640 | 0.2076 | 0.0537 | 0.4740 | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |
| Checkpoint 6 — 18k (18021 episodes) | 18,021 | -0.0135 | 0.3262 | -0.0338 | 0.3131 | 0.1058 | 0.9180 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 7 — 21k (21010 episodes) | 21,010 | -0.1248 | 0.2293 | 0.0043 | 0.1734 | 0.2720 | 1.7258 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 8 — 24k (24014 episodes) | 24,014 | -0.0333 | 0.3355 | -0.0068 | 0.2184 | 0.2924 | 1.7113 | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |
| Checkpoint 9 — 27k (27022 episodes) | 27,022 | -0.2361 | 0.3335 | 0.2881 | 0.2405 | 0.3603 | 2.4042 | {'correlation': 0.0, 'p_value': 1.0} |
| Checkpoint 10 — 30k (30017 episodes) | 30,017 | 0.3412 | 0.4502 | 0.0557 | 0.3101 | 0.3960 | 2.5013 | {'correlation': 0.0, 'p_value': 1.0} |
| Final — 30k episodes | 30,017 | — | 0.3700 | -0.3867 | 0.2225 | 0.6206 | 2.5174 | {'correlation': 0.0, 'p_value': 1.0} |

---

## Checkpoint 1 — 3k (3005 episodes)

**Episodes:** 3,005  
**Success:** 189  
**Failure:** 311  
**Threshold μ:** 3.740

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8248 |
| Coherence (Success) | 0.2335 |
| Coherence (Failure) | 0.2880 |
| Gradient Magnitude (Success) | 0.0907 |
| Gradient Magnitude (Failure) | 0.0633 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0575 |
| Clusters | 4,206 |
| Noise Fraction | 0.0285 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.219 | 0.065 | 0.078 |
| locked_door | 0.219 | -0.000 | 0.199 | 0.420 |
| open_door | 0.065 | 0.199 | 0.000 | 0.166 |
| target_ball | 0.078 | 0.420 | 0.166 | 0.000 |

---

## Checkpoint 2 — 6k (6007 episodes)

**Episodes:** 6,007  
**Success:** 178  
**Failure:** 322  
**Threshold μ:** 3.704

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.2865 |
| Coherence (Success) | 0.2782 |
| Coherence (Failure) | 0.1285 |
| Gradient Magnitude (Success) | 0.0866 |
| Gradient Magnitude (Failure) | 0.0382 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0542 |
| Clusters | 4,090 |
| Noise Fraction | 0.0252 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.372 | 0.107 | 0.495 |
| locked_door | 0.372 | 0.000 | 0.432 | 0.565 |
| open_door | 0.107 | 0.432 | -0.000 | 0.895 |
| target_ball | 0.495 | 0.565 | 0.895 | 0.000 |

---

## Checkpoint 3 — 9k (9001 episodes)

**Episodes:** 9,001  
**Success:** 270  
**Failure:** 230  
**Threshold μ:** 4.190

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.8187 |
| Coherence (Success) | 0.2555 |
| Coherence (Failure) | 0.3741 |
| Gradient Magnitude (Success) | 0.0673 |
| Gradient Magnitude (Failure) | 0.0810 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0636 |
| Clusters | 3,977 |
| Noise Fraction | 0.0294 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.147 | 0.070 | 0.158 |
| locked_door | 0.147 | 0.000 | 0.108 | 0.279 |
| open_door | 0.070 | 0.108 | 0.000 | 0.299 |
| target_ball | 0.158 | 0.279 | 0.299 | 0.000 |

---

## Checkpoint 4 — 12k (12007 episodes)

**Episodes:** 12,007  
**Success:** 177  
**Failure:** 323  
**Threshold μ:** 3.694

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.7284 |
| Coherence (Success) | 0.3261 |
| Coherence (Failure) | 0.3484 |
| Gradient Magnitude (Success) | 0.1068 |
| Gradient Magnitude (Failure) | 0.0678 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.0433 |
| Clusters | 3,692 |
| Noise Fraction | 0.0214 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': -0.4140393356054126, 'p_value': 0.41443008250091623} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.050 | 0.051 | 0.179 |
| locked_door | 0.050 | 0.000 | 0.154 | 0.087 |
| open_door | 0.051 | 0.154 | -0.000 | 0.306 |
| target_ball | 0.179 | 0.087 | 0.306 | -0.000 |

---

## Checkpoint 5 — 15k (15002 episodes)

**Episodes:** 15,002  
**Success:** 297  
**Failure:** 203  
**Threshold μ:** 4.578

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1805 |
| Coherence (Success) | 0.4200 |
| Coherence (Failure) | 0.0640 |
| Gradient Magnitude (Success) | 0.2076 |
| Gradient Magnitude (Failure) | 0.0537 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.4740 |
| Clusters | 2,947 |
| Noise Fraction | 0.0225 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.8280786712108252, 'p_value': 0.04179468045604518} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.892 | 0.737 | 0.227 |
| locked_door | 0.892 | 0.000 | 0.115 | 1.001 |
| open_door | 0.737 | 0.115 | 0.000 | 0.852 |
| target_ball | 0.227 | 1.001 | 0.852 | 0.000 |

---

## Checkpoint 6 — 18k (18021 episodes)

**Episodes:** 18,021  
**Success:** 409  
**Failure:** 91  
**Threshold μ:** 5.574

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0135 |
| Coherence (Success) | 0.3262 |
| Coherence (Failure) | -0.0338 |
| Gradient Magnitude (Success) | 0.3131 |
| Gradient Magnitude (Failure) | 0.1058 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 0.9180 |
| Clusters | 1,537 |
| Noise Fraction | 0.0528 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.351 | 0.175 | 0.449 |
| locked_door | 0.351 | -0.000 | 0.183 | 0.957 |
| open_door | 0.175 | 0.183 | 0.000 | 0.532 |
| target_ball | 0.449 | 0.957 | 0.532 | 0.000 |

---

## Checkpoint 7 — 21k (21010 episodes)

**Episodes:** 21,010  
**Success:** 460  
**Failure:** 40  
**Threshold μ:** 5.796

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.1248 |
| Coherence (Success) | 0.2293 |
| Coherence (Failure) | 0.0043 |
| Gradient Magnitude (Success) | 0.1734 |
| Gradient Magnitude (Failure) | 0.2720 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7258 |
| Clusters | 995 |
| Noise Fraction | 0.0586 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.237 | 0.049 | 0.323 |
| locked_door | 0.237 | 0.000 | 0.223 | 0.778 |
| open_door | 0.049 | 0.223 | 0.000 | 0.429 |
| target_ball | 0.323 | 0.778 | 0.429 | 0.000 |

---

## Checkpoint 8 — 24k (24014 episodes)

**Episodes:** 24,014  
**Success:** 469  
**Failure:** 31  
**Threshold μ:** 5.888

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.0333 |
| Coherence (Success) | 0.3355 |
| Coherence (Failure) | -0.0068 |
| Gradient Magnitude (Success) | 0.2184 |
| Gradient Magnitude (Failure) | 0.2924 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 1.7113 |
| Clusters | 1,022 |
| Noise Fraction | 0.0637 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.2070196678027063, 'p_value': 0.69390663403457} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.104 | 0.055 | 0.303 |
| locked_door | 0.104 | 0.000 | 0.061 | 0.562 |
| open_door | 0.055 | 0.061 | 0.000 | 0.425 |
| target_ball | 0.303 | 0.562 | 0.425 | 0.000 |

---

## Checkpoint 9 — 27k (27022 episodes)

**Episodes:** 27,022  
**Success:** 485  
**Failure:** 15  
**Threshold μ:** 5.948

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | -0.2361 |
| Coherence (Success) | 0.3335 |
| Coherence (Failure) | 0.2881 |
| Gradient Magnitude (Success) | 0.2405 |
| Gradient Magnitude (Failure) | 0.3603 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.4042 |
| Clusters | 894 |
| Noise Fraction | 0.0689 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.066 | 0.017 | 0.279 |
| locked_door | 0.066 | 0.000 | 0.069 | 0.495 |
| open_door | 0.017 | 0.069 | 0.000 | 0.288 |
| target_ball | 0.279 | 0.495 | 0.288 | 0.000 |

---

## Checkpoint 10 — 30k (30017 episodes)

**Episodes:** 30,017  
**Success:** 488  
**Failure:** 12  
**Threshold μ:** 5.950

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | 0.3412 |
| Coherence (Success) | 0.4502 |
| Coherence (Failure) | 0.0557 |
| Gradient Magnitude (Success) | 0.3101 |
| Gradient Magnitude (Failure) | 0.3960 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5013 |
| Clusters | 801 |
| Noise Fraction | 0.0733 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | 0.000 | 0.081 | 0.026 | 0.203 |
| locked_door | 0.081 | 0.000 | 0.089 | 0.433 |
| open_door | 0.026 | 0.089 | 0.000 | 0.203 |
| target_ball | 0.203 | 0.433 | 0.203 | 0.000 |

---

## Final — 30k episodes

**Episodes:** 30,017  
**Success:** 493  
**Failure:** 7  
**Threshold μ:** 5.974

### Gradient Metrics

| Metric | Value |
|---|---|
| Opposition Score | — |
| Coherence (Success) | 0.3700 |
| Coherence (Failure) | -0.3867 |
| Gradient Magnitude (Success) | 0.2225 |
| Gradient Magnitude (Failure) | 0.6206 |

### Activation Metrics

| Metric | Value |
|---|---|
| Activation Separation | 2.5174 |
| Clusters | 771 |
| Noise Fraction | 0.0654 |

### RSA

| Metric | Value |
|---|---|
| Alignment (Spearman ρ) | {'correlation': 0.0, 'p_value': 1.0} |

**Representational Dissimilarity Matrix:**

| | key | locked_door | open_door | target_ball |
|---|---|---|---|---|
| key | -0.000 | 0.088 | 0.028 | 0.198 |
| locked_door | 0.088 | -0.000 | 0.094 | 0.444 |
| open_door | 0.028 | 0.094 | 0.000 | 0.199 |
| target_ball | 0.198 | 0.444 | 0.199 | 0.000 |
