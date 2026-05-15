# PPOSAC-tracker

Dissertation codebase: PPO and Rainbow DQN trained on Crafter, with gradient analysis, activation analysis, and RSA.

## Install

```bash
pip install crafter gymnasium torch stable-baselines3 tyro torch_ac umap-learn hdbscan scipy tensorboard tqdm
```

---

## Training

### PPO - train + analyse (main entry point)

Trains across 5 seeds, runs analysis at every checkpoint, writes markdown reports.

```bash
python ppo/train_and_analyze.py
python ppo/train_and_analyze.py --seeds 1 2 3 --total_timesteps 5_000_000
```

### Rainbow - train + analyse (main entry point)

```bash
python rainbow/train_and_analyze.py
python rainbow/train_and_analyze.py --seeds 1 2 3
```

### Train only (no analysis)

```bash
python ppo/train.py --total_timesteps 10_000_000 --experiment_root my_ppo_run
python rainbow/train.py --T-max 10000000 --experiment-root my_rainbow_run
```

---

## Analysis

### Re-run analysis on existing checkpoints

```bash
# Single checkpoint
python analyze_checkpoint.py --algorithm rainbow --checkpoint_path path/to/checkpoint.pt ...

# Re-run corrected MoR analysis across all seeds
python run_corrected_analysis.py \
    --experiment_root rainbow_experiment_root \
    --output_root corrected_analysis_results
```

### Full dissertation pipeline (all analysis phases)

Runs phases B–I (fixed-threshold, scalar ablation, frozen RSA, EPS sensitivity, etc.) in dependency order.

```bash
python run_pipeline.py                        # all phases
python run_pipeline.py --device cuda
python run_pipeline.py --skip D G H I        # critical path only
python run_pipeline.py --only B F            # specific phases
python run_pipeline.py --include_e           # include Phase E (slow - reruns training)
```

Phases: B=corrected analysis, C=scalar DQN ablation, D=fixed-threshold, E=counterfactual gradient, F=robustness report, G=PPO cross-seed, H=frozen RSA, I=EPS sensitivity.

### Full Rainbow pipeline (train all seeds + all phases)

```bash
python run_rainbow_full_pipeline.py
python run_rainbow_full_pipeline.py --device cuda --experiment_root rainbow_v2
```

---

## Dissertation figures

Generates all PDFs into `GRAPHS/`.

```bash
python dissertation_graphs/run_all.py \
    --ppo_root ppo_experiment_root \
    --rainbow_root rainbow_experiment_root \
    --rainbow_v2_root rainbow_v2
```

---

## Project structure

```
ppo/
  train.py               PPO training (SB3 CnnPolicy)
  train_and_analyze.py   Train + analyse entry point
  gradients.py           Gradient computation
  activations.py         Activation extraction
  sampling.py            Evaluation rollouts
rainbow/
  train.py               Rainbow DQN training
  train_and_analyze.py   Train + analyse entry point
  agent.py               Rainbow agent (adapted from Kaixhin/Rainbow)
  model.py               DQN model with NoisyLinear + dueling heads
  memory.py              Prioritised replay buffer
  gradients.py           G_uniform / G_IS / G_reward gradient variants
  activations.py         Activation extraction
  moment_of_reward.py    MORA sub-partition analysis
shared/
  achievements.py        Crafter achievement definitions + EPS scoring
  activation_utils.py    Hook-based activation extraction, UMAP, HDBSCAN
  gradient_utils.py      OnlineGradientAggregator, cosine similarity
  rsa.py                 Representational Similarity Analysis
  thresholding.py        Episode partitioning (EPS / percentile / fixed)
  reporting.py           Markdown report generation
  graphing.py            Longitudinal plot generation
  storage.py             Analysis result save/load
wrappers.py              Crafter gymnasium wrapper + achievement injection
analyze_checkpoint.py    Core analysis pipeline (called by all train_and_analyze scripts)
run_pipeline.py          All post-training analysis phases
run_rainbow_full_pipeline.py  Full Rainbow training + all phases
dissertation_graphs/     Figure generation scripts for the dissertation
experiments/             Ablation and validation experiments
```
