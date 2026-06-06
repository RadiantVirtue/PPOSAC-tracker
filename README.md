# PPOSAC-tracker

Dissertation codebase: RL agents trained on Crafter, with gradient, activation, and RSA analysis to investigate credit assignment.

## Install

```bash
pip install crafter gymnasium torch stable-baselines3 umap-learn hdbscan scipy mlflow tqdm
```

---

## How to run

### Train + evaluate + analyse (new pipeline)

```bash
python -m training.trainer --entity ppo_crafter --n_steps 1_000_000 --eval_every 50_000
```

### View results

```bash
mlflow ui --backend-store-uri mlruns/
```

Then open `http://localhost:5000` - experiment `ppo_crafter` shows longitudinal metric plots and RDM artifacts.

### Adding a new entity (algorithm + environment)

See [ADDING_ENTITIES.md](ADDING_ENTITIES.md) for step-by-step instructions and skeleton code.

---

## New project structure

```
core/                   HARD-CODED - data structures, Entity protocol, shared utilities
  data.py               Canonical dataclasses: EpisodeData, EvaluationBatch, GradientResult,
                          ActivationResult, AnalysisResult
  entity.py             Entity protocol - the interface every entity must satisfy
  metrics.py            Scalar metrics: opposition_score, coherence, gradient_magnitude,
                          activation_separation, centroid_cosine_distance
  gradient_utils.py     OnlineGradientAggregator, cosine_similarity_flat
  activation_utils.py   extract_activations (hook-based), reduce_dimensions (UMAP),
                          cluster_activations (HDBSCAN), compute_centroids
  thresholding.py       partition_episodes (eps / percentile / fixed modes)

entities/               SOFT-CODED - one file per algorithm+environment pair
  definitions/
    crafter.py          22 Crafter achievements: names, labels, groups, materials
  ppo_crafter.py        PPO (Stable-Baselines3) + Crafter entity

training/               HARD-CODED - orchestration and episode collection
  run_config.py         RunConfig dataclass - single source of truth for all parameters
  trainer.py            Top-level orchestrator; triggers eval + analysis at each checkpoint
  eval_runner.py        Generic episode collection loop; saves EvaluationBatch to temp file
  achievement_tracker.py Per-episode RSA frame logging (separate from EPS scoring)

analysis/               HARD-CODED - analysis pipeline
  pipeline.py           Orchestrator: load temp -> partition -> analyse -> log -> delete
  gradient_analyzer.py  Delegates to entity.compute_gradients()
  activation_analyzer.py UMAP + HDBSCAN + centroids for success and failure groups
  rsa_analyzer.py       Cosine-dissimilarity RDM; Spearman rho per achievement group

storage/                HARD-CODED - persistence
  temp_store.py         EvaluationBatch <-> compressed .npz (~400-700 MB per batch);
                          deleted immediately after analysis
  mlflow_logger.py      AnalysisResult -> MLflow metrics and RDM artifacts

README.md               This file
ADDING_ENTITIES.md      How to add a new entity with skeleton code
```

### Legacy files (kept, not yet ported)

```
ppo/                    Original PPO training + analysis scripts
rainbow/                Original Rainbow DQN training + analysis scripts
shared/                 Original shared utilities (superseded by core/)
wrappers.py             Original Crafter wrappers (superseded by entities/ppo_crafter.py)
analyze_checkpoint.py   Original analysis entry point
run_pipeline.py         Multi-phase dissertation pipeline
dissertation_graphs/    Figure generation scripts
experiments/            Ablation experiments
```

---

## Achievement concepts

Three distinct concepts - kept strictly separate in the code:

| Concept | Purpose | Where |
|---------|---------|-------|
| **Achievement counting** | Compute EPS per episode. Used for success/failure partitioning. | `entity.compute_eps()` - SOFT |
| **Achievement frame logging** | Collect first obs frame per achievement unlock. Used as RSA stimuli. | `training/achievement_tracker.py` - HARD |
| **Training-level tracking** | First global step each achievement is ever seen. For graphing. | Tabled for later phase |
