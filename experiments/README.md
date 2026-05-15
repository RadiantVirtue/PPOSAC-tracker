# experiments/

Experiment orchestration scripts. All scripts add the PPOSAC-tracker root to
`sys.path` automatically, so they can be run from any working directory:

```bash
cd PPOSAC-tracker/
python experiments/<script>.py [args]
```

## Robustness experiments (Issues #1–#4)

| Script | Issue | Purpose |
|--------|-------|---------|
| `run_scalar_ablation.py` | #1 | Train non-distributional ScalarDQN and compute MORA ratio. Gates the headline credit-assignment claim. Run after corrected analysis (Phase B). |
| `run_fixed_threshold_analysis.py` | #2 | Re-run RQ1 longitudinal analysis with thresholds fixed across checkpoints. Use `--pool_all_seeds` to derive thresholds from all 5 seeds' final checkpoints (eliminates seed-anchor degree of freedom). |
| `run_gradient_validation.py` | #4 | Train a fresh Rainbow run with gradient capture enabled. Compute `cos(G_training, G_counterfactual)` to validate the counterfactual gradient framework. |

## Null-model and appendix (Issues #5, #7)

| Script | Issue | Purpose |
|--------|-------|---------|
| `compare_ppo_seeds.py` | #5 | Cross-seed cosine similarity `cos(G_success^i, G_success^j)` for all 10 PPO seed pairs. N=10 pairs, 95% bootstrap CI. |
| `ablation_experiments.py` | #7 | Hyperparameter ablation runs (training and analysis params). |
| `ablation_graphing.py` | #7 | Figures for hyperparameter ablation results. |

## Utilities

| Script | Purpose |
|--------|---------|
| `compare_old_vs_corrected.py` | Side-by-side comparison of pre-fix vs post-fix analysis JSONs. |
| `smoke_test_next_obs_fix.py` | Pipeline correctness test for the next-obs fix. |

## Key shared scripts (remain at root)

- `analyze_checkpoint.py` — core analysis orchestrator (imported by many scripts)
- `run_corrected_analysis.py` — main corrected analysis runner (Phase B)
- `report_robustness_stats.py` — Issue #10 CI reporting
- `train_and_analyze.py` — training + analysis loop
- `analyze_range.py` — batch analysis runner
- `regenerate_graphs.py` — graph regeneration utility
