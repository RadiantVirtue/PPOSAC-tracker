"""training - HARD-CODED orchestration pipeline.

All scripts here are hard-coded. Entity-specific behaviour is injected via
the Entity protocol and RunConfig. No algorithm-specific branches exist here.

Files:
    run_config.py          - RunConfig dataclass; single source of truth for all
                             tunable parameters across training and analysis
    trainer.py             - top-level orchestrator; calls entity.train() with a
                             checkpoint callback; triggers eval + analysis
    eval_runner.py         - generic episode collection loop; creates EvaluationBatch
                             and saves to a compressed temp file
    achievement_tracker.py - per-episode RSA frame logging; receives achievement
                             definitions (names, label_map) as constructor inputs;
                             NOT used for EPS scoring
"""
