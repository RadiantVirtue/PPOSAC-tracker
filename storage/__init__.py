"""storage - HARD-CODED persistence layer.

All scripts here are hard-coded. All data uses canonical types from core/data.py.

DO NOT add algorithm-specific serialisation here.

Files:
    temp_store.py    - EvaluationBatch <-> compressed .npz; deleted after analysis
                       Estimated size: ~400-700 MB per 500-episode batch
    mlflow_logger.py - AnalysisResult -> MLflow metrics (per step) and artifacts
"""
