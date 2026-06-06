"""analysis - HARD-CODED analysis pipeline.

All scripts here are hard-coded. No algorithm-specific branches.
Entity-specific behaviour is injected via the Entity protocol.

DO NOT add algorithm-specific if/elif branches here.

Files:
    pipeline.py            - orchestrator: load temp batch -> partition ->
                             analyse -> log to MLflow -> delete temp file
    gradient_analyzer.py   - thin delegation to entity.compute_gradients()
    activation_analyzer.py - activation extraction (entity.preprocess_obs +
                             entity.hook_layer), UMAP, HDBSCAN, centroids
    rsa_analyzer.py        - cosine-dissimilarity RDM; Spearman rho per group
                             in entity.achievement_groups
"""
