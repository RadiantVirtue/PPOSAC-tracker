"""core - HARD-CODED shared infrastructure.

Contains data structures, the Entity protocol, and algorithm-agnostic utility
functions (metrics, gradient aggregation, activation extraction, thresholding).

DO NOT add algorithm-specific or environment-specific logic here.
DO NOT import from entities/ here.
All modules here run identically regardless of which entity is used.

Files:
    data.py             - canonical dataclasses: EpisodeData, EvaluationBatch,
                          GradientResult, ActivationResult, AnalysisResult
    entity.py           - Entity protocol; the interface every entity must satisfy
    metrics.py          - scalar metric functions: opposition_score, coherence,
                          gradient_magnitude, activation_separation, rsa_alignment
    gradient_utils.py   - OnlineGradientAggregator, cosine_similarity_flat
    activation_utils.py - extract_activations, reduce_dimensions (UMAP),
                          cluster_activations (HDBSCAN), compute_centroids
    thresholding.py     - partition_episodes (eps / percentile / fixed modes)
"""
