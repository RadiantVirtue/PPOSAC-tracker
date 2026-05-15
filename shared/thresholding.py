import numpy as np


# split episodes into Success / Failure using one of three modes:
#
#   mode="eps"        — existing behaviour: split at mean EPS score.
#                       Returns (success, failure, mu) where mu is a float.
#
#   mode="percentile" — rank episodes by raw return; bottom percentile_x%
#                       become failure, top percentile_x% become success,
#                       middle episodes are discarded.
#                       Returns (success, failure, (lower, upper)) where the
#                       tuple contains the two return cutoffs.
#
#   mode="fixed"      — apply pre-determined cutoffs (lower, upper) from
#                       fixed_thresholds=(lower_return, upper_return).
#                       Episodes with return <= lower → failure;
#                       episodes with return >= upper → success; rest discarded.
#                       Returns (success, failure, (lower, upper)).
#                       Use this to eliminate partition-boundary confounds in
#                       the RQ1 longitudinal analysis (Experiment 2).
def partition_episodes(episodes, scores, mode="eps", percentile_x=25,
                       fixed_thresholds=None):
    if mode == "fixed":
        if fixed_thresholds is None:
            raise ValueError("mode='fixed' requires fixed_thresholds=(lower, upper)")
        lower, upper = fixed_thresholds
        scores_arr = np.array(scores)
        failure = [ep for ep, s in zip(episodes, scores) if s <= lower]
        success = [ep for ep, s in zip(episodes, scores) if s >= upper]
        return success, failure, (float(lower), float(upper))
    elif mode == "percentile":
        scores_arr = np.array(scores)
        sorted_idx = np.argsort(scores_arr, kind="stable")
        n = len(episodes)
        n_each = max(1, int(n * percentile_x / 100))
        failure_idx = set(sorted_idx[:n_each])
        success_idx = set(sorted_idx[n - n_each:])
        failure = [ep for i, ep in enumerate(episodes) if i in failure_idx]
        success = [ep for i, ep in enumerate(episodes) if i in success_idx]
        lower = float(scores_arr[sorted_idx[n_each - 1]])
        upper = float(scores_arr[sorted_idx[n - n_each]])
        return success, failure, (lower, upper)
    else:
        mu = float(np.mean(scores))
        success = [ep for ep, s in zip(episodes, scores) if s >= mu]
        failure = [ep for ep, s in zip(episodes, scores) if s < mu]
        return success, failure, mu
