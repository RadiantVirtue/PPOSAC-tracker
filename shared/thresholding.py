import numpy as np


# split episodes into Success / Failure using one of two modes:
#
#   mode="eps"        — existing behaviour: split at mean EPS score.
#                       Returns (success, failure, mu) where mu is a float.
#
#   mode="percentile" — rank episodes by raw return; bottom percentile_x%
#                       become failure, top percentile_x% become success,
#                       middle episodes are discarded.
#                       Returns (success, failure, (lower, upper)) where the
#                       tuple contains the two return cutoffs.
def partition_episodes(episodes, scores, mode="eps", percentile_x=25):
    if mode == "percentile":
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
