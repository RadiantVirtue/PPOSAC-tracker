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
        lower = float(np.percentile(scores, percentile_x))
        upper = float(np.percentile(scores, 100 - percentile_x))
        success = [ep for ep, s in zip(episodes, scores) if s >= upper]
        failure = [ep for ep, s in zip(episodes, scores) if s <= lower]
        return success, failure, (lower, upper)
    else:
        mu = float(np.mean(scores))
        success = [ep for ep, s in zip(episodes, scores) if s >= mu]
        failure = [ep for ep, s in zip(episodes, scores) if s < mu]
        return success, failure, mu
