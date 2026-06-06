"""Episode partitioning into success / failure groups."""
import numpy as np


def partition_episodes(episodes, scores, mode="eps", percentile_x=25,
                       fixed_thresholds=None):
    """Split episodes into success / failure using one of three modes.

    Args:
        episodes:          list of EpisodeData
        scores:            list or array of EPS scores (one per episode)
        mode:              "eps"        — split at mean EPS score
                           "percentile" — bottom/top percentile_x% by score
                           "fixed"      — apply pre-determined cutoffs
        percentile_x:      percentage for percentile mode (default 25)
        fixed_thresholds:  (lower, upper) tuple for fixed mode

    Returns:
        (success_episodes, failure_episodes, threshold)
        threshold is a float for "eps" mode, (lower, upper) tuple otherwise.
    """
    if mode == "fixed":
        if fixed_thresholds is None:
            raise ValueError("mode='fixed' requires fixed_thresholds=(lower, upper)")
        lower, upper = fixed_thresholds
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

    else:  # "eps" — split at mean
        mu = float(np.mean(scores))
        success = [ep for ep, s in zip(episodes, scores) if s >= mu]
        failure = [ep for ep, s in zip(episodes, scores) if s < mu]
        return success, failure, mu
