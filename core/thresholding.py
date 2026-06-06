"""Episode partitioning into success / failure groups."""
import numpy as np


def partition_episodes(episodes, scores, percentile_x: int = 25):
    """Split episodes into top/bottom percentile_x% by EPS score.

    Args:
        episodes:     list of EpisodeData
        scores:       list or array of EPS scores (one per episode)
        percentile_x: percentage for each tail (default 25 → top/bottom 25%)

    Returns:
        (success_episodes, failure_episodes, (lower_threshold, upper_threshold))
    """
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
