import numpy as np


# return mu = mean EPS across all episodes
def compute_threshold(eps_scores):
    return float(np.mean(eps_scores))


# split episodes into Success (EPS >= mu) and Failure (EPS < mu)
# returns: (success_list, failure_list, mu)
def partition_episodes(episodes, eps_scores):
    mu = compute_threshold(eps_scores)
    success = [ep for ep, eps in zip(episodes, eps_scores) if eps >= mu]
    failure = [ep for ep, eps in zip(episodes, eps_scores) if eps < mu]
    return success, failure, mu
