"""MORA-Informed Priority Tracker for Rainbow DQN.

Implements the gradient-coherence-based priority modifier described in
dissertation Section 8.2.2 (Eq 8.1):

    p_i = |δ_i|^α · m_{r=0}

where m_{r=0} = clamp(1 − cos(G_{r=0 in success}, G_{failure}), 0, 1) + ε

G_{r=0 in success} is the mean gradient of preparatory (r=0) transitions
within the last k successful episodes.  G_{failure} is the mean gradient of
all transitions in the last k failed episodes.  The modifier is highest (≈1+ε)
when preparatory transitions oppose the failure gradient, and lowest (≈ε) when
they are most aligned with it - directly targeting the preparatory-failure
alignment identified in the MORA analysis rather than the internal r>0/r≤0
class opposition used in earlier formulations.

Usage:
    tracker = MORAPriorityTracker(k=10)

    # Inside training loop, BEFORE the main loss.backward():
    #   1. Run a r=0-masked backward (retain_graph=True) to get grad_flat_r0.
    #   2. zero_grad, then run the main backward.
    #   3. clip_grad_norm_, then collect grad_flat (all-transition gradient).
    tracker.accumulate(grad_flat, grad_flat_r0=grad_flat_r0)

    # At episode end (done=True):
    tracker.episode_end(ep_return)
    tracker.log_m(log_path, episode_count)

    # In update_priorities(), apply tracker.m to r=0 transitions.
"""
import csv
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class MORAPriorityTracker:
    """Rolling gradient-coherence modifier for PER priority adjustment.

    Episode success/failure classification uses the same percentile partitioning
    as the dissertation's analysis pipeline (Section 6.5):
      - top    percentile_x % of recent returns  → success
      - bottom percentile_x % of recent returns  → failure
      - middle (100 - 2*percentile_x) %          → discarded (ambiguous)

    This mirrors mode="percentile" in the analysis pipeline and avoids the
    instability of a median split when returns are symmetrically distributed.

    Parameters
    ----------
    k : int
        Rolling window size - number of past successful (and failed) episodes
        whose mean gradients are averaged to form G_success / G_failure.
    epsilon : float
        Floor added to the modifier so priorities never collapse to zero.
        m ∈ [epsilon, 1 + epsilon].  Default: 0.01.
    percentile_x : int
        Percentile cutoff for success/failure classification (default 25).
        Top percentile_x% = success, bottom percentile_x% = failure.
    window : int
        Number of recent episode returns used to compute the rolling
        percentile thresholds.  25 is responsive over ~100 total episodes
        in a 1M-step run without being noise-dominated.
    """

    def __init__(
        self,
        k: int,
        epsilon: float = 0.01,
        percentile_x: int = 25,
        window: int = 25,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.epsilon = epsilon
        self.percentile_x = percentile_x
        self.window = window

        # Rolling buffers of per-episode mean gradient vectors (CPU tensors).
        # _preparatory_grads: r=0-only gradients from successful episodes (G_{r=0 in success}).
        # _failure_grads: all-transition gradients from failure episodes (G_failure).
        self._preparatory_grads: deque = deque(maxlen=k)
        self._failure_grads: deque = deque(maxlen=k)

        # Running sum for current episode - all transitions (used for failure classification).
        self._episode_accum: Optional[torch.Tensor] = None
        self._episode_n: int = 0  # number of learn() calls in current episode

        # Running sum for r=0 transitions only (used for success classification).
        self._episode_r0_accum: Optional[torch.Tensor] = None
        self._episode_r0_n: int = 0

        # Rolling window of recent episode returns for percentile thresholds
        self._recent_returns: deque = deque(maxlen=window)

        # Current modifier value.  Starts at epsilon (no-op) until both
        # success and failure deques have at least one entry.
        self.m: float = epsilon

        # Per-batch r≤0 fraction (set by update_priorities for logging)
        self._last_frac_neg: float = float("nan")

    # ------------------------------------------------------------------
    # Called every learn() step
    # ------------------------------------------------------------------

    def accumulate(
        self,
        grad_flat: torch.Tensor,
        grad_flat_r0: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one learn()-step gradient into the current episode sums.

        grad_flat     - 1-D CPU tensor of μ-parameters, all transitions (σ excluded).
        grad_flat_r0  - same shape, but computed from r=0 transitions only via a
                        separate masked backward; None if the batch had no r=0 transitions.
        """
        g = grad_flat.cpu()
        if self._episode_accum is None:
            self._episode_accum = g.clone()
        else:
            self._episode_accum.add_(g)
        self._episode_n += 1

        if grad_flat_r0 is not None:
            g0 = grad_flat_r0.cpu()
            if self._episode_r0_accum is None:
                self._episode_r0_accum = g0.clone()
            else:
                self._episode_r0_accum.add_(g0)
            self._episode_r0_n += 1

    # ------------------------------------------------------------------
    # Called at episode end (done=True in training loop)
    # ------------------------------------------------------------------

    def episode_end(self, ep_return: float) -> None:
        """Finalise the current episode gradient, classify success/failure,
        update the rolling deques, and recompute m.

        Classification mirrors the dissertation's percentile partitioning
        (mode="percentile", percentile_x=25 by default):
          - top    percentile_x % → success  → pushed to _success_grads
          - bottom percentile_x % → failure  → pushed to _failure_grads
          - middle 50 %           → discarded (ambiguous; does not update deques)

        If no learn() calls happened this episode (warm-up phase before
        learn_start), the episode is skipped without updating m.
        """
        if self._episode_n == 0 or self._episode_accum is None:
            self._episode_accum = None
            self._episode_n = 0
            self._episode_r0_accum = None
            self._episode_r0_n = 0
            return

        # Episode mean gradients (CPU, detached)
        mean_grad_all = (self._episode_accum / self._episode_n).detach()

        # Update rolling return window and compute percentile thresholds
        self._recent_returns.append(ep_return)
        recent = list(self._recent_returns)

        # Need at least 2 distinct values to compute meaningful percentiles
        if len(recent) >= 2:
            low_thresh = float(np.percentile(recent, self.percentile_x))
            high_thresh = float(np.percentile(recent, 100 - self.percentile_x))

            if ep_return >= high_thresh:
                # Success: push the r=0-only gradient (preparatory signal).
                # Skip if no r=0 transitions were seen this episode.
                if self._episode_r0_n > 0 and self._episode_r0_accum is not None:
                    mean_grad_r0 = (self._episode_r0_accum / self._episode_r0_n).detach()
                    self._preparatory_grads.append(mean_grad_r0)
            elif ep_return <= low_thresh:
                # Failure: push the all-transition gradient.
                self._failure_grads.append(mean_grad_all)
            # else: middle band - discard

        # Recompute m only when both sides have data.
        # c_buffer = cos(G_{r=0 in success}, G_failure): high when preparatory
        # transitions align with failure - modifier then suppresses them most.
        if self._preparatory_grads and self._failure_grads:
            G_prep = torch.stack(list(self._preparatory_grads)).mean(dim=0)
            G_f = torch.stack(list(self._failure_grads)).mean(dim=0)
            c = F.cosine_similarity(
                G_prep.flatten().unsqueeze(0),
                G_f.flatten().unsqueeze(0),
            ).item()
            self.m = float(np.clip(1.0 - c, 0.0, 1.0)) + self.epsilon

        # Reset episode accumulators
        self._episode_accum = None
        self._episode_n = 0
        self._episode_r0_accum = None
        self._episode_r0_n = 0

    # ------------------------------------------------------------------
    # Called from update_priorities() for per-batch sanity tracking
    # ------------------------------------------------------------------

    def set_last_frac_neg(self, frac: float) -> None:
        """Store the fraction of the most recent batch classified r≤0."""
        self._last_frac_neg = frac

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_m(self, log_path: str, episode: int) -> None:
        """Append one row to mora_modifier_log.csv.

        Columns: episode, m, frac_neg_batch
        - m: current modifier value
        - frac_neg_batch: fraction of most recent batch classified r≤0
          (NaN until the first update_priorities call)

        If frac_neg_batch is consistently near 0 or 1, the n-step return
        reward-sign proxy is likely misfiring.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["episode", "m", "frac_neg_batch"])
            writer.writerow([episode, f"{self.m:.6f}", f"{self._last_frac_neg:.4f}"])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MORAPriorityTracker(k={self.k}, epsilon={self.epsilon}, "
            f"m={self.m:.4f}, "
            f"preparatory_buf={len(self._preparatory_grads)}/{self.k}, "
            f"failure_buf={len(self._failure_grads)}/{self.k})"
        )


class OutcomeTracker:
    """Lightweight episode outcome classifier using percentile partitioning.

    Uses the same rolling-window percentile logic as MORAPriorityTracker so that
    outcome-conditioned atom weighting and MORA sampling use a consistent
    success/failure classification.

    Returns +1 (success), -1 (failure), or 0 (neutral) per episode.
    Used by rainbow/train.py to label replay buffer transitions for
    outcome-conditioned atom weighting in agent.learn().

    Parameters
    ----------
    percentile_x : int
        Top/bottom percentile cutoff (default 25).
        Top percentile_x% = success (+1), bottom percentile_x% = failure (-1),
        middle (100 - 2*percentile_x)% = neutral (0).
    window : int
        Rolling window of recent episode returns for computing thresholds.
    """

    def __init__(self, percentile_x: int = 25, window: int = 25):
        self.percentile_x = percentile_x
        self.window = window
        self._recent_returns: deque = deque(maxlen=window)

    def classify(self, ep_return: float) -> int:
        """Classify episode return as success (+1), failure (-1), or neutral (0).

        Appends ep_return to the rolling window before computing thresholds,
        matching MORAPriorityTracker.episode_end() behaviour.
        """
        self._recent_returns.append(ep_return)
        recent = list(self._recent_returns)
        if len(recent) < 2:
            return 0  # insufficient history - treat as neutral
        low_thresh = float(np.percentile(recent, self.percentile_x))
        high_thresh = float(np.percentile(recent, 100 - self.percentile_x))
        if ep_return >= high_thresh:
            return 1
        elif ep_return <= low_thresh:
            return -1
        return 0

    def __repr__(self) -> str:
        return (
            f"OutcomeTracker(percentile_x={self.percentile_x}, "
            f"window={self.window}, "
            f"buf={len(self._recent_returns)}/{self.window})"
        )
