import os
import pickle

import numpy as np
import torch

from sac.tagged_buffer import TaggedReplayBuffer


class TaggedPERBuffer(TaggedReplayBuffer):
    """TaggedReplayBuffer with Prioritized Experience Replay (PER).

    Extends TaggedReplayBuffer with a SumTree for O(log N) priority-weighted
    sampling and importance-sampling (IS) weight correction. All episode tagging
    functionality (backfill_episode, get_all_valid) is inherited unchanged.

    sample() returns (batch_dict, is_weights_tensor, leaf_indices_array) instead
    of just batch_dict. Callers must call update_priorities() after each Q-update.

    Args:
        per_alpha:      priority exponent (0=uniform, 1=greedy). Default 0.6.
        per_beta_start: initial IS exponent (anneals to per_beta_end). Default 0.4.
        per_beta_end:   final IS exponent (fully unbiased). Default 1.0.
        per_beta_steps: number of sample() calls over which beta anneals. Default 500_000.
        per_epsilon:    small constant added to |TD error| to prevent zero priority. Default 1e-6.
    """

    def __init__(self, buffer_size, obs_shape, n_actions, device,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0,
                 per_beta_steps=500_000, per_epsilon=1e-6):
        super().__init__(buffer_size, obs_shape, n_actions, device)

        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_epsilon = per_epsilon

        # SumTree: flat float64 array of size 2*N - 1.
        # Leaf i is at position (N - 1 + i). Internal nodes store sum of children.
        self._leaf_offset = buffer_size - 1
        self._tree = np.zeros(2 * buffer_size - 1, dtype=np.float64)
        self._max_priority = 1.0   # new transitions are inserted at max priority
        self._beta_step = 0

    # ------------------------------------------------------------------
    # SumTree internals
    # ------------------------------------------------------------------

    def _tree_update(self, leaf_idx, priority):
        """Set leaf priority and propagate sum changes up to the root. O(log N)."""
        node = self._leaf_offset + leaf_idx
        self._tree[node] = priority
        while node > 0:
            node = (node - 1) // 2
            self._tree[node] = self._tree[2 * node + 1] + self._tree[2 * node + 2]

    def _tree_sample(self, value):
        """Walk the tree from root for cumulative-sum value. Returns (leaf_idx, priority). O(log N)."""
        node = 0
        while node < self._leaf_offset:
            left = 2 * node + 1
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = left + 1
        return node - self._leaf_offset, self._tree[node]

    # ------------------------------------------------------------------
    # Public interface (overrides)
    # ------------------------------------------------------------------

    def add(self, obs, next_obs, action, reward, done, episode_id):
        leaf_idx = self.pos   # capture BEFORE super().add() increments pos
        super().add(obs, next_obs, action, reward, done, episode_id)
        # New transitions start at maximum observed priority (ensures they are sampled at least once)
        self._tree_update(leaf_idx, self._max_priority ** self.per_alpha)

    def sample(self, batch_size):
        """Priority-weighted stratified sample.

        Returns:
            batch:        dict of tensors (same keys as TaggedReplayBuffer.sample)
            is_weights:   (batch_size,) float32 tensor, max-normalised IS correction weights
            leaf_indices: (batch_size,) int64 numpy array for update_priorities()
        """
        size = self.buffer_size if self.full else self.pos
        total = self._tree[0]

        # Anneal beta linearly from per_beta_start → per_beta_end
        t = min(1.0, self._beta_step / self.per_beta_steps)
        beta = self.per_beta_start + t * (self.per_beta_end - self.per_beta_start)
        self._beta_step += 1

        # Stratified sampling: divide [0, total] into batch_size segments
        segment = total / batch_size
        leaf_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            # Clamp v to valid range to handle floating-point rounding at the top
            v = min(v, total - 1e-12)
            leaf_idx, priority = self._tree_sample(v)
            # Guard: leaf_idx must be within the valid buffer range
            leaf_idx = min(leaf_idx, size - 1)
            leaf_indices[i] = leaf_idx
            priorities[i] = max(priority, 1e-12)   # guard against exact zero

        # Importance-sampling weights: w_i = (N * P(i))^{-beta}, normalised by max(w)
        sampling_probs = priorities / total
        is_weights = (size * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()   # normalise so max weight = 1
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        batch = {
            "observations":      torch.tensor(self.observations[leaf_indices],      dtype=torch.float32).to(self.device),
            "next_observations": torch.tensor(self.next_observations[leaf_indices], dtype=torch.float32).to(self.device),
            "actions":           torch.tensor(self.actions[leaf_indices]).to(self.device),
            "rewards":           torch.tensor(self.rewards[leaf_indices],           dtype=torch.float32).to(self.device),
            "dones":             torch.tensor(self.dones[leaf_indices],             dtype=torch.float32).to(self.device),
        }
        return batch, is_weights_t, leaf_indices

    def update_priorities(self, leaf_indices, td_errors):
        """Update priorities for a batch of transitions after a Q-network update.

        Args:
            leaf_indices: numpy array of int64 leaf indices returned by sample()
            td_errors:    numpy array of per-sample |TD error| values (detached)
        """
        new_priorities = (np.abs(td_errors) + self.per_epsilon) ** self.per_alpha
        self._max_priority = max(self._max_priority, float(new_priorities.max()))
        for idx, p in zip(leaf_indices, new_priorities):
            self._tree_update(int(idx), float(p))

    # ------------------------------------------------------------------
    # Serialisation (override to include PER state)
    # ------------------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "observations":      self.observations,
            "next_observations": self.next_observations,
            "actions":           self.actions,
            "rewards":           self.rewards,
            "dones":             self.dones,
            "episode_ids":       self.episode_ids,
            "episode_returns":   self.episode_returns,
            "episode_eps":       self.episode_eps,
            "pos":               self.pos,
            "full":              self.full,
            # PER-specific
            "_tree":             self._tree,
            "_max_priority":     self._max_priority,
            "_beta_step":        self._beta_step,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, device="cuda"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obs_shape = data["observations"].shape[1:]
        buf = cls(data["observations"].shape[0], obs_shape, 1, device)
        for key in data:
            setattr(buf, key, data[key])
        # If loading an old-style TaggedReplayBuffer pickle (no tree), rebuild
        # uniform priorities for all valid transitions so sample() works.
        if "_tree" not in data:
            size = buf.buffer_size if buf.full else buf.pos
            uniform_p = 1.0 ** buf.per_alpha
            for i in range(size):
                buf._tree_update(i, uniform_p)
            buf._max_priority = 1.0
        return buf
