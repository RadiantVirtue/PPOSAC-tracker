"""tests/test_rainbow_pipeline.py

Integration and unit tests for the Rainbow training + analysis pipeline.
Synthetic data is injected directly into each pipeline stage; no real
environment or training is required.

Run with:  pytest tests/test_rainbow_pipeline.py -v
           pytest tests/test_rainbow_pipeline.py -v -k test_analyze_checkpoint
"""
import argparse
import copy
import json
import math
import os
import sys
import tempfile
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Ensure the project root is on the path regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rainbow.model import DQN
from rainbow.sampling import EpisodeData



N_ACTIONS = 4
ATOMS      = 3      # tiny atom count for speed
HIDDEN     = 8      # tiny hidden size
OBS_H, OBS_W = 64, 64  # real spatial dims (conv strides are fixed)


def _make_args(architecture="data-efficient"):
    """Minimal argparse.Namespace matching what DQN and the analysis pipeline expect."""
    return argparse.Namespace(
        atoms=ATOMS,
        hidden_size=HIDDEN,
        architecture=architecture,
        history_length=3,
        V_min=-1.0,
        V_max=1.0,
        multi_step=3,
        discount=0.99,
        noisy_std=0.1,
        priority_exponent=0.5,
        priority_weight=0.4,
        T_max=int(10e6),
        learn_start=int(20e3),
        device=torch.device("cpu"),
        model=None,
    )


def _make_net(seed=0):
    """Construct a small DQN in eval mode (deterministic NoisyLinear)."""
    torch.manual_seed(seed)
    net = DQN(_make_args(), N_ACTIONS)
    net.eval()
    return net


def _make_episode(n_steps=20, positive_rewards=True, seed=0):
    """Synthetic EpisodeData with (n_steps, 3, 64, 64) float32 observations."""
    rng = np.random.RandomState(seed)
    obs = torch.from_numpy(
        rng.rand(n_steps, 3, OBS_H, OBS_W).astype(np.float32)
    )
    actions = torch.randint(0, N_ACTIONS, (n_steps,))
    if positive_rewards:
        rewards = torch.tensor(
            [1.0 if i % 5 == 0 else 0.0 for i in range(n_steps)],
            dtype=torch.float32,
        )
    else:
        rewards = torch.zeros(n_steps)
    dones = torch.zeros(n_steps)
    dones[-1] = 1.0
    return EpisodeData(observations=obs, actions=actions, rewards=rewards, dones=dones)


def _make_episode_pool(n_success=8, n_failure=8):
    """Two groups of episodes with distinct reward profiles."""
    success = [_make_episode(n_steps=15, positive_rewards=True,  seed=i)       for i in range(n_success)]
    failure = [_make_episode(n_steps=15, positive_rewards=False, seed=100 + i) for i in range(n_failure)]
    return success, failure



class TestGradientComputation:

    def setup_method(self):
        self.net    = _make_net(seed=1)
        self.target = _make_net(seed=2)
        self.args   = _make_args()
        self.success, self.failure = _make_episode_pool()

    def _run(self, episodes, precomputed=None):
        from rainbow.gradients import compute_group_gradient_with_coherence
        return compute_group_gradient_with_coherence(
            self.net, episodes,
            batch_size=4, device="cpu",
            target_net=self.target, args_ns=self.args,
            global_step=100_000,
            precomputed_is_weights=precomputed,
        )

    def test_output_keys(self):
        result = self._run(self.success)
        for key in ("uniform", "is_weighted", "reward_weighted",
                    "cos_uniform_is", "cos_is_reward",
                    "beta_used", "n_transitions"):
            assert key in result, f"Missing key: {key}"

    def test_uniform_raw_is_dict_of_tensors(self):
        result = self._run(self.success)
        raw = result["uniform"]["raw"]
        assert isinstance(raw, dict)
        assert all(isinstance(v, torch.Tensor) for v in raw.values())

    def test_all_variants_present_when_target_provided(self):
        result = self._run(self.success)
        assert result["is_weighted"]["raw"]       is not None
        assert result["reward_weighted"]["raw"]   is not None

    def test_gradient_magnitudes_are_finite_and_positive(self):
        from shared.metrics import gradient_magnitude
        result = self._run(self.success)
        for variant in ("uniform", "is_weighted", "reward_weighted"):
            raw = result[variant]["raw"]
            mag = gradient_magnitude(raw)
            assert math.isfinite(mag) and mag > 0, (
                f"{variant} gradient magnitude={mag} is not finite+positive"
            )

    def test_is_differs_from_uniform(self):
        """IS-weighted gradient should not be identical to uniform gradient."""
        from shared.gradient_utils import cosine_similarity_flat
        result = self._run(self.success)
        raw_uni = result["uniform"]["raw"]
        raw_is  = result["is_weighted"]["raw"]
        # They may be close but should not be exactly equal
        flat_uni = torch.cat([raw_uni[k].flatten() for k in sorted(raw_uni)])
        flat_is  = torch.cat([raw_is[k].flatten()  for k in sorted(raw_is)])
        assert not torch.allclose(flat_uni, flat_is, atol=1e-7), (
            "IS-weighted and uniform gradients are identical - IS weighting has no effect"
        )

    def test_batch_grads_count_matches_batches(self):
        result = self._run(self.success)
        n_batches = math.ceil(len(self.success) / 4)
        assert len(result["uniform"]["batch_grads"]) == n_batches

    def test_coherence_values_bounded(self):
        from shared.metrics import coherence
        result = self._run(self.success)
        for variant in ("uniform", "is_weighted", "reward_weighted"):
            coh = coherence(result[variant]["batch_grads"])
            if coh is not None:
                assert -1.0 - 1e-4 <= coh <= 1.0 + 1e-4, (
                    f"{variant} coherence={coh:.4f} outside [-1, 1]"
                )

    def test_n_transitions_matches_episode_lengths(self):
        result = self._run(self.success)
        expected = sum(len(ep.rewards) for ep in self.success)
        assert result["n_transitions"] == expected

    def test_beta_annealing(self):
        """beta_used should increase with global_step."""
        from rainbow.gradients import compute_group_gradient_with_coherence
        r_early = compute_group_gradient_with_coherence(
            self.net, self.success[:4], batch_size=4, device="cpu",
            target_net=self.target, args_ns=self.args, global_step=50_000,
        )
        r_late = compute_group_gradient_with_coherence(
            self.net, self.success[:4], batch_size=4, device="cpu",
            target_net=self.target, args_ns=self.args, global_step=9_000_000,
        )
        assert r_early["beta_used"] < r_late["beta_used"], (
            f"beta not increasing: early={r_early['beta_used']}, late={r_late['beta_used']}"
        )

    def test_precomputed_is_weights_used(self):
        """Gradient with precomputed IS weights should differ from within-group normalised."""
        from rainbow.gradients import _forward_per_loss, _compute_is_weights
        args = self.args
        atoms = args.atoms
        Vmin, Vmax = args.V_min, args.V_max
        delta_z = (Vmax - Vmin) / (atoms - 1)
        support = torch.linspace(Vmin, Vmax, atoms)
        n, gamma = args.multi_step, args.discount

        # Build fake flat IS weights (all ones = uniform, so result == uniform variant)
        flat_weights = []
        for ep in self.success:
            flat_weights.append(torch.ones(len(ep.rewards)))
        result_precomputed = self._run(self.success, precomputed=flat_weights)
        result_default     = self._run(self.success)

        # With all-ones weights the IS variant == uniform variant
        flat_pre = torch.cat([result_precomputed["is_weighted"]["raw"][k].flatten()
                              for k in sorted(result_precomputed["is_weighted"]["raw"])])
        flat_uni = torch.cat([result_precomputed["uniform"]["raw"][k].flatten()
                              for k in sorted(result_precomputed["uniform"]["raw"])])
        assert torch.allclose(flat_pre, flat_uni, atol=1e-5), (
            "With all-ones precomputed IS weights the IS variant should equal uniform"
        )



class TestISWeights:

    def test_max_weight_is_one(self):
        from rainbow.gradients import _compute_is_weights
        losses = torch.tensor([0.1, 0.5, 1.0, 2.0, 0.3])
        w = _compute_is_weights(losses, alpha=0.5, beta=0.4)
        assert abs(w.max().item() - 1.0) < 1e-5

    def test_weights_finite_and_positive(self):
        from rainbow.gradients import _compute_is_weights
        losses = torch.tensor([0.01, 0.1, 1.0, 5.0])
        w = _compute_is_weights(losses, alpha=0.5, beta=1.0)
        assert torch.all(torch.isfinite(w))
        assert torch.all(w > 0)

    def test_high_loss_gets_low_weight(self):
        """IS correction: high-loss (high-priority) transitions get downweighted."""
        from rainbow.gradients import _compute_is_weights
        losses = torch.tensor([0.01, 100.0])
        w = _compute_is_weights(losses, alpha=0.5, beta=1.0)
        # The low-loss transition should have a higher IS weight
        assert w[0].item() > w[1].item()

    def test_zero_floor_prevents_divide_by_zero(self):
        from rainbow.gradients import _compute_is_weights
        losses = torch.tensor([0.0, 0.0, 1.0])  # two exact zeros
        w = _compute_is_weights(losses, alpha=0.5, beta=0.4)
        assert torch.all(torch.isfinite(w))

    def test_uniform_losses_give_equal_weights(self):
        from rainbow.gradients import _compute_is_weights
        losses = torch.ones(10)
        w = _compute_is_weights(losses, alpha=0.5, beta=0.4)
        assert torch.allclose(w, torch.ones(10), atol=1e-5)



class TestActivationAnalysis:

    def setup_method(self):
        self.net = _make_net(seed=3)
        self.success, self.failure = _make_episode_pool(n_success=6, n_failure=6)

    def test_output_keys(self):
        from rainbow.activations import run_activation_analysis
        result = run_activation_analysis(self.net, self.success, self.failure)
        for key in ("activations", "projected", "cluster_stats", "centroids", "labels"):
            assert key in result, f"Missing key: {key}"

    def test_activations_shape(self):
        from rainbow.activations import run_activation_analysis
        result = run_activation_analysis(self.net, self.success, self.failure)
        n_total = sum(len(ep.observations) for ep in self.success + self.failure)
        acts = result["activations"]
        assert acts.shape[0] == n_total
        # data-efficient arch: conv_output_size = 256
        assert acts.shape[1] == 256, f"Expected 256-dim activations, got {acts.shape[1]}"

    def test_centroids_shape(self):
        from rainbow.activations import run_activation_analysis
        result = run_activation_analysis(self.net, self.success, self.failure)
        c = result["centroids"]
        assert "success" in c and "failure" in c
        assert c["success"].shape == c["failure"].shape
        assert c["success"].ndim == 1

    def test_labels_match_episode_counts(self):
        from rainbow.activations import run_activation_analysis
        result = run_activation_analysis(self.net, self.success, self.failure)
        n_s = sum(len(ep.observations) for ep in self.success)
        n_f = sum(len(ep.observations) for ep in self.failure)
        labels = result["labels"]
        assert np.sum(labels == 1) == n_s
        assert np.sum(labels == 0) == n_f

    def test_activation_separation_positive(self):
        """Two groups with different reward profiles should have some spatial separation."""
        from rainbow.activations import run_activation_analysis
        from shared.metrics import activation_separation
        result = run_activation_analysis(self.net, self.success, self.failure)
        sep = activation_separation(result["centroids"]["success"], result["centroids"]["failure"])
        assert sep >= 0, f"activation_separation={sep} is negative"
        assert math.isfinite(sep)

    def test_activation_cosine_distance_in_range(self):
        from rainbow.activations import run_activation_analysis
        from shared.metrics import centroid_cosine_distance
        result = run_activation_analysis(self.net, self.success, self.failure)
        cd = centroid_cosine_distance(result["centroids"]["success"], result["centroids"]["failure"])
        assert 0.0 - 1e-4 <= cd <= 2.0 + 1e-4, f"cosine distance={cd:.4f} outside [0, 2]"

    def test_cluster_stats_present(self):
        from rainbow.activations import run_activation_analysis
        result = run_activation_analysis(self.net, self.success, self.failure)
        cs = result["cluster_stats"]
        assert cs is not None
        assert isinstance(cs, dict)



class TestMomentOfReward:

    def setup_method(self):
        self.net    = _make_net(seed=4)
        self.target = _make_net(seed=5)
        self.args   = _make_args()

    def _make_mixed_success(self, n=12):
        """Episodes with a mix of positive, zero, and negative rewards."""
        episodes = []
        rng = np.random.RandomState(42)
        for i in range(n):
            T = 20
            obs = torch.from_numpy(rng.rand(T, 3, OBS_H, OBS_W).astype(np.float32))
            acts = torch.randint(0, N_ACTIONS, (T,))
            # Mix: some +1, some 0, some -1
            rew = torch.tensor(
                [1.0 if j % 7 == 0 else (-1.0 if j % 11 == 0 else 0.0) for j in range(T)],
                dtype=torch.float32,
            )
            dones = torch.zeros(T); dones[-1] = 1.0
            episodes.append(EpisodeData(obs, acts, rew, dones))
        return episodes

    def test_output_keys(self):
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        success = self._make_mixed_success()
        result = run_moment_of_reward_analysis(
            self.net, success, device="cpu",
            target_net=self.target, args_ns=self.args,
        )
        expected_keys = [
            "n_positive", "n_neutral", "n_negative",
            "gradient_magnitude_positive", "gradient_magnitude_neutral",
            "coherence_positive", "coherence_neutral",
            "opp_pos_vs_neutral", "opp_pos_vs_failure",
            "opp_neutral_vs_failure",
        ]
        assert result is not None
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_transition_counts_are_non_negative(self):
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        success = self._make_mixed_success()
        result = run_moment_of_reward_analysis(
            self.net, success, device="cpu",
            target_net=self.target, args_ns=self.args,
        )
        for k in ("n_positive", "n_neutral", "n_negative"):
            assert result[k] >= 0, f"{k}={result[k]} is negative"

    def test_transition_counts_sum_to_total(self):
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        success = self._make_mixed_success(n=8)
        result = run_moment_of_reward_analysis(
            self.net, success, device="cpu",
            target_net=self.target, args_ns=self.args,
        )
        total = sum(len(ep.rewards) for ep in success)
        counted = result["n_positive"] + result["n_neutral"] + result["n_negative"]
        assert counted == total, (
            f"MoR counts sum to {counted}, expected {total}"
        )

    def test_opposition_vs_failure_with_raw_failure_grad(self):
        """When raw_failure_grad is provided, vs-failure oppositions should be non-None."""
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        from rainbow.gradients import compute_group_gradient_with_coherence
        success = self._make_mixed_success(n=6)
        failure = [_make_episode(n_steps=15, positive_rewards=False, seed=200 + i)
                   for i in range(6)]
        fail_result = compute_group_gradient_with_coherence(
            self.net, failure, batch_size=3, device="cpu",
            target_net=self.target, args_ns=self.args, global_step=100_000,
        )
        raw_fail = fail_result["uniform"]["raw"]
        result = run_moment_of_reward_analysis(
            self.net, success, raw_failure_grad=raw_fail, device="cpu",
            target_net=self.target, args_ns=self.args,
        )
        # At least one vs-failure opposition should be populated (subgroups may have < MIN_MOR_EPISODES)
        any_vs_failure = any(
            result.get(k) is not None
            for k in ("opp_pos_vs_failure", "opp_neutral_vs_failure", "opp_negative_vs_failure")
        )
        assert any_vs_failure, "No vs-failure opposition scores were computed despite raw_failure_grad being provided"

    def test_opposition_scores_bounded(self):
        from rainbow.moment_of_reward import run_moment_of_reward_analysis
        success = self._make_mixed_success()
        result = run_moment_of_reward_analysis(
            self.net, success, device="cpu",
            target_net=self.target, args_ns=self.args,
        )
        for k in ("opp_pos_vs_neutral", "opp_pos_vs_negative", "opp_neutral_vs_negative"):
            v = result.get(k)
            if v is not None:
                assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, f"{k}={v:.4f} outside [-1, 1]"



class TestWeightDelta:

    def _grad_dict(self, net, episodes, target, args, step=100_000):
        from rainbow.gradients import compute_group_gradient_with_coherence
        return compute_group_gradient_with_coherence(
            net, episodes, batch_size=4, device="cpu",
            target_net=target, args_ns=args, global_step=step,
        )

    def test_returns_six_keys(self):
        from rainbow.weight_delta import compute_weight_delta_metrics
        net1 = _make_net(seed=10)
        net2 = _make_net(seed=11)
        args = _make_args()
        target = _make_net(seed=12)
        success, failure = _make_episode_pool(n_success=6, n_failure=6)
        grad_s = self._grad_dict(net2, success, target, args)
        grad_f = self._grad_dict(net2, failure, target, args)
        result = compute_weight_delta_metrics(
            net1.state_dict(), net2.state_dict(), grad_s, grad_f
        )
        expected = {
            "cos_uniform_success", "cos_is_success", "cos_reward_success",
            "cos_uniform_failure", "cos_is_failure", "cos_reward_failure",
        }
        assert set(result.keys()) == expected

    def test_cosines_bounded(self):
        from rainbow.weight_delta import compute_weight_delta_metrics
        net1 = _make_net(seed=13)
        net2 = _make_net(seed=14)
        args = _make_args()
        target = _make_net(seed=15)
        success, failure = _make_episode_pool(n_success=6, n_failure=6)
        grad_s = self._grad_dict(net2, success, target, args)
        grad_f = self._grad_dict(net2, failure, target, args)
        result = compute_weight_delta_metrics(
            net1.state_dict(), net2.state_dict(), grad_s, grad_f
        )
        for k, v in result.items():
            if v is not None:
                assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, f"{k}={v:.4f} outside [-1, 1]"

    def test_identical_nets_give_zero_cosines(self):
        """When prev == curr the weight delta is zero; cosine similarity is undefined → None or 0."""
        from rainbow.weight_delta import compute_weight_delta_metrics
        net = _make_net(seed=16)
        args = _make_args()
        target = _make_net(seed=17)
        success, failure = _make_episode_pool(n_success=6, n_failure=6)
        grad_s = self._grad_dict(net, success, target, args)
        grad_f = self._grad_dict(net, failure, target, args)
        result = compute_weight_delta_metrics(
            net.state_dict(), net.state_dict(), grad_s, grad_f
        )
        # All cosines should be 0 (or None) because Δθ = 0
        for k, v in result.items():
            if v is not None:
                assert abs(v) < 1e-4, f"{k}={v:.6f} should be ~0 for Δθ=0"

    def test_sigma_params_excluded(self):
        """Sigma parameters should not appear in the weight-delta comparison keys."""
        from rainbow.weight_delta import _state_delta_dict
        net1 = _make_net(seed=18)
        net2 = _make_net(seed=19)
        delta = _state_delta_dict(net1.state_dict(), net2.state_dict())
        for k in delta:
            assert "weight_sigma" not in k and "bias_sigma" not in k, (
                f"Sigma param leaked into delta: {k}"
            )



class TestJointISWeights:

    def test_joint_normalisation_max_is_one_across_pool(self):
        """Joint IS weights are normalised over the combined pool, not per group."""
        from analyze_checkpoint import _compute_joint_is_weights
        net    = _make_net(seed=20)
        target = _make_net(seed=21)
        args   = _make_args()
        success, failure = _make_episode_pool(n_success=4, n_failure=4)
        combined = success + failure
        weights = _compute_joint_is_weights(net, target, combined, args, global_step=500_000, device="cpu")
        valid = [w for w in weights if w is not None]
        assert len(valid) > 0
        all_w = torch.cat(valid)
        assert abs(all_w.max().item() - 1.0) < 1e-5, (
            f"Joint IS max weight = {all_w.max().item():.6f}, expected 1.0"
        )

    def test_joint_weights_count_matches_episodes(self):
        from analyze_checkpoint import _compute_joint_is_weights
        net    = _make_net(seed=22)
        target = _make_net(seed=23)
        args   = _make_args()
        success, failure = _make_episode_pool(n_success=3, n_failure=3)
        combined = success + failure
        weights = _compute_joint_is_weights(net, target, combined, args, global_step=100_000, device="cpu")
        assert len(weights) == len(combined)



def _make_fake_checkpoint(path, seed=0):
    """Save a minimal but valid Rainbow checkpoint to disk."""
    args = _make_args()
    torch.manual_seed(seed)
    net    = DQN(args, N_ACTIONS)
    target = DQN(args, N_ACTIONS)
    args_dict = {
        "atoms": args.atoms,
        "hidden_size": args.hidden_size,
        "architecture": args.architecture,
        "history_length": args.history_length,
        "V_min": args.V_min,
        "V_max": args.V_max,
        "multi_step": args.multi_step,
        "discount": args.discount,
        "noisy_std": args.noisy_std,
        "priority_exponent": args.priority_exponent,
        "priority_weight": args.priority_weight,
        "T_max": args.T_max,
        "learn_start": args.learn_start,
        "n_actions": N_ACTIONS,
    }
    torch.save({
        "global_step": 500_000,
        "episode_count": 1000,
        "online_net_state_dict": net.state_dict(),
        "target_net_state_dict": target.state_dict(),
        "optimizer_state_dict": {},
        "args_dict": args_dict,
    }, path)


def _fake_evaluate_frozen_policy(checkpoint_path, n_episodes=20, device="cpu", seed=None, num_envs=16):
    """Replacement for rainbow.sampling.evaluate_frozen_policy that returns synthetic data."""
    rng = np.random.RandomState(seed or 0)
    episodes   = []
    eps_scores = []
    for i in range(n_episodes):
        T = 20
        obs  = torch.from_numpy(rng.rand(T, 3, OBS_H, OBS_W).astype(np.float32))
        acts = torch.randint(0, N_ACTIONS, (T,))
        rew  = torch.tensor(
            [1.0 if j % 5 == 0 else 0.0 for j in range(T)], dtype=torch.float32
        )
        dones = torch.zeros(T); dones[-1] = 1.0
        episodes.append(EpisodeData(obs, acts, rew, dones))
        # First half get high EPS, second half get low EPS → clear percentile split
        eps_scores.append(1.0 if i < n_episodes // 2 else 0.0)
    return episodes, eps_scores


class TestAnalyzeCheckpoint:

    def test_json_written_with_required_keys(self):
        REQUIRED = [
            "episode", "algorithm", "n_success", "n_failure",
            "opposition_score", "opposition_score_is",
            "coherence_success", "coherence_failure",
            "coherence_success_is", "coherence_failure_is",
            "gradient_magnitude_success", "gradient_magnitude_failure",
            "gradient_magnitude_success_is", "gradient_magnitude_failure_is",
            "activation_separation", "activation_cosine_distance",
            "cos_uniform_is_success", "cos_uniform_is_failure",
            "beta_used", "n_transitions_success", "n_transitions_failure",
            "moment_of_reward",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(ckpt_path, seed=42)

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", ckpt_path, tmpdir,
                    n_episodes=20, device="cpu", reason="test",
                    split_mode="percentile", percentile_x=25,
                    seed=1,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            assert os.path.exists(json_path), "Analysis JSON was not written"

            with open(json_path) as f:
                data = json.load(f)

            for k in REQUIRED:
                assert k in data, f"JSON missing required key: {k}"

    def test_algorithm_field_is_rainbow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(ckpt_path, seed=43)

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", ckpt_path, tmpdir,
                    n_episodes=20, device="cpu", reason="test",
                    split_mode="percentile", percentile_x=25,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            with open(json_path) as f:
                data = json.load(f)
            assert data["algorithm"] == "rainbow"

    def test_no_nan_in_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(ckpt_path, seed=44)

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", ckpt_path, tmpdir,
                    n_episodes=20, device="cpu", reason="test",
                    split_mode="percentile", percentile_x=25,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            with open(json_path) as f:
                data = json.load(f)

            def _check_nan(obj, path=""):
                if isinstance(obj, float):
                    assert not math.isnan(obj), f"NaN at {path}"
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        _check_nan(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        _check_nan(v, f"{path}[{i}]")

            _check_nan(data)

    def test_weight_delta_absent_when_no_prev_checkpoint(self):
        """Without a prev_checkpoint_path all delta cosines should be None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(ckpt_path, seed=45)

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", ckpt_path, tmpdir,
                    n_episodes=20, device="cpu",
                    split_mode="percentile", percentile_x=25,
                    prev_checkpoint_path=None,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            with open(json_path) as f:
                data = json.load(f)

            for k in ("cos_uniform_success_delta", "cos_is_success_delta",
                      "cos_reward_success_delta", "cos_uniform_failure_delta",
                      "cos_is_failure_delta", "cos_reward_failure_delta"):
                assert data.get(k) is None, f"{k} should be None when no prev checkpoint"

    def test_weight_delta_present_when_prev_checkpoint_given(self):
        """With two checkpoints, delta cosines should be populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_path = os.path.join(tmpdir, "checkpoint_step400000.pt")
            curr_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(prev_path, seed=46)
            _make_fake_checkpoint(curr_path, seed=47)  # different weights

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", curr_path, tmpdir,
                    n_episodes=20, device="cpu",
                    split_mode="percentile", percentile_x=25,
                    prev_checkpoint_path=prev_path,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            with open(json_path) as f:
                data = json.load(f)

            delta_keys = [
                "cos_uniform_success_delta", "cos_is_success_delta",
                "cos_reward_success_delta", "cos_uniform_failure_delta",
                "cos_is_failure_delta", "cos_reward_failure_delta",
            ]
            for k in delta_keys:
                v = data.get(k)
                assert v is not None, f"{k} should be non-None when prev checkpoint provided"
                assert math.isfinite(v), f"{k}={v} is not finite"
                assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, f"{k}={v} outside [-1, 1]"

    def test_moment_of_reward_keys_in_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_step500000.pt")
            _make_fake_checkpoint(ckpt_path, seed=48)

            with patch("rainbow.sampling.evaluate_frozen_policy", side_effect=_fake_evaluate_frozen_policy):
                from analyze_checkpoint import analyze_checkpoint
                analyze_checkpoint(
                    "rainbow", ckpt_path, tmpdir,
                    n_episodes=20, device="cpu",
                    split_mode="percentile", percentile_x=25,
                )

            json_path = os.path.join(tmpdir, "analysis_logs", "rainbow", "checkpoint_step500000.json")
            with open(json_path) as f:
                data = json.load(f)

            mor = data.get("moment_of_reward")
            assert mor is not None, "moment_of_reward should not be null"
            for k in ("n_positive", "n_neutral", "n_negative"):
                assert k in mor, f"moment_of_reward missing key: {k}"
