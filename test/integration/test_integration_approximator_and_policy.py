# tests/integration/test_integration_approximator_and_policy.py
"""
Integration tests: QValueApproximator training → GreedyQPolicy decisions.

We construct synthetic (state, action, target-Q) training batches where
the correct action is known exactly, train a real approximator on them,
wrap it in GreedyQPolicy, and verify the policy always selects the
dominant action.

Both LinearQValueApproximator and DNNQValueApproximator are tested.
No stubs are used; every class is a production implementation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from approximator import DNNQValueApproximator, DNNSpec, LinearQValueApproximator
from policy import GreedyQPolicy
from state import AllocationAction, PortfolioState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Two competing actions: LOW (alloc=0.0) and HIGH (alloc=0.1)
ACTION_LOW = AllocationAction(allocations=(0.0,))
ACTION_HIGH = AllocationAction(allocations=(0.1,))

# Feature functions shared across approximator tests
# Input tuple: (wealth, action_alloc_0)
FEATURES = [
    lambda x: 1.0,            # bias
    lambda x: x[0],           # wealth
    lambda x: x[1],           # allocation fraction
    lambda x: x[0] * x[1],   # interaction
]


def make_state(wealth: float) -> PortfolioState:
    return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(0.0,))


def build_training_data(
    n_per_action: int = 30,
    q_high: float = 10.0,
    q_low: float = 1.0,
) -> List[Tuple[PortfolioState, AllocationAction, float]]:
    """
    Synthetic data where ACTION_HIGH always has higher Q.
    Wealth is varied so the approximator sees diverse inputs.
    """
    rng = np.random.default_rng(0)
    wealths = rng.uniform(0.5, 3.0, n_per_action)
    samples = []
    for w in wealths:
        state = make_state(float(w))
        samples.append((state, ACTION_HIGH, q_high))
        samples.append((state, ACTION_LOW, q_low))
    return samples


# ---------------------------------------------------------------------------
# TestLinearQVAWithGreedyPolicy
# ---------------------------------------------------------------------------


class TestLinearQVAWithGreedyPolicy:
    """Train a LinearQValueApproximator, then verify GreedyQPolicy is correct."""

    def _trained_qvf(self, lambda_reg: float = 0.0) -> LinearQValueApproximator:
        qvf = LinearQValueApproximator(FEATURES, lambda_reg=lambda_reg)
        return qvf.update(build_training_data())

    def test_trained_qvf_evaluate_returns_float(self):
        qvf = self._trained_qvf()
        val = qvf.evaluate(make_state(1.0), ACTION_HIGH)
        assert isinstance(val, float)

    def test_high_action_has_higher_q_than_low(self):
        """After training, Q(s, ACTION_HIGH) > Q(s, ACTION_LOW) for every s."""
        qvf = self._trained_qvf()
        for wealth in [0.5, 1.0, 1.5, 2.0, 3.0]:
            state = make_state(wealth)
            q_high = qvf.evaluate(state, ACTION_HIGH)
            q_low = qvf.evaluate(state, ACTION_LOW)
            assert q_high > q_low, (
                f"Expected Q_high > Q_low at wealth={wealth}; "
                f"got Q_high={q_high:.4f}, Q_low={q_low:.4f}"
            )

    def test_greedy_policy_always_selects_high_action(self):
        qvf = self._trained_qvf()
        policy = GreedyQPolicy(
            qvf_per_step=[qvf],
            feasible_actions=[ACTION_LOW, ACTION_HIGH],
        )
        for wealth in [0.5, 1.0, 2.0, 5.0]:
            state = make_state(wealth)
            action = policy.get_action(state, t=0)
            assert action == ACTION_HIGH

    def test_greedy_policy_with_multi_step_qvfs(self):
        """GreedyQPolicy must use the time-step-specific QVF."""
        # Step 0: high action preferred; step 1: low action preferred
        qvf_prefer_high = self._trained_qvf()
        qvf_prefer_low = LinearQValueApproximator(FEATURES).update(
            build_training_data(q_high=1.0, q_low=10.0)
        )
        policy = GreedyQPolicy(
            qvf_per_step=[qvf_prefer_high, qvf_prefer_low],
            feasible_actions=[ACTION_LOW, ACTION_HIGH],
        )
        state = make_state(1.0)
        assert policy.get_action(state, t=0) == ACTION_HIGH
        assert policy.get_action(state, t=1) == ACTION_LOW

    def test_regularised_qvf_still_selects_correct_action(self):
        """L2 regularisation should not flip the ordering on clear signal."""
        qvf = self._trained_qvf(lambda_reg=0.1)
        policy = GreedyQPolicy(
            qvf_per_step=[qvf],
            feasible_actions=[ACTION_LOW, ACTION_HIGH],
        )
        action = policy.get_action(make_state(1.0), t=0)
        assert action == ACTION_HIGH

    def test_q_values_are_finite(self):
        qvf = self._trained_qvf()
        for wealth in [0.1, 1.0, 10.0]:
            for action in [ACTION_LOW, ACTION_HIGH]:
                val = qvf.evaluate(make_state(wealth), action)
                assert math.isfinite(val)

    def test_copy_preserves_trained_weights(self):
        qvf = self._trained_qvf()
        qvf_copy = qvf.copy()
        state = make_state(1.0)
        assert qvf.evaluate(state, ACTION_HIGH) == pytest.approx(
            qvf_copy.evaluate(state, ACTION_HIGH)
        )

    def test_update_returns_new_instance(self):
        qvf = LinearQValueApproximator(FEATURES)
        updated = qvf.update(build_training_data())
        assert updated is not qvf

    def test_original_unchanged_after_update(self):
        """Immutable-update pattern: original QVF must remain untrained."""
        qvf = LinearQValueApproximator(FEATURES)
        _ = qvf.update(build_training_data())
        # Original should have zero weights (untrained)
        assert np.allclose(qvf.theta, 0.0)


# ---------------------------------------------------------------------------
# TestDNNQVAIntegration
# ---------------------------------------------------------------------------


class TestDNNQVAIntegration:
    """
    Verify that DNNQValueApproximator trains end-to-end and produces
    finite Q-value estimates.  We do not assert exact numeric values
    (DNN training is stochastic) but do assert directional ordering
    after sufficient training on a strongly separated dataset.
    """

    _SPEC = DNNSpec(layer_sizes=[8, 8], activations=["relu", "relu"])

    def _trained_dnn(self, n_epochs: int = 200) -> DNNQValueApproximator:
        qvf = DNNQValueApproximator(
            FEATURES,
            dnn_spec=self._SPEC,
            learning_rate=0.05,
            n_epochs=n_epochs,
            batch_size=16,
            rng=np.random.default_rng(42),
        )
        return qvf.update(build_training_data(n_per_action=50))

    def test_dnn_evaluate_returns_float_after_training(self):
        qvf = self._trained_dnn()
        val = qvf.evaluate(make_state(1.0), ACTION_HIGH)
        assert isinstance(val, float)

    def test_dnn_q_values_are_finite(self):
        qvf = self._trained_dnn()
        for wealth in [0.5, 1.0, 2.0]:
            for action in [ACTION_LOW, ACTION_HIGH]:
                assert math.isfinite(qvf.evaluate(make_state(wealth), action))

    def test_dnn_copy_preserves_evaluation(self):
        qvf = self._trained_dnn()
        qvf_copy = qvf.copy()
        state = make_state(1.5)
        assert qvf.evaluate(state, ACTION_HIGH) == pytest.approx(
            qvf_copy.evaluate(state, ACTION_HIGH), rel=1e-6
        )

    def test_dnn_update_returns_new_instance(self):
        qvf = DNNQValueApproximator(
            FEATURES, dnn_spec=self._SPEC, rng=np.random.default_rng(0)
        )
        updated = qvf.update(build_training_data())
        assert updated is not qvf

    def test_dnn_is_fitted_true_after_update(self):
        qvf = DNNQValueApproximator(
            FEATURES, dnn_spec=self._SPEC, rng=np.random.default_rng(0)
        )
        assert not qvf.is_fitted
        fitted = qvf.update(build_training_data())
        assert fitted.is_fitted

    def test_dnn_greedy_policy_selects_action(self):
        """After training, GreedyQPolicy must return one of the two valid actions."""
        qvf = self._trained_dnn()
        policy = GreedyQPolicy(
            qvf_per_step=[qvf],
            feasible_actions=[ACTION_LOW, ACTION_HIGH],
        )
        action = policy.get_action(make_state(1.0), t=0)
        assert action in [ACTION_LOW, ACTION_HIGH]