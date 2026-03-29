"""
Unit tests for approximator.py — LinearQValueApproximator,
DNNQValueApproximator, DNNSpec, and _extract_input.

Test philosophy
---------------
* All targets for LinearQValueApproximator are hand-computed so failures
  are immediately diagnosable without running a reference implementation.
* DNN tests concentrate on structural contracts (immutability, predictted state,
  weight independence) plus a convergence sanity check on a trivially
  learnable constant target.  Per-weight numerical assertions are avoided
  because they would couple tests to the exact SGD trajectory.
* Fixtures are narrow — each test creates exactly what it needs — to keep
  failure messages clear and avoid hidden coupling between cases.
"""
from __future__ import annotations
import sys
from pathlib import Path
import math
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from approximator import (
    DNNQValueApproximator,
    DNNSpec,
    LinearQValueApproximator,
    QValueApproximator,
    _extract_input,
)
from state import AllocationAction, PortfolioState


# ===========================================================================
# Helpers — state / action / sample factories
# ===========================================================================


def make_state(wealth: float = 1.0, alloc: float = 0.0) -> PortfolioState:
    return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(alloc,))


def make_action(alloc: float = 0.0) -> AllocationAction:
    return AllocationAction(allocations=(alloc,))


def make_sample(
    wealth: float = 1.0,
    alloc: float = 0.0,
    target: float = 0.0,
) -> Tuple[PortfolioState, AllocationAction, float]:
    return (make_state(wealth, alloc), make_action(alloc), target)


# ===========================================================================
# Standard feature functions
# ===========================================================================

# Single-feature sets that yield analytic OLS solutions

f_wealth: callable = lambda x: x[0]          # x[0] = wealth
f_alloc:  callable = lambda x: x[1]          # x[1] = first action alloc
f_const:  callable = lambda x: 1.0           # bias term
f_wealth_sq: callable = lambda x: x[0] ** 2  # quadratic wealth feature


# ===========================================================================
# Shared DNN spec helpers
# ===========================================================================


def small_spec(activations: List[str] | None = None) -> DNNSpec:
    """One hidden layer of width 8, relu by default."""
    return DNNSpec(layer_sizes=[8], activations=activations)


def shallow_spec() -> DNNSpec:
    """No hidden layers — reduces DNN to linear regression."""
    return DNNSpec(layer_sizes=[], activations=[], output_activation="linear")


def deep_spec() -> DNNSpec:
    return DNNSpec(layer_sizes=[16, 8, 4])


# ===========================================================================
# _extract_input
# ===========================================================================


class TestExtractInput:
    def test_single_asset_layout(self):
        state = make_state(wealth=2.5, alloc=0.3)
        action = make_action(alloc=0.7)
        result = _extract_input(state, action)
        assert result == (2.5, 0.7)

    def test_multi_asset_includes_all_allocs(self):
        state = PortfolioState(wealth=1.0, prices=(1.0, 1.0), allocations=(0.3, 0.2))
        action = AllocationAction(allocations=(0.4, 0.5))
        result = _extract_input(state, action)
        assert result == (1.0, 0.4, 0.5)

    def test_returns_floats(self):
        result = _extract_input(make_state(1), make_action(0))
        assert all(isinstance(v, float) for v in result)

    def test_state_alloc_not_in_output(self):
        """The current portfolio alloc (in state) should NOT appear in x."""
        state = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.99,))
        action = AllocationAction(allocations=(0.1,))
        x = _extract_input(state, action)
        # Only wealth=1.0 and action alloc=0.1 should be present
        assert x == (1.0, 0.1)

    def test_wealth_is_first_element(self):
        state = make_state(wealth=3.14)
        action = make_action(alloc=0.5)
        assert _extract_input(state, action)[0] == pytest.approx(3.14)


# ===========================================================================
# DNNSpec
# ===========================================================================


class TestDNNSpec:
    def test_valid_construction_single_hidden(self):
        spec = DNNSpec(layer_sizes=[32])
        assert spec.layer_sizes == [32]

    def test_valid_construction_multi_hidden(self):
        spec = DNNSpec(layer_sizes=[64, 32, 16])
        assert len(spec.layer_sizes) == 3

    def test_default_activations_all_relu(self):
        spec = DNNSpec(layer_sizes=[8, 4])
        assert spec.activations == ["relu", "relu"]

    def test_explicit_activations_stored(self):
        spec = DNNSpec(layer_sizes=[8, 4], activations=["tanh", "sigmoid"])
        assert spec.activations == ["tanh", "sigmoid"]

    def test_default_output_activation_is_linear(self):
        spec = DNNSpec(layer_sizes=[8])
        assert spec.output_activation == "linear"

    def test_custom_output_activation(self):
        spec = DNNSpec(layer_sizes=[8], output_activation="tanh")
        assert spec.output_activation == "tanh"

    def test_empty_layer_sizes_valid(self):
        spec = DNNSpec(layer_sizes=[], activations=[])
        assert spec.layer_sizes == []
        assert spec.activations == []

    def test_mismatched_activations_length_raises(self):
        with pytest.raises(ValueError, match="len(activations)"):
            DNNSpec(layer_sizes=[8, 4], activations=["relu"])

    def test_invalid_hidden_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            DNNSpec(layer_sizes=[8], activations=["softmax"])

    def test_invalid_output_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            DNNSpec(layer_sizes=[8], output_activation="softmax")

    def test_zero_layer_size_raises(self):
        with pytest.raises(ValueError, match="layer_sizes must all be >= 1"):
            DNNSpec(layer_sizes=[0])

    def test_negative_layer_size_raises(self):
        with pytest.raises(ValueError, match="layer_sizes must all be >= 1"):
            DNNSpec(layer_sizes=[8, -1])

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "linear"])
    def test_all_valid_activations_accepted(self, act: str):
        spec = DNNSpec(layer_sizes=[4], activations=[act])
        assert spec.activations == [act]

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "linear"])
    def test_all_valid_output_activations_accepted(self, act: str):
        spec = DNNSpec(layer_sizes=[4], output_activation=act)
        assert spec.output_activation == act


# ===========================================================================
# LinearQValueApproximator — constructor
# ===========================================================================


class TestLinearConstructor:
    def test_stores_features(self):
        feats = [f_wealth, f_alloc]
        approx = LinearQValueApproximator(feats)
        assert approx._features == feats

    def test_initialises_theta_to_zeros(self):
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        assert np.all(approx._theta == 0.0)

    def test_theta_length_equals_n_features(self):
        for n in [1, 3, 5]:
            approx = LinearQValueApproximator([f_const] * n)
            assert len(approx._theta) == n

    def test_lambda_reg_stored(self):
        approx = LinearQValueApproximator([f_wealth], lambda_reg=0.5)
        assert approx._lambda_reg == pytest.approx(0.5)

    def test_zero_lambda_reg_valid(self):
        LinearQValueApproximator([f_wealth], lambda_reg=0.0)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_reg"):
            LinearQValueApproximator([f_wealth], lambda_reg=-0.1)


# ===========================================================================
# LinearQValueApproximator — evaluate
# ===========================================================================


class TestLinearEvaluate:
    def test_zero_theta_returns_zero(self):
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        assert approx.evaluate(make_state(2.0), make_action(0.5)) == pytest.approx(0.0)

    def test_single_feature_known_theta(self):
        """θ = [3.0], φ = (wealth=2.0,) → Q = 6.0"""
        approx = LinearQValueApproximator([f_wealth])
        approx._theta = np.array([3.0])
        assert approx.evaluate(make_state(2.0), make_action(0.0)) == pytest.approx(6.0)

    def test_two_features_known_theta(self):
        """θ = [2.0, 1.0], φ = (wealth=3.0, alloc=0.5) → Q = 6.5"""
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        approx._theta = np.array([2.0, 1.0])
        assert approx.evaluate(make_state(3.0), make_action(0.5)) == pytest.approx(6.5)

    def test_constant_feature_bias_only(self):
        """θ = [7.0], bias feature → Q = 7.0 for any (s, a)"""
        approx = LinearQValueApproximator([f_const])
        approx._theta = np.array([7.0])
        for w in [0.5, 1.0, 100.0]:
            assert approx.evaluate(make_state(w), make_action(0.0)) == pytest.approx(7.0)

    def test_returns_python_float(self):
        approx = LinearQValueApproximator([f_wealth])
        result = approx.evaluate(make_state(), make_action())
        assert isinstance(result, float)

    def test_negative_theta_gives_negative_q(self):
        """θ = [-1], wealth = 3.0 → Q = -3.0 (sign is correctly propagated)."""
        approx = LinearQValueApproximator([f_wealth])
        approx._theta = np.array([-1.0])
        assert approx.evaluate(make_state(3.0), make_action(0.0)) == pytest.approx(-3.0)
    
# ===========================================================================
# LinearQValueApproximator — update (OLS)
# ===========================================================================


class TestLinearUpdate:
    def test_update_returns_new_instance(self):
        approx = LinearQValueApproximator([f_wealth])
        new = approx.update([make_sample(wealth=1.0, target=2.0)])
        assert new is not approx

    def test_update_is_qvalue_approximator_subclass(self):
        approx = LinearQValueApproximator([f_wealth])
        new = approx.update([make_sample()])
        assert isinstance(new, QValueApproximator)

    def test_update_does_not_mutate_original_theta(self):
        approx = LinearQValueApproximator([f_wealth])
        theta_before = approx._theta.copy()
        approx.update([make_sample(wealth=1.0, target=99.0)])
        np.testing.assert_array_equal(approx._theta, theta_before)

    def test_empty_samples_returns_copy_with_same_theta(self):
        approx = LinearQValueApproximator([f_wealth])
        approx._theta = np.array([5.0])
        new = approx.update([])
        assert new is not approx
        np.testing.assert_array_equal(new._theta, approx._theta)

    # ── Analytic OLS verification ─────────────────────────────────────────

    def test_single_feature_ols_exact(self):
        """
        Feature = wealth, 1 sample: wealth=2, target=6.
        OLS solution: θ = [6/2] = [3.0].
        """
        approx = LinearQValueApproximator([f_wealth])
        new = approx.update([make_sample(wealth=2.0, alloc=0.0, target=6.0)])
        assert new._theta[0] == pytest.approx(3.0, rel=1e-8)

    def test_two_feature_ols_recovers_true_theta(self):
        """
        Features: wealth, action_alloc.
        True θ = [2, 1].  Three linearly independent samples produce an
        exactly determined system:
          wealth=1, alloc=0 → target=2   : 1*2 + 0*1 = 2  ✓
          wealth=2, alloc=1 → target=5   : 2*2 + 1*1 = 5  ✓
          wealth=1, alloc=1 → target=3   : 1*2 + 1*1 = 3  ✓
        """
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        samples = [
            make_sample(wealth=1.0, alloc=0.0, target=2.0),
            make_sample(wealth=2.0, alloc=1.0, target=5.0),
            make_sample(wealth=1.0, alloc=1.0, target=3.0),
        ]
        new = approx.update(samples)
        np.testing.assert_allclose(new._theta, [2.0, 1.0], atol=1e-9)

    def test_ols_predictions_match_targets_on_predictting_data(self):
        """After exact OLS predict the predictions must reproduce training targets."""
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        samples = [
            make_sample(wealth=1.0, alloc=0.0, target=2.0),
            make_sample(wealth=2.0, alloc=1.0, target=5.0),
            make_sample(wealth=1.0, alloc=1.0, target=3.0),
        ]
        new = approx.update(samples)
        for s, a, t in samples:
            assert new.evaluate(s, a) == pytest.approx(t, abs=1e-9)

    def test_constant_target_ols(self):
        """
        With feature = bias (const=1), any constant target y → θ = [y].
        """
        approx = LinearQValueApproximator([f_const])
        samples = [make_sample(target=7.5) for _ in range(10)]
        new = approx.update(samples)
        assert new._theta[0] == pytest.approx(7.5, rel=1e-8)

    def test_overdetermined_system_minimises_residuals(self):
        """
        N > D samples from a noisy linear model.
        OLS must reduce MSE vs the initial zero-theta prediction.
        """
        rng = np.random.default_rng(0)
        true_theta = np.array([2.0, -1.0])
        approx = LinearQValueApproximator([f_wealth, f_alloc])

        samples = []
        for _ in range(50):
            w = float(rng.uniform(0.5, 2.0))
            al = float(rng.uniform(0.0, 1.0))
            t = float(true_theta[0] * w + true_theta[1] * al + rng.normal(0, 0.01))
            samples.append(make_sample(wealth=w, alloc=al, target=t))

        new = approx.update(samples)
        mse_before = np.mean(
            [(approx.evaluate(s, a) - t) ** 2 for s, a, t in samples]
        )
        mse_after = np.mean(
            [(new.evaluate(s, a) - t) ** 2 for s, a, t in samples]
        )
        assert mse_after < mse_before

    def test_update_chaining_uses_latest_theta(self):
        """
        Successive update() calls must each refpredict from scratch using the
        new data, not accumulate previous predicts.
        """
        approx = LinearQValueApproximator([f_wealth])
        first = approx.update([make_sample(wealth=1.0, target=4.0)])   # θ = [4]
        second = first.update([make_sample(wealth=2.0, target=2.0)])   # θ = [1]
        assert second._theta[0] == pytest.approx(1.0, rel=1e-8)


# ===========================================================================
# LinearQValueApproximator — update (Ridge)
# ===========================================================================


class TestLinearRidge:
    def test_ridge_returns_smaller_theta_norm(self):
        """
        Ridge regularisation must shrink ||θ|| toward zero compared with OLS
        on the same data.
        """
        samples = [
            make_sample(wealth=1.0, alloc=0.0, target=2.0),
            make_sample(wealth=2.0, alloc=1.0, target=5.0),
            make_sample(wealth=1.0, alloc=1.0, target=3.0),
        ]
        ols = LinearQValueApproximator([f_wealth, f_alloc], lambda_reg=0.0)
        ridge = LinearQValueApproximator([f_wealth, f_alloc], lambda_reg=10.0)
        new_ols = ols.update(samples)
        new_ridge = ridge.update(samples)
        assert np.linalg.norm(new_ridge._theta) < np.linalg.norm(new_ols._theta)

    def test_high_lambda_pushes_theta_toward_zero(self):
        """Very large λ → θ ≈ 0."""
        samples = [make_sample(wealth=2.0, target=4.0) for _ in range(5)]
        approx = LinearQValueApproximator([f_wealth], lambda_reg=1e8)
        new = approx.update(samples)
        assert abs(new._theta[0]) < 1e-4

    def test_ridge_does_not_mutate_original(self):
        approx = LinearQValueApproximator([f_wealth], lambda_reg=1.0)
        original_theta = approx._theta.copy()
        approx.update([make_sample(wealth=1.0, target=5.0)])
        np.testing.assert_array_equal(approx._theta, original_theta)


# ===========================================================================
# LinearQValueApproximator — copy
# ===========================================================================


class TestLinearCopy:
    def test_copy_returns_different_object(self):
        approx = LinearQValueApproximator([f_wealth])
        assert approx.copy() is not approx

    def test_copy_theta_values_equal(self):
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        approx._theta = np.array([3.0, -1.5])
        clone = approx.copy()
        np.testing.assert_array_equal(clone._theta, approx._theta)

    def test_copy_theta_is_independent(self):
        """Mutating the copy's theta must not affect the original."""
        approx = LinearQValueApproximator([f_wealth])
        approx._theta = np.array([2.0])
        clone = approx.copy()
        clone._theta[0] = 999.0
        assert approx._theta[0] == pytest.approx(2.0)

    def test_copy_preserves_lambda_reg(self):
        approx = LinearQValueApproximator([f_wealth], lambda_reg=0.7)
        assert approx.copy()._lambda_reg == pytest.approx(0.7)

    def test_copy_preserves_features(self):
        feats = [f_wealth, f_const]
        approx = LinearQValueApproximator(feats)
        assert approx.copy()._features == feats


# ===========================================================================
# LinearQValueApproximator — theta property
# ===========================================================================


class TestLinearThetaProperty:
    def test_theta_returns_correct_values(self):
        approx = LinearQValueApproximator([f_wealth, f_alloc])
        approx._theta = np.array([1.0, 2.0])
        np.testing.assert_array_equal(approx.theta, [1.0, 2.0])

    def test_theta_returns_copy_not_view(self):
        approx = LinearQValueApproximator([f_wealth])
        approx._theta = np.array([5.0])
        t = approx.theta
        t[0] = 999.0
        assert approx._theta[0] == pytest.approx(5.0)


# ===========================================================================
# LinearQValueApproximator — repr
# ===========================================================================


class TestLinearRepr:
    def test_repr_contains_class_name(self):
        assert "LinearQValueApproximator" in repr(LinearQValueApproximator([f_wealth]))

    def test_repr_contains_n_features(self):
        approx = LinearQValueApproximator([f_wealth, f_alloc, f_const])
        assert "3" in repr(approx)

    def test_repr_contains_lambda_reg(self):
        approx = LinearQValueApproximator([f_wealth], lambda_reg=0.25)
        assert "0.25" in repr(approx)


# ===========================================================================
# DNNQValueApproximator — constructor
# ===========================================================================


class TestDNNConstructor:
    def test_valid_construction(self):
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), learning_rate=0.01
        )
        assert approx is not None

    def test_zero_lr_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DNNQValueApproximator([f_wealth], small_spec(), learning_rate=0.0)

    def test_negative_lr_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DNNQValueApproximator([f_wealth], small_spec(), learning_rate=-0.1)

    def test_zero_epochs_raises(self):
        with pytest.raises(ValueError, match="n_epochs"):
            DNNQValueApproximator([f_wealth], small_spec(), n_epochs=0)

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            DNNQValueApproximator([f_wealth], small_spec(), batch_size=0)

    def test_layer_sizes_built_correctly(self):
        """_layer_sizes = [n_feat] + hidden_sizes + [1]."""
        approx = DNNQValueApproximator(
            [f_wealth, f_alloc],
            DNNSpec(layer_sizes=[16, 8]),
        )
        assert approx._layer_sizes == [2, 16, 8, 1]

    def test_weights_none_before_update(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert approx._weights is None

    def test_is_fitted_false_before_update(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert not approx.is_fitted

    def test_custom_rng_stored(self):
        rng = np.random.default_rng(42)
        approx = DNNQValueApproximator([f_wealth], small_spec(), rng=rng)
        assert approx._rng is rng


# ===========================================================================
# DNNQValueApproximator — evaluate (unpredictted)
# ===========================================================================


class TestDNNEvaluateUnpredictted:
    def test_returns_zero_before_update(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        result = approx.evaluate(make_state(), make_action())
        assert result == pytest.approx(0.0)

    def test_returns_float_type(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert isinstance(approx.evaluate(make_state(), make_action()), float)


# ===========================================================================
# DNNQValueApproximator — update (structural)
# ===========================================================================


class TestDNNUpdateStructural:
    def test_update_returns_new_instance(self):
        approx = DNNQValueApproximator([f_wealth], small_spec(), rng=np.random.default_rng(0))
        new = approx.update([make_sample()])
        assert new is not approx

    def test_update_returns_qvalue_approximator_subclass(self):
        approx = DNNQValueApproximator([f_wealth], small_spec(), rng=np.random.default_rng(0))
        new = approx.update([make_sample()])
        assert isinstance(new, QValueApproximator)

    def test_update_sets_is_fitted_true(self):
        approx = DNNQValueApproximator([f_wealth], small_spec(), rng=np.random.default_rng(0))
        new = approx.update([make_sample()])
        assert new.is_fitted

    def test_update_does_not_mutate_original_is_fitted(self):
        approx = DNNQValueApproximator([f_wealth], small_spec(), rng=np.random.default_rng(0))
        approx.update([make_sample()])
        assert not approx.is_fitted

    def test_empty_samples_returns_unpredictted_copy(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        new = approx.update([])
        assert new is not approx
        assert not new.is_fitted

    def test_empty_samples_predictted_copy_keeps_weights(self):
        """Calling update([]) on an already-predictted approx preserves weights."""
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), rng=np.random.default_rng(0)
        )
        predictted = approx.update([make_sample()])
        weights_before = [w.copy() for w in predictted._weights.weights]
        still_predictted = predictted.update([])
        for wb, wa in zip(weights_before, still_predictted._weights.weights):
            np.testing.assert_array_equal(wb, wa)

    def test_update_does_not_mutate_original_weights(self):
        """
        After predictting, calling update() again on the predictted instance must
        not change its stored weights.
        """
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), rng=np.random.default_rng(1)
        )
        predictted = approx.update([make_sample(wealth=1.0, target=2.0)])
        weights_snapshot = [w.copy() for w in predictted._weights.weights]

        # Perform a second update — must not affect predictted
        predictted.update([make_sample(wealth=2.0, target=4.0)])

        for snap, current in zip(weights_snapshot, predictted._weights.weights):
            np.testing.assert_array_equal(snap, current)

    def test_evaluate_returns_finite_float_after_update(self):
        approx = DNNQValueApproximator(
            [f_wealth, f_alloc], small_spec(), rng=np.random.default_rng(2)
        )
        predictted = approx.update([make_sample(wealth=1.0, alloc=0.1, target=3.0)])
        result = predictted.evaluate(make_state(1.0, 0.1), make_action(0.1))
        assert math.isfinite(result)
        assert isinstance(result, float)


# ===========================================================================
# DNNQValueApproximator — convergence sanity check
# ===========================================================================


class TestDNNConvergence:
    """
    The network must reduce MSE toward a constant target after sufficient
    training.  We use a very simple learnable function and generous
    hyperparameters to make this robust across platforms.
    """

    def _mse(
        self,
        approx: DNNQValueApproximator,
        samples: List,
    ) -> float:
        return float(
            np.mean([(approx.evaluate(s, a) - t) ** 2 for s, a, t in samples])
        )

    def test_constant_target_mse_decreases(self):
        """
        Target = 5.0 for all samples.  After training, MSE must be
        significantly smaller than before (which is 25.0 = 5² from zero init).
        """
        samples = [make_sample(wealth=float(i), target=5.0) for i in range(1, 11)]
        approx = DNNQValueApproximator(
            [f_wealth],
            DNNSpec(layer_sizes=[16]),
            learning_rate=0.05,
            n_epochs=500,
            batch_size=10,
            rng=np.random.default_rng(0),
        )
        predictted = approx.update(samples)
        mse_before = self._mse(approx, samples)   # ≈ 25.0 (zero weights)
        mse_after = self._mse(predictted, samples)
        assert mse_after < mse_before * 0.5, (
            f"Expected MSE to drop by at least 50%; "
            f"before={mse_before:.4f}, after={mse_after:.4f}"
        )

    def test_linear_target_mse_decreases(self):
        """
        Target = 2 * wealth.  DNN with sufficient capacity should predict this.
        """
        samples = [
            make_sample(wealth=float(i) * 0.5, target=float(i) * 1.0)
            for i in range(1, 9)
        ]
        approx = DNNQValueApproximator(
            [f_wealth],
            DNNSpec(layer_sizes=[32, 16]),
            learning_rate=0.02,
            n_epochs=500,
            batch_size=8,
            rng=np.random.default_rng(7),
        )
        predictted = approx.update(samples)
        mse_before = self._mse(approx, samples)
        mse_after = self._mse(predictted, samples)
        assert mse_after < mse_before

    def test_shallow_spec_linear_output(self):
        """
        No hidden layers + linear output = linear model.
        Should converge on linear data.
        """
        samples = [
            make_sample(wealth=float(w), alloc=0.0, target=3.0 * float(w))
            for w in range(1, 6)
        ]
        approx = DNNQValueApproximator(
            [f_wealth],
            shallow_spec(),
            learning_rate=0.05,
            n_epochs=1000,
            batch_size=5,
            rng=np.random.default_rng(3),
        )
        predictted = approx.update(samples)
        mse_after = self._mse(predictted, samples)
        assert mse_after < 0.5, f"Expected near-zero MSE; got {mse_after:.4f}"


# ===========================================================================
# DNNQValueApproximator — copy
# ===========================================================================


class TestDNNCopy:
    def test_copy_returns_different_object(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert approx.copy() is not approx

    def test_copy_before_update_is_not_predictted(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert not approx.copy().is_fitted

    def test_copy_after_update_is_fitted(self):
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), rng=np.random.default_rng(0)
        )
        predictted = approx.update([make_sample()])
        assert predictted.copy().is_fitted

    def test_copy_weights_are_numerically_equal(self):
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), rng=np.random.default_rng(0)
        )
        predictted = approx.update([make_sample(wealth=1.0, target=2.0)])
        clone = predictted.copy()
        for W_orig, W_copy in zip(
            predictted._weights.weights, clone._weights.weights
        ):
            np.testing.assert_array_equal(W_orig, W_copy)

    def test_copy_weights_are_independent(self):
        """Mutating clone weights must not affect original."""
        approx = DNNQValueApproximator(
            [f_wealth], small_spec(), rng=np.random.default_rng(0)
        )
        predictted = approx.update([make_sample()])
        clone = predictted.copy()
        # Zero out all clone weights
        for W in clone._weights.weights:
            W[:] = 0.0
        # Original must be unchanged
        for W_orig in predictted._weights.weights:
            assert not np.all(W_orig == 0.0), (
                "Original weights were modified when clone weights were zeroed"
            )

    def test_copy_preserves_hyperparameters(self):
        approx = DNNQValueApproximator(
            [f_wealth],
            small_spec(),
            learning_rate=0.03,
            n_epochs=200,
            batch_size=16,
        )
        clone = approx.copy()
        assert clone._lr == pytest.approx(0.03)
        assert clone._n_epochs == 200
        assert clone._batch_size == 16


# ===========================================================================
# DNNQValueApproximator — activation functions
# ===========================================================================


class TestDNNActivations:
    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid"])
    def test_all_activations_produce_finite_output(self, act: str):
        spec = DNNSpec(layer_sizes=[8], activations=[act])
        approx = DNNQValueApproximator(
            [f_wealth, f_alloc],
            spec,
            learning_rate=0.01,
            n_epochs=10,
            rng=np.random.default_rng(0),
        )
        predictted = approx.update([
            make_sample(wealth=1.0, alloc=0.1, target=2.0),
            make_sample(wealth=2.0, alloc=0.2, target=4.0),
        ])
        result = predictted.evaluate(make_state(1.5, 0.15), make_action(0.15))
        assert math.isfinite(result)

    def test_deep_network_runs_without_error(self):
        approx = DNNQValueApproximator(
            [f_wealth, f_alloc],
            deep_spec(),
            learning_rate=0.005,
            n_epochs=10,
            rng=np.random.default_rng(0),
        )
        samples = [make_sample(wealth=float(i), alloc=0.1, target=float(i))
                   for i in range(1, 6)]
        predictted = approx.update(samples)
        for s, a, _ in samples:
            assert math.isfinite(predictted.evaluate(s, a))

    def test_multi_feature_multi_layer(self):
        """Smoke test: no exceptions with 3 features and 3 hidden layers."""
        feats = [f_wealth, f_alloc, f_const]
        spec = DNNSpec(layer_sizes=[16, 8, 4], activations=["relu", "tanh", "sigmoid"])
        approx = DNNQValueApproximator(feats, spec, learning_rate=0.01, n_epochs=5,
                                        rng=np.random.default_rng(99))
        samples = [make_sample(wealth=float(i), alloc=0.1, target=float(i) * 2)
                   for i in range(1, 6)]
        predictted = approx.update(samples)
        for s, a, _ in samples:
            assert math.isfinite(predictted.evaluate(s, a))


# ===========================================================================
# DNNQValueApproximator — repr
# ===========================================================================


class TestDNNRepr:
    def test_repr_contains_class_name(self):
        approx = DNNQValueApproximator([f_wealth], small_spec())
        assert "DNNQValueApproximator" in repr(approx)

    def test_repr_contains_n_features(self):
        approx = DNNQValueApproximator([f_wealth, f_alloc], small_spec())
        assert "2" in repr(approx)

    def test_repr_contains_layer_sizes(self):
        spec = DNNSpec(layer_sizes=[32, 16])
        approx = DNNQValueApproximator([f_wealth], spec)
        r = repr(approx)
        assert "32" in r or "16" in r

    def test_repr_contains_lr(self):
        approx = DNNQValueApproximator([f_wealth], small_spec(), learning_rate=0.007)
        assert "0.007" in repr(approx)


# ===========================================================================
# Shared ABC contract — both approximators
# ===========================================================================


class TestQValueApproximatorContract:
    """
    Verify that both concrete classes honour the QValueApproximator ABC
    and behave identically on structural invariants.
    """

    @pytest.fixture(
        params=["linear", "dnn"],
        ids=["LinearQVF", "DNNQVF"],
    )
    def fresh_approx(self, request: pytest.FixtureRequest) -> QValueApproximator:
        if request.param == "linear":
            return LinearQValueApproximator([f_wealth, f_alloc])
        rng = np.random.default_rng(0)
        return DNNQValueApproximator(
            [f_wealth, f_alloc],
            DNNSpec(layer_sizes=[8]),
            learning_rate=0.01,
            n_epochs=5,
            rng=rng,
        )

    def test_is_qvalue_approximator_instance(self, fresh_approx):
        assert isinstance(fresh_approx, QValueApproximator)

    def test_update_returns_qvalue_approximator(self, fresh_approx):
        result = fresh_approx.update([make_sample(wealth=1.0, target=2.0)])
        assert isinstance(result, QValueApproximator)

    def test_update_returns_new_object(self, fresh_approx):
        result = fresh_approx.update([make_sample()])
        assert result is not fresh_approx

    def test_copy_returns_qvalue_approximator(self, fresh_approx):
        assert isinstance(fresh_approx.copy(), QValueApproximator)

    def test_copy_returns_new_object(self, fresh_approx):
        assert fresh_approx.copy() is not fresh_approx

    def test_evaluate_returns_python_float(self, fresh_approx):
        result = fresh_approx.evaluate(make_state(), make_action())
        assert isinstance(result, float)

    def test_evaluate_is_finite(self, fresh_approx):
        result = fresh_approx.evaluate(make_state(1.0), make_action(0.1))
        assert math.isfinite(result)