# test/unit/test_returns.py
from __future__ import annotations
import sys
import math
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from returns import (
    ConstantRisklessReturn,
    StepwiseRisklessReturn,
    NormalReturnDistribution,
    MultivariateNormalReturnDistribution,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ===========================================================================
# ConstantRisklessReturn
# ===========================================================================


class TestConstantRisklessReturnConstructor:
    def test_valid_positive_rate(self):
        m = ConstantRisklessReturn(rate=0.05)
        assert m.rate == 0.05

    def test_valid_zero_rate(self):
        m = ConstantRisklessReturn(rate=0.0)
        assert m.rate == 0.0

    def test_valid_negative_rate_above_minus_one(self):
        """A slightly negative rate is economically unusual but not invalid."""
        m = ConstantRisklessReturn(rate=-0.5)
        assert m.rate == -0.5

    def test_rate_exactly_minus_one_raises(self):
        with pytest.raises(ValueError, match=r"rate"):
            ConstantRisklessReturn(rate=-1.0)

    def test_rate_below_minus_one_raises(self):
        with pytest.raises(ValueError, match=r"rate"):
            ConstantRisklessReturn(rate=-2.0)

    def test_is_frozen_dataclass(self):
        """Mutation of a frozen dataclass must raise."""
        m = ConstantRisklessReturn(rate=0.03)
        with pytest.raises((AttributeError, TypeError)):
            m.rate = 0.04  # type: ignore[misc]


class TestConstantRisklessReturnGetRate:
    @pytest.fixture
    def model(self):
        return ConstantRisklessReturn(rate=0.02)

    def test_returns_rate_at_t_zero(self, model):
        assert model.get_rate(0) == 0.02

    def test_returns_same_rate_for_large_t(self, model):
        assert model.get_rate(1_000) == 0.02

    def test_returns_same_rate_for_all_t(self, model):
        rates = [model.get_rate(t) for t in range(50)]
        assert all(r == 0.02 for r in rates)

    def test_negative_rate_returned_correctly(self):
        m = ConstantRisklessReturn(rate=-0.01)
        assert m.get_rate(5) == -0.01


# ===========================================================================
# StepwiseRisklessReturn
# ===========================================================================


class TestStepwiseRisklessReturnConstructor:
    def test_single_rate_is_valid(self):
        m = StepwiseRisklessReturn(rates=(0.03,))
        assert m.rates == (0.03,)

    def test_multiple_rates_are_valid(self):
        m = StepwiseRisklessReturn(rates=(0.01, 0.02, 0.03))
        assert len(m.rates) == 3

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match=r"rates"):
            StepwiseRisklessReturn(rates=())

    def test_is_frozen_dataclass(self):
        m = StepwiseRisklessReturn(rates=(0.01, 0.02))
        with pytest.raises((AttributeError, TypeError)):
            m.rates = (0.05,)  # type: ignore[misc]


class TestStepwiseRisklessReturnGetRate:
    @pytest.fixture
    def model(self):
        # Three distinct rates for easy assertions
        return StepwiseRisklessReturn(rates=(0.01, 0.02, 0.03))

    def test_t_zero_returns_first_rate(self, model):
        assert math.isclose(model.get_rate(0), 0.01)

    def test_t_one_returns_second_rate(self, model):
        assert math.isclose(model.get_rate(1), 0.02)

    def test_t_last_in_bounds(self, model):
        assert math.isclose(model.get_rate(2), 0.03)

    def test_t_beyond_length_returns_last_rate(self, model):
        """Flat extrapolation: any t ≥ len(rates) returns rates[-1]."""
        for t in [3, 10, 100, 999]:
            assert math.isclose(model.get_rate(t), 0.03), (
                f"Expected 0.03 for t={t}, got {model.get_rate(t)}"
            )

    def test_single_rate_model_always_returns_that_rate(self):
        m = StepwiseRisklessReturn(rates=(0.05,))
        for t in range(10):
            assert math.isclose(m.get_rate(t), 0.05)

    def test_negative_rates_indexed_correctly(self):
        m = StepwiseRisklessReturn(rates=(-0.01, 0.00, 0.01))
        assert math.isclose(m.get_rate(0), -0.01)
        assert math.isclose(m.get_rate(1), 0.00)
        assert math.isclose(m.get_rate(2), 0.01)


# ===========================================================================
# NormalReturnDistribution
# ===========================================================================


class TestNormalReturnDistributionConstructor:
    def test_valid_positive_sigma(self):
        d = NormalReturnDistribution(mu=0.08, sigma=0.20)
        assert d.mean == 0.08
        assert d.variance == pytest.approx(0.04)

    def test_valid_zero_sigma(self):
        d = NormalReturnDistribution(mu=0.05, sigma=0.0)
        assert d.variance == 0.0

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match=r"sigma"):
            NormalReturnDistribution(mu=0.0, sigma=-0.1)

    def test_negative_mu_is_valid(self):
        """Negative expected return is financially meaningful."""
        d = NormalReturnDistribution(mu=-0.05, sigma=0.10)
        assert d.mean == -0.05


class TestNormalReturnDistributionProperties:
    def test_mean_property(self):
        assert NormalReturnDistribution(mu=0.12, sigma=0.15).mean == 0.12

    def test_variance_property_nonzero_sigma(self):
        d = NormalReturnDistribution(mu=0.0, sigma=0.25)
        assert math.isclose(d.variance, 0.0625, rel_tol=1e-12)

    def test_variance_zero_when_sigma_zero(self):
        d = NormalReturnDistribution(mu=0.1, sigma=0.0)
        assert d.variance == 0.0


class TestNormalReturnDistributionSample:
    def test_sample_returns_float(self):
        d = NormalReturnDistribution(mu=0.05, sigma=0.10)
        result = d.sample(t=0, rng=make_rng())
        assert isinstance(result, float)

    def test_sample_deterministic_when_sigma_zero(self):
        """Degenerate distribution: every draw must equal mu."""
        d = NormalReturnDistribution(mu=0.07, sigma=0.0)
        for t in range(10):
            assert math.isclose(d.sample(t=t, rng=make_rng(t)), 0.07)

    def test_sample_seeded_reproducibility(self):
        d = NormalReturnDistribution(mu=0.08, sigma=0.20)
        r1 = d.sample(t=0, rng=make_rng(7))
        r2 = d.sample(t=0, rng=make_rng(7))
        assert r1 == r2

    def test_sample_different_seeds_differ(self):
        d = NormalReturnDistribution(mu=0.08, sigma=0.20)
        r1 = d.sample(t=0, rng=make_rng(1))
        r2 = d.sample(t=0, rng=make_rng(2))
        assert r1 != r2

    def test_sample_statistical_mean(self):
        """
        Draw N samples and verify the empirical mean is within 3σ/√N of μ.
        The bound is deterministic given the fixed seed.
        """
        N = 50_000
        mu, sigma = 0.08, 0.20
        d = NormalReturnDistribution(mu=mu, sigma=sigma)
        rng = make_rng(0)
        draws = [d.sample(t=i, rng=rng) for i in range(N)]
        empirical_mean = sum(draws) / N
        tol = 4 * sigma / math.sqrt(N)          # very conservative 4-sigma bound
        assert abs(empirical_mean - mu) < tol, (
            f"Empirical mean {empirical_mean:.6f} too far from μ={mu}"
        )

    def test_sample_statistical_variance(self):
        """Empirical variance should be close to σ²."""
        N = 50_000
        mu, sigma = 0.0, 0.30
        d = NormalReturnDistribution(mu=mu, sigma=sigma)
        rng = make_rng(1)
        draws = [d.sample(t=i, rng=rng) for i in range(N)]
        empirical_var = sum((x - mu) ** 2 for x in draws) / N
        assert abs(empirical_var - sigma ** 2) < 0.01, (
            f"Empirical variance {empirical_var:.6f} too far from σ²={sigma**2}"
        )

    def test_sample_t_argument_does_not_affect_iid_distribution(self):
        """
        NormalReturnDistribution is i.i.d.: t should not shift the distribution.
        Verify by comparing means of samples drawn at t=0 vs t=999.
        """
        d = NormalReturnDistribution(mu=0.05, sigma=0.10)
        rng_a = make_rng(42)
        rng_b = make_rng(42)
        s0 = d.sample(t=0, rng=rng_a)
        s999 = d.sample(t=999, rng=rng_b)
        # Same RNG seed → same draw regardless of t
        assert s0 == s999

    def test_sample_default_rng_does_not_crash(self):
        """Calling sample() without providing an RNG must not raise."""
        d = NormalReturnDistribution(mu=0.05, sigma=0.10)
        result = d.sample(t=0)
        assert isinstance(result, float)


class TestNormalReturnDistributionRepr:
    def test_repr_contains_class_name(self):
        r = repr(NormalReturnDistribution(mu=0.08, sigma=0.20))
        assert "NormalReturnDistribution" in r

    def test_repr_contains_mu_and_sigma(self):
        r = repr(NormalReturnDistribution(mu=0.08, sigma=0.20))
        assert "0.08" in r
        assert "0.2" in r


# ===========================================================================
# MultivariateNormalReturnDistribution
# ===========================================================================


# ---- Shared fixtures / helpers ------------------------------------------- #

def diagonal_cov(variances: Tuple[float, ...]) -> np.ndarray:
    """Build a diagonal covariance matrix from a tuple of variances."""
    return np.diag(np.array(variances, dtype=float))


SIMPLE_MUS = (0.05, 0.10)
SIMPLE_COV = diagonal_cov((0.04, 0.09))          # σ₁=0.2, σ₂=0.3


class TestMultivariateNormalConstructorValid:
    def test_single_asset(self):
        d = MultivariateNormalReturnDistribution(
            mus=[0.05], cov=np.array([[0.04]])
        )
        assert d.n_assets == 1

    def test_two_assets_diagonal(self):
        d = MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )
        assert d.n_assets == 2

    def test_three_assets_correlated(self):
        mus = [0.05, 0.08, 0.12]
        cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ])
        d = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
        assert d.n_assets == 3

    def test_zero_covariance_matrix_allowed(self):
        """Σ = 0 is PSD (degenerate), so it must be accepted."""
        d = MultivariateNormalReturnDistribution(
            mus=[0.05, 0.10], cov=np.zeros((2, 2))
        )
        assert d.n_assets == 2


class TestMultivariateNormalConstructorInvalid:
    def test_empty_mus_raises(self):
        with pytest.raises(ValueError, match=r"asset"):
            MultivariateNormalReturnDistribution(mus=[], cov=np.zeros((0, 0)))

    def test_cov_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape"):
            MultivariateNormalReturnDistribution(
                mus=[0.05, 0.10], cov=np.eye(3)      # 3×3 for 2 assets
            )

    def test_cov_non_square_raises(self):
        with pytest.raises(ValueError, match=r"shape"):
            MultivariateNormalReturnDistribution(
                mus=[0.05, 0.10], cov=np.ones((2, 3))
            )

    def test_asymmetric_cov_raises(self):
        cov = np.array([[0.04, 0.01],
                        [0.02, 0.09]])               # off-diagonals differ
        with pytest.raises(ValueError, match=r"symmetric"):
            MultivariateNormalReturnDistribution(mus=[0.05, 0.10], cov=cov)

    def test_negative_definite_cov_raises(self):
        """A negative-definite matrix is not PSD."""
        cov = np.array([[-0.04, 0.00],
                        [0.00, -0.09]])
        with pytest.raises(ValueError, match=r"[Pp]ositive"):
            MultivariateNormalReturnDistribution(mus=[0.05, 0.10], cov=cov)

    def test_negative_eigenvalue_cov_raises(self):
        """Symmetric matrix with one negative eigenvalue must be rejected."""
        # Construct: eigenvalues 1 and -0.5
        Q = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        cov = Q @ np.diag([1.0, -0.5]) @ Q.T
        with pytest.raises(ValueError, match=r"[Pp]ositive"):
            MultivariateNormalReturnDistribution(mus=[0.0, 0.0], cov=cov)


class TestMultivariateNormalProperties:
    @pytest.fixture
    def dist(self):
        return MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )

    def test_n_assets(self, dist):
        assert dist.n_assets == 2

    def test_means_tuple(self, dist):
        assert dist.means == SIMPLE_MUS

    def test_means_is_tuple_type(self, dist):
        assert isinstance(dist.means, tuple)

    def test_covariance_matrix_values(self, dist):
        np.testing.assert_array_almost_equal(dist.covariance_matrix, SIMPLE_COV)

    def test_covariance_matrix_is_copy(self, dist):
        """Mutating the returned matrix must not affect internal state."""
        cov_out = dist.covariance_matrix
        cov_out[0, 0] = 999.0
        np.testing.assert_array_almost_equal(
            dist.covariance_matrix, SIMPLE_COV,
            err_msg="covariance_matrix should return a defensive copy",
        )

    def test_covariance_matrix_is_ndarray(self, dist):
        assert isinstance(dist.covariance_matrix, np.ndarray)

    def test_diagonal_entries_match_individual_variances(self, dist):
        cov = dist.covariance_matrix
        assert math.isclose(cov[0, 0], 0.04, rel_tol=1e-12)
        assert math.isclose(cov[1, 1], 0.09, rel_tol=1e-12)


class TestMultivariateNormalSample:
    @pytest.fixture
    def dist(self):
        return MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )

    def test_sample_returns_tuple(self, dist):
        result = dist.sample(t=0, rng=make_rng())
        assert isinstance(result, tuple)

    def test_sample_length_equals_n_assets(self, dist):
        result = dist.sample(t=0, rng=make_rng())
        assert len(result) == 2

    def test_sample_elements_are_floats(self, dist):
        result = dist.sample(t=0, rng=make_rng())
        assert all(isinstance(x, float) for x in result)

    def test_sample_seeded_reproducibility(self, dist):
        r1 = dist.sample(t=0, rng=make_rng(7))
        r2 = dist.sample(t=0, rng=make_rng(7))
        assert r1 == r2

    def test_sample_different_seeds_differ(self, dist):
        r1 = dist.sample(t=0, rng=make_rng(1))
        r2 = dist.sample(t=0, rng=make_rng(2))
        assert r1 != r2

    def test_sample_default_rng_does_not_crash(self, dist):
        result = dist.sample(t=0)
        assert len(result) == 2

    def test_sample_deterministic_zero_covariance(self):
        """With Σ=0 every draw must equal μ exactly."""
        mus = (0.05, 0.10)
        d = MultivariateNormalReturnDistribution(
            mus=mus, cov=np.zeros((2, 2))
        )
        for seed in range(5):
            draw = d.sample(t=0, rng=make_rng(seed))
            assert math.isclose(draw[0], 0.05, abs_tol=1e-10), (
                f"seed={seed}: draw[0]={draw[0]}"
            )
            assert math.isclose(draw[1], 0.10, abs_tol=1e-10), (
                f"seed={seed}: draw[1]={draw[1]}"
            )

    def test_sample_statistical_means(self):
        """Empirical means of N draws should be close to μ."""
        N = 60_000
        mus = (0.05, 0.10)
        cov = diagonal_cov((0.04, 0.09))
        d = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
        rng = make_rng(99)
        draws = [d.sample(t=i, rng=rng) for i in range(N)]
        emp_mean_0 = sum(x[0] for x in draws) / N
        emp_mean_1 = sum(x[1] for x in draws) / N
        assert abs(emp_mean_0 - mus[0]) < 4 * math.sqrt(0.04 / N), (
            f"Asset 0: empirical mean {emp_mean_0:.5f} too far from {mus[0]}"
        )
        assert abs(emp_mean_1 - mus[1]) < 4 * math.sqrt(0.09 / N), (
            f"Asset 1: empirical mean {emp_mean_1:.5f} too far from {mus[1]}"
        )

    def test_sample_statistical_variances(self):
        """Empirical variances of N draws should be close to diagonal entries."""
        N = 60_000
        mus = (0.0, 0.0)
        variances = (0.04, 0.16)
        cov = diagonal_cov(variances)
        d = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
        rng = make_rng(77)
        draws = np.array([d.sample(t=i, rng=rng) for i in range(N)])
        emp_var_0 = float(np.var(draws[:, 0]))
        emp_var_1 = float(np.var(draws[:, 1]))
        assert abs(emp_var_0 - variances[0]) < 0.01, (
            f"Asset 0: empirical variance {emp_var_0:.4f} vs {variances[0]}"
        )
        assert abs(emp_var_1 - variances[1]) < 0.02, (
            f"Asset 1: empirical variance {emp_var_1:.4f} vs {variances[1]}"
        )

    def test_sample_correlated_assets_covariance(self):
        """
        For two correlated assets draw N samples and verify the off-diagonal
        empirical covariance is close to the true value.
        """
        N = 80_000
        mus = [0.0, 0.0]
        cov_true = np.array([[0.04, 0.03],
                             [0.03, 0.09]])
        d = MultivariateNormalReturnDistribution(mus=mus, cov=cov_true)
        rng = make_rng(55)
        draws = np.array([d.sample(t=i, rng=rng) for i in range(N)])
        emp_cov = np.cov(draws.T, ddof=1)
        # Off-diagonal covariance check
        assert abs(emp_cov[0, 1] - 0.03) < 0.01, (
            f"Empirical covariance {emp_cov[0,1]:.4f} too far from 0.03"
        )

    def test_sample_t_does_not_affect_iid_distribution(self):
        """
        t is passed through but must not shift the draw for an i.i.d. model.
        Two calls with the same seeded RNG but different t should be equal.
        """
        d = MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )
        r_t0 = d.sample(t=0, rng=make_rng(3))
        r_t99 = d.sample(t=99, rng=make_rng(3))
        assert r_t0 == r_t99


class TestMultivariateNormalRepr:
    def test_repr_contains_class_name(self):
        d = MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )
        assert "MultivariateNormalReturnDistribution" in repr(d)

    def test_repr_contains_n_assets(self):
        d = MultivariateNormalReturnDistribution(
            mus=SIMPLE_MUS, cov=SIMPLE_COV
        )
        assert "2" in repr(d)