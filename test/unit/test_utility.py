from __future__ import annotations
import sys
import math
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utility import CRRAUtility, ExponentialUtility, LogUtility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEALTH_SAMPLES = [0.01, 0.5, 1.0, 2.0, 10.0, 100.0]


# ===========================================================================
# CRRAUtility
# ===========================================================================


class TestCRRAUtilityConstructor:
    def test_valid_gamma_zero(self):
        u = CRRAUtility(gamma=0.0)
        assert u.gamma == 0.0

    def test_valid_gamma_one(self):
        u = CRRAUtility(gamma=1.0)
        assert math.isclose(u.gamma, 1.0)

    def test_valid_gamma_large(self):
        u = CRRAUtility(gamma=10.0)
        assert u.gamma == 10.0

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            CRRAUtility(gamma=-0.1)

    def test_gamma_property_is_stored(self):
        u = CRRAUtility(gamma=3.5)
        assert u.gamma == 3.5


class TestCRRAUtilityEvaluate:
    def test_gamma_one_equals_log(self):
        """CRRA with γ=1 must return ln(W)."""
        u = CRRAUtility(gamma=1.0)
        for w in WEALTH_SAMPLES:
            assert math.isclose(u.evaluate(w), math.log(w), rel_tol=1e-12)

    def test_gamma_two_formula(self):
        """U(W) = W^(1-2)/(1-2) = -1/W."""
        u = CRRAUtility(gamma=2.0)
        for w in WEALTH_SAMPLES:
            expected = (w ** (1.0 - 2.0)) / (1.0 - 2.0)  # = -1/w
            assert math.isclose(u.evaluate(w), expected, rel_tol=1e-12)

    def test_gamma_half_formula(self):
        """U(W) = W^0.5 / 0.5 = 2√W."""
        u = CRRAUtility(gamma=0.5)
        for w in WEALTH_SAMPLES:
            expected = (w ** 0.5) / 0.5
            assert math.isclose(u.evaluate(w), expected, rel_tol=1e-12)

    def test_gamma_zero_is_linear(self):
        """U(W) = W^1 / 1 = W (risk-neutral linear utility)."""
        u = CRRAUtility(gamma=0.0)
        for w in WEALTH_SAMPLES:
            assert math.isclose(u.evaluate(w), w, rel_tol=1e-12)

    def test_zero_wealth_raises(self):
        u = CRRAUtility(gamma=2.0)
        with pytest.raises(ValueError, match="positive"):
            u.evaluate(0.0)

    def test_negative_wealth_raises(self):
        u = CRRAUtility(gamma=2.0)
        with pytest.raises(ValueError, match="positive"):
            u.evaluate(-1.0)

    def test_strictly_increasing(self):
        """Higher wealth should yield higher utility for all γ ≥ 0."""
        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
            u = CRRAUtility(gamma=gamma)
            vals = [u.evaluate(w) for w in WEALTH_SAMPLES]
            assert vals == sorted(vals), f"Not increasing for gamma={gamma}"


class TestCRRAUtilityMarginal:
    def test_marginal_formula(self):
        """U'(W) = W^{-γ}."""
        for gamma in [0.0, 0.5, 1.0, 2.0, 4.0]:
            u = CRRAUtility(gamma=gamma)
            for w in WEALTH_SAMPLES:
                expected = w ** (-gamma)
                assert math.isclose(u.marginal(w), expected, rel_tol=1e-12), (
                    f"gamma={gamma}, w={w}"
                )

    def test_marginal_zero_wealth_raises(self):
        u = CRRAUtility(gamma=2.0)
        with pytest.raises(ValueError, match="positive"):
            u.marginal(0.0)

    def test_marginal_negative_wealth_raises(self):
        u = CRRAUtility(gamma=2.0)
        with pytest.raises(ValueError, match="positive"):
            u.marginal(-5.0)

    def test_marginal_strictly_decreasing_for_risk_averse(self):
        """Marginal utility must decrease with wealth when γ > 0 (concavity)."""
        for gamma in [0.5, 1.0, 2.0, 5.0]:
            u = CRRAUtility(gamma=gamma)
            margs = [u.marginal(w) for w in WEALTH_SAMPLES]
            assert margs == sorted(margs, reverse=True), (
                f"Marginal not decreasing for gamma={gamma}"
            )

    def test_marginal_constant_for_gamma_zero(self):
        """γ=0 → U'(W) = 1 for all W (linear utility has constant marginal)."""
        u = CRRAUtility(gamma=0.0)
        for w in WEALTH_SAMPLES:
            assert math.isclose(u.marginal(w), 1.0, rel_tol=1e-12)


class TestCRRAUtilityRiskAversion:
    def test_is_risk_averse_positive_gamma(self):
        for gamma in [0.1, 1.0, 2.0, 10.0]:
            assert CRRAUtility(gamma=gamma).is_risk_averse is True

    def test_is_not_risk_averse_gamma_zero(self):
        """γ=0 is linear (risk-neutral); is_risk_averse should be False."""
        assert CRRAUtility(gamma=0.0).is_risk_averse is False


class TestCRRAUtilityRepr:
    def test_repr_contains_class_and_gamma(self):
        r = repr(CRRAUtility(gamma=3.0))
        assert "CRRAUtility" in r
        assert "3.0" in r


# ===========================================================================
# ExponentialUtility
# ===========================================================================


class TestExponentialUtilityConstructor:
    def test_valid_alpha(self):
        u = ExponentialUtility(alpha=1.0)
        assert u.alpha == 1.0

    def test_valid_small_alpha(self):
        u = ExponentialUtility(alpha=0.001)
        assert math.isclose(u.alpha, 0.001)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ExponentialUtility(alpha=0.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ExponentialUtility(alpha=-1.0)

    def test_alpha_property_stored(self):
        u = ExponentialUtility(alpha=2.5)
        assert u.alpha == 2.5


class TestExponentialUtilityEvaluate:
    def test_formula(self):
        """U(W) = -exp(-α·W) / α."""
        for alpha in [0.5, 1.0, 2.0]:
            u = ExponentialUtility(alpha=alpha)
            for w in [-1.0, 0.0, 0.5, 1.0, 5.0]:
                expected = -math.exp(-alpha * w) / alpha
                assert math.isclose(u.evaluate(w), expected, rel_tol=1e-12), (
                    f"alpha={alpha}, w={w}"
                )

    def test_defined_for_negative_wealth(self):
        """Unlike CRRA/Log, Exponential utility accepts negative wealth."""
        u = ExponentialUtility(alpha=1.0)
        result = u.evaluate(-10.0)
        assert math.isfinite(result)

    def test_defined_for_zero_wealth(self):
        u = ExponentialUtility(alpha=1.0)
        expected = -math.exp(0.0) / 1.0  # = -1
        assert math.isclose(u.evaluate(0.0), expected, rel_tol=1e-12)

    def test_strictly_increasing(self):
        """Higher wealth → higher utility."""
        u = ExponentialUtility(alpha=1.0)
        wealth_range = [-2.0, -0.5, 0.0, 1.0, 5.0, 20.0]
        vals = [u.evaluate(w) for w in wealth_range]
        assert vals == sorted(vals)

    def test_utility_always_negative(self):
        """U(W) = -exp(-αW)/α < 0 for all W and α > 0."""
        u = ExponentialUtility(alpha=1.0)
        for w in [-10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            assert u.evaluate(w) < 0


class TestExponentialUtilityMarginal:
    def test_formula(self):
        """U'(W) = exp(-α·W)."""
        for alpha in [0.5, 1.0, 2.0]:
            u = ExponentialUtility(alpha=alpha)
            for w in [-1.0, 0.0, 1.0, 5.0]:
                expected = math.exp(-alpha * w)
                assert math.isclose(u.marginal(w), expected, rel_tol=1e-12)

    def test_marginal_always_positive(self):
        """Strictly increasing utility ⟹ U'(W) > 0."""
        u = ExponentialUtility(alpha=1.0)
        for w in [-10.0, 0.0, 10.0]:
            assert u.marginal(w) > 0

    def test_marginal_strictly_decreasing(self):
        """Concavity ⟹ U'(W) decreasing."""
        u = ExponentialUtility(alpha=1.0)
        wealth_range = [-5.0, 0.0, 1.0, 5.0, 10.0]
        margs = [u.marginal(w) for w in wealth_range]
        assert margs == sorted(margs, reverse=True)


class TestExponentialUtilityRiskAversion:
    def test_is_risk_averse_always_true(self):
        for alpha in [0.1, 1.0, 10.0]:
            assert ExponentialUtility(alpha=alpha).is_risk_averse is True


class TestExponentialUtilityRepr:
    def test_repr_contains_class_and_alpha(self):
        r = repr(ExponentialUtility(alpha=0.5))
        assert "ExponentialUtility" in r
        assert "0.5" in r


# ===========================================================================
# LogUtility
# ===========================================================================


class TestLogUtilityEvaluate:
    def test_formula(self):
        """U(W) = ln(W)."""
        u = LogUtility()
        for w in WEALTH_SAMPLES:
            assert math.isclose(u.evaluate(w), math.log(w), rel_tol=1e-12)

    def test_zero_wealth_raises(self):
        u = LogUtility()
        with pytest.raises(ValueError, match="positive"):
            u.evaluate(0.0)

    def test_negative_wealth_raises(self):
        u = LogUtility()
        with pytest.raises(ValueError, match="positive"):
            u.evaluate(-1.0)

    def test_evaluate_at_one_is_zero(self):
        """ln(1) = 0."""
        assert math.isclose(LogUtility().evaluate(1.0), 0.0, abs_tol=1e-15)

    def test_strictly_increasing(self):
        u = LogUtility()
        vals = [u.evaluate(w) for w in WEALTH_SAMPLES]
        assert vals == sorted(vals)


class TestLogUtilityMarginal:
    def test_formula(self):
        """U'(W) = 1/W."""
        u = LogUtility()
        for w in WEALTH_SAMPLES:
            assert math.isclose(u.marginal(w), 1.0 / w, rel_tol=1e-12)

    def test_zero_wealth_raises(self):
        u = LogUtility()
        with pytest.raises(ValueError, match="positive"):
            u.marginal(0.0)

    def test_negative_wealth_raises(self):
        u = LogUtility()
        with pytest.raises(ValueError, match="positive"):
            u.marginal(-1.0)

    def test_marginal_strictly_decreasing(self):
        """1/W is strictly decreasing for W > 0."""
        u = LogUtility()
        margs = [u.marginal(w) for w in WEALTH_SAMPLES]
        assert margs == sorted(margs, reverse=True)


class TestLogUtilityRiskAversion:
    def test_is_risk_averse(self):
        assert LogUtility().is_risk_averse is True


class TestLogUtilityRepr:
    def test_repr(self):
        assert repr(LogUtility()) == "LogUtility()"


# ===========================================================================
# Cross-class consistency
# ===========================================================================


class TestCrossClassConsistency:
    """LogUtility must be numerically identical to CRRAUtility(γ=1)."""

    def test_evaluate_log_equals_crra_gamma_one(self):
        log_u = LogUtility()
        crra_u = CRRAUtility(gamma=1.0)
        for w in WEALTH_SAMPLES:
            assert math.isclose(log_u.evaluate(w), crra_u.evaluate(w), rel_tol=1e-12), (
                f"Mismatch at w={w}: LogUtility={log_u.evaluate(w)}, "
                f"CRRAUtility(1)={crra_u.evaluate(w)}"
            )

    def test_marginal_log_equals_crra_gamma_one(self):
        log_u = LogUtility()
        crra_u = CRRAUtility(gamma=1.0)
        for w in WEALTH_SAMPLES:
            assert math.isclose(log_u.marginal(w), crra_u.marginal(w), rel_tol=1e-12), (
                f"Mismatch at w={w}"
            )

    def test_all_utility_functions_are_risk_averse_except_crra_zero(self):
        """Sanity check: every implemented utility except γ=0 is risk-averse."""
        risk_averse = [
            CRRAUtility(gamma=0.5),
            CRRAUtility(gamma=1.0),
            CRRAUtility(gamma=2.0),
            ExponentialUtility(alpha=1.0),
            LogUtility(),
        ]
        for u in risk_averse:
            assert u.is_risk_averse, f"{u!r} should be risk-averse"

        assert not CRRAUtility(gamma=0.0).is_risk_averse