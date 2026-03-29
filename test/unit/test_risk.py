import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import risk
from risk import RiskAversionModel, ConstantRiskAversion, TimeVaryingRiskAversion

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

# A simple linear glide path from gamma=2 (growth) to gamma=6 (conservative).
HORIZON = 40
GLIDE_PATH = [2.0 + (6.0 - 2.0) * t / (HORIZON - 1) for t in range(HORIZON)]

WEALTH_LEVELS = [0.5, 1.0, 5.0, 10.0, 100.0]


# ===========================================================================
# Module import
# ===========================================================================


def test_import_risk():
    assert risk is not None


def test_all_classes_exported():
    for name in ["RiskAversionModel", "ConstantRiskAversion", "TimeVaryingRiskAversion"]:
        assert hasattr(risk, name), f"Missing export: {name}"


# ===========================================================================
# Abstract base class
# ===========================================================================


def test_risk_aversion_model_is_abstract():
    """RiskAversionModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        RiskAversionModel()


def test_partial_subclass_cannot_be_instantiated():
    """A subclass that omits get_gamma must still be uninstantiable."""
    class Incomplete(RiskAversionModel):
        pass  # get_gamma not implemented

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_with_get_gamma_can_be_instantiated():
    """A complete subclass must be instantiable."""
    class Minimal(RiskAversionModel):
        def get_gamma(self, t, wealth=None):
            return 1.0

    m = Minimal()
    assert m.get_gamma(0) == 1.0


# ===========================================================================
# ConstantRiskAversion — construction and validation
# ===========================================================================


class TestConstantRiskAversionConstruction:
    def test_valid_gamma_zero(self):
        """gamma=0 is valid; represents a risk-neutral investor."""
        m = ConstantRiskAversion(gamma=0.0)
        assert m.gamma == pytest.approx(0.0)

    def test_valid_gamma_one(self):
        """gamma=1 corresponds to log utility."""
        m = ConstantRiskAversion(gamma=1.0)
        assert m.gamma == pytest.approx(1.0)

    def test_valid_gamma_large(self):
        m = ConstantRiskAversion(gamma=10.0)
        assert m.gamma == pytest.approx(10.0)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ConstantRiskAversion(gamma=-0.1)

    def test_negative_gamma_large_raises(self):
        with pytest.raises(ValueError):
            ConstantRiskAversion(gamma=-5.0)

    def test_is_risk_aversion_model_instance(self):
        assert isinstance(ConstantRiskAversion(gamma=2.0), RiskAversionModel)


# ===========================================================================
# ConstantRiskAversion — get_gamma
# ===========================================================================


class TestConstantRiskAversionGetGamma:
    def test_returns_same_gamma_at_every_time_step(self):
        m = ConstantRiskAversion(gamma=3.0)
        for t in range(100):
            assert m.get_gamma(t) == pytest.approx(3.0)

    def test_wealth_argument_is_ignored(self):
        """Wealth must have no effect on a constant model."""
        m = ConstantRiskAversion(gamma=2.0)
        for w in WEALTH_LEVELS:
            assert m.get_gamma(t=5, wealth=w) == pytest.approx(2.0)

    def test_gamma_zero_returns_zero(self):
        m = ConstantRiskAversion(gamma=0.0)
        assert m.get_gamma(0) == pytest.approx(0.0)
        assert m.get_gamma(99) == pytest.approx(0.0)

    def test_negative_time_raises(self):
        m = ConstantRiskAversion(gamma=2.0)
        with pytest.raises(ValueError):
            m.get_gamma(t=-1)

    def test_returns_exact_constructor_value(self):
        gamma = 4.567
        m = ConstantRiskAversion(gamma=gamma)
        assert m.get_gamma(1000) == pytest.approx(gamma)

    def test_consistent_with_crra_utility_constraint(self):
        """
        CRRAUtility in utility.py requires gamma >= 0.
        All values returned by ConstantRiskAversion must satisfy this.
        """
        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
            m = ConstantRiskAversion(gamma=gamma)
            assert m.get_gamma(0) >= 0.0


# ===========================================================================
# TimeVaryingRiskAversion — construction and validation
# ===========================================================================


class TestTimeVaryingConstruction:
    def test_valid_single_element_schedule(self):
        m = TimeVaryingRiskAversion([3.0])
        assert m.horizon == 1

    def test_valid_glide_path(self):
        m = TimeVaryingRiskAversion(GLIDE_PATH)
        assert m.horizon == HORIZON

    def test_accepts_tuple_input(self):
        """Should accept any Sequence, not just list."""
        m = TimeVaryingRiskAversion(tuple([1.0, 2.0, 3.0]))
        assert m.horizon == 3

    def test_accepts_generator_input(self):
        m = TimeVaryingRiskAversion(x * 0.5 for x in range(1, 6))
        assert m.horizon == 5

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TimeVaryingRiskAversion([])

    def test_negative_gamma_in_schedule_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TimeVaryingRiskAversion([1.0, 2.0, -0.5, 3.0])

    def test_negative_gamma_at_index_zero_raises(self):
        with pytest.raises(ValueError):
            TimeVaryingRiskAversion([-1.0, 2.0, 3.0])

    def test_negative_gamma_at_last_index_raises(self):
        with pytest.raises(ValueError):
            TimeVaryingRiskAversion([1.0, 2.0, -0.1])

    def test_zero_gamma_is_valid_in_schedule(self):
        """gamma=0 (risk-neutral) should be accepted at any position."""
        m = TimeVaryingRiskAversion([0.0, 1.0, 2.0])
        assert m.get_gamma(0) == pytest.approx(0.0)

    def test_is_risk_aversion_model_instance(self):
        assert isinstance(TimeVaryingRiskAversion([2.0, 4.0]), RiskAversionModel)


# ===========================================================================
# TimeVaryingRiskAversion — horizon property
# ===========================================================================


class TestTimeVaryingHorizon:
    def test_horizon_matches_schedule_length(self):
        for n in [1, 5, 10, 40, 100]:
            schedule = [float(i) for i in range(n)]
            m = TimeVaryingRiskAversion(schedule)
            assert m.horizon == n

    def test_horizon_is_read_only_integer(self):
        m = TimeVaryingRiskAversion([1.0, 2.0, 3.0])
        assert isinstance(m.horizon, int)


# ===========================================================================
# TimeVaryingRiskAversion — get_gamma within schedule
# ===========================================================================


class TestTimeVaryingGetGammaInSchedule:
    def test_returns_correct_value_at_each_step(self):
        schedule = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = TimeVaryingRiskAversion(schedule)
        for t, expected in enumerate(schedule):
            assert m.get_gamma(t) == pytest.approx(expected)

    def test_single_step_schedule(self):
        m = TimeVaryingRiskAversion([7.5])
        assert m.get_gamma(0) == pytest.approx(7.5)

    def test_glide_path_is_monotone_increasing(self):
        """
        In a standard retirement glide path, risk aversion must increase
        monotonically over time (investor becomes more conservative).
        """
        m = TimeVaryingRiskAversion(GLIDE_PATH)
        values = [m.get_gamma(t) for t in range(HORIZON)]
        assert all(values[i] <= values[i + 1] for i in range(HORIZON - 1))

    def test_glide_path_start_and_end_values(self):
        m = TimeVaryingRiskAversion(GLIDE_PATH)
        assert m.get_gamma(0) == pytest.approx(2.0)
        assert m.get_gamma(HORIZON - 1) == pytest.approx(6.0)

    def test_wealth_argument_is_ignored(self):
        """Wealth must not influence the returned gamma in this implementation."""
        m = TimeVaryingRiskAversion([2.0, 4.0, 6.0])
        for w in WEALTH_LEVELS:
            assert m.get_gamma(t=1, wealth=w) == pytest.approx(4.0)

    def test_negative_time_raises(self):
        m = TimeVaryingRiskAversion([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            m.get_gamma(t=-1)


# ===========================================================================
# TimeVaryingRiskAversion — clamping behaviour past the horizon
# ===========================================================================


class TestTimeVaryingClampingPastHorizon:
    def test_t_equal_to_horizon_returns_last_value(self):
        """t == horizon is one step past the schedule; should clamp to last."""
        m = TimeVaryingRiskAversion([1.0, 2.0, 5.0])
        assert m.get_gamma(3) == pytest.approx(5.0)

    def test_large_t_returns_last_value(self):
        m = TimeVaryingRiskAversion([1.0, 2.0, 5.0])
        assert m.get_gamma(1000) == pytest.approx(5.0)

    def test_clamping_models_terminal_risk_aversion(self):
        """
        After a glide path ends (e.g. at retirement), the investor's risk
        aversion should remain at its terminal level indefinitely.
        """
        terminal_gamma = GLIDE_PATH[-1]
        m = TimeVaryingRiskAversion(GLIDE_PATH)
        for t in [HORIZON, HORIZON + 10, HORIZON + 100]:
            assert m.get_gamma(t) == pytest.approx(terminal_gamma), (
                f"Clamping failed at t={t}"
            )

    def test_clamping_consistent_with_last_in_schedule_value(self):
        schedule = [0.5, 1.0, 2.0, 4.0]
        m = TimeVaryingRiskAversion(schedule)
        last = m.get_gamma(len(schedule) - 1)
        for t in range(len(schedule), len(schedule) + 20):
            assert m.get_gamma(t) == pytest.approx(last)


# ===========================================================================
# Consistency with utility.py constraints
# ===========================================================================


class TestConsistencyWithUtility:
    def test_constant_model_all_gammas_satisfy_crra_constraint(self):
        """
        Every value returned by the model must be a valid gamma for
        CRRAUtility (i.e. >= 0).
        """
        m = ConstantRiskAversion(gamma=3.5)
        for t in range(50):
            assert m.get_gamma(t) >= 0.0

    def test_time_varying_all_gammas_satisfy_crra_constraint(self):
        m = TimeVaryingRiskAversion(GLIDE_PATH)
        for t in range(HORIZON + 10):
            assert m.get_gamma(t) >= 0.0

    def test_both_models_share_same_interface(self):
        """Both concrete classes must answer get_gamma with the same signature."""
        models = [
            ConstantRiskAversion(gamma=2.0),
            TimeVaryingRiskAversion(GLIDE_PATH),
        ]
        for m in models:
            result = m.get_gamma(t=0)
            assert isinstance(result, float)
            result_with_wealth = m.get_gamma(t=0, wealth=10.0)
            assert isinstance(result_with_wealth, float)

    def test_log_utility_gamma_one_is_valid_for_both_models(self):
        """
        gamma=1 is the log utility special case in utility.py.
        Both models must return it cleanly.
        """
        assert ConstantRiskAversion(gamma=1.0).get_gamma(0) == pytest.approx(1.0)
        assert TimeVaryingRiskAversion([1.0, 1.0, 1.0]).get_gamma(1) == pytest.approx(1.0)