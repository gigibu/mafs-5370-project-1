"""
Comprehensive test suite for mdp.py, returns.py, and utility.py.

Covers all four scenario combinations:
    N=1 / T=1   — single asset, single period
    N=1 / T=10  — single asset, full horizon
    N=4 / T=1   — max assets, single period
    N=4 / T=10  — max assets, full horizon
"""
import math
import sys
import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state import ActionSpace, AllocationAction, PortfolioState
from returns import (
    ConstantRisklessReturn,
    MultivariateNormalReturnDistribution,
    NormalReturnDistribution,
    StepwiseRisklessReturn,
)
from utility import CRRAUtility, ExponentialUtility, LogUtility
from mdp import MultiAssetMDP, SingleAssetMDP

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

FINE_GRID = [round(x * 0.1, 10) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
REBALANCE_LIMIT = 0.10


def make_single_mdp(
    mu: float = 0.05,
    sigma: float = 0.0,
    r: float = 0.02,
    gamma: float = 2.0,
    time_steps: int = 1,
    seed: int = 42,
) -> SingleAssetMDP:
    """Factory: deterministic (sigma=0) single-asset MDP by default."""
    return SingleAssetMDP(
        risky_return=NormalReturnDistribution(mu=mu, sigma=sigma),
        riskless_return=ConstantRisklessReturn(rate=r),
        utility=CRRAUtility(gamma=gamma),
        action_space=ActionSpace(FINE_GRID, n_assets=1),
        time_steps=time_steps,
        rng=np.random.default_rng(seed),
    )


def make_multi_mdp(
    n_assets: int = 4,
    mus: tuple = None,
    sigma: float = 0.0,
    r: float = 0.02,
    gamma: float = 2.0,
    time_steps: int = 1,
    seed: int = 42,
) -> MultiAssetMDP:
    """Factory: deterministic (sigma=0) multi-asset MDP by default."""
    if mus is None:
        mus = tuple([0.05] * n_assets)
    cov = np.eye(n_assets) * sigma ** 2
    return MultiAssetMDP(
        risky_returns=MultivariateNormalReturnDistribution(mus=list(mus), cov=cov),
        riskless_return=ConstantRisklessReturn(rate=r),
        utility=CRRAUtility(gamma=gamma),
        action_space=ActionSpace(FINE_GRID, n_assets=n_assets),
        time_steps=time_steps,
        rng=np.random.default_rng(seed),
    )


def make_initial_state(
    n_assets: int = 1,
    wealth: float = 1.0,
    alloc: float = 0.0,
) -> PortfolioState:
    return PortfolioState(
        wealth=wealth,
        prices=tuple([1.0] * n_assets),
        allocations=tuple([alloc] * n_assets),
    )


def hold_action(state: PortfolioState) -> AllocationAction:
    """Return an action that exactly replicates the current allocations."""
    return AllocationAction(allocations=state.allocations)


# ===========================================================================
# Returns — ConstantRisklessReturn
# ===========================================================================


class TestConstantRisklessReturn:
    def test_rate_constant_across_all_periods(self):
        model = ConstantRisklessReturn(rate=0.02)
        for t in range(15):
            assert model.get_rate(t) == pytest.approx(0.02)

    def test_zero_rate(self):
        model = ConstantRisklessReturn(rate=0.0)
        assert model.get_rate(0) == pytest.approx(0.0)

    def test_rate_below_minus_one_raises(self):
        with pytest.raises(ValueError):
            ConstantRisklessReturn(rate=-2.0)

    def test_rate_exactly_minus_one_raises(self):
        with pytest.raises(ValueError):
            ConstantRisklessReturn(rate=-1.0)

    def test_frozen_immutable(self):
        model = ConstantRisklessReturn(rate=0.02)
        with pytest.raises((AttributeError, TypeError)):
            model.rate = 0.05  # type: ignore[misc]


# ===========================================================================
# Returns — StepwiseRisklessReturn
# ===========================================================================


class TestStepwiseRisklessReturn:
    def test_reads_correct_rate_per_step(self):
        model = StepwiseRisklessReturn(rates=(0.01, 0.02, 0.03))
        assert model.get_rate(0) == pytest.approx(0.01)
        assert model.get_rate(1) == pytest.approx(0.02)
        assert model.get_rate(2) == pytest.approx(0.03)

    def test_extrapolates_last_rate_beyond_horizon(self):
        model = StepwiseRisklessReturn(rates=(0.01, 0.05))
        assert model.get_rate(10) == pytest.approx(0.05)
        assert model.get_rate(100) == pytest.approx(0.05)

    def test_single_rate_entry(self):
        model = StepwiseRisklessReturn(rates=(0.03,))
        for t in range(5):
            assert model.get_rate(t) == pytest.approx(0.03)

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError):
            StepwiseRisklessReturn(rates=())


# ===========================================================================
# Returns — NormalReturnDistribution
# ===========================================================================


class TestNormalReturnDistribution:
    def test_deterministic_at_sigma_zero(self):
        dist = NormalReturnDistribution(mu=0.08, sigma=0.0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert dist.sample(0, rng) == pytest.approx(0.08)

    def test_mean_property(self):
        dist = NormalReturnDistribution(mu=0.07, sigma=0.15)
        assert dist.mean == pytest.approx(0.07)

    def test_variance_property(self):
        dist = NormalReturnDistribution(mu=0.05, sigma=0.20)
        assert dist.variance == pytest.approx(0.04)

    def test_sample_is_float(self):
        dist = NormalReturnDistribution(mu=0.05, sigma=0.20)
        s = dist.sample(0, np.random.default_rng(1))
        assert isinstance(s, float)

    def test_sample_varies_with_different_seeds(self):
        dist = NormalReturnDistribution(mu=0.05, sigma=0.20)
        samples = {dist.sample(0, np.random.default_rng(seed)) for seed in range(20)}
        assert len(samples) > 1, "Stochastic distribution produced constant samples"

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            NormalReturnDistribution(mu=0.05, sigma=-0.1)

    def test_uses_own_rng_when_none_provided(self):
        dist = NormalReturnDistribution(mu=0.05, sigma=0.20)
        s = dist.sample(0)  # should not raise
        assert isinstance(s, float)


# ===========================================================================
# Returns — MultivariateNormalReturnDistribution
# ===========================================================================


class TestMultivariateNormalReturnDistribution:
    def test_deterministic_at_zero_covariance(self):
        mus = [0.05, 0.07, 0.03]
        cov = np.zeros((3, 3))
        dist = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
        sample = dist.sample(0, np.random.default_rng(0))
        assert sample == pytest.approx(tuple(mus), abs=1e-9)

    def test_n_assets_reflects_input_length(self):
        for n in [1, 2, 3, 4]:
            mus = [0.05] * n
            cov = np.eye(n) * 0.04
            dist = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
            assert dist.n_assets == n

    def test_means_property(self):
        mus = [0.05, 0.07, 0.09]
        dist = MultivariateNormalReturnDistribution(mus=mus, cov=np.eye(3) * 0.01)
        assert dist.means == pytest.approx(mus)

    def test_covariance_matrix_returned_as_copy(self):
        cov = np.eye(2) * 0.04
        dist = MultivariateNormalReturnDistribution(mus=[0.05, 0.07], cov=cov)
        returned_cov = dist.covariance_matrix
        returned_cov[0, 0] = 999.0  # mutate copy
        assert dist.covariance_matrix[0, 0] == pytest.approx(0.04)

    def test_sample_length_matches_n_assets(self):
        mus = [0.05, 0.07, 0.03, 0.04]
        dist = MultivariateNormalReturnDistribution(mus=mus, cov=np.eye(4) * 0.01)
        sample = dist.sample(0, np.random.default_rng(0))
        assert len(sample) == 4

    def test_sample_elements_are_floats(self):
        dist = MultivariateNormalReturnDistribution(
            mus=[0.05, 0.07], cov=np.eye(2) * 0.04
        )
        sample = dist.sample(0, np.random.default_rng(0))
        for s in sample:
            assert isinstance(s, float)

    def test_non_square_cov_raises(self):
        with pytest.raises(ValueError, match=r"shape"):
            MultivariateNormalReturnDistribution(mus=[0.05, 0.07], cov=np.eye(3))

    def test_asymmetric_cov_raises(self):
        cov = np.array([[0.04, 0.01], [0.02, 0.04]])
        with pytest.raises(ValueError, match="symmetric"):
            MultivariateNormalReturnDistribution(mus=[0.05, 0.07], cov=cov)

    def test_non_psd_cov_raises(self):
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3 and -1
        with pytest.raises(ValueError, match="semi-definite"):
            MultivariateNormalReturnDistribution(mus=[0.05, 0.07], cov=cov)

    def test_empty_mus_raises(self):
        with pytest.raises(ValueError):
            MultivariateNormalReturnDistribution(mus=[], cov=np.zeros((0, 0)))

    def test_correlated_assets_sample_varies(self):
        mus = [0.05, 0.07]
        cov = np.array([[0.04, 0.02], [0.02, 0.04]])
        dist = MultivariateNormalReturnDistribution(mus=mus, cov=cov)
        samples = [dist.sample(0, np.random.default_rng(s)) for s in range(20)]
        unique = {s for s in samples}
        assert len(unique) > 1


# ===========================================================================
# Utility — CRRAUtility
# ===========================================================================


class TestCRRAUtility:
    def test_gamma_one_equals_log(self):
        u = CRRAUtility(gamma=1.0)
        assert u.evaluate(math.e) == pytest.approx(1.0, rel=1e-9)
        assert u.evaluate(1.0) == pytest.approx(0.0)

    def test_gamma_two(self):
        u = CRRAUtility(gamma=2.0)
        # U(W) = W^(1-2)/(1-2) = W^(-1)/(-1) = -1/W
        assert u.evaluate(2.0) == pytest.approx(-0.5)
        assert u.evaluate(4.0) == pytest.approx(-0.25)

    def test_gamma_zero_is_linear(self):
        u = CRRAUtility(gamma=0.0)
        # U(W) = W^1 / 1 = W
        assert u.evaluate(3.0) == pytest.approx(3.0)

    def test_marginal_gamma_two(self):
        u = CRRAUtility(gamma=2.0)
        # U'(W) = W^(-2)
        assert u.marginal(2.0) == pytest.approx(0.25)
        assert u.marginal(4.0) == pytest.approx(0.0625)

    def test_is_risk_averse_positive_gamma(self):
        assert CRRAUtility(gamma=2.0).is_risk_averse

    def test_not_risk_averse_at_gamma_zero(self):
        assert not CRRAUtility(gamma=0.0).is_risk_averse

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            CRRAUtility(gamma=-1.0)

    def test_zero_wealth_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            CRRAUtility(gamma=2.0).evaluate(0.0)

    def test_marginal_zero_wealth_raises(self):
        with pytest.raises(ValueError):
            CRRAUtility(gamma=2.0).marginal(0.0)

    def test_utility_strictly_increasing(self):
        u = CRRAUtility(gamma=2.0)
        for w1, w2 in [(1.0, 2.0), (0.5, 1.5), (2.0, 3.0)]:
            assert u.evaluate(w1) < u.evaluate(w2)

    def test_utility_strictly_concave(self):
        """Midpoint must exceed the secant line: U((a+b)/2) > (U(a)+U(b))/2."""
        u = CRRAUtility(gamma=2.0)
        for a, b in [(1.0, 3.0), (0.5, 2.5)]:
            mid = (a + b) / 2
            assert u.evaluate(mid) > (u.evaluate(a) + u.evaluate(b)) / 2


# ===========================================================================
# Utility — ExponentialUtility
# ===========================================================================


class TestExponentialUtility:
    def test_evaluate_at_zero_wealth(self):
        u = ExponentialUtility(alpha=1.0)
        # U(0) = -exp(0)/1 = -1
        assert u.evaluate(0.0) == pytest.approx(-1.0)

    def test_evaluate_at_one_wealth(self):
        u = ExponentialUtility(alpha=1.0)
        # U(1) = -exp(-1)
        assert u.evaluate(1.0) == pytest.approx(-math.exp(-1.0))

    def test_marginal_at_zero(self):
        u = ExponentialUtility(alpha=1.0)
        # U'(W) = exp(-α*W); at W=0: exp(0)=1
        assert u.marginal(0.0) == pytest.approx(1.0)

    def test_marginal_decays_with_wealth(self):
        u = ExponentialUtility(alpha=1.0)
        assert u.marginal(1.0) < u.marginal(0.0)

    def test_is_risk_averse(self):
        assert ExponentialUtility(alpha=2.0).is_risk_averse

    def test_utility_strictly_increasing(self):
        u = ExponentialUtility(alpha=1.0)
        assert u.evaluate(0.0) < u.evaluate(1.0) < u.evaluate(5.0)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ExponentialUtility(alpha=0.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            ExponentialUtility(alpha=-1.0)

    def test_defined_at_negative_wealth(self):
        """CARA utility is valid for all W ∈ ℝ, including W < 0."""
        u = ExponentialUtility(alpha=1.0)
        v = u.evaluate(-1.0)  # should not raise
        assert v < u.evaluate(0.0)


# ===========================================================================
# Utility — LogUtility
# ===========================================================================


class TestLogUtility:
    def test_evaluate_at_one(self):
        assert LogUtility().evaluate(1.0) == pytest.approx(0.0)

    def test_evaluate_at_e(self):
        assert LogUtility().evaluate(math.e) == pytest.approx(1.0)

    def test_marginal_is_reciprocal(self):
        u = LogUtility()
        for w in [0.5, 1.0, 2.0, 5.0]:
            assert u.marginal(w) == pytest.approx(1.0 / w)

    def test_zero_wealth_raises(self):
        with pytest.raises(ValueError):
            LogUtility().evaluate(0.0)

    def test_is_risk_averse(self):
        assert LogUtility().is_risk_averse

    def test_matches_crra_gamma_one(self):
        log_u = LogUtility()
        crra_u = CRRAUtility(gamma=1.0)
        for w in [0.1, 0.5, 1.0, 2.0, 10.0]:
            assert log_u.evaluate(w) == pytest.approx(crra_u.evaluate(w), rel=1e-9)


# ===========================================================================
# SingleAssetMDP — construction and properties
# ===========================================================================


class TestSingleAssetMDPConstruction:
    def test_valid_construction_stores_time_steps(self):
        mdp = make_single_mdp(time_steps=5)
        assert mdp.time_steps == 5

    def test_zero_time_steps_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            SingleAssetMDP(
                risky_return=NormalReturnDistribution(0.05, 0.0),
                riskless_return=ConstantRisklessReturn(0.02),
                utility=CRRAUtility(2.0),
                action_space=ActionSpace(FINE_GRID, n_assets=1),
                time_steps=0,
            )

    def test_negative_time_steps_raises(self):
        with pytest.raises(ValueError):
            SingleAssetMDP(
                risky_return=NormalReturnDistribution(0.05, 0.0),
                riskless_return=ConstantRisklessReturn(0.02),
                utility=CRRAUtility(2.0),
                action_space=ActionSpace(FINE_GRID, n_assets=1),
                time_steps=-3,
            )

    def test_default_rng_is_created_when_none(self):
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(0.05, 0.0),
            riskless_return=ConstantRisklessReturn(0.02),
            utility=CRRAUtility(2.0),
            action_space=ActionSpace(FINE_GRID, n_assets=1),
            time_steps=1,
        )
        assert mdp._rng is not None

    def test_repr_contains_key_info(self):
        mdp = make_single_mdp(time_steps=3)
        r = repr(mdp)
        assert "T=3" in r


# ===========================================================================
# SingleAssetMDP — is_terminal
# ===========================================================================


class TestSingleAssetMDPIsTerminal:
    @pytest.mark.parametrize("T", [1, 3, 5, 10])
    def test_terminal_exactly_at_T(self, T):
        mdp = make_single_mdp(time_steps=T)
        assert mdp.is_terminal(T)

    @pytest.mark.parametrize("T", [1, 3, 5, 10])
    def test_not_terminal_before_T(self, T):
        mdp = make_single_mdp(time_steps=T)
        for t in range(T):
            assert not mdp.is_terminal(t), f"is_terminal({t}) should be False for T={T}"

    def test_terminal_past_T(self):
        mdp = make_single_mdp(time_steps=5)
        assert mdp.is_terminal(6)
        assert mdp.is_terminal(100)

    def test_terminal_at_zero_when_T_is_one(self):
        mdp = make_single_mdp(time_steps=1)
        assert not mdp.is_terminal(0)
        assert mdp.is_terminal(1)


# ===========================================================================
# SingleAssetMDP — _compute_next_wealth
# ===========================================================================


class TestSingleAssetComputeNextWealth:
    """Isolated unit tests for the deterministic wealth-transition formula."""

    def setup_method(self):
        self.mdp = make_single_mdp()

    def test_all_cash_uses_only_riskless_return(self):
        w = self.mdp._compute_next_wealth(1.0, 0.0, risky_return=0.20, riskless_rate=0.02)
        assert w == pytest.approx(1.02)

    def test_fully_invested_uses_only_risky_return(self):
        w = self.mdp._compute_next_wealth(1.0, 1.0, risky_return=0.10, riskless_rate=0.02)
        assert w == pytest.approx(1.10)

    def test_half_half_allocation(self):
        # W * (0.5 * 1.10 + 0.5 * 1.02) = 1.06
        w = self.mdp._compute_next_wealth(1.0, 0.5, risky_return=0.10, riskless_rate=0.02)
        assert w == pytest.approx(1.06)

    def test_wealth_scales_proportionally(self):
        w1 = self.mdp._compute_next_wealth(1.0, 0.4, risky_return=0.08, riskless_rate=0.02)
        w2 = self.mdp._compute_next_wealth(3.0, 0.4, risky_return=0.08, riskless_rate=0.02)
        assert w2 == pytest.approx(3.0 * w1)

    def test_equal_returns_allocation_irrelevant(self):
        """When R = r, portfolio return equals r regardless of allocation."""
        for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            w = self.mdp._compute_next_wealth(1.0, theta, risky_return=0.02, riskless_rate=0.02)
            assert w == pytest.approx(1.02), f"Failed at theta={theta}"

    def test_negative_risky_return_reduces_wealth(self):
        w = self.mdp._compute_next_wealth(1.0, 1.0, risky_return=-0.20, riskless_rate=0.02)
        assert w == pytest.approx(0.80)
        assert w > 0  # wealth stays positive

    def test_large_initial_wealth(self):
        w = self.mdp._compute_next_wealth(
            100_000.0, 0.5, risky_return=0.10, riskless_rate=0.02
        )
        assert w == pytest.approx(100_000.0 * 1.06)


# ===========================================================================
# Scenario: N=1, T=1
# ===========================================================================


class TestScenarioN1T1:
    """Single risky asset, one period — fundamental building block."""

    def test_step_returns_portfolio_state_and_float_reward(self):
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        s1, reward = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert isinstance(s1, PortfolioState)
        assert isinstance(reward, float)

    def test_all_cash_wealth_is_riskless_return(self):
        """θ=0 → W_1 = W_0 * (1 + r) = 1.02."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        s1, _ = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert s1.wealth == pytest.approx(1.02)

    def test_fully_invested_wealth_is_risky_return(self):
        """θ=1 → W_1 = W_0 * (1 + μ) = 1.05 (with σ=0)."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=0.05, sigma=0.0),
            riskless_return=ConstantRisklessReturn(rate=0.02),
            utility=CRRAUtility(gamma=2.0),
            action_space=space,
            time_steps=1,
            rng=np.random.default_rng(0),
        )
        s0 = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(1.0,))
        s1, _ = mdp.step(s0, AllocationAction(allocations=(1.0,)), t=0)
        assert s1.wealth == pytest.approx(1.05)

    def test_mixed_allocation_wealth(self):
        """θ=0.5, μ=0.10, r=0.02 → W_1 = 0.5*1.10 + 0.5*1.02 = 1.06."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        s0 = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.5,))
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=0.10, sigma=0.0),
            riskless_return=ConstantRisklessReturn(rate=0.02),
            utility=CRRAUtility(gamma=2.0),
            action_space=space,
            time_steps=1,
            rng=np.random.default_rng(0),
        )
        s1, _ = mdp.step(s0, hold_action(s0), t=0)
        assert s1.wealth == pytest.approx(1.06)

    def test_terminal_step_reward_equals_utility_of_wealth(self):
        """The single step in T=1 is terminal → reward = U(W_1)."""
        gamma = 2.0
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, gamma=gamma, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        s1, reward = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert reward == pytest.approx(CRRAUtility(gamma).evaluate(s1.wealth))

    def test_non_terminal_step_reward_is_zero(self):
        """For T=2, the transition at t=0 is not terminal → reward=0."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, time_steps=2)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        _, reward = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert reward == pytest.approx(0.0)

    def test_price_updates_with_risky_return(self):
        """X(1) = X(0) * (1 + μ) = 1.05 when σ=0."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        s1, _ = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert s1.prices[0] == pytest.approx(1.05)

    def test_next_state_records_chosen_allocation(self):
        """The action's allocation must be carried into the next state."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        a = AllocationAction(allocations=(0.1,))
        s1, _ = mdp.step(s0, a, t=0)
        assert s1.allocations == (0.1,)

    def test_infeasible_action_raises_value_error(self):
        """Jump of 0.5 from θ=0 exceeds the 10% rebalancing limit."""
        mdp = make_single_mdp(time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        a = AllocationAction(allocations=(0.5,))
        with pytest.raises(ValueError):
            mdp.step(s0, a, t=0)

    def test_sample_next_state_equals_step_for_deterministic_dist(self):
        """With σ=0, step and sample_next_state produce identical results."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, time_steps=1, seed=7)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        a = AllocationAction(allocations=(0.0,))
        s1_step, r1 = mdp.step(s0, a, t=0)
        mdp._rng = np.random.default_rng(7)
        s1_samp, r2 = mdp.sample_next_state(s0, a, t=0)
        assert s1_step.wealth == pytest.approx(s1_samp.wealth)
        assert r1 == pytest.approx(r2)

    def test_get_feasible_actions_returns_allocation_actions(self):
        mdp = make_single_mdp(time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        actions = mdp.get_feasible_actions(s0)
        assert len(actions) > 0
        assert all(isinstance(a, AllocationAction) for a in actions)

    def test_get_feasible_actions_all_valid(self):
        mdp = make_single_mdp(time_steps=1)
        s0 = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.5,))
        for a in mdp.get_feasible_actions(s0):
            assert abs(a.allocations[0] - 0.5) <= REBALANCE_LIMIT + 1e-9

    def test_wealth_positive_after_step(self):
        mdp = make_single_mdp(mu=-0.05, sigma=0.0, r=-0.005, time_steps=1)
        s0 = make_initial_state(n_assets=1, alloc=0.0)
        s1, _ = mdp.step(s0, AllocationAction(allocations=(0.0,)), t=0)
        assert s1.wealth > 0


# ===========================================================================
# Scenario: N=1, T=10
# ===========================================================================


class TestScenarioN1T10:
    """Single risky asset over the full 10-period horizon."""

    HORIZON = 10

    def _run_episode(
        self,
        mdp: SingleAssetMDP,
        initial_alloc: float = 0.0,
        delta: float = 0.1,
    ) -> Tuple[List[PortfolioState], List[float]]:
        """Deterministic trajectory: increase allocation by `delta` each step."""
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,),
            allocations=(initial_alloc,),
        )
        states, rewards = [s], []
        for t in range(self.HORIZON):
            cur = s.allocations[0]
            new_alloc = round(min(cur + delta, 1.0), 10)
            a = AllocationAction(allocations=(new_alloc,))
            s, r = mdp.step(s, a, t)
            states.append(s)
            rewards.append(r)
        return states, rewards

    def test_episode_produces_T_plus_one_states(self):
        mdp = make_single_mdp(time_steps=self.HORIZON)
        states, rewards = self._run_episode(mdp)
        assert len(states) == self.HORIZON + 1
        assert len(rewards) == self.HORIZON

    def test_only_final_reward_is_nonzero(self):
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, time_steps=self.HORIZON)
        _, rewards = self._run_episode(mdp)
        for i, r in enumerate(rewards[:-1]):
            assert r == pytest.approx(0.0), f"Non-zero reward at non-terminal t={i}"
        assert rewards[-1] != pytest.approx(0.0)

    def test_terminal_reward_equals_utility_of_final_wealth(self):
        gamma = 3.0
        mdp = make_single_mdp(
            mu=0.05, sigma=0.0, r=0.02, gamma=gamma, time_steps=self.HORIZON
        )
        states, rewards = self._run_episode(mdp)
        expected = CRRAUtility(gamma).evaluate(states[-1].wealth)
        assert rewards[-1] == pytest.approx(expected)

    def test_wealth_grows_monotonically_with_positive_deterministic_returns(self):
        """With σ=0 and positive μ, r: wealth is strictly increasing."""
        mdp = make_single_mdp(mu=0.05, sigma=0.0, r=0.02, time_steps=self.HORIZON)
        states, _ = self._run_episode(mdp, delta=0.0, initial_alloc=0.5)
        for i in range(1, len(states)):
            assert states[i].wealth > states[i - 1].wealth

    def test_prices_compound_at_risky_return_rate(self):
        """X(t) = (1+μ)^t for deterministic returns (σ=0)."""
        mu = 0.05
        mdp = make_single_mdp(mu=mu, sigma=0.0, r=0.02, time_steps=self.HORIZON)
        states, _ = self._run_episode(mdp)
        for t, s in enumerate(states):
            assert s.prices[0] == pytest.approx((1.0 + mu) ** t, rel=1e-9)

    def test_allocation_reaches_one_in_ten_steps(self):
        """Starting at θ=0, increasing by 0.1 each step: θ=1 at t=10."""
        mdp = make_single_mdp(time_steps=self.HORIZON)
        states, _ = self._run_episode(mdp, initial_alloc=0.0, delta=0.1)
        assert states[-1].allocations[0] == pytest.approx(1.0)

    def test_rebalancing_constraint_enforced_every_step(self):
        mdp = make_single_mdp(time_steps=self.HORIZON)
        states, _ = self._run_episode(mdp)
        for prev, curr in zip(states[:-1], states[1:]):
            delta = abs(curr.allocations[0] - prev.allocations[0])
            assert delta <= REBALANCE_LIMIT + 1e-9

    def test_all_states_are_valid_throughout(self):
        mdp = make_single_mdp(time_steps=self.HORIZON)
        states, _ = self._run_episode(mdp)
        for s in states:
            assert s.wealth > 0
            assert s.n_assets == 1
            assert all(p > 0 for p in s.prices)
            assert 0.0 <= sum(s.allocations) <= 1.0 + 1e-9

    def test_stochastic_runs_produce_different_final_wealth(self):
        """With σ > 0, independent seeds must yield different final wealth."""
        final_wealths = set()
        for seed in range(25):
            mdp = make_single_mdp(
                mu=0.05, sigma=0.20, r=0.02, time_steps=self.HORIZON, seed=seed
            )
            s = make_initial_state(n_assets=1, alloc=0.5)   # <-- start with 50% risky
            for t in range(self.HORIZON):
                s, _ = mdp.step(s, AllocationAction(allocations=(0.5,)), t)  # <-- stay 50% risky
            final_wealths.add(round(s.wealth, 6))
        assert len(final_wealths) > 1

    def test_exponential_utility_as_terminal_reward(self):
        alpha = 1.5
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=0.05, sigma=0.0),
            riskless_return=ConstantRisklessReturn(0.02),
            utility=ExponentialUtility(alpha=alpha),
            action_space=ActionSpace(FINE_GRID, n_assets=1),
            time_steps=self.HORIZON,
            rng=np.random.default_rng(0),
        )
        states, rewards = self._run_episode(mdp)
        expected = ExponentialUtility(alpha).evaluate(states[-1].wealth)
        assert rewards[-1] == pytest.approx(expected)

    def test_log_utility_as_terminal_reward(self):
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=0.05, sigma=0.0),
            riskless_return=ConstantRisklessReturn(0.02),
            utility=LogUtility(),
            action_space=ActionSpace(FINE_GRID, n_assets=1),
            time_steps=self.HORIZON,
            rng=np.random.default_rng(0),
        )
        states, rewards = self._run_episode(mdp)
        assert rewards[-1] == pytest.approx(LogUtility().evaluate(states[-1].wealth))

    def test_is_terminal_boundary_at_horizon_ten(self):
        mdp = make_single_mdp(time_steps=self.HORIZON)
        assert not mdp.is_terminal(self.HORIZON - 1)
        assert mdp.is_terminal(self.HORIZON)


# ===========================================================================
# MultiAssetMDP — construction
# ===========================================================================


class TestMultiAssetMDPConstruction:
    def test_valid_four_asset_construction(self):
        mdp = make_multi_mdp(n_assets=4, time_steps=5)
        assert mdp.n_assets == 4
        assert mdp.time_steps == 5

    def test_zero_time_steps_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            MultiAssetMDP(
                risky_returns=MultivariateNormalReturnDistribution(
                    mus=[0.05] * 2, cov=np.eye(2) * 0.01
                ),
                riskless_return=ConstantRisklessReturn(0.02),
                utility=CRRAUtility(2.0),
                action_space=ActionSpace(FINE_GRID, n_assets=2),
                time_steps=0,
            )

    def test_n_assets_mismatch_raises(self):
        """action_space for 4 assets, distribution for 2 assets → ValueError."""
        dist = MultivariateNormalReturnDistribution(
            mus=[0.05, 0.07], cov=np.eye(2) * 0.01
        )
        with pytest.raises(ValueError, match="n_assets"):
            MultiAssetMDP(
                risky_returns=dist,
                riskless_return=ConstantRisklessReturn(0.02),
                utility=CRRAUtility(2.0),
                action_space=ActionSpace(FINE_GRID, n_assets=4),
                time_steps=1,
            )

    def test_repr_contains_n_assets_and_T(self):
        mdp = make_multi_mdp(n_assets=3, time_steps=7)
        r = repr(mdp)
        assert "n_assets=3" in r
        assert "T=7" in r


# ===========================================================================
# MultiAssetMDP — _compute_next_wealth
# ===========================================================================


class TestMultiAssetComputeNextWealth:
    def setup_method(self):
        self.mdp = make_multi_mdp(n_assets=4)

    def test_all_cash_uses_only_riskless_return(self):
        w = self.mdp._compute_next_wealth(
            1.0,
            allocations=(0.0, 0.0, 0.0, 0.0),
            risky_returns=(0.10, 0.20, 0.30, 0.40),
            riskless_rate=0.02,
        )
        assert w == pytest.approx(1.02)

    def test_equal_weight_all_same_return(self):
        """25% * 4 assets, each returning 5% → W_1 = 1.05."""
        w = self.mdp._compute_next_wealth(
            1.0,
            allocations=(0.25, 0.25, 0.25, 0.25),
            risky_returns=(0.05, 0.05, 0.05, 0.05),
            riskless_rate=0.02,
        )
        assert w == pytest.approx(1.05)

    def test_mixed_allocation_exact_arithmetic(self):
        # 30% in asset A (+10%), 20% in asset B (+5%), 50% cash (+2%)
        # W = 0.30*1.10 + 0.20*1.05 + 0.50*1.02 = 0.33 + 0.21 + 0.51 = 1.05
        w = self.mdp._compute_next_wealth(
            1.0,
            allocations=(0.3, 0.2, 0.0, 0.0),
            risky_returns=(0.10, 0.05, 0.0, 0.0),
            riskless_rate=0.02,
        )
        assert w == pytest.approx(0.30 * 1.10 + 0.20 * 1.05 + 0.50 * 1.02)

    def test_wealth_scales_proportionally_with_initial_wealth(self):
        kwargs = dict(
            allocations=(0.2, 0.2, 0.1, 0.1),
            risky_returns=(0.07, 0.05, 0.03, 0.01),
            riskless_rate=0.02,
        )
        w1 = self.mdp._compute_next_wealth(1.0, **kwargs)
        w2 = self.mdp._compute_next_wealth(4.0, **kwargs)
        assert w2 == pytest.approx(4.0 * w1)

    def test_single_asset_equivalent_to_single_asset_mdp_formula(self):
        """Multi-asset formula with N=1 must match the single-asset formula."""
        single_mdp = make_single_mdp()
        multi_mdp = make_multi_mdp(n_assets=1)
        theta, R, r = 0.4, 0.08, 0.02
        w_single = single_mdp._compute_next_wealth(1.0, theta, R, r)
        w_multi = multi_mdp._compute_next_wealth(1.0, (theta,), (R,), r)
        assert w_single == pytest.approx(w_multi)


# ===========================================================================
# Scenario: N=4, T=1
# ===========================================================================


class TestScenarioN4T1:
    """Four risky assets, one period."""

    def test_step_returns_four_asset_state(self):
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.1, 0.0, 0.0, 0.0))
        s1, _ = mdp.step(s0, a, t=0)
        assert s1.n_assets == 4

    def test_all_cash_deterministic_wealth(self):
        """All cash → W_1 = (1 + r) = 1.02."""
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, r=0.02, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.0, 0.0, 0.0, 0.0))
        s1, _ = mdp.step(s0, a, t=0)
        assert s1.wealth == pytest.approx(1.02)

    def test_equal_weight_all_same_return(self):
        """25% per asset, all returning 5%, 0% cash → W_1 = 1.05."""
        mus = (0.05, 0.05, 0.05, 0.05)
        mdp = make_multi_mdp(n_assets=4, mus=mus, sigma=0.0, r=0.02, time_steps=1)
        s0 = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.25, 0.25, 0.25, 0.25),
        )
        s1, _ = mdp.step(s0, hold_action(s0), t=0)
        assert s1.wealth == pytest.approx(1.05)

    def test_prices_update_independently_per_asset(self):
        """Each price must compound by its own return."""
        mus = (0.10, 0.05, 0.02, -0.01)
        mdp = make_multi_mdp(n_assets=4, mus=mus, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.0, 0.0, 0.0, 0.0))
        s1, _ = mdp.step(s0, a, t=0)
        for i, mu in enumerate(mus):
            assert s1.prices[i] == pytest.approx(1.0 * (1.0 + mu))

    def test_terminal_reward_equals_utility_four_assets(self):
        gamma = 2.0
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, r=0.02, gamma=gamma, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.0, 0.0, 0.0, 0.0))
        s1, reward = mdp.step(s0, a, t=0)
        assert reward == pytest.approx(CRRAUtility(gamma).evaluate(s1.wealth))

    def test_non_terminal_step_four_assets_reward_zero(self):
        """T=2 → step at t=0 is not terminal."""
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=2)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.0, 0.0, 0.0, 0.0))
        _, reward = mdp.step(s0, a, t=0)
        assert reward == pytest.approx(0.0)

    def test_infeasible_jump_raises(self):
        """Δθ = 0.5 >> REBALANCE_LIMIT must be rejected."""
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.5, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            mdp.step(s0, a, t=0)

    def test_per_asset_rebalancing_respected(self):
        """Exactly at the limit is valid; one tick over is not."""
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=1)
        s0 = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.1, 0.1),
        )
        # Valid: each delta = 0.1
        a_valid = AllocationAction(allocations=(0.3, 0.1, 0.1, 0.1))
        s1, _ = mdp.step(s0, a_valid, t=0)
        assert s1.n_assets == 4

        # Invalid: asset 0 delta = 0.11 > REBALANCE_LIMIT
        a_invalid = AllocationAction(allocations=(0.31, 0.2, 0.1, 0.1))
        with pytest.raises(ValueError):
            mdp.step(s0, a_invalid, t=0)

    def test_get_feasible_actions_all_have_four_assets(self):
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        for a in mdp.get_feasible_actions(s0):
            assert a.n_assets == 4

    def test_new_state_carries_chosen_allocations(self):
        mdp = make_multi_mdp(n_assets=4, sigma=0.0, time_steps=1)
        s0 = make_initial_state(n_assets=4, alloc=0.0)
        a = AllocationAction(allocations=(0.1, 0.1, 0.0, 0.0))
        s1, _ = mdp.step(s0, a, t=0)
        assert s1.allocations == (0.1, 0.1, 0.0, 0.0)

    def test_action_space_size_is_1001(self):
        """C(14,4)=1001 sum-feasible actions on an 11-point grid for N=4."""
        space = ActionSpace(FINE_GRID, n_assets=4)
        assert len(space) == 1001


# ===========================================================================
# Scenario: N=4, T=10
# ===========================================================================


class TestScenarioN4T10:
    """Four risky assets over the full 10-period horizon."""

    HORIZON = 10
    N_ASSETS = 4
    MUS = (0.07, 0.05, 0.03, 0.01)

    def _make_mdp(
        self,
        sigma: float = 0.0,
        mus: tuple = MUS,
        r: float = 0.02,
        gamma: float = 2.0,
        seed: int = 42,
    ) -> MultiAssetMDP:
        cov = np.eye(self.N_ASSETS) * sigma ** 2
        return MultiAssetMDP(
            risky_returns=MultivariateNormalReturnDistribution(
                mus=list(mus), cov=cov
            ),
            riskless_return=ConstantRisklessReturn(rate=r),
            utility=CRRAUtility(gamma=gamma),
            action_space=ActionSpace(FINE_GRID, n_assets=self.N_ASSETS),
            time_steps=self.HORIZON,
            rng=np.random.default_rng(seed),
        )

    def _run_episode(
        self,
        mdp: MultiAssetMDP,
        step_size: float = 0.1,
    ) -> Tuple[List[PortfolioState], List[float]]:
        """
        Simulate 10 steps: increase assets 0 and 1 by step_size each period,
        capped at 0.5 per asset. Uses only values in FINE_GRID.
        """
        s = make_initial_state(n_assets=self.N_ASSETS, alloc=0.0)
        states, rewards = [s], []
        for t in range(self.HORIZON):
            cur = s.allocations
            new_0 = round(min(cur[0] + step_size, 0.5), 10)
            new_1 = round(min(cur[1] + step_size, 0.5), 10)
            a = AllocationAction(allocations=(new_0, new_1, cur[2], cur[3]))
            s, r = mdp.step(s, a, t)
            states.append(s)
            rewards.append(r)
        return states, rewards

    def test_episode_produces_correct_number_of_steps(self):
        states, rewards = self._run_episode(self._make_mdp())
        assert len(states) == self.HORIZON + 1
        assert len(rewards) == self.HORIZON

    def test_only_final_reward_is_nonzero(self):
        _, rewards = self._run_episode(self._make_mdp())
        for i, r in enumerate(rewards[:-1]):
            assert r == pytest.approx(0.0), f"Unexpected reward at t={i}"
        assert rewards[-1] != pytest.approx(0.0)

    def test_terminal_reward_matches_utility_of_final_wealth(self):
        gamma = 2.0
        states, rewards = self._run_episode(self._make_mdp(gamma=gamma))
        expected = CRRAUtility(gamma).evaluate(states[-1].wealth)
        assert rewards[-1] == pytest.approx(expected)

    def test_all_states_have_four_assets(self):
        for s in self._run_episode(self._make_mdp())[0]:
            assert s.n_assets == self.N_ASSETS

    def test_all_states_have_positive_wealth(self):
        for s in self._run_episode(self._make_mdp())[0]:
            assert s.wealth > 0

    def test_all_states_have_positive_prices(self):
        for s in self._run_episode(self._make_mdp())[0]:
            assert all(p > 0 for p in s.prices)

    def test_all_states_have_valid_allocations(self):
        for s in self._run_episode(self._make_mdp())[0]:
            assert sum(s.allocations) <= 1.0 + 1e-9
            assert all(a >= 0 for a in s.allocations)

    def test_prices_compound_independently_over_full_horizon(self):
        """X_i(t) = (1+μᵢ)^t for deterministic returns (σ=0)."""
        states, _ = self._run_episode(self._make_mdp(sigma=0.0))
        for t, s in enumerate(states):
            for i, (price, mu) in enumerate(zip(s.prices, self.MUS)):
                expected = (1.0 + mu) ** t
                assert price == pytest.approx(expected, rel=1e-6), (
                    f"Asset {i} price mismatch at t={t}"
                )

    def test_rebalancing_constraint_respected_every_step(self):
        states, _ = self._run_episode(self._make_mdp())
        for prev, curr in zip(states[:-1], states[1:]):
            for new_a, old_a in zip(curr.allocations, prev.allocations):
                assert abs(new_a - old_a) <= REBALANCE_LIMIT + 1e-9

    def test_wealth_increases_deterministically_with_positive_returns(self):
        """All cash (delta=0.0), positive r → wealth strictly increasing."""
        states, _ = self._run_episode(self._make_mdp(sigma=0.0), step_size=0.0)
        for i in range(1, len(states)):
            assert states[i].wealth >= states[i - 1].wealth

    def test_stochastic_episodes_produce_distinct_final_wealth(self):
        """With σ > 0, different seeds must yield different trajectories."""
        final_wealths = set()
        for seed in range(15):
            mdp = self._make_mdp(sigma=0.15, seed=seed)
            states, _ = self._run_episode(mdp)
            final_wealths.add(round(states[-1].wealth, 5))
        assert len(final_wealths) > 1

    def test_is_terminal_boundary_horizon_ten(self):
        mdp = self._make_mdp()
        assert not mdp.is_terminal(9)
        assert mdp.is_terminal(10)
        assert mdp.is_terminal(11)

    def test_get_feasible_actions_non_empty_throughout_episode(self):
        """Every state visited during the episode must have at least one feasible action."""
        mdp = self._make_mdp()
        states, _ = self._run_episode(mdp)
        # exclude terminal state (no action needed there)
        for s in states[:-1]:
            assert len(mdp.get_feasible_actions(s)) > 0

    def test_different_utility_functions_give_different_final_rewards(self):
        """The same trajectory must yield different terminal rewards under CRRA vs Exp."""
        crra_mdp = self._make_mdp(gamma=2.0)
        exp_mdp = MultiAssetMDP(
            risky_returns=MultivariateNormalReturnDistribution(
                mus=list(self.MUS), cov=np.zeros((4, 4))
            ),
            riskless_return=ConstantRisklessReturn(rate=0.02),
            utility=ExponentialUtility(alpha=1.0),
            action_space=ActionSpace(FINE_GRID, n_assets=4),
            time_steps=self.HORIZON,
            rng=np.random.default_rng(42),
        )
        _, crra_rewards = self._run_episode(crra_mdp)
        _, exp_rewards = self._run_episode(exp_mdp)
        assert not math.isclose(crra_rewards[-1], exp_rewards[-1])

    def test_correlated_returns_do_not_violate_state_invariants(self):
        """Full correlation structure must not produce invalid states."""
        cov = np.array([
            [0.04, 0.02, 0.01, 0.005],
            [0.02, 0.04, 0.01, 0.005],
            [0.01, 0.01, 0.04, 0.005],
            [0.005, 0.005, 0.005, 0.04],
        ])
        mdp = MultiAssetMDP(
            risky_returns=MultivariateNormalReturnDistribution(
                mus=list(self.MUS), cov=cov
            ),
            riskless_return=ConstantRisklessReturn(0.02),
            utility=CRRAUtility(2.0),
            action_space=ActionSpace(FINE_GRID, n_assets=4),
            time_steps=self.HORIZON,
            rng=np.random.default_rng(0),
        )
        states, _ = self._run_episode(mdp)
        for s in states:
            assert s.wealth > 0
            assert all(p > 0 for p in s.prices)
            assert s.n_assets == 4