# tests/integration/test_integration_mdp_dynamics.py
"""
Integration tests: market components → MDP dynamics.

Verifies that SingleAssetMDP and MultiAssetMDP produce correct wealth
transitions, price evolution, and terminal rewards when wired with real
NormalReturnDistribution, ConstantRisklessReturn, CRRAUtility, LogUtility,
and ActionSpace.  No stubs are used here.

Key insight: setting sigma=0 makes returns deterministic, allowing exact
arithmetic verification of every wealth path.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mdp import MultiAssetMDP, SingleAssetMDP
from returns import (
    ConstantRisklessReturn,
    MultivariateNormalReturnDistribution,
    NormalReturnDistribution,
)
from state import ActionSpace, AllocationAction, PortfolioState
from utility import CRRAUtility, LogUtility


# ---------------------------------------------------------------------------
# Module-level constants (used across all test classes)
# ---------------------------------------------------------------------------

R = 0.02        # riskless rate
MU = 0.10       # risky expected return
GAMMA = 2.0     # risk-aversion coefficient
T = 3           # investment horizon
W0 = 1.0        # initial wealth


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def single_asset_mdp(
    sigma: float = 0.0,
    time_steps: int = T,
    gamma: float = GAMMA,
    rng=None,
) -> SingleAssetMDP:
    return SingleAssetMDP(
        risky_return=NormalReturnDistribution(mu=MU, sigma=sigma),
        riskless_return=ConstantRisklessReturn(rate=R),
        utility=CRRAUtility(gamma=gamma),
        action_space=ActionSpace(choices=[0.0, 0.1, 0.5, 1.0], n_assets=1),
        time_steps=time_steps,
        rng=rng,
    )


def init_state(wealth: float = W0) -> PortfolioState:
    return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(0.0,))


def two_asset_mdp(time_steps: int = 2) -> MultiAssetMDP:
    """Two-asset MDP with zero variance for deterministic wealth checks."""
    mus = [0.08, 0.12]
    cov = np.zeros((2, 2))
    return MultiAssetMDP(
        risky_returns=MultivariateNormalReturnDistribution(mus=mus, cov=cov),
        riskless_return=ConstantRisklessReturn(rate=R),
        utility=CRRAUtility(gamma=GAMMA),
        action_space=ActionSpace(choices=[0.0, 0.1, 0.5], n_assets=2),
        time_steps=time_steps,
    )


# ---------------------------------------------------------------------------
# TestSingleAssetMDPDeterministicDynamics
# ---------------------------------------------------------------------------


class TestSingleAssetMDPDeterministicDynamics:
    """
    Wealth transition formula (sigma=0, so R_t = mu):
        W_{t+1} = W_t * [theta*(1+mu) + (1-theta)*(1+r)]
                = W_t * (1 + r + theta*(mu - r))
    """

    def _factor(self, theta: float) -> float:
        return 1.0 + R + theta * (MU - R)

    def test_zero_allocation_grows_at_riskless_rate(self):
        mdp = single_asset_mdp()
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.wealth == pytest.approx(W0 * (1.0 + R))

    def test_full_allocation_grows_at_risky_return(self):
        mdp = single_asset_mdp()
        state = init_state()
        action = AllocationAction(allocations=(1.0,))
        # Use a state already at allocation=1.0 to satisfy rebalancing limit
        state_full = PortfolioState(wealth=W0, prices=(1.0,), allocations=(1.0,))
        next_state, _ = mdp.step(state_full, action, t=0)
        assert next_state.wealth == pytest.approx(W0 * (1.0 + MU))

    def test_mixed_allocation_blends_returns(self):
        """theta=0.1 → exact blend of risky and riskless."""
        mdp = single_asset_mdp()
        state = init_state()
        action = AllocationAction(allocations=(0.1,))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.wealth == pytest.approx(W0 * self._factor(0.1))

    def test_two_non_terminal_steps_compound_correctly(self):
        """Wealth at t=2 from two deterministic steps."""
        mdp = single_asset_mdp()
        theta = 0.1
        action = AllocationAction(allocations=(theta,))
        state = init_state()
        state, _ = mdp.step(state, action, t=0)
        state, _ = mdp.step(state, action, t=1)
        expected = W0 * (self._factor(theta) ** 2)
        assert state.wealth == pytest.approx(expected)

    def test_price_evolves_as_gross_risky_return(self):
        """Price_{t+1} = price_t * (1 + R_t). With sigma=0: price_1 = 1+mu."""
        mdp = single_asset_mdp()
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.prices[0] == pytest.approx(1.0 + MU)

    def test_allocations_in_next_state_match_action(self):
        mdp = single_asset_mdp()
        state = init_state()
        action = AllocationAction(allocations=(0.1,))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.allocations == pytest.approx((0.1,))

    def test_wealth_is_positive_after_step(self):
        mdp = single_asset_mdp()
        state = init_state(2.5)
        action = AllocationAction(allocations=(0.0,))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.wealth > 0.0

    def test_sample_next_state_identical_to_step(self):
        """sample_next_state() is an alias for step(); must give same result."""
        rng = np.random.default_rng(42)
        mdp = single_asset_mdp(rng=rng)
        rng2 = np.random.default_rng(42)
        mdp2 = single_asset_mdp(rng=rng2)
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        s1, r1 = mdp.step(state, action, t=0)
        s2, r2 = mdp2.sample_next_state(state, action, t=0)
        assert s1.wealth == pytest.approx(s2.wealth)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# TestSingleAssetMDPRewards
# ---------------------------------------------------------------------------


class TestSingleAssetMDPRewards:
    def test_non_terminal_reward_is_zero(self):
        mdp = single_asset_mdp(time_steps=3)
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        _, reward = mdp.step(state, action, t=0)   # t+1=1, not terminal
        assert reward == pytest.approx(0.0)

    def test_penultimate_step_is_also_non_terminal(self):
        mdp = single_asset_mdp(time_steps=3)
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        _, reward = mdp.step(state, action, t=1)   # t+1=2, not terminal
        assert reward == pytest.approx(0.0)

    def test_terminal_reward_equals_crra_utility(self):
        """At t=T-1, reward = CRRAUtility(gamma).evaluate(W_{T})."""
        gamma = 2.0
        mdp = single_asset_mdp(time_steps=2, gamma=gamma)
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        next_state, reward = mdp.step(state, action, t=1)   # t+1=2=T
        expected = CRRAUtility(gamma).evaluate(next_state.wealth)
        assert reward == pytest.approx(expected)

    def test_terminal_reward_log_utility(self):
        """LogUtility: terminal reward = ln(W_T)."""
        mdp = SingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=MU, sigma=0.0),
            riskless_return=ConstantRisklessReturn(rate=R),
            utility=LogUtility(),
            action_space=ActionSpace(choices=[0.0, 0.1, 1.0], n_assets=1),
            time_steps=1,
        )
        state = init_state()
        action = AllocationAction(allocations=(0.0,))
        next_state, reward = mdp.step(state, action, t=0)
        assert reward == pytest.approx(math.log(next_state.wealth))

    def test_is_terminal_flags(self):
        mdp = single_asset_mdp(time_steps=3)
        assert not mdp.is_terminal(0)
        assert not mdp.is_terminal(2)
        assert mdp.is_terminal(3)
        assert mdp.is_terminal(10)

    def test_time_steps_property(self):
        mdp = single_asset_mdp(time_steps=5)
        assert mdp.time_steps == 5


# ---------------------------------------------------------------------------
# TestSingleAssetMDPFeasibility
# ---------------------------------------------------------------------------


class TestSingleAssetMDPFeasibility:
    def test_feasible_actions_nonempty_from_initial_state(self):
        mdp = single_asset_mdp()
        assert len(mdp.get_feasible_actions(init_state())) > 0

    def test_feasible_actions_are_allocation_actions(self):
        mdp = single_asset_mdp()
        for action in mdp.get_feasible_actions(init_state()):
            assert isinstance(action, AllocationAction)

    def test_infeasible_action_raises_value_error(self):
        """Jump of 0.5 pp from alloc=0.0 violates REBALANCE_LIMIT=0.10."""
        mdp = single_asset_mdp()
        state = init_state()          # allocs=(0.0,)
        bad = AllocationAction(allocations=(0.5,))   # delta=0.5 > 0.10
        with pytest.raises(ValueError):
            mdp.step(state, bad, t=0)

    def test_feasible_actions_all_have_sum_leq_one(self):
        mdp = single_asset_mdp()
        for action in mdp.get_feasible_actions(init_state()):
            assert sum(action.allocations) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# TestMultiAssetMDPDynamics
# ---------------------------------------------------------------------------


class TestMultiAssetMDPDynamics:
    def test_all_cash_grows_at_riskless_rate(self):
        mdp = two_asset_mdp()
        state = PortfolioState(wealth=1.0, prices=(1.0, 1.0), allocations=(0.0, 0.0))
        action = AllocationAction(allocations=(0.0, 0.0))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.wealth == pytest.approx(1.0 + R)

    def test_prices_evolve_independently(self):
        """Price_i(t+1) = Price_i(t) * (1 + R_i).  With sigma=0: R_i = mu_i."""
        mdp = two_asset_mdp()
        state = PortfolioState(wealth=1.0, prices=(1.0, 1.0), allocations=(0.0, 0.0))
        action = AllocationAction(allocations=(0.0, 0.0))
        next_state, _ = mdp.step(state, action, t=0)
        assert next_state.prices[0] == pytest.approx(1.0 + 0.08)
        assert next_state.prices[1] == pytest.approx(1.0 + 0.12)

    def test_mixed_two_asset_wealth_formula(self):
        """W_{t+1} = W_t * [theta1*(1+R1) + theta2*(1+R2) + cash*(1+r)]."""
        mdp = two_asset_mdp()
        state = PortfolioState(wealth=2.0, prices=(1.0, 1.0), allocations=(0.0, 0.0))
        action = AllocationAction(allocations=(0.1, 0.1))
        next_state, _ = mdp.step(state, action, t=0)
        # cash = 1 - 0.1 - 0.1 = 0.8
        expected = 2.0 * (0.1 * 1.08 + 0.1 * 1.12 + 0.8 * 1.02)
        assert next_state.wealth == pytest.approx(expected)

    def test_multi_asset_non_terminal_reward_zero(self):
        mdp = two_asset_mdp(time_steps=2)
        state = PortfolioState(wealth=1.0, prices=(1.0, 1.0), allocations=(0.0, 0.0))
        action = AllocationAction(allocations=(0.0, 0.0))
        _, reward = mdp.step(state, action, t=0)
        assert reward == pytest.approx(0.0)

    def test_multi_asset_terminal_reward_is_utility(self):
        mdp = two_asset_mdp(time_steps=1)
        state = PortfolioState(wealth=1.0, prices=(1.0, 1.0), allocations=(0.0, 0.0))
        action = AllocationAction(allocations=(0.0, 0.0))
        next_state, reward = mdp.step(state, action, t=0)
        assert reward == pytest.approx(CRRAUtility(GAMMA).evaluate(next_state.wealth))

    def test_n_assets_property(self):
        assert two_asset_mdp().n_assets == 2