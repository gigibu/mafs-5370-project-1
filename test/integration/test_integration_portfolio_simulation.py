# tests/integration/test_integration_portfolio_simulation.py
"""
Integration tests: extended MDP → PortfolioSimulator → trajectory statistics.

PortfolioSimulator requires two attributes that are absent from the
production MDPs (which expose time_steps, not n_steps, and have no
initial_state() factory).  The ExtendedSingleAssetMDP subclass defined
here adds those two attributes and is the *only* non-production code in
this file.

Policy choices are limited to allocations ≤ 0.10 so that every step
satisfies the REBALANCE_LIMIT constraint imposed by ActionSpace.is_valid().
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mdp import SingleAssetMDP
from policy import AnalyticalMertonPolicy, Policy
from returns import ConstantRisklessReturn, NormalReturnDistribution
from simulator import PortfolioSimulator
from state import ActionSpace, AllocationAction, PortfolioState
from utility import CRRAUtility


# ---------------------------------------------------------------------------
# Bridge: add n_steps + initial_state() that PortfolioSimulator requires
# ---------------------------------------------------------------------------


class ExtendedSingleAssetMDP(SingleAssetMDP):
    """
    Thin subclass that adds the two attributes PortfolioSimulator needs.

    SingleAssetMDP exposes time_steps; PortfolioSimulator calls n_steps.
    SingleAssetMDP has no initial_state factory; PortfolioSimulator calls
    initial_state(wealth).
    """

    @property
    def n_steps(self) -> int:
        return self.time_steps

    def initial_state(self, wealth: float) -> PortfolioState:
        return PortfolioState(
            wealth=float(wealth),
            prices=(1.0,),
            allocations=(0.0,),
        )


# ---------------------------------------------------------------------------
# Concrete policies used in tests
# ---------------------------------------------------------------------------


class AllCashPolicy(Policy):
    """Always places everything in the risk-free account (alloc = 0.0)."""

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        return AllocationAction(allocations=(0.0,))


class TenPercentRiskyPolicy(Policy):
    """Always targets 10% in the risky asset (within REBALANCE_LIMIT)."""

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        return AllocationAction(allocations=(0.1,))


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

R = 0.02
MU_DET = 0.10     # deterministic risky return (sigma=0)
GAMMA = 2.0


def make_mdp(
    sigma: float = 0.0,
    time_steps: int = 3,
    gamma: float = GAMMA,
) -> ExtendedSingleAssetMDP:
    return ExtendedSingleAssetMDP(
        risky_return=NormalReturnDistribution(mu=MU_DET, sigma=sigma),
        riskless_return=ConstantRisklessReturn(rate=R),
        utility=CRRAUtility(gamma=gamma),
        action_space=ActionSpace(choices=[0.0, 0.1, 0.5, 1.0], n_assets=1),
        time_steps=time_steps,
    )


# ---------------------------------------------------------------------------
# TestPortfolioSimulatorTrajectory
# ---------------------------------------------------------------------------


class TestPortfolioSimulatorTrajectory:
    def test_simulate_path_length_equals_n_steps(self):
        mdp = make_mdp(time_steps=4)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        assert len(path) == 4

    def test_simulate_path_single_step(self):
        mdp = make_mdp(time_steps=1)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        assert len(path) == 1

    def test_all_intermediate_rewards_are_zero(self):
        """Reward is non-zero only at the terminal step."""
        mdp = make_mdp(time_steps=4)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        for _, _, reward in path[:-1]:
            assert reward == pytest.approx(0.0)

    def test_terminal_reward_is_nonzero(self):
        mdp = make_mdp(time_steps=3)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        _, _, terminal_reward = path[-1]
        assert terminal_reward != pytest.approx(0.0)

    def test_terminal_reward_is_finite(self):
        mdp = make_mdp(time_steps=3)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        _, _, terminal_reward = path[-1]
        assert math.isfinite(terminal_reward)

    def test_each_element_is_state_action_reward_tuple(self):
        mdp = make_mdp(time_steps=2)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        for element in path:
            state, action, reward = element
            assert isinstance(state, PortfolioState)
            assert isinstance(action, AllocationAction)
            assert isinstance(reward, float)

    def test_states_have_increasing_wealth_under_all_cash_deterministic(self):
        """With sigma=0 and all-cash policy, wealth grows monotonically at rate r."""
        mdp = make_mdp(sigma=0.0, time_steps=4)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        path = sim.simulate_path()
        wealths = [step[0].wealth for step in path]
        for i in range(1, len(wealths)):
            assert wealths[i] == pytest.approx(wealths[i - 1] * (1.0 + R))


# ---------------------------------------------------------------------------
# TestSimulateMany
# ---------------------------------------------------------------------------


class TestSimulateMany:
    def test_simulate_many_returns_correct_count(self):
        mdp = make_mdp()
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        paths = sim.simulate_many(num_paths=7)
        assert len(paths) == 7

    def test_each_path_has_correct_length(self):
        n_steps = 5
        mdp = make_mdp(time_steps=n_steps)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        paths = sim.simulate_many(num_paths=4)
        for path in paths:
            assert len(path) == n_steps

    def test_simulate_many_one_path(self):
        mdp = make_mdp()
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        paths = sim.simulate_many(num_paths=1)
        assert len(paths) == 1

    def test_invalid_num_paths_raises(self):
        mdp = make_mdp()
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        with pytest.raises(ValueError):
            sim.simulate_many(num_paths=0)


# ---------------------------------------------------------------------------
# TestExpectedTerminalUtility
# ---------------------------------------------------------------------------


class TestExpectedTerminalUtility:
    def test_expected_terminal_utility_is_finite(self):
        mdp = make_mdp(sigma=0.05)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        result = sim.expected_terminal_utility(num_paths=20)
        assert math.isfinite(result)

    def test_deterministic_utility_exact(self):
        """
        With sigma=0, all-cash policy, and T=2:
            W_T = 1.0 * (1.02)^2 = 1.0404
            U(W_T) = W_T^(1-2) / (1-2) = -1 / W_T
        E[U] must equal this exactly, regardless of num_paths.
        """
        mdp = make_mdp(sigma=0.0, time_steps=2, gamma=2.0)
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        result = sim.expected_terminal_utility(num_paths=5)
        terminal_wealth = 1.0 * ((1.0 + R) ** 2)
        expected = CRRAUtility(2.0).evaluate(terminal_wealth)
        assert result == pytest.approx(expected)

    def test_higher_risky_allocation_changes_utility(self):
        """Deterministically: 10% risky → different W_T → different E[U]."""
        mdp_cash = make_mdp(sigma=0.0, time_steps=2)
        mdp_risky = make_mdp(sigma=0.0, time_steps=2)
        sim_cash = PortfolioSimulator(mdp_cash, AllCashPolicy(), initial_wealth=1.0)
        sim_risky = PortfolioSimulator(mdp_risky, TenPercentRiskyPolicy(), initial_wealth=1.0)
        u_cash = sim_cash.expected_terminal_utility(num_paths=1)
        u_risky = sim_risky.expected_terminal_utility(num_paths=1)
        # Both are finite and different (MU > R, so 10% risky gives more wealth)
        assert math.isfinite(u_cash)
        assert math.isfinite(u_risky)
        assert u_cash != pytest.approx(u_risky)

    def test_expected_utility_invalid_paths_raises(self):
        mdp = make_mdp()
        sim = PortfolioSimulator(mdp, AllCashPolicy(), initial_wealth=1.0)
        with pytest.raises(ValueError):
            sim.expected_terminal_utility(num_paths=0)


# ---------------------------------------------------------------------------
# TestMertonPolicySimulation
# ---------------------------------------------------------------------------


class TestMertonPolicySimulation:
    """
    Parameters chosen so that the Merton fraction π* = 0.10, which sits
    exactly at the REBALANCE_LIMIT threshold and is therefore feasible from
    every state whose current allocation is 0.0 or 0.10.

        π* = (μ - r) / (σ² · γ) = (0.04 - 0.02) / (0.04 · 5) = 0.10
    """

    MU_M, R_M, SIGMA_M, GAMMA_M = 0.04, 0.02, 0.20, 5.0

    def _merton_mdp(self, time_steps: int = 3) -> ExtendedSingleAssetMDP:
        return ExtendedSingleAssetMDP(
            risky_return=NormalReturnDistribution(mu=self.MU_M, sigma=0.0),
            riskless_return=ConstantRisklessReturn(rate=self.R_M),
            utility=CRRAUtility(gamma=self.GAMMA_M),
            action_space=ActionSpace(choices=[0.0, 0.1], n_assets=1),
            time_steps=time_steps,
        )

    def test_merton_policy_fraction_value(self):
        policy = AnalyticalMertonPolicy(
            mu=self.MU_M, r=self.R_M, sigma=self.SIGMA_M, gamma=self.GAMMA_M
        )
        assert policy.optimal_fraction() == pytest.approx(0.10)

    def test_merton_simulation_path_runs(self):
        mdp = self._merton_mdp()
        policy = AnalyticalMertonPolicy(
            mu=self.MU_M, r=self.R_M, sigma=self.SIGMA_M, gamma=self.GAMMA_M
        )
        sim = PortfolioSimulator(mdp, policy, initial_wealth=1.0)
        path = sim.simulate_path()
        assert len(path) == 3

    def test_merton_terminal_reward_is_finite(self):
        mdp = self._merton_mdp()
        policy = AnalyticalMertonPolicy(
            mu=self.MU_M, r=self.R_M, sigma=self.SIGMA_M, gamma=self.GAMMA_M
        )
        sim = PortfolioSimulator(mdp, policy, initial_wealth=1.0)
        path = sim.simulate_path()
        _, _, terminal_reward = path[-1]
        assert math.isfinite(terminal_reward)

    def test_merton_allocations_constant_across_steps(self):
        """π* is constant, so every action in the path should have alloc[0] ≈ 0.10."""
        mdp = self._merton_mdp()
        policy = AnalyticalMertonPolicy(
            mu=self.MU_M, r=self.R_M, sigma=self.SIGMA_M, gamma=self.GAMMA_M
        )
        sim = PortfolioSimulator(mdp, policy, initial_wealth=1.0)
        path = sim.simulate_path()
        for _, action, _ in path:
            assert action.allocations[0] == pytest.approx(0.10)