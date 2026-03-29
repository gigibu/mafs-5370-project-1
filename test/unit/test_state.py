import os
import sys
import math
import random
import pytest
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import state
from state import (
    PortfolioState,
    AllocationAction,
    ActionSpace,
    MAX_ASSETS,
    REBALANCE_LIMIT,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

FINE_GRID = [round(x * 0.1, 10) for x in range(11)]   # [0.0, 0.1, ..., 1.0]
COARSE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_state(
    n_assets: int,
    wealth: float = 1.0,
    price: float = 1.0,
    alloc: float = 0.0,
) -> PortfolioState:
    """Create a uniform PortfolioState with identical prices and allocations."""
    return PortfolioState(
        wealth=wealth,
        prices=tuple([price] * n_assets),
        allocations=tuple([alloc] * n_assets),
    )


def make_action(n_assets: int, alloc: float) -> AllocationAction:
    """Create a uniform AllocationAction with identical allocations."""
    return AllocationAction(allocations=tuple([alloc] * n_assets))


def apply_returns(
    state: PortfolioState,
    action: AllocationAction,
    asset_returns: tuple[float, ...],
    risk_free_rate: float = 0.02,
) -> PortfolioState:
    """
    Simple one-period portfolio transition used in multi-step scenario tests.

    The portfolio return is:
        R_p = Σ θᵢ * Rᵢ + (1 - Σθᵢ) * r

    where Rᵢ = asset_returns[i] (gross return, e.g. 1.05 = +5%)
    and r = risk_free_rate (per-period, e.g. 0.02 = 2%).

    New prices: X(t+1) = X(t) * Rᵢ
    New allocations: carried forward from the chosen action (rebalance happens
    at end of period, before observation of new prices).
    """
    risky_fraction = sum(action.allocations)
    cash_fraction = 1.0 - risky_fraction
    portfolio_return = (
        sum(a * r for a, r in zip(action.allocations, asset_returns))
        + cash_fraction * (1.0 + risk_free_rate)
    )
    new_wealth = state.wealth * portfolio_return
    new_prices = tuple(
        p * r for p, r in zip(state.prices, asset_returns)
    )
    return PortfolioState(
        wealth=new_wealth,
        prices=new_prices,
        allocations=action.allocations,
    )


# ===========================================================================
# Module smoke test
# ===========================================================================


def test_import_state():
    assert state is not None


def test_constants_exported():
    assert MAX_ASSETS == 4
    assert REBALANCE_LIMIT == pytest.approx(0.10)


# ===========================================================================
# PortfolioState — valid construction
# ===========================================================================


class TestPortfolioStateValid:
    def test_single_asset_unit_initial_state(self):
        """Canonical N=1 starting state."""
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        assert s.wealth == pytest.approx(1.0)
        assert s.prices == (1.0,)
        assert s.allocations == (0.0,)

    def test_four_asset_initial_state(self):
        """Canonical N=4 starting state: all prices 1.0, no allocation yet."""
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.0, 0.0, 0.0, 0.0),
        )
        assert s.n_assets == 4

    def test_fully_invested_single_asset(self):
        """100% in risky asset is valid (sum of allocations == 1)."""
        s = PortfolioState(wealth=10.0, prices=(1.2,), allocations=(1.0,))
        assert s.cash_fraction == pytest.approx(0.0)

    def test_mixed_allocation(self):
        s = PortfolioState(
            wealth=50.0,
            prices=(1.0, 1.1, 0.9),
            allocations=(0.3, 0.3, 0.2),
        )
        assert s.cash_fraction == pytest.approx(0.2)
        assert s.n_assets == 3

    def test_wealth_can_be_large(self):
        s = PortfolioState(wealth=1e7, prices=(1.0,), allocations=(0.5,))
        assert s.wealth == pytest.approx(1e7)

    def test_prices_can_vary_widely(self):
        """Prices drift far from 1.0 after many steps — still valid."""
        s = PortfolioState(
            wealth=3.0,
            prices=(0.01, 15.7, 3.3, 0.5),
            allocations=(0.1, 0.1, 0.1, 0.1),
        )
        assert s.n_assets == 4


# ===========================================================================
# PortfolioState — invalid construction
# ===========================================================================


class TestPortfolioStateInvalid:
    def test_zero_wealth_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            PortfolioState(wealth=0.0, prices=(1.0,), allocations=(0.0,))

    def test_negative_wealth_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            PortfolioState(wealth=-1.0, prices=(1.0,), allocations=(0.0,))

    def test_zero_price_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            PortfolioState(wealth=1.0, prices=(0.0,), allocations=(0.0,))

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            PortfolioState(wealth=1.0, prices=(-0.5,), allocations=(0.0,))

    def test_negative_allocation_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            PortfolioState(wealth=1.0, prices=(1.0,), allocations=(-0.1,))

    def test_over_leveraged_raises(self):
        """sum(allocations) > 1.0 must be rejected."""
        with pytest.raises(ValueError, match="1.0"):
            PortfolioState(wealth=1.0, prices=(1.0,), allocations=(1.1,))

    def test_over_leveraged_multi_asset_raises(self):
        with pytest.raises(ValueError):
            PortfolioState(
                wealth=1.0,
                prices=(1.0, 1.0),
                allocations=(0.6, 0.5),  # sum = 1.1
            )

    def test_too_many_assets_raises(self):
        """N=5 exceeds MAX_ASSETS=4 and must be rejected."""
        with pytest.raises(ValueError, match=str(MAX_ASSETS)):
            PortfolioState(
                wealth=1.0,
                prices=(1.0, 1.0, 1.0, 1.0, 1.0),
                allocations=(0.0, 0.0, 0.0, 0.0, 0.0),
            )

    def test_zero_assets_raises(self):
        with pytest.raises(ValueError):
            PortfolioState(wealth=1.0, prices=(), allocations=())

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            PortfolioState(
                wealth=1.0,
                prices=(1.0, 1.0),
                allocations=(0.5,),
            )

    def test_frozen_prevents_mutation(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.5,))
        with pytest.raises((AttributeError, TypeError)):
            s.wealth = 2.0  # type: ignore[misc]


# ===========================================================================
# PortfolioState — properties
# ===========================================================================


class TestPortfolioStateProperties:
    def test_n_assets_single(self):
        assert make_state(1).n_assets == 1

    def test_n_assets_four(self):
        assert make_state(4).n_assets == 4

    def test_cash_fraction_all_cash(self):
        s = make_state(2, alloc=0.0)
        assert s.cash_fraction == pytest.approx(1.0)

    def test_cash_fraction_fully_invested(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(1.0,))
        assert s.cash_fraction == pytest.approx(0.0)

    def test_cash_fraction_partial(self):
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0),
            allocations=(0.2, 0.3, 0.1),
        )
        assert s.cash_fraction == pytest.approx(0.4)

    def test_state_is_hashable(self):
        """Frozen dataclass must be usable as a dict key."""
        s = make_state(1)
        d = {s: 42}
        assert d[s] == 42

    def test_two_identical_states_are_equal(self):
        s1 = make_state(2, wealth=5.0, price=1.1, alloc=0.3)
        s2 = make_state(2, wealth=5.0, price=1.1, alloc=0.3)
        assert s1 == s2

    def test_two_different_states_are_not_equal(self):
        s1 = make_state(1, wealth=1.0)
        s2 = make_state(1, wealth=2.0)
        assert s1 != s2


# ===========================================================================
# AllocationAction — valid construction
# ===========================================================================


class TestAllocationActionValid:
    def test_zero_allocation(self):
        a = AllocationAction(allocations=(0.0,))
        assert a.cash_fraction == pytest.approx(1.0)

    def test_full_single_asset(self):
        a = AllocationAction(allocations=(1.0,))
        assert a.cash_fraction == pytest.approx(0.0)

    def test_partial_multi_asset(self):
        a = AllocationAction(allocations=(0.3, 0.3, 0.2))
        assert a.cash_fraction == pytest.approx(0.2)
        assert a.n_assets == 3

    def test_four_asset_action(self):
        a = AllocationAction(allocations=(0.25, 0.25, 0.25, 0.25))
        assert a.n_assets == 4
        assert a.cash_fraction == pytest.approx(0.0)

    def test_action_is_frozen_hashable(self):
        a = AllocationAction(allocations=(0.5,))
        d = {a: "stored"}
        assert d[a] == "stored"


# ===========================================================================
# AllocationAction — invalid construction
# ===========================================================================


class TestAllocationActionInvalid:
    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError):
            AllocationAction(allocations=())

    def test_negative_allocation_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            AllocationAction(allocations=(-0.1,))

    def test_frozen_prevents_mutation(self):
        a = AllocationAction(allocations=(0.5,))
        with pytest.raises((AttributeError, TypeError)):
            a.allocations = (0.6,)  # type: ignore[misc]


# ===========================================================================
# ActionSpace — construction and validation
# ===========================================================================


class TestActionSpaceConstruction:
    def test_single_asset_fine_grid(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        assert space.n_assets == 1
        assert len(space) == 11  # [0.0, 0.1, ..., 1.0]

    def test_deduplicated_choices(self):
        space = ActionSpace([0.0, 0.5, 0.5, 1.0], n_assets=1)
        assert len(space.get_choices()) == 3

    def test_sorted_choices(self):
        space = ActionSpace([1.0, 0.0, 0.5], n_assets=1)
        assert space.get_choices() == pytest.approx([0.0, 0.5, 1.0])

    def test_two_asset_sum_constraint(self):
        """With choices [0.0, 0.5, 1.0] and N=2, valid combos have sum ≤ 1."""
        space = ActionSpace([0.0, 0.5, 1.0], n_assets=2)
        for action in space.get_all_actions():
            assert sum(action.allocations) <= 1.0 + 1e-9

    def test_four_asset_all_actions_obey_sum_constraint(self):
        space = ActionSpace(COARSE_GRID, n_assets=4)
        for action in space.get_all_actions():
            assert sum(action.allocations) <= 1.0 + 1e-9

    def test_n_assets_five_raises(self):
        """N=5 exceeds MAX_ASSETS=4."""
        with pytest.raises(ValueError, match=str(MAX_ASSETS)):
            ActionSpace(FINE_GRID, n_assets=5)

    def test_n_assets_zero_raises(self):
        with pytest.raises(ValueError):
            ActionSpace(FINE_GRID, n_assets=0)

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ActionSpace([], n_assets=1)

    def test_out_of_range_choice_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ActionSpace([0.0, 1.1], n_assets=1)

    def test_negative_choice_raises(self):
        with pytest.raises(ValueError):
            ActionSpace([-0.1, 0.5], n_assets=1)

    def test_repr_contains_n_assets(self):
        space = ActionSpace(COARSE_GRID, n_assets=2)
        assert "n_assets=2" in repr(space)


# ===========================================================================
# ActionSpace — get_choices / get_all_actions
# ===========================================================================


class TestActionSpaceGetters:
    def test_get_choices_returns_copy(self):
        """Mutation of the returned list must not affect the ActionSpace."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        choices = space.get_choices()
        choices.append(99.9)
        assert 99.9 not in space.get_choices()

    def test_get_all_actions_returns_copy(self):
        space = ActionSpace(COARSE_GRID, n_assets=1)
        actions = space.get_all_actions()
        original_len = len(actions)
        actions.clear()
        assert len(space.get_all_actions()) == original_len

    def test_all_actions_are_allocation_action_instances(self):
        space = ActionSpace(COARSE_GRID, n_assets=2)
        for a in space.get_all_actions():
            assert isinstance(a, AllocationAction)

    def test_all_actions_have_correct_n_assets(self):
        for n in range(1, MAX_ASSETS + 1):
            space = ActionSpace(COARSE_GRID, n_assets=n)
            for a in space.get_all_actions():
                assert a.n_assets == n


# ===========================================================================
# ActionSpace — is_valid
# ===========================================================================


class TestActionSpaceIsValid:
    def test_hold_action_is_always_valid(self):
        """An action that exactly replicates the current allocation is always valid."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        for alloc in [0.0, 0.3, 0.5, 1.0]:
            s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(alloc,))
            a = AllocationAction(allocations=(alloc,))
            assert space.is_valid(a, s), f"Hold action failed at alloc={alloc}"

    def test_increase_within_limit_is_valid(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.4,))
        a = AllocationAction(allocations=(0.5,))  # +0.10 == REBALANCE_LIMIT
        assert ActionSpace(FINE_GRID, n_assets=1).is_valid(a, s)

    def test_decrease_within_limit_is_valid(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.4,))
        a = AllocationAction(allocations=(0.3,))  # -0.10 == REBALANCE_LIMIT
        assert ActionSpace(FINE_GRID, n_assets=1).is_valid(a, s)

    def test_increase_beyond_limit_is_invalid(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.4,))
        a = AllocationAction(allocations=(0.51,))  # +0.11 > REBALANCE_LIMIT
        assert not ActionSpace([0.0, 0.25, 0.51, 1.0], n_assets=1).is_valid(a, s)

    def test_decrease_beyond_limit_is_invalid(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.4,))
        a = AllocationAction(allocations=(0.29,))  # -0.11 > REBALANCE_LIMIT
        assert not ActionSpace([0.0, 0.29, 1.0], n_assets=1).is_valid(a, s)

    def test_wrong_dimension_is_invalid(self):
        """Action with 2 assets is invalid from a 1-asset state."""
        space = ActionSpace(FINE_GRID, n_assets=2)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        a = AllocationAction(allocations=(0.0, 0.0))
        assert not space.is_valid(a, s)

    def test_multi_asset_all_within_limit(self):
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.2, 0.1),
        )
        a = AllocationAction(allocations=(0.3, 0.1, 0.2, 0.1))  # each ±0.1
        assert space.is_valid(a, s)

    def test_multi_asset_one_exceeds_limit(self):
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.2, 0.1),
        )
        a = AllocationAction(allocations=(0.31, 0.2, 0.2, 0.1))  # asset 0: +0.11
        assert not space.is_valid(a, s)

# ===========================================================================
# ActionSpace — feasible_actions
# ===========================================================================


class TestFeasibleActions:
    def test_feasible_set_is_subset_of_all_actions(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = make_state(1, alloc=0.5)
        feasible = space.feasible_actions(s)
        all_actions = set(map(lambda a: a.allocations, space.get_all_actions()))
        for a in feasible:
            assert a.allocations in all_actions

    def test_feasible_set_contains_hold_action(self):
        """The current allocation is always in the feasible set."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        for alloc in [0.0, 0.3, 0.5, 0.8, 1.0]:
            s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(alloc,))
            feasible = space.feasible_actions(s)
            assert any(a.allocations == (alloc,) for a in feasible), (
                f"Hold action missing from feasible set at alloc={alloc}"
            )

    def test_feasible_set_respects_rebalance_limit(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.5,))
        for a in space.feasible_actions(s):
            assert abs(a.allocations[0] - 0.5) <= REBALANCE_LIMIT + 1e-9

    def test_feasible_set_at_zero_allocation(self):
        """From all-cash state, agent can only increase allocations by up to 10%."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = make_state(1, alloc=0.0)
        feasible_fracs = sorted(a.allocations[0] for a in space.feasible_actions(s))
        assert max(feasible_fracs) <= REBALANCE_LIMIT + 1e-9

    def test_feasible_set_at_full_allocation(self):
        """From fully invested state, agent can only decrease by up to 10%."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(1.0,))
        feasible_fracs = sorted(a.allocations[0] for a in space.feasible_actions(s))
        assert min(feasible_fracs) >= 1.0 - REBALANCE_LIMIT - 1e-9

    def test_feasible_set_four_assets(self):
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.1, 0.1),
        )
        for a in space.feasible_actions(s):
            for new, cur in zip(a.allocations, s.allocations):
                assert abs(new - cur) <= REBALANCE_LIMIT + 1e-9
            assert sum(a.allocations) <= 1.0 + 1e-9


# ===========================================================================
# ActionSpace — sampling
# ===========================================================================


class TestActionSpaceSampling:
    def test_sample_uniform_returns_valid_action(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        a = space.sample_uniform()
        assert isinstance(a, AllocationAction)
        assert a in space.get_all_actions()

    def test_sample_feasible_returns_feasible_action(self):
        space = ActionSpace(FINE_GRID, n_assets=2)
        s = make_state(2, alloc=0.3)
        a = space.sample_feasible(s)
        assert space.is_valid(a, s)

    def test_sample_feasible_distribution(self):
        """Sample 200 times and verify all sampled actions are feasible."""
        random.seed(42)
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.5,))
        for _ in range(200):
            a = space.sample_feasible(s)
            assert space.is_valid(a, s)


# ===========================================================================
# Scenario: N=1, T=1 (single asset, single period)
# ===========================================================================


class TestScenarioN1T1:
    """
    Minimal portfolio: one risky asset, one period.
    Validates the fundamental rebalancing mechanic on a single transition.
    """

    def test_initial_state_valid(self):
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        assert s.n_assets == 1
        assert s.wealth == pytest.approx(1.0)

    def test_feasible_actions_from_cash_state(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        feasible = space.feasible_actions(s)
        # From θ=0 we can move to θ ∈ {0.0, 0.1} (Δ ≤ 0.1)
        fracs = [a.allocations[0] for a in feasible]
        assert pytest.approx(max(fracs)) == REBALANCE_LIMIT
        assert pytest.approx(min(fracs)) == 0.0

    def test_single_period_transition_wealth_grows(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        s0 = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        a = AllocationAction(allocations=(0.1,))
        assert space.is_valid(a, s0)
        s1 = apply_returns(s0, a, asset_returns=(1.10,), risk_free_rate=0.02)
        # 10% in risky (+10% return) + 90% in cash (+2%) = 0.10*1.10 + 0.90*1.02 = 1.028
        assert s1.wealth == pytest.approx(1.028)
        assert s1.prices == pytest.approx((1.10,))

    def test_single_period_action_is_recorded_in_state(self):
        s0 = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        a = AllocationAction(allocations=(0.1,))
        s1 = apply_returns(s0, a, asset_returns=(1.0,))
        assert s1.allocations == (0.1,)

    def test_infeasible_action_rejected(self):
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        a = AllocationAction(allocations=(0.2,))  # Δ = 0.2 > 0.1
        assert not space.is_valid(a, s)


# ===========================================================================
# Scenario: N=1, T=10 (single asset, full 10-period horizon)
# ===========================================================================


class TestScenarioN1T10:
    """
    Single risky asset over a 10-period horizon.
    Tests that states chain correctly and the rebalancing constraint is
    enforced at every step.
    """

    HORIZON = 10
    GROSS_RETURN = 1.05   # +5% per period
    RISK_FREE    = 0.02   # +2% per period
    SPACE        = ActionSpace(FINE_GRID, n_assets=1)

    def _run_trajectory(
        self,
        initial_alloc: float,
        delta_per_step: float,
        gross_return: float = GROSS_RETURN,
    ) -> list[PortfolioState]:
        """Simulate a deterministic trajectory; each step shifts allocation by delta."""
        states = []
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(initial_alloc,))
        states.append(s)
        for _ in range(self.HORIZON):
            cur = s.allocations[0]
            new_alloc = round(min(max(cur + delta_per_step, 0.0), 1.0), 10)
            a = AllocationAction(allocations=(new_alloc,))
            if self.SPACE.is_valid(a, s):
                s = apply_returns(s, a, asset_returns=(gross_return,), risk_free_rate=self.RISK_FREE)
                states.append(s)
        return states

    def test_trajectory_produces_t_plus_one_states(self):
        trajectory = self._run_trajectory(0.0, delta_per_step=0.1)
        assert len(trajectory) == self.HORIZON + 1

    def test_wealth_monotonically_increases_with_positive_returns(self):
        trajectory = self._run_trajectory(0.5, delta_per_step=0.0)
        for i in range(1, len(trajectory)):
            assert trajectory[i].wealth > trajectory[i - 1].wealth

    def test_prices_compound_correctly(self):
        """After T steps with constant gross_return r, price == r^T."""
        trajectory = self._run_trajectory(0.5, delta_per_step=0.0)
        for t, s in enumerate(trajectory):
            expected_price = self.GROSS_RETURN ** t
            assert s.prices[0] == pytest.approx(expected_price, rel=1e-6)

    def test_rebalancing_constraint_enforced_every_step(self):
        """Every consecutive state pair must satisfy the per-step rebalancing limit."""
        trajectory = self._run_trajectory(0.0, delta_per_step=0.1)
        for prev, curr in zip(trajectory[:-1], trajectory[1:]):
            delta = abs(curr.allocations[0] - prev.allocations[0])
            assert delta <= REBALANCE_LIMIT + 1e-9

    def test_allocation_cannot_jump_more_than_limit(self):
        """Attempting a jump of 0.2 must be blocked."""
        space = ActionSpace(FINE_GRID, n_assets=1)
        s = PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))
        a = AllocationAction(allocations=(0.2,))
        assert not space.is_valid(a, s)

    def test_full_investment_reachable_in_10_steps(self):
        """Starting at 0 and increasing by 0.1/step, θ=1.0 is reached exactly at t=10."""
        trajectory = self._run_trajectory(0.0, delta_per_step=0.1)
        final_alloc = trajectory[-1].allocations[0]
        assert final_alloc == pytest.approx(1.0)

    def test_state_is_always_valid_throughout_trajectory(self):
        trajectory = self._run_trajectory(0.5, delta_per_step=0.0)
        for s in trajectory:
            assert s.wealth > 0
            assert 0.0 <= s.cash_fraction <= 1.0 + 1e-9
            assert all(0.0 <= a <= 1.0 for a in s.allocations)


# ===========================================================================
# Scenario: N=4, T=1 (maximum valid assets, single period)
# ===========================================================================


class TestScenarioN4T1:
    """
    Four risky assets, one period.
    Validates multi-asset state construction, action validity, and the
    per-asset rebalancing constraint.
    """

    def test_four_asset_initial_state(self):
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.0, 0.0, 0.0, 0.0),
        )
        assert s.n_assets == 4
        assert s.cash_fraction == pytest.approx(1.0)

    def test_equal_weight_initial_allocation(self):
        """25%/25%/25%/25% is a common benchmark starting allocation."""
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.25, 0.25, 0.25, 0.25),
        )
        assert s.cash_fraction == pytest.approx(0.0)

    def test_action_space_size_four_assets(self):
        """
        With choices [0.0, 0.1, ..., 1.0] (11 values) and N=4 assets,
        the number of sum-feasible actions is C(14, 4) = 1001.
        """
        space = ActionSpace(FINE_GRID, n_assets=4)
        assert len(space) == 1001

    def test_per_asset_rebalancing_in_multi_asset(self):
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.2, 0.1),
        )
        # Move each asset by exactly the limit
        a_valid = AllocationAction(allocations=(0.3, 0.1, 0.2, 0.1))
        a_invalid = AllocationAction(allocations=(0.31, 0.2, 0.2, 0.1))
        assert space.is_valid(a_valid, s)
        assert not space.is_valid(a_invalid, s)

    def test_single_period_transition_four_assets(self):
        s0 = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.1, 0.1, 0.1, 0.1),
        )
        a = AllocationAction(allocations=(0.2, 0.1, 0.1, 0.1))  # asset 0: +0.1
        returns = (1.10, 1.05, 1.03, 0.95)
        s1 = apply_returns(s0, a, asset_returns=returns, risk_free_rate=0.02)
        # Portfolio return: 0.2*1.10 + 0.1*1.05 + 0.1*1.03 + 0.1*0.95 + 0.5*1.02
        expected = 0.2*1.10 + 0.1*1.05 + 0.1*1.03 + 0.1*0.95 + 0.5*1.02
        assert s1.wealth == pytest.approx(expected)
        assert s1.n_assets == 4

    def test_feasible_actions_non_empty_four_assets(self):
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0,) * 4,
            allocations=(0.2, 0.2, 0.1, 0.1),
        )
        assert len(space.feasible_actions(s)) > 0


# ===========================================================================
# Scenario: N=4, T=10 (maximum assets, full 10-period horizon)
# ===========================================================================


class TestScenarioN4T10:
    """
    Four risky assets over a 10-period horizon.
    Simulates the full problem scope, verifying that the machinery
    remains consistent over repeated transitions.
    """

    HORIZON      = 10
    RISK_FREE    = 0.02
    ASSET_RETURNS = (1.07, 1.05, 1.03, 1.01)  # deterministic for reproducibility

    def _run_trajectory(self) -> list[PortfolioState]:
        """
        Simulate a 10-step trajectory: start fully in cash, increase
        the first two assets by 0.05 each per step while holding the others.
        """
        space = ActionSpace(FINE_GRID, n_assets=4)
        s = PortfolioState(
            wealth=1.0,
            prices=(1.0, 1.0, 1.0, 1.0),
            allocations=(0.0, 0.0, 0.0, 0.0),
        )
        states = [s]
        step = 0.05  # move two assets by 5pp per period (within 10pp limit)
        for t in range(self.HORIZON):
            cur = s.allocations
            new_0 = round(min(cur[0] + step, 1.0), 10)
            new_1 = round(min(cur[1] + step, 1.0), 10)
            a = AllocationAction(allocations=(new_0, new_1, cur[2], cur[3]))
            assert space.is_valid(a, s), f"Action invalid at step {t}"
            s = apply_returns(s, a, asset_returns=self.ASSET_RETURNS, risk_free_rate=self.RISK_FREE)
            states.append(s)
        return states

    def test_trajectory_length(self):
        assert len(self._run_trajectory()) == self.HORIZON + 1

    def test_wealth_grows_throughout(self):
        trajectory = self._run_trajectory()
        for i in range(1, len(trajectory)):
            assert trajectory[i].wealth > trajectory[i - 1].wealth, (
                f"Wealth did not grow at step {i}"
            )

    def test_all_states_have_four_assets(self):
        for s in self._run_trajectory():
            assert s.n_assets == 4

    def test_rebalancing_constraint_every_step(self):
        trajectory = self._run_trajectory()
        for prev, curr in zip(trajectory[:-1], trajectory[1:]):
            for new_a, old_a in zip(curr.allocations, prev.allocations):
                assert abs(new_a - old_a) <= REBALANCE_LIMIT + 1e-9

    def test_prices_compound_independently(self):
        """Each price series must compound at its own gross return rate."""
        trajectory = self._run_trajectory()
        for t, s in enumerate(trajectory):
            for i, (price, gross_ret) in enumerate(zip(s.prices, self.ASSET_RETURNS)):
                expected = gross_ret ** t
                assert price == pytest.approx(expected, rel=1e-6), (
                    f"Asset {i} price mismatch at t={t}"
                )

    def test_all_states_valid_throughout(self):
        for s in self._run_trajectory():
            assert s.wealth > 0
            assert 0.0 <= sum(s.allocations) <= 1.0 + 1e-9
            assert all(p > 0 for p in s.prices)


# ===========================================================================
# Scenario: N=5 — exceeds MAX_ASSETS, must be rejected
# ===========================================================================


class TestScenarioN5Rejected:
    """
    The problem specification caps the portfolio at 4 risky assets.
    Every entry point that would create an N=5 state or action space must
    raise a ValueError immediately, before any computation proceeds.
    """

    def test_portfolio_state_five_prices_raises(self):
        with pytest.raises(ValueError, match=str(MAX_ASSETS)):
            PortfolioState(
                wealth=1.0,
                prices=(1.0,) * 5,
                allocations=(0.0,) * 5,
            )

    def test_action_space_five_assets_raises(self):
        with pytest.raises(ValueError, match=str(MAX_ASSETS)):
            ActionSpace(FINE_GRID, n_assets=5)

    def test_error_message_references_max_assets(self):
        """Error messages must be actionable — they must name the limit."""
        with pytest.raises(ValueError) as exc_info:
            PortfolioState(
                wealth=1.0,
                prices=(1.0,) * 5,
                allocations=(0.0,) * 5,
            )
        assert str(MAX_ASSETS) in str(exc_info.value)

    def test_six_assets_also_rejected(self):
        with pytest.raises(ValueError):
            PortfolioState(
                wealth=1.0,
                prices=(1.0,) * 6,
                allocations=(0.0,) * 6,
            )
        with pytest.raises(ValueError):
            ActionSpace(FINE_GRID, n_assets=6)