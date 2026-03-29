"""
Unit tests for policy.py — GreedyQPolicy, RandomPolicy, AnalyticalMertonPolicy.

Test philosophy
---------------
* GreedyQPolicy tests verify argmax selection, correct QVF dispatch per
  time step, and boundary handling.
* RandomPolicy tests verify delegation to action_space.sample() and
  independence from state and time.
* AnalyticalMertonPolicy tests verify the closed-form formula with
  hand-computed expected values, monotonicity properties, and correct
  propagation into AllocationAction.
* All three classes are tested against the Policy ABC contract via a
  shared parametrised fixture.
"""
from __future__ import annotations

import math
import sys
import numpy as np
import pytest
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from approximator import QValueApproximator
from policy import (
    AnalyticalMertonPolicy,
    GreedyQPolicy,
    Policy,
    RandomPolicy,
)
from state import AllocationAction, PortfolioState


# ===========================================================================
# Concrete QValueApproximator stubs
# ===========================================================================


class ConstantQVF(QValueApproximator):
    """Returns a fixed constant for every (state, action) pair."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        return float(self._value)

    def update(self, samples):
        return self

    def copy(self):
        return ConstantQVF(self._value)


class MappedQVF(QValueApproximator):
    """
    Returns Q values from a dict keyed by action.allocations tuple.
    Raises KeyError if an unmapped action is queried (catches test bugs).
    """

    def __init__(self, q_map: Dict[tuple, float]) -> None:
        self._q_map = q_map

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        return float(self._q_map[action.allocations])

    def update(self, samples):
        return self

    def copy(self):
        return MappedQVF(dict(self._q_map))


# ===========================================================================
# Stub action space (duck-typed; satisfies RandomPolicy's interface)
# ===========================================================================


class StubActionSpace:
    """
    Minimal duck-typed action space for RandomPolicy tests.
    Supports a fixed return value and call counting.
    """

    def __init__(
        self,
        actions: list[AllocationAction],
        rng: np.random.Generator | None = None,
    ) -> None:
        self._actions = actions
        self._rng = rng or np.random.default_rng(0)
        self.sample_call_count: int = 0
        self._fixed: AllocationAction | None = None

    def sample(self) -> AllocationAction:
        self.sample_call_count += 1
        if self._fixed is not None:
            return self._fixed
        idx = int(self._rng.integers(len(self._actions)))
        return self._actions[idx]

    def __repr__(self) -> str:
        return f"StubActionSpace(n={len(self._actions)})"


# ===========================================================================
# Helpers
# ===========================================================================


def make_state(wealth: float = 1.0, alloc: float = 0.0) -> PortfolioState:
    return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(alloc,))


def make_state_n(wealth: float = 1.0, n: int = 1) -> PortfolioState:
    return PortfolioState(
        wealth=wealth,
        prices=tuple(1.0 for _ in range(n)),
        allocations=tuple(0.0 for _ in range(n)),
    )


def make_action(alloc: float = 0.0) -> AllocationAction:
    return AllocationAction(allocations=(alloc,))


# ===========================================================================
# GreedyQPolicy — constructor
# ===========================================================================


class TestGreedyQPolicyConstructor:
    def test_empty_qvf_per_step_raises(self):
        with pytest.raises(ValueError, match="qvf_per_step"):
            GreedyQPolicy([], [make_action()])

    def test_empty_feasible_actions_raises(self):
        with pytest.raises(ValueError, match="feasible_actions"):
            GreedyQPolicy([ConstantQVF()], [])

    def test_stores_qvfs(self):
        qvf = ConstantQVF(1.0)
        policy = GreedyQPolicy([qvf], [make_action()])
        assert policy._qvf_per_step == [qvf]

    def test_stores_feasible_actions(self):
        actions = [make_action(0.0), make_action(0.5)]
        policy = GreedyQPolicy([ConstantQVF()], actions)
        assert policy._feasible_actions == actions

    def test_is_policy_instance(self):
        assert isinstance(GreedyQPolicy([ConstantQVF()], [make_action()]), Policy)

    def test_multiple_time_steps_stored(self):
        qvfs = [ConstantQVF(float(i)) for i in range(5)]
        policy = GreedyQPolicy(qvfs, [make_action()])
        assert len(policy._qvf_per_step) == 5

    def test_sequences_converted_to_list(self):
        """Passing tuples as sequences must work."""
        policy = GreedyQPolicy(
            (ConstantQVF(),),
            (make_action(0.0), make_action(1.0)),
        )
        assert isinstance(policy._qvf_per_step, list)
        assert isinstance(policy._feasible_actions, list)


# ===========================================================================
# GreedyQPolicy — get_action
# ===========================================================================


class TestGreedyQPolicyGetAction:
    def test_single_action_always_returned(self):
        """With only one feasible action it is always selected."""
        action = make_action(0.3)
        policy = GreedyQPolicy([ConstantQVF(5.0)], [action])
        assert policy.get_action(make_state(), t=0) == action

    def test_returns_argmax_action(self):
        """The action with the highest Q value must be returned."""
        a_low = make_action(0.0)
        a_mid = make_action(0.5)
        a_high = make_action(1.0)
        q_map = {(0.0,): 1.0, (0.5,): 3.0, (1.0,): 2.0}
        policy = GreedyQPolicy([MappedQVF(q_map)], [a_low, a_mid, a_high])
        assert policy.get_action(make_state(), t=0) == a_mid

    def test_returns_allocation_action_type(self):
        policy = GreedyQPolicy([ConstantQVF()], [make_action()])
        assert isinstance(policy.get_action(make_state(), t=0), AllocationAction)

    def test_uses_correct_qvf_at_t0(self):
        """QVF at t=0 is consulted for step 0."""
        a1 = make_action(0.2)
        a2 = make_action(0.8)
        qvf0 = MappedQVF({(0.2,): 10.0, (0.8,): 1.0})   # a1 wins
        qvf1 = MappedQVF({(0.2,): 1.0,  (0.8,): 10.0})  # a2 wins
        policy = GreedyQPolicy([qvf0, qvf1], [a1, a2])
        assert policy.get_action(make_state(), t=0) == a1

    def test_uses_correct_qvf_at_t1(self):
        """QVF at t=1 is consulted for step 1."""
        a1 = make_action(0.2)
        a2 = make_action(0.8)
        qvf0 = MappedQVF({(0.2,): 10.0, (0.8,): 1.0})
        qvf1 = MappedQVF({(0.2,): 1.0,  (0.8,): 10.0})
        policy = GreedyQPolicy([qvf0, qvf1], [a1, a2])
        assert policy.get_action(make_state(), t=1) == a2

    def test_negative_t_raises_index_error(self):
        policy = GreedyQPolicy([ConstantQVF()], [make_action()])
        with pytest.raises(IndexError):
            policy.get_action(make_state(), t=-1)

    def test_t_equal_to_n_steps_raises_index_error(self):
        policy = GreedyQPolicy([ConstantQVF()], [make_action()])
        with pytest.raises(IndexError):
            policy.get_action(make_state(), t=1)

    def test_t_far_out_of_range_raises_index_error(self):
        qvfs = [ConstantQVF() for _ in range(3)]
        policy = GreedyQPolicy(qvfs, [make_action()])
        with pytest.raises(IndexError):
            policy.get_action(make_state(), t=100)

    def test_argmax_independent_of_state_wealth(self):
        """
        When Q depends only on action (not state), the same action must be
        selected regardless of wealth.
        """
        a_best = make_action(0.7)
        a_other = make_action(0.2)
        q_map = {(0.7,): 99.0, (0.2,): 0.0}
        policy = GreedyQPolicy([MappedQVF(q_map)], [a_best, a_other])
        for w in [0.5, 1.0, 5.0, 100.0]:
            assert policy.get_action(make_state(w), t=0) == a_best

    def test_all_negative_q_values_still_finds_argmax(self):
        """argmax must work even when every Q-value is negative."""
        a_less_bad = make_action(0.4)
        a_worse = make_action(0.1)
        q_map = {(0.4,): -1.0, (0.1,): -5.0}
        policy = GreedyQPolicy([MappedQVF(q_map)], [a_less_bad, a_worse])
        assert policy.get_action(make_state(), t=0) == a_less_bad

    def test_many_time_steps_dispatch(self):
        """Policy selects the per-step argmax action correctly at every t."""
        n_steps = 10
        actions = [make_action(float(i) * 0.1) for i in range(5)]
        qvfs = []
        for t in range(n_steps):
            best = actions[t % 5].allocations
            q_map = {a.allocations: (10.0 if a.allocations == best else 0.0)
                     for a in actions}
            qvfs.append(MappedQVF(q_map))
        policy = GreedyQPolicy(qvfs, actions)
        for t in range(n_steps):
            assert policy.get_action(make_state(), t=t) == actions[t % 5]


# ===========================================================================
# GreedyQPolicy — repr
# ===========================================================================


class TestGreedyQPolicyRepr:
    def test_repr_contains_class_name(self):
        assert "GreedyQPolicy" in repr(GreedyQPolicy([ConstantQVF()], [make_action()]))

    def test_repr_contains_n_steps(self):
        policy = GreedyQPolicy([ConstantQVF(), ConstantQVF(), ConstantQVF()], [make_action()])
        assert "3" in repr(policy)

    def test_repr_contains_n_actions(self):
        actions = [make_action(float(i) * 0.1) for i in range(4)]
        policy = GreedyQPolicy([ConstantQVF()], actions)
        assert "4" in repr(policy)


# ===========================================================================
# RandomPolicy
# ===========================================================================


class TestRandomPolicy:
    def _make_policy(
        self, n_actions: int = 3
    ) -> tuple[RandomPolicy, StubActionSpace]:
        actions = [make_action(float(i) / n_actions) for i in range(n_actions)]
        space = StubActionSpace(actions, rng=np.random.default_rng(0))
        return RandomPolicy(space), space

    def test_is_policy_instance(self):
        policy, _ = self._make_policy()
        assert isinstance(policy, Policy)

    def test_get_action_calls_sample_exactly_once(self):
        policy, space = self._make_policy()
        space.sample_call_count = 0
        policy.get_action(make_state(), t=0)
        assert space.sample_call_count == 1

    def test_get_action_returns_allocation_action(self):
        policy, _ = self._make_policy()
        assert isinstance(policy.get_action(make_state(), t=0), AllocationAction)

    def test_get_action_returns_value_from_sample(self):
        """get_action must return exactly what action_space.sample() returns."""
        fixed_action = make_action(0.42)
        space = StubActionSpace([fixed_action])
        space._fixed = fixed_action
        policy = RandomPolicy(space)
        assert policy.get_action(make_state(), t=0) == fixed_action

    def test_get_action_ignores_state_wealth(self):
        """sample() is called once regardless of state wealth."""
        policy, space = self._make_policy()
        for w in [0.5, 1.0, 5.0, 50.0]:
            space.sample_call_count = 0
            policy.get_action(make_state(w), t=7)
            assert space.sample_call_count == 1

    def test_get_action_ignores_time_step(self):
        """sample() is called once regardless of t."""
        policy, space = self._make_policy()
        for t in [0, 1, 99]:
            space.sample_call_count = 0
            policy.get_action(make_state(), t=t)
            assert space.sample_call_count == 1

    def test_stores_action_space_reference(self):
        _, space = self._make_policy()
        policy = RandomPolicy(space)
        assert policy._action_space is space

    def test_repr_contains_class_name(self):
        policy, _ = self._make_policy()
        assert "RandomPolicy" in repr(policy)


# ===========================================================================
# AnalyticalMertonPolicy — constructor
# ===========================================================================


class TestAnalyticalMertonConstructor:
    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.0, gamma=2.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=-0.1, gamma=2.0)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=0.0)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=-1.0)

    def test_valid_construction_stores_params(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert p._mu    == pytest.approx(0.1)
        assert p._r     == pytest.approx(0.02)
        assert p._sigma == pytest.approx(0.2)
        assert p._gamma == pytest.approx(2.0)

    def test_is_policy_instance(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert isinstance(p, Policy)

    def test_r_greater_than_mu_is_valid(self):
        """r > mu is valid and produces a negative (short) fraction."""
        p = AnalyticalMertonPolicy(mu=0.01, r=0.05, sigma=0.2, gamma=2.0)
        assert p.optimal_fraction() < 0.0

    def test_params_stored_as_float(self):
        p = AnalyticalMertonPolicy(mu=1, r=0, sigma=1, gamma=1)
        assert isinstance(p._mu, float)
        assert isinstance(p._r, float)
        assert isinstance(p._sigma, float)
        assert isinstance(p._gamma, float)


# ===========================================================================
# AnalyticalMertonPolicy — optimal_fraction
# ===========================================================================


class TestAnalyticalMertonOptimalFraction:
    def test_known_value_1(self):
        """
        μ=0.12, r=0.02, σ=0.2, γ=2.
        π* = (0.12 − 0.02) / (0.04 · 2) = 0.10 / 0.08 = 1.25
        """
        p = AnalyticalMertonPolicy(mu=0.12, r=0.02, sigma=0.2, gamma=2.0)
        assert p.optimal_fraction() == pytest.approx(1.25)

    def test_known_value_2(self):
        """
        μ=0.1, r=0.05, σ=0.1, γ=5.
        π* = 0.05 / (0.01 · 5) = 1.0
        """
        p = AnalyticalMertonPolicy(mu=0.1, r=0.05, sigma=0.1, gamma=5.0)
        assert p.optimal_fraction() == pytest.approx(1.0)

    def test_returns_python_float(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert isinstance(p.optimal_fraction(), float)

    def test_positive_when_mu_gt_r(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert p.optimal_fraction() > 0.0

    def test_negative_when_mu_lt_r(self):
        p = AnalyticalMertonPolicy(mu=0.01, r=0.05, sigma=0.2, gamma=2.0)
        assert p.optimal_fraction() < 0.0

    def test_zero_when_mu_equals_r(self):
        p = AnalyticalMertonPolicy(mu=0.05, r=0.05, sigma=0.2, gamma=2.0)
        assert p.optimal_fraction() == pytest.approx(0.0)

    def test_higher_gamma_reduces_fraction(self):
        """Greater risk aversion → smaller risky position."""
        p_low  = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=1.0)
        p_high = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=4.0)
        assert p_high.optimal_fraction() < p_low.optimal_fraction()

    def test_higher_sigma_reduces_fraction(self):
        """Higher volatility → smaller risky position."""
        p_low  = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.1, gamma=2.0)
        p_high = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.4, gamma=2.0)
        assert p_high.optimal_fraction() < p_low.optimal_fraction()

    @pytest.mark.parametrize("mu, r, sigma, gamma, expected", [
        # (mu − r) / (sigma² · gamma)
        (0.08, 0.02, 0.2, 1.0,  1.50),   # 0.06 / 0.04      = 1.5
        (0.08, 0.02, 0.2, 2.0,  0.75),   # 0.06 / 0.08      = 0.75
        (0.08, 0.02, 0.2, 3.0,  0.50),   # 0.06 / 0.12      = 0.5
        (0.10, 0.10, 0.2, 2.0,  0.00),   # mu == r           → 0
        (0.05, 0.10, 0.5, 2.0, -0.10),   # −0.05 / 0.50     = −0.1
    ])
    def test_parametrized_known_values(
        self, mu: float, r: float, sigma: float, gamma: float, expected: float
    ) -> None:
        p = AnalyticalMertonPolicy(mu=mu, r=r, sigma=sigma, gamma=gamma)
        assert p.optimal_fraction() == pytest.approx(expected, rel=1e-8)


# ===========================================================================
# AnalyticalMertonPolicy — get_action
# ===========================================================================


class TestAnalyticalMertonGetAction:
    def test_returns_allocation_action(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert isinstance(p.get_action(make_state(), t=0), AllocationAction)

    def test_single_asset_fraction_correct(self):
        """π* = 1.25 must appear as the sole allocation."""
        p = AnalyticalMertonPolicy(mu=0.12, r=0.02, sigma=0.2, gamma=2.0)
        action = p.get_action(make_state(1.0), t=0)
        assert action.allocations[0] == pytest.approx(1.25)

    def test_single_asset_tuple_length_one(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert len(p.get_action(make_state(), t=0).allocations) == 1

    def test_multi_asset_first_slot_is_merton_fraction(self):
        """Three-asset state: allocs = (π*, 0.0, 0.0)."""
        p = AnalyticalMertonPolicy(mu=0.12, r=0.02, sigma=0.2, gamma=2.0)
        action = p.get_action(make_state_n(1.0, n=3), t=0)
        assert len(action.allocations) == 3
        assert action.allocations[0] == pytest.approx(1.25)
        assert action.allocations[1] == pytest.approx(0.0)
        assert action.allocations[2] == pytest.approx(0.0)

    def test_action_equals_optimal_fraction(self):
        """First allocation must equal optimal_fraction()."""
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        action = p.get_action(make_state(), t=0)
        assert action.allocations[0] == pytest.approx(p.optimal_fraction())

    def test_action_independent_of_wealth(self):
        """Merton fraction is constant across all wealth levels."""
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        fractions = [
            p.get_action(make_state(w), t=0).allocations[0]
            for w in [0.5, 1.0, 2.0, 10.0, 100.0]
        ]
        for f in fractions:
            assert f == pytest.approx(fractions[0])

    def test_action_independent_of_time(self):
        """Merton fraction is constant across all time steps."""
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        fractions = [
            p.get_action(make_state(), t=t).allocations[0]
            for t in range(10)
        ]
        for f in fractions:
            assert f == pytest.approx(fractions[0])

    def test_negative_fraction_short_position(self):
        """μ < r → negative fraction (short risky asset)."""
        p = AnalyticalMertonPolicy(mu=0.01, r=0.05, sigma=0.2, gamma=2.0)
        action = p.get_action(make_state(), t=0)
        assert action.allocations[0] < 0.0
        assert action.allocations[0] == pytest.approx(p.optimal_fraction())


# ===========================================================================
# AnalyticalMertonPolicy — repr
# ===========================================================================


class TestAnalyticalMertonRepr:
    def test_repr_contains_class_name(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert "AnalyticalMertonPolicy" in repr(p)

    def test_repr_contains_mu(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)
        assert "0.1" in repr(p)

    def test_repr_contains_sigma(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.3, gamma=2.0)
        assert "0.3" in repr(p)

    def test_repr_contains_gamma(self):
        p = AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=3.5)
        assert "3.5" in repr(p)


# ===========================================================================
# Shared Policy ABC contract — all three concrete classes
# ===========================================================================


class TestPolicyContract:
    """
    Structural invariants that every Policy subclass must satisfy.
    """

    @pytest.fixture(
        params=["greedy", "random", "merton"],
        ids=["GreedyQPolicy", "RandomPolicy", "AnalyticalMertonPolicy"],
    )
    def policy(self, request: pytest.FixtureRequest) -> Policy:
        if request.param == "greedy":
            return GreedyQPolicy([ConstantQVF(1.0)], [make_action(0.5)])
        if request.param == "random":
            return RandomPolicy(StubActionSpace([make_action(0.5)]))
        return AnalyticalMertonPolicy(mu=0.1, r=0.02, sigma=0.2, gamma=2.0)

    def test_is_policy_instance(self, policy: Policy) -> None:
        assert isinstance(policy, Policy)

    def test_get_action_returns_allocation_action(self, policy: Policy) -> None:
        result = policy.get_action(make_state(), t=0)
        assert isinstance(result, AllocationAction)

    def test_get_action_allocations_are_finite(self, policy: Policy) -> None:
        result = policy.get_action(make_state(), t=0)
        assert all(math.isfinite(a) for a in result.allocations)

    def test_repr_is_non_empty_string(self, policy: Policy) -> None:
        r = repr(policy)
        assert isinstance(r, str) and len(r) > 0