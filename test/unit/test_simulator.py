# test/unit/test_simulator.py
"""
Unit tests for simulator.py — StateSampler, ForwardStateSampler,
PortfolioSimulator.

Test philosophy
---------------
* All MDP, Distribution, ActionSpace, and Policy dependencies are replaced
  with narrow, hand-written stubs so tests are isolated from those modules.
* ForwardStateSampler tests verify correct delegation to the distribution and
  action space, the number of MDP steps taken, and the cumulative wealth
  propagation.
* PortfolioSimulator tests cover trajectory structure, policy dispatch,
  wealth/reward correctness, Monte Carlo averaging, and constructor
  validation.
* Stubs are minimal and transparent: they expose counters and logs so tests
  can assert on internal call patterns without mocking frameworks.
"""
from __future__ import annotations
import math
import sys
import numpy as np
import pytest
from typing import List, Optional, Tuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from policy import Policy
from simulator import ForwardStateSampler, PortfolioSimulator, StateSampler
from state import AllocationAction, PortfolioState


# ===========================================================================
# Stubs — MDP
# ===========================================================================


class StubMDP:
    """
    Deterministic MDP stub.

    Dynamics: new_wealth = old_wealth * growth.
    Reward: 0 for non-terminal steps; utility_fn(new_wealth) at the last step.
    Provides full call logging so tests can assert on step indices and states.
    """

    def __init__(
        self,
        n_steps: int = 5,
        growth: float = 1.0,
        utility_fn=None,
    ) -> None:
        self.n_steps = n_steps
        self._growth = growth
        self._utility = utility_fn if utility_fn is not None else (lambda w: w)
        # (wealth_before, action_allocations, t)
        self.step_calls: List[Tuple[float, tuple, int]] = []

    def initial_state(self, wealth: float) -> PortfolioState:
        return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(0.0,))

    def step(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        self.step_calls.append((state.wealth, action.allocations, t))
        new_wealth = state.wealth * self._growth
        next_state = PortfolioState(
            wealth=new_wealth,
            prices=(1.0,),
            allocations=action.allocations,
        )
        is_terminal = t == self.n_steps - 1
        reward = float(self._utility(new_wealth)) if is_terminal else 0.0
        return next_state, reward


# ===========================================================================
# Stubs — Distribution
# ===========================================================================


class FixedDistribution:
    """Always returns the same value; counts calls."""

    def __init__(self, value: float = 1.0) -> None:
        self._value = float(value)
        self.sample_count: int = 0

    def sample(self) -> float:
        self.sample_count += 1
        return self._value


# ===========================================================================
# Stubs — ActionSpace
# ===========================================================================


class FixedActionSpace:
    """Always returns the same action; counts calls."""

    def __init__(self, action: Optional[AllocationAction] = None) -> None:
        self._action = action or AllocationAction(allocations=(0.5,))
        self.sample_count: int = 0

    def sample(self) -> AllocationAction:
        self.sample_count += 1
        return self._action


# ===========================================================================
# Stubs — Policy
# ===========================================================================


class ConstantPolicy(Policy):
    """Returns the same action for every (state, t); records all calls."""

    def __init__(self, action: Optional[AllocationAction] = None) -> None:
        self._action = action or AllocationAction(allocations=(0.5,))
        # list of (state, t) pairs passed to get_action
        self.calls: List[Tuple[PortfolioState, int]] = []

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        self.calls.append((state, t))
        return self._action


class TimeDependentPolicy(Policy):
    """Cycles through a list of actions by time step index."""

    def __init__(self, actions: List[AllocationAction]) -> None:
        self._actions = actions

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        return self._actions[t % len(self._actions)]


# ===========================================================================
# Helpers
# ===========================================================================


def make_action(alloc: float = 0.5) -> AllocationAction:
    return AllocationAction(allocations=(alloc,))


# ===========================================================================
# StateSampler — ABC contract
# ===========================================================================


class TestStateSamplerABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            StateSampler()  # type: ignore[abstract]

    def test_concrete_without_sample_state_raises(self):
        class Incomplete(StateSampler):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_with_sample_state_works(self):
        class Minimal(StateSampler):
            def sample_state(self, t: int) -> PortfolioState:
                return PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))

        result = Minimal().sample_state(0)
        assert isinstance(result, PortfolioState)


# ===========================================================================
# ForwardStateSampler — constructor
# ===========================================================================


class TestForwardStateSamplerConstructor:
    def test_stores_mdp_reference(self):
        mdp = StubMDP()
        fss = ForwardStateSampler(mdp, FixedDistribution(), FixedActionSpace())
        assert fss._mdp is mdp

    def test_stores_distribution_reference(self):
        dist = FixedDistribution(2.0)
        fss = ForwardStateSampler(StubMDP(), dist, FixedActionSpace())
        assert fss._initial_wealth_dist is dist

    def test_stores_action_space_reference(self):
        space = FixedActionSpace()
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(), space)
        assert fss._action_space is space

    def test_is_state_sampler_instance(self):
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(), FixedActionSpace())
        assert isinstance(fss, StateSampler)


# ===========================================================================
# ForwardStateSampler — sample_state
# ===========================================================================


class TestForwardStateSamplerSampleState:
    # ── Boundary: t = 0 ─────────────────────────────────────────────────

    def test_t0_returns_portfolio_state(self):
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(1.0), FixedActionSpace())
        assert isinstance(fss.sample_state(0), PortfolioState)

    def test_t0_wealth_equals_sampled_wealth(self):
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(3.0), FixedActionSpace())
        assert fss.sample_state(0).wealth == pytest.approx(3.0)

    def test_t0_does_not_call_mdp_step(self):
        mdp = StubMDP()
        fss = ForwardStateSampler(mdp, FixedDistribution(), FixedActionSpace())
        fss.sample_state(0)
        assert len(mdp.step_calls) == 0

    def test_t0_does_not_call_action_space_sample(self):
        space = FixedActionSpace()
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(), space)
        fss.sample_state(0)
        assert space.sample_count == 0

    def test_t0_calls_distribution_once(self):
        dist = FixedDistribution()
        fss = ForwardStateSampler(StubMDP(), dist, FixedActionSpace())
        fss.sample_state(0)
        assert dist.sample_count == 1

    # ── Rolling forward: t > 0 ──────────────────────────────────────────

    @pytest.mark.parametrize("t", [1, 2, 3, 4])
    def test_action_space_called_exactly_t_times(self, t: int):
        space = FixedActionSpace()
        fss = ForwardStateSampler(StubMDP(n_steps=5), FixedDistribution(), space)
        fss.sample_state(t)
        assert space.sample_count == t

    @pytest.mark.parametrize("t", [1, 2, 3, 4])
    def test_mdp_step_called_exactly_t_times(self, t: int):
        mdp = StubMDP(n_steps=5)
        fss = ForwardStateSampler(mdp, FixedDistribution(), FixedActionSpace())
        fss.sample_state(t)
        assert len(mdp.step_calls) == t

    def test_step_indices_passed_in_order(self):
        """mdp.step must receive step indices 0, 1, ..., t−1 in order."""
        mdp = StubMDP(n_steps=5)
        fss = ForwardStateSampler(mdp, FixedDistribution(), FixedActionSpace())
        fss.sample_state(4)
        step_indices = [call[2] for call in mdp.step_calls]
        assert step_indices == [0, 1, 2, 3]

    def test_wealth_accumulates_via_growth(self):
        """
        growth=2.0, initial_wealth=1.0, t=3 → wealth = 1.0 * 2^3 = 8.0.
        """
        mdp = StubMDP(n_steps=5, growth=2.0)
        fss = ForwardStateSampler(mdp, FixedDistribution(1.0), FixedActionSpace())
        state = fss.sample_state(3)
        assert state.wealth == pytest.approx(8.0)

    def test_initial_wealth_from_distribution(self):
        """
        growth=1.0, initial_wealth=7.0, t=2 → state.wealth = 7.0.
        """
        mdp = StubMDP(n_steps=5, growth=1.0)
        fss = ForwardStateSampler(mdp, FixedDistribution(7.0), FixedActionSpace())
        assert fss.sample_state(2).wealth == pytest.approx(7.0)

    def test_each_call_draws_fresh_wealth(self):
        """Every call to sample_state must draw from the distribution once."""
        dist = FixedDistribution(1.0)
        fss = ForwardStateSampler(StubMDP(), dist, FixedActionSpace())
        for _ in range(5):
            fss.sample_state(0)
        assert dist.sample_count == 5

    def test_result_is_portfolio_state_for_all_valid_t(self):
        fss = ForwardStateSampler(StubMDP(n_steps=4), FixedDistribution(), FixedActionSpace())
        for t in range(4):
            assert isinstance(fss.sample_state(t), PortfolioState)

    # ── Validation ──────────────────────────────────────────────────────

    def test_negative_t_raises_value_error(self):
        fss = ForwardStateSampler(StubMDP(n_steps=5), FixedDistribution(), FixedActionSpace())
        with pytest.raises(ValueError, match="t"):
            fss.sample_state(-1)

    def test_t_equal_n_steps_raises_value_error(self):
        fss = ForwardStateSampler(StubMDP(n_steps=3), FixedDistribution(), FixedActionSpace())
        with pytest.raises(ValueError):
            fss.sample_state(3)

    def test_t_far_out_of_range_raises_value_error(self):
        fss = ForwardStateSampler(StubMDP(n_steps=3), FixedDistribution(), FixedActionSpace())
        with pytest.raises(ValueError):
            fss.sample_state(100)

    # ── Last valid step ──────────────────────────────────────────────────

    def test_t_equals_n_steps_minus_1_is_valid(self):
        """t = n_steps − 1 is the last non-terminal step and must not raise."""
        fss = ForwardStateSampler(StubMDP(n_steps=4), FixedDistribution(), FixedActionSpace())
        result = fss.sample_state(3)
        assert isinstance(result, PortfolioState)


# ===========================================================================
# ForwardStateSampler — repr
# ===========================================================================


class TestForwardStateSamplerRepr:
    def test_repr_contains_class_name(self):
        fss = ForwardStateSampler(StubMDP(), FixedDistribution(), FixedActionSpace())
        assert "ForwardStateSampler" in repr(fss)

    def test_repr_contains_n_steps(self):
        fss = ForwardStateSampler(StubMDP(n_steps=9), FixedDistribution(), FixedActionSpace())
        assert "9" in repr(fss)


# ===========================================================================
# PortfolioSimulator — constructor
# ===========================================================================


class TestPortfolioSimulatorConstructor:
    def test_stores_mdp(self):
        mdp = StubMDP()
        sim = PortfolioSimulator(mdp, ConstantPolicy(), 1.0)
        assert sim._mdp is mdp

    def test_stores_policy(self):
        policy = ConstantPolicy()
        sim = PortfolioSimulator(StubMDP(), policy, 1.0)
        assert sim._policy is policy

    def test_stores_initial_wealth(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 3.14)
        assert sim._initial_wealth == pytest.approx(3.14)

    def test_initial_wealth_stored_as_float(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 2)
        assert isinstance(sim._initial_wealth, float)

    def test_zero_initial_wealth_raises(self):
        with pytest.raises(ValueError, match="initial_wealth"):
            PortfolioSimulator(StubMDP(), ConstantPolicy(), 0.0)

    def test_negative_initial_wealth_raises(self):
        with pytest.raises(ValueError, match="initial_wealth"):
            PortfolioSimulator(StubMDP(), ConstantPolicy(), -5.0)

    def test_very_small_positive_wealth_is_valid(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 1e-10)
        assert sim._initial_wealth == pytest.approx(1e-10)


# ===========================================================================
# PortfolioSimulator — simulate_path
# ===========================================================================


class TestSimulatePath:
    # ── Return-type and structural invariants ────────────────────────────

    def test_returns_list(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        assert isinstance(sim.simulate_path(), list)

    @pytest.mark.parametrize("n_steps", [1, 3, 5, 10])
    def test_length_equals_n_steps(self, n_steps: int):
        sim = PortfolioSimulator(StubMDP(n_steps=n_steps), ConstantPolicy(), 1.0)
        assert len(sim.simulate_path()) == n_steps

    def test_each_element_is_three_tuple(self):
        sim = PortfolioSimulator(StubMDP(n_steps=4), ConstantPolicy(), 1.0)
        for item in sim.simulate_path():
            assert isinstance(item, tuple) and len(item) == 3

    def test_state_element_is_portfolio_state(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        for state, _, _ in sim.simulate_path():
            assert isinstance(state, PortfolioState)

    def test_action_element_is_allocation_action(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        for _, action, _ in sim.simulate_path():
            assert isinstance(action, AllocationAction)

    def test_reward_element_is_float(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        for _, _, reward in sim.simulate_path():
            assert isinstance(reward, float)

    def test_all_rewards_are_finite(self):
        sim = PortfolioSimulator(StubMDP(n_steps=5, growth=1.1), ConstantPolicy(), 1.0)
        for _, _, reward in sim.simulate_path():
            assert math.isfinite(reward)

    # ── Initial state ────────────────────────────────────────────────────

    def test_first_state_has_initial_wealth(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 2.5)
        state, _, _ = sim.simulate_path()[0]
        assert state.wealth == pytest.approx(2.5)

    # ── Policy dispatch ──────────────────────────────────────────────────

    def test_policy_called_n_steps_times(self):
        policy = ConstantPolicy()
        sim = PortfolioSimulator(StubMDP(n_steps=5), policy, 1.0)
        sim.simulate_path()
        assert len(policy.calls) == 5

    def test_policy_called_with_consecutive_time_steps(self):
        """get_action must receive t = 0, 1, …, n_steps − 1 in order."""
        policy = ConstantPolicy()
        sim = PortfolioSimulator(StubMDP(n_steps=4), policy, 1.0)
        sim.simulate_path()
        time_steps = [t for _, t in policy.calls]
        assert time_steps == [0, 1, 2, 3]

    def test_policy_receives_correct_wealth_at_each_step(self):
        """
        With growth=2.0 and initial_wealth=1.0 the policy should see
        wealths 1.0, 2.0, 4.0 at t=0, 1, 2.
        """
        policy = ConstantPolicy()
        sim = PortfolioSimulator(StubMDP(n_steps=3, growth=2.0), policy, 1.0)
        sim.simulate_path()
        wealth_sequence = [s.wealth for s, _ in policy.calls]
        assert wealth_sequence == pytest.approx([1.0, 2.0, 4.0])

    # ── Wealth trajectory ────────────────────────────────────────────────

    def test_state_wealth_follows_growth_factor(self):
        """
        States recorded in the path are the pre-action states.
        growth=3.0, initial_wealth=1.0 → wealths 1, 3, 9, 27.
        """
        sim = PortfolioSimulator(StubMDP(n_steps=4, growth=3.0), ConstantPolicy(), 1.0)
        wealths = [s.wealth for s, _, _ in sim.simulate_path()]
        assert wealths == pytest.approx([1.0, 3.0, 9.0, 27.0])

    # ── Action correctness ───────────────────────────────────────────────

    def test_actions_in_path_match_policy_output(self):
        a0 = make_action(0.2)
        a1 = make_action(0.8)
        policy = TimeDependentPolicy([a0, a1])
        sim = PortfolioSimulator(StubMDP(n_steps=4), policy, 1.0)
        path = sim.simulate_path()
        assert path[0][1] == a0
        assert path[1][1] == a1
        assert path[2][1] == a0
        assert path[3][1] == a1

    # ── Reward correctness ───────────────────────────────────────────────

    def test_non_terminal_rewards_are_zero(self):
        sim = PortfolioSimulator(StubMDP(n_steps=5, growth=1.0), ConstantPolicy(), 1.0)
        for _, _, reward in sim.simulate_path()[:-1]:
            assert reward == pytest.approx(0.0)

    def test_terminal_reward_equals_utility_of_terminal_wealth(self):
        """
        growth=2.0, n_steps=3, initial_wealth=1.0 → W_T = 8.0.
        utility = log → terminal reward = log(8.0).
        """
        utility = lambda w: np.log(w)
        sim = PortfolioSimulator(
            StubMDP(n_steps=3, growth=2.0, utility_fn=utility),
            ConstantPolicy(),
            1.0,
        )
        _, _, terminal_reward = sim.simulate_path()[-1]
        assert terminal_reward == pytest.approx(float(np.log(8.0)))

    # ── Determinism ──────────────────────────────────────────────────────

    def test_deterministic_system_gives_identical_paths(self):
        sim = PortfolioSimulator(StubMDP(n_steps=4, growth=1.5), ConstantPolicy(), 2.0)
        path1 = sim.simulate_path()
        path2 = sim.simulate_path()
        for (s1, a1, r1), (s2, a2, r2) in zip(path1, path2):
            assert s1.wealth == pytest.approx(s2.wealth)
            assert a1 == a2
            assert r1 == pytest.approx(r2)

    # ── Single step ──────────────────────────────────────────────────────

    def test_single_step_mdp(self):
        """n_steps=1 must produce exactly one tuple with terminal reward."""
        utility = lambda w: w * 3.0
        sim = PortfolioSimulator(
            StubMDP(n_steps=1, growth=2.0, utility_fn=utility),
            ConstantPolicy(),
            4.0,
        )
        path = sim.simulate_path()
        assert len(path) == 1
        state, _, reward = path[0]
        assert state.wealth == pytest.approx(4.0)
        # terminal_wealth = 4.0 * 2.0 = 8.0; utility = 24.0
        assert reward == pytest.approx(24.0)


# ===========================================================================
# PortfolioSimulator — simulate_many
# ===========================================================================


class TestSimulateMany:
    def test_returns_list(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        assert isinstance(sim.simulate_many(2), list)

    @pytest.mark.parametrize("n", [1, 3, 10, 50])
    def test_length_equals_num_paths(self, n: int):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        assert len(sim.simulate_many(n)) == n

    def test_each_path_has_n_steps_entries(self):
        sim = PortfolioSimulator(StubMDP(n_steps=4), ConstantPolicy(), 1.0)
        for path in sim.simulate_many(5):
            assert len(path) == 4

    def test_each_path_is_a_list(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        for path in sim.simulate_many(3):
            assert isinstance(path, list)

    def test_zero_num_paths_raises(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 1.0)
        with pytest.raises(ValueError, match="num_paths"):
            sim.simulate_many(0)

    def test_negative_num_paths_raises(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 1.0)
        with pytest.raises(ValueError, match="num_paths"):
            sim.simulate_many(-5)

    def test_paths_are_distinct_objects(self):
        sim = PortfolioSimulator(StubMDP(n_steps=2), ConstantPolicy(), 1.0)
        paths = sim.simulate_many(3)
        assert paths[0] is not paths[1]
        assert paths[1] is not paths[2]

    def test_deterministic_paths_are_numerically_identical(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3, growth=2.0), ConstantPolicy(), 1.0)
        paths = sim.simulate_many(5)
        for path in paths[1:]:
            for (s0, a0, r0), (s1, a1, r1) in zip(paths[0], path):
                assert s0.wealth == pytest.approx(s1.wealth)
                assert a0 == a1
                assert r0 == pytest.approx(r1)

    def test_each_element_has_correct_types(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        for path in sim.simulate_many(2):
            for state, action, reward in path:
                assert isinstance(state, PortfolioState)
                assert isinstance(action, AllocationAction)
                assert isinstance(reward, float)


# ===========================================================================
# PortfolioSimulator — expected_terminal_utility
# ===========================================================================


class TestExpectedTerminalUtility:
    def test_returns_float(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        assert isinstance(sim.expected_terminal_utility(5), float)

    def test_returns_finite_value(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3, growth=1.1), ConstantPolicy(), 1.0)
        assert math.isfinite(sim.expected_terminal_utility(10))

    def test_deterministic_utility_value(self):
        """
        growth=2.0, n_steps=3, initial_wealth=1.0 → W_T = 8.0.
        identity utility → E[U] = 8.0.
        """
        utility = lambda w: w
        sim = PortfolioSimulator(
            StubMDP(n_steps=3, growth=2.0, utility_fn=utility),
            ConstantPolicy(),
            1.0,
        )
        assert sim.expected_terminal_utility(1) == pytest.approx(8.0)

    def test_deterministic_value_independent_of_num_paths(self):
        """Same deterministic system → same E[U] regardless of num_paths."""
        utility = lambda w: np.log(w)
        sim = PortfolioSimulator(
            StubMDP(n_steps=4, growth=1.5, utility_fn=utility),
            ConstantPolicy(),
            2.0,
        )
        u1 = sim.expected_terminal_utility(1)
        u10 = sim.expected_terminal_utility(10)
        u100 = sim.expected_terminal_utility(100)
        assert u1 == pytest.approx(u10)
        assert u10 == pytest.approx(u100)

    def test_constant_terminal_reward_returns_that_constant(self):
        """If every terminal reward = k, then E[U] = k exactly."""
        sim = PortfolioSimulator(
            StubMDP(n_steps=2, growth=1.0, utility_fn=lambda w: 7.77),
            ConstantPolicy(),
            1.0,
        )
        assert sim.expected_terminal_utility(100) == pytest.approx(7.77)

    def test_zero_num_paths_raises(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 1.0)
        with pytest.raises(ValueError, match="num_paths"):
            sim.expected_terminal_utility(0)

    def test_negative_num_paths_raises(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 1.0)
        with pytest.raises(ValueError, match="num_paths"):
            sim.expected_terminal_utility(-3)

    def test_higher_initial_wealth_gives_higher_utility(self):
        """identity utility + growth ≥ 1 → more wealth → higher E[U]."""
        utility = lambda w: w
        mdp = StubMDP(n_steps=3, growth=2.0, utility_fn=utility)
        sim_low = PortfolioSimulator(mdp, ConstantPolicy(), 1.0)
        sim_high = PortfolioSimulator(mdp, ConstantPolicy(), 5.0)
        assert sim_high.expected_terminal_utility(5) > sim_low.expected_terminal_utility(5)

    def test_uses_last_step_reward_not_sum(self):
        """
        Non-terminal rewards are 0 in StubMDP; the function must return the
        terminal reward only, not the sum of all rewards.
        """
        utility = lambda w: 42.0
        sim = PortfolioSimulator(
            StubMDP(n_steps=5, growth=1.0, utility_fn=utility),
            ConstantPolicy(),
            1.0,
        )
        # Sum of all rewards would be 42.0 (others are 0); terminal is 42.0.
        # Both are equal here, so verify with utility=1 and sum would differ.
        utility2 = lambda w: 10.0
        sim2 = PortfolioSimulator(
            StubMDP(n_steps=3, growth=1.0, utility_fn=utility2),
            ConstantPolicy(),
            1.0,
        )
        # sum of all rewards = 0+0+10 = 10; terminal reward = 10.
        assert sim2.expected_terminal_utility(1) == pytest.approx(10.0)


# ===========================================================================
# PortfolioSimulator — repr
# ===========================================================================


class TestPortfolioSimulatorRepr:
    def test_repr_contains_class_name(self):
        sim = PortfolioSimulator(StubMDP(n_steps=3), ConstantPolicy(), 1.0)
        assert "PortfolioSimulator" in repr(sim)

    def test_repr_contains_n_steps(self):
        sim = PortfolioSimulator(StubMDP(n_steps=7), ConstantPolicy(), 1.0)
        assert "7" in repr(sim)

    def test_repr_contains_initial_wealth(self):
        sim = PortfolioSimulator(StubMDP(), ConstantPolicy(), 3.5)
        assert "3.5" in repr(sim)

    def test_repr_is_nonempty_string(self):
        r = repr(PortfolioSimulator(StubMDP(), ConstantPolicy(), 1.0))
        assert isinstance(r, str) and len(r) > 0