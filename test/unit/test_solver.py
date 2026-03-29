"""
Unit tests for BackwardInductionSolver and GreedyPolicy in solver.py.

Test strategy
-------------
* QValueApproximator and Policy are abstract; we provide lightweight stubs
  (ConstantQVF, PerActionQVF) that expose recorded fit()/predict() call logs,
  letting us assert on Bellman targets without a real regression backend.
* StateSampler is stubbed with FixedStateSampler, which cycles a pre-built
  list of PortfolioState objects.
* All MDPs are real SingleAssetMDP instances with seeded RNGs for determinism.
* sigma=0 / mu=const is used wherever target values must be computed by hand.
"""
from __future__ import annotations

import math
import sys
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import solver
from solver import (
    BackwardInductionSolver,
    GreedyPolicy,
    Policy,
    QValueApproximator,
    StateSampler,
)
from mdp import SingleAssetMDP
from returns import ConstantRisklessReturn, NormalReturnDistribution
from state import ActionSpace, AllocationAction, PortfolioState
from utility import LogUtility


# ===========================================================================
# Stubs
# ===========================================================================


class ConstantQVF(QValueApproximator):
    """
    Always predicts a fixed scalar value.
    Records every call to fit() for inspection.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        # Each element: (sa_pairs, targets) tuple from one fit() call
        self.fit_calls: List[Tuple[list, list]] = []

    def fit(self, sa_pairs: list, targets: list) -> None:
        self.fit_calls.append((list(sa_pairs), list(targets)))

    def predict(self, state: PortfolioState, action: AllocationAction) -> float:
        return self.value

    # ---- satisfy QValueApproximator abstract interface ----
    def update(self, samples) -> "ConstantQVF":
        """Delegates to fit(); satisfies the ABC's update() requirement."""
        sa_pairs = [(s, a) for s, a, _ in samples]
        targets = [t for _, _, t in samples]
        self.fit(sa_pairs, targets)
        return self

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        """Delegates to predict(); satisfies the ABC's evaluate() requirement."""
        return self.predict(state, action)
    # -------------------------------------------------------

    def copy(self) -> "ConstantQVF":
        return ConstantQVF(self.value)


class PerActionQVF(QValueApproximator):
    """
    Returns a pre-specified value per action allocation tuple.
    Useful for testing greedy argmax selection.
    """

    def __init__(
        self,
        values: Dict[tuple, float],
        default: float = 0.0,
    ) -> None:
        self.values = values
        self.default = default
        self.fit_calls: List[Tuple[list, list]] = []

    def fit(self, sa_pairs: list, targets: list) -> None:
        self.fit_calls.append((list(sa_pairs), list(targets)))

    def predict(self, state: PortfolioState, action: AllocationAction) -> float:
        return self.values.get(action.allocations, self.default)

    # ---- satisfy QValueApproximator abstract interface ----
    def update(self, samples) -> "PerActionQVF":
        """Delegates to fit(); satisfies the ABC's update() requirement."""
        sa_pairs = [(s, a) for s, a, _ in samples]
        targets = [t for _, _, t in samples]
        self.fit(sa_pairs, targets)
        return self

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        """Delegates to predict(); satisfies the ABC's evaluate() requirement."""
        return self.predict(state, action)
    # -------------------------------------------------------

    def copy(self) -> "PerActionQVF":
        return PerActionQVF(dict(self.values), self.default)


class FixedStateSampler(StateSampler):
    """Returns a fixed list of states, cycling when n > len(states)."""

    def __init__(self, states: List[PortfolioState]) -> None:
        self._states = states

    def sample(self, n: int) -> List[PortfolioState]:
        return [self._states[i % len(self._states)] for i in range(n)]


# ===========================================================================
# Factory helpers
# ===========================================================================


def make_mdp(
    *,
    mu: float = 0.05,
    sigma: float = 0.10,
    r: float = 0.02,
    time_steps: int = 2,
    seed: int = 0,
    choices: List[float] | None = None,
) -> SingleAssetMDP:
    """Build a seeded SingleAssetMDP with LogUtility."""
    if choices is None:
        choices = [0.0, 0.1]
    return SingleAssetMDP(
        risky_return=NormalReturnDistribution(mu=mu, sigma=sigma),
        riskless_return=ConstantRisklessReturn(rate=r),
        utility=LogUtility(),
        action_space=ActionSpace(choices=choices, n_assets=1),
        time_steps=time_steps,
        rng=np.random.default_rng(seed),
    )


def make_state(
    alloc: float = 0.0,
    wealth: float = 1.0,
) -> PortfolioState:
    return PortfolioState(wealth=wealth, prices=(1.0,), allocations=(alloc,))


def make_sampler(
    n_states: int = 3,
    alloc: float = 0.0,
    base_wealth: float = 1.0,
) -> FixedStateSampler:
    states = [make_state(alloc, base_wealth + i * 0.1) for i in range(n_states)]
    return FixedStateSampler(states)


def make_solver(
    *,
    mdp: SingleAssetMDP | None = None,
    qvf: QValueApproximator | None = None,
    sampler: StateSampler | None = None,
    num_samples: int = 3,
    error_tol: float = 1e-4,
    gamma: float = 1.0,
) -> BackwardInductionSolver:
    return BackwardInductionSolver(
        mdp=mdp or make_mdp(),
        initial_qvf=qvf or ConstantQVF(0.0),
        state_sampler=sampler or make_sampler(),
        num_state_samples=num_samples,
        error_tolerance=error_tol,
        gamma=gamma,
    )


# ===========================================================================
# Constructor validation
# ===========================================================================


class TestBackwardInductionSolverConstructor:
    def test_valid_construction_succeeds(self):
        solver = make_solver()
        assert solver is not None

    def test_num_samples_zero_raises(self):
        with pytest.raises(ValueError, match="num_state_samples"):
            make_solver(num_samples=0)

    def test_num_samples_negative_raises(self):
        with pytest.raises(ValueError, match="num_state_samples"):
            make_solver(num_samples=-1)

    def test_num_samples_one_is_valid(self):
        make_solver(num_samples=1)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            make_solver(gamma=0.0)

    def test_gamma_above_one_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            make_solver(gamma=1.01)

    def test_gamma_exactly_one_is_valid(self):
        make_solver(gamma=1.0)

    def test_gamma_small_positive_is_valid(self):
        make_solver(gamma=0.001)

    def test_negative_error_tolerance_raises(self):
        with pytest.raises(ValueError, match="error_tolerance"):
            make_solver(error_tol=-1e-9)

    def test_zero_error_tolerance_is_valid(self):
        make_solver(error_tol=0.0)

    def test_params_are_stored(self):
        mdp = make_mdp(time_steps=5)
        qvf = ConstantQVF(1.0)
        sampler = make_sampler(n_states=4)
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=qvf,
            state_sampler=sampler,
            num_state_samples=7,
            error_tolerance=0.01,
            gamma=0.9,
        )
        assert solver._mdp is mdp
        assert solver._initial_qvf is qvf
        assert solver._state_sampler is sampler
        assert solver._num_state_samples == 7
        assert math.isclose(solver._error_tolerance, 0.01)
        assert math.isclose(solver._gamma, 0.9)


# ===========================================================================
# solve() — structural invariants
# ===========================================================================


class TestSolveStructure:
    @pytest.mark.parametrize("T", [1, 2, 3, 5])
    def test_returns_list_of_length_T(self, T: int):
        solver = make_solver(mdp=make_mdp(time_steps=T))
        result = solver.solve()
        assert len(result) == T

    def test_all_elements_are_qvf_instances(self):
        T = 3
        result = make_solver(mdp=make_mdp(time_steps=T)).solve()
        for i, qvf in enumerate(result):
            assert isinstance(qvf, QValueApproximator), (
                f"Element at index {i} is not a QValueApproximator"
            )

    def test_fit_called_exactly_once_per_step(self):
        T = 4
        solver = make_solver(mdp=make_mdp(time_steps=T), num_samples=2)
        qvfs = solver.solve()
        total_fits = sum(len(q.fit_calls) for q in qvfs)
        assert total_fits == T

    def test_each_step_gets_distinct_qvf_object(self):
        """copy() must produce a new object for every time step."""
        T = 3
        result = make_solver(mdp=make_mdp(time_steps=T)).solve()
        ids = [id(q) for q in result]
        assert len(set(ids)) == T, "Expected a distinct QVF object per time step"

    def test_prototype_qvf_never_mutated(self):
        """solve() must call copy() and fit only the copies, never the prototype."""
        prototype = ConstantQVF(0.0)
        make_solver(
            mdp=make_mdp(time_steps=4),
            qvf=prototype,
            num_samples=2,
        ).solve()
        assert len(prototype.fit_calls) == 0, (
            "prototype.fit() must never be called; only copies should be fitted"
        )

    def test_t1_solve_returns_single_qvf(self):
        T = 1
        result = make_solver(mdp=make_mdp(time_steps=T), num_samples=1).solve()
        assert len(result) == 1
        assert len(result[0].fit_calls) == 1


# ===========================================================================
# _fit_one_step() — terminal step targets
# ===========================================================================


class TestFitOneStepTerminal:
    """
    At t = T−1 (i.e. is_terminal(t+1) is True), target = U(W_{t+1}).
    No bootstrapping from a next-step QVF occurs.
    """

    def _get_targets(
        self,
        mdp: SingleAssetMDP,
        sampler: FixedStateSampler,
        gamma: float = 1.0,
    ) -> List[float]:
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=ConstantQVF(0.0),
            state_sampler=sampler,
            num_state_samples=len(sampler._states),
            error_tolerance=0.0,
            gamma=gamma,
        )
        # T=1 → _fit_one_step(0, None) is the sole terminal step
        qvf = solver._fit_one_step(t=0, next_step_qvf=None)
        return qvf.fit_calls[0][1]  # list of targets

    def test_targets_are_finite(self):
        mdp = make_mdp(time_steps=1, seed=0)
        targets = self._get_targets(mdp, make_sampler(n_states=2))
        assert all(math.isfinite(t) for t in targets)

    def test_terminal_target_equals_log_utility_of_new_wealth(self):
        """
        Deterministic case: sigma=0, choices=[0.0] only → all-cash transition.
        W_{t+1} = 1.0 * (1 + 0.02) = 1.02 → target = ln(1.02).
        """
        r = 0.02
        mdp = make_mdp(mu=0.0, sigma=0.0, r=r, time_steps=1, seed=0, choices=[0.0])
        sampler = FixedStateSampler([make_state(alloc=0.0, wealth=1.0)])
        targets = self._get_targets(mdp, sampler)
        expected = math.log(1.0 * (1.0 + r))
        assert len(targets) == 1
        assert math.isclose(targets[0], expected, rel_tol=1e-9), (
            f"Expected ln(1.02)={expected:.8f}, got {targets[0]:.8f}"
        )

    def test_terminal_target_scales_with_initial_wealth(self):
        """
        With W=2.0 and all-cash: W_{t+1} = 2.02 → target = ln(2.02).
        """
        r = 0.02
        mdp = make_mdp(mu=0.0, sigma=0.0, r=r, time_steps=1, seed=0, choices=[0.0])
        sampler = FixedStateSampler([make_state(alloc=0.0, wealth=2.0)])
        targets = self._get_targets(mdp, sampler)
        expected = math.log(2.0 * (1.0 + r))
        assert math.isclose(targets[0], expected, rel_tol=1e-9)

    def test_gamma_does_not_change_terminal_target(self):
        """
        gamma must not affect targets at the terminal step since no
        bootstrap occurs.  Results for gamma=1.0 and gamma=0.5 must be equal.
        """
        r = 0.02
        m1 = make_mdp(mu=0.0, sigma=0.0, r=r, time_steps=1, seed=3, choices=[0.0])
        m2 = make_mdp(mu=0.0, sigma=0.0, r=r, time_steps=1, seed=3, choices=[0.0])
        sampler = FixedStateSampler([make_state(alloc=0.0, wealth=1.0)])
        t_g1 = self._get_targets(m1, sampler, gamma=1.0)
        t_g5 = self._get_targets(m2, sampler, gamma=0.5)
        assert math.isclose(t_g1[0], t_g5[0], rel_tol=1e-12)


# ===========================================================================
# _fit_one_step() — non-terminal step targets
# ===========================================================================


class TestFitOneStepNonTerminal:
    """
    At t < T−1 (non-terminal), target = gamma * max_{a'} Q_{t+1}(s', a').
    The reward at non-terminal steps is 0 by MDP construction.
    """

    def _get_targets(
        self,
        next_qvf_value: float,
        gamma: float = 1.0,
        choices: List[float] | None = None,
    ) -> List[float]:
        """
        T=2, t=0 is non-terminal (is_terminal(1)=False).
        next_qvf returns a constant value for all (s', a') pairs.
        """
        if choices is None:
            choices = [0.0]
        mdp = make_mdp(mu=0.0, sigma=0.0, r=0.02, time_steps=2, seed=0, choices=choices)
        sampler = FixedStateSampler([make_state(alloc=0.0, wealth=1.0)])
        next_qvf = ConstantQVF(value=next_qvf_value)
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=ConstantQVF(0.0),
            state_sampler=sampler,
            num_state_samples=1,
            error_tolerance=0.0,
            gamma=gamma,
        )
        qvf = solver._fit_one_step(t=0, next_step_qvf=next_qvf)
        return qvf.fit_calls[0][1]

    def test_target_equals_bootstrap_when_gamma_one(self):
        """target = 1.0 * V when constant QVF predicts V everywhere."""
        V = 4.5
        targets = self._get_targets(next_qvf_value=V, gamma=1.0)
        assert len(targets) == 1
        assert math.isclose(targets[0], V, rel_tol=1e-9)

    def test_gamma_discounts_bootstrap(self):
        """target = 0.8 * 4.5 = 3.6."""
        V, gamma = 4.5, 0.8
        targets = self._get_targets(next_qvf_value=V, gamma=gamma)
        assert math.isclose(targets[0], gamma * V, rel_tol=1e-9)

    def test_zero_bootstrap_produces_zero_target(self):
        targets = self._get_targets(next_qvf_value=0.0, gamma=1.0)
        assert all(math.isclose(t, 0.0, abs_tol=1e-12) for t in targets)

    def test_negative_bootstrap_propagates(self):
        """Negative Q-values (common early in training) must propagate correctly."""
        V = -3.0
        targets = self._get_targets(next_qvf_value=V, gamma=1.0)
        assert all(math.isclose(t, V, rel_tol=1e-9) for t in targets)

    def test_gamma_half_on_negative_value(self):
        V, gamma = -6.0, 0.5
        targets = self._get_targets(next_qvf_value=V, gamma=gamma)
        assert math.isclose(targets[0], gamma * V, rel_tol=1e-9)


# ===========================================================================
# _fit_one_step() — (state, action) pair counts
# ===========================================================================


class TestFitOneStepPairCounts:
    def test_pairs_equal_states_times_feasible_actions(self):
        """
        N states × K feasible actions = N·K (state, action) pairs fed to fit().

        choices=[0.0, 0.1] and alloc=0.0:
          - (0.0,): |0.0−0.0|=0.0 ≤ 0.10 ✓
          - (0.1,): |0.1−0.0|=0.1 ≤ 0.10 ✓
          → K=2 feasible actions per state.
        """
        n_states = 4
        choices = [0.0, 0.1]
        mdp = make_mdp(time_steps=1, seed=0, choices=choices)
        states = [make_state(alloc=0.0, wealth=1.0 + i * 0.5) for i in range(n_states)]
        sampler = FixedStateSampler(states)
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=ConstantQVF(0.0),
            state_sampler=sampler,
            num_state_samples=n_states,
            error_tolerance=0.0,
            gamma=1.0,
        )
        qvf = solver._fit_one_step(t=0, next_step_qvf=None)
        sa_pairs, targets = qvf.fit_calls[0]
        assert len(sa_pairs) == n_states * 2
        assert len(targets) == n_states * 2

    def test_single_action_single_state(self):
        mdp = make_mdp(time_steps=1, seed=0, choices=[0.0])
        sampler = FixedStateSampler([make_state(alloc=0.0)])
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=ConstantQVF(0.0),
            state_sampler=sampler,
            num_state_samples=1,
            error_tolerance=0.0,
            gamma=1.0,
        )
        qvf = solver._fit_one_step(t=0, next_step_qvf=None)
        sa_pairs, targets = qvf.fit_calls[0]
        assert len(sa_pairs) == 1
        assert len(targets) == 1


# ===========================================================================
# extract_policy()
# ===========================================================================


class TestExtractPolicy:
    def test_returns_policy_instance(self):
        solver = make_solver(mdp=make_mdp(time_steps=2))
        policy = solver.extract_policy(solver.solve())
        assert isinstance(policy, Policy)

    def test_returns_greedy_policy(self):
        solver = make_solver(mdp=make_mdp(time_steps=2))
        policy = solver.extract_policy(solver.solve())
        assert isinstance(policy, GreedyPolicy)

    def test_policy_qvfs_match_solve_output(self):
        solver = make_solver(mdp=make_mdp(time_steps=3))
        qvfs = solver.solve()
        policy: GreedyPolicy = solver.extract_policy(qvfs)  # type: ignore[assignment]
        assert policy._qvf_per_step is not qvfs     # not the same list object
        assert len(policy._qvf_per_step) == len(qvfs)
        for i, (pq, sq) in enumerate(zip(policy._qvf_per_step, qvfs)):
            assert pq is sq, f"QVF at index {i} should be the same object"


# ===========================================================================
# GreedyPolicy.get_action()
# ===========================================================================


class TestGreedyPolicyGetAction:
    """
    GreedyPolicy must return argmax_a Q_t(s, a) over feasible actions.
    """

    @pytest.fixture
    def two_action_policy(self):
        """
        Two feasible actions from alloc=0.0:
          (0.0,) → Q=1.0,  (0.1,) → Q=5.0
        Greedy policy must always pick (0.1,).
        """
        mdp = make_mdp(time_steps=1, choices=[0.0, 0.1], seed=0)
        qvf = PerActionQVF(values={(0.0,): 1.0, (0.1,): 5.0})
        policy = GreedyPolicy(qvf_per_step=[qvf], mdp=mdp)
        state = make_state(alloc=0.0, wealth=1.0)
        return policy, state

    def test_picks_highest_q_action(self, two_action_policy):
        policy, state = two_action_policy
        action = policy.get_action(state, t=0)
        assert action.allocations == (0.1,), (
            f"Expected (0.1,) with Q=5.0; got {action.allocations}"
        )

    def test_returns_allocation_action_instance(self, two_action_policy):
        policy, state = two_action_policy
        action = policy.get_action(state, t=0)
        assert isinstance(action, AllocationAction)

    def test_lowest_q_action_is_never_picked(self, two_action_policy):
        policy, state = two_action_policy
        action = policy.get_action(state, t=0)
        assert action.allocations != (0.0,)

    def test_uses_correct_time_step_qvf(self):
        """
        With T=2 and two distinct QVFs:
          t=0 QVF prefers (0.1,),  t=1 QVF prefers (0.0,).
        """
        mdp = make_mdp(time_steps=2, choices=[0.0, 0.1], seed=0)
        qvf0 = PerActionQVF(values={(0.0,): 0.0, (0.1,): 10.0})   # prefer (0.1,)
        qvf1 = PerActionQVF(values={(0.0,): 10.0, (0.1,): 0.0})   # prefer (0.0,)
        policy = GreedyPolicy(qvf_per_step=[qvf0, qvf1], mdp=mdp)

        # From alloc=0.0 at t=0, feasible: {(0.0,), (0.1,)}
        a0 = policy.get_action(make_state(alloc=0.0), t=0)
        # From alloc=0.1 at t=1, feasible: {(0.0,), (0.1,)}
        a1 = policy.get_action(make_state(alloc=0.1), t=1)

        assert a0.allocations == (0.1,), f"t=0: expected (0.1,), got {a0.allocations}"
        assert a1.allocations == (0.0,), f"t=1: expected (0.0,), got {a1.allocations}"

    def test_single_feasible_action_always_selected(self):
        """When only one action is reachable from a state, it must be returned."""
        mdp = make_mdp(time_steps=1, choices=[0.0], seed=0)
        policy = GreedyPolicy(qvf_per_step=[ConstantQVF(0.0)], mdp=mdp)
        action = policy.get_action(make_state(alloc=0.0), t=0)
        assert action.allocations == (0.0,)

    def test_no_feasible_actions_raises_runtime_error(self):
        """
        choices=[0.5] only; from alloc=0.0 the single action (0.5,) is
        infeasible (|0.5−0.0|=0.5 > REBALANCE_LIMIT=0.1) → RuntimeError.
        """
        mdp = make_mdp(time_steps=1, choices=[0.5], seed=0)
        policy = GreedyPolicy(qvf_per_step=[ConstantQVF(0.0)], mdp=mdp)
        with pytest.raises(RuntimeError):
            policy.get_action(make_state(alloc=0.0), t=0)

    def test_three_actions_picks_correct_maximum(self):
        """
        Three feasible actions with distinct Q-values; verify the global max
        is selected even when it is neither the first nor the last in the list.
        """
        mdp = make_mdp(time_steps=1, choices=[0.0, 0.05, 0.1], seed=0)
        # From alloc=0.0 all three are within REBALANCE_LIMIT=0.1
        qvf = PerActionQVF(values={(0.0,): 1.0, (0.05,): 9.0, (0.1,): 3.0})
        policy = GreedyPolicy(qvf_per_step=[qvf], mdp=mdp)
        action = policy.get_action(make_state(alloc=0.0), t=0)
        assert action.allocations == (0.05,), (
            f"Expected (0.05,) with Q=9.0; got {action.allocations}"
        )


# ===========================================================================
# End-to-end: solve → extract_policy → simulate episode
# ===========================================================================


class TestEndToEnd:
    def test_policy_produces_valid_actions_throughout_episode(self):
        """
        Run a full episode under the solved policy; every action must:
          - be an AllocationAction
          - have allocations inside the declared choices grid
          - be reachable from the preceding state
        """
        T = 4
        choices = [0.0, 0.1]
        mdp = make_mdp(time_steps=T, choices=choices, seed=42)
        sampler = FixedStateSampler(
            [make_state(alloc=0.0, wealth=1.0 + i * 0.2) for i in range(5)]
        )
        solver = BackwardInductionSolver(
            mdp=mdp,
            initial_qvf=ConstantQVF(0.0),
            state_sampler=sampler,
            num_state_samples=5,
            error_tolerance=1e-6,
            gamma=0.99,
        )
        qvfs = solver.solve()
        policy = solver.extract_policy(qvfs)

        state = make_state(alloc=0.0, wealth=1.0)
        for t in range(T):
            action = policy.get_action(state, t)
            assert isinstance(action, AllocationAction)
            assert action.allocations[0] in choices, (
                f"t={t}: action {action.allocations} not in grid {choices}"
            )
            state, _ = mdp.step(state, action, t)

    def test_solve_and_policy_length_match_horizon(self):
        for T in [1, 3, 6]:
            mdp = make_mdp(time_steps=T)
            solver = make_solver(mdp=mdp)
            qvfs = solver.solve()
            policy: GreedyPolicy = solver.extract_policy(qvfs)  # type: ignore[assignment]
            assert len(qvfs) == T
            assert len(policy._qvf_per_step) == T

    def test_deterministic_mdp_same_episode_across_runs(self):
        """
        With sigma=0 the MDP is deterministic.
        Two identical solvers must produce policies that trace identical episodes.
        """
        T = 3
        choices = [0.0, 0.1]

        def run() -> List[float]:
            mdp = make_mdp(mu=0.0, sigma=0.0, r=0.02, time_steps=T,
                           seed=0, choices=choices)
            sampler = FixedStateSampler([make_state(alloc=0.0, wealth=1.0)])
            solver = BackwardInductionSolver(
                mdp=mdp,
                initial_qvf=ConstantQVF(0.0),
                state_sampler=sampler,
                num_state_samples=1,
                error_tolerance=0.0,
                gamma=1.0,
            )
            policy = solver.extract_policy(solver.solve())
            state = make_state(alloc=0.0, wealth=1.0)
            wealth_path = [state.wealth]
            for t in range(T):
                action = policy.get_action(state, t)
                state, _ = mdp.step(state, action, t)
                wealth_path.append(state.wealth)
            return wealth_path

        w1 = run()
        w2 = run()
        assert w1 == w2, f"Deterministic MDP produced different paths: {w1} vs {w2}"


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    def test_solver_repr_contains_class_name(self):
        assert "BackwardInductionSolver" in repr(make_solver())

    def test_solver_repr_contains_T(self):
        r = repr(make_solver(mdp=make_mdp(time_steps=7)))
        assert "7" in r

    def test_solver_repr_contains_num_samples(self):
        r = repr(make_solver(num_samples=13))
        assert "13" in r

    def test_solver_repr_contains_gamma(self):
        r = repr(make_solver(gamma=0.95))
        assert "0.95" in r

    def test_greedy_policy_repr_contains_class_name(self):
        solver = make_solver(mdp=make_mdp(time_steps=3))
        policy = solver.extract_policy(solver.solve())
        assert "GreedyPolicy" in repr(policy)

    def test_greedy_policy_repr_contains_T(self):
        T = 5
        solver = make_solver(mdp=make_mdp(time_steps=T))
        policy = solver.extract_policy(solver.solve())
        assert str(T) in repr(policy)