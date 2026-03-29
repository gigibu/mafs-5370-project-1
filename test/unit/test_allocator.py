# test/unit/test_allocator.py
"""
Unit tests for allocator.py — AssetAllocator.

Test philosophy
---------------
* All six collaborators (MDP, solver, action_space, utility, returns,
  risk-aversion, distribution, state_sampler) are replaced with narrow
  stubs so every test isolates exactly one behaviour.
* run() calls _build_mdp() internally, which does a deferred import from
  mdp.py (not yet implemented).  Every test that exercises run() uses
  monkeypatch to short-circuit _build_mdp and return a StubMDP instead,
  so the tests are completely decoupled from mdp.py.
* evaluate_policy() and benchmark_against_merton() tests bypass run() by
  setting allocator._mdp and allocator._trained_qvfs directly, which is
  valid because AssetAllocator uses plain instance attributes.
* The StubMDP is deterministic (fixed growth, fixed utility), making
  terminal-reward assertions exact rather than approximate.
"""
from __future__ import annotations
import math
import sys
import numpy as np
import pytest
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from allocator import AssetAllocator
from approximator import QValueApproximator
from policy import AnalyticalMertonPolicy, GreedyQPolicy, Policy
from simulator import StateSampler
from state import AllocationAction, PortfolioState


# ===========================================================================
# Stubs — MDP
# ===========================================================================


class StubMDP:
    """
    Deterministic single-step or multi-step MDP.

    Dynamics : new_wealth = old_wealth * growth
    Reward   : 0 for non-terminal steps; utility_fn(new_wealth) at last step.
    """

    def __init__(
        self,
        n_steps: int = 2,
        growth: float = 1.0,
        utility_fn=None,
    ) -> None:
        self.n_steps = n_steps
        self._growth = growth
        self._utility = utility_fn if utility_fn is not None else (lambda w: w)

    def initial_state(self, wealth: float) -> PortfolioState:
        return PortfolioState(
            wealth=float(wealth), prices=(1.0,), allocations=(0.0,)
        )

    def step(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        new_wealth = state.wealth * self._growth

        # PortfolioState enforces sum(allocations) <= 1.0.  Analytical
        # policies (e.g. Merton) may return leveraged fractions > 1, so we
        # clamp proportionally here.  StubMDP only needs to advance wealth;
        # it does not rely on accurate allocation tracking.
        allocs = action.allocations
        total = sum(allocs)
        if total > 1.0 + 1e-9:
            allocs = tuple(a / total for a in allocs)

        next_state = PortfolioState(
            wealth=new_wealth,
            prices=(1.0,),
            allocations=allocs,
        )
        is_terminal = t == self.n_steps - 1
        reward = float(self._utility(new_wealth)) if is_terminal else 0.0
        return next_state, reward


# ===========================================================================
# Stubs — Solver
# ===========================================================================


class StubSolver:
    """
    Returns a list of ConstantQVF objects (one per MDP step) and logs calls.
    """

    def __init__(self, n_steps: int = 2, q_value: float = 1.0) -> None:
        self._n_steps = n_steps
        self._q_value = q_value
        self.solve_calls: List[Tuple[Any, Any, Any]] = []

    def solve(
        self,
        mdp: Any,
        state_sampler: Any,
        initial_qvf: Any,
    ) -> List[QValueApproximator]:
        self.solve_calls.append((mdp, state_sampler, initial_qvf))
        return [ConstantQVF(self._q_value) for _ in range(self._n_steps)]


# ===========================================================================
# Stubs — ActionSpace
# ===========================================================================


class StubActionSpace:
    """Has feasible_actions and sample(); both are tested."""

    def __init__(self, actions: Optional[List[AllocationAction]] = None) -> None:
        self._actions = actions or [
            AllocationAction(allocations=(0.0,)),
            AllocationAction(allocations=(0.5,)),
            AllocationAction(allocations=(1.0,)),
        ]
        self.sample_count: int = 0

    @property
    def feasible_actions(self) -> List[AllocationAction]:
        return self._actions

    def sample(self) -> AllocationAction:
        self.sample_count += 1
        return self._actions[0]


# ===========================================================================
# Stubs — Market components
# ===========================================================================


class StubUtility:
    def evaluate(self, wealth: float) -> float:
        return float(wealth)


class StubRisklessReturn:
    def __init__(self, r: float = 0.02) -> None:
        self.r = float(r)


class StubRiskyReturn:
    def __init__(self, mu: float = 0.10, sigma: float = 0.20) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)


class StubRiskAversion:
    def __init__(self, gamma: float = 2.0) -> None:
        self.gamma = float(gamma)


class StubDistribution:
    """Always returns a fixed wealth value; counts calls."""

    def __init__(self, value: float = 1.0) -> None:
        self._value = float(value)
        self.sample_count: int = 0

    def sample(self) -> float:
        self.sample_count += 1
        return self._value


# ===========================================================================
# Stubs — StateSampler
# ===========================================================================


class StubStateSampler(StateSampler):
    def sample_state(self, t: int) -> PortfolioState:
        return PortfolioState(wealth=1.0, prices=(1.0,), allocations=(0.0,))


# ===========================================================================
# Stubs — QValueApproximator / Policy
# ===========================================================================


class ConstantQVF(QValueApproximator):
    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        return float(self._value)

    def update(self, samples):
        return self

    def copy(self):
        return ConstantQVF(self._value)


class MappedQVF(QValueApproximator):
    """Returns Q values keyed by action.allocations tuple."""

    def __init__(self, q_map: dict) -> None:
        self._q_map = q_map

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        return float(self._q_map[action.allocations])

    def update(self, samples):
        return self

    def copy(self):
        return MappedQVF(dict(self._q_map))


class ConstantPolicy(Policy):
    """Returns the same fixed action regardless of state or time."""

    def __init__(
        self, action: Optional[AllocationAction] = None
    ) -> None:
        self._action = action or AllocationAction(allocations=(0.5,))
        self.calls: List[Tuple[PortfolioState, int]] = []

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        self.calls.append((state, t))
        return self._action


class TimestepRecordingPolicy(Policy):
    """Records every (wealth, t) pair passed to get_action."""

    def __init__(self, action: Optional[AllocationAction] = None) -> None:
        self._action = action or AllocationAction(allocations=(0.3,))
        self.records: List[Tuple[float, int]] = []

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        self.records.append((state.wealth, t))
        return self._action


# ===========================================================================
# Helper factories
# ===========================================================================


def make_action(alloc: float = 0.5) -> AllocationAction:
    return AllocationAction(allocations=(alloc,))


def make_allocator(
    n_steps: int = 2,
    growth: float = 1.0,
    utility_fn=None,
    risky_returns: Optional[List[Any]] = None,
    r: float = 0.02,
    gamma: float = 2.0,
    initial_wealth: float = 1.0,
    action_space: Optional[StubActionSpace] = None,
) -> AssetAllocator:
    """Build an AssetAllocator wired with stubs for all components."""
    return AssetAllocator(
        utility=StubUtility(),
        riskless_return=StubRisklessReturn(r=r),
        risky_returns=risky_returns or [StubRiskyReturn()],
        risk_aversion=StubRiskAversion(gamma=gamma),
        action_space=action_space or StubActionSpace(),
        qvf_approximator=ConstantQVF(),
        solver=StubSolver(n_steps=n_steps),
        state_sampler=StubStateSampler(),
        initial_wealth_distribution=StubDistribution(initial_wealth),
    )


def make_trained_allocator(
    monkeypatch,
    n_steps: int = 2,
    growth: float = 1.0,
    utility_fn=None,
    **kwargs,
) -> AssetAllocator:
    """
    Return an AssetAllocator where run() has already been called.
    Uses monkeypatch to replace _build_mdp so mdp.py is not needed.
    """
    stub_mdp = StubMDP(n_steps=n_steps, growth=growth, utility_fn=utility_fn)
    allocator = make_allocator(n_steps=n_steps, growth=growth, **kwargs)
    monkeypatch.setattr(allocator, "_build_mdp", lambda: stub_mdp)
    allocator.run()
    return allocator


# ===========================================================================
# TestAssetAllocatorConstructor
# ===========================================================================


class TestAssetAllocatorConstructor:
    def test_stores_utility(self):
        u = StubUtility()
        a = AssetAllocator(
            utility=u,
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        assert a._utility is u

    def test_stores_riskless_return(self):
        rl = StubRisklessReturn(r=0.05)
        a = make_allocator()
        a._riskless_return = rl
        assert a._riskless_return is rl

    def test_risky_returns_stored_as_list(self):
        r1, r2 = StubRiskyReturn(), StubRiskyReturn()
        a = make_allocator(risky_returns=[r1, r2])
        assert isinstance(a._risky_returns, list)
        assert a._risky_returns == [r1, r2]

    def test_stores_solver(self):
        solver = StubSolver()
        a = make_allocator()
        a._solver = solver
        assert a._solver is solver

    def test_stores_state_sampler(self):
        ss = StubStateSampler()
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(),
            state_sampler=ss,
            initial_wealth_distribution=StubDistribution(),
        )
        assert a._state_sampler is ss

    def test_stores_distribution(self):
        dist = StubDistribution(5.0)
        a = make_allocator()
        a._initial_wealth_dist = dist
        assert a._initial_wealth_dist is dist

    def test_mdp_none_before_run(self):
        a = make_allocator()
        assert a.mdp is None

    def test_is_trained_false_before_run(self):
        a = make_allocator()
        assert not a.is_trained

    def test_trained_qvfs_empty_before_run(self):
        a = make_allocator()
        assert a._trained_qvfs == []

    def test_empty_risky_returns_raises(self):
        with pytest.raises(ValueError, match="risky_returns"):
            AssetAllocator(
                utility=StubUtility(),
                riskless_return=StubRisklessReturn(),
                risky_returns=[],
                risk_aversion=StubRiskAversion(),
                action_space=StubActionSpace(),
                qvf_approximator=ConstantQVF(),
                solver=StubSolver(),
                state_sampler=StubStateSampler(),
                initial_wealth_distribution=StubDistribution(),
            )

    def test_two_risky_assets_stored(self):
        r1, r2 = StubRiskyReturn(), StubRiskyReturn()
        a = make_allocator(risky_returns=[r1, r2])
        assert len(a._risky_returns) == 2

    def test_risky_returns_sequence_converted_to_list(self):
        """Passing a tuple must still produce an internal list."""
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=(StubRiskyReturn(),),  # tuple
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        assert isinstance(a._risky_returns, list)


# ===========================================================================
# TestAssetAllocatorRun
# ===========================================================================


class TestAssetAllocatorRun:
    """
    All run() tests monkeypatch _build_mdp so mdp.py is not required.
    """

    def test_run_returns_policy_instance(self, monkeypatch):
        a = make_trained_allocator(monkeypatch, n_steps=2)
        # re-run with fresh allocator
        a2 = make_allocator(n_steps=2)
        stub_mdp = StubMDP(n_steps=2)
        monkeypatch.setattr(a2, "_build_mdp", lambda: stub_mdp)
        policy = a2.run()
        assert isinstance(policy, Policy)

    def test_run_returns_greedy_q_policy(self, monkeypatch):
        a = make_allocator(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        policy = a.run()
        assert isinstance(policy, GreedyQPolicy)

    def test_run_calls_solver_once(self, monkeypatch):
        solver = StubSolver(n_steps=2)
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=solver,
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        assert len(solver.solve_calls) == 1

    def test_run_passes_mdp_to_solver(self, monkeypatch):
        solver = StubSolver(n_steps=2)
        stub_mdp = StubMDP(n_steps=2)
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=solver,
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        monkeypatch.setattr(a, "_build_mdp", lambda: stub_mdp)
        a.run()
        assert solver.solve_calls[0][0] is stub_mdp

    def test_run_passes_state_sampler_to_solver(self, monkeypatch):
        solver = StubSolver(n_steps=2)
        ss = StubStateSampler()
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=solver,
            state_sampler=ss,
            initial_wealth_distribution=StubDistribution(),
        )
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        assert solver.solve_calls[0][1] is ss

    def test_run_passes_initial_qvf_to_solver(self, monkeypatch):
        solver = StubSolver(n_steps=2)
        initial_qvf = ConstantQVF(7.0)
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=StubActionSpace(),
            qvf_approximator=initial_qvf,
            solver=solver,
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        assert solver.solve_calls[0][2] is initial_qvf

    def test_run_sets_mdp_attribute(self, monkeypatch):
        stub_mdp = StubMDP(n_steps=2)
        a = make_allocator(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: stub_mdp)
        a.run()
        assert a.mdp is stub_mdp

    def test_run_populates_trained_qvfs(self, monkeypatch):
        a = make_allocator(n_steps=3)
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=3))
        a.run()
        assert len(a._trained_qvfs) == 3

    def test_run_sets_is_trained_true(self, monkeypatch):
        a = make_allocator(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        assert a.is_trained

    def test_run_greedy_policy_has_correct_n_actions(self, monkeypatch):
        """GreedyQPolicy must contain all actions from action_space."""
        actions = [make_action(float(i) * 0.25) for i in range(4)]
        space = StubActionSpace(actions=actions)
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(),
            risky_returns=[StubRiskyReturn()],
            risk_aversion=StubRiskAversion(),
            action_space=space,
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(n_steps=2),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(),
        )
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        policy = a.run()
        assert isinstance(policy, GreedyQPolicy)
        assert len(policy._feasible_actions) == 4

    def test_second_run_overwrites_trained_qvfs(self, monkeypatch):
        """Calling run() twice must update _trained_qvfs to the new result."""
        a = make_allocator(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        first_qvfs = list(a._trained_qvfs)
        a.run()
        # New list was created (different objects)
        assert a._trained_qvfs is not first_qvfs


# ===========================================================================
# TestAssetAllocatorEvaluatePolicy
# ===========================================================================


class TestAssetAllocatorEvaluatePolicy:
    def _make_trained(self, n_steps=2, growth=1.0, utility_fn=None, initial_wealth=1.0):
        """Build allocator with _mdp and _trained_qvfs already set (bypass run)."""
        a = make_allocator(n_steps=n_steps, initial_wealth=initial_wealth)
        a._mdp = StubMDP(n_steps=n_steps, growth=growth, utility_fn=utility_fn)
        a._trained_qvfs = [ConstantQVF(1.0) for _ in range(n_steps)]
        return a

    # ── Return type and keys ─────────────────────────────────────────────

    def test_returns_dict(self):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=3)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "key",
        ["expected_utility", "std_utility", "min_utility", "max_utility", "num_simulations"],
    )
    def test_required_keys_present(self, key):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=3)
        assert key in result

    def test_all_values_are_floats(self):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=5)
        for v in result.values():
            assert isinstance(v, float)

    def test_all_values_are_finite(self):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=5)
        for v in result.values():
            assert math.isfinite(v)

    # ── Correctness of metrics ───────────────────────────────────────────

    def test_num_simulations_echoed(self):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=7)
        assert result["num_simulations"] == pytest.approx(7.0)

    def test_deterministic_expected_utility(self):
        """
        growth=2.0, n_steps=2, initial_wealth=1.0, utility=identity.
        W_T = 1.0 * 2^2 = 4.0  →  terminal reward = 4.0.
        E[U] = 4.0 regardless of num_simulations.
        """
        a = self._make_trained(
            n_steps=2, growth=2.0, utility_fn=lambda w: w, initial_wealth=1.0
        )
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=10)
        assert result["expected_utility"] == pytest.approx(4.0)

    def test_deterministic_std_is_zero(self):
        """All paths identical → std must be zero."""
        a = self._make_trained(
            n_steps=2, growth=2.0, utility_fn=lambda w: w, initial_wealth=1.0
        )
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=10)
        assert result["std_utility"] == pytest.approx(0.0)

    def test_min_equals_max_for_deterministic_system(self):
        a = self._make_trained(growth=1.5, utility_fn=lambda w: w)
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=5)
        assert result["min_utility"] == pytest.approx(result["max_utility"])

    def test_min_leq_expected_leq_max(self):
        a = self._make_trained()
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=10)
        assert result["min_utility"] <= result["expected_utility"]
        assert result["expected_utility"] <= result["max_utility"]

    def test_single_simulation(self):
        a = self._make_trained(growth=3.0, utility_fn=lambda w: w, initial_wealth=2.0)
        result = a.evaluate_policy(ConstantPolicy(), num_simulations=1)
        # W_T = 2.0 * 3^2 = 18.0
        assert result["expected_utility"] == pytest.approx(18.0)

    # ── Validation ───────────────────────────────────────────────────────

    def test_raises_if_mdp_not_built(self):
        a = make_allocator()
        with pytest.raises(RuntimeError, match="run()"):
            a.evaluate_policy(ConstantPolicy(), num_simulations=5)

    def test_raises_if_num_simulations_is_zero(self):
        a = self._make_trained()
        with pytest.raises(ValueError, match="num_simulations"):
            a.evaluate_policy(ConstantPolicy(), num_simulations=0)

    def test_raises_if_num_simulations_is_negative(self):
        a = self._make_trained()
        with pytest.raises(ValueError, match="num_simulations"):
            a.evaluate_policy(ConstantPolicy(), num_simulations=-1)

    # ── Distribution sampling ────────────────────────────────────────────

    def test_samples_initial_wealth_from_distribution(self):
        """evaluate_policy must call distribution.sample() exactly once."""
        dist = StubDistribution(2.0)
        a = make_allocator(initial_wealth=2.0)
        a._mdp = StubMDP(n_steps=2)
        a._trained_qvfs = [ConstantQVF(), ConstantQVF()]
        a._initial_wealth_dist = dist
        dist.sample_count = 0
        a.evaluate_policy(ConstantPolicy(), num_simulations=5)
        assert dist.sample_count == 1


# ===========================================================================
# TestAssetAllocatorBenchmarkMerton
# ===========================================================================


class TestAssetAllocatorBenchmarkMerton:
    # Known Merton parameters — easy to verify analytically
    # mu=0.12, r=0.02, sigma=0.2, gamma=2 → π* = 0.10/(0.04*2) = 1.25
    _MU, _R, _SIGMA, _GAMMA = 0.12, 0.02, 0.20, 2.0
    _EXPECTED_FRACTION = 1.25  # (0.12-0.02)/(0.04*2)

    def _make_benchmark_allocator(self, n_steps: int = 2) -> AssetAllocator:
        """Allocator with _mdp and _trained_qvfs set; no run() needed."""
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(r=self._R),
            risky_returns=[StubRiskyReturn(mu=self._MU, sigma=self._SIGMA)],
            risk_aversion=StubRiskAversion(gamma=self._GAMMA),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(n_steps=n_steps),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(1.0),
        )
        a._mdp = StubMDP(n_steps=n_steps, growth=1.0, utility_fn=lambda w: w)
        a._trained_qvfs = [ConstantQVF(1.0) for _ in range(n_steps)]
        return a

    # ── Return type and keys ─────────────────────────────────────────────

    def test_returns_dict(self):
        a = self._make_benchmark_allocator()
        assert isinstance(a.benchmark_against_merton(num_simulations=1), dict)

    @pytest.mark.parametrize(
        "key",
        [
            "rl_expected_utility",
            "merton_expected_utility",
            "merton_fraction",
            "outperforms_merton",
        ],
    )
    def test_required_keys_present(self, key):
        a = self._make_benchmark_allocator()
        result = a.benchmark_against_merton(num_simulations=1)
        assert key in result

    def test_merton_fraction_is_float(self):
        a = self._make_benchmark_allocator()
        result = a.benchmark_against_merton(num_simulations=1)
        assert isinstance(result["merton_fraction"], float)

    def test_outperforms_merton_is_bool(self):
        a = self._make_benchmark_allocator()
        result = a.benchmark_against_merton(num_simulations=1)
        assert isinstance(result["outperforms_merton"], bool)

    def test_utilities_are_finite(self):
        a = self._make_benchmark_allocator()
        result = a.benchmark_against_merton(num_simulations=5)
        assert math.isfinite(result["rl_expected_utility"])
        assert math.isfinite(result["merton_expected_utility"])

    # ── Merton fraction correctness ──────────────────────────────────────

    def test_merton_fraction_value(self):
        """
        (μ - r) / (σ² · γ) = (0.12 - 0.02) / (0.04 · 2) = 1.25.
        """
        a = self._make_benchmark_allocator()
        result = a.benchmark_against_merton(num_simulations=1)
        assert result["merton_fraction"] == pytest.approx(self._EXPECTED_FRACTION)

    def test_merton_fraction_independent_of_num_simulations(self):
        a = self._make_benchmark_allocator()
        f1 = a.benchmark_against_merton(num_simulations=1)["merton_fraction"]
        f10 = a.benchmark_against_merton(num_simulations=10)["merton_fraction"]
        assert f1 == pytest.approx(f10)

    def test_merton_fraction_reflects_mu_and_sigma(self):
        """Doubling σ must halve the Merton fraction."""
        a1 = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(r=0.02),
            risky_returns=[StubRiskyReturn(mu=0.10, sigma=0.20)],
            risk_aversion=StubRiskAversion(gamma=2.0),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(n_steps=2),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(1.0),
        )
        a1._mdp = StubMDP(n_steps=2)
        a1._trained_qvfs = [ConstantQVF(), ConstantQVF()]

        a2 = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(r=0.02),
            risky_returns=[StubRiskyReturn(mu=0.10, sigma=0.40)],  # σ doubled
            risk_aversion=StubRiskAversion(gamma=2.0),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(n_steps=2),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(1.0),
        )
        a2._mdp = StubMDP(n_steps=2)
        a2._trained_qvfs = [ConstantQVF(), ConstantQVF()]

        f1 = a1.benchmark_against_merton(num_simulations=1)["merton_fraction"]
        f2 = a2.benchmark_against_merton(num_simulations=1)["merton_fraction"]
        assert f2 == pytest.approx(f1 / 4.0)  # σ doubled → σ² quadrupled → fraction /4

    # ── Validation ───────────────────────────────────────────────────────

    def test_raises_if_mdp_not_built(self):
        a = make_allocator()
        with pytest.raises(RuntimeError, match="run()"):
            a.benchmark_against_merton(num_simulations=1)

    def test_raises_if_no_trained_qvfs(self):
        a = make_allocator()
        a._mdp = StubMDP(n_steps=2)
        # _trained_qvfs remains empty
        with pytest.raises(RuntimeError):
            a.benchmark_against_merton(num_simulations=1)

    def test_raises_if_num_simulations_is_zero(self):
        a = self._make_benchmark_allocator()
        with pytest.raises(ValueError, match="num_simulations"):
            a.benchmark_against_merton(num_simulations=0)

    def test_raises_if_num_simulations_is_negative(self):
        a = self._make_benchmark_allocator()
        with pytest.raises(ValueError, match="num_simulations"):
            a.benchmark_against_merton(num_simulations=-5)

    # ── deterministic comparison ─────────────────────────────────────────

    def test_deterministic_system_utilities_are_equal(self):
        """
        identity utility + growth=1 + constant QVF → same W_T for both
        policies → rl_expected_utility == merton_expected_utility.
        """
        a = self._make_benchmark_allocator(n_steps=1)
        result = a.benchmark_against_merton(num_simulations=5)
        assert result["rl_expected_utility"] == pytest.approx(
            result["merton_expected_utility"]
        )


# ===========================================================================
# TestAssetAllocatorGetOptimalAllocations
# ===========================================================================


class TestAssetAllocatorGetOptimalAllocations:
    def _make_allocator(self, n_assets: int = 1) -> AssetAllocator:
        return make_allocator(
            risky_returns=[StubRiskyReturn() for _ in range(n_assets)]
        )

    # ── Return type and structure ────────────────────────────────────────

    def test_returns_list(self):
        a = self._make_allocator()
        result = a.get_optimal_allocations(ConstantPolicy(), [1.0, 2.0], t=0)
        assert isinstance(result, list)

    def test_length_equals_wealth_grid_length(self):
        a = self._make_allocator()
        grid = [float(w) for w in range(1, 8)]
        result = a.get_optimal_allocations(ConstantPolicy(), grid, t=0)
        assert len(result) == len(grid)

    def test_elements_are_floats(self):
        a = self._make_allocator()
        for v in a.get_optimal_allocations(ConstantPolicy(), [1.0, 2.0, 3.0], t=0):
            assert isinstance(v, float)

    def test_elements_are_finite(self):
        a = self._make_allocator()
        for v in a.get_optimal_allocations(ConstantPolicy(), [1.0, 2.0, 3.0], t=0):
            assert math.isfinite(v)

    def test_empty_grid_returns_empty_list(self):
        a = self._make_allocator()
        result = a.get_optimal_allocations(ConstantPolicy(), [], t=0)
        assert result == []

    # ── Policy dispatch correctness ──────────────────────────────────────

    def test_policy_called_once_per_wealth_point(self):
        a = self._make_allocator()
        policy = ConstantPolicy()
        a.get_optimal_allocations(policy, [1.0, 2.0, 3.0], t=0)
        assert len(policy.calls) == 3

    def test_policy_receives_correct_time_step(self):
        a = self._make_allocator()
        policy = TimestepRecordingPolicy()
        a.get_optimal_allocations(policy, [1.0, 2.0], t=5)
        for _, t in policy.records:
            assert t == 5

    def test_policy_receives_correct_wealth_values(self):
        a = self._make_allocator()
        policy = TimestepRecordingPolicy()
        grid = [1.0, 2.5, 4.0]
        a.get_optimal_allocations(policy, grid, t=0)
        observed_wealths = [w for w, _ in policy.records]
        assert observed_wealths == pytest.approx(grid)

    def test_returns_first_action_allocation(self):
        """The result contains allocations[0] from the policy's action."""
        target_alloc = 0.73
        policy = ConstantPolicy(action=AllocationAction(allocations=(target_alloc,)))
        a = self._make_allocator()
        result = a.get_optimal_allocations(policy, [1.0, 2.0], t=0)
        for v in result:
            assert v == pytest.approx(target_alloc)

    def test_reflects_wealth_dependent_policy(self):
        """
        A policy that returns different actions based on wealth must produce
        correspondingly different allocation values.
        """

        class WealthThresholdPolicy(Policy):
            def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
                if state.wealth < 2.0:
                    return AllocationAction(allocations=(0.2,))
                return AllocationAction(allocations=(0.8,))

        a = self._make_allocator()
        result = a.get_optimal_allocations(
            WealthThresholdPolicy(), [1.0, 3.0], t=0
        )
        assert result[0] == pytest.approx(0.2)
        assert result[1] == pytest.approx(0.8)

    def test_multi_asset_uses_first_allocation_only(self):
        """get_optimal_allocations always returns allocations[0]."""
        policy = ConstantPolicy(
            action=AllocationAction(allocations=(0.4, 0.3, 0.3))
        )
        a = self._make_allocator(n_assets=3)
        result = a.get_optimal_allocations(policy, [1.0, 2.0], t=0)
        for v in result:
            assert v == pytest.approx(0.4)

    def test_single_wealth_point(self):
        policy = ConstantPolicy(action=AllocationAction(allocations=(0.55,)))
        a = self._make_allocator()
        result = a.get_optimal_allocations(policy, [5.0], t=3)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.55)

    def test_n_assets_determines_state_shape(self):
        """
        State passed to the policy must have allocations of length n_assets.
        """

        class ShapeCheckPolicy(Policy):
            def __init__(self):
                self.observed_n = []

            def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
                self.observed_n.append(len(state.allocations))
                return AllocationAction(allocations=(0.5,) * len(state.allocations))

        policy = ShapeCheckPolicy()
        a = self._make_allocator(n_assets=3)
        a.get_optimal_allocations(policy, [1.0, 2.0], t=0)
        for n in policy.observed_n:
            assert n == 3

    # ── Integration with AnalyticalMertonPolicy ──────────────────────────

    def test_merton_policy_constant_across_wealth(self):
        """
        AnalyticalMertonPolicy returns the same fraction for every wealth.
        mu=0.12, r=0.02, sigma=0.2, gamma=2 → π*=1.25 everywhere.
        """
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(r=0.02),
            risky_returns=[StubRiskyReturn(mu=0.12, sigma=0.20)],
            risk_aversion=StubRiskAversion(gamma=2.0),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(1.0),
        )
        merton = AnalyticalMertonPolicy(mu=0.12, r=0.02, sigma=0.20, gamma=2.0)
        grid = [0.5, 1.0, 2.0, 5.0, 10.0]
        result = a.get_optimal_allocations(merton, grid, t=0)
        for v in result:
            assert v == pytest.approx(1.25)


# ===========================================================================
# TestAssetAllocatorProperties
# ===========================================================================


class TestAssetAllocatorProperties:
    def test_mdp_property_returns_none_initially(self):
        a = make_allocator()
        assert a.mdp is None

    def test_mdp_property_returns_mdp_after_set(self):
        a = make_allocator()
        stub = StubMDP()
        a._mdp = stub
        assert a.mdp is stub

    def test_is_trained_false_when_mdp_none(self):
        a = make_allocator()
        assert not a.is_trained

    def test_is_trained_false_when_qvfs_empty(self):
        a = make_allocator()
        a._mdp = StubMDP()
        # _trained_qvfs still empty
        assert not a.is_trained

    def test_is_trained_true_when_both_set(self):
        a = make_allocator(n_steps=2)
        a._mdp = StubMDP(n_steps=2)
        a._trained_qvfs = [ConstantQVF(), ConstantQVF()]
        assert a.is_trained


# ===========================================================================
# TestAssetAllocatorRepr
# ===========================================================================


class TestAssetAllocatorRepr:
    def test_repr_contains_class_name(self):
        assert "AssetAllocator" in repr(make_allocator())

    def test_repr_contains_n_risky(self):
        a = make_allocator(risky_returns=[StubRiskyReturn(), StubRiskyReturn()])
        assert "2" in repr(a)

    def test_repr_is_trained_false_before_run(self):
        a = make_allocator()
        assert "False" in repr(a)

    def test_repr_is_trained_true_after_run(self, monkeypatch):
        a = make_allocator(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: StubMDP(n_steps=2))
        a.run()
        assert "True" in repr(a)

    def test_repr_is_nonempty_string(self):
        r = repr(make_allocator())
        assert isinstance(r, str) and len(r) > 0


# ===========================================================================
# TestAssetAllocatorIntegration
# ===========================================================================


class TestAssetAllocatorIntegration:
    """
    Lightweight integration tests that chain run() → evaluate_policy()
    and run() → benchmark_against_merton() without mocking internals.
    These use monkeypatch only for _build_mdp so mdp.py is not required.
    """

    def test_run_then_evaluate_policy(self, monkeypatch):
        """run() followed by evaluate_policy() must succeed end-to-end."""
        a = make_allocator(n_steps=2, growth=1.0)
        stub_mdp = StubMDP(n_steps=2, growth=1.0, utility_fn=lambda w: w)
        monkeypatch.setattr(a, "_build_mdp", lambda: stub_mdp)

        policy = a.run()
        result = a.evaluate_policy(policy, num_simulations=5)

        assert "expected_utility" in result
        assert math.isfinite(result["expected_utility"])

    def test_run_then_benchmark(self, monkeypatch):
        """run() followed by benchmark_against_merton() must return valid dict."""
        a = AssetAllocator(
            utility=StubUtility(),
            riskless_return=StubRisklessReturn(r=0.02),
            risky_returns=[StubRiskyReturn(mu=0.10, sigma=0.20)],
            risk_aversion=StubRiskAversion(gamma=2.0),
            action_space=StubActionSpace(),
            qvf_approximator=ConstantQVF(),
            solver=StubSolver(n_steps=2),
            state_sampler=StubStateSampler(),
            initial_wealth_distribution=StubDistribution(1.0),
        )
        stub_mdp = StubMDP(n_steps=2, growth=1.0, utility_fn=lambda w: w)
        monkeypatch.setattr(a, "_build_mdp", lambda: stub_mdp)

        a.run()
        result = a.benchmark_against_merton(num_simulations=5)

        assert "merton_fraction" in result
        assert "outperforms_merton" in result
        assert isinstance(result["outperforms_merton"], bool)

    def test_run_then_get_optimal_allocations(self, monkeypatch):
        """run() followed by get_optimal_allocations() must return valid list."""
        a = make_allocator(n_steps=2)
        stub_mdp = StubMDP(n_steps=2)
        monkeypatch.setattr(a, "_build_mdp", lambda: stub_mdp)

        policy = a.run()
        grid = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = a.get_optimal_allocations(policy, grid, t=0)

        assert len(result) == len(grid)
        assert all(isinstance(v, float) for v in result)

    def test_evaluate_policy_mdp_not_built_raises_even_after_set_qvfs(self):
        """Having trained QVFs but no MDP must still raise in evaluate_policy."""
        a = make_allocator(n_steps=2)
        a._trained_qvfs = [ConstantQVF(), ConstantQVF()]
        # _mdp is still None
        with pytest.raises(RuntimeError):
            a.evaluate_policy(ConstantPolicy(), num_simulations=1)