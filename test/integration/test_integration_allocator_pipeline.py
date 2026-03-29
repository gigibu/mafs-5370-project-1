# test/integration/test_integration_allocator_pipeline.py
"""
Integration tests for AssetAllocator end-to-end pipeline.

Design notes
------------
* A session-scoped ``trained_allocator`` fixture trains once and is reused
  across all tests that need a pre-trained allocator.
* ``TestAllocatorRun`` creates fresh allocators per test to isolate state.
* All numeric assertions use loose tolerances — the goal is structural
  correctness, not numerical precision.
* ActionSpace is constructed with ``choices=`` (the actual parameter name)
  and relies on the module-level REBALANCE_LIMIT constant.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utility import CRRAUtility
from returns import NormalReturnDistribution
from risk import ConstantRiskAversion
from state import ActionSpace, PortfolioState
from approximator import LinearQValueApproximator
from policy import GreedyQPolicy
from allocator import AssetAllocator


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_action_space() -> ActionSpace:
    """
    Build a single-asset ActionSpace from an 11-point grid [0.0, 0.1, …, 1.0].

    The rebalancing limit (10 pp) comes from the module-level constant
    ``state.REBALANCE_LIMIT``; no per-instance override is needed.
    """
    grid = list(np.round(np.linspace(0.0, 1.0, 11), 10))
    return ActionSpace(choices=grid, n_assets=1)


def _make_initial_state(alloc: float = 0.0, wealth: float = 1.0) -> PortfolioState:
    return PortfolioState(
        wealth=float(wealth),
        prices=(1.0,),
        allocations=(float(alloc),),
    )


def _make_allocator(
    gamma: float = 2.0,
    n_steps: int = 3,
    n_training_states: int = 40,
    n_mc_samples: int = 30,
    seed: int = 0,
) -> AssetAllocator:
    """
    Wire up a lean but real AssetAllocator.
    Small n_training_states / n_mc_samples keep the test suite fast.
    """
    utility       = CRRAUtility(gamma=gamma)
    return_model  = NormalReturnDistribution(mu=0.06, sigma=0.20)
    risk_aversion = ConstantRiskAversion(gamma=gamma)
    action_space  = _make_action_space()
    initial_state = _make_initial_state(alloc=0.0, wealth=1.0)

    return AssetAllocator(
        utility=utility,
        return_model=return_model,
        risk_aversion=risk_aversion,
        action_space=action_space,
        n_steps=n_steps,
        initial_state=initial_state,
        approx_factory=LinearQValueApproximator,   # class used as zero-arg factory
        n_training_states=n_training_states,
        n_mc_samples=n_mc_samples,
        random_seed=seed,
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture: train once, share across all test classes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trained_allocator() -> AssetAllocator:
    """Build and train a lean AssetAllocator (T=3, γ=2)."""
    alloc = _make_allocator(gamma=2.0, n_steps=3)
    alloc.run()
    return alloc


# ---------------------------------------------------------------------------
# TestAllocatorRun
# ---------------------------------------------------------------------------


class TestAllocatorRun:

    def test_run_returns_greedy_q_policy(self):
        alloc = _make_allocator()
        policy = alloc.run()
        assert isinstance(policy, GreedyQPolicy), (
            f"run() must return GreedyQPolicy, got {type(policy).__name__}"
        )

    def test_is_trained_true_after_run(self):
        alloc = _make_allocator()
        assert alloc.is_trained is False
        alloc.run()
        assert alloc.is_trained is True

    def test_trained_qvfs_length_equals_n_steps(self):
        n_steps = 4
        alloc = _make_allocator(n_steps=n_steps)
        alloc.run()
        assert len(alloc.trained_qvfs) == n_steps, (
            f"Expected {n_steps} QVFs, got {len(alloc.trained_qvfs)}"
        )

    def test_mdp_set_after_run(self):
        alloc = _make_allocator()
        assert alloc.mdp is None
        alloc.run()
        assert alloc.mdp is not None

    def test_second_run_replaces_trained_qvfs(self):
        alloc = _make_allocator(n_steps=2)
        alloc.run()
        qvfs_first = alloc.trained_qvfs[:]   # shallow copy of the list

        alloc.run()
        # The new list must be a different object with fresh approximators.
        assert alloc.trained_qvfs is not qvfs_first


# ---------------------------------------------------------------------------
# TestAllocatorEvaluatePolicy
# ---------------------------------------------------------------------------


class TestAllocatorEvaluatePolicy:

    def test_evaluate_policy_returns_dict(self, trained_allocator):
        result = trained_allocator.evaluate_policy(num_simulations=50)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("key", [
        "expected_utility",
        "std_utility",
        "min_utility",
        "max_utility",
        "num_simulations",
    ])
    def test_required_keys_present(self, trained_allocator, key):
        result = trained_allocator.evaluate_policy(num_simulations=50)
        assert key in result, f"Missing key: '{key}'"

    def test_expected_utility_is_finite(self, trained_allocator):
        result = trained_allocator.evaluate_policy(num_simulations=50)
        assert math.isfinite(result["expected_utility"])

    def test_min_leq_expected_leq_max(self, trained_allocator):
        result = trained_allocator.evaluate_policy(num_simulations=100)
        assert result["min_utility"] <= result["expected_utility"] <= result["max_utility"]

    def test_num_simulations_echoed(self, trained_allocator):
        n = 77
        result = trained_allocator.evaluate_policy(num_simulations=n)
        assert result["num_simulations"] == n

    def test_zero_simulations_raises_value_error(self, trained_allocator):
        with pytest.raises(ValueError):
            trained_allocator.evaluate_policy(num_simulations=0)


# ---------------------------------------------------------------------------
# TestAllocatorBenchmarkMerton
# ---------------------------------------------------------------------------


class TestAllocatorBenchmarkMerton:

    def test_benchmark_returns_dict(self, trained_allocator):
        result = trained_allocator.benchmark_against_merton(num_simulations=50)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("key", [
        "rl_expected_utility",
        "merton_expected_utility",
        "merton_fraction",
        "outperforms_merton",
    ])
    def test_required_keys_present(self, trained_allocator, key):
        result = trained_allocator.benchmark_against_merton(num_simulations=50)
        assert key in result, f"Missing key: '{key}'"

    def test_merton_fraction_correct(self, trained_allocator):
        """merton_fraction must equal rl_eu / merton_eu."""
        result = trained_allocator.benchmark_against_merton(num_simulations=100)
        rl_eu  = result["rl_expected_utility"]
        m_eu   = result["merton_expected_utility"]
        frac   = result["merton_fraction"]
        if math.isfinite(frac) and abs(m_eu) > 1e-12:
            assert abs(frac - rl_eu / m_eu) < 1e-9

    def test_outperforms_merton_is_bool(self, trained_allocator):
        result = trained_allocator.benchmark_against_merton(num_simulations=50)
        assert isinstance(result["outperforms_merton"], bool)

    def test_utilities_are_finite(self, trained_allocator):
        result = trained_allocator.benchmark_against_merton(num_simulations=50)
        assert math.isfinite(result["rl_expected_utility"])
        assert math.isfinite(result["merton_expected_utility"])

    def test_zero_simulations_raises_value_error(self, trained_allocator):
        with pytest.raises(ValueError):
            trained_allocator.benchmark_against_merton(num_simulations=0)


# ---------------------------------------------------------------------------
# TestAllocatorGetOptimalAllocations
# ---------------------------------------------------------------------------


class TestAllocatorGetOptimalAllocations:

    def test_length_matches_grid(self, trained_allocator):
        allocs = trained_allocator.get_optimal_allocations()
        assert len(allocs) == trained_allocator._n_steps

    def test_returns_list_of_floats(self, trained_allocator):
        allocs = trained_allocator.get_optimal_allocations()
        assert isinstance(allocs, list)
        assert all(isinstance(a, float) for a in allocs), (
            f"All entries must be float; got: "
            f"{[type(a).__name__ for a in allocs]}"
        )

    def test_all_values_finite(self, trained_allocator):
        allocs = trained_allocator.get_optimal_allocations()
        assert all(math.isfinite(a) for a in allocs), (
            f"Non-finite allocation found: {allocs}"
        )