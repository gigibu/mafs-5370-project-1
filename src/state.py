# State and action representations
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import List, Sequence

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_ASSETS: int = 4
"""Hard upper bound on the number of risky assets, per the problem specification."""

REBALANCE_LIMIT: float = 0.10
"""
Maximum per-asset allocation change allowed in a single time step.
Inspired by transaction-cost / broker-throttle models: the agent cannot
move more than 10 percentage points of wealth into or out of any single
risky asset per period.
"""


# ---------------------------------------------------------------------------
# PortfolioState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioState:
    """
    Complete, self-contained description of the portfolio at a single time step.

    Frozen so it can be used as a dictionary key (e.g. in a tabular value
    function that buckets continuous wealth) or stored in sets.

    Attributes
    ----------
    wealth : float
        Current portfolio value W(t). Must be strictly positive.
    prices : tuple[float, ...]
        Normalised price vector X(t) ∈ ℝⁿ, with X(0) = 1 for each asset.
        Length must satisfy 1 ≤ n ≤ MAX_ASSETS. All entries strictly positive.
    allocations : tuple[float, ...]
        Current allocation fractions θ(t) — the fraction of total wealth
        already invested in each risky asset.  Must satisfy:
          - len(allocations) == len(prices)
          - θᵢ ≥ 0 for all i  (no short-selling in held state)
          - Σθᵢ ≤ 1            (no leverage in held state)
    """

    wealth: float
    prices: tuple[float, ...]
    allocations: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.wealth <= 0:
            raise ValueError(
                f"Wealth must be strictly positive, got {self.wealth}."
            )
        n = len(self.prices)
        if not (1 <= n <= MAX_ASSETS):
            raise ValueError(
                f"Number of assets must be between 1 and {MAX_ASSETS} "
                f"(inclusive), got {n}."
            )
        if len(self.allocations) != n:
            raise ValueError(
                f"allocations length ({len(self.allocations)}) must equal "
                f"prices length ({n})."
            )
        for i, p in enumerate(self.prices):
            if p <= 0:
                raise ValueError(
                    f"All asset prices must be strictly positive; "
                    f"got {p} at index {i}."
                )
        for i, a in enumerate(self.allocations):
            if a < 0:
                raise ValueError(
                    f"Allocation fractions must be non-negative (no short-selling); "
                    f"got {a} at index {i}."
                )
        total = sum(self.allocations)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Total allocation {total:.6f} exceeds 1.0 (no leverage allowed "
                f"in a held portfolio state)."
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Number of risky assets held in this state."""
        return len(self.prices)

    @property
    def cash_fraction(self) -> float:
        """Fraction of wealth held in the risk-free cash account."""
        return 1.0 - sum(self.allocations)


# ---------------------------------------------------------------------------
# AllocationAction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationAction:
    """
    Represents the *target* allocation after a single rebalancing step.

    This class is used by the discrete RL action space and enforces the
    feasibility constraints that apply to any action the agent may select:
    no short-selling and no leverage.  Analytical policies (e.g. Merton)
    that are permitted to produce unconstrained fractions should use the
    ``MertonAction`` subclass instead.

    Attributes
    ----------
    allocations : tuple[float, ...]
        Target fraction of wealth to place in each risky asset.
        Constraints:
          - len ≥ 1
          - all entries ≥ 0   (no short-selling)
          - sum ≤ 1           (no leverage)
    """

    allocations: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.allocations) == 0:
            raise ValueError(
                "AllocationAction must specify at least one asset allocation."
            )
        for i, a in enumerate(self.allocations):
            if a < 0:
                raise ValueError(
                    f"Target allocations must be non-negative (no short-selling); "
                    f"got {a} at index {i}."
                )
        # NOTE: the sum > 1.0 (leverage) check is intentionally absent here.
        # AllocationAction is a plain data container.  Whether a leveraged
        # allocation is admissible is a policy enforced by ActionSpace, not by
        # this dataclass.


    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Number of risky assets targeted by this action."""
        return len(self.allocations)

    @property
    def cash_fraction(self) -> float:
        """Implied cash fraction after rebalancing."""
        return 1.0 - sum(self.allocations)


# ---------------------------------------------------------------------------
# MertonAction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MertonAction(AllocationAction):
    """
    Unconstrained allocation produced by ``AnalyticalMertonPolicy``.

    The Merton (1969/1971) closed-form solution does not impose
    non-negativity or no-leverage constraints: the optimal fraction π*
    can be negative (short the risky asset when μ < r) or greater than 1
    (leverage when the Sharpe ratio is high relative to risk aversion).

    This subclass preserves the ``AllocationAction`` interface so it is
    accepted everywhere an ``AllocationAction`` is expected, but overrides
    ``__post_init__`` to drop the range checks that are only appropriate
    for the discrete RL action space.
    """

    def __post_init__(self) -> None:
        if len(self.allocations) == 0:
            raise ValueError(
                "MertonAction must specify at least one asset allocation."
            )
        # Intentionally no non-negativity or sum ≤ 1 checks:
        # the Merton solution is unconstrained.


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------


class ActionSpace:
    """
    Defines and manages the discrete set of allowable allocation choices.

    The full action space is the Cartesian product of a per-asset allocation
    grid, pre-filtered to remove leveraged combinations (Σθᵢ > 1).  The
    feasible subset from a given state additionally enforces the per-period
    rebalancing limit (REBALANCE_LIMIT = 10 pp per asset).

    Parameters
    ----------
    choices : Sequence[float]
        Discrete grid of per-asset allocation fractions, e.g. [0.0, 0.1, ..., 1.0].
        Duplicates are removed; the list is stored sorted. All values must
        lie in [0, 1].
    n_assets : int, optional
        Number of risky assets (default 1). Must be between 1 and MAX_ASSETS.
    """

    def __init__(self, choices: Sequence[float], n_assets: int = 1) -> None:
        choices = list(choices)
        if len(choices) == 0:
            raise ValueError("choices must contain at least one value.")
        for c in choices:
            if c < 0 or c > 1:
                raise ValueError(
                    f"All allocation choices must lie in [0, 1], got {c}."
                )
        if not (1 <= n_assets <= MAX_ASSETS):
            raise ValueError(
                f"n_assets must be between 1 and {MAX_ASSETS}, got {n_assets}."
            )
        self._choices: List[float] = sorted(set(choices))
        self._n_assets: int = n_assets
        self._all_actions: List[AllocationAction] = self._build_action_set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_action_set(self) -> List[AllocationAction]:
        """
        Enumerate all per-asset allocation combinations and keep only those
        where the total allocation does not exceed 1.0 (no leverage).
        """
        valid: List[AllocationAction] = []
        for combo in product(self._choices, repeat=self._n_assets):
            if sum(combo) <= 1.0 + 1e-9:
                valid.append(AllocationAction(allocations=tuple(combo)))
        return valid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Number of risky assets this action space was built for."""
        return self._n_assets

    def get_choices(self) -> List[float]:
        """Return the sorted, deduplicated per-asset grid values."""
        return list(self._choices)

    def get_all_actions(self) -> List[AllocationAction]:
        """Return all sum-feasible actions (Σθᵢ ≤ 1)."""
        return list(self._all_actions)

    def is_valid(self, action: AllocationAction, state: PortfolioState) -> bool:
        """
        Return True if and only if ``action`` is reachable from ``state``.

        Checks applied in order:
          1. Dimension: action.n_assets == state.n_assets
          2. Non-negativity: all target allocations ≥ 0
          3. No-leverage: Σ target allocations ≤ 1
          4. Rebalancing limit: |θᵢ_new − θᵢ_current| ≤ REBALANCE_LIMIT for every i
        """
        if action.n_assets != state.n_assets:
            return False
        if any(a < 0 for a in action.allocations):
            return False
        if sum(action.allocations) > 1.0 + 1e-9:
            return False
        for new_alloc, current_alloc in zip(action.allocations, state.allocations):
            if abs(new_alloc - current_alloc) > REBALANCE_LIMIT + 1e-9:
                return False
        return True

    def feasible_actions(self, state: PortfolioState) -> List[AllocationAction]:
        """Return every pre-built action that is valid from ``state``."""
        return [a for a in self._all_actions if self.is_valid(a, state)]

    def sample_uniform(self) -> AllocationAction:
        """Sample a uniformly random action from the full sum-feasible set."""
        if not self._all_actions:
            raise RuntimeError("Action space is empty.")
        return random.choice(self._all_actions)

    def sample_feasible(self, state: PortfolioState) -> AllocationAction:
        """Sample a uniformly random action from the rebalancing-feasible set."""
        feasible = self.feasible_actions(state)
        if not feasible:
            raise RuntimeError(
                f"No feasible actions from state with allocations "
                f"{state.allocations}. This should not happen with a "
                "well-constructed grid that includes the current allocation."
            )
        return random.choice(feasible)

    def __len__(self) -> int:
        return len(self._all_actions)

    def __repr__(self) -> str:
        return (
            f"ActionSpace("
            f"n_assets={self._n_assets}, "
            f"grid_size={len(self._choices)}, "
            f"total_actions={len(self._all_actions)})"
        )