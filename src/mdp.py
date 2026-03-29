# MDP definition: single-asset and multi-asset portfolio allocation
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from state import ActionSpace, AllocationAction, PortfolioState
from returns import (
    MultiAssetReturnDistribution,
    RisklessReturnModel,
    RiskyReturnDistribution,
)
from utility import UtilityFunction


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AssetAllocationMDP(ABC):
    """
    Abstract MDP for the discrete-time asset allocation problem.

    The state is a PortfolioState (wealth + price vector + current allocations)
    and the action is an AllocationAction (target allocation fractions).
    Reward is non-zero only at the terminal time step, where it equals U(W_T).
    """

    @abstractmethod
    def step(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        """
        Execute one MDP step, sampling a return from the risky distribution.

        Returns
        -------
        next_state : PortfolioState
            The portfolio state after applying the action and observing returns.
        reward : float
            U(W_{t+1}) if t+1 is the terminal step, else 0.0.
        """
        ...

    @abstractmethod
    def sample_next_state(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        """
        Draw a single (next_state, reward) sample from the transition kernel.
        Semantically identical to step(); provided as a named alias so RL
        solvers can call it without conflating simulation and planning APIs.
        """
        ...

    @abstractmethod
    def is_terminal(self, t: int) -> bool:
        """Return True if t is at or beyond the final time step."""
        ...

    @abstractmethod
    def get_feasible_actions(
        self,
        state: PortfolioState,
    ) -> List[AllocationAction]:
        """
        Return all actions reachable from state, respecting both the
        no-leverage constraint and the per-period rebalancing limit.
        """
        ...


# ---------------------------------------------------------------------------
# Single-asset MDP
# ---------------------------------------------------------------------------


class SingleAssetMDP(AssetAllocationMDP):
    """
    Concrete MDP for one risky asset plus cash.

    Wealth transition:
        W_{t+1} = θ·W_t·(1+R_t) + (1−θ)·W_t·(1+r_t)

    where θ ∈ [0,1] is the fraction of wealth in the risky asset,
    R_t ~ risky_return, and r_t is the riskless rate.

    Price evolution:
        X(t+1) = X(t)·(1+R_t)       with X(0) = 1.

    Parameters
    ----------
    risky_return   : RiskyReturnDistribution
        Distribution of per-period net returns for the risky asset.
    riskless_return : RisklessReturnModel
        Deterministic per-period risk-free rate.
    utility        : UtilityFunction
        Terminal utility function applied to W_T.
    action_space   : ActionSpace
        Discrete allocation grid and feasibility checker (n_assets=1).
    time_steps     : int
        Investment horizon T. Must be ≥ 1. Terminal step is t = T.
    rng            : np.random.Generator, optional
        Seeded RNG for reproducible experiments. A fresh default_rng() is
        created if not supplied.
    """

    def __init__(
        self,
        risky_return: RiskyReturnDistribution,
        riskless_return: RisklessReturnModel,
        utility: UtilityFunction,
        action_space: ActionSpace,
        time_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if time_steps < 1:
            raise ValueError(
                f"time_steps must be at least 1, got {time_steps}."
            )
        self._risky_return = risky_return
        self._riskless_return = riskless_return
        self._utility = utility
        self._action_space = action_space
        self._time_steps = time_steps
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_steps(self) -> int:
        """Investment horizon T."""
        return self._time_steps

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def is_terminal(self, t: int) -> bool:
        """Return True iff t ≥ T (at or past the end of the horizon)."""
        return t >= self._time_steps

    def get_feasible_actions(
        self,
        state: PortfolioState,
    ) -> List[AllocationAction]:
        """Delegate feasibility check to the ActionSpace."""
        return self._action_space.feasible_actions(state)

    def _compute_next_wealth(
        self,
        wealth: float,
        allocation: float,
        risky_return: float,
        riskless_rate: float,
    ) -> float:
        """
        Core wealth-transition equation for one risky asset.

        W_{t+1} = W_t · [θ·(1+R) + (1−θ)·(1+r)]

        Parameters
        ----------
        wealth       : current wealth W_t
        allocation   : fraction θ ∈ [0,1] invested in the risky asset
        risky_return : net return R_t on the risky asset
        riskless_rate: net risk-free rate r_t

        Separated from step() for independent unit-testability.
        """
        return wealth * (
            allocation * (1.0 + risky_return)
            + (1.0 - allocation) * (1.0 + riskless_rate)
        )

    def _transition(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        """
        Shared implementation used by both step() and sample_next_state().

        Raises ValueError if the action is infeasible from state.
        """
        if not self._action_space.is_valid(action, state):
            raise ValueError(
                f"Action {action.allocations} is infeasible from state "
                f"(wealth={state.wealth:.4f}, allocs={state.allocations}) at t={t}. "
                "Check rebalancing limits and no-leverage constraints."
            )

        theta = action.allocations[0]
        r_risky = self._risky_return.sample(t, self._rng)
        r_riskless = self._riskless_return.get_rate(t)

        new_wealth = self._compute_next_wealth(
            state.wealth, theta, r_risky, r_riskless
        )
        new_price = state.prices[0] * (1.0 + r_risky)

        next_state = PortfolioState(
            wealth=new_wealth,
            prices=(new_price,),
            allocations=action.allocations,
        )

        # Reward is non-zero only at the terminal transition
        reward = (
            self._utility.evaluate(new_wealth)
            if self.is_terminal(t + 1)
            else 0.0
        )
        return next_state, reward

    def step(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        return self._transition(state, action, t)

    def sample_next_state(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        return self._transition(state, action, t)

    def __repr__(self) -> str:
        return (
            f"SingleAssetMDP(T={self._time_steps}, "
            f"risky={self._risky_return!r}, "
            f"utility={self._utility!r})"
        )


# ---------------------------------------------------------------------------
# Multi-asset MDP
# ---------------------------------------------------------------------------


class MultiAssetMDP(AssetAllocationMDP):
    """
    Concrete MDP for N risky assets (1 ≤ N ≤ MAX_ASSETS=4) plus cash.

    Wealth transition:
        W_{t+1} = W_t · [Σᵢ θᵢ·(1+Rᵢ_t) + (1−Σθᵢ)·(1+r_t)]

    Price evolution (per asset):
        Xᵢ(t+1) = Xᵢ(t)·(1+Rᵢ_t)     with Xᵢ(0) = 1.

    Joint returns (R₁,...,Rₙ) are drawn from a MultiAssetReturnDistribution
    (typically multivariate normal), supporting full correlation structure.

    Parameters
    ----------
    risky_returns  : MultiAssetReturnDistribution
        Joint distribution of per-period net returns for all risky assets.
    riskless_return : RisklessReturnModel
        Deterministic per-period risk-free rate.
    utility        : UtilityFunction
        Terminal utility function applied to W_T.
    action_space   : ActionSpace
        Must have n_assets matching risky_returns.n_assets.
    time_steps     : int
        Investment horizon T. Must be ≥ 1.
    rng            : np.random.Generator, optional
        Seeded RNG for reproducibility.
    """

    def __init__(
        self,
        risky_returns: MultiAssetReturnDistribution,
        riskless_return: RisklessReturnModel,
        utility: UtilityFunction,
        action_space: ActionSpace,
        time_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if time_steps < 1:
            raise ValueError(
                f"time_steps must be at least 1, got {time_steps}."
            )
        if action_space.n_assets != risky_returns.n_assets:
            raise ValueError(
                f"action_space.n_assets ({action_space.n_assets}) must equal "
                f"risky_returns.n_assets ({risky_returns.n_assets}). "
                "The action space must be built for the same number of assets "
                "as the return distribution."
            )
        self._risky_returns = risky_returns
        self._riskless_return = riskless_return
        self._utility = utility
        self._action_space = action_space
        self._time_steps = time_steps
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_steps(self) -> int:
        """Investment horizon T."""
        return self._time_steps

    @property
    def n_assets(self) -> int:
        """Number of risky assets."""
        return self._risky_returns.n_assets

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def is_terminal(self, t: int) -> bool:
        """Return True iff t ≥ T."""
        return t >= self._time_steps

    def get_feasible_actions(
        self,
        state: PortfolioState,
    ) -> List[AllocationAction]:
        """Delegate feasibility check to the ActionSpace."""
        return self._action_space.feasible_actions(state)

    def _compute_next_wealth(
        self,
        wealth: float,
        allocations: tuple[float, ...],
        risky_returns: tuple[float, ...],
        riskless_rate: float,
    ) -> float:
        """
        Core wealth-transition equation for N risky assets.

        W_{t+1} = W_t · [Σᵢ θᵢ·(1+Rᵢ) + (1−Σθᵢ)·(1+r)]

        Parameters
        ----------
        wealth        : current wealth W_t
        allocations   : fractions (θ₁,...,θₙ) in each risky asset
        risky_returns : sampled net returns (R₁,...,Rₙ)
        riskless_rate : net risk-free rate r_t
        """
        risky_contribution = sum(
            theta * (1.0 + ret)
            for theta, ret in zip(allocations, risky_returns)
        )
        cash_fraction = 1.0 - sum(allocations)
        cash_contribution = cash_fraction * (1.0 + riskless_rate)
        return wealth * (risky_contribution + cash_contribution)

    def _transition(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        """Shared implementation for step() and sample_next_state()."""
        if not self._action_space.is_valid(action, state):
            raise ValueError(
                f"Action {action.allocations} is infeasible from state "
                f"(wealth={state.wealth:.4f}, allocs={state.allocations}) at t={t}. "
                "Check rebalancing limits and no-leverage constraints."
            )

        r_risky: tuple[float, ...] = self._risky_returns.sample(t, self._rng)
        r_riskless: float = self._riskless_return.get_rate(t)

        new_wealth = self._compute_next_wealth(
            state.wealth, action.allocations, r_risky, r_riskless
        )
        new_prices = tuple(
            p * (1.0 + r) for p, r in zip(state.prices, r_risky)
        )

        next_state = PortfolioState(
            wealth=new_wealth,
            prices=new_prices,
            allocations=action.allocations,
        )

        reward = (
            self._utility.evaluate(new_wealth)
            if self.is_terminal(t + 1)
            else 0.0
        )
        return next_state, reward

    def step(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        return self._transition(state, action, t)

    def sample_next_state(
        self,
        state: PortfolioState,
        action: AllocationAction,
        t: int,
    ) -> Tuple[PortfolioState, float]:
        return self._transition(state, action, t)

    def __repr__(self) -> str:
        return (
            f"MultiAssetMDP(n_assets={self.n_assets}, T={self._time_steps}, "
            f"risky={self._risky_returns!r}, utility={self._utility!r})"
        )