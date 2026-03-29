"""
Backward induction RL solver for finite-horizon asset allocation MDPs.

Implements the Q-value backward induction algorithm:
  1. At t = T−1 (terminal transition): target = U(W_T)  (direct reward, no bootstrap)
  2. At t < T−1 (non-terminal):        target = γ · max_{a'} Q_{t+1}(s', a')

The resulting per-step Q-value approximators are wrapped in a GreedyPolicy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── QValueApproximator ────────────────────────────────────────────────────────
from approximator import QValueApproximator

# ── Policy ────────────────────────────────────────────────────────────────────
from policy import Policy

# ── MDP / state ───────────────────────────────────────────────────────────────
from mdp import AssetAllocationMDP
from state import AllocationAction, PortfolioState


# ---------------------------------------------------------------------------
# StateSampler
# ---------------------------------------------------------------------------


class StateSampler(ABC):
    """
    Abstract base class for state-space samplers.

    During backward induction, a StateSampler generates the portfolio states
    used to build regression targets at each time step.
    """

    @abstractmethod
    def sample(self, n: int) -> List[PortfolioState]:
        """Return a list of n PortfolioState samples."""
        ...


# ---------------------------------------------------------------------------
# Abstract solver interface
# ---------------------------------------------------------------------------


class RLSolver(ABC):
    """
    Abstract base class for RL solvers.
    Defines the interface that any backward or forward RL algorithm must satisfy,
    making it easy to swap in LSPI, fitted Q-iteration, or model-free methods later.
    """

    @abstractmethod
    def solve(self) -> Sequence[QValueApproximator]:
        """
        Run the full solving procedure.
        Returns a sequence of Q-value approximators, one per time step.
        """
        ...

    @abstractmethod
    def extract_policy(
        self, qvf_per_step: Sequence[QValueApproximator]
    ) -> Policy:
        """
        Extract the greedy policy from a solved sequence of Q-value functions.
        """
        ...


# ---------------------------------------------------------------------------
# Backward induction solver
# ---------------------------------------------------------------------------


class BackwardInductionSolver(RLSolver):
    """
    Backward induction over the finite horizon.

    At each time step t (from T−1 down to 0), fits a Q-value function
    using sampled (state, action, Bellman-target) tuples, then passes the
    fitted function forward as the bootstrap estimate for the previous step.

    Parameters
    ----------
    mdp               : AssetAllocationMDP
        The environment. Its ``step()`` method is called to generate
        one-step transitions for each sampled (state, action) pair.
    initial_qvf       : QValueApproximator
        Prototype approximator. Each time step receives a fresh ``copy()``.
        The prototype itself is never modified.
    state_sampler     : StateSampler
        Generates PortfolioState samples for regression target construction.
    num_state_samples : int
        Number of states to draw per time step. Must be ≥ 1.
    error_tolerance   : float
        Stored for reference / iterative variants; not actively used in the
        single-pass backward induction sweep. Must be ≥ 0.
    gamma             : float
        Discount factor applied to bootstrapped Q-values. Must be in (0, 1].
        Defaults to 1.0 (undiscounted finite-horizon problem).
    """

    def __init__(
        self,
        mdp: AssetAllocationMDP,
        initial_qvf: QValueApproximator,
        state_sampler: StateSampler,
        num_state_samples: int,
        error_tolerance: float,
        gamma: float = 1.0,
    ) -> None:
        if num_state_samples < 1:
            raise ValueError(
                f"num_state_samples must be at least 1, got {num_state_samples}."
            )
        if not (0.0 < gamma <= 1.0):
            raise ValueError(
                f"gamma must be in (0, 1]; got {gamma}."
            )
        if error_tolerance < 0:
            raise ValueError(
                f"error_tolerance must be non-negative; got {error_tolerance}."
            )
        self._mdp = mdp
        self._initial_qvf = initial_qvf
        self._state_sampler = state_sampler
        self._num_state_samples = num_state_samples
        self._error_tolerance = error_tolerance
        self._gamma = gamma

    # ------------------------------------------------------------------
    # RLSolver interface
    # ------------------------------------------------------------------

    def solve(self) -> List[QValueApproximator]:
        """
        Run backward induction from t = T−1 down to t = 0.

        Returns
        -------
        qvf_per_step : List[QValueApproximator]
            Length-T list; index t holds the fitted Q-value approximator
            for time step t.
        """
        T = self._mdp.time_steps
        qvf_per_step: List[Optional[QValueApproximator]] = [None] * T
        next_qvf: Optional[QValueApproximator] = None

        for t in range(T - 1, -1, -1):
            current_qvf = self._fit_one_step(t, next_qvf)
            qvf_per_step[t] = current_qvf
            next_qvf = current_qvf

        return qvf_per_step  # type: ignore[return-value]

    def extract_policy(
        self, qvf_per_step: Sequence[QValueApproximator]
    ) -> Policy:
        """
        Wrap the solved Q-value sequence in a GreedyPolicy.

        At each time step t the policy returns argmax_{a ∈ feasible} Q_t(s, a).
        """
        return GreedyPolicy(
            qvf_per_step=list(qvf_per_step),
            mdp=self._mdp,
        )

    # ------------------------------------------------------------------
    # Core backward-induction step
    # ------------------------------------------------------------------

    def _fit_one_step(
        self,
        t: int,
        next_step_qvf: Optional[QValueApproximator],
    ) -> QValueApproximator:
        """
        Fit Q_t using one-step Bellman targets.

        Target rule
        -----------
        - t+1 is terminal  →  target(s, a) = reward              (= U(W_{t+1}))
        - t+1 is interior  →  target(s, a) = γ · max_{a'} Q_{t+1}(s', a')
          (reward is 0 at non-terminal steps by MDP construction)

        Parameters
        ----------
        t             : current time step being fitted
        next_step_qvf : fitted QVF for step t+1, or None when t+1 is terminal
        """
        is_terminal_transition = self._mdp.is_terminal(t + 1)

        states = self._state_sampler.sample(self._num_state_samples)

        sa_pairs: List[Tuple[PortfolioState, AllocationAction]] = []
        targets: List[float] = []

        for state in states:
            feasible = self._mdp.get_feasible_actions(state)
            if not feasible:
                continue

            for action in feasible:
                next_state, reward = self._mdp.step(state, action, t)

                if is_terminal_transition:
                    # reward = U(W_{t+1}); no future to bootstrap
                    target = reward
                else:
                    assert next_step_qvf is not None, (
                        "_fit_one_step: next_step_qvf must not be None "
                        "for non-terminal transitions."
                    )
                    next_feasible = self._mdp.get_feasible_actions(next_state)
                    bootstrap = (
                        max(next_step_qvf.predict(next_state, a) for a in next_feasible)
                        if next_feasible
                        else 0.0
                    )
                    # reward = 0 at non-terminal step (by MDP construction)
                    target = self._gamma * bootstrap

                sa_pairs.append((state, action))
                targets.append(target)

        new_qvf = self._initial_qvf.copy()
        new_qvf.fit(sa_pairs, targets)
        return new_qvf

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BackwardInductionSolver("
            f"T={self._mdp.time_steps}, "
            f"num_state_samples={self._num_state_samples}, "
            f"gamma={self._gamma})"
        )


# ---------------------------------------------------------------------------
# Greedy policy
# ---------------------------------------------------------------------------


class GreedyPolicy(Policy):
    """
    Time-indexed greedy policy extracted from a solved Q-value sequence.

    At time t, returns argmax_{a ∈ feasible(s)} Q_t(s, a).

    Parameters
    ----------
    qvf_per_step : List[QValueApproximator]
        Fitted Q-value approximators indexed by time step.
    mdp          : AssetAllocationMDP
        Used to enumerate feasible actions at each (state, t) pair.
    """

    def __init__(
        self,
        qvf_per_step: List[QValueApproximator],
        mdp: AssetAllocationMDP,
    ) -> None:
        self._qvf_per_step = qvf_per_step
        self._mdp = mdp

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        """Return the action that maximises Q_t(state, ·) over feasible actions."""
        feasible = self._mdp.get_feasible_actions(state)
        if not feasible:
            raise RuntimeError(
                f"GreedyPolicy: no feasible actions from state "
                f"(wealth={state.wealth:.4f}, allocs={state.allocations}) at t={t}."
            )
        qvf = self._qvf_per_step[t]
        return max(feasible, key=lambda a: qvf.predict(state, a))

    def __repr__(self) -> str:
        return f"GreedyPolicy(T={len(self._qvf_per_step)})"