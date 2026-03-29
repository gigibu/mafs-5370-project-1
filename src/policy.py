# src/policy.py
"""Policy representation for portfolio allocation RL."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from state import AllocationAction, MertonAction, PortfolioState
from approximator import QValueApproximator


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Policy(ABC):
    """Abstract base class for a policy π(s) → a."""

    @abstractmethod
    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        """Return the action prescribed by this policy at state s and time t."""
        ...


# ---------------------------------------------------------------------------
# GreedyQPolicy
# ---------------------------------------------------------------------------

class GreedyQPolicy(Policy):
    """
    A deterministic greedy policy derived from a Q-value function.
    At each (state, time) pair, selects argmax_a Q(s, a).

    Accepts ``qvf_per_step`` as a backward-compatible alias for ``qvfs``.

    Construction-time validation
    ----------------------------
    ``feasible_actions=None`` (or omitted) means "not yet known"; the policy
    is constructed successfully and will raise RuntimeError at call time if
    invoked with no registered actions.

    ``feasible_actions=[]`` (an explicitly empty sequence) is treated as a
    caller error and raises ValueError immediately, because passing a
    zero-length list signals that the caller computed an action set and found
    it empty — almost certainly a bug.
    """

    def __init__(
        self,
        qvfs: Optional[Sequence[QValueApproximator]] = None,
        feasible_actions: Optional[Sequence[AllocationAction]] = None,
        *,
        qvf_per_step: Optional[Sequence[QValueApproximator]] = None,
    ) -> None:
        # ── Resolve qvf_per_step alias ────────────────────────────────
        if qvfs is None and qvf_per_step is not None:
            qvfs = qvf_per_step
        if qvfs is None:
            qvfs = []

        if len(qvfs) == 0:
            raise ValueError(
                "qvf_per_step must contain at least one QValueApproximator."
            )

        # Distinguish None (not provided) from [] (explicitly empty).
        # An explicitly empty sequence is almost always a caller bug.
        if feasible_actions is not None and len(feasible_actions) == 0:
            raise ValueError(
                "feasible_actions must contain at least one AllocationAction."
            )

        self._qvf_per_step: list[QValueApproximator] = list(qvfs)
        self._feasible_actions: list[AllocationAction] = (
            list(feasible_actions) if feasible_actions is not None else []
        )

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        """Return the greedy action argmax_a Q_t(s, a) for time step t."""
        if not (0 <= t < len(self._qvf_per_step)):
            raise IndexError(
                f"Time step t={t} is out of range for "
                f"{len(self._qvf_per_step)} time step(s)."
            )
        if not self._feasible_actions:
            raise RuntimeError(
                f"GreedyQPolicy has no feasible actions registered; "
                f"cannot select an action at t={t}."
            )
        qvf = self._qvf_per_step[t]
        return max(
            self._feasible_actions,
            key=lambda a: qvf.evaluate(state, a),
        )

    def __repr__(self) -> str:
        return (
            f"GreedyQPolicy("
            f"n_steps={len(self._qvf_per_step)}, "
            f"n_actions={len(self._feasible_actions)})"
        )


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class RandomPolicy(Policy):
    """Uniformly random policy over the action space."""

    def __init__(self, action_space: "ActionSpace") -> None:  # type: ignore[name-defined]
        self._action_space = action_space

    def get_action(self, state: PortfolioState, t: int) -> AllocationAction:
        return self._action_space.sample()

    def __repr__(self) -> str:
        return f"RandomPolicy(action_space={self._action_space!r})"


# ---------------------------------------------------------------------------
# AnalyticalMertonPolicy
# ---------------------------------------------------------------------------

class AnalyticalMertonPolicy(Policy):
    """Closed-form Merton (1969/1971) constant-fraction policy."""

    def __init__(
        self,
        mu: float,
        r: float,
        sigma: float,
        gamma: float,
    ) -> None:
        if sigma <= 0:
            raise ValueError(
                f"sigma (volatility) must be strictly positive; got {sigma}."
            )
        if gamma <= 0:
            raise ValueError(
                f"gamma (risk-aversion coefficient) must be strictly positive; "
                f"got {gamma}."
            )
        self._mu    = float(mu)
        self._r     = float(r)
        self._sigma = float(sigma)
        self._gamma = float(gamma)

    def optimal_fraction(self) -> float:
        """π* = (μ − r) / (σ² · γ).  Raw value, no clipping."""
        return (self._mu - self._r) / (self._sigma ** 2 * self._gamma)

    def get_action(self, state: PortfolioState, t: int) -> MertonAction:
        raw_fraction = self.optimal_fraction()
        n_assets     = len(state.allocations)
        allocations  = (raw_fraction,) + (0.0,) * (n_assets - 1)
        return MertonAction(allocations=allocations)

    def __repr__(self) -> str:
        return (
            f"AnalyticalMertonPolicy("
            f"mu={self._mu}, r={self._r}, "
            f"sigma={self._sigma}, gamma={self._gamma})"
        )