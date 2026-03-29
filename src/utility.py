# Utility functions: CRRA, Exponential (CARA), Log
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class UtilityFunction(ABC):
    """
    Interface for a strictly increasing utility function over terminal wealth.
    All concrete implementations must be strictly concave (risk-averse) unless
    explicitly documented otherwise (e.g. risk-neutral linear utility).
    """

    @abstractmethod
    def evaluate(self, wealth: float) -> float:
        """Compute U(W). Raises ValueError if wealth is outside the domain."""
        ...

    @abstractmethod
    def marginal(self, wealth: float) -> float:
        """Compute U'(W) — first derivative with respect to wealth."""
        ...

    @property
    @abstractmethod
    def is_risk_averse(self) -> bool:
        """True iff U''(W) < 0 everywhere in the domain."""
        ...


class CRRAUtility(UtilityFunction):
    """
    Constant Relative Risk Aversion (CRRA) utility:

        U(W) = W^(1−γ) / (1−γ)    for γ ≠ 1
        U(W) = ln(W)               for γ = 1  (log utility)

    The coefficient of relative risk aversion is −W·U''(W)/U'(W) = γ.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient. γ > 0 implies risk-aversion; γ = 0 gives
        linear (risk-neutral) utility; γ = 1 recovers log utility.
    """

    def __init__(self, gamma: float) -> None:
        if gamma < 0:
            raise ValueError(
                f"gamma must be non-negative for a well-behaved utility function; "
                f"got {gamma}."
            )
        self._gamma = gamma

    @property
    def gamma(self) -> float:
        return self._gamma

    def evaluate(self, wealth: float) -> float:
        if wealth <= 0:
            raise ValueError(
                f"CRRA utility requires strictly positive wealth; got {wealth}."
            )
        if math.isclose(self._gamma, 1.0, rel_tol=1e-9):
            return math.log(wealth)
        return (wealth ** (1.0 - self._gamma)) / (1.0 - self._gamma)

    def marginal(self, wealth: float) -> float:
        if wealth <= 0:
            raise ValueError(
                f"CRRA marginal utility requires strictly positive wealth; got {wealth}."
            )
        return wealth ** (-self._gamma)

    @property
    def is_risk_averse(self) -> bool:
        return self._gamma > 0

    def __repr__(self) -> str:
        return f"CRRAUtility(gamma={self._gamma})"


class ExponentialUtility(UtilityFunction):
    """
    Constant Absolute Risk Aversion (CARA) / Exponential utility:

        U(W) = −exp(−α·W) / α

    The coefficient of absolute risk aversion is −U''(W)/U'(W) = α.
    Unlike CRRA, this utility is defined for all W ∈ ℝ (including negative wealth).

    Parameters
    ----------
    alpha : float
        Risk aversion coefficient. Must be strictly positive.
    """

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError(
                f"alpha must be strictly positive for risk-averse CARA utility; "
                f"got {alpha}."
            )
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def evaluate(self, wealth: float) -> float:
        return -math.exp(-self._alpha * wealth) / self._alpha

    def marginal(self, wealth: float) -> float:
        return math.exp(-self._alpha * wealth)

    @property
    def is_risk_averse(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ExponentialUtility(alpha={self._alpha})"


class LogUtility(UtilityFunction):
    """
    Logarithmic utility U(W) = ln(W).

    This is algebraically equivalent to CRRA with γ = 1, but provided as a
    named class for readability. The Kelly criterion is the optimal policy
    under this utility.
    """

    def evaluate(self, wealth: float) -> float:
        if wealth <= 0:
            raise ValueError(
                f"Log utility requires strictly positive wealth; got {wealth}."
            )
        return math.log(wealth)

    def marginal(self, wealth: float) -> float:
        if wealth <= 0:
            raise ValueError(
                f"Log marginal utility requires strictly positive wealth; got {wealth}."
            )
        return 1.0 / wealth

    @property
    def is_risk_averse(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "LogUtility()"