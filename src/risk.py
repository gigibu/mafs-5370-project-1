# Risk aversion models
from abc import ABC, abstractmethod
from typing import List, Sequence


class RiskAversionModel(ABC):
    """
    Abstract base class for the risk aversion parameter used by utility functions.

    Separating this from the utility function allows the solver to query risk
    aversion independently — for example, to compute the analytical Merton
    fraction alpha* = (mu - r) / (gamma * sigma^2) — without needing to hold a
    full utility object.

    The interface is deliberately minimal: it returns the Arrow-Pratt *relative*
    risk aversion coefficient gamma, which is the natural parameter for both
    CRRAUtility and LogUtility. For ExponentialUtility (CARA), gamma here would
    correspond to the absolute risk aversion coefficient alpha, and the caller is
    responsible for routing the value to the correct utility constructor.

    The optional `wealth` argument is accepted throughout so that future
    wealth-dependent extensions (e.g. habit-formation models) can subclass this
    without breaking the interface.
    """

    @abstractmethod
    def get_gamma(self, t: int, wealth: float = None) -> float:
        """
        Return the risk aversion coefficient at time step t.

        Parameters
        ----------
        t : int
            Discrete time step. Must be non-negative.
        wealth : float, optional
            Current wealth level. Ignored in the two concrete implementations
            below but retained for interface compatibility.

        Returns
        -------
        float
            The risk aversion coefficient, always >= 0.
        """
        ...


class ConstantRiskAversion(RiskAversionModel):
    """
    A single fixed gamma across all time steps and wealth levels.

    This is the standard assumption in classic CRRA-based Merton models:
    the investor's risk preference is stable and does not depend on age,
    wealth, or market conditions.

    Parameters
    ----------
    gamma : float
        The relative risk aversion coefficient. Must be >= 0, in line with
        the constraint enforced by CRRAUtility in utility.py.
        gamma = 0  → risk-neutral (linear utility)
        gamma = 1  → log utility
        gamma > 1  → increasingly risk-averse
    """

    def __init__(self, gamma: float) -> None:
        if gamma < 0:
            raise ValueError(
                f"Risk aversion coefficient gamma must be non-negative, got {gamma}."
            )
        self.gamma = gamma

    def get_gamma(self, t: int, wealth: float = None) -> float:
        """Return the fixed gamma, ignoring t and wealth."""
        if t < 0:
            raise ValueError(f"Time step t must be non-negative, got {t}.")
        return self.gamma


class TimeVaryingRiskAversion(RiskAversionModel):
    """
    Risk aversion that follows a deterministic schedule over the investment horizon.

    Practical usage
    ---------------
    The canonical real-world application is the *glide path* in target-date
    retirement funds (e.g. Vanguard Target Retirement, Fidelity Freedom funds).
    An investor holds a high-risk (low gamma) portfolio early in life and
    gradually shifts to a conservative (high gamma) allocation as retirement
    approaches. The U.S. Department of Labor's Qualified Default Investment
    Alternative (QDIA) regulations are largely built around this pattern.

    Behaviour past the end of the schedule
    ---------------------------------------
    When t >= len(schedule), the last value in the schedule is held constant.
    This models a post-retirement phase where the investor's risk aversion has
    reached its terminal level and no further adjustment is made.

    Parameters
    ----------
    gamma_schedule : Sequence[float]
        An ordered list of gamma values, one per discrete time step.
        Must be non-empty. All values must be >= 0, consistent with
        CRRAUtility's constraint in utility.py.

    Examples
    --------
    A 40-step glide path starting at gamma=2 (growth-oriented) and ending
    at gamma=6 (conservative):

        schedule = [2.0 + (6.0 - 2.0) * t / 39 for t in range(40)]
        model = TimeVaryingRiskAversion(schedule)
    """

    def __init__(self, gamma_schedule: Sequence[float]) -> None:
        schedule = list(gamma_schedule)
        if len(schedule) == 0:
            raise ValueError("gamma_schedule must contain at least one value.")
        for i, gamma in enumerate(schedule):
            if gamma < 0:
                raise ValueError(
                    f"All gamma values must be non-negative; "
                    f"got {gamma} at index {i}."
                )
        self._schedule: List[float] = schedule

    @property
    def horizon(self) -> int:
        """Number of explicitly scheduled time steps."""
        return len(self._schedule)

    def get_gamma(self, t: int, wealth: float = None) -> float:
        """
        Return the scheduled gamma at time t.

        For t within [0, horizon - 1], returns the scheduled value.
        For t >= horizon, clamps to the final scheduled value (terminal phase).

        Parameters
        ----------
        t : int
            Must be non-negative.
        wealth : float, optional
            Ignored; accepted for interface compatibility.
        """
        if t < 0:
            raise ValueError(f"Time step t must be non-negative, got {t}.")
        index = min(t, self.horizon - 1)
        return self._schedule[index]