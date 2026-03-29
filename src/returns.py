# Return models: riskless and risky distributions
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class RisklessReturnModel(ABC):
    """Interface for the risk-free (cash) return model."""

    @abstractmethod
    def get_rate(self, t: int) -> float:
        """Return the per-period riskless rate at time step t (e.g. 0.02 = 2%)."""
        ...


class RiskyReturnDistribution(ABC):
    """Interface for a single-asset risky return distribution."""

    @abstractmethod
    def sample(self, t: int, rng: Optional[np.random.Generator] = None) -> float:
        """Draw one net return sample (e.g. 0.05 = +5%) for period t."""
        ...

    @property
    @abstractmethod
    def mean(self) -> float:
        """Expected per-period net return."""
        ...

    @property
    @abstractmethod
    def variance(self) -> float:
        """Variance of per-period net return."""
        ...


class MultiAssetReturnDistribution(ABC):
    """Interface for a joint return distribution over n risky assets."""

    @abstractmethod
    def sample(
        self,
        t: int,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[float, ...]:
        """
        Draw one joint net-return sample for all assets at period t.
        Returns a tuple of length n_assets.
        """
        ...

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of risky assets."""
        ...

    @property
    @abstractmethod
    def means(self) -> tuple[float, ...]:
        """Expected per-period net return for each asset."""
        ...

    @property
    @abstractmethod
    def covariance_matrix(self) -> np.ndarray:
        """n×n covariance matrix of per-period returns."""
        ...


# ---------------------------------------------------------------------------
# Concrete riskless models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantRisklessReturn(RisklessReturnModel):
    """
    A flat riskless rate that does not vary over time.

    Parameters
    ----------
    rate : float
        Per-period risk-free rate (e.g. 0.02 = 2%).
        Must satisfy rate > -1 to avoid negative gross return.
    """

    rate: float

    def __post_init__(self) -> None:
        if self.rate <= -1.0:
            raise ValueError(
                f"Risk-free rate must be > -1 (otherwise gross return ≤ 0); "
                f"got {self.rate}."
            )

    def get_rate(self, t: int) -> float:
        return self.rate


@dataclass(frozen=True)
class StepwiseRisklessReturn(RisklessReturnModel):
    """
    A piecewise-constant riskless rate: rates[t] applies at period t.
    If t ≥ len(rates), the last rate is used (flat extrapolation).

    Parameters
    ----------
    rates : tuple[float, ...]
        Per-period risk-free rates, one entry per time step.
    """

    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rates) == 0:
            raise ValueError("rates must contain at least one entry.")

    def get_rate(self, t: int) -> float:
        idx = min(t, len(self.rates) - 1)
        return self.rates[idx]


# ---------------------------------------------------------------------------
# Concrete single-asset risky distribution
# ---------------------------------------------------------------------------


class NormalReturnDistribution(RiskyReturnDistribution):
    """
    i.i.d. normal returns: R_t ~ N(μ, σ²).

    Parameters
    ----------
    mu    : expected net return per period (e.g. 0.08 = 8%).
    sigma : standard deviation per period (e.g. 0.20 = 20%). Must be ≥ 0.
            When sigma = 0, sample() always returns mu (deterministic).
    """

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")
        self._mu = mu
        self._sigma = sigma

    def sample(
        self,
        t: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        generator = rng if rng is not None else np.random.default_rng()
        return float(generator.normal(self._mu, self._sigma))

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return self._sigma ** 2

    def __repr__(self) -> str:
        return f"NormalReturnDistribution(mu={self._mu}, sigma={self._sigma})"


# ---------------------------------------------------------------------------
# Concrete multi-asset risky distribution
# ---------------------------------------------------------------------------


class MultivariateNormalReturnDistribution(MultiAssetReturnDistribution):
    """
    Joint multivariate normal return distribution: R_t ~ MVN(μ, Σ).

    Supports correlated assets via a full covariance matrix.
    When Σ = 0 (zero matrix), sample() returns μ deterministically.

    Parameters
    ----------
    mus : Sequence[float]
        Expected net returns per period for each asset.
    cov : np.ndarray, shape (n, n)
        Covariance matrix. Must be symmetric and positive semi-definite.
    """

    def __init__(
        self,
        mus: Sequence[float],
        cov: np.ndarray,
    ) -> None:
        mus_arr = np.asarray(mus, dtype=float)
        cov_arr = np.asarray(cov, dtype=float)
        n = len(mus_arr)
        if n == 0:
            raise ValueError("mus must specify at least one asset.")
        if cov_arr.shape != (n, n):
            raise ValueError(
                f"Covariance matrix shape {cov_arr.shape} does not match "
                f"number of assets {n}. Expected ({n}, {n})."
            )
        if not np.allclose(cov_arr, cov_arr.T, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(cov_arr)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                "Covariance matrix must be positive semi-definite "
                f"(minimum eigenvalue: {eigenvalues.min():.6g})."
            )
        self._mus: tuple[float, ...] = tuple(float(m) for m in mus_arr)
        self._cov: np.ndarray = cov_arr.copy()

    def sample(
        self,
        t: int,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[float, ...]:
        generator = rng if rng is not None else np.random.default_rng()
        draws = generator.multivariate_normal(np.array(self._mus), self._cov)
        return tuple(float(x) for x in draws)

    @property
    def n_assets(self) -> int:
        return len(self._mus)

    @property
    def means(self) -> tuple[float, ...]:
        return self._mus

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._cov.copy()

    def __repr__(self) -> str:
        return (
            f"MultivariateNormalReturnDistribution("
            f"mus={self._mus}, n_assets={self.n_assets})"
        )