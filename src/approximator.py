# src/approximator.py
"""
Q-value function approximation backends.

Both backends follow an **immutable-update pattern**: update() returns a
new fitted instance and never mutates self.  This keeps intermediate QVFs
safe to cache during backward induction.

Feature function contract
-------------------------
Each feature function receives a single flat tuple
    x = (wealth, action_alloc_0, …, action_alloc_{K-1})
and returns one float.  Both approximators build this tuple internally via
_extract_input(), so callers never construct it manually.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .state import AllocationAction, PortfolioState
except ImportError:
    from state import AllocationAction, PortfolioState  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Input extractor
# ---------------------------------------------------------------------------


def _extract_input(
    state: PortfolioState,
    action: AllocationAction,
) -> Tuple[float, ...]:
    """
    Map (PortfolioState, AllocationAction) → flat numeric tuple.

    Layout: (wealth, action_alloc_0, …, action_alloc_{K-1}).
    """
    return (float(state.wealth), *map(float, action.allocations))


# ---------------------------------------------------------------------------
# Default feature functions
# ---------------------------------------------------------------------------

#: Default polynomial feature set used when no feature_functions are supplied
#: to LinearQValueApproximator.  Covers a (wealth, alloc) input tuple and
#: provides enough expressive power for a concave-utility Q-function.
DEFAULT_LINEAR_FEATURES: List[Callable[[Tuple[float, ...]], float]] = [
    lambda x: 1.0,              # bias / intercept
    lambda x: x[0],             # wealth
    lambda x: x[1],             # risky allocation (asset 0)
    lambda x: x[0] ** 2,        # wealth²  (captures concavity in W)
    lambda x: x[1] ** 2,        # allocation²
    lambda x: x[0] * x[1],      # wealth × allocation interaction
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class QValueApproximator(ABC):
    """
    Abstract base class for Q-value function approximation Q(s, a).
    """

    @abstractmethod
    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        """Return the estimated Q-value for a (state, action) pair."""
        ...

    @abstractmethod
    def update(
        self,
        samples: Sequence[Tuple[PortfolioState, AllocationAction, float]],
    ) -> "QValueApproximator":
        """
        Update the approximator given a batch of (state, action, target) samples.
        Returns a new updated instance (immutable-update pattern).
        """
        ...

    @abstractmethod
    def copy(self) -> "QValueApproximator":
        """Return a deep copy of the current approximator."""
        ...


# ---------------------------------------------------------------------------
# DNNSpec
# ---------------------------------------------------------------------------


@dataclass
class DNNSpec:
    """
    Architecture specification for DNNQValueApproximator.
    """

    layer_sizes: List[int]
    activations: Optional[List[str]] = field(default=None)
    output_activation: str = "linear"

    VALID_ACTIVATIONS: frozenset = field(
        default=frozenset({"relu", "tanh", "sigmoid", "linear"}),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.activations is None:
            self.activations = ["relu"] * len(self.layer_sizes)

        if len(self.activations) != len(self.layer_sizes):
            raise ValueError(
                f"lenactivations ({len(self.activations)}) must equal "
                f"len(layer_sizes) ({len(self.layer_sizes)}): provide one "
                f"activation name per hidden layer."
            )

        for s in self.layer_sizes:
            if s < 1:
                raise ValueError(
                    f"layer_sizes must all be >= 1; got {s}."
                )

        for a in list(self.activations) + [self.output_activation]:
            if a not in self.VALID_ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{a}'; "
                    f"valid choices: {self.VALID_ACTIVATIONS}."
                )


# ---------------------------------------------------------------------------
# Linear Q-value approximator
# ---------------------------------------------------------------------------


class LinearQValueApproximator(QValueApproximator):
    """
    Linear function approximation: Q(s, a) = θᵀ φ(s, a).

    Parameters
    ----------
    feature_functions : Sequence of callables, each f : Tuple[float, …] → float.
                        If omitted, ``DEFAULT_LINEAR_FEATURES`` are used —
                        a polynomial basis over (wealth, allocation).
    lambda_reg        : L2 regularisation weight (Ridge regression).
                        0 = ordinary least squares (no regularisation).
    """

    def __init__(
        self,
        feature_functions: Optional[Sequence[Callable[[Tuple[float, ...]], float]]] = None,
        *,
        lambda_reg: float = 0.0,
    ) -> None:
        if lambda_reg < 0.0:
            raise ValueError(f"lambda_reg must be >= 0; got {lambda_reg}.")
        if feature_functions is None:
            feature_functions = DEFAULT_LINEAR_FEATURES
        self._features: List[Callable] = list(feature_functions)
        self._lambda_reg = float(lambda_reg)
        self._theta: np.ndarray = np.zeros(len(self._features), dtype=float)

    # ── QValueApproximator interface ──────────────────────────────────────

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        return float(self._theta @ self._build_phi(state, action))

    def update(
        self,
        samples: Sequence[Tuple[PortfolioState, AllocationAction, float]],
    ) -> "LinearQValueApproximator":
        """
        Fit θ via (regularised) least squares on the provided samples.
        Returns a *new* instance; self is not modified.
        """
        new = self.copy()
        if not samples:
            return new

        Phi = np.array(
            [self._build_phi(s, a) for s, a, _ in samples], dtype=float
        )  # (N, D)
        y = np.array([t for _, _, t in samples], dtype=float)  # (N,)
        D = Phi.shape[1]

        if self._lambda_reg > 0.0:
            A = Phi.T @ Phi + self._lambda_reg * np.eye(D)
            b = Phi.T @ y
            new._theta = np.linalg.solve(A, b)
        else:
            new._theta, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

        return new

    def copy(self) -> "LinearQValueApproximator":
        clone = LinearQValueApproximator(self._features, lambda_reg=self._lambda_reg)
        clone._theta = self._theta.copy()
        return clone

    # ── accessors ─────────────────────────────────────────────────────────

    @property
    def theta(self) -> np.ndarray:
        """Return a copy of the current weight vector θ."""
        return self._theta.copy()

    def __repr__(self) -> str:
        return (
            f"LinearQValueApproximator("
            f"n_features={len(self._features)}, "
            f"lambda_reg={self._lambda_reg})"
        )

    # ── private ───────────────────────────────────────────────────────────

    def _build_phi(
        self,
        state: PortfolioState,
        action: AllocationAction,
    ) -> np.ndarray:
        x = _extract_input(state, action)
        return np.array([f(x) for f in self._features], dtype=float)


# ---------------------------------------------------------------------------
# DNN internals
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_d(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _tanh_d(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _sigmoid_d(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)


_ACT_FN: dict = {
    "relu":    (_relu,    _relu_d),
    "tanh":    (np.tanh,  _tanh_d),
    "sigmoid": (_sigmoid, _sigmoid_d),
    "linear":  (lambda x: x, lambda x: np.ones_like(x)),
}


class _MLPWeights:
    """Mutable container for all layer parameters of a feedforward network."""

    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        self.weights = weights
        self.biases = biases

    def copy(self) -> "_MLPWeights":
        return _MLPWeights(
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases],
        )

    def step(
        self,
        dW_list: List[np.ndarray],
        db_list: List[np.ndarray],
        lr: float,
    ) -> None:
        for i, (dW, db) in enumerate(zip(dW_list, db_list)):
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db


# ---------------------------------------------------------------------------
# DNN Q-value approximator
# ---------------------------------------------------------------------------


class DNNQValueApproximator(QValueApproximator):
    """
    Multi-layer perceptron Q-value approximator trained with mini-batch SGD.
    """

    def __init__(
        self,
        feature_functions: Optional[Sequence[Callable[[Tuple[float, ...]], float]]] = None,
        dnn_spec: Optional[DNNSpec] = None,
        learning_rate: float = 0.1,
        *,
        n_epochs: int = 100,
        batch_size: int = 32,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0; got {learning_rate}.")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1; got {n_epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}.")

        if feature_functions is None:
            feature_functions = DEFAULT_LINEAR_FEATURES
        if dnn_spec is None:
            dnn_spec = DNNSpec(layer_sizes=[32, 16])

        self._features: List[Callable] = list(feature_functions)
        self._spec = dnn_spec
        self._lr = float(learning_rate)
        self._n_epochs = int(n_epochs)
        self._batch_size = int(batch_size)
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng(0)
        )
        self._weights: Optional[_MLPWeights] = None

        n_in = len(self._features)
        self._layer_sizes: List[int] = [n_in] + list(dnn_spec.layer_sizes) + [1]

    def evaluate(self, state: PortfolioState, action: AllocationAction) -> float:
        if self._weights is None:
            return 0.0
        x = self._build_phi(state, action)[np.newaxis, :]
        _, post = self._forward(x)
        return float(post[-1].item())

    def update(
        self,
        samples: Sequence[Tuple[PortfolioState, AllocationAction, float]],
    ) -> "DNNQValueApproximator":
        new = self.copy()
        if not samples:
            return new

        if new._weights is None:
            new._weights = new._init_weights()

        X = np.array(
            [new._build_phi(s, a) for s, a, _ in samples], dtype=float
        )
        y = np.array([t for _, _, t in samples], dtype=float)
        n = len(y)

        for _ in range(new._n_epochs):
            perm = new._rng.permutation(n)
            for start in range(0, n, new._batch_size):
                idx = perm[start : start + new._batch_size]
                X_b, y_b = X[idx], y[idx]
                pre_acts, post_acts = new._forward(X_b)
                dW, db = new._backward(pre_acts, post_acts, y_b, new._weights)
                new._weights.step(dW, db, new._lr)

        return new

    def copy(self) -> "DNNQValueApproximator":
        clone = DNNQValueApproximator(
            self._features,
            self._spec,
            self._lr,
            n_epochs=self._n_epochs,
            batch_size=self._batch_size,
            rng=deepcopy(self._rng),
        )
        clone._layer_sizes = list(self._layer_sizes)
        clone._weights = self._weights.copy() if self._weights is not None else None
        return clone

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    def __repr__(self) -> str:
        return (
            f"DNNQValueApproximator("
            f"n_features={len(self._features)}, "
            f"layers={self._spec.layer_sizes}, "
            f"lr={self._lr})"
        )

    def _build_phi(self, state: PortfolioState, action: AllocationAction) -> np.ndarray:
        x = _extract_input(state, action)
        return np.array([f(x) for f in self._features], dtype=float)

    def _init_weights(self) -> _MLPWeights:
        all_acts = list(self._spec.activations) + [self._spec.output_activation]
        Ws, bs = [], []
        for i, (fan_in, fan_out) in enumerate(
            zip(self._layer_sizes[:-1], self._layer_sizes[1:])
        ):
            scale = (
                np.sqrt(2.0 / fan_in)
                if all_acts[i] == "relu"
                else np.sqrt(1.0 / fan_in)
            )
            W = self._rng.normal(0.0, scale, (fan_in, fan_out))
            b = np.zeros(fan_out, dtype=float)
            Ws.append(W)
            bs.append(b)
        return _MLPWeights(Ws, bs)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_acts = list(self._spec.activations) + [self._spec.output_activation]
        pre_acts: List[np.ndarray] = []
        post_acts: List[np.ndarray] = [X]

        for i, (W, b) in enumerate(zip(self._weights.weights, self._weights.biases)):
            z = post_acts[-1] @ W + b
            pre_acts.append(z)
            fn, _ = _ACT_FN[all_acts[i]]
            post_acts.append(fn(z))

        return pre_acts, post_acts

    def _backward(
        self,
        pre_acts: List[np.ndarray],
        post_acts: List[np.ndarray],
        targets: np.ndarray,
        weights: _MLPWeights,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_acts = list(self._spec.activations) + [self._spec.output_activation]
        n = targets.shape[0]
        n_layers = len(weights.weights)

        dW_list: List[Optional[np.ndarray]] = [None] * n_layers
        db_list: List[Optional[np.ndarray]] = [None] * n_layers

        y_hat = post_acts[-1].squeeze(-1)
        err = (y_hat - targets) / n
        _, act_d = _ACT_FN[all_acts[-1]]
        delta = err[:, np.newaxis] * act_d(pre_acts[-1])

        for i in range(n_layers - 1, -1, -1):
            dW_list[i] = post_acts[i].T @ delta
            db_list[i] = delta.sum(axis=0)
            if i > 0:
                _, act_d = _ACT_FN[all_acts[i - 1]]
                delta = (delta @ weights.weights[i].T) * act_d(pre_acts[i - 1])

        return dW_list, db_list  # type: ignore[return-value]