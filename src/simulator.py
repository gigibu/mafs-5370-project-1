# src/simulator.py
"""Portfolio path simulator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np

from policy import Policy
from state import AllocationAction, PortfolioState


class StateSampler(ABC):
    """
    Abstract base class for generating a distribution of states at time t.
    Needed by the solver to draw training states at each backward step.
    """

    @abstractmethod
    def sample_state(self, t: int) -> PortfolioState:
        """Draw a single sample of the non-terminal state distribution at time t."""
        ...


class ForwardStateSampler(StateSampler):
    """
    Generates state samples by rolling forward from time 0.
    Uses the MDP's transition dynamics and a random exploration policy,
    mirroring the states_sampler_func approach in the reference codebase.
    """

    def __init__(
        self,
        mdp: "AssetAllocationMDP",
        initial_wealth_distribution: "Distribution",
        action_space: "ActionSpace",
    ) -> None:
        self._mdp = mdp
        self._initial_wealth_dist = initial_wealth_distribution
        self._action_space = action_space

    def sample_state(self, t: int) -> PortfolioState:
        """
        Roll forward from t=0 to t using random actions.

        At t=0, returns the initial state directly without any MDP steps.
        For t>0, takes t random actions through the MDP transition dynamics
        and returns the resulting state.

        Raises ValueError for t < 0 or t >= n_steps.
        """
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}.")
        if t >= self._mdp.n_steps:
            raise ValueError(
                f"t={t} is out of range; MDP has {self._mdp.n_steps} step(s)."
            )

        wealth = float(self._initial_wealth_dist.sample())
        state = self._mdp.initial_state(wealth)

        for step in range(t):
            action = self._action_space.sample()
            state, _ = self._mdp.step(state, action, step)

        return state

    def __repr__(self) -> str:
        return f"ForwardStateSampler(n_steps={self._mdp.n_steps})"


class PortfolioSimulator:
    """
    Simulates full portfolio trajectories under a given policy.
    Used for evaluation, plotting wealth paths, and computing performance
    metrics.  Separated from the solver so simulation concerns do not
    pollute optimisation logic.
    """

    def __init__(
        self,
        mdp: "AssetAllocationMDP",
        policy: Policy,
        initial_wealth: float,
    ) -> None:
        if initial_wealth <= 0.0:
            raise ValueError(
                f"initial_wealth must be strictly positive, got {initial_wealth}."
            )
        self._mdp = mdp
        self._policy = policy
        self._initial_wealth = float(initial_wealth)

    def simulate_path(
        self,
    ) -> List[Tuple[PortfolioState, AllocationAction, float]]:
        """
        Simulate a single trajectory from t=0 to T.

        Returns a list of (state_t, action_t, reward_t) tuples of length
        n_steps, where state_t is the portfolio state before the action at
        step t, action_t is the policy's prescribed allocation, and reward_t
        is the reward received after transitioning to t+1.
        """
        state = self._mdp.initial_state(self._initial_wealth)
        trajectory: List[Tuple[PortfolioState, AllocationAction, float]] = []

        for t in range(self._mdp.n_steps):
            action = self._policy.get_action(state, t)
            next_state, reward = self._mdp.step(state, action, t)
            trajectory.append((state, action, float(reward)))
            state = next_state

        return trajectory

    def simulate_many(
        self, num_paths: int
    ) -> List[List[Tuple[PortfolioState, AllocationAction, float]]]:
        """Simulate num_paths independent trajectories for Monte Carlo analysis."""
        if num_paths < 1:
            raise ValueError(f"num_paths must be >= 1, got {num_paths}.")
        return [self.simulate_path() for _ in range(num_paths)]

    def expected_terminal_utility(self, num_paths: int) -> float:
        """
        Estimate E[U(W_T)] by averaging the terminal reward over simulated
        paths.  The terminal reward is taken as the reward at the last time
        step (index n_steps − 1) in each simulated trajectory.
        """
        if num_paths < 1:
            raise ValueError(f"num_paths must be >= 1, got {num_paths}.")
        paths = self.simulate_many(num_paths)
        terminal_rewards = [path[-1][2] for path in paths]
        return float(np.mean(terminal_rewards))

    def __repr__(self) -> str:
        return (
            f"PortfolioSimulator("
            f"n_steps={self._mdp.n_steps}, "
            f"initial_wealth={self._initial_wealth})"
        )