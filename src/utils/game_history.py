"""
Defines the functionality for the replay-buffer and parameter scheduling.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from src.utils.game import GameState


@dataclass
class GameHistory:
    """
    Data container for keeping track of game trajectories.
    """
    observations: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    MCTS_value_estimation: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    observed_returns: list = field(default_factory=list)
    terminated: bool = False

    def __len__(self) -> int:
        """Get length of current stored trajectory"""
        return len(self.observations)

    def capture(self, state: GameState, pi: np.ndarray,
                r: float, v: float) -> None:
        """Take a snapshot of the current state of the environment and the
        search results"""
        self.observations.append(state.observation)
        self.probabilities.append(pi)
        self.MCTS_value_estimation.append(v)
        self.rewards.append(r)
        self.actions.append(state.action)

    def refresh(self) -> None:
        """Clear all statistics within the class"""
        self.observations.clear()
        self.probabilities.clear()
        self.MCTS_value_estimation.clear()
        self.rewards.clear()
        self.actions.clear()
        self.observed_returns.clear()
        self.terminated = False

    def compute_returns(self, gamma) -> None:
        """Computes the n-step returns assuming that the last recorded snapshot
        was a terminal state
        :param gamma:
        """
        self.observed_returns = [0] * len(self.rewards)
        observed_return = 0

        for t in range(len(self.rewards) - 1, -1, -1):
            observed_return = self.rewards[t] + gamma * observed_return
            self.observed_returns[t] = observed_return
        return
