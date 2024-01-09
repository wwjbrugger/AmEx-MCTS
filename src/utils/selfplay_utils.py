"""
Defines the functionality for prioritized sampling, the replay-buffer, min-max normalization, and parameter scheduling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import typing

import numpy as np
from src.utils.game import GameState


@dataclass
class GameHistory:
    """
    Data container for keeping track of game trajectories.
    """
    observations: list = field(default_factory=list)        # o_t: State Observations
    players: list = field(default_factory=list)             # p_t: Current player
    probabilities: list = field(default_factory=list)       # pi_t: Probability vector of MCTS for the action
    MCTS_value_estimation: list = field(default_factory=list)      # v_t: MCTS value estimation
    rewards: list = field(default_factory=list)             # u_t+1: Observed reward after performing a_t+1
    actions: list = field(default_factory=list)             # a_t+1: Action leading to transition s_t -> s_t+1
    observed_returns: list = field(default_factory=list)    # z_t: Training targets for the value function
    terminated: bool = False                                # Whether the environment has terminated

    def __len__(self) -> int:
        """Get length of current stored trajectory"""
        return len(self.observations)

    def capture(self, state: GameState, pi: np.ndarray, r: float, v: float) -> None:
        """Take a snapshot of the current state of the environment and the search results"""
        self.observations.append(state.observation)
        self.actions.append(state.action)
        self.probabilities.append(pi)
        self.rewards.append(r)
        self.MCTS_value_estimation.append(v)

    def terminate(self, formula_started_from='', found_equation='') -> None:
        """Take a snapshot of the terminal state of the environment"""
        # self.probabilities.append(np.zeros_like(self.probabilities[-1]))
        # self.rewards.append(0)         # Reward past u_T
        # self.MCTS_value_estimation.append(0)
        self.formula_started_from = formula_started_from# Bootstrap: Future possible reward = 0
        self.found_equation = found_equation

        self.terminated = True

    def refresh(self) -> None:
        """Clear all statistics within the class"""
        all([x.clear() for x in vars(self).values() if type(x) == list])
        self.terminated = False

    def compute_returns(self, args, gamma: float = 1, look_ahead: typing.Optional[int] = None) -> None:
        """Computes the n-step returns assuming that the last recorded snapshot was a terminal state
        :param args:
        """
        self.observed_returns = list()
        horizon = len(self.rewards)
        for t in range(len(self.rewards)):
            discounted_rewards = [np.power(gamma, k - t) * self.rewards[k] for k in range(t, horizon)]
            observed_return = sum(discounted_rewards) #+ bootstrap
            if args.average_policy_if_wrong and observed_return < args.maximum_reward - 0.99 :
                self.probabilities[t][self.probabilities[t] > 0] = 1/np.count_nonzero(self.probabilities[t])
            self.observed_returns.append(observed_return)
        return


