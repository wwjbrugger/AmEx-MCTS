"""
Define the abstract Game class for providing a structure/ interface for agent environments.

"""
from abc import ABC, abstractmethod
import typing

import numpy as np


class GameState:
    canonical_state: typing.Any     # s_t
    observation: np.ndarray         # o_t
    action: int     # a_t
    player: int     # player_t
    done: bool      # I(s_t = s_T)
    hash_previous_state: str
    hash : str

    def __init__(self, syntax_tree, observation, done=False, hash=None,
                 production_action=None, residual_calculated=False,
                 previous_state=None):
        self.syntax_tree = syntax_tree
        self.observation = observation
        self.done = done
        self.y_calc = None
        self.hash = hash
        self.residual_calculated = residual_calculated
        self.production_action = production_action
        self.previous_state = previous_state

    def __str__(self):
        return self.hash


class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass this class and implement the
    functions below.
    """

    def __init__(self, n_players: int = 1) -> None:
        """
        Initialize the base variables for the Game class.
        :param n_players: int The number of players/ adversaries within the implementation of Game (either 1 or 2)
        :raise NotImplementedError: Error raised for n_players larger than 2.
        """
        self.n_players = n_players
        self.n_symmetries = 1
        if self.n_players > 2:
            raise NotImplementedError(f"Environments for more than 2 agents are not yet supported, {n_players} > 2")
        self.max_path_length = None

    @abstractmethod
    def getInitialState(self) -> GameState:
        """
        Initialize the environment and get the root-state wrapped in a GameState data structure.
        :return: GameState Data structure containing the specifics of the current environment state.
        """

    @abstractmethod
    def getDimensions(self) -> typing.Tuple[int, ...]:
        """
        Get the raw observation dimensions visible for a learning algorithm.
        :return: tuple of integers representing the dimensions of observation data.
        """

    @abstractmethod
    def getActionSize(self) -> int:
        """
        Get the number of atomic actions in the environment.
        :return: int The number of atomic actions in the environment.
        """

    @abstractmethod
    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        """
        Perform an action in the environment and observe the transition and reward.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :param action: int Integer action to perform on the environment.
        :return: tuple containing the next environment state in a GameState object, along with a float reward.
        """

    @abstractmethod
    def getLegalMoves(self, state: GameState) -> np.ndarray:
        """
        Determine the legal moves at the provided environment state.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :return: np.ndarray Array of length |action_space| with 0s for illegal and 1s for legal moves.
        """

    @abstractmethod
    def getGameEnded(self, state: GameState, **kwargs) -> typing.Union[float, int]:
        """
        Determine whether the given state is a terminal state.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :return: float or int Always returns 0 until the game ends, then a terminal reward is returned.
        """

    @abstractmethod
    def buildObservation(self, state: GameState) -> np.ndarray:
        """
        Compute some representation of the GameState, to be used as the input of a neural network.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :return: np.ndarray Some game-specific representation of the current environment state.
        """

    @abstractmethod
    def getSymmetries(self, state: GameState, pi: np.ndarray) -> typing.List:
        """
        @DEPRECATED: future will replace state with GameHistory to get symmetries over observation trajectories.
        Compute every possible symmetry of the provided environment state with correctly oriented pi-vectors.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :param pi: np.ndarray Raw move probability vector of size |action-space|.
        :return: A list of the form [(state, pi)] where each tuple is a symmetrical form of the state and the
                 corresponding pi vector. This can be used for diversifying training examples.
        """

    @abstractmethod
    def getHash(self, state: GameState) -> typing.Union[str, bytes, int]:
        """
        Compute a hashable representation of the provided environment state, h: StateSpace -> Universe
        :param state: GameState Data structure containing the specifics of the current environment state.
        :return: Some hashable datatype representing the provided GameState.
        """

    def close(self, state: GameState) -> None:
        """
        Clean up necessary variables within the environment/ class. If any.
        :param state: GameState Data structure containing the specifics of the current environment state.
        """

    def render(self, state: GameState):
        """
        Base method for generating a visual rendering of the game implementation.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :raises NotImplementedError: Error raised if the child class did not implement a rendering method.
        """
        raise NotImplementedError(f"Render method not implemented for Game: {self}")
