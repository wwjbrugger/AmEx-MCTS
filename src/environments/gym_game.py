import copy
import typing

import gym
import numpy as np
from pcfg import PCFG
import src.environments
from src.utils.get_grammar import add_prior
from src.utils.game import Game, GameState


class GymGameState(GameState):
    def __init__(self, env, observation, production_action=None,
                 previous_state=None, done=False):
        super().__init__(syntax_tree=None, observation=observation,
                         done=done,
                         production_action=production_action,
                         previous_state=previous_state)
        self.env = env
        self.hash = str(observation["obs"])


class GymGame(Game):

    def __init__(self, args):
        super().__init__()
        if args.env_str == "CartPole-v1":
            self.env = CartPoleWrapper(gym.make(args.env_str))
        elif args.env_str == "CliffWalking-v0":
            self.env = CliffWrapper(gym.make(args.env_str))
        else:
            self.env = gym.make(args.env_str)
        a_size = self.getActionSize()
        s = [f"S -> S [{1.0 / a_size}]\n"] * a_size  # uniform prior

        self.grammar = PCFG.fromstring("".join(s))
        self.max_path_length = self.env.spec.max_episode_steps
        add_prior(self.grammar, args)

    def getInitialState(self) -> GymGameState:
        env = copy.deepcopy(self.env)
        obs, _ = env.reset()
        return GymGameState(env, {"last_symbol": "S", "obs": obs})

    def getDimensions(self) -> typing.Tuple[int, ...]:
        return self.env.observation_space.shape

    def getActionSize(self) -> int:
        return self.env.action_space.n  # noqa

    def getNextState(self, state: GymGameState, action: int, **kwargs) -> \
            typing.Tuple[GameState, float]:
        env = copy.deepcopy(state.env)
        obs, reward, terminated, truncated, __ = env.step(action)
        s = GymGameState(env, {"last_symbol": "S", "obs": obs},
                         production_action=action,
                         previous_state=state, done=terminated or truncated)
        return s, reward

    def getLegalMoves(self, state: GameState) -> np.ndarray:
        return np.ones(self.env.action_space.n)  # noqa

    def getGameEnded(self, state: GameState, **kwargs) -> typing.Union[
          float, int]:
        return GameState.done

    def buildObservation(self, state: GameState) -> np.ndarray:
        return state.observation

    def getSymmetries(self, state: GameState, pi: np.ndarray) -> typing.List:
        pass

    def getHash(self, state: GameState) -> typing.Union[str, bytes, int]:
        return state.hash

    def __del__(self):
        self.env.close()


class CartPoleWrapper(gym.Wrapper):
    def step(self, action):
        obs, _, term, trunc, info = super().step(action)
        done = term or trunc
        return obs, -1.005 * done + 0.005, done, trunc, info


class CliffWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        done = term or trunc
        return obs, done + 1 + reward, done, trunc, info
