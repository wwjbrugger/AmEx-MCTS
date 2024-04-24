import copy
import typing

import gym
import numpy as np
from pcfg import PCFG
import src.environments
from src.utils.get_grammar import add_prior
from src.utils.game import Game, GameState
import compiler_gym
from compiler_gym.wrappers.core import CompilerEnvWrapper
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ActionType
from typing import Iterable, Optional
from compiler_gym.spaces import Commandline, CommandlineFlag
from src.equation_classes.MaxList import MaxList
import math
class GymGameState(GameState):
    def __init__(self, env, observation, production_action=None,
                 previous_state=None, done=False):
        super().__init__(syntax_tree=None, observation=observation,
                         done=done,
                         production_action=production_action,
                         previous_state=previous_state)
        self.env = env
        self.hash = str(observation["obs"])


    def __del__(self):
        self.env.close()



class GymGame(Game):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.env_str == "CartPole-v1":
            self.env = CartPoleWrapper(gym.make(args.env_str))
        elif args.env_str == "CliffWalking-v0":
            self.env = CliffWrapper(gym.make(args.env_str))
        elif args.env_str in list(gym.envs.registry.keys()):
            self.env = gym.make(args.env_str)

        else:
            self.env = CompilerGymWrapper(
                compiler_gym.make(  # creates a new environment (same as gym.make)
                    "llvm-v0",  # selects the compiler to use
                    benchmark=args.env_str,  # selects the program to compile
                    observation_space="Autophase",  # selects the observation space
                    reward_space="IrInstructionCountOz",  # selects the optimization target
                ),
                max_episode_steps=args.max_episode_steps,
                args=args
            )
        self.env.reset(seed=args.seed)
        a_size = self.getActionSize()

        s = [f"S -> S [{1.0 / a_size}]\n"] * a_size  # uniform prior

        self.grammar = PCFG.fromstring("".join(s))
        self.max_path_length = self.env.spec.max_episode_steps
        add_prior(self.grammar, args)


    def getInitialState(self) -> GymGameState:
        env = self.env
        obs, _ = env.reset()
        return GymGameState(env, {"last_symbol": "S", "obs": obs})

    def getDimensions(self) -> typing.Tuple[int, ...]:
        return self.env.observation_space.shape

    def getActionSize(self) -> int:
        return self.env.action_space.n  # noqa

    def getNextState(self, state: GymGameState, action: int, **kwargs) -> \
            typing.Tuple[GameState, float]:
        env = state.env
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
        return str(state.env.selected_actions)

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


class CompilerGymWrapper(CompilerEnvWrapper):
    max_list = MaxList(10)

    def __init__(self, env: CompilerEnv, max_episode_steps: Optional[int] = None,
                 args = None):
        super().__init__(env=env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.selected_actions = []
        self.args = args
        terminal = CommandlineFlag(
            name="end-of-episode",
            flag="# end-of-episode",
            description="End the episode",
        )
        self.action_space = Commandline(
            items=[
                      CommandlineFlag(
                          name=name,
                          flag=flag,
                          description=description,
                      )
                      for name, flag, description in zip(
                    env.action_space.names,
                    env.action_space.flags,
                    env.action_space.descriptions,
                )
                  ]
                  + [terminal],
            name=f"{type(self).__name__}<{env.action_space.name}>",
        )
        self.terminal_action: int = len(self.action_space.flags) - 1

    def reset(self, seed=None):
        reset_info = super().reset()
        self._elapsed_steps = 0
        self.selected_actions = []
        return (self.selected_actions, {})

    def step(self, action):
        reward = [0.0]
        done = False
        trunc = False
        info = ''
        terminal_action_selected = action == self.terminal_action
        # if terminal_action_selected and len(self.selected_actions) == 0:
        #     return [], 0, True, True, info
        if not terminal_action_selected:
            self.selected_actions.append(action)
        if ((len(self.selected_actions) >= self._max_episode_steps)
        or terminal_action_selected) :
            obs, reward, done, info = self.multistep(
                self.selected_actions,
                observation_spaces=[self.observation_space],
                reward_spaces=[self.reward_space],
                observations=[self.observation_space_spec],
                rewards=[self.reward_space]
            )
            if math.isfinite(reward[0]):
                CompilerGymWrapper.max_list.add(
                    state=self.selected_actions,
                    key=reward
                )

            if len(self.selected_actions) >= self._max_episode_steps:
                trunc = True
            else:
                trunc = False
        if terminal_action_selected:
            self.selected_actions.append(action)
            done= True

        return self.selected_actions, reward[-1], done, trunc, info

    def multistep(self, actions: Iterable[ActionType], **kwargs):
        env = self.create_env()
        actions = list(actions)
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = env.multistep(actions, **kwargs)
        self._elapsed_steps += len(actions)
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        env.close()
        return observation, reward, done, info

    def create_env(self):
        env = CompilerGymWrapper(
            compiler_gym.make(  # creates a new environment (same as gym.make)
                "llvm-v0",  # selects the compiler to use
                benchmark=self.args.env_str,  # selects the program to compile
                observation_space="Autophase",  # selects the observation space
                reward_space="IrInstructionCountOz",  # selects the optimization target
            ),
            max_episode_steps=self.args.max_episode_steps,
            args=self.args
        )
        env.reset(seed=self.args.seed)
        return env
    def fork(self) -> "TimeLimit":
        """Fork the wrapped environment.

        The time limit state of the forked environment is the same as the source
        state.
        """
        fkd = type(self)(env=self.env.fork(), max_episode_steps=self._max_episode_steps)
        fkd._elapsed_steps = self._elapsed_steps  # pylint: disable=protected-access
        return fkd


