"""
Contains logic for performing Monte Carlo Tree Search having access to the
environment.

The class is an adaptation of AlphaZero-General's MCTS search to accommodate
non-adversarial environments (MDPs).
The MCTS returns both the estimated root-value and action
probabilities. The MCTS also discounts backed up rewards given that gamma < 1.

Notes:
 -  Adapted from https://github.com/suragnair/alpha-zero-general and https://github.com/kaesve/muzero/tree/master
"""
import copy
import typing
import numpy as np
from src.environments.gym_game import GymGameState
from src.utils.logging import get_log_obj
import queue
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class GreedyBestFirstSearch:

    def __init__(self, game, args) -> None:
        self.game = game
        self.args = args

        self.logger = get_log_obj(args=args)

    def run_search(self, state: GymGameState, num_mcts_sims) -> typing.Tuple[np.ndarray, float]:
        s_0 = self.game.getHash(state=state)

        priority_queue = queue.PriorityQueue()

        S = []

        q = np.ones(self.game.getActionSize()) * self.args.minimum_reward

        for num_sim in range(num_mcts_sims):
            # select
            if priority_queue.empty():
                break
            item = priority_queue.get()
            v = item.priority
            node = item.item
            moves = self.game.getLegalMoves(node).astype(bool)
            child = None
            for action in range(len(moves)):
                if moves[action]:
                    child, r = self.game.getNextState(node, action)
                    if child.hash not in S:
                        S.append(child.hash)
                        priority_queue.put(PrioritizedItem(self.simulate(child) + r, child))
                        # add parent again when not fully expanded
                        priority_queue.put(
                            PrioritizedItem(v, node))
                        break

            if child is None:
                continue

            # backprop
            while child.previous_state != state:
                child = child.previous_state
            q[child.production_action] = max(v, q[child.production_action])

        return q, max(q)

    def simulate(self, state, num=1):
        return np.mean([self.rollout_gym(state) for _ in range(num)])

    def rollout_gym(self, state):
        env = copy.deepcopy(state.env)
        done = state.done
        ret = 0.0
        gamma = 1.0

        while not done:
            _, r, term, trunc, __ = env.step(env.action_space.sample())
            done = (term or trunc)
            ret += gamma * r
            gamma *= self.args.gamma
        return ret
