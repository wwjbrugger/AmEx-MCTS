"""
Score Bounded MCTS - Tristan Cazenave and Abdallah Saffidine
"""
import typing
from src.utils.game import GameState
from src.classic_mcts import ClassicMCTS


class ScoreBoundedMCTS(ClassicMCTS):
    """
    This class handles the MCTS tree while having access to the environment
    logic.
    """

    def __init__(self, game, args) -> None:
        """
        Initialize all requisite variables for performing MCTS for AlphaZero.

        :param game: Game Implementation of Game class for environment logic.
        :param args: Data structure containing parameters for the tree search.
        """
        super().__init__(game=game, args=args)
        self.pessimistic_q = {}
        self.optimistic_q = {}

    def clear_tree(self) -> None:
        """ Clear all statistics stored in the current search tree """
        super().clear_tree()
        self.pessimistic_q = {}
        self.optimistic_q = {}

    def initialize_root(self, state: GameState) -> \
            typing.Tuple[bytes, float]:
        s_0_hash, v_0 = super().initialize_root(state)

        self.pessimistic_q[s_0_hash] = self.args.minimum_reward
        self.optimistic_q[s_0_hash] = self.args.maximum_reward

        return s_0_hash, v_0

    def rollout_for_valid_moves(self, a, state_hash,
                                state, path):
        value = super().rollout_for_valid_moves(a, state_hash, state, path)

        # update bounds if next state is done
        next_state = self.Ssa[(state_hash, a)]
        if next_state.done:
            self.pessimistic_q[next_state.hash] = self.Rsa[(state_hash, a)]
            self.optimistic_q[next_state.hash] = self.Rsa[(state_hash, a)]
        else:
            self.pessimistic_q[next_state.hash] = self.args.minimum_reward
            self.optimistic_q[next_state.hash] = self.args.maximum_reward

        return value

    def backup(self, a, state_hash, value):
        mct_return = self.Rsa[(state_hash, a)] + \
                     self.args.gamma * value  # (Discounted) Value of the current node
        if (state_hash, a) in self.Qsa:
            if self.args.risk_seeking:
                self.Qsa[(state_hash, a)] = max(self.Qsa[(state_hash, a)], mct_return)
            else:
                self.Qsa[(state_hash, a)] = (self.times_edge_s_a_was_visited[(state_hash, a)] *
                                             self.Qsa[(state_hash, a)] + mct_return) / \
                                   (self.times_edge_s_a_was_visited[(state_hash, a)] + 1)
            self.times_edge_s_a_was_visited[(state_hash, a)] += 1
        else:  # new node
            self.Qsa[(state_hash, a)] = mct_return
            self.times_edge_s_a_was_visited[(state_hash, a)] = 1

        self.times_s_was_visited[state_hash] += 1

        # update bounds
        next_hash = self.Ssa[(state_hash, a)].hash
        if self.pessimistic_q[state_hash] < self.pessimistic_q[next_hash]:
            self.pessimistic_q[state_hash] = self.pessimistic_q[next_hash]

        if self.optimistic_q[state_hash] > self.optimistic_q[next_hash]:
            max_value = self.pessimistic_q[state_hash]
            # using copy of the actual list for save removal
            for action in range(len(self.valid_moves_for_s[state_hash][:])):
                if (state_hash, action) in self.Ssa:
                    nsa = self.Ssa[(state_hash, action)]

                    if self.optimistic_q[nsa.hash] > max_value:
                        max_value = self.optimistic_q[nsa.hash]
                    elif sum(self.valid_moves_for_s[state_hash]) > 1 and \
                            self.optimistic_q[nsa.hash] < self.pessimistic_q[state_hash]:
                        self.valid_moves_for_s[state_hash, action] = 0
                else:
                    max_value = self.args.maximum_reward
                    break
            self.optimistic_q[state_hash] = max_value

        return mct_return

