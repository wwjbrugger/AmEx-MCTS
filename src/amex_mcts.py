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
import typing
import numpy as np
from src.utils.game import GameState
from src.utils.utils import tie_breaking_argmax
from src.classic_mcts import ClassicMCTS


class AmEx_MCTS(ClassicMCTS):
    """
    This class handles the MCTS tree while having access to the environment
    logic.
    """

    def __init__(self, game, args, **kwargs) -> None:
        """
        Initialize all requisite variables for performing MCTS for AlphaZero.

        :param game: Game Implementation of Game class for environment logic.
        :param args: Data structure containing parameters for the tree search.
        """
        super().__init__(game=game, args=args, **kwargs)
        self.not_completely_explored_moves_for_s = {}
        self.states = {}

    def run_mcts(self, state: GameState, num_mcts_sims,
                 temperature: float) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs 'num_MCTS_sims' simulations of MCTS starting
        from the provided root GameState.

        Before the search we only clear statistics stored inside the tree.
        In this way we ensure that reward bounds get refreshed over time/
        don't get affected by strong reward scaling in past searches.
        This implementation, thus, reuses state transitions from past searches.
        This may influence memory usage.

        Our estimation of the root-value of the MCTS tree search is based on a
        sample average of each backed-up MCTS value.

        Illegal moves are masked before computing the action probabilities.

        :param state: GameState Data structure containing the current state of
        the environment.
        :param num_mcts_sims: The number of simulations to perform
        :param temperature: float Visit count exponentiation factor. A value of
        0 = Greedy, +infinity = uniformly random.
        :return: tuple (pi, v) The move probabilities of MCTS and the estimated
        root-value of the policy.
        """
        # Refresh value bounds in the tree
        self.temperature = temperature

        # Initialize the root variables needed for MCTS.
        s_0_hash, v_0 = self.initialize_root(state=state)

        # Aggregate root state value over MCTS back-propagated values
        mct_return_list = []
        not_completely_explored = np.any(self.not_completely_explored_moves_for_s[state.hash])
        for num_sim in range(num_mcts_sims):
            if not_completely_explored:
                mct_return, not_completely_explored = self._search(
                    state=state
                )
                mct_return_list.append(mct_return)
            else:
                break

        if not_completely_explored:
            # MCTS Visit count array for each edge 'a' from root node 's_0_hash'.
            move_probabilities = self.calculate_move_probabilities(
                s_0_hash,
                self.times_edge_s_a_was_visited
            )
            v = (np.max(mct_return_list) * num_mcts_sims + v_0) / (num_mcts_sims + 1)
        else:
            self.update_full_exploration_to_root_node(state_hash=s_0_hash)
            # MCTS q-values array for each edge 'a' from root node 's_0_hash'.
            move_probabilities = self.calculate_move_probabilities(s_0_hash,
                                                                   self.Qsa,
                                                                   True)
            v = self.Qsa[(s_0_hash, tie_breaking_argmax(move_probabilities))]

        return move_probabilities, v

    def update_full_exploration_to_root_node(self, state_hash):
        """
        After running the simulation, the root node of the simulation be fully explored.
         However, its parent node is not informed about this,so we are addressing this issue now.
        :param state_hash:
        :return:
        """
        if not self.states[state_hash].previous_state is None:
            state = self.states[state_hash]
            # check if parent node think the child node is not full explored
            if self.not_completely_explored_moves_for_s[state.previous_state.hash][state.production_action]:
                self.not_completely_explored_moves_for_s[state.previous_state.hash][state.production_action] = False
                if np.any(self.not_completely_explored_moves_for_s[state.previous_state.hash]):
                    self.update_full_exploration_to_root_node(state_hash=state.previous_state.hash)

    def clear_tree(self) -> None:
        """ Clear all statistics stored in the current search tree """
        super().clear_tree()
        self.not_completely_explored_moves_for_s = {}
        self.states = {}

    def initialize_root(self, state: GameState) -> \
            typing.Tuple[bytes, float]:
        """
        Perform initial inference for the root state.
        Additionally, mask the illegal moves in the network prior and
        initialize all statistics for starting the MCTS search.

        :param state: GameState Data structure containing the current state of
        the environment.
        :return: tuple (hash, root_value) The hash of the environment state
        and inferred root-value.
        """
        s_0_hash = self.game.getHash(state=state)
        if s_0_hash in self.states:
            # state was already visited before
            v_0 = 0
        else:
            self.states[s_0_hash] = state
            state.previous_state = None
            state.production_action = None

            self.Ps[s_0_hash], v_0 = self.get_prior_and_value(state)
            # Mask the prior for illegal moves, and re-normalize accordingly.
            self.valid_moves_for_s[s_0_hash] = self.game.getLegalMoves(state).astype(bool)
            self.not_completely_explored_moves_for_s[s_0_hash] = \
                self.game.getLegalMoves(state).astype(bool)

            self.Ps[s_0_hash] *= self.valid_moves_for_s[s_0_hash]
            self.Ps[s_0_hash] = self.Ps[s_0_hash] / (1e-8 + np.sum(self.Ps[s_0_hash]))

            # Sum of visit counts of the edges/ children and legal moves.
            self.times_s_was_visited[s_0_hash] = 0

        return s_0_hash, v_0

    def _search(self, state: GameState,
                path: typing.Tuple[int, ...] = tuple(), ) -> (float, bool):
        """
        Recursively perform MCTS search inside the actual environments with
        search-paths guided by the PUCT formula.

        Selection chooses an action for expanding/ traversing the edge (s, a)
        within the tree search.

        If an edge is expanded, we perform a step within the environment (with
        action a) and observe the state transition, reward, and infer the new
        move probabilities, and state value. If an edge is traversed, we simply
        look up earlier inferred/observed values from the class dictionaries.

        During backup, we update the current value estimates of an edge Q(s, a)
        using an average. Note that backed-up values get discounted for
        gamma < 1.

        The actual search-path 'path' is kept as a debugging-variable, it
        currently has no practical use. This method may raise a recursion error
        if the environment creates cycles, this should be highly improbable for
        most environments. If this does occur, the environment can be altered
        to terminate after n visits to some cycle.

        :param state: GameState Numerical prediction of the state by the
        encoder/ dynamics model.
        :param path: tuple of integers representing the tree search-path of the
         current function call.
        :return: float The backed-up discounted/ Monte-Carlo returns (dependent
         on gamma) of the tree search.
        :raises RecursionError: When cycles occur within the search path, the
        search can get stuck *ad infinitum*.
        """
        state_hash = self.game.getHash(state=state)
        # SELECT
        a, a_max = self.select_action_with_highest_upper_confidence_bound(state_hash)
        # EXPAND and SIMULATE
        if (state_hash, a) not in self.Ssa:
            value = self.rollout_for_valid_moves(
                a=a,
                state_hash=state_hash,
                state=state,
                path=path
            )
            pass

        elif not self.Ssa[(state_hash, a)].done:
            # walk known part of the net
            value, _ = self._search(
                state=self.Ssa[(state_hash, a)],
                path=path + (a,)
            )

        else:  # is in Ssa and done
            raise RuntimeError(f"State is in Ssa and done."
                               f"This should not happen! "
                               f"State hash: {state_hash}, action: {a} "
                               f"Ssa entry {self.Ssa[(state_hash, a)].hash}")

        # BACKUP
        mct_return = self.backup(a=a,
                                 state_hash=state_hash,
                                 value=value,
                                 a_max=a_max)

        not_subtree_completed = np.any(self.not_completely_explored_moves_for_s[state_hash])

        if not not_subtree_completed and state.previous_state is not None:
            self.Qsa[(state.previous_state.hash, state.production_action)] = \
              self.args.gamma * np.max([self.Qsa[(state_hash, action)] for action, valid in enumerate(self.valid_moves_for_s[state_hash]) if valid])

            self.not_completely_explored_moves_for_s[state.previous_state.hash][state.production_action] \
                &= not_subtree_completed
        return mct_return, not_subtree_completed

    def rollout_for_valid_moves(self, a, state_hash,
                                state, path):
        # explore new part of the tree
        value = 0
        next_state, reward = self.game.getNextState(
            state=state,
            action=a
        )
        next_state_hash = self.game.getHash(state=next_state)
        if next_state_hash in self.states:
            # We are visiting a state we already explored before
            # This path is closed and the statistics of the existing
            # path is returned
            self.not_completely_explored_moves_for_s[state_hash][a] = False
            next_state = self.states[next_state_hash]
            self.Ssa[(state_hash, a)] = next_state
            previous_state = next_state.previous_state
            production_action = next_state.production_action
            self.Rsa[(state_hash, a)] = reward

            if previous_state is not None:  # start node has no previous state
                return self.Qsa[(previous_state.hash, production_action)]
            else:
                return 0.0  # todo return something universally good
                # return np.max([self.Qsa[(s, a)] for a in self.game.getLegalMoves(s)])
        # Transition statistics.
        self.Rsa[(state_hash, a)] = reward
        self.Ssa[(state_hash, a)] = next_state
        self.times_s_was_visited[next_state_hash] = 0
        self.visits_roll_out += 1
        self.states[next_state_hash] = next_state

        # Inference for non-terminal nodes.
        if not next_state.done:
            # Build network input for inference.
            prior, value = self.get_prior_and_value(state=next_state)
            self.Ps[next_state_hash] = prior
            self.valid_moves_for_s[next_state_hash] = self.game.getLegalMoves(
                state=next_state
            ).astype(bool)
            self.not_completely_explored_moves_for_s[next_state_hash] =\
                self.game.getLegalMoves(
                state=next_state
            ).astype(bool)
            if self.args.depth_first_search:
                value_search, not_subtree_completed = self._search(
                    state=next_state,
                    path=path + (a,)
                )
                if not_subtree_completed:
                    value = (value_search + value) / 2
                else:
                    value = value_search
        else:
            # next state is done
            self.not_completely_explored_moves_for_s[next_state_hash] = [False] * self.action_size
            self.not_completely_explored_moves_for_s[state_hash][a] = False
            self.times_s_was_visited[next_state_hash] += 1  # debug value
            if reward >= 0.98:
                self.states_explored_till_perfect_fit = len(self.times_s_was_visited)
        return value

    def backup(self, a, state_hash, value, a_max):
        mct_return = self.Rsa[(state_hash, a)] + \
                     self.args.gamma * value  # (Discounted) Value of the current node
        if (state_hash, a) in self.Qsa:
            if self.args.risk_seeking:
                self.Qsa[(state_hash, a)] = max(self.Qsa[(state_hash, a)], mct_return)
            else:
                # update path for a_select
                self.Qsa[(state_hash, a)] = (self.times_edge_s_a_was_visited[(state_hash, a)] *
                                             self.Qsa[(state_hash, a)] + mct_return) / \
                                   (self.times_edge_s_a_was_visited[(state_hash, a)] + 1)

                # but do not backprop worse values than what would've been done
                if a != a_max and mct_return < self.Qsa[(state_hash, a_max)]:
                    mct_return = self.Qsa[(state_hash, a_max)]

            self.times_edge_s_a_was_visited[(state_hash, a_max)] += 1
        else:
            self.Qsa[(state_hash, a)] = mct_return
            self.times_edge_s_a_was_visited[(state_hash, a)] = 0  # initialize
            self.times_edge_s_a_was_visited[(state_hash, a_max)] = 1
        self.times_s_was_visited[state_hash] += 1
        return mct_return

    def select_action_with_highest_upper_confidence_bound(self, state_hash):
        confidence_bounds = []
        for a in range(self.action_size):
            ucb = self.compute_ucb(state_hash, a)
            confidence_bounds.append(ucb)
        confidence_bounds = np.asarray(confidence_bounds)

        # Get masked argmax.
        a = tie_breaking_argmax(np.where(self.not_completely_explored_moves_for_s[state_hash],
                                confidence_bounds,
                                -np.inf))  # never choose these actions!
        # Get valid arg_max
        a_max = tie_breaking_argmax(np.where(self.valid_moves_for_s[state_hash],
                                    confidence_bounds,
                                    -np.inf))  # never choose these actions!

        # for the unlikely event that a and a_max should be the same,
        # but because of the tie_breaking_argmax are not we have to check if
        # a_max really already exist
        if not (state_hash, a_max) in self.Qsa:
            a = a_max
        return a, a_max
