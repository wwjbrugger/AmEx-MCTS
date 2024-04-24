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
from src.utils.game import GameState
from src.utils.logging import get_log_obj
from src.utils.utils import tie_breaking_argmax
import random


class ClassicMCTS:
    """
    This class handles the MCTS tree while having access to the environment
    logic.
    """

    def __init__(self, game, args) -> None:
        """
        Initialize all requisite variables for performing MCTS.
        :param game: Game Implementation of Game class for environment logic.
        :param args: Data structure containing parameters for the tree search.
        """
        self.game = game
        self.args = args

        # Static helper variables.
        self.action_size = game.getActionSize()

        self.Qsa = {}  # stores Q values for s, a
        self.Ssa = {}  # stores state transitions for s, a
        self.Rsa = {}  # stores R values for s, a
        self.times_edge_s_a_was_visited = {}  # stores visit counts of edges
        self.times_s_was_visited = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy
        self.valid_moves_for_s = {}  # stores game.getValidMoves for board s
        self.visits_done_state = 0 # Count visits of done states.
        self.visits_roll_out = 0 # Count how often a new state is explored
        self.logger = get_log_obj(args=args)

        self.temperature = None # exponentiation factor
        self.states_explored_till_perfect_fit = -1

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
        s_0, v_0 = self.initialize_root(state=state)

        # Aggregate root state value over MCTS back-propagated values
        mct_return_list = []
        for num_sim in range(num_mcts_sims):
            mct_return = self._search(
                state=state
            )
            mct_return_list.append(mct_return)

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        move_probabilities = self.calculate_move_probabilities(
            s_0,
            self.times_edge_s_a_was_visited
        )
        v = (np.max(mct_return_list) * num_mcts_sims + v_0) / (num_mcts_sims + 1)
        return move_probabilities, v

    def calculate_move_probabilities(self, s_0, source_dict, softmax=False):
        possible_actions = np.where(self.valid_moves_for_s[s_0])[0]
        action_utilities = np.zeros(len(possible_actions))
        for i, a in enumerate(possible_actions):
            if (s_0, a) in source_dict:
                action_utilities[i] = source_dict[(s_0, a)]

        move_probabilities = np.zeros(self.action_size)

        if self.temperature == 0:  # Greedy selection. One hot encode the most visited paths (randomly break ties).
            max_index = possible_actions[tie_breaking_argmax(action_utilities)]
            move_probabilities[max_index] = 1.0
        else:
            if softmax:
                try:
                    # Qsa
                    action_utilities = np.exp(action_utilities / self.temperature)
                except FloatingPointError:
                    move_probabilities = np.zeros(self.action_size)
                    max_index = possible_actions[tie_breaking_argmax(action_utilities)]
                    move_probabilities[max_index] = 1.0
                    return move_probabilities
            # else:  # counts
            move_probabilities[possible_actions] = action_utilities
            try:
                move_probabilities = np.divide(move_probabilities, np.sum(move_probabilities))
            except FloatingPointError:
                move_probabilities = np.zeros(self.action_size)
                max_index = possible_actions[tie_breaking_argmax(action_utilities)]
                move_probabilities[max_index] = 1.0
                return move_probabilities
        return move_probabilities

    def clear_tree(self) -> None:
        """ Clear all statistics stored in the current search tree """
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ssa = {}  # stores state transitions for s, a
        self.Rsa = {}  # stores R values for s, a
        self.times_edge_s_a_was_visited = {}  # stores visit counts of edges
        self.times_s_was_visited = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.valid_moves_for_s = {}  # stores game.getValidMoves for board s
        self.visits_done_state = 0
        self.visits_roll_out = 0
        self.states_explored_till_perfect_fit = -1

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
        s_0 = self.game.getHash(state=state)

        self.Ps[s_0], v_0 = self.get_prior_and_value(state)
        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.valid_moves_for_s[s_0] = self.game.getLegalMoves(state).astype(bool)

        self.Ps[s_0] *= self.valid_moves_for_s[s_0]
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/children and legal moves.
        self.times_s_was_visited[s_0] = 0

        return s_0, v_0

    def get_prior_and_value(self, state):
        if self.args.prior_source == 'grammar':
            prior = self.game.grammar.prior_dict[state.observation['last_symbol']]
        else:  # elif self.args.prior_source == 'uniform':
            prior = self.game.grammar.prior_dict[state.observation['last_symbol']]

        if self.args.env_str == "Equation":
            value = self.rollout_equation(state)
        else:
            value = self.rollout_gym(state)

        return prior, value

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

        # terminate todo
        if len(path) > self.game.max_path_length:
            return self.args.minimum_reward

        # SELECT
        a = self.select_action_with_highest_upper_confidence_bound(state_hash)
        # EXPAND and SIMULATE
        if (state_hash, a) not in self.Ssa:
            # ask neural net what to do next
            value = self.rollout_for_valid_moves(
                a=a,
                state_hash=state_hash,
                state=state,
                path=path
            )
            pass

        elif not self.Ssa[(state_hash, a)].done:
            # walk known part of the net
            value = self._search(
                state=self.Ssa[(state_hash, a)],
                path=path + (a,)
            )

        else:  # is in Ssa and done
            value = 0
            self.visits_done_state += 1

        # BACKUP
        mct_return = self.backup(
            a=a,
            state_hash=state_hash,
            value=value
        )
        return mct_return

    def rollout_for_valid_moves(self, a, state_hash,
                                state, path):
        # explore new part of the tree
        value = 0
        next_state, reward = self.game.getNextState(
            state=state,
            action=a
        )
        next_state_hash = self.game.getHash(state=next_state)
        # Transition statistics.
        self.Rsa[(state_hash, a)] = reward
        self.Ssa[(state_hash, a)] = next_state
        self.times_s_was_visited[next_state_hash] = 0
        self.visits_roll_out += 1

        # Inference for non-terminal nodes.
        if not next_state.done:
            # Build network input for inference.
            prior, value = self.get_prior_and_value(state=next_state)
            self.Ps[next_state_hash] = prior
            self.valid_moves_for_s[next_state_hash] = self.game.getLegalMoves(
                state=next_state
            )
            # todo maybe delete depth_first
            if self.args.depth_first_search:
                value_search = self._search(
                    state=next_state,
                    path=path + (a,)
                )
                value = (value_search + value) / 2
        else:
            # next state is done
            if reward >= 0.98:
                self.states_explored_till_perfect_fit = len(self.times_s_was_visited)
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
        else:
            self.Qsa[(state_hash, a)] = mct_return
            self.times_edge_s_a_was_visited[(state_hash, a)] = 1
        self.times_s_was_visited[state_hash] += 1
        return mct_return

    def select_action_with_highest_upper_confidence_bound(self, state_hash):
        confidence_bounds = []
        for a in range(self.action_size):
            ucb = self.compute_ucb(state_hash, a)
            confidence_bounds.append(ucb)
        confidence_bounds = np.asarray(confidence_bounds)
        # Get masked argmax.
        a = tie_breaking_argmax(np.where(self.valid_moves_for_s[state_hash],
                                         confidence_bounds,
                                         -np.inf))  # never choose these actions!
        return a

    def compute_ucb(self, state_hash: bytes, a: int) -> float:
        """
        Compute the UCB for an edge (s, a) within the MCTS tree:

            PUCT(s, a) = Q(s, a) + P(s, a) * sqrt(visits_s) / (1 + visits_s')
                        * c1

        Illegal edges are returned as zeros.

        :param state_hash: hash Key of the current state inside the MCTS tree.
        :param a: int Action key representing the path to reach the child node
        from path (s, a)
        :return: float Upper confidence bound with neural network prior
        """
        if state_hash in self.valid_moves_for_s and not self.valid_moves_for_s[state_hash][a]:
            return 0.0  # todo handle transpositions

        if (state_hash, a) in self.Qsa:
            times_s_a_visited = self.times_edge_s_a_was_visited[(state_hash, a)]
            q_value = self.Qsa[(state_hash, a)]
        else:
            times_s_a_visited = 0
            q_value = 0

        if self.args.use_puct:
            # Standard PUCT formula from the AlphaZero paper
            exploration = self.args.c1 * self.Ps[state_hash][a] * np.sqrt(
                self.times_s_was_visited[state_hash] + 1) / (1 + times_s_a_visited)
        else:
            # Standard UCT/UCB1 formula
            if times_s_a_visited == 0:
                return 1e8

            if self.times_s_was_visited[state_hash] == 0:
                denominator = 1.0
            else:
                denominator = np.log(self.times_s_was_visited[state_hash])

            exploration = self.args.c1 * np.sqrt(denominator / times_s_a_visited)

        return q_value + exploration
    
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
        env.close()
        return ret

    def rollout_equation(self, state):
        ret = 0.0
        gamma = 1.0
        while not state.done:
            possible_actions = state.syntax_tree.get_possible_moves(
                node_id=state.syntax_tree.nodes_to_expand[0]
            )
            random_action = random.choice(possible_actions)
            state, r = self.game.getNextState(
                state=state,
                action=random_action
            )
            ret += gamma * r
            gamma *= self.args.gamma
        return ret

    def __del__(self):
        self.clear_tree()
