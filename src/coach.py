"""
Define the base self-play/ data gathering class. This class should work with
any MCTS-based neural network learning algorithm like AlphaZero or MuZero.
Self-play, model-fitting, and pitting is performed sequentially on a
single-thread n this default implementation.
"""
import logging
import copy
import os
import sys
import typing
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque

import numpy as np
from tqdm import trange

from src.utils.game_history import GameHistory
from datetime import datetime
import wandb
from src.utils.logging import get_log_obj


class Coach:
    """
    This class controls the self-play and learning loop.
    """

    def __init__(self, game, args, mcts_engine,
                 run_name) -> None:
        """
        Initialize the self-play class with an environment, an agent to train,
        requisite hyperparameters, an MCTS search engine,
        and an agent-interface.
        :param run_name:
        :param game: Game Implementation of Game class for environment logic.
        :param args Data structure containing parameters for self-play.
        :param mcts_engine: Class  for performing MCTS using the neural_net.
        """

        self.game = game
        self.args = args

        self.step = 0

        # Initialize replay buffer and helper variable
        self.trainExamplesHistory = deque(
            maxlen=self.args.buffer_size)

        # Initialize network and search engine
        self.rule_predictor = mcts_engine.rule_predictor
        self.test_predictor = copy.deepcopy(self.rule_predictor)
        self.mcts = mcts_engine

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"{args.path}/out/logs/{run_name}"
        self.logger = get_log_obj(args=args, name='Coach')

    @staticmethod
    def get_checkpoint_file(iteration: int) -> str:
        """ Helper function to format model checkpoint filenames """
        return f'checkpoint_{iteration}.pth.tar'

    def sample_batch(self, histories: typing.Sequence[GameHistory]) \
            -> typing.Tuple[typing.List, np.ndarray, np.ndarray]:
        """
        Sample a batch of data from the current replay buffer.
        Construct a batch of data-targets for gradient optimization of the
        AlphaZero neural network.

        The procedure samples a list of game and inside-game coordinates of
        length 'batch_size'. This is done uniformly. Using this list of
        coordinates, we sample the according games, and the according points of
        times within the game to generate neural network inputs, targets.

        The targets for the neural network consist of MCTS move probability
        vectors and TD/ Monte-Carlo returns.

        :param histories: List of GameHistory objects. Contains all
        game-trajectories in the replay-buffer.
        :return: Lists of training examples:
            (observations, move-probabilities, TD/MC-returns)
        """
        # Generate coordinates within the replay buffer to sample from.
        obs = []
        probs = []
        returns = []
        for _ in range(self.args.batch_size):
            row = np.random.randint(0, len(histories) - 1)
            col = np.random.randint(0, len(histories[row]) - 1)

            # Collect training examples for AlphaZero: (o_t, pi_t, v_t)
            obs.append(histories[row].observations[col])
            probs.append(histories[row].probabilities[col])
            returns.append(histories[row].observed_returns[col])
        return (obs, np.array(probs, dtype=np.float32),
                np.array(returns, dtype=np.float32))

    def execute_one_game(self, temp=None) -> GameHistory:
        """
        Perform one episode of self-play for gathering data to train neural
        networks on.

        The implementation details of the neural networks/ agents, temperature
        schedule, data storage is kept highly transparent on this side of the
        algorithm. Hence, for implementation details see the specific
        implementations of the function calls.

        At every step we record a snapshot of the state into a GameHistory
        object, this includes the observation, MCTS search statistics,
        performed action, and observed rewards. After the end of the episode,
        we close the GameHistory object and compute internal target values.

        :return: GameHistory Data structure containing all observed states and
        statistics required for network training.
        """
        history = GameHistory()
        state = self.game.getInitialState()
        # Update MCTS visit count temperature according to an episode or weight
        # update schedule.
        if temp is None:
            temp = self.get_temperature()
        while not state.done:
            # Compute the move probability vector and state value using MCTS
            # for the current state of the environment.
            pi, v = self.get_mcts_action(state, temp)

            next_state, r = self.game.getNextState(
                state=state,
                action=state.action,
            )

            history.capture(
                state=state,
                pi=pi,
                r=r,
                v=v
            )

            # Update state of control
            state = next_state
        history.terminated = True
        self.log_mcts_results(history)

        history.compute_returns(gamma=self.args.gamma)
        return history

    def get_mcts_action(self, state, temp):
        self.mcts.clear_tree()
        self.mcts.initialize_root(state)
        pi, v = self.mcts.run_mcts(
            state=state,
            num_mcts_sims=self.args.num_mcts_sims,
            temperature=temp
        )
        state.action = np.random.choice(len(pi), p=pi)
        return pi, v

    def log_mcts_results(self, history):
        # Cleanup environment and GameHistory
        # self.logger.info("Initial guess of NN: ")
        # initial_hash = list(self.mcts.Ps.keys())[0]
        # for i in np.where(self.mcts.valid_moves_for_s[initial_hash])[0]:
        #     if (initial_hash, i) in self.mcts.Qsa:
        #         self.logger.info(
        #             f"""
        #             Ps: {self.mcts.Ps[initial_hash][:]}|
        #             mcts: {history.probabilities[0][:]}|
        #             Qsa: {self.mcts.Qsa[[(initial_hash, q) for i ]]}|
        #             #Ssa: {self.mcts.times_edge_s_a_was_visited[(
        #                 initial_hash, i)]}|"""
        #         )
        pass

    def get_temperature(self):
        try:
            temp = self.args.temp_0 * np.exp(
                self.args.temperature_decay * self.step
            )
        except FloatingPointError:
            temp = self.args.temp_0
        return temp

    def learn(self) -> None:
        """
        Control the data gathering and weight optimization loop. Perform
        'num_selfplay_iterations' iterations of self-play to gather data, each
        of 'num_episodes' episodes. After every self-play iteration, train the
        neural network with the accumulated data. If specified, the previous
        neural network weights are evaluated against the newly fitted neural
        network weights, the newly fitted weights are then accepted based on
        some specified win/ lose ratio. Neural network weights and the replay
        buffer are stored after every iteration. Note that for highly granular
        vision based environments, that the replay buffer may grow to large
        sizes.
        """
        self.load_train_examples(self.step)
        while self.step < self.args.num_training_epochs:
            self.logger.info(f'----------------ITERATION '
                             f'{self.step}----------------')

            wandb_dict = {"iteration": self.step}

            # Gather training data.
            if True:  # self.args.generate_new_training_data:
                wandb_dict["return"], num_new_data = self.gather_data(
                    num_games=self.args.num_selfplay_iterations)
                self.rule_predictor.env_steps += num_new_data
                wandb_dict["env_steps"] = self.rule_predictor.env_steps
                self.save_train_examples(self.step)

            # Update neural network.
            if self.step > self.args.cold_start_iterations:
                wandb_dict["value_loss"], wandb_dict["pi_loss"] = (
                    self.update_network())
                wandb_dict["updates"] = (self.step *
                                         self.args.num_gradient_steps)
                # save model
                if False:
                    self.rule_predictor.save_checkpoint(
                        epoch=self.step,
                        path=self.args.path / 'saved_models' / f"{self.step}.tar"
                    )
                    self.logger.debug(
                        f"Saved checkpoint for epoch {self.step}:"
                        f"{self.args.path}"
                    )
                # test
                if self.step % self.args.test_frequency == 0:
                    self.rule_predictor.net.eval()
                    mean_reward = 0
                    for _ in range(self.args.num_test_epochs):
                        h = self.execute_one_game(temp=0.0)
                        mean_reward += h.observed_returns[0]
                    wandb_dict["test_reward"] = (mean_reward /
                                                 self.args.num_test_epochs)
                    self.rule_predictor.net.train()
            self.step += 1

            wandb.log(wandb_dict)

    def update_network(self) -> typing.Tuple[float, float]:
        # Backpropagation
        pi_train_loss = 0
        value_train_loss = 0
        for _ in range(self.args.num_gradient_steps):
            batch = self.sample_batch(self.trainExamplesHistory)
            value_loss, pi_loss = self.rule_predictor.train(batch)
            value_train_loss += value_loss
            pi_train_loss += pi_loss
        return (value_train_loss / self.args.num_gradient_steps,
                pi_train_loss / self.args.num_gradient_steps)

    def gather_data(self, num_games) -> typing.Tuple[float, int]:
        mean_reward = 0
        num_data_points = 0
        for _ in range(num_games):
            h = self.execute_one_game()
            mean_reward += h.observed_returns[0]
            num_data_points += len(h)
            self.trainExamplesHistory.append(h)
        return mean_reward / num_games, num_data_points

    def save_train_examples(self, iteration: int) -> None:
        """
        Store the current accumulated data to a compressed file using pickle.
        Note that for highly dimensional environments, that the stored files
        may be considerably large and that storing/ loading the data may
        introduce a significant bottleneck to the runtime of the algorithm.
        :param iteration: int Current iteration of the self-play. Used as
        indexing value for the data filename.
        """
        folder = self.args.path / 'saved_models'

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder / f"buffer_{iteration}.examples"
        with open(filename, "wb+") as f:
            Pickler(f, protocol=HIGHEST_PROTOCOL).dump(
                self.trainExamplesHistory)

        # Don't hog up storage space and clean up old
        # (never to be used again) data.
        old_checkpoint = folder / f"buffer_{iteration - 1}.examples"
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def load_train_examples(self, iteration: int) -> None:
        """
        Load in a previously generated replay buffer from the path specified in
        the .json arguments.
        """
        if iteration == 0:
            self.logger.warning("Just starting training. Using empty replay"
                                "buffer.")
        elif len(self.args.replay_buffer_path) >= 1:
            if os.path.isfile(self.args.replay_buffer_path):
                with open(self.args.replay_buffer_path, "rb") as f:
                    self.logger.info(
                        f"Replay buffer {self.args.replay_buffer_path} found."
                        f"Read it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.warning("No replay buffer found. Using an empty "
                                    "one.")
        else:
            folder = self.args.path / 'saved_models'

            filename = folder / f"buffer_{iteration}.examples"

            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    self.logger.info(
                        f"Replay buffer {iteration}  found. Reading it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.warning("No replay buffer found. Using an empty "
                                    "one.")
