import unittest
from src.visualize.visualize_mcts_tree import visualize_mcts
import wandb
import numpy as np

from src.utils.logging import get_log_obj
from src.mcts_end_game import MCTSEndGame
from src.classic_mcts import ClassicMCTS
from src.utils.get_grammar import read_grammar_file
from src.environments.find_equation_game import FindEquationGame


class VisualizationTest(unittest.TestCase):
    def setUp(self) -> None:
        class Namespace():
            def __init__(self):
                pass
        self.args = Namespace()

        self.args.num_sims = 10
        self.args.gamma = 0.99
        self.args.c1 = 1.25
        self.args.depth_first_search = True
        self.args.risk_seeking = True
        self.args.render_mode = 'human'
        self.args.wandb = 'disabled'
        self.args.logging_level = 20
        self.args.prior_source = 'grammar'
        self.args.num_MCTS_sims = 100


    # Visualization needs Gaphviz to be installed.
    # On Windows the installation is non-trivial to run the test on all devises
    # the test for visualization is commented out
    # Installation guide for PxGraphviz:
    #  https://pygraphviz.github.io/documentation/stable/install.html
    def test_equation_EndgameMCTS(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "ClassicMCTS"
        self.args.data_path = 'data/nguyen'
        self.args.max_elements_in_list = 30
        self.args.max_len_datasets = 10
        self.args.max_depth_of_tree = 6
        self.args.max_branching_factor = 2
        self.args.minimum_reward = -1
        self.args.maximum_reward = 9.0
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        logger = get_log_obj(args=self.args, name="MCTSEndGame")
        grammar = read_grammar_file(args=self.args)
        game = FindEquationGame(
            grammar=grammar,
            args=self.args,
            train_test_or_val='train'
        )

        if self.args.mcts_engine == 'EndgameMCTS':
            mcts_engine = MCTSEndGame(game=game, args=self.args)
        elif self.args.mcts_engine == 'ClassicMCTS':
            mcts_engine = ClassicMCTS(game=game, args=self.args)
        else:
            raise AssertionError(f"args.engine: {self.args.engine} not defined")
        state = game.getInitialState()
        i = 0

        while not state.done:
            pi, v = mcts_engine.runMCTS(
                state=state,
                num_MCTS_sims=self.args.num_MCTS_sims,
                temperature=1.
            )
            a = np.argmax(pi).item()
            next_state, r = game.getNextState(
                state=state,
                action=a,
            )
            state = next_state
        visualize_mcts(mcts_engine)

