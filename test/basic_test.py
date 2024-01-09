import unittest

import wandb

from src.start_mcts import run_gym, run_equation
from definitions import ROOT_DIR


class BasicTest(unittest.TestCase):
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
        self.args.prior_source = 'uniform'
        self.args.num_MCTS_sims = 500
        self.args.use_puct = False

    def test_equation_EndgameMCTS(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "EndgameMCTS"
        self.args.data_path = 'data/nguyen'
        self.args.prior_source = 'grammar'
        self.args.max_elements_in_list = 30
        self.args.max_len_datasets = 10
        self.args.max_depth_of_tree = 6
        self.args.max_branching_factor = 2
        self.args.minimum_reward = -1
        self.args.maximum_reward = 9.0
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_equation(self.args)

    def test_equation_EndgameMCTS_rollout(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "EndgameMCTS"
        self.args.data_path = 'data/nguyen'
        self.args.prior_source = 'grammar'
        self.args.max_elements_in_list = 30
        self.args.max_len_datasets = 10
        self.args.max_depth_of_tree = 6
        self.args.max_branching_factor = 2
        self.args.minimum_reward = -1
        self.args.maximum_reward = 9.0
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_equation(self.args)

    def test_equation_ClassicMCTS(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "ClassicMCTS"
        self.args.data_path = 'data/nguyen'
        self.args.prior_source = 'grammar'
        self.args.max_elements_in_list = 30
        self.args.max_len_datasets = 10
        self.args.max_depth_of_tree = 6
        self.args.max_branching_factor = 2
        self.args.minimum_reward = -1
        self.args.maximum_reward = 9.0
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_equation(self.args)
        
    def test_Chain_v0_EndgameMCTS(self):
        self.args.env_str = 'Chain-v0'
        self.args.mcts_engine = "EndgameMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_Chain_v0_ClassicMCTS(self):
        self.args.env_str = 'Chain-v0'
        self.args.mcts_engine = "ClassicMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_CartPole_v1_EndgameMCTS(self):
        self.args.env_str = 'ChainLoop-v0'
        self.args.mcts_engine = "EndgameMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_CartPole_v1_ClassicMCTS(self):
        self.args.env_str = 'ChainLoop-v0'
        self.args.mcts_engine = "ClassicMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)
