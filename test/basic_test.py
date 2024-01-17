import unittest

import wandb

from src.start_mcts import run_gym, run_equation
from definitions import ROOT_DIR


class BasicTest(unittest.TestCase):
    def setUp(self) -> None:
        class Namespace:
            pass
        self.args = Namespace()
        self.args.gamma = 0.99
        self.args.c1 = 1.41421356237
        self.args.depth_first_search = False
        self.args.risk_seeking = True  # uses max instead of mean
        self.args.render_mode = 'human'
        self.args.wandb = 'disabled'
        self.args.logging_level = 20
        self.args.prior_source = 'uniform'
        self.args.num_MCTS_sims = 500
        self.args.use_puct = False
        self.args.seed = 42
        self.args.minimum_reward = 0

    def test_equation_AmEx_MCTS(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "AmEx_MCTS"
        self.args.data_path = 'data/nguyen_11'
        self.args.prior_source = 'grammar'
        self.args.max_elements_in_list = 30
        self.args.max_len_datasets = 10
        self.args.max_depth_of_tree = 6
        self.args.max_branching_factor = 2
        self.args.minimum_reward = -1
        self.args.maximum_reward = 1.0
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_equation(self.args)

    def test_equation_AmEx_MCTS_rollout(self):
        self.args.env_str = 'Equation'
        self.args.mcts_engine = "AmEx_MCTS"
        self.args.data_path = 'data/nguyen_11'
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
        self.args.data_path = 'data/nguyen_11'
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
        
    def test_Chain_v0_AmEx_MCTS(self):
        self.args.env_str = 'Chain-v0'
        self.args.mcts_engine = "AmEx_MCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_Chain_v0_ClassicMCTS(self):
        self.args.env_str = 'Chain-v0'
        self.args.mcts_engine = "ClassicMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_ChainLoop_v0_AmEx_MCTS(self):
        self.args.env_str = 'ChainLoop-v0'
        self.args.mcts_engine = "AmEx_MCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)

    def test_ChainLoop_v0_ClassicMCTS(self):
        self.args.env_str = 'ChainLoop-v0'
        self.args.mcts_engine = "ClassicMCTS"
        wandb.init(project="MCTSEndgame", config=self.args,
                   mode=self.args.wandb)
        result, disc = run_gym(self.args)
