import argparse
from src.utils.parse_args import str2bool
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AmEx_MCTS Experiments')
    parser.add_argument('--env-str', type=str,
                        help='The gym environment string', required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--num-sims', type=int,
                        dest='num_MCTS_sims', required=True,
                        help='The number of simulations to perform per step')
    parser.add_argument('--gamma', type=float,
                        help='The discount factor', default=0.99)
    parser.add_argument('--c1', type=float, default=1.41421356237,
                        help='The UCB exploration coefficient')
    parser.add_argument('--depth-first-search', type=str2bool,
                        help='Performs a depth first search if true',
                        default=False)
    parser.add_argument('--risk-seeking', type=str2bool,
                        help='Acts risk seeking if true', required=True)
    parser.add_argument('--wandb', type=str,
                        choices=["online", "offline", "disabled"],
                        default="offline",
                        help='The wandb mode')
    parser.add_argument('--logging-level', type=int, default=40,
                        help='The logging level')
    parser.add_argument("--mcts-engine", type=str, required=True,
                        choices=['ClassicMCTS', 'AmEx_MCTS'],
                        help="Which Engine should be used in MCTS?")
    parser.add_argument('--prior-source', type=str, required=True,
                        choices=['uniform', 'grammar'])
    parser.add_argument('--use-puct', type=str2bool, required=True,
                        help='Uses the PUCT formula when true, UCB1 otherwise')
    
    #  parameter only needed for gym
    parser.add_argument('--render-mode', type=str,
                        choices=["human", "rgb_array", None], default=None,
                        help='The render mode for the gym environment')
    
    # parameter only needed for equation discovery
    parser.add_argument("--data-path", type=Path,
                        help="path to preprocessed dataset", required=False)
    parser.add_argument('--max-elements-in-list', type=int,
                        help='How many of the best results should be saved?')
    parser.add_argument(
        "--max-len-datasets", type=int,
        help="Number of samples from dataset which is the input into NN"
    )
    parser.add_argument('--max-depth-of-tree', type=int,
                        help='Maximum depth of generated equations')
    parser.add_argument('--max-branching-factor', type=float,
                        help='The maximal number of children in the equation'
                             'tree')
    parser.add_argument('--minimum-reward', type=float)
    parser.add_argument('--maximum-reward', type=float)

    return parser.parse_args()
