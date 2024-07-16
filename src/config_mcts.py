import argparse
from src.utils.parse_args import str2bool
from pathlib import Path


def parse_args(a0=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AmEx-MCTS Experiments')
    parser.add_argument('--experiment-name', type=str,
                        help='The Experiment name', required=False)
    parser.add_argument('--env-str', type=str,
                        help='The gym environment string', required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--num-sims', type=int,
                        dest='num_mcts_sims', required=True,
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
                        choices=['uniform', 'grammar', 'neural_net'])
    parser.add_argument('--use-puct', type=str2bool, required=True,
                        help='Uses the PUCT formula when true, UCB1 otherwise')
    parser.add_argument('--max_elements_in_best_list', type=int,
                        default=10,
                        help='How many of the best results should be saved?')
    parser.add_argument("--path", type=Path,
                        help="path for wandb", required=False)

    #  parameter only needed for gym
    parser.add_argument('--render-mode', type=str,
                        choices=["human", "rgb_array", None], default=None,
                        help='The render mode for the gym environment')

    # parameter only needed for equation discovery
    parser.add_argument("--data-path", type=Path, default="data",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument(
        "--max-len-datasets", type=int,
        help="Number of samples from dataset which is the input into NN"
    )
    parser.add_argument('--max-depth-of-tree', type=int,
                        help='Maximum depth of generated equations')
    parser.add_argument('--max-branching-factor', type=float,
                        help='The maximal number of children in the equation'
                             'tree')
    parser.add_argument('--minimum-reward', type=float,
                        required=True)
    parser.add_argument('--maximum-reward', type=float)
    parser.add_argument('--max_episode_steps', type=int,
                        default=None)

    if a0:
        group = parser.add_argument_group('AlphaZero')
        group.add_argument('--lr', type=float,
                           help='The learning rate', default=1e-3)
        group.add_argument('--weight-decay', type=float,
                           help='Adam weight decay factor', default=0.1)
        group.add_argument('--buffer-size', type=int,
                           help='The replay buffer size', default=None)
        group.add_argument('--num-training-epochs', type=int,
                           help='The number of epochs to train for',
                           default=200)
        group.add_argument('--num-selfplay-iterations', type=int,
                           help='Number of games to play to collect new data',
                           default=3)
        group.add_argument('--cold-start-iterations', type=int,
                           help='Number of games to play before starting'
                                'updating the neural net',
                           default=8)
        group.add_argument('--temp_0', type=float,
                           help='The initial temperature for action selection',
                           default=1.0)
        group.add_argument('--temperature-decay', type=float,
                           help='The temperature decay',
                           default=0.0)
        group.add_argument('--num-gradient-steps', type=int,
                           help='Number of gradient updates per iteration',
                           default=3)
        group.add_argument('--batch-size', type=int,
                           help='The batch size',
                           default=32)
        group.add_argument('--test-frequency', type=int,
                           help='When to test the net',
                           default=5)
        group.add_argument('--num-test-epochs', type=int,
                           help='How many games to play for testing',
                           default=5)

    return parser.parse_args()
