import wandb

import numpy as np

from src.utils.logging import get_log_obj

from datetime import datetime

import torch.nn as nn

from src.environments.gym_game import GymGame, make_env
from src.coach import Coach
from src.amex_mcts import AmEx_MCTS
from src.classic_mcts import ClassicMCTS
from src.config_mcts import parse_args
import random
import pathlib

from src.utils.rule_predictor import RulePredictor, NN


def train_a0(args) -> None:
    """
    Train an AlphaZero agent on the given environment with the specified
    configuration. If specified within the configuration file, the function
    will load in a previous model along with previously generated data.
    :param args:
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_log_obj(args=args, name=args.mcts_engine)
    logger.info("------------Starting AmEx-Zero------------")

    env = make_env(args.env_str, args.max_episode_steps)
    game = GymGame(args, env)
    wandb.config.update(game.env.spec.kwargs)

    net = init_net(game, args)

    rule_predictor = RulePredictor(net=net, args=args)

    if args.mcts_engine == 'AmEx_MCTS':
        mcts_engine = AmEx_MCTS(game=game, args=args,
                                rule_predictor=rule_predictor)
    elif args.mcts_engine == 'ClassicMCTS':
        mcts_engine = ClassicMCTS(game=game, args=args,
                                  rule_predictor=rule_predictor)
    else:
        raise AssertionError(f"p_args.mcts_engine: {args.mcts_engine} not"
                             f" defined")

    coach = Coach(game=game, args=args,  mcts_engine=mcts_engine,
                  run_name=args.experiment_name)

    coach.learn()


def init_net(game, args) -> nn.Module:
    return NN(game)


if __name__ == '__main__':
    p_args = parse_args(a0=True)

    if p_args.experiment_name is None:
        p_args.experiment_name = f"{p_args.env_str}_{np.random.randint(9999)}"

    if p_args.path is None:
        root_path = pathlib.Path(__file__).parent.parent.resolve()
        p_args.path = (
                root_path / 'Experiments' / p_args.env_str /
                f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}_{p_args.seed}"
        )
    p_args.path.mkdir(parents=True)

    wandb.init(project="AmExZero", config=p_args.__dict__, save_code=False,
               mode=p_args.wandb, dir=p_args.path, name=p_args.experiment_name,
               entity='mctsengame')

    train_a0(p_args)
