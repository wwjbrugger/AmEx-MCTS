import wandb

import numpy as np

from src.utils.logging import get_log_obj

from src.environments.gym_game import GymGame, make_env
from src.amex_mcts import AmEx_MCTS
from src.classic_mcts import ClassicMCTS
from src.config_mcts import parse_args
from src.utils.get_grammar import read_grammar_file
from src.environments.find_equation_game import FindEquationGame
import random
import time


def run_gym(args) -> (float, float):
    """
    Method to run MCTS approaches for gym environments
    :param args:
    :return: undiscounted_return and  discounted_return of the MCTS
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_log_obj(args=args, name=args.mcts_engine)

    env = make_env(args.env_str, args.max_episode_steps)
    game = GymGame(args, env)
    wandb.config.update(game.env.spec.kwargs)

    if args.mcts_engine == 'AmEx_MCTS':
        mcts_engine = AmEx_MCTS(game=game, args=args)
    elif args.mcts_engine == 'ClassicMCTS':
        mcts_engine = ClassicMCTS(game=game, args=args)
    else:
        raise AssertionError(f"args.engine: {args.engine} not defined")
    state = game.getInitialState()

    i = 0
    undiscounted_return = 0
    discounted_return = 0
    gamma = 1.0
    start_time = time.time()
    next_state = None

    while not state.done:
        pi, v = mcts_engine.run_mcts(state=state,
                                     num_mcts_sims=args.num_MCTS_sims,
                                     temperature=0)

        a = np.argmax(pi).item()
        logger.debug(f"{i}, {a}, {pi}, {v}, {state.observation['obs']}")
        next_state, r = game.getNextState(
            state=state,
            action=a,
        )
        state = next_state
        i += 1
        discounted_return += gamma * r
        undiscounted_return += r
        gamma *= args.gamma
    end_time = time.time()
    wandb.log({
        "mcts_actions: ": next_state.hash,
        "run_time": end_time - start_time
    })

    # Cleanup environment and GameHistory
    env.close()

    # Log results
    logger.debug(f"Return: {undiscounted_return}")
    logger.debug(f"Discounted Return: {discounted_return}")

    return undiscounted_return, discounted_return


def run_equation(args) -> (float, float):
    """
    Method to run MCTS approaches for equation discovery
    :param args:
    :return:
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_log_obj(args=args, name=args.mcts_engine)
    grammar = read_grammar_file(args=args)
    game = FindEquationGame(
        grammar=grammar,
        args=args,
        train_test_or_val='train'
    )

    if args.mcts_engine == 'AmEx_MCTS':
        mcts_engine = AmEx_MCTS(game=game, args=args)
    elif args.mcts_engine == 'ClassicMCTS':
        mcts_engine = ClassicMCTS(game=game, args=args)
    else:
        raise AssertionError(f"args.engine: {args.engine} not defined")
    state = game.getInitialState()
    i = 0
    undiscounted_return = 0
    discounted_return = 0
    gamma = 1.0

    while not state.done:
        pi, v = mcts_engine.run_mcts(state=state,
                                     num_mcts_sims=args.num_MCTS_sims,
                                     temperature=1.)
        a = np.argmax(pi).item()
        next_state, r = game.getNextState(
            state=state,
            action=a,
        )
        state = next_state
        i += 1
        discounted_return += gamma * r
        undiscounted_return += r
        gamma *= args.gamma

        logger.debug(
            f"{i}, {a}, {pi}, {v},"
            f"{next_state.observation['current_tree_representation_str']}"
        )

        # Log results
        logger.debug(f"Return: {undiscounted_return}")
        logger.debug(f"Discounted Return: {discounted_return}")

    return undiscounted_return, discounted_return


if __name__ == '__main__':
    parser_args = parse_args()

    wandb.init(project="MCTSEndgame", config=parser_args.__dict__,
               mode=parser_args.wandb)

    if parser_args.env_str == "Equation":
        result, disc = run_equation(parser_args)
    else:
        result, disc = run_gym(parser_args)

    wandb.log({"Return": result})
    wandb.log({"Discounted Return": disc})
