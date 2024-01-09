import wandb
import copy


import numpy as np

from src.utils.logging import get_log_obj

from src.environments.gym_game import GymGame
from src.mcts_end_game import MCTSEndGame
from src.classic_mcts import ClassicMCTS
from src.config_mcts import parse_args
from src.utils.get_grammar import read_grammar_file
from src.environments.find_equation_game import FindEquationGame
import random

def run_gym(args) -> (float, float):
    logger = get_log_obj(args=args, name="MCTSEndGame")
    game = GymGame(args)
    wandb.config.update(game.env.spec.kwargs)

    if args.mcts_engine == 'EndgameMCTS':
        mcts_engine = MCTSEndGame(game=game, args=args)
    elif args.mcts_engine == 'ClassicMCTS':
        mcts_engine = ClassicMCTS(game=game, args=args)
    else:
        raise AssertionError(f"args.engine: {args.engine} not defined")
    state = game.getInitialState()
    env = copy.deepcopy(state.env)
    env.unwrapped.render_mode = args.render_mode

    i = 0
    undiscounted_return = 0
    discounted_return = 0
    gamma = 1.0

    while not state.done:
        pi, v = mcts_engine.runMCTS(
            state=state,
            num_MCTS_sims=args.num_MCTS_sims,
            temperature=1.
        )

        a = np.argmax(pi).item()
        logger.debug(f"{i}, {a}, {pi}, {v}, {state.observation['obs']}")
        next_state, r = game.getNextState(
            state=state,
            action=a,
        )
        env.step(a)
        state = next_state
        i += 1
        discounted_return += gamma * r
        undiscounted_return += r
        gamma *= args.gamma

    # Cleanup environment and GameHistory
    game.close(state)
    env.close()

    return undiscounted_return, discounted_return


def run_equation(args) -> (float, float):
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_log_obj(args=args, name="MCTSEndGame")
    grammar = read_grammar_file(args=args)
    game = FindEquationGame(
        grammar=grammar,
        args=args,
        train_test_or_val='train'
    )

    if args.mcts_engine == 'EndgameMCTS':
        mcts_engine = MCTSEndGame(game=game, args=args)
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
        pi, v = mcts_engine.runMCTS(
            state=state,
            num_MCTS_sims=args.num_MCTS_sims,
            temperature=1.
        )
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

        logger.debug(f"{i}, {a}, {pi}, {v}, {next_state.observation['current_tree_representation_str']}")

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
