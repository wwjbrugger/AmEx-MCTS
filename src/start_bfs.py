import wandb

import numpy as np

from src.best_first import BestFirstSearch
from src.utils.logging import get_log_obj

from src.environments.gym_game import GymGame, make_env
from src.config_mcts import parse_args
import random
import time
from utils.utils import tie_breaking_argmax


def run_gym(args) -> (float, float):
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_log_obj(args=args, name=args.mcts_engine)

    env = make_env(args.env_str, args.max_episode_steps)
    game = GymGame(args, env)
    wandb.config.update(game.env.spec.kwargs)

    mcts_engine = BestFirstSearch(game=game, args=args)

    state = game.getInitialState()

    i = 0
    undiscounted_return = 0
    discounted_return = 0
    gamma = 1.0
    start_time = time.time()
    next_state = None

    while not state.done:
        pi, v = mcts_engine.run_search(state=state,
                                     num_mcts_sims=args.num_MCTS_sims)

        a = tie_breaking_argmax(pi)
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


if __name__ == '__main__':
    parser_args = parse_args()
    parser_args.mcts_engine = "BestFirstSearch"

    wandb.init(project="MCTSEndgame", config=parser_args.__dict__,
               mode=parser_args.wandb, entity="mctsengame")

    result, disc = run_gym(parser_args)

    wandb.log({"Return": result})
    wandb.log({"Discounted Return": disc})
