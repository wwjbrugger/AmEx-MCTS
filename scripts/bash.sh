export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 5 --mcts_engine EndgameMCTS
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py Chain-v0 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py Chain-v1 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py Chain-v2 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py Chain-v3 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py ChainLoop-v0 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py ChainLoop-v1 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py ChainLoop-v2 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py ChainLoop-v3 --logging-level 40 --wandb online --num-sims 250

python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 5
python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 10
python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 25
python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 50
python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 100
python src/start_mcts.py FrozenLakeNotSlippery-v1 --logging-level 40 --wandb online --num-sims 250