# AmEx-MCTS: *Amplifying Exploration in Monte Carlo Tree Search by Focusing on the Unknown*

MCTS implementation for deterministic discrete action space MDPs, ensuring each terminal state is visited only once.

## Installation
All packages are specified for Python version 3.11

```bash
pip install -r requirements.txt
``` 
pygraphviz is used to visualize the search trees. 
If you want to use this function, please follow the installation instructions for pygraphviz (https://pygraphviz.github.io/documentation/stable/install.html)

## Sample Usage

The entry point to AmEx-MCTS and Classic-MCTS is the [src/start_mcts.py](src/start_mcts.py) file.
The possible parameters are described in src/config_mcts.py.  
You can simply try AmEx-MCTS by running the tests in [test/basic_test.py](test/basic_test.py)
The visualization of the search trees can be tested in [test/test_visualize_mcts.py](test/test_visualize_mcts.py)

## Experiments

In [create_scripts_gym](create_scripts/create_scripts_gym.py), you find the bash commands to replicate the experiments reported in our paper including all hyperparameters.

The experiments are the cartesian product of the following parampeter: 
- Seed [0 - 24]
- Environments ['Chain-v0', 'Chain-v1', 'Chain-v2', 'Chain-v3', 'ChainLoop-v0', 'ChainLoop-v1', 'ChainLoop-v2', 'ChainLoop-v3', 'FrozenLakeNotSlippery-v1']
- MCTS simulation steps [5, 10, 25, 50, 100, 250]
- MCTS Algorithm ['AmEx_MCTS', 'ClassicMCTS']
