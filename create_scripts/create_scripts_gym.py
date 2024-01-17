import itertools
from pathlib import Path


def run():
    parameter_list_dict = {
        'seeds': range(25),
        'script_folder': ['scripts_gym'],
        'output_folder': ['output_gym'],
        ## General
        'experiment_name': [''],
        'logging_level': ['30'],
        'wandb': ['online'],
        'env_str': ['Chain-v0', 'Chain-v1', 'Chain-v2', 'Chain-v3', #]
                    'ChainLoop-v0', 'ChainLoop-v1', 'ChainLoop-v2', 'ChainLoop-v3',
                    'FrozenLakeNotSlippery-v1'
                    ],
        'num_MCTS_sims': [5, 10, 25, 50, 100, 250],
        'gamma': [0.99],
        'c1': [1.414213562373095],
        'depth_first_search':[False],
        'risk_seeking': [True,False],
        'mcts_engine': [ 'AmEx_MCTS', 'ClassicMCTS'],
        'prior_source': ['uniform'],
        'use_puct': [False],
    }

    cartesian_product = itertools.product(
        *parameter_list_dict.values(),
        repeat=1
    )
    with open(f"experiments.sh", "a") as file1:
        file1.writelines("export PYTHONPATH=$PYTHONPATH:$(pwd) \n")
        for values in cartesian_product:
            settings_one_script = dict(zip(parameter_list_dict.keys(), values))
            settings_one_script['experiment_name'] = create_experiment_name(
                settings_one_script)
            write_script(settings_one_script= settings_one_script, file1=file1)



def create_experiment_name(settings_one_script):
    experiment_name = f"{settings_one_script['env_str']}" \
                      f"_{settings_one_script['num_MCTS_sims']}" \
                      f"_{settings_one_script['prior_source']}" \
                      f"_{settings_one_script['mcts_engine']}" \
                      f"_rs_{settings_one_script['risk_seeking']}"
    return experiment_name




def write_script(file1, settings_one_script):
        file1.writelines(f"python src/start_mcts.py  ")
        ## General
        file1.writelines(f"--env-str {settings_one_script['env_str']}  ")
        file1.writelines(f"--seed {settings_one_script['seeds']}  ")
        file1.writelines(f"--gamma {settings_one_script['gamma']}  ")
        file1.writelines(f"--num-sims {settings_one_script['num_MCTS_sims']}  ")
        file1.writelines(f"--c1 {settings_one_script['c1']}  ")
        file1.writelines(
            f"--depth-first-search {settings_one_script['depth_first_search']}  ")
        file1.writelines(
            f"--risk-seeking {settings_one_script['risk_seeking']}  ")
        file1.writelines(f"--wandb {settings_one_script['wandb']}  ")
        file1.writelines(
            f"--logging-level {settings_one_script['logging_level']}  ")
        file1.writelines(
            f"--mcts-engine {settings_one_script['mcts_engine']}  ")
        file1.writelines(
            f"--prior-source {settings_one_script['prior_source']}  ")
        file1.writelines(
            f"--use-puct {settings_one_script['use_puct']} \n ")


if __name__ == '__main__':
    run()
