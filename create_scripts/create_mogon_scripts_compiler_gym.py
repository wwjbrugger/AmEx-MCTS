import itertools
from pathlib import Path


def run():
    parameter_list_dict = {
        'job_array_size': ['8-15'],
        'script_folder': ['scripts_compiler_gym'],
        'output_folder': ['output_compiler_gym'],
        ## General
        'experiment_name': ['compiler_gym_sb'
        ],
        'minutes_to_run': [24],
        'logging_level': ['30'],
        'wandb': ['offline'],
        'env_str': [
            'cbench-v1/adpcm',
            'cbench-v1/bitcount',
            'cbench-v1/blowfish',
            'cbench-v1/bzip2',
            'cbench-v1/crc32',
            'cbench-v1/dijkstra',
            'cbench-v1/ghostscript',
            'cbench-v1/gsm',
            'cbench-v1/ispell',
            'cbench-v1/jpeg-c',
            'cbench-v1/jpeg-d',
            'cbench-v1/lame',
            'cbench-v1/patricia',
            'cbench-v1/qsort',
            'cbench-v1/rijndael',
            'cbench-v1/sha',
            'cbench-v1/stringsearch',
            'cbench-v1/stringsearch2',
            'cbench-v1/susan',
            'cbench-v1/tiff2bw',
            'cbench-v1/tiff2rgba',
            'cbench-v1/tiffdither',
            'cbench-v1/tiffmedian',
        ],
        'num_MCTS_sims': [500],
        'gamma': [0.99],
        'c1': [1.25],
        'depth_first_search': [False],
        'risk_seeking': [False, True],
        'mcts_engine': ['ScoreBoundedMCTS'],  # 'ClassicMCTS', later only with risk_seeking true
        'prior_source': ['uniform'],
        'use_puct': [False],
        'max_episode_steps': [20],
        'minimum-reward': [0],
        'maximum-reward': [2]
    }
    create_output_folder(parameter_list_dict)

    cartesian_product = itertools.product(
        *parameter_list_dict.values(),
        repeat=1
    )
    created_experiments = []
    for values in cartesian_product:
        settings_one_script = dict(zip(parameter_list_dict.keys(), values))
        settings_one_script['experiment_name'] = create_experiment_name(
            settings_one_script)
        write_script(settings_one_script)
        created_experiments.append(settings_one_script['experiment_name'])

    write_experiment_names_to_file(
        experiment_list=created_experiments,
        script_folder=parameter_list_dict['script_folder'][0]
    )
    write_sbatch_comments_to_file(
        experiment_list=created_experiments,
        script_folder=parameter_list_dict['script_folder'][0]
    )


def create_experiment_name(settings_one_script):
    experiment_name = f"{settings_one_script['env_str']}" \
                      f"_{settings_one_script['num_MCTS_sims']}" \
                      f"_{settings_one_script['prior_source']}" \
                      f"_{settings_one_script['mcts_engine']}" \
                      f"_rs_{settings_one_script['risk_seeking']}"
    return experiment_name


def create_output_folder(parameter_list_dict):
    path = Path(
        f"../{parameter_list_dict['script_folder'][0]}/"
        f"{parameter_list_dict['output_folder'][0]}")
    path.mkdir(exist_ok=True, parents=True)
    with open(path / 'delete_me.txt', "w") as file:
        file.writelines(
            "This file exists for Mogon to create the output folder. ")


def write_script(settings_one_script):
    with open(f"../{settings_one_script['script_folder']}/" +
                  (f"{settings_one_script['experiment_name']}.sh".replace('/','_')), "w") as file1:
        write_SBATCH_commants(settings_one_script, file1)
        write_prepare_enviroment(file1)
        write_python_call(settings_one_script, file1)


def write_SBATCH_commants(settings_one_script, file1):
    file1.write(f"#!/bin/bash\n")
    file1.write(
        f"#SBATCH --job-name={settings_one_script['experiment_name'].replace('/','_')} \n")

    file1.writelines("#SBATCH -p smp \n")
    file1.writelines("#SBATCH --account=m2_datamining \n")
    file1.writelines(
        f"#SBATCH --time={int(settings_one_script['minutes_to_run'] * 60)} \n")
    file1.writelines(f"#SBATCH --array={settings_one_script['job_array_size']}%2 \n")
    file1.writelines("#SBATCH --tasks=1 \n")
    file1.writelines("#SBATCH --nodes=1 \n")
    file1.writelines("#SBATCH --cpus-per-task=4 \n")
    file1.writelines("#SBATCH --mem=57GB \n")
    file1.writelines("\n")
    file1.writelines("#SBATCH -o \%x_\%j_profile.out \n")
    file1.writelines("#SBATCH -C anyarch \n")
    file1.writelines(f"#SBATCH -o {settings_one_script['output_folder']}"
                     f"/\%x_\%j.out \n")
    file1.writelines(f"#SBATCH -e {settings_one_script['output_folder']}"
                     f"/\%x_\%j.err \n")
    file1.writelines("#SBATCH --mail-user=%u@uni-mainz.de \n")
    file1.writelines("#SBATCH --mail-type=FAIL \n")
    file1.writelines("\n")


def write_prepare_enviroment(file1):
    file1.writelines("module purge \n")
    file1.writelines("module load "
                     "lang/Python/3.11.2-GCCcore-11.4.0-bare\n"
                     )
    file1.writelines("cd ..\n")
    file1.writelines("export PYTHONPATH=$PYTHONPATH:$(pwd) \n")
    file1.writelines("export COMPILER_GYM_SITE_DATA=/home/$USER/AmEx-MCTS/compiler_gym_site_data \n")
    file1.writelines(
        "export http_proxy=http://webproxy.zdv.uni-mainz.de:8888 \n")
    file1.writelines(
        "export https_proxy=https://webproxy.zdv.uni-mainz.de:8888 \n")
    file1.writelines("source venv/bin/activate\n")
    file1.writelines("wandb offline \n")
    file1.writelines("\n")
    file1.writelines("\n")


def write_python_call(settings_one_script, file1):
    file1.writelines(f"srun python "
                     f"src/start_mcts.py \\\n")
    ## General
    file1.writelines(f"--env-str {settings_one_script['env_str']} \\\n")
    file1.writelines(f"--seed $SLURM_ARRAY_TASK_ID \\\n")
    file1.writelines(f"--gamma {settings_one_script['gamma']} \\\n")
    file1.writelines(f"--num-sims {settings_one_script['num_MCTS_sims']} \\\n")
    file1.writelines(f"--c1 {settings_one_script['c1']} \\\n")
    file1.writelines(
        f"--depth-first-search {settings_one_script['depth_first_search']}"
        f" \\\n")
    file1.writelines(
        f"--risk-seeking {settings_one_script['risk_seeking']} \\\n")
    file1.writelines(f"--wandb {settings_one_script['wandb']} \\\n")
    file1.writelines(
        f"--logging-level {settings_one_script['logging_level']} \\\n")
    file1.writelines(
        f"--mcts-engine {settings_one_script['mcts_engine']} \\\n")
    file1.writelines(
        f"--prior-source {settings_one_script['prior_source']} \\\n")
    file1.writelines(
        f"--use-puct {settings_one_script['use_puct']} \\\n")
    file1.writelines(
        f"--max_episode_steps {settings_one_script['max_episode_steps']} \\\n")
    file1.writelines(
        f"--minimum-reward {settings_one_script['minimum-reward']} \\\n")
    file1.writelines(
        f"--maximum-reward {settings_one_script['maximum-reward']} \\\n")


def write_experiment_names_to_file(experiment_list, script_folder):
    with open(f"../{script_folder}/experiment_name.txt", "a") as file2:
        for experiment_name in experiment_list:
            file2.write(f"\"{experiment_name.replace('/','_')}\",\n")


def write_sbatch_comments_to_file(experiment_list, script_folder):
    with open(f"../{script_folder}/sbatch_comments.txt", "a") as file2:
        for experiment_name in experiment_list:
            file2.write(f"sbatch {experiment_name.replace('/','_')}.sh \n")


if __name__ == '__main__':
    run()
