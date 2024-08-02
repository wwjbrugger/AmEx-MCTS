#!/bin/bash
#SBATCH --job-name=cbench-v1_sha_500_uniform_ScoreBoundedMCTS_rs_True 
#SBATCH -p smp 
#SBATCH --account=m2_datamining 
#SBATCH --time=1440 
#SBATCH --array=8-15%2 
#SBATCH --tasks=1 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=57GB 

#SBATCH -o \%x_\%j_profile.out 
#SBATCH -C anyarch 
#SBATCH -o output_compiler_gym/\%x_\%j.out 
#SBATCH -e output_compiler_gym/\%x_\%j.err 
#SBATCH --mail-user=%u@uni-mainz.de 
#SBATCH --mail-type=FAIL 

module purge 
module load lang/Python/3.11.2-GCCcore-11.4.0-bare
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd) 
export COMPILER_GYM_SITE_DATA=/home/$USER/AmEx-MCTS/compiler_gym_site_data 
export http_proxy=http://webproxy.zdv.uni-mainz.de:8888 
export https_proxy=https://webproxy.zdv.uni-mainz.de:8888 
source venv/bin/activate
wandb offline 


srun python src/start_mcts.py \
--env-str cbench-v1/sha \
--seed $SLURM_ARRAY_TASK_ID \
--gamma 0.99 \
--num-sims 500 \
--c1 1.25 \
--depth-first-search False \
--risk-seeking True \
--wandb offline \
--logging-level 30 \
--mcts-engine ScoreBoundedMCTS \
--prior-source uniform \
--use-puct False \
--max_episode_steps 20 \
--minimum-reward 0 \
--maximum-reward 2 \
