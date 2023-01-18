#!/bin/bash

# Slurm sbatch options
#SBATCH -o run.sh.log-%j
#SBATCH --gres=gpu:volta:1

# # Loading the required module
source /etc/profile
module load anaconda/2020a 
module load mpi/openmpi-4.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gridsan/sidnayak/.mujoco/mujoco200/bin

# Run the script        
# script to iterate through different hyperparameters
envs="NutAssemblyDense"
seeds=0
exp_names="rl"
# Run the script
mkdir ${exp_names}_${envs}
# echo "${exp_names}_${envs}_${seeds}"
python -m RL.sac.sac --env_name ${envs} --seed ${seeds} --exp_name ${exp_names} 2>&1 | tee ${exp_names}_${envs}/out_${envs}_${seeds}