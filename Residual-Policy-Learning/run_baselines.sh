#!/bin/bash

# Slurm sbatch options
# add -g volta:1 -s 20 for 1 GPU
#SBATCH -o run.sh.log-%j

# Loading the required module
source /etc/profile
module load anaconda/2020a 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gridsan/sidnayak/.mujoco/mujoco200/bin

# Run the script        
# script to iterate through different envs
# Get the baselines for just pure control
seeds=(0 1 2 3 4)
envs=("FetchPushImperfect" "FetchPushSlippery" "FetchSlideFrictionControl" "FetchSlideSlapControl" "FetchPickAndPlacePerfect" "FetchPickAndPlaceSticky")
epochs=(200 200 500 500 500 500)
mkdir baseline_outputs
# Run the script
for i in ${!seeds[@]}; do
    for j in ${!envs[@]}; do
        echo "out_${envs[$j]}_${seeds[$i]} ${epochs[$j]}"
        python -m RL.baseline --exp_name 'baseline' --env_name ${envs[$j]} --n_epochs ${epochs[$j]} --seed ${seeds[$i]} 2>&1 | tee baseline_outputs/out_${envs[$j]}_${seeds[$i]}
    done
done
