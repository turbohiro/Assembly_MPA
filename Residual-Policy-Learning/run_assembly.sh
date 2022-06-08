#!/bin/bash

envs=("NutAssembly" "NutAssemblyHand" "NutAssemblyHandSticky")
seeds=(1 2)
exp_names=("rl" "res" "res")
epochs=(500 500 500)
# Run the script
for i in ${!seeds[@]}; do
    # run for different seeds
    for j in ${!envs[@]}; do
        # run for all different envs
        echo "Env Name: ${envs[$j]} | Seed: ${seeds[$i]} | Exp Name: ${exp_names[$j]} | out_${envs[$j]}_${seeds[i]}"
        mpirun python -m RL.ddpg.ddpg_mpi --env_name ${envs[$j]} --seed ${seeds[$i]} --n_epochs ${epochs[$j]}  --exp_name ${exp_names[$j]} 2>&1 | tee exp_outputs/out_${envs[$j]}_${seeds[i]}
    done
done