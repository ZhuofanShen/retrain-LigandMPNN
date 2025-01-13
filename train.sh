#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 12
#SBATCH --mem=128g
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%N.%j.log
#SBATCH --error=slurm.%N.%j.err
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --begin=now
#SBATCH --open-mode=append

ulimit -n 65535
python ./train.py \
       --num_neighbors 32 \
       --atom_context_num 25 \
       --model_type "ligand_mpnn" \
       --cpus_per_task 12 \
       --path_for_outputs "./exp_020" \
       --path_for_training_data "training" \
       --num_examples_per_epoch 1000 \
       --save_model_every_n_epochs 50
