#!/bin/bash
#SBATCH --clusters=amarel
#SBATCH --partition=gpu
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=3000
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%N.%j.log
#SBATCH --error=slurm.%N.%j.err
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --begin=now
#SBATCH --open-mode=append

python ./train.py \
       --cpus_per_task 1 \
       --path_for_outputs "./exp_020" \
       --path_for_training_data "sample" \
       --num_examples_per_epoch 100 \
       --save_model_every_n_epochs 50
