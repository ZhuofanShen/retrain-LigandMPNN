#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=preprocess_dataset
#SBATCH --cpus-per-task 1
#SBATCH --mem=3000
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%N.%j.log
#SBATCH --error=slurm.%N.%j.err
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --begin=now
#SBATCH --open-mode=append

python preprocess_training_data.py ${1}/train.json
python preprocess_training_data.py ${1}/valid.json

