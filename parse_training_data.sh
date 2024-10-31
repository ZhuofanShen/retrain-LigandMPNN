#!/bin/bash
#SBATCH --clusters=amarel
#SBATCH --partition=gpu
#SBATCH --job-name=binarize
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

python parse_training_data.py sample/train.json
python parse_training_data.py sample/valid.json

