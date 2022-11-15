#!/bin/bash

#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -t 1:00:00
#SBATCH --job-name="TRAINDQN"

python main.py --train_dqn --use_cuda