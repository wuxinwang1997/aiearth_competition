#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7
source activate aiearth
python tools/train_net.py
