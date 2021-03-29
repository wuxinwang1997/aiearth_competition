#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7
module load nvidia/cuda/10.1
source activate keras-aiearth
python tools/train_net.py DATASETS.SODA '(False)'
