#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7
module load nvidia/cuda/10.1
source activate aiearth
python tools/train_net.py OUTPUT_DIR '("./usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py OUTPUT_DIR '("./usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda/")' DATASETS.SODA '(True)' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)'
