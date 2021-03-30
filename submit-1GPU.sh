#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7
module load nvidia/cuda/10.1
source activate aiearth-fusion
python tools/train_net.py GENERATE '(True)' DATASETS.SODA '(False)'
python tools/train_net.py GENERATE '(True)' DATASETS.SODA '(True)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold00/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold01/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold02/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold03/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold04/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'

python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold10/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold11/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold12/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold13/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold14/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold1/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'

python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold20/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold21/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold22/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold23/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold24/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold2/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'

python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold30/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold31/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold32/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold33/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold34/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold3/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'

python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold40/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold41/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold42/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold43/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python tools/train_net.py GENERATE '(False)' DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-soda-fold44/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("./usr_data/model_data/seresnet18-lr1e4-sst-epoch30-cmip-fold4/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
