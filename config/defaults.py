# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import sys
import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = False
_C.SEED = 66
_C.VERBOSE = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cpu"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.PRETRAINED_IMAGENET = '' # '/home/wangxiang/dat01/WWX/aiearth/pretrained/resnet50.pth'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root dir of dataset
_C.DATASETS.ROOT_DIR = "/home/wangxiang/dat01/WWX/aiearth/data/enso_round1_train_20210201/"
# Fold to validate

_C.DATASETS.X_DIM = 72
_C.DATASETS.Y_DIM = 24
_C.DATASETS.Z_DIM = 48
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.SCHEDULER_NAME = "CosineAnnealingWarmRestarts"
_C.SOLVER.COS_CPOCH = 5
_C.SOLVER.T_MUL = 2

_C.SOLVER.MAX_EPOCHS = 80

_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY_BN = 0

_C.SOLVER.WARMUP_EPOCHS = 5

_C.SOLVER.EARLY_STOP_PATIENCE = 40

_C.SOLVER.TRAIN_CHECKPOINT = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.WEIGHT = os.path.abspath(os.path.join(os.getcwd(), "../usr_data/model_data/"))+"/best-model.bin"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../usr_data/model_data/"))
_C.RESULT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../result"))
