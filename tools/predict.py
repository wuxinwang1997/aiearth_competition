# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
sys.path.append('.')
from config import cfg
from data import make_test_data_loader
from engine.predicter import Predicter
from modeling import build_model
import random
import torch
import numpy as np
from utils.logger import setup_logger

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def predict(cfg):
    seed_everything(cfg.SEED)
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT, map_location=cfg.MODEL.DEVICE)['model_state_dict'])
    device = cfg.MODEL.DEVICE

    test_loader = make_test_data_loader(cfg)

    predicter = Predicter(model=model, device=device, cfg=cfg, test_loader=test_loader)
    predicter.predict()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    result_dir = cfg.RESULT_DIR
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)
    predict(cfg)


if __name__ == '__main__':
    main()
