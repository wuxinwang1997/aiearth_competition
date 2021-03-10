# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .multiresnet import MultiResnet
from .model import Model

def build_model(cfg):
    model = Model(cfg)
    return model
