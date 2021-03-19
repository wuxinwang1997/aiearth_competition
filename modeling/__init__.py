# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simplest import SimplestCNN


def build_model(cfg):
    model = SimplestCNN(cfg)
    return model
