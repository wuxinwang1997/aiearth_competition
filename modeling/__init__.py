# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .miltiresnet import MultiResnet


def build_model(cfg):
    model = MultiResnet(cfg)
    return model
