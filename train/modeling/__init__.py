# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simpleresnet import SimpleResnet


def build_model(cfg):
    model = SimpleResnet(cfg)
    return model
