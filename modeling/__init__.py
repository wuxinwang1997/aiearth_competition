# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .model import AIEarthModel


def build_model(cfg):
    model = AIEarthModel(cfg)
    return model
