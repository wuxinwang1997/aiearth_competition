# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simplecnn import SimpleCNN
from .simplepcb import PCB
from .model import AIEarthModel
def build_model(cfg):
    model = SimpleCNN(cfg)
    return model
