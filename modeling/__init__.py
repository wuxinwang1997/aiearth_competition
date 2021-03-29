# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simplecnn import SimpleCNN
from .simplepcb import PCB
from .model import AIEarthModel
from .threedcnn import generate_model
def build_model(cfg):
    model = generate_model(cfg.MODEL.BACKBONE.DEPTH)
    return model
