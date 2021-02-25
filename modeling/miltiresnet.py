# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models
from .simpleresnet import SimpleResnet

class MultiResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.models = []
        for i in range(24):
            simpleresnet = SimpleResnet()
            self.models.append(simpleresnet)

    def forward(self, x):
        preds = []
        for i in range(24):
            preds.append(self.models[i](x))
        return preds