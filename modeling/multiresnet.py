# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from .simplecnn import SimpleCNN

class MultiResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cnn = nn.ModuleList([SimpleCNN(cfg) for i in range(10)])

    def forward(self, x):
        sst, t300, ua, va = x
        outputs = []
        for i in range(10):
            outputs.append(self.cnn[i](torch.cat([sst, t300, ua, va], dim=1)))

        return outputs
