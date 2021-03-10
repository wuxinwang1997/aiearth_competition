# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from .multiresnet import MultiResnet

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.multicnn = nn.ModuleList([MultiResnet(cfg) for i in range(24)])

    def forward(self, x):
        sst, t300, ua, va = x
        outputs = []
        for i in range(24):
            outputs.append(self.multicnn[i](sst[:, i:i+3], t300[:, i:i+3], ua[:,i:i+3], va[:,i:i+3]))

        return outputs
