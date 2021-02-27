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
        self.model = nn.ModuleList(SimpleResnet(cfg) for i in range(24))

    def forward(self, x):
        res = torch.zeros((x.shape[0], 24), dtype=torch.float).to('cuda')

        for i, model in enumerate(self.model):
            y = model(x)
            res[:, i] = y.squeeze(1)

        return res
