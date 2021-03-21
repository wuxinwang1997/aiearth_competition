# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from .multiresnet import MultiResnet


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.multicnn = nn.ModuleList([MultiResnet(cfg) for i in range(24)])

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.tensor(np.zeros((batch_size, 24, 10)))
        for i in range(24):
            output[:, i, :] = self.multicnn[i](x)

        return output
