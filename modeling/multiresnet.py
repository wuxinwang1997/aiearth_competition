# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from statistics import mean

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from .simplecnn import SimpleCNN


class MultiResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.cnn = nn.ModuleList([SimpleCNN(cfg) for i in range(10)])

    def forward(self, x):
        # Input 12 month sst
        batch_size = self.config.SOLVER.IMS_PER_BATCH
        all_pred = torch.tensor(np.zeros((10, batch_size, 1)))
        for i in range(10):
            one_pred = self.cnn[i](x[:, i:i + 3, :, :])
            all_pred[i, :, :] = one_pred
        all_pred = torch.transpose(all_pred, 0, 1)
        all_pred = torch.flatten(all_pred, start_dim=1, end_dim=2)
        # mean_pred = torch.mean(all_pred, 1)

        return all_pred
