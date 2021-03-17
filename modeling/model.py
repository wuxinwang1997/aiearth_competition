# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
from .backbone import build_resnet_backbone, SimpleResNet


class AIEarthModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.backbones = nn.ModuleList([SimpleResNet(cfg) for i in range(4)])
        self.backbones = nn.ModuleList([SimpleResNet(cfg) for i in range(1)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 128))
        self.lstm = nn.LSTM(input_size=3 * 4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(512, affine=False)
        self.linear = nn.Linear(128, 24)

    def forward(self, x):
        # sst, t300, ua, va = x
        sst = x[0]

        sst = self.backbones[0](sst)
        # t300 = self.backbones[1](t300)
        # ua = self.backbones[2](ua)
        # va = self.backbones[3](va)

        sst = torch.flatten(sst, start_dim=2)
        # t300 = torch.flatten(t300, start_dim=2)
        # ua = torch.flatten(ua, start_dim=2)
        # va = torch.flatten(va, start_dim=2)

        # output = torch.cat([sst, t300, ua, va], dim=-1)
        output = torch.cat([sst, ], dim=-1)
        output = self.batch_norm(output)
        output, _ = self.lstm(output)
        output = self.avgpool(output).squeeze(dim=-2)
        output = self.linear(output)

        return output
