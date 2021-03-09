# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models
from .simplecnn import SimpleCNN


class MultiResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cnn = nn.ModuleList([SimpleCNN(cfg) for i in range(4)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 128))
        # original
        # self.lstm = nn.LSTM(input_size=3 * 4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        # upsample 2
        self.lstm = nn.LSTM(input_size=10 * 4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(512, affine=False)
        self.linear = nn.Linear(128, 24)

    def forward(self, x):
        # torch.Size([64, 12, 24, 72])
        sst, t300, ua, va = x

        sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # torch.Size([64, 12, 48, 144])
        sst = sample(sst)
        t300 = sample(t300)
        ua = sample(ua)
        va = sample(va)

        # torch.Size([64, 512, 1, 3])
        # upsample 2: torch.Size([64, 512, 2, 5])
        # torch.Size([64, 512, 3, 9])
        sst = self.cnn[0](sst)
        t300 = self.cnn[1](t300)
        ua = self.cnn[2](ua)
        va = self.cnn[3](va)

        # torch.Size([64, 512, 3])
        # upsample 2: torch.Size([64, 512, 10])
        sst = torch.flatten(sst, start_dim=2)
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)

        # torch.Size([64, 512, 12])
        # upsample 2: torch.Size([64, 512, 40])
        output = torch.cat([sst, t300, ua, va], dim=-1)
        output = self.batch_norm(output)
        # torch.Size([64, 512, 128])
        output, _ = self.lstm(output)
        # torch.Size([64, 1, 128]) -> torch.Size([64, 128])
        output = self.avgpool(output).squeeze(dim=-2)
        # torch.Size([64, 24])
        output = self.linear(output)

        return output
