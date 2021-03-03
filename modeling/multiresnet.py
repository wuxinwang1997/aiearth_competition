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
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in [3]]) 
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in [3]])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in [3]])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in [3]])
        self.avgpool = nn.AdaptiveAvgPool2d((1,128))
        self.lstm = nn.LSTM(input_size=1540 * 4 ,hidden_size=64,num_layers=2,batch_first=True,bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.linear = nn.Linear(128, 24)

    def forward(self, x):
        sst, t300, ua, va = x
        for conv1 in self.conv1:
            sst = conv1(sst)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        
        sst = torch.flatten(sst, start_dim=2)
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)

        output = torch.cat([sst, t300, ua, va], dim=-1)
        output = self.batch_norm(output)
        output, _ = self.lstm(output)
        output = self.avgpool(output).squeeze(dim=-2)
        output = self.linear(output)

        return output
