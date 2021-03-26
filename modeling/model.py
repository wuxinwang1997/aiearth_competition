# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import copy
from .resnet import build_resnet_backbone
class AIEarthModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cnn = build_resnet_backbone(cfg)
        cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(cnn.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.cnn = nn.ModuleList([copy.deepcopy(cnn) for i in range(12)])
        self.avgpools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for i in range(12)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 128))
        self.lstm = nn.LSTM(input_size=512, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.fc = nn.Linear(128, 24)
    def forward(self, x):
        outputs = []
        for i in range(12):
            outputs.append(torch.unsqueeze(self.avgpools[i](self.cnn[i](torch.unsqueeze(x[:,i,:,:], dim=1))),dim=1))
        outputs = torch.squeeze(torch.cat(outputs, dim=1))
        outputs = self.batch_norm(outputs)
        outputs, _ = self.lstm(outputs)
        outputs = self.avgpool(outputs).squeeze(dim=-2)
        outputs = self.fc(outputs)
        return outputs
