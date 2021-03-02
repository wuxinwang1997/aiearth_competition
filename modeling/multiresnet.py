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
        self.backbones = nn.ModuleList(SimpleResnet(cfg) for i in range(4))
        self.avgpool1 = nn.AdaptiveAvgPool2d((22,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,128))
        self.lstm = nn.LSTM(input_size=1540 * 4 ,hidden_size=64,num_layers=2,batch_first=True,bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.linear = nn.Linear(128, 24)
        self.device = cfg.MODEL.DEVICE

    def forward(self, x):
        # res = torch.zeros((x.shape[0], 24), dtype=torch.float).to(self.device)
        outputs = []
        for i, backbone in enumerate(self.backbones):
            input = x[:, i*12:(i+1)*12, :, :] 
            outputs.append(torch.flatten(backbone(input), start_dim=2))
        output = torch.cat(outputs, dim=-1)
        output = self.batch_norm(output)
        output = self.lstm(output)[0]
        output = self.avgpool2(output).squeeze(dim=-2)
        output = self.linear(output)
        # res[:, i] = y.squeeze(1)

        return output
