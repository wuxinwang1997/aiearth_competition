# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models

class SimpleResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        if cfg.MODEL.PRETRAINED_IMAGENET is not '':
            self.model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED_IMAGENET))
        self.model.conv1 = nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 24)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        if cfg.MODEL.PRETRAINED_CMIP is not '':
            self.load_state_dict(torch.load(cfg.MODEL.PRETRAINED_CMIP)['model_state_dict'])

    def forward(self, x):

        return self.model(x)


