# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.model = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in [3]]) 
        #self.effnet = EfficientNet.from_pretrained(model_name='efficientnet-b0',
        #                                           weights_path=cfg.MODEL.PRETRAINED_IMAGENET,
        #                                           advprop=True,
        #                                           in_channels=12,
        #                                           image_size=[cfg.DATASETS.X_DIM,cfg.DATASETS.Y_DIM])
        resnet = models.resnet18(pretrained=False)
        if cfg.MODEL.PRETRAINED_IMAGENET is not '':
           resnet.load_state_dict(torch.load(cfg.MODEL.PRETRAINED_IMAGENET))
        resnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        #for conv in self.model:
        #    x = conv(x)  # batch * 12 * (24 - 2) * (72 -2)
        #x = self.effnet.extract_features(x)
        return self.model(x)


