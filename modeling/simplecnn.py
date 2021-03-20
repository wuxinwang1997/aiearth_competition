# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=(4, 8), stride=1, padding=(1, 3), padding_mode='zeros'),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=(2, 4), stride=1, padding=(1, 2), padding_mode='zeros'),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=(2, 4), stride=1, padding=(0, 1), padding_mode='zeros'),
            nn.Tanh(),
        )
        self.fc = nn.Linear(1275, 1)

    def forward(self, x):
        # Input 3 month sst
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
