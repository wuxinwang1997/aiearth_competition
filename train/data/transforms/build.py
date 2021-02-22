# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torchvision import transforms

def get_train_transforms(cfg):
    return transforms.ToTensor()

def get_valid_transforms(cfg):
    return transforms.ToTensor()

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = get_train_transforms(cfg)
    else:
        transform = get_valid_transforms(cfg)

    return transform
