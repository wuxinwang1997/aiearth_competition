# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import netCDF4 as nc4
import numpy as np
import torch.utils.data as data
from .datasets.dataset import TrainDataset
from .transforms.build import build_transforms
from .collate_batch import collate_batch

def prepare_data(cfg):
        """Download data if needed."""
        data_train = []
        data_val = []
        data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + cfg.DATASETS.DATA_NAME + '_train.nc')
        label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + cfg.DATASETS.DATA_NAME + '_label.nc')
        sample_size = data.variables['sst'].shape[0]
        train_data = np.zeros((sample_size, cfg.DATASETS.Z_DIM, cfg.DATASETS.Y_DIM, cfg.DATASETS.X_DIM))
        train_data[:, 0:12, :, :] = data.variables['sst'][:, 0:12, :, :]
        train_data[:, 12:24, :, :] = data.variables['t300'][:, 0:12, :, :]
        train_data[:, 24:36, :, :] = data.variables['ua'][:, 0:12, :, :]
        train_data[:, 36:48, :, :] = data.variables['va'][:, 0:12, :, :]
        data = np.array(train_data)
        data[np.isnan(data)] = 0
        label = label.variables['nino']
        label = np.array(label)
        for i in range(sample_size):
            if i < int(sample_size * 0.8):
                data_train.append((data[i, :, :, :], label[i, 12:]))
            else:
                data_val.append((data[i, :, :, :], label[i, 12:]))
        return data_train, data_val

def build_dataset(cfg):
    data_train, data_val = prepare_data(cfg)
    train_dataset = TrainDataset(
        data_list = data_train,
        transforms=build_transforms(cfg, is_train=True),
    )

    val_dataset = TrainDataset(
        data_list=data_val,
        transforms=build_transforms(cfg, is_train=False),
    )

    return train_dataset, val_dataset

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH

    train_dataset, val_dataset = build_dataset(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False)

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader