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

def prepare_cmip_data(cfg):
        mean = []
        std = []
        cmip_data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'CMIP_train.nc')
        cmip_label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'CMIP_label.nc')
        sample_size = cmip_data.variables['sst'].shape[0]
        train_data = np.zeros((sample_size, cfg.DATASETS.Z_DIM, cfg.DATASETS.Y_DIM, cfg.DATASETS.X_DIM))
        sst = np.array(cmip_data.variables['sst'][:, 0:12, :, :])
        for i in range(12):
            sst = np.nan_to_num(sst)
            #sst[:, i, :, :][np.isnan(sst[:, i, :, :])] = np.nanmean(sst[:, i, :, :])
            mean.append(np.nanmean(sst[:, i, :, :]))
            std.append(np.nanstd(sst[:, i, :, :]))
        train_data[:, 0:12, :, :] = sst
        t300 = np.array(cmip_data.variables['t300'][:, 0:12, :, :])
        for i in range(12):
            t300 = np.nan_to_num(t300)
            # t300[:, i, :, :][np.isnan(t300[:, i, :, :])] = np.nanmean(t300[:, i, :, :])
            mean.append(np.nanmean(t300[:, i, :, :]))
            std.append(np.nanstd(t300[:, i, :, :]))
        train_data[:, 12:24, :, :] = t300
        ua = np.array(cmip_data.variables['ua'][:, 0:12, :, :])
        for i in range(12):
            ua = np.nan_to_num(ua)
            # ua[:, i, :, :][np.isnan(ua[:, i, :, :])] = np.nanmean(ua[:, i, :, :])
            mean.append(np.nanmean(ua[:, i, :, :]))
            std.append(np.nanstd(ua[:, i, :, :]))
        train_data[:, 24:36, :, :] = ua
        va = np.array(cmip_data.variables['va'][:, 0:12, :, :])
        for i in range(12):
            va = np.nan_to_num(va)
            #va[:, i, :, :][np.isnan(va[:, i, :, :])] = np.nanmean(va[:, i, :, :])
            mean.append(np.nanmean(va[:, i, :, :]))
            std.append(np.nanstd(va[:, i, :, :]))
        train_data[:, 36:48, :, :] = va
        cmip_data = train_data
        cmip_label = cmip_label.variables['nino']
        cmip_label = np.array(cmip_label)

        return cmip_data, cmip_label, mean, std


def prepare_soda_data(cfg):
    mean = []
    std = []
    soda_data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_train.nc')
    soda_label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_label.nc')
    sample_size = soda_data.variables['sst'].shape[0]
    train_data = np.zeros((sample_size, cfg.DATASETS.Z_DIM, cfg.DATASETS.Y_DIM, cfg.DATASETS.X_DIM))
    sst = np.array(soda_data.variables['sst'][:, 0:12, :, :])
    for i in range(12):
        mean.append(np.nanmean(sst[:, i, :, :]))
        std.append(np.nanstd(sst[:, i, :, :]))
    train_data[:, 0:12, :, :] = sst
    t300 = np.array(soda_data.variables['t300'][:, 0:12, :, :])
    for i in range(12):
        mean.append(np.nanmean(t300[:, i, :, :]))
        std.append(np.nanstd(t300[:, i, :, :]))
    train_data[:, 12:24, :, :] = t300
    ua = np.array(soda_data.variables['ua'][:, 0:12, :, :])
    for i in range(12):
        mean.append(np.nanmean(ua[:, i, :, :]))
        std.append(np.nanstd(ua[:, i, :, :]))
    train_data[:, 24:36, :, :] = ua
    va = np.array(soda_data.variables['va'][:, 0:12, :, :])
    for i in range(12):
        mean.append(np.nanmean(va[:, i, :, :]))
        std.append(np.nanstd(va[:, i, :, :]))
    train_data[:, 36:48, :, :] = va
    soda_data = train_data
    soda_label = soda_label.variables['nino']
    soda_label = np.array(soda_label)

    return soda_data, soda_label, mean, std

def prepare_data(cfg):
    data_train = []
    data_val = []
    cmip_data, cmip_label, cmip_mean, cmip_std = prepare_cmip_data(cfg)
    soda_data, soda_label, soda_mean, soda_std = prepare_soda_data(cfg)
    for cmip_6 in range(15):
        for year in range(151):
        #    if year not in [6, 7, 8, 9, 13]:
            if year < int(151 * 1):
                data_train.append((cmip_data[151*cmip_6 + year, :, :, :], cmip_label[151*cmip_6 + year, 12:]))
            else:
                data_val.append((cmip_data[151*cmip_6 + year, :, :, :], cmip_label[151*cmip_6 + year, 12:]))

    for cmip_5 in range(17):
        for year in range(140):
            if year < int(140 * 1):
                data_train.append((cmip_data[2264+140*cmip_5 + year, :, :, :], cmip_label[2264+140*cmip_5 + year, 12:]))
            else:
                data_val.append((cmip_data[2264+140*cmip_5 + year, :, :, :], cmip_label[2264+140*cmip_5 + year, 12:]))

    for year in range(100):
        if year < int(100 * 0):
            data_train.append((soda_data[year, :, :, :], soda_label[year, 12:]))
        else:
            data_val.append((soda_data[year, :, :, :], soda_label[year, 12:]))

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
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False)

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader
