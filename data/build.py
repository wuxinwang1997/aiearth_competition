# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import netCDF4 as nc4
import numpy as np
import torch.utils.data as data
from .datasets.dataset import EarthDataset
from .transforms.build import build_transforms
from .collate_batch import collate_batch

def prepare_cmip_data(cfg):
    cmip_data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'CMIP_train.nc')
    cmip_label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'CMIP_label.nc')
    cmip_sst = cmip_data['sst'][:, 0:12].values
    cmip_sst = np.nan_to_num(cmip_sst)
    cmip_t300 = cmip_data['t300'][:, 0:12].values
    cmip_t300 = np.nan_to_num(cmip_t300)
    cmip_ua = cmip_data['ua'][:, 0:12, :, :].values
    cmip_ua = np.nan_to_num(cmip_ua)
    cmip_va = cmip_data['va'][:, 0:12, :, :].values
    cmip_va = np.nan_to_num(cmip_va)
    cmip_label = cmip_label['nino'][:,12:36].values
    cmip_label = np.array(cmip_label)
    dict_cmip = {
        'sst':cmip_sst,
        't300':cmip_t300,
        'ua':cmip_ua,
        'va': cmip_va,
        'label': cmip_label
        }
    return dict_cmip


def prepare_soda_data(cfg):
    soda_data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_train.nc')
    soda_label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_label.nc')
    soda_sst = soda_data['sst'][:, 0:12].values
    soda_t300 = soda_data['t300'][:, 0:12].values
    soda_ua = soda_data['ua'][:, 0:12].values
    soda_va = soda_data['va'][:, 0:12].values
    soda_label = soda_label['nino'][:, 12:36].values
    dict_soda = {
        'sst':soda_sst,
        't300':soda_t300,
        'ua':soda_ua,
        'va':soda_va,
        'label':soda_label
        }
    return dict_soda

def build_dataset(cfg):
    dict_cmip, dict_soda = prepare_cmip_data(cfg), prepare_soda_data(cfg)
    train_dataset = EarthDataset(
        data_dict = dict_cmip,
        transforms=build_transforms(cfg, is_train=True),
    )

    val_dataset = EarthDataset(
        data_dict = dict_soda,
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
