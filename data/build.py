# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import socket

import netCDF4 as nc4
from scipy import interpolate
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from .datasets.dataset import EarthDataset, TestDataset
from .transforms.build import build_transforms
import matplotlib.pyplot as plt
from .collate_batch import collate_batch


def fill_nan(cmip_data):
    sample_size = cmip_data['sst'].shape[0]
    train_data = np.zeros((4, sample_size, 12, 24, 72))
    cmip6 = np.zeros((4, 15, 151, 12, 24, 72))
    cmip5 = np.zeros((4, 17, 140, 12, 24, 72))

    print('decompose')
    for i, var in enumerate(['sst', 't300', 'ua', 'va']):
        for j in range(15):
            for k in range(151):
                cmip6[i, j, k, :, :] = cmip_data[var][j * 151 + k, 0:12, :, :]
        for j in range(17):
            for k in range(140):
                cmip5[i, j, k, :, :] = cmip_data[var][j * 140 + k + 15 * 151, 0:12, :, :]

    print('fill nan')
    for data in [cmip6, cmip5]:
        for i in range(4):
            nan_idx = np.argwhere(np.isnan(data[i]))
            if not nan_idx.shape[0]:
                continue
            nan_df = pd.DataFrame(nan_idx)
            yx_unique_nan = nan_df.groupby([3, 4]).size().reset_index(name='Freq')
            for idx, row in yx_unique_nan.iterrows():
                y = row[3]
                x = row[4]
                for year in range(151):
                    for month in range(12):
                        pt = data[i, :, year, month, y, x]
                        pt[np.isnan(pt)] = np.nanmean(pt)
    print('reshape')
    for i in range(4):
        year = 0
        while year < 15 * 151:
            train_data[i, year, :, :, :] = cmip6[i, int(year / 151), year % 151, :, :, :]
            year += 1
        while year < 15 * 151 + 17 * 140:
            train_data[i, year, :, :, :] = cmip5[i, int((year - 151 * 15) / 140), year % 140, :, :, :]
            year += 1

    return train_data


def upsample(a, ratio):
    height = a.shape[0]
    width = a.shape[1]
    y = np.array(range(height))
    x = np.array(range(width))
    f = interpolate.interp2d(x, y, a, kind='linear')

    xnew = np.linspace(0, width, width * ratio)
    ynew = np.linspace(0, height, height * ratio)
    znew = f(xnew, ynew)

    return znew


def prepare_cmip_data(cfg):
    if socket.gethostname() == 'lujingzedeMacBook-Pro.local':
        root_dir = '/Users/lujingze/Programming/ai-earth/data/enso_round1_train_20210201/'
    else:
        root_dir = cfg.DATASETS.ROOT_DIR
    ratio = cfg.DATASETS.UP_RATIO
    cmip_data = nc4.Dataset(root_dir + 'CMIP_train.nc').variables
    # shape: (4, years, 12, 24, 72)
    cmip_data = fill_nan(cmip_data)
    cmip_label = nc4.Dataset(root_dir + 'CMIP_label.nc').variables

    dict_cmip = dict()
    for i, var in enumerate(['sst', 't300', 'ua', 'va']):
        tmp = np.array(cmip_data[i][:, 0:12, :, :])
        tmp = np.nan_to_num(tmp)
        tmp = torch.tensor(tmp)
        tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
        # months = tmp.shape[0]
        # up_tmp = np.zeros((months, 24 * ratio, 72 * ratio))
        # for j in range(months):
        #     up_tmp[j] = upsample(tmp[j], ratio)
        # dict_cmip[var] = up_tmp
        dict_cmip[var] = tmp

    tmp = np.array(cmip_label['nino'][:, 12:24])
    last_year_nino = np.array(cmip_label['nino'][-1, -12:].reshape((1, 12)))
    tmp = np.concatenate((tmp, last_year_nino), axis=0)
    dict_cmip['label'] = tmp.flatten()

    return dict_cmip


def prepare_soda_data(cfg):
    if socket.gethostname() == 'lujingzedeMacBook-Pro.local':
        root_dir = '/Users/lujingze/Programming/ai-earth/data/enso_round1_train_20210201/'
    else:
        root_dir = cfg.DATASETS.ROOT_DIR
    soda_data = nc4.Dataset(root_dir + 'SODA_train.nc').variables
    soda_label = nc4.Dataset(root_dir + 'SODA_label.nc').variables

    soda = dict()
    for var in ['sst', 't300', 'ua', 'va']:
        tmp = np.array(soda_data[var][:, 0:12, :, :])
        tmp = np.nan_to_num(tmp)
        tmp = torch.tensor(tmp)
        tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
        soda[var] = tmp.numpy()

    tmp = np.array(soda_label['nino'][:, 12:24])
    last_year_nino = np.array(soda_label['nino'][-1, -12:].reshape((1, 12)))
    tmp = np.concatenate((tmp, last_year_nino), axis=0)
    soda['label'] = tmp.flatten()

    # soda['label'] = np.array(soda_label['nino'][:, 12:36])
    # soda['label'] = np.array(soda['label']).flatten()

    dict_soda = dict()
    for var in ['sst', 't300', 'ua', 'va', 'label']:
        dict_soda[var] = soda[var]

    return dict_soda


def prepare_test_data(cfg):
    test_path = cfg.DATASETS.TEST_DIR
    files = [x for x in os.listdir(test_path) if x.endswith('.npy')]
    test_sst = np.zeros((len(files), 12, 24, 72))
    test_t300 = np.zeros((len(files), 12, 24, 72))
    test_ua = np.zeros((len(files), 12, 24, 72))
    test_va = np.zeros((len(files), 12, 24, 72))
    for i in range(len(files)):
        file = np.load(test_path + files[i])
        sst, t300, ua, va = np.split(file, 4, axis=3)
        test_sst[i, :, :, :] = sst.transpose(3, 0, 1, 2)
        test_t300[i, :, :, :] = t300.transpose(3, 0, 1, 2)
        test_ua[i, :, :, :] = ua.transpose(3, 0, 1, 2)
        test_va[i, :, :, :] = va.transpose(3, 0, 1, 2)

    dict_test = {
        'sst': test_sst,
        't300': test_t300,
        'ua': test_ua,
        'va': test_va,
        'name': np.array(files)
    }

    return dict_test


def build_dataset(cfg):
    dict_cmip, dict_soda = prepare_cmip_data(cfg), prepare_soda_data(cfg)
    train_dataset = EarthDataset(
        data_dict=dict_cmip,
        transforms=build_transforms(cfg, is_train=True),
    )

    val_dataset = EarthDataset(
        data_dict=dict_soda,
        transforms=build_transforms(cfg, is_train=False),
    )

    return train_dataset, val_dataset


def build_test_dataset(cfg):
    dict_test = prepare_test_data(cfg)
    test_dataset = TestDataset(
        data_dict=dict_test,
        transforms=build_transforms(cfg, is_train=False),
    )

    return test_dataset


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


def make_test_data_loader(cfg):
    batch_size = cfg.TEST.IMS_PER_BATCH

    test_dataset = build_test_dataset(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False)

    return test_loader
