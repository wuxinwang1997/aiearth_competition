# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import socket

import netCDF4 as nc4
import numpy as np
import torch
import torch.utils.data as data
from .datasets.dataset import EarthDataset, TestDataset
from .transforms.build import build_transforms
from .collate_batch import collate_batch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def prepare_cmip_data(cfg):
    root_dir = cfg.DATASETS.ROOT_DIR
    cmip_data = nc4.Dataset(root_dir + 'CMIP_train.nc').variables
    cmip_label = nc4.Dataset(root_dir + 'CMIP_label.nc').variables

    cmip = dict()
    for var in ['sst', 't300', 'ua', 'va']:
        tmp = np.array(cmip_data[var][:, 0:12, :, :])
        tmp = np.nan_to_num(tmp)
        tmp = torch.tensor(tmp)
        tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
        cmip[var] = tmp.numpy()
    tmp = np.array(cmip_label['nino'][:, 12:24])
    last_year_nino = np.array(cmip_label['nino'][-1, -12:].reshape((1, 12)))
    tmp = np.concatenate((tmp, last_year_nino), axis=0)
    cmip['label'] = tmp.flatten()

    dict_cmip = dict()
    for var in ['sst', 't300', 'ua', 'va', 'label']:
        dict_cmip[var] = cmip[var]
    return dict_cmip


def prepare_soda_data(cfg):
    #soda_data = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_train.nc')
    #soda_label = nc4.Dataset(cfg.DATASETS.ROOT_DIR + 'SODA_label.nc')
    #soda_sst = np.array(soda_data.variables['sst'][:, 0:12, :, :])
    #soda_t300 = np.array(soda_data.variables['t300'][:, 0:12, :, :])
    #soda_ua = np.array(soda_data.variables['ua'][:, 0:12, :, :])
    #soda_va = np.array(soda_data.variables['va'][:, 0:12, :, :])
    #soda_label = np.array(soda_label.variables['nino'][:, 12:36])
    #dict_soda = {
    #   'sst':soda_sst,
    #   't300':soda_t300,
    #   'ua':soda_ua,
    #   'va':soda_va,
    #   'label':soda_label
    #}
    #return dict_soda
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

    #soda['label'] = np.array(soda_label['nino'][:, 12:36])
    #soda['label'] = np.array(soda['label']).flatten()

    dict_soda = dict()
    for var in ['sst', 't300', 'ua', 'va', 'label']:
        dict_soda[var] = soda[var]
    return dict_soda

def prepare_test_data(cfg):
    test_path = cfg.DATASETS.TEST_DIR
    files = os.listdir(test_path)
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
    train_dataset_cmip = EarthDataset(
        data_dict=dict_cmip,
        transforms=build_transforms(cfg, is_train=True),
    )

    val_dataset_soda = EarthDataset(
        data_dict=dict_soda,
        transforms=build_transforms(cfg, is_train=False),
    )
    len_val = int(val_dataset_soda.len*0.2)
    train_dataset, val_dataset = data.random_split(val_dataset_soda, lengths=[val_dataset_soda.len-len_val, len_val] , generator=torch.Generator().manual_seed(cfg.SEED))
    #train_dataset = data.ConcatDataset([train_dataset_cmip, train_dataset_soda])

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
        shuffle=True)

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
