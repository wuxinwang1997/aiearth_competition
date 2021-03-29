# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import random
import os
import socket

import netCDF4 as nc4
import numpy as np


def prepare_cmip_data(cfg):
    if socket.gethostname() == 'lujingzedeMacBook-Pro.local':
        root_dir = '/Users/lujingze/Programming/ai-earth/data/enso_round1_train_20210201/'
    else:
        root_dir = cfg.DATASETS.ROOT_DIR
    cmip_data = nc4.Dataset(root_dir + 'CMIP_train.nc').variables

    all_one_year = []
    all_half_year = []
    # TODO: Use other data
    for start_month in range(0, 12, 4):
        one_year = cmip_data['sst'][:, start_month:start_month+12, :, :]
        one_year = np.nan_to_num(one_year)
        one_year = np.expand_dims(one_year, axis=4)
        all_one_year.append(one_year)
        half_year = cmip_data['sst'][:, start_month+12:start_month+24, :, :]
        half_year = np.nan_to_num(half_year)
        half_year = np.expand_dims(half_year, axis=4)
        all_half_year.append(half_year)
    all_one_year = np.concatenate(all_one_year, axis=0)
    all_half_year = np.concatenate(all_half_year, axis=0)
    # TODO: make use of last 2 years
    return all_one_year, all_half_year


def prepare_soda_data(cfg):
    if socket.gethostname() == 'lujingzedeMacBook-Pro.local':
        root_dir = '/Users/lujingze/Programming/ai-earth/data/enso_round1_train_20210201/'
    else:
        root_dir = cfg.DATASETS.ROOT_DIR
    soda_data = nc4.Dataset(root_dir + 'SODA_train.nc').variables

    all_one_year = []
    all_half_year = []
    # TODO: Use other data
    for start_month in range(0, 12, 1):
        one_year = soda_data['sst'][:, start_month:start_month+12, :, :]
        one_year = np.nan_to_num(one_year)
        one_year = np.expand_dims(one_year, axis=4)
        all_one_year.append(one_year)
        half_year = soda_data['sst'][:, start_month+12:start_month+24, :, :]
        half_year = np.nan_to_num(half_year)
        half_year = np.expand_dims(half_year, axis=4)
        all_half_year.append(half_year)
    all_one_year = np.concatenate(all_one_year, axis=0)
    all_half_year = np.concatenate(all_half_year, axis=0)
    # TODO: make use of last 2 years
    return all_one_year, all_half_year


def prepare_test_data(cfg):
    test_path = cfg.DATASETS.TEST_DIR
    files = [x for x in os.listdir(test_path) if x.endswith('.npy')]
    test_sst = np.zeros((len(files), 12, 24, 72))
    # test_t300 = np.zeros((len(files), 12, 24, 72))
    # test_ua = np.zeros((len(files), 12, 24, 72))
    # test_va = np.zeros((len(files), 12, 24, 72))
    for i in range(len(files)):
        file = np.load(test_path + files[i])
        sst, t300, ua, va = np.split(file, 4, axis=3)
        test_sst[i, :, :, :] = sst.transpose(3, 0, 1, 2)
        # test_t300[i, :, :, :] = t300.transpose(3, 0, 1, 2)
        # test_ua[i, :, :, :] = ua.transpose(3, 0, 1, 2)
        # test_va[i, :, :, :] = va.transpose(3, 0, 1, 2)

    dict_test = {
        'sst': test_sst,
        # 't300': test_t300,
        #   'ua': test_ua,
        #   'va': test_va,
        'name': np.array(files)
    }
    return dict_test


def build_dataset(cfg):
    if not cfg.DATASETS.SODA:
        all_one_year, all_half_year = prepare_cmip_data(cfg)
    else:
        all_one_year, all_half_year = prepare_soda_data(cfg)

    np.random.shuffle(all_one_year)
    np.random.shuffle(all_half_year)

    return all_one_year, all_half_year
